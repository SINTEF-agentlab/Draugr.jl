"""
    amg_cycle!(x, b, hierarchy, config)

Apply one V-cycle of AMG to improve the solution `x` for the system `Ax = b`.

The V-cycle descends through all levels with pre-smoothing, restriction, and
then ascends with prolongation and post-smoothing. At the coarsest level a
direct solver is used.

The backend and block_size are taken from the hierarchy (set during `amg_setup`).
"""
function amg_cycle!(x::AbstractVector{Tv}, b::AbstractVector{Tv},
                    hierarchy::AMGHierarchy{Tv, Ti},
                    config::AMGConfig=AMGConfig()) where {Tv, Ti}
    backend = hierarchy.backend
    block_size = hierarchy.block_size
    nlevels = length(hierarchy.levels)
    if nlevels == 0
        # Direct solve only
        copyto!(hierarchy.coarse_b, b)
        ldiv!(hierarchy.coarse_x, hierarchy.coarse_factor, hierarchy.coarse_b)
        copyto!(x, hierarchy.coarse_x)
        return x
    end
    # V-cycle: descend
    _vcycle_descend!(x, b, hierarchy, config, 1; backend=backend, block_size=block_size)
    return x
end

# ── KA kernel for residual: r = b - A*x ─────────────────────────────────────

@kernel function residual_kernel!(r, @Const(b), @Const(x),
                                  @Const(nzval), @Const(colval), @Const(rp))
    i = @index(Global)
    @inbounds begin
        Ax_i = zero(eltype(r))
        for nz in rp[i]:(rp[i+1]-1)
            j = colval[nz]
            Ax_i += nzval[nz] * x[j]
        end
        r[i] = b[i] - Ax_i
    end
end

"""
    compute_residual!(r, A, x, b; backend=DEFAULT_BACKEND, block_size=64)

Compute residual r = b - A*x using a KA kernel. No allocations.
"""
function compute_residual!(r::AbstractVector, A::CSRMatrix,
                           x::AbstractVector, b::AbstractVector;
                           backend=DEFAULT_BACKEND, block_size::Int=64)
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    kernel! = residual_kernel!(backend, block_size)
    kernel!(r, b, x, nzv, cv, rp; ndrange=n)
    _synchronize(backend)
    return r
end

function _vcycle_descend!(x::AbstractVector{Tv}, b::AbstractVector{Tv},
                          hierarchy::AMGHierarchy{Tv, Ti},
                          config::AMGConfig, lvl::Int;
                          backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    nlevels = length(hierarchy.levels)
    level = hierarchy.levels[lvl]
    A = level.A
    P = level.P
    r = level.r
    xc = level.xc
    bc = level.bc
    # Pre-smoothing
    smooth!(x, A, b, level.smoother; steps=config.pre_smoothing_steps, backend=backend, block_size=block_size)
    # Compute residual: r = b - A*x (parallelized, no allocations)
    compute_residual!(r, A, x, b; backend=backend, block_size=block_size)
    # Restrict residual to coarse grid
    restrict!(bc, level.Pt_map, P, r; backend=backend, block_size=block_size)
    # Solve on coarse grid
    fill!(xc, zero(Tv))
    if lvl < nlevels
        # Recurse (W-cycle: recurse twice, V-cycle: once)
        n_recurse = config.cycle_type == :W ? 2 : 1
        for _ in 1:n_recurse
            _vcycle_descend!(xc, bc, hierarchy, config, lvl + 1; backend=backend, block_size=block_size)
        end
    else
        # Direct solve at coarsest level
        copyto!(hierarchy.coarse_b, bc)
        ldiv!(hierarchy.coarse_x, hierarchy.coarse_factor, hierarchy.coarse_b)
        copyto!(xc, hierarchy.coarse_x)
    end
    # Prolongate and correct: x += P * xc
    prolongate!(x, P, xc; backend=backend, block_size=block_size)
    # Post-smoothing
    smooth!(x, A, b, level.smoother; steps=config.post_smoothing_steps, backend=backend, block_size=block_size)
    return x
end

"""
    amg_solve!(x, b, hierarchy, config; tol=1e-10, maxiter=100)

Solve Ax = b using AMG V-cycles. Returns the solution in `x` and the number of
iterations performed. Uses pre-allocated residual buffer from hierarchy to avoid
allocations.

The backend and block_size are taken from the hierarchy (set during `amg_setup`).
"""
function amg_solve!(x::AbstractVector{Tv}, b::AbstractVector{Tv},
                    hierarchy::AMGHierarchy{Tv, Ti},
                    config::AMGConfig=AMGConfig();
                    tol::Real=1e-10, maxiter::Int=100) where {Tv, Ti}
    backend = hierarchy.backend
    block_size = hierarchy.block_size
    t_solve = config.verbose >= 1 ? time() : 0.0
    bnorm = norm(b)
    if bnorm == 0
        fill!(x, zero(Tv))
        return x, 0
    end
    # Get the finest level matrix
    if length(hierarchy.levels) > 0
        A = hierarchy.levels[1].A
    else
        # Only direct solve
        copyto!(hierarchy.coarse_b, b)
        ldiv!(hierarchy.coarse_x, hierarchy.coarse_factor, hierarchy.coarse_b)
        copyto!(x, hierarchy.coarse_x)
        return x, 1
    end
    # Use pre-allocated residual buffer (no allocations!)
    r = hierarchy.solve_r
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    rkernel! = residual_kernel!(backend, block_size)
    rnorm = bnorm
    for iter in 1:maxiter
        amg_cycle!(x, b, hierarchy, config)
        # Check convergence using parallelized residual computation (kernel pre-built)
        rkernel!(r, b, x, nzv, cv, rp; ndrange=n)
        _synchronize(backend)
        rnorm = norm(r)
        rel_norm = rnorm / bnorm
        if config.verbose >= 2
            Printf.@printf("  AMG iter %4d: residual norm = %.6e, relative = %.6e\n",
                            iter, rnorm, rel_norm)
        end
        if rel_norm < tol
            if config.verbose >= 1
                t_solve = time() - t_solve
                Printf.@printf("AMG solve converged: %d iterations, %.4f s, final residual %.2e\n",
                                iter, t_solve, rel_norm)
            end
            return x, iter
        end
    end
    if config.verbose >= 1
        t_solve = time() - t_solve
        Printf.@printf("AMG solve did NOT converge: %d iterations, %.4f s, final residual %.2e\n",
                        maxiter, t_solve, rnorm / bnorm)
    end
    return x, maxiter
end
