"""
    amg_cycle!(x, b, hierarchy, config; backend=CPU())

Apply one V-cycle of AMG to improve the solution `x` for the system `Ax = b`.

The V-cycle descends through all levels with pre-smoothing, restriction, and
then ascends with prolongation and post-smoothing. At the coarsest level a
direct solver is used.
"""
function amg_cycle!(x::AbstractVector{Tv}, b::AbstractVector{Tv},
                    hierarchy::AMGHierarchy{Tv, Ti},
                    config::AMGConfig=AMGConfig();
                    backend=CPU()) where {Tv, Ti}
    nlevels = length(hierarchy.levels)
    if nlevels == 0
        # Direct solve only
        copyto!(hierarchy.coarse_b, b)
        ldiv!(hierarchy.coarse_x, hierarchy.coarse_factor, hierarchy.coarse_b)
        copyto!(x, hierarchy.coarse_x)
        return x
    end
    # V-cycle: descend
    _vcycle_descend!(x, b, hierarchy, config, 1; backend=backend)
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
    compute_residual!(r, A, x, b; backend=CPU())

Compute residual r = b - A*x using a KA kernel. No allocations.
"""
function compute_residual!(r::AbstractVector, A::StaticSparsityMatrixCSR,
                           x::AbstractVector, b::AbstractVector;
                           backend=CPU())
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    kernel! = residual_kernel!(backend, 64)
    kernel!(r, b, x, nzv, cv, rp; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return r
end

function _vcycle_descend!(x::AbstractVector{Tv}, b::AbstractVector{Tv},
                          hierarchy::AMGHierarchy{Tv, Ti},
                          config::AMGConfig, lvl::Int;
                          backend=CPU()) where {Tv, Ti}
    nlevels = length(hierarchy.levels)
    level = hierarchy.levels[lvl]
    A = level.A
    P = level.P
    r = level.r
    xc = level.xc
    bc = level.bc
    # Pre-smoothing
    smooth!(x, A, b, level.smoother; steps=config.pre_smoothing_steps, backend=backend)
    # Compute residual: r = b - A*x (parallelized, no allocations)
    compute_residual!(r, A, x, b; backend=backend)
    # Restrict residual to coarse grid
    restrict!(bc, P, r; backend=backend)
    # Solve on coarse grid
    fill!(xc, zero(Tv))
    if lvl < nlevels
        # Recurse
        _vcycle_descend!(xc, bc, hierarchy, config, lvl + 1; backend=backend)
    else
        # Direct solve at coarsest level
        copyto!(hierarchy.coarse_b, bc)
        ldiv!(hierarchy.coarse_x, hierarchy.coarse_factor, hierarchy.coarse_b)
        copyto!(xc, hierarchy.coarse_x)
    end
    # Prolongate and correct: x += P * xc
    prolongate!(x, P, xc; backend=backend)
    # Post-smoothing
    smooth!(x, A, b, level.smoother; steps=config.post_smoothing_steps, backend=backend)
    return x
end

"""
    amg_solve!(x, b, hierarchy, config; tol=1e-10, maxiter=100, backend=CPU())

Solve Ax = b using AMG V-cycles. Returns the solution in `x` and the number of
iterations performed. Uses pre-allocated residual buffer from hierarchy to avoid
allocations.
"""
function amg_solve!(x::AbstractVector{Tv}, b::AbstractVector{Tv},
                    hierarchy::AMGHierarchy{Tv, Ti},
                    config::AMGConfig=AMGConfig();
                    tol::Real=1e-10, maxiter::Int=100,
                    backend=CPU()) where {Tv, Ti}
    t_solve = config.verbose ? time() : 0.0
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
    rnorm = bnorm
    for iter in 1:maxiter
        amg_cycle!(x, b, hierarchy, config; backend=backend)
        # Check convergence using parallelized residual computation
        compute_residual!(r, A, x, b; backend=backend)
        rnorm = norm(r)
        if rnorm / bnorm < tol
            if config.verbose
                t_solve = time() - t_solve
                Printf.@printf("AMG solve converged: %d iterations, %.4f s, final residual %.2e\n",
                                iter, t_solve, rnorm / bnorm)
            end
            return x, iter
        end
    end
    if config.verbose
        t_solve = time() - t_solve
        Printf.@printf("AMG solve did NOT converge: %d iterations, %.4f s, final residual %.2e\n",
                        maxiter, t_solve, rnorm / bnorm)
    end
    return x, maxiter
end
