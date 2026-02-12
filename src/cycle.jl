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
    # Compute residual: r = b - A*x
    mul!(r, A, x)
    @inbounds for i in 1:length(r)
        r[i] = b[i] - r[i]
    end
    # Restrict residual to coarse grid
    restrict!(bc, P, r)
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
    prolongate!(x, P, xc)
    # Post-smoothing
    smooth!(x, A, b, level.smoother; steps=config.post_smoothing_steps, backend=backend)
    return x
end

"""
    amg_solve!(x, b, hierarchy, config; tol=1e-10, maxiter=100, backend=CPU())

Solve Ax = b using AMG V-cycles. Returns the solution in `x` and the number of
iterations performed.
"""
function amg_solve!(x::AbstractVector{Tv}, b::AbstractVector{Tv},
                    hierarchy::AMGHierarchy{Tv, Ti},
                    config::AMGConfig=AMGConfig();
                    tol::Real=1e-10, maxiter::Int=100,
                    backend=CPU()) where {Tv, Ti}
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
    r = similar(x)
    for iter in 1:maxiter
        amg_cycle!(x, b, hierarchy, config; backend=backend)
        # Check convergence
        mul!(r, A, x)
        @inbounds for i in 1:length(r)
            r[i] = b[i] - r[i]
        end
        rnorm = norm(r)
        if rnorm / bnorm < tol
            return x, iter
        end
    end
    return x, maxiter
end
