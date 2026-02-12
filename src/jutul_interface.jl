# ══════════════════════════════════════════════════════════════════════════════
# Jutul Preconditioner Interface
# ══════════════════════════════════════════════════════════════════════════════

import Jutul

"""
    ParallelAMGPreconditioner <: Jutul.JutulPreconditioner

AMG preconditioner implementing the Jutul preconditioner interface.
Can be used as a preconditioner in Jutul's linear solvers.

# Constructor

    ParallelAMGPreconditioner(; kwargs...)

Keyword arguments are forwarded to `AMGConfig`.
"""
mutable struct ParallelAMGPreconditioner <: Jutul.JutulPreconditioner
    config::AMGConfig
    hierarchy::Union{Nothing, AMGHierarchy}
    dim::Union{Nothing, Tuple{Int,Int}}
end

function ParallelAMGPreconditioner(; kwargs...)
    config = AMGConfig(; kwargs...)
    return ParallelAMGPreconditioner(config, nothing, nothing)
end

"""
    update_preconditioner!(prec::ParallelAMGPreconditioner, A, b, context, executor)

Update (or build) the AMG preconditioner for matrix `A`.
If the hierarchy has not been built yet, performs a full setup.
Otherwise, performs an in-place resetup (same sparsity, new coefficients).
"""
function Jutul.update_preconditioner!(prec::ParallelAMGPreconditioner,
                                      A::StaticSparsityMatrixCSR, b, context, executor)
    if isnothing(prec.hierarchy)
        prec.hierarchy = amg_setup(A, prec.config)
    else
        amg_resetup!(prec.hierarchy, A, prec.config)
    end
    prec.dim = size(A)
    return prec
end

function Jutul.update_preconditioner!(prec::ParallelAMGPreconditioner,
                                      A, b, context, executor)
    A_csr = static_csr_from_csc(A)
    return Jutul.update_preconditioner!(prec, A_csr, b, context, executor)
end

"""
    apply!(x, prec::ParallelAMGPreconditioner, y)

Apply the AMG preconditioner: solve approximately `A*x = y` using one V-cycle.
"""
function Jutul.apply!(x, prec::ParallelAMGPreconditioner, y)
    fill!(x, zero(eltype(x)))
    amg_cycle!(x, y, prec.hierarchy, prec.config)
    return x
end

"""
    operator_nrows(prec::ParallelAMGPreconditioner)

Return the number of rows the preconditioner operates on.
"""
function Jutul.operator_nrows(prec::ParallelAMGPreconditioner)
    if isnothing(prec.dim)
        return 0
    end
    return prec.dim[1]
end
