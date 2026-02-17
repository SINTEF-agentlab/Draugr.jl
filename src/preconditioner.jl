# ══════════════════════════════════════════════════════════════════════════════
# Preconditioner Base Types
# ══════════════════════════════════════════════════════════════════════════════

"""
    AbstractDraugrPreconditioner

Abstract base type for AMG preconditioners. Extension modules define concrete
subtypes that inherit from the appropriate solver-specific abstract types
(e.g., `Jutul.JutulPreconditioner` or LinearSolve-compatible types).
"""
abstract type AbstractDraugrPreconditioner end

"""
    DraugrPreconditioner

Concrete AMG preconditioner that can be used standalone or with solver-specific
extensions. Stores the AMG configuration and hierarchy.

# Constructor

    DraugrPreconditioner(; solver=nothing, kwargs...)

Keyword arguments are forwarded to `AMGConfig`. The `solver` argument is an
optional `Symbol` (e.g., `:jutul`, `:linearsolve`) that selects a solver-specific
preconditioner subtype via `setup_specific_preconditioner`.
When `solver` is `nothing`, creates a standalone preconditioner.
"""
mutable struct DraugrPreconditioner <: AbstractDraugrPreconditioner
    config::AMGConfig
    hierarchy::Union{Nothing, AMGHierarchy}
    dim::Union{Nothing, Tuple{Int,Int}}
end

function DraugrPreconditioner(; solver::Union{Nothing, Symbol}=nothing, kwargs...)
    config = AMGConfig(; kwargs...)
    if isnothing(solver)
        return DraugrPreconditioner(config, nothing, nothing)
    else
        return setup_specific_preconditioner(Val(solver); kwargs...)
    end
end

"""
    setup_specific_preconditioner(::Val{S}; kwargs...) where S

Dispatch function for creating solver-specific preconditioner subtypes.
Extension modules add methods for specific solver symbols (e.g., `:jutul`, `:linearsolve`).
"""
function setup_specific_preconditioner(::Val{S}; kwargs...) where S
    error("Solver :$S is not supported. Load the appropriate extension package.")
end

"""
    preconditioner_update!(prec::DraugrPreconditioner, A::CSRMatrix)

Update (or build) the AMG preconditioner for a CSRMatrix.
If the hierarchy has not been built yet, performs a full setup.
Otherwise, performs an in-place resetup (same sparsity, new coefficients).
"""
function preconditioner_update!(prec::DraugrPreconditioner, A_csr::CSRMatrix)
    if isnothing(prec.hierarchy)
        prec.hierarchy = amg_setup(A_csr, prec.config)
    else
        amg_resetup!(prec.hierarchy, A_csr, prec.config)
    end
    prec.dim = size(A_csr)
    return prec
end

"""
    preconditioner_apply!(x, prec::AbstractDraugrPreconditioner, y)

Apply the AMG preconditioner: solve approximately `A*x = y` using one V-cycle.
"""
function preconditioner_apply!(x, prec::AbstractDraugrPreconditioner, y)
    fill!(x, zero(eltype(x)))
    amg_cycle!(x, y, prec.hierarchy, prec.config)
    return x
end

"""
    preconditioner_nrows(prec::AbstractDraugrPreconditioner)

Return the number of rows the preconditioner operates on.
"""
function preconditioner_nrows(prec::AbstractDraugrPreconditioner)
    if isnothing(prec.dim)
        return 0
    end
    return prec.dim[1]
end

"""
    LinearAlgebra.ldiv!(x, prec::AbstractDraugrPreconditioner, b)

Apply the preconditioner as a left-division: approximately solve `A*x = b`.
This enables usage with LinearSolve.jl and other packages that expect `ldiv!`.
"""
function LinearAlgebra.ldiv!(x::AbstractVector, prec::AbstractDraugrPreconditioner, b::AbstractVector)
    return preconditioner_apply!(x, prec, b)
end
