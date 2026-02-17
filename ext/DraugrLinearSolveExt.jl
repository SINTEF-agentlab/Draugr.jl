module DraugrLinearSolveExt

using Draugr
using LinearSolve
using LinearAlgebra
using SparseArrays

# ── LinearSolve Preconditioner ───────────────────────────────────────────────

"""
    DraugrPreconditionerLinearSolve <: AbstractDraugrPreconditioner

AMG preconditioner for use with LinearSolve.jl. Supports `ldiv!(y, P, x)`
which is the interface required by LinearSolve.jl iterative solvers.

Can be constructed via `DraugrPreconditioner(solver=:linearsolve; kwargs...)`.
"""
mutable struct DraugrPreconditionerLinearSolve <: AbstractDraugrPreconditioner
    config::AMGConfig
    hierarchy::Union{Nothing, AMGHierarchy}
    dim::Union{Nothing, Tuple{Int,Int}}
end

function Draugr.setup_specific_preconditioner(::Val{:linearsolve}; kwargs...)
    config = AMGConfig(; kwargs...)
    return DraugrPreconditionerLinearSolve(config, nothing, nothing)
end

"""
    LinearAlgebra.ldiv!(x, prec::DraugrPreconditionerLinearSolve, b)

Apply the AMG preconditioner: approximately solve `A*x = b` using one V-cycle.
This is the interface required by LinearSolve.jl.
"""
function LinearAlgebra.ldiv!(x::AbstractVector, prec::DraugrPreconditionerLinearSolve, b::AbstractVector)
    return Draugr.preconditioner_apply!(x, prec, b)
end

"""
    update!(prec::DraugrPreconditionerLinearSolve, A; kwargs...)

Update the AMG preconditioner with a new matrix `A`.
Accepts `SparseMatrixCSC` or `CSRMatrix`.
"""
function update!(prec::DraugrPreconditionerLinearSolve, A::SparseMatrixCSC)
    A_csr = Draugr.csr_from_csc(A)
    return _update_csr!(prec, A_csr)
end

function update!(prec::DraugrPreconditionerLinearSolve, A_csr::CSRMatrix)
    return _update_csr!(prec, A_csr)
end

function _update_csr!(prec::DraugrPreconditionerLinearSolve, A_csr::CSRMatrix)
    if isnothing(prec.hierarchy)
        prec.hierarchy = Draugr.amg_setup(A_csr, prec.config)
    else
        Draugr.amg_resetup!(prec.hierarchy, A_csr, prec.config)
    end
    prec.dim = size(A_csr)
    return prec
end

end # module
