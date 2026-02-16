module ParallelAMGLinearSolveExt

using ParallelAMG
using LinearSolve
using LinearAlgebra
using SparseArrays

# ── LinearSolve Preconditioner ───────────────────────────────────────────────

"""
    ParallelAMGPreconditionerLinearSolve <: AbstractParallelAMGPreconditioner

AMG preconditioner for use with LinearSolve.jl. Supports `ldiv!(y, P, x)`
which is the interface required by LinearSolve.jl iterative solvers.

Can be constructed via `ParallelAMGPreconditioner(solver=:linearsolve; kwargs...)`.
"""
mutable struct ParallelAMGPreconditionerLinearSolve <: AbstractParallelAMGPreconditioner
    config::AMGConfig
    hierarchy::Union{Nothing, AMGHierarchy}
    dim::Union{Nothing, Tuple{Int,Int}}
end

function ParallelAMG.setup_specific_preconditioner(::Val{:linearsolve}; kwargs...)
    config = AMGConfig(; kwargs...)
    return ParallelAMGPreconditionerLinearSolve(config, nothing, nothing)
end

"""
    LinearAlgebra.ldiv!(x, prec::ParallelAMGPreconditionerLinearSolve, b)

Apply the AMG preconditioner: approximately solve `A*x = b` using one V-cycle.
This is the interface required by LinearSolve.jl.
"""
function LinearAlgebra.ldiv!(x::AbstractVector, prec::ParallelAMGPreconditionerLinearSolve, b::AbstractVector)
    return ParallelAMG.preconditioner_apply!(x, prec, b)
end

"""
    update!(prec::ParallelAMGPreconditionerLinearSolve, A; kwargs...)

Update the AMG preconditioner with a new matrix `A`.
Accepts `SparseMatrixCSC` or `CSRMatrix`.
"""
function update!(prec::ParallelAMGPreconditionerLinearSolve, A::SparseMatrixCSC)
    A_csr = ParallelAMG.csr_from_csc(A)
    return _update_csr!(prec, A_csr)
end

function update!(prec::ParallelAMGPreconditionerLinearSolve, A_csr::CSRMatrix)
    return _update_csr!(prec, A_csr)
end

function _update_csr!(prec::ParallelAMGPreconditionerLinearSolve, A_csr::CSRMatrix)
    if isnothing(prec.hierarchy)
        prec.hierarchy = ParallelAMG.amg_setup(A_csr, prec.config)
    else
        ParallelAMG.amg_resetup!(prec.hierarchy, A_csr, prec.config)
    end
    prec.dim = size(A_csr)
    return prec
end

end # module
