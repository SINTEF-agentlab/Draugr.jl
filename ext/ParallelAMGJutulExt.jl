module ParallelAMGJutulExt

using ParallelAMG
using Jutul
using Jutul.StaticCSR: StaticSparsityMatrixCSR, static_sparsity_sparse,
                       nthreads, minbatch
import Jutul.StaticCSR: colvals
using SparseArrays
using LinearAlgebra
using KernelAbstractions

# ── StaticCSR helpers ────────────────────────────────────────────────────────

"""
    static_csr_from_csc(A::SparseMatrixCSC)

Create a `StaticSparsityMatrixCSR` from a `SparseMatrixCSC` by transposing internally.
"""
function ParallelAMG.static_csr_from_csc(A::SparseMatrixCSC)
    return StaticSparsityMatrixCSR(sparse(A'))
end

"""
    rowptr(S::StaticSparsityMatrixCSR)

Return the row pointer array.
"""
ParallelAMG.rowptr(S::StaticSparsityMatrixCSR) = SparseArrays.getcolptr(S.At)

"""
    find_nz_index(A::StaticSparsityMatrixCSR, row, col)

Find the index in the nonzero array for entry (row, col). Returns 0 if not found.
"""
function ParallelAMG.find_nz_index(A::StaticSparsityMatrixCSR, row::Integer, col::Integer)
    cv = colvals(A)
    for nz in nzrange(A, row)
        @inbounds if cv[nz] == col
            return nz
        end
    end
    return 0
end

# ── CSR conversion from StaticSparsityMatrixCSR ──────────────────────────────

"""
    csr_from_static(A::StaticSparsityMatrixCSR) -> CSRMatrix

Convert a `StaticSparsityMatrixCSR` to the internal `CSRMatrix`
representation by extracting its raw CSR vectors.
"""
function ParallelAMG.csr_from_static(A::StaticSparsityMatrixCSR{Tv, Ti}) where {Tv, Ti}
    rp = collect(ParallelAMG.rowptr(A))
    cv = collect(colvals(A))
    nzv = collect(nonzeros(A))
    return CSRMatrix(rp, cv, nzv, size(A, 1), size(A, 2))
end

"""
    csr_copy_nzvals!(dest::CSRMatrix, src::StaticSparsityMatrixCSR)

Copy nonzero values from a `StaticSparsityMatrixCSR` into an existing
`CSRMatrix` with the same sparsity pattern.
"""
function ParallelAMG.csr_copy_nzvals!(dest::CSRMatrix{Tv}, src::StaticSparsityMatrixCSR{Tv};
                                      backend=ParallelAMG.DEFAULT_BACKEND, block_size::Int=64) where Tv
    nzv_d = nonzeros(dest)
    nzv_s = nonzeros(src)
    n = length(nzv_d)
    kernel! = ParallelAMG.copy_kernel!(backend, block_size)
    kernel!(nzv_d, nzv_s; ndrange=n)
    ParallelAMG._synchronize(backend)
    return dest
end

# ── AMG setup/resetup entry points for StaticSparsityMatrixCSR ───────────────

"""
    amg_setup(A::StaticSparsityMatrixCSR, config; backend) -> AMGHierarchy

External API entry point: convert `StaticSparsityMatrixCSR` to `CSRMatrix` once
and forward to the general CSRMatrix-based setup.
"""
function ParallelAMG.amg_setup(A::StaticSparsityMatrixCSR{Tv, Ti}, config::AMGConfig=AMGConfig();
                               backend=ParallelAMG.DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    return ParallelAMG.amg_setup(ParallelAMG.csr_from_static(A), config; backend=backend, block_size=block_size)
end

"""
    amg_resetup!(hierarchy, A_new::StaticSparsityMatrixCSR, config)

External API entry point for StaticSparsityMatrixCSR resetup.
"""
function ParallelAMG.amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                                  A_new::StaticSparsityMatrixCSR{Tv, Ti},
                                  config::AMGConfig=AMGConfig()) where {Tv, Ti}
    backend = hierarchy.backend
    block_size = hierarchy.block_size
    nlevels = length(hierarchy.levels)
    if nlevels == 0
        A_csr = ParallelAMG.csr_from_static(A_new)
        ParallelAMG._update_coarse_solver!(hierarchy, A_csr; backend=backend, block_size=block_size)
        return hierarchy
    end
    level1 = hierarchy.levels[1]
    ParallelAMG.csr_copy_nzvals!(level1.A, A_new; backend=backend, block_size=block_size)
    ParallelAMG.update_smoother!(level1.smoother, level1.A; backend=backend, block_size=block_size)
    for lvl in 1:(nlevels - 1)
        level = hierarchy.levels[lvl]
        next_level = hierarchy.levels[lvl + 1]
        ParallelAMG.galerkin_product!(next_level.A, level.A, level.P, level.R_map; backend=backend, block_size=block_size)
        ParallelAMG.update_smoother!(next_level.smoother, next_level.A; backend=backend, block_size=block_size)
    end
    last_level = hierarchy.levels[nlevels]
    ParallelAMG._recompute_coarsest_dense!(hierarchy, last_level; backend=backend)
    hierarchy.coarse_factor = lu(hierarchy.coarse_A)
    return hierarchy
end

# ── Smoother wrappers for StaticSparsityMatrixCSR ────────────────────────────

function ParallelAMG.build_smoother(A::StaticSparsityMatrixCSR, smoother_type::ParallelAMG.SmootherType;
                                    ω::Real=2.0/3.0, backend=ParallelAMG.DEFAULT_BACKEND, block_size::Int=64)
    A_csr = ParallelAMG.csr_from_static(A)
    return ParallelAMG.build_smoother(A_csr, smoother_type, ω; backend=backend, block_size=block_size)
end

function ParallelAMG.update_smoother!(smoother::ParallelAMG.AbstractSmoother, A::StaticSparsityMatrixCSR;
                                      backend=ParallelAMG.DEFAULT_BACKEND, block_size::Int=64)
    A_csr = ParallelAMG.csr_from_static(A)
    return ParallelAMG.update_smoother!(smoother, A_csr; backend=backend, block_size=block_size)
end

function ParallelAMG.smooth!(x::AbstractVector, A::StaticSparsityMatrixCSR, b::AbstractVector,
                             smoother::ParallelAMG.AbstractSmoother; steps::Int=1,
                             backend=ParallelAMG.DEFAULT_BACKEND, block_size::Int=64)
    A_csr = ParallelAMG.csr_from_static(A)
    return ParallelAMG.smooth!(x, A_csr, b, smoother; steps=steps, backend=backend, block_size=block_size)
end

# ── Jutul Preconditioner ─────────────────────────────────────────────────────

"""
    ParallelAMGPreconditionerJutul <: Jutul.JutulPreconditioner

AMG preconditioner implementing the Jutul preconditioner interface.
Can be used as a preconditioner in Jutul's linear solvers.
"""
mutable struct ParallelAMGPreconditionerJutul <: Jutul.JutulPreconditioner
    config::AMGConfig
    hierarchy::Union{Nothing, AMGHierarchy}
    dim::Union{Nothing, Tuple{Int,Int}}
end

function ParallelAMG.setup_specific_preconditioner(::Val{:jutul}; kwargs...)
    config = AMGConfig(; kwargs...)
    return ParallelAMGPreconditionerJutul(config, nothing, nothing)
end

function Jutul.update_preconditioner!(prec::ParallelAMGPreconditionerJutul,
                                      A::StaticSparsityMatrixCSR, b, context, executor)
    if isnothing(prec.hierarchy)
        prec.hierarchy = ParallelAMG.amg_setup(A, prec.config)
    else
        ParallelAMG.amg_resetup!(prec.hierarchy, A, prec.config)
    end
    prec.dim = size(A)
    return prec
end

function Jutul.update_preconditioner!(prec::ParallelAMGPreconditionerJutul,
                                      A, b, context, executor)
    A_csr = ParallelAMG.static_csr_from_csc(A)
    return Jutul.update_preconditioner!(prec, A_csr, b, context, executor)
end

function Jutul.apply!(x, prec::ParallelAMGPreconditionerJutul, y)
    fill!(x, zero(eltype(x)))
    ParallelAMG.amg_cycle!(x, y, prec.hierarchy, prec.config)
    return x
end

function Jutul.operator_nrows(prec::ParallelAMGPreconditionerJutul)
    if isnothing(prec.dim)
        return 0
    end
    return prec.dim[1]
end

end # module
