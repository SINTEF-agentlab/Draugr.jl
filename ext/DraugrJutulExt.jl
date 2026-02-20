module DraugrJutulExt

using Draugr
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
function Draugr.static_csr_from_csc(A::SparseMatrixCSC)
    return StaticSparsityMatrixCSR(sparse(A'))
end

"""
    rowptr(S::StaticSparsityMatrixCSR)

Return the row pointer array.
"""
Draugr.rowptr(S::StaticSparsityMatrixCSR) = SparseArrays.getcolptr(S.At)

"""
    find_nz_index(A::StaticSparsityMatrixCSR, row, col)

Find the index in the nonzero array for entry (row, col). Returns 0 if not found.
"""
function Draugr.find_nz_index(A::StaticSparsityMatrixCSR, row::Integer, col::Integer)
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
    csr_from_static(A::StaticSparsityMatrixCSR; do_collect=false) -> CSRMatrix

Convert a `StaticSparsityMatrixCSR` to the internal `CSRMatrix`
representation by extracting its raw CSR vectors.

When `do_collect` is `false` (default), the resulting `CSRMatrix` directly
references the internal arrays of the source matrix without copying.
When `do_collect` is `true`, `collect` is called to produce independent copies.
"""
function Draugr.csr_from_static(A::StaticSparsityMatrixCSR{Tv, Ti}; do_collect::Bool=false) where {Tv, Ti}
    rp = Draugr.rowptr(A)
    cv = colvals(A)
    nzv = nonzeros(A)
    if do_collect
        rp = collect(rp)
        cv = collect(cv)
        nzv = collect(nzv)
    end
    return CSRMatrix(rp, cv, nzv, size(A, 1), size(A, 2))
end

"""
    csr_copy_nzvals!(dest::CSRMatrix, src::StaticSparsityMatrixCSR)

Copy nonzero values from a `StaticSparsityMatrixCSR` into an existing
`CSRMatrix` with the same sparsity pattern.
"""
function Draugr.csr_copy_nzvals!(dest::CSRMatrix{Tv}, src::StaticSparsityMatrixCSR{Tv};
                                      backend=Draugr.DEFAULT_BACKEND, block_size::Int=64) where Tv
    nzv_d = nonzeros(dest)
    nzv_s = nonzeros(src)
    n = length(nzv_d)
    kernel! = Draugr.copy_kernel!(backend, block_size)
    kernel!(nzv_d, nzv_s; ndrange=n)
    Draugr._synchronize(backend)
    return dest
end

# ── AMG setup/resetup entry points for StaticSparsityMatrixCSR ───────────────

"""
    amg_setup(A::StaticSparsityMatrixCSR, config; backend) -> AMGHierarchy

External API entry point: convert `StaticSparsityMatrixCSR` to `CSRMatrix` once
and forward to the general CSRMatrix-based setup.
"""
function Draugr.amg_setup(A::StaticSparsityMatrixCSR{Tv, Ti}, config::AMGConfig=AMGConfig();
                               backend=Draugr.DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    return Draugr.amg_setup(Draugr.csr_from_static(A), config; backend=backend, block_size=block_size)
end

"""
    amg_resetup!(hierarchy, A_new::StaticSparsityMatrixCSR, config)

External API entry point for StaticSparsityMatrixCSR resetup. Converts to the
internal `CSRMatrix` and forwards to the main `CSRMatrix`-based resetup.
"""
function Draugr.amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                                  A_new::StaticSparsityMatrixCSR{Tv, Ti},
                                  config::AMGConfig=AMGConfig();
                                  partial::Bool=true) where {Tv, Ti}
    A_csr = Draugr.csr_from_static(A_new)
    return Draugr.amg_resetup!(hierarchy, A_csr, config; partial=partial)
end

# ── Smoother wrappers for StaticSparsityMatrixCSR ────────────────────────────

function Draugr.build_smoother(A::StaticSparsityMatrixCSR, smoother_type::Draugr.SmootherType;
                                    ω::Real=2.0/3.0, backend=Draugr.DEFAULT_BACKEND, block_size::Int=64)
    A_csr = Draugr.csr_from_static(A)
    return Draugr.build_smoother(A_csr, smoother_type, ω; backend=backend, block_size=block_size)
end

function Draugr.update_smoother!(smoother::Draugr.AbstractSmoother, A::StaticSparsityMatrixCSR;
                                      backend=Draugr.DEFAULT_BACKEND, block_size::Int=64)
    A_csr = Draugr.csr_from_static(A)
    return Draugr.update_smoother!(smoother, A_csr; backend=backend, block_size=block_size)
end

function Draugr.smooth!(x::AbstractVector, A::StaticSparsityMatrixCSR, b::AbstractVector,
                             smoother::Draugr.AbstractSmoother; steps::Int=1,
                             backend=Draugr.DEFAULT_BACKEND, block_size::Int=64)
    A_csr = Draugr.csr_from_static(A)
    return Draugr.smooth!(x, A_csr, b, smoother; steps=steps, backend=backend, block_size=block_size)
end

# ── Jutul Preconditioner ─────────────────────────────────────────────────────

"""
    DraugrPreconditionerJutul <: Jutul.JutulPreconditioner

AMG preconditioner implementing the Jutul preconditioner interface.
Can be used as a preconditioner in Jutul's linear solvers.
"""
mutable struct DraugrPreconditionerJutul <: Jutul.JutulPreconditioner
    config::AMGConfig
    hierarchy::Union{Nothing, AMGHierarchy}
    dim::Union{Nothing, Tuple{Int,Int}}
end

function Draugr.setup_specific_preconditioner(::Val{:jutul}; kwargs...)
    config = AMGConfig(; kwargs...)
    return DraugrPreconditionerJutul(config, nothing, nothing)
end

function Jutul.update_preconditioner!(prec::DraugrPreconditionerJutul,
                                      A::StaticSparsityMatrixCSR, b, context, executor)
    if isnothing(prec.hierarchy)
        prec.hierarchy = Draugr.amg_setup(A, prec.config)
    else
        Draugr.amg_resetup!(prec.hierarchy, A, prec.config)
    end
    prec.dim = size(A)
    return prec
end

function Jutul.update_preconditioner!(prec::DraugrPreconditionerJutul,
                                      A, b, context, executor)
    A_csr = Draugr.static_csr_from_csc(A)
    return Jutul.update_preconditioner!(prec, A_csr, b, context, executor)
end

function Jutul.apply!(x, prec::DraugrPreconditionerJutul, y)
    fill!(x, zero(eltype(x)))
    Draugr.amg_cycle!(x, y, prec.hierarchy, prec.config)
    return x
end

function Jutul.operator_nrows(prec::DraugrPreconditionerJutul)
    if isnothing(prec.dim)
        return 0
    end
    return prec.dim[1]
end

end # module
