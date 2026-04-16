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
    amg_resetup!(hierarchy, A_new::StaticSparsityMatrixCSR, config; partial=true, update_P=false)

External API entry point for StaticSparsityMatrixCSR resetup. Converts to the
internal `CSRMatrix` and forwards to the main `CSRMatrix`-based resetup.
"""
function Draugr.amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                                  A_new::StaticSparsityMatrixCSR{Tv, Ti},
                                  config::AMGConfig=AMGConfig();
                                  partial::Bool=true,
                                  update_P::Bool=false) where {Tv, Ti}
    A_csr = Draugr.csr_from_static(A_new)
    return Draugr.amg_resetup!(hierarchy, A_csr, config; partial=partial, update_P=update_P)
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
                             backend=Draugr.DEFAULT_BACKEND, block_size::Int=64,
                             residual::Union{Nothing, AbstractVector}=nothing)
    A_csr = Draugr.csr_from_static(A)
    return Draugr.smooth!(x, A_csr, b, smoother; steps=steps, backend=backend, block_size=block_size, residual=residual)
end

# ── Shared helpers ────────────────────────────────────────────────────────────

"""
    _smart_amg_resetup!(hierarchy, A, config)

Select the most efficient in-place resetup strategy based on what data is
available in the hierarchy:

1. `partial=true, update_P=true`  — when the first level has a `P_update_map`
   (CF-splitting coarsening with `allow_partial_resetup=true`). Recomputes
   interpolation weights from the new matrix values before updating the Galerkin
   products, giving the best quality with minimal cost.

2. `partial=true, update_P=false` — when only a `R_map` (restriction map) is
   available. Updates smoother weights and Galerkin products without recomputing
   the prolongation.

3. `partial=false` — full hierarchy rebuild when neither map is available.
"""
function _smart_amg_resetup!(hierarchy::AMGHierarchy, A, config::AMGConfig)
    if isempty(hierarchy.levels)
        # Trivial / coarsest-only hierarchy: just update the direct solver
        Draugr.amg_resetup!(hierarchy, A, config; partial=true)
        return hierarchy
    end
    lvl1 = hierarchy.levels[1]
    if lvl1.P_update_map !== nothing && lvl1.R_map !== nothing
        Draugr.amg_resetup!(hierarchy, A, config; partial=true, update_P=true)
    elseif lvl1.R_map !== nothing
        Draugr.amg_resetup!(hierarchy, A, config; partial=true, update_P=false)
    else
        Draugr.amg_resetup!(hierarchy, A, config; partial=false)
    end
    return hierarchy
end

# ── Jutul Preconditioner ─────────────────────────────────────────────────────

"""
    DraugrPreconditionerJutul <: Jutul.JutulPreconditioner

AMG preconditioner implementing the Jutul preconditioner interface.
Can be used as a preconditioner in Jutul's linear solvers.

On first call, performs a full AMG setup. On subsequent calls, uses the smartest
available in-place resetup strategy:
- `partial=true, update_P=true` when a `P_update_map` is present (best quality).
- `partial=true` when only a restriction map is present.
- `partial=false` (full rebuild) otherwise.
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
        _smart_amg_resetup!(prec.hierarchy, A, prec.config)
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
    Draugr.amg_cycle!(x, y, prec.hierarchy, prec.config; residual=y)
    return x
end

function Jutul.operator_nrows(prec::DraugrPreconditionerJutul)
    if isnothing(prec.dim)
        return 0
    end
    return prec.dim[1]
end

# ── Jutul Partial Update Preconditioner ──────────────────────────────────────

"""
    DraugrPreconditionerJutulPartial <: Jutul.JutulPreconditioner

Variant of `DraugrPreconditionerJutul` designed for repeated solves where the
matrix sparsity pattern is fixed but values change (e.g. nonlinear simulations).

On first call, performs a full AMG setup with `allow_partial_resetup=true` so that
restriction maps (and, for CF-splitting coarsening, prolongation update maps) are
precomputed.  On subsequent calls, the cheapest available resetup is used:

- `update_P=true`  when the prolongation update map is available — recomputes
  interpolation weights without rebuilding the coarsening.
- `partial=true`   when only restriction maps are available — recomputes
  smoother and Galerkin products but keeps the prolongation fixed.
- `partial=false`  (full rebuild) when no maps are available.

Create via `DraugrPreconditioner(solver=:jutul_partial; kwargs...)`.
"""
mutable struct DraugrPreconditionerJutulPartial <: Jutul.JutulPreconditioner
    config::AMGConfig
    hierarchy::Union{Nothing, AMGHierarchy}
    dim::Union{Nothing, Tuple{Int,Int}}
end

function Draugr.setup_specific_preconditioner(::Val{:jutul_partial}; kwargs...)
    config = AMGConfig(; kwargs...)
    return DraugrPreconditionerJutulPartial(config, nothing, nothing)
end

function Jutul.update_preconditioner!(prec::DraugrPreconditionerJutulPartial,
                                      A::StaticSparsityMatrixCSR, b, context, executor)
    if isnothing(prec.hierarchy)
        prec.hierarchy = Draugr.amg_setup(A, prec.config)
    else
        _smart_amg_resetup!(prec.hierarchy, A, prec.config)
    end
    prec.dim = size(A)
    return prec
end

function Jutul.update_preconditioner!(prec::DraugrPreconditionerJutulPartial,
                                      A, b, context, executor)
    A_csr = Draugr.static_csr_from_csc(A)
    return Jutul.update_preconditioner!(prec, A_csr, b, context, executor)
end

function Jutul.apply!(x, prec::DraugrPreconditionerJutulPartial, y)
    fill!(x, zero(eltype(x)))
    Draugr.amg_cycle!(x, y, prec.hierarchy, prec.config; residual=y)
    return x
end

function Jutul.operator_nrows(prec::DraugrPreconditionerJutulPartial)
    if isnothing(prec.dim)
        return 0
    end
    return prec.dim[1]
end

end # module
