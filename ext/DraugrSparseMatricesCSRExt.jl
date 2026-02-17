module DraugrSparseMatricesCSRExt

using Draugr
using SparseMatricesCSR
using SparseArrays
using LinearAlgebra
using KernelAbstractions

# ── CSR conversion from SparseMatrixCSR ──────────────────────────────────────

"""
    csr_from_sparse_csr(A::SparseMatrixCSR{Bi}; do_collect=false) -> CSRMatrix

Convert a `SparseMatrixCSR` from SparseMatricesCSR.jl to the internal `CSRMatrix`
representation by extracting its raw CSR vectors.

Handles both 0-based and 1-based indexing (Bi parameter).

When `do_collect` is `false` (default) and indexing is 1-based, the resulting
`CSRMatrix` directly references the internal arrays without copying.
When `do_collect` is `true`, `collect` is called to produce independent copies.
Zero-based indexing always requires a copy since the indices must be converted.
"""
function csr_from_sparse_csr(A::SparseMatrixCSR{Bi, Tv, Ti}; do_collect::Bool=false) where {Bi, Tv, Ti}
    rp = SparseMatricesCSR.getrowptr(A)
    cv = SparseMatricesCSR.getcolval(A)
    nzv = nonzeros(A)
    # SparseMatrixCSR uses Bi-based indexing; convert to 1-based if needed
    if Bi != 1
        # Zero-based indexing always requires a copy
        offset = Ti(1 - Bi)
        rp = collect(rp) .+ offset
        cv = collect(cv) .+ offset
        nzv = collect(nzv)
    elseif do_collect
        rp = collect(rp)
        cv = collect(cv)
        nzv = collect(nzv)
    end
    return CSRMatrix(rp, cv, nzv, size(A, 1), size(A, 2))
end

# ── AMG setup/resetup entry points for SparseMatrixCSR ───────────────────────

"""
    amg_setup(A::SparseMatrixCSR, config; backend, block_size) -> AMGHierarchy

AMG setup accepting a `SparseMatrixCSR` from SparseMatricesCSR.jl.
Converts to the internal `CSRMatrix` format and forwards to the standard setup.
"""
function Draugr.amg_setup(A::SparseMatrixCSR{Bi, Tv, Ti}, config::AMGConfig=AMGConfig();
                               backend=Draugr.DEFAULT_BACKEND, block_size::Int=64) where {Bi, Tv, Ti}
    A_csr = csr_from_sparse_csr(A)
    return Draugr.amg_setup(A_csr, config; backend=backend, block_size=block_size)
end

"""
    amg_resetup!(hierarchy, A_new::SparseMatrixCSR, config)

AMG resetup accepting a `SparseMatrixCSR` from SparseMatricesCSR.jl.
"""
function Draugr.amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                                  A_new::SparseMatrixCSR{Bi, Tv, Ti},
                                  config::AMGConfig=AMGConfig()) where {Bi, Tv, Ti}
    A_csr = csr_from_sparse_csr(A_new)
    backend = hierarchy.backend
    block_size = hierarchy.block_size
    nlevels = length(hierarchy.levels)
    if nlevels == 0
        Draugr._update_coarse_solver!(hierarchy, A_csr; backend=backend, block_size=block_size)
        return hierarchy
    end
    level1 = hierarchy.levels[1]
    Draugr._copy_nzvals!(level1.A, A_csr; backend=backend, block_size=block_size)
    Draugr.update_smoother!(level1.smoother, level1.A; backend=backend, block_size=block_size)
    for lvl in 1:(nlevels - 1)
        level = hierarchy.levels[lvl]
        next_level = hierarchy.levels[lvl + 1]
        Draugr.galerkin_product!(next_level.A, level.A, level.P, level.R_map; backend=backend, block_size=block_size)
        Draugr.update_smoother!(next_level.smoother, next_level.A; backend=backend, block_size=block_size)
    end
    last_level = hierarchy.levels[nlevels]
    Draugr._recompute_coarsest_dense!(hierarchy, last_level; backend=backend)
    hierarchy.coarse_factor = lu(hierarchy.coarse_A)
    return hierarchy
end

# ── Smoother wrappers for SparseMatrixCSR ────────────────────────────────────

function Draugr.build_smoother(A::SparseMatrixCSR, smoother_type::Draugr.SmootherType;
                                    ω::Real=2.0/3.0, backend=Draugr.DEFAULT_BACKEND, block_size::Int=64)
    A_csr = csr_from_sparse_csr(A)
    return Draugr.build_smoother(A_csr, smoother_type, ω; backend=backend, block_size=block_size)
end

function Draugr.update_smoother!(smoother::Draugr.AbstractSmoother, A::SparseMatrixCSR;
                                      backend=Draugr.DEFAULT_BACKEND, block_size::Int=64)
    A_csr = csr_from_sparse_csr(A)
    return Draugr.update_smoother!(smoother, A_csr; backend=backend, block_size=block_size)
end

function Draugr.smooth!(x::AbstractVector, A::SparseMatrixCSR, b::AbstractVector,
                             smoother::Draugr.AbstractSmoother; steps::Int=1,
                             backend=Draugr.DEFAULT_BACKEND, block_size::Int=64)
    A_csr = csr_from_sparse_csr(A)
    return Draugr.smooth!(x, A_csr, b, smoother; steps=steps, backend=backend, block_size=block_size)
end

end # module
