module ParallelAMGJLArraysExt

using ParallelAMG
using JLArrays
using JLArrays: JLSparseMatrixCSR
using KernelAbstractions
using LinearAlgebra

"""
    csr_from_gpu(A::JLSparseMatrixCSR) -> CSRMatrix

Convert a JLArrays `JLSparseMatrixCSR` to the internal `CSRMatrix` representation
by extracting its raw GPU arrays. The resulting `CSRMatrix` is backed by
`JLVector`s and can be used directly with KernelAbstractions kernels.
"""
function ParallelAMG.csr_from_gpu(A::JLSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    rp = A.rowPtr
    cv = A.colVal
    nzv = A.nzVal
    return CSRMatrix(rp, cv, nzv, size(A, 1), size(A, 2))
end

"""
    amg_setup(A::JLSparseMatrixCSR, config; backend, block_size) -> AMGHierarchy

AMG setup accepting a JLArrays sparse CSR matrix. Unwraps the GPU arrays into
a `CSRMatrix` and forwards to the standard setup with `JLBackend()`.
"""
function ParallelAMG.amg_setup(A::JLSparseMatrixCSR{Tv, Ti},
                               config::AMGConfig=AMGConfig();
                               backend=JLBackend(),
                               block_size::Int=64) where {Tv, Ti}
    A_csr = ParallelAMG.csr_from_gpu(A)
    return ParallelAMG.amg_setup(A_csr, config; backend=backend, block_size=block_size)
end

"""
    amg_resetup!(hierarchy, A_new::JLSparseMatrixCSR, config)

AMG resetup accepting a JLArrays sparse CSR matrix. Uses backend and block_size
from the hierarchy.
"""
function ParallelAMG.amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                                  A_new::JLSparseMatrixCSR{Tv, Ti},
                                  config::AMGConfig=AMGConfig()) where {Tv, Ti}
    backend = hierarchy.backend
    block_size = hierarchy.block_size
    A_csr = ParallelAMG.csr_to_cpu(ParallelAMG.csr_from_gpu(A_new))
    nlevels = length(hierarchy.levels)
    if nlevels == 0
        ParallelAMG._update_coarse_solver!(hierarchy, A_csr; block_size=block_size)
        return hierarchy
    end
    # Copy nonzero values from new matrix into existing level 1
    level1 = hierarchy.levels[1]
    ParallelAMG._copy_nzvals!(level1.A, A_csr; block_size=block_size)
    ParallelAMG.update_smoother!(level1.smoother, level1.A; block_size=block_size)
    # Update subsequent levels via Galerkin products
    for lvl in 1:(nlevels - 1)
        level = hierarchy.levels[lvl]
        next_level = hierarchy.levels[lvl + 1]
        ParallelAMG.galerkin_product!(next_level.A, level.A, level.P, level.R_map; block_size=block_size)
        ParallelAMG.update_smoother!(next_level.smoother, next_level.A; block_size=block_size)
    end
    # Recompute coarsest dense matrix and LU
    last_level = hierarchy.levels[nlevels]
    ParallelAMG._recompute_coarsest_dense!(hierarchy, last_level)
    hierarchy.coarse_factor = lu(hierarchy.coarse_A)
    return hierarchy
end

end # module
