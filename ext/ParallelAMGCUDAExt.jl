module ParallelAMGCUDAExt

using ParallelAMG
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using KernelAbstractions, LinearAlgebra

"""
    csr_from_gpu(A::CuSparseMatrixCSR) -> CSRMatrix

Convert a CUDA `CuSparseMatrixCSR` to the internal `CSRMatrix` representation
by extracting its raw GPU arrays. The resulting `CSRMatrix` is backed by
`CuVector`s and can be used directly with KernelAbstractions kernels.
"""
function ParallelAMG.csr_from_gpu(A::CuSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    rp = A.rowPtr
    cv = A.colVal
    nzv = A.nzVal
    return CSRMatrix(rp, cv, nzv, size(A, 1), size(A, 2))
end

"""
    amg_setup(A::CuSparseMatrixCSR, config; backend) -> AMGHierarchy

AMG setup accepting a CUDA sparse CSR matrix. Unwraps the GPU arrays into
a `CSRMatrix` and forwards to the standard setup with `CUDABackend()`.
"""
function ParallelAMG.amg_setup(A::CuSparseMatrixCSR{Tv, Ti},
                               config::AMGConfig=AMGConfig();
                               backend=CUDABackend()) where {Tv, Ti}
    A_csr = ParallelAMG.csr_from_gpu(A)
    return ParallelAMG.amg_setup(A_csr, config; backend=backend)
end

"""
    amg_resetup!(hierarchy, A_new::CuSparseMatrixCSR, config)

AMG resetup accepting a CUDA sparse CSR matrix. Unwraps the GPU arrays,
converts to CPU, and copies values into the existing hierarchy.
"""
function ParallelAMG.amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                                  A_new::CuSparseMatrixCSR{Tv, Ti},
                                  config::AMGConfig=AMGConfig();
                                  backend=CUDABackend()) where {Tv, Ti}
    A_csr = ParallelAMG.csr_to_cpu(ParallelAMG.csr_from_gpu(A_new))
    block_size = config.block_size
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
