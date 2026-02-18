module DraugrCUDAExt

using Draugr
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using KernelAbstractions, LinearAlgebra

"""
    csr_from_gpu(A::CuSparseMatrixCSR) -> CSRMatrix

Convert a CUDA `CuSparseMatrixCSR` to the internal `CSRMatrix` representation
by extracting its raw GPU arrays. The resulting `CSRMatrix` is backed by
`CuVector`s and can be used directly with KernelAbstractions kernels.
"""
function Draugr.csr_from_gpu(A::CuSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    rp = A.rowPtr
    cv = A.colVal
    nzv = A.nzVal
    return CSRMatrix(rp, cv, nzv, size(A, 1), size(A, 2))
end

"""
    amg_setup(A::CuSparseMatrixCSR, config; backend, block_size) -> AMGHierarchy

AMG setup accepting a CUDA sparse CSR matrix. Unwraps the GPU arrays into
a `CSRMatrix` and forwards to the standard setup with `CUDABackend()`.
"""
function Draugr.amg_setup(A::CuSparseMatrixCSR{Tv, Ti},
                               config::AMGConfig=AMGConfig();
                               backend=CUDABackend(),
                               block_size::Int=64,
                               allow_partial_resetup::Bool=true) where {Tv, Ti}
    A_csr = Draugr.csr_from_gpu(A)
    return Draugr.amg_setup(A_csr, config; backend=backend, block_size=block_size,
                            allow_partial_resetup=allow_partial_resetup)
end

"""
    amg_resetup!(hierarchy, A_new::CuSparseMatrixCSR, config)

AMG resetup accepting a CUDA sparse CSR matrix. Converts to a CPU
`CSRMatrix` and forwards to the main `CSRMatrix`-based resetup.
"""
function Draugr.amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                                  A_new::CuSparseMatrixCSR{Tv, Ti},
                                  config::AMGConfig=AMGConfig();
                                  partial::Bool=true,
                                  allow_partial_resetup::Bool=true) where {Tv, Ti}
    A_csr = Draugr.csr_to_cpu(Draugr.csr_from_gpu(A_new))
    return Draugr.amg_resetup!(hierarchy, A_csr, config; partial=partial,
                               allow_partial_resetup=allow_partial_resetup)
end

end # module
