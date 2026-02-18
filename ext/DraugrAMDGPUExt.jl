module DraugrAMDGPUExt

using Draugr
using AMDGPU
using AMDGPU: ROCSparseMatrixCSR
using KernelAbstractions, LinearAlgebra

"""
    csr_from_gpu(A::ROCSparseMatrixCSR) -> CSRMatrix

Convert an AMDGPU `ROCSparseMatrixCSR` to the internal `CSRMatrix` representation
by extracting its raw GPU arrays. The resulting `CSRMatrix` is backed by
`ROCVector`s and can be used directly with KernelAbstractions kernels.
"""
function Draugr.csr_from_gpu(A::ROCSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    rp = A.rowPtr
    cv = A.colVal
    nzv = A.nzVal
    return CSRMatrix(rp, cv, nzv, size(A, 1), size(A, 2))
end

"""
    amg_setup(A::ROCSparseMatrixCSR, config; backend, block_size) -> AMGHierarchy

AMG setup accepting an AMDGPU sparse CSR matrix. Unwraps the GPU arrays into
a `CSRMatrix` and forwards to the standard setup with `ROCBackend()`.
"""
function Draugr.amg_setup(A::ROCSparseMatrixCSR{Tv, Ti},
                               config::AMGConfig=AMGConfig();
                               backend=ROCBackend(),
                               block_size::Int=64) where {Tv, Ti}
    A_csr = Draugr.csr_from_gpu(A)
    return Draugr.amg_setup(A_csr, config; backend=backend, block_size=block_size)
end

"""
    amg_resetup!(hierarchy, A_new::ROCSparseMatrixCSR, config)

AMG resetup accepting an AMDGPU sparse CSR matrix. Converts to a CPU
`CSRMatrix` and forwards to the main `CSRMatrix`-based resetup.
"""
function Draugr.amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                                  A_new::ROCSparseMatrixCSR{Tv, Ti},
                                  config::AMGConfig=AMGConfig();
                                  partial::Bool=true) where {Tv, Ti}
    A_csr = Draugr.csr_to_cpu(Draugr.csr_from_gpu(A_new))
    return Draugr.amg_resetup!(hierarchy, A_csr, config; partial=partial)
end

end # module
