module DraugrJLArraysExt

using Draugr
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
function Draugr.csr_from_gpu(A::JLSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
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
function Draugr.amg_setup(A::JLSparseMatrixCSR{Tv, Ti},
                               config::AMGConfig=AMGConfig();
                               backend=JLBackend(),
                               block_size::Int=64,
                               allow_partial_resetup::Bool=true) where {Tv, Ti}
    A_csr = Draugr.csr_from_gpu(A)
    return Draugr.amg_setup(A_csr, config; backend=backend, block_size=block_size,
                            allow_partial_resetup=allow_partial_resetup)
end

"""
    amg_resetup!(hierarchy, A_new::JLSparseMatrixCSR, config)

AMG resetup accepting a JLArrays sparse CSR matrix. Converts to a CPU
`CSRMatrix` and forwards to the main `CSRMatrix`-based resetup.
"""
function Draugr.amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                                  A_new::JLSparseMatrixCSR{Tv, Ti},
                                  config::AMGConfig=AMGConfig();
                                  partial::Bool=true,
                                  allow_partial_resetup::Bool=true) where {Tv, Ti}
    A_csr = Draugr.csr_to_cpu(Draugr.csr_from_gpu(A_new))
    return Draugr.amg_resetup!(hierarchy, A_csr, config; partial=partial,
                               allow_partial_resetup=allow_partial_resetup)
end

end # module
