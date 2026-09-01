module DraugrCUDAExt

using Draugr
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using LinearAlgebra

"""
    csr_from_gpu(A::CuSparseMatrixCSR) -> CSRMatrix

Expose cuSPARSE CSR buffers as Draugr's backend-neutral CSR representation.
All PMIS/direct symbolic setup is implemented in Draugr with
KernelAbstractions; this extension supplies only CUDA sparse-library calls.
"""
function Draugr.csr_from_gpu(A::CuSparseMatrixCSR{Tv, Ti}) where {Tv, Ti}
    return CSRMatrix(A.rowPtr, A.colVal, A.nzVal, size(A, 1), size(A, 2))
end

"""CUDA specialization of the Galerkin sparse product `P' * A * P`."""
function Draugr.compute_coarse_sparsity(
        A_fine::CSRMatrix{Tv, Ti, <:CUDA.CuArray, <:CUDA.CuArray, <:CUDA.CuArray},
        P::Draugr.ProlongationOp{Ti, Tv, <:CUDA.CuArray, <:CUDA.CuArray},
        Pt_map::Draugr.TransposeMap, n_coarse::Int; kwargs...) where {Tv, Ti}
    P_sparse = CuSparseMatrixCSR(P.rowptr, P.colval, P.nzval, (P.nrow, P.ncol))
    A_sparse = CuSparseMatrixCSR(A_fine.rowptr, A_fine.colval, A_fine.nzval,
                                  (A_fine.nrow, A_fine.ncol))
    A_coarse = CuSparseMatrixCSR(transpose(P_sparse) * A_sparse * P_sparse)
    return Draugr.csr_from_gpu(A_coarse), nothing
end

"""
    amg_setup(A::CuSparseMatrixCSR, config; backend, block_size) -> AMGHierarchy

AMG setup accepting a CUDA sparse CSR matrix. It unwraps the GPU buffers and
forwards to Draugr's backend-neutral setup implementation.
"""
function Draugr.amg_setup(A::CuSparseMatrixCSR{Tv, Ti},
                          config::AMGConfig=AMGConfig();
                          backend=CUDABackend(),
                          block_size::Int=64) where {Tv, Ti}
    return Draugr.amg_setup(Draugr.csr_from_gpu(A), config;
        backend=backend, block_size=block_size)
end

"""CUDA CSR resetup entry point that retains the input buffers on device."""
function Draugr.amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                             A_new::CuSparseMatrixCSR{Tv, Ti},
                             config::AMGConfig=AMGConfig();
                             partial::Bool=true,
                             update_P::Bool=false) where {Tv, Ti}
    return Draugr.amg_resetup!(hierarchy, Draugr.csr_from_gpu(A_new), config;
        partial=partial, update_P=update_P)
end

end # module
