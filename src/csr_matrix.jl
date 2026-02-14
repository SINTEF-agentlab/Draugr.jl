# ══════════════════════════════════════════════════════════════════════════════
# Internal CSR matrix representation
# ══════════════════════════════════════════════════════════════════════════════

"""
    CSRMatrix{Tv, Ti, Vr, Vc, Vv}

Lightweight CSR matrix holding raw row-pointer, column-index, and value vectors.
This is the internal representation used throughout the AMG hierarchy, decoupled
from Jutul's `StaticSparsityMatrixCSR`.  External API entry points accept
`StaticSparsityMatrixCSR` and convert to `CSRMatrix` once at the boundary.

The vector types are parameterized as `AbstractVector` subtypes so the format
works with GPU arrays or other custom vector implementations.
"""
struct CSRMatrix{Tv, Ti<:Integer, Vr<:AbstractVector{Ti}, Vc<:AbstractVector{Ti}, Vv<:AbstractVector{Tv}}
    rowptr::Vr
    colval::Vc
    nzval::Vv
    nrow::Int
    ncol::Int
end

Base.size(A::CSRMatrix) = (A.nrow, A.ncol)
Base.size(A::CSRMatrix, d::Int) = d == 1 ? A.nrow : (d == 2 ? A.ncol : 1)
SparseArrays.nnz(A::CSRMatrix) = length(A.nzval)
SparseArrays.nonzeros(A::CSRMatrix) = A.nzval
SparseArrays.nzrange(A::CSRMatrix, row::Integer) = A.rowptr[row]:(A.rowptr[row+1]-1)

function Base.getindex(A::CSRMatrix{Tv}, i::Integer, j::Integer) where Tv
    @boundscheck (1 <= i <= A.nrow && 1 <= j <= A.ncol) || throw(BoundsError(A, (i, j)))
    for nz in A.rowptr[i]:(A.rowptr[i+1]-1)
        @inbounds A.colval[nz] == j && return A.nzval[nz]
    end
    return zero(Tv)
end

"""Return the column-values vector for CSRMatrix."""
colvals(A::CSRMatrix) = A.colval

"""Return the row-pointer vector for CSRMatrix."""
rowptr(A::CSRMatrix) = A.rowptr

"""
    csr_to_cpu(A::CSRMatrix) -> CSRMatrix

Convert a CSRMatrix with GPU arrays to a CSRMatrix with CPU arrays.
If the arrays are already CPU arrays, this is a no-op (returns same object).
"""
function csr_to_cpu(A::CSRMatrix{Tv, Ti, <:Array, <:Array, <:Array}) where {Tv, Ti}
    return A  # already CPU
end

function csr_to_cpu(A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    return CSRMatrix(Array(A.rowptr), Array(A.colval), Array(A.nzval), A.nrow, A.ncol)
end

"""
    csr_from_static(A::StaticSparsityMatrixCSR) -> CSRMatrix

Convert a Jutul `StaticSparsityMatrixCSR` to the internal `CSRMatrix`
representation by extracting its raw CSR vectors. This is the single
conversion point; all internal functions work only with `CSRMatrix`.
"""
function csr_from_static(A::StaticSparsityMatrixCSR{Tv, Ti}) where {Tv, Ti}
    rp = collect(rowptr(A))
    cv = collect(colvals(A))
    nzv = collect(nonzeros(A))
    return CSRMatrix(rp, cv, nzv, size(A, 1), size(A, 2))
end

"""
    csr_copy_nzvals!(dest::CSRMatrix, src::StaticSparsityMatrixCSR)

Copy nonzero values from a `StaticSparsityMatrixCSR` into an existing
`CSRMatrix` with the same sparsity pattern.
"""
function csr_copy_nzvals!(dest::CSRMatrix{Tv}, src::StaticSparsityMatrixCSR{Tv};
                          backend=DEFAULT_BACKEND, block_size::Int=64) where Tv
    nzv_d = nonzeros(dest)
    nzv_s = nonzeros(src)
    n = length(nzv_d)
    kernel! = copy_kernel!(backend, block_size)
    kernel!(nzv_d, nzv_s; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return dest
end

@kernel function copy_kernel!(dst, @Const(src))
    i = @index(Global)
    @inbounds dst[i] = src[i]
end

"""
    LinearAlgebra.mul!(y, A::CSRMatrix, x)

CSR matrix-vector product y = A * x.
"""
function LinearAlgebra.mul!(y::AbstractVector, A::CSRMatrix, x::AbstractVector)
    @inbounds for i in 1:A.nrow
        s = zero(eltype(y))
        for nz in A.rowptr[i]:(A.rowptr[i+1]-1)
            s += A.nzval[nz] * x[A.colval[nz]]
        end
        y[i] = s
    end
    return y
end

"""
    _get_backend(arr)

Infer the KernelAbstractions backend from an array using `KernelAbstractions.get_backend`.
Returns the appropriate backend for GPU arrays (e.g., CUDABackend, MetalBackend, JLBackend)
or CPU backend for regular Arrays.
"""
function _get_backend(arr::AbstractVector)
    return KernelAbstractions.get_backend(arr)
end

"""
    _synchronize(backend)

Synchronize the backend. This wraps `KernelAbstractions.synchronize` with a fallback
for backends that don't implement it (e.g., JLBackend), since some mock GPU backends
execute kernels synchronously.
"""
function _synchronize(backend)
    if hasmethod(KernelAbstractions.synchronize, Tuple{typeof(backend)})
        KernelAbstractions.synchronize(backend)
    end
    return nothing
end
