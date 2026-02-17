# ══════════════════════════════════════════════════════════════════════════════
# Internal CSR matrix representation
# ══════════════════════════════════════════════════════════════════════════════

"""
    CSRMatrix{Tv, Ti, Vr, Vc, Vv}

Lightweight CSR matrix holding raw row-pointer, column-index, and value vectors.
This is the internal representation used throughout the AMG hierarchy.
External API entry points accept various sparse CSR formats and convert to
`CSRMatrix` once at the boundary.

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
SparseArrays.nzrange(A::CSRMatrix, row::Integer) = A.rowptr[row]:(A.rowptr[row+1]-one(eltype(A.rowptr)))

function Base.getindex(A::CSRMatrix{Tv}, i::Integer, j::Integer) where Tv
    @boundscheck (1 <= i <= A.nrow && 1 <= j <= A.ncol) || throw(BoundsError(A, (i, j)))
    for nz in nzrange(A, i)
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
    _to_device(ref::CSRMatrix, v::AbstractVector)

Copy a vector `v` to the same device as `ref`'s arrays.
If `ref` is already on CPU, returns the vector as-is.
"""
function _to_device(ref::CSRMatrix{Tv, Ti, <:Array, <:Array, <:Array}, v::AbstractVector) where {Tv, Ti}
    return v  # already CPU
end

function _to_device(ref::CSRMatrix, v::AbstractVector)
    be = _get_backend(ref.nzval)
    dev = KernelAbstractions.allocate(be, eltype(v), length(v))
    copyto!(dev, v)
    return dev
end

"""
    _csr_to_device(ref, A_cpu) -> CSRMatrix

Copy a CPU CSRMatrix to the same device as `ref`'s arrays.
"""
function _csr_to_device(ref::CSRMatrix, A_cpu::CSRMatrix)
    rp = _to_device(ref, A_cpu.rowptr)
    cv = _to_device(ref, A_cpu.colval)
    nzv = _to_device(ref, A_cpu.nzval)
    return CSRMatrix(rp, cv, nzv, A_cpu.nrow, A_cpu.ncol)
end

"""
    csr_from_csc(A::SparseMatrixCSC; do_collect=false) -> CSRMatrix

Convert a `SparseMatrixCSC` to a `CSRMatrix` by transposing.

When `do_collect` is `false` (default), the resulting `CSRMatrix` directly
references the internal arrays of the transposed matrix without copying.
When `do_collect` is `true`, `collect` is called to produce independent copies.
"""
function csr_from_csc(A::SparseMatrixCSC{Tv, Ti}; do_collect::Bool=false) where {Tv, Ti}
    At = sparse(A')
    rp = SparseArrays.getcolptr(At)
    cv = SparseArrays.rowvals(At)
    nzv = nonzeros(At)
    if do_collect
        rp = collect(rp)
        cv = collect(cv)
        nzv = collect(nzv)
    end
    return CSRMatrix(rp, cv, nzv, size(A, 1), size(A, 2))
end

"""
    csr_from_raw(rowptr, colval, nzval, nrow, ncol; index_base=1) -> CSRMatrix

Create a `CSRMatrix` from raw CSR arrays. When `index_base=0`, the row-pointer
and column-index arrays are assumed to use zero-based indexing (as in C/C++) and
are converted to one-based indexing **in-place** (no copy). When `index_base=1`
(the default), arrays are used as-is.

The caller must own the arrays (e.g., via `copy`) because this function may
mutate `rowptr` and `colval` when `index_base=0`.
"""
function csr_from_raw(rowptr::AbstractVector{Ti}, colval::AbstractVector{Ti},
                      nzval::AbstractVector{Tv}, nrow::Int, ncol::Int;
                      index_base::Int=1) where {Tv, Ti<:Integer}
    if index_base == 0
        rowptr .+= Ti(1)
        colval .+= Ti(1)
    elseif index_base != 1
        throw(ArgumentError("index_base must be 0 or 1, got $index_base"))
    end
    return CSRMatrix(rowptr, colval, nzval, nrow, ncol)
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

"""
    _allocate_vector(A::CSRMatrix, ::Type{Tv}, n) -> AbstractVector{Tv}

Allocate a zero-filled vector of length `n` on the same device as the CSR matrix `A`.
Uses `KernelAbstractions.zeros` with the backend inferred from the matrix arrays.
"""
function _allocate_vector(A::CSRMatrix, ::Type{Tv}, n::Int) where Tv
    be = _get_backend(A.nzval)
    return KernelAbstractions.zeros(be, Tv, n)
end

"""
    _allocate_undef_vector(A::CSRMatrix, ::Type{Tv}, n) -> AbstractVector{Tv}

Allocate an uninitialized vector of length `n` on the same device as the CSR matrix `A`.
Uses `KernelAbstractions.allocate` with the backend inferred from the matrix arrays.
"""
function _allocate_undef_vector(A::CSRMatrix, ::Type{Tv}, n::Int) where Tv
    be = _get_backend(A.nzval)
    return KernelAbstractions.allocate(be, Tv, n)
end

"""
    _allocate_dense_matrix(A::CSRMatrix, ::Type{Tv}, m, n) -> AbstractMatrix{Tv}

Allocate a zero-filled m×n dense matrix on the same device as the CSR matrix `A`.
Uses `KernelAbstractions.zeros` with the backend inferred from the matrix arrays.
"""
function _allocate_dense_matrix(A::CSRMatrix, ::Type{Tv}, m::Int, n::Int) where Tv
    be = _get_backend(A.nzval)
    return KernelAbstractions.zeros(be, Tv, m, n)
end
