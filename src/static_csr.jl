"""
    StaticSparsityMatrixCSR{Tv, Ti}

A CSR (Compressed Sparse Row) matrix with static sparsity pattern, compatible with the
StaticCSR format from the Jutul package. The sparsity pattern is fixed at construction
time, but nonzero values can be modified in-place.

Internally stores the transpose as a `SparseMatrixCSC` so that row-oriented access
maps directly to CSC column access.
"""
struct StaticSparsityMatrixCSR{Tv, Ti<:Integer} <: SparseArrays.AbstractSparseMatrix{Tv, Ti}
    At::SparseMatrixCSC{Tv, Ti}
    function StaticSparsityMatrixCSR(At::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
        return new{Tv, Ti}(At)
    end
end

function StaticSparsityMatrixCSR(m::Integer, n::Integer, rowptr::Vector{Ti},
                                  colval::Vector{Ti}, nzval::Vector{Tv}) where {Tv, Ti<:Integer}
    At = SparseMatrixCSC(n, m, rowptr, colval, nzval)
    return StaticSparsityMatrixCSR(At)
end

"""
    static_sparsity_sparse(I, J, V, [m, n])

Construct a `StaticSparsityMatrixCSR` from COO-format arrays.
"""
function static_sparsity_sparse(I, J, V, m=maximum(I), n=maximum(J))
    A = sparse(J, I, V, n, m)
    return StaticSparsityMatrixCSR(A)
end

"""
    static_csr_from_csc(A::SparseMatrixCSC)

Create a `StaticSparsityMatrixCSR` from a `SparseMatrixCSC` by transposing internally.
"""
function static_csr_from_csc(A::SparseMatrixCSC)
    return StaticSparsityMatrixCSR(sparse(A'))
end

Base.size(S::StaticSparsityMatrixCSR) = reverse(size(S.At))
Base.getindex(S::StaticSparsityMatrixCSR, I::Integer, J::Integer) = S.At[J, I]
SparseArrays.nnz(S::StaticSparsityMatrixCSR) = nnz(S.At)
SparseArrays.nonzeros(S::StaticSparsityMatrixCSR) = nonzeros(S.At)
Base.isstored(S::StaticSparsityMatrixCSR, I::Integer, J::Integer) = Base.isstored(S.At, J, I)
SparseArrays.nzrange(S::StaticSparsityMatrixCSR, row::Integer) = SparseArrays.nzrange(S.At, row)

"""
    colvals(S::StaticSparsityMatrixCSR)

Return the column indices of the nonzero entries (analogous to `rowvals` for CSC).
"""
colvals(S::StaticSparsityMatrixCSR) = SparseArrays.rowvals(S.At)

"""
    rowptr(S::StaticSparsityMatrixCSR)

Return the row pointer array.
"""
rowptr(S::StaticSparsityMatrixCSR) = SparseArrays.getcolptr(S.At)

function Base.show(io::IO, ::MIME"text/plain", A::StaticSparsityMatrixCSR)
    m, n = size(A)
    print(io, "$m×$n StaticSparsityMatrixCSR{$(eltype(A))} with $(nnz(A)) stored entries")
end

function LinearAlgebra.mul!(y::AbstractVector, A::StaticSparsityMatrixCSR, x::AbstractVector,
                            α::Number, β::Number)
    n = size(A, 1)
    size(A, 2) == length(x) || throw(DimensionMismatch())
    length(y) == n || throw(DimensionMismatch())
    nzv = nonzeros(A)
    cv = colvals(A)
    if β != 1
        if β == 0
            fill!(y, zero(eltype(y)))
        else
            rmul!(y, β)
        end
    end
    @inbounds for row in 1:n
        v = zero(eltype(y))
        for nz in nzrange(A, row)
            col = cv[nz]
            v += nzv[nz] * x[col]
        end
        y[row] += α * v
    end
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, A::StaticSparsityMatrixCSR, x::AbstractVector)
    return mul!(y, A, x, one(eltype(A)), zero(eltype(y)))
end

"""
    find_nz_index(A::StaticSparsityMatrixCSR, row, col)

Find the index in the nonzero array for entry (row, col). Returns 0 if not found.
"""
function find_nz_index(A::StaticSparsityMatrixCSR, row::Integer, col::Integer)
    cv = colvals(A)
    for nz in nzrange(A, row)
        @inbounds if cv[nz] == col
            return nz
        end
    end
    return 0
end
