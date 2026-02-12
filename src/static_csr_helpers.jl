"""
    static_csr_from_csc(A::SparseMatrixCSC)

Create a `StaticSparsityMatrixCSR` from a `SparseMatrixCSC` by transposing internally.
"""
function static_csr_from_csc(A::SparseMatrixCSC)
    return StaticSparsityMatrixCSR(sparse(A'))
end

"""
    rowptr(S::StaticSparsityMatrixCSR)

Return the row pointer array.
"""
rowptr(S::StaticSparsityMatrixCSR) = SparseArrays.getcolptr(S.At)

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
