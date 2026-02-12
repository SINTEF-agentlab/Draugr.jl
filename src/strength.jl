"""
    strength_graph(A, θ)

Compute strength-of-connection for matrix `A` using threshold `θ`.
Returns a boolean CSR matrix `S` where `S[i,j] = true` if j is a strong connection of i.

A connection (i,j) is strong if |A[i,j]| ≥ θ * max_{k≠i} |A[i,k]|
"""
function strength_graph(A::StaticSparsityMatrixCSR{Tv, Ti}, θ::Real) where {Tv, Ti}
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    is_strong = Vector{Bool}(undef, nnz(A))
    @inbounds for row in 1:n
        rng = nzrange(A, row)
        # Find maximum off-diagonal magnitude
        max_offdiag = zero(real(Tv))
        for nz in rng
            col = cv[nz]
            if col != row
                max_offdiag = max(max_offdiag, abs(nzv[nz]))
            end
        end
        threshold = θ * max_offdiag
        # Mark strong connections
        for nz in rng
            col = cv[nz]
            if col != row
                is_strong[nz] = abs(nzv[nz]) >= threshold
            else
                is_strong[nz] = false
            end
        end
    end
    return is_strong
end

"""
    strong_neighbors(A, is_strong, row)

Return an iterator of strongly connected column indices for a given row.
"""
function strong_neighbors(A::StaticSparsityMatrixCSR, is_strong::Vector{Bool}, row::Integer)
    cv = colvals(A)
    return (cv[nz] for nz in nzrange(A, row) if is_strong[nz])
end
