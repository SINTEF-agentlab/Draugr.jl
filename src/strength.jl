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

"""
    _apply_max_row_sum(A, threshold)

Apply max row sum dependency weakening (as in hypre). For rows where
(Σ_j |a_{i,j}|) / |a_{i,i}| > threshold, off-diagonal entries are scaled down
so that the absolute row sum ratio is brought to the threshold. This weakens
dependencies in rows that are far from diagonal dominance.

This is used only for strength-of-connection computation, not for the actual
matrix used in the solve.

Returns a new matrix with the same sparsity pattern but modified coefficients.
"""
function _apply_max_row_sum(A::StaticSparsityMatrixCSR{Tv, Ti}, threshold::Real) where {Tv, Ti}
    n = size(A, 1)
    cv = colvals(A)
    nzv_old = nonzeros(A)
    nzv_new = copy(nzv_old)
    rp = rowptr(A)
    @inbounds for i in 1:n
        a_ii = zero(Tv)
        abs_offdiag_sum = zero(real(Tv))
        for nz in rp[i]:(rp[i+1]-1)
            j = cv[nz]
            if j == i
                a_ii = nzv_old[nz]
            else
                abs_offdiag_sum += abs(nzv_old[nz])
            end
        end
        abs_aii = abs(a_ii)
        abs_aii < eps(real(Tv)) && continue
        # Check if absolute row sum ratio exceeds threshold
        # ratio = (|a_ii| + sum|off-diag|) / |a_ii| = 1 + sum|off-diag|/|a_ii|
        abs_row_sum_ratio = (abs_aii + abs_offdiag_sum) / abs_aii
        if abs_row_sum_ratio > threshold && abs_offdiag_sum > eps(real(Tv))
            # Scale off-diag so that (|a_ii| + α * sum|off-diag|) / |a_ii| = threshold
            # α = (threshold * |a_ii| - |a_ii|) / sum|off-diag|
            #   = |a_ii| * (threshold - 1) / sum|off-diag|
            α = abs_aii * (threshold - one(real(Tv))) / abs_offdiag_sum
            α = clamp(α, zero(real(Tv)), one(real(Tv)))
            for nz in rp[i]:(rp[i+1]-1)
                j = cv[nz]
                if j != i
                    nzv_new[nz] = Tv(α) * nzv_old[nz]
                end
            end
        end
    end
    return StaticSparsityMatrixCSR(size(A, 1), size(A, 2),
                                    collect(rp), collect(cv), nzv_new)
end
