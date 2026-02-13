"""
    strength_graph(A, θ)

Compute strength-of-connection for matrix `A` using threshold `θ`.
Returns a boolean CSR matrix `S` where `S[i,j] = true` if j is a strong connection of i.

A connection (i,j) is strong if |A[i,j]| ≥ θ * max_{k≠i} |A[i,k]|
"""
function strength_graph(A::CSRMatrix{Tv, Ti}, θ::Real) where {Tv, Ti}
    return strength_graph(A, θ, AbsoluteStrength())
end

"""
    strength_graph(A, θ, ::AbsoluteStrength)

Absolute-value strength: |a_{i,j}| ≥ θ * max_{k≠i} |a_{i,k}|.
"""
function strength_graph(A::CSRMatrix{Tv, Ti}, θ::Real, ::AbsoluteStrength) where {Tv, Ti}
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
    strength_graph(A, θ, ::SignedStrength)

Sign-aware (classical Ruge-Stüben) strength of connection for non-M-matrices.
Only off-diagonals with opposite sign from the diagonal are considered for strong
connections. This is critical for reservoir simulation where some off-diagonals
may have "wrong" (positive) sign.

A connection (i,j) is strong if:
  sign(a_{i,j}) ≠ sign(a_{i,i})  AND  |a_{i,j}| ≥ θ * max_{k: sign(a_{i,k})≠sign(a_{i,i})} |a_{i,k}|

If all off-diagonals have the same sign as the diagonal (no "proper" connections),
falls back to absolute-value strength for that row.
"""
function strength_graph(A::CSRMatrix{Tv, Ti}, θ::Real, ::SignedStrength) where {Tv, Ti}
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    is_strong = Vector{Bool}(undef, nnz(A))
    @inbounds for row in 1:n
        rng = nzrange(A, row)
        # Find diagonal
        a_ii = zero(Tv)
        for nz in rng
            if cv[nz] == row
                a_ii = nzv[nz]
                break
            end
        end
        diag_sign = sign(real(a_ii))
        if diag_sign == 0
            diag_sign = one(real(Tv))  # default to positive if zero diagonal
        end
        # Find max magnitude among opposite-sign off-diagonals
        max_opposite = zero(real(Tv))
        has_opposite = false
        for nz in rng
            col = cv[nz]
            if col != row && sign(real(nzv[nz])) != diag_sign
                max_opposite = max(max_opposite, abs(nzv[nz]))
                has_opposite = true
            end
        end
        if has_opposite
            threshold = θ * max_opposite
            for nz in rng
                col = cv[nz]
                if col != row && sign(real(nzv[nz])) != diag_sign
                    is_strong[nz] = abs(nzv[nz]) >= threshold
                else
                    is_strong[nz] = false
                end
            end
        else
            # Fallback: no opposite-sign connections, use absolute value
            max_offdiag = zero(real(Tv))
            for nz in rng
                col = cv[nz]
                if col != row
                    max_offdiag = max(max_offdiag, abs(nzv[nz]))
                end
            end
            threshold = θ * max_offdiag
            for nz in rng
                col = cv[nz]
                if col != row
                    is_strong[nz] = abs(nzv[nz]) >= threshold
                else
                    is_strong[nz] = false
                end
            end
        end
    end
    return is_strong
end

"""
    strength_graph(A, θ, config::AMGConfig)

Dispatch strength computation based on config's strength_type.
"""
function strength_graph(A::CSRMatrix, θ::Real, config::AMGConfig)
    return strength_graph(A, θ, config.strength_type)
end

"""
    strong_neighbors(A, is_strong, row)

Return an iterator of strongly connected column indices for a given row.
"""
function strong_neighbors(A::CSRMatrix, is_strong::Vector{Bool}, row::Integer)
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
function _apply_max_row_sum(A::CSRMatrix{Tv, Ti}, threshold::Real) where {Tv, Ti}
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
    return CSRMatrix{Tv, Ti}(collect(rp), collect(cv), nzv_new, size(A, 1), size(A, 2))
end

# ── Safe diagonal helpers ────────────────────────────────────────────────────

"""
    _safe_inv_diag(d, row_norm)

Compute a safe inverse of diagonal entry `d`, using `row_norm` as a fallback
scale. Returns `1/d` when `|d|` is large enough, otherwise returns a safe
value that avoids Inf/NaN.
"""
function _safe_inv_diag(d::Tv, row_norm::Real) where Tv
    abs_d = abs(d)
    threshold = eps(real(Tv)) * max(one(real(Tv)), convert(real(Tv), row_norm))
    if abs_d > threshold
        return one(Tv) / d
    else
        # Return zero for truly zero diagonal (isolated node)
        return zero(Tv)
    end
end

"""
    _safe_threshold(::Type{Tv}, scale)

Compute a safe threshold for near-zero checks: eps(Tv) * max(1, scale).
"""
function _safe_threshold(::Type{Tv}, scale::Real) where Tv
    return eps(real(Tv)) * max(one(real(Tv)), convert(real(Tv), scale))
end
