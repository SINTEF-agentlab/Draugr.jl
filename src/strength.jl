"""
    strength_graph(A, θ)

Compute strength-of-connection for matrix `A` using threshold `θ`.
Returns a boolean CSR matrix `S` where `S[i,j] = true` if j is a strong connection of i.

A connection (i,j) is strong if |A[i,j]| ≥ θ * max_{k≠i} |A[i,k]|

The backend is inferred from the array type of A's data, so GPU arrays
will use GPU kernels automatically.
"""
function strength_graph(A::CSRMatrix{Tv, Ti}, θ::Real;
                        backend=DEFAULT_BACKEND, block_size::Int=64,
                        is_strong=nothing) where {Tv, Ti}
    return strength_graph(A, θ, AbsoluteStrength(); backend=backend, block_size=block_size, is_strong=is_strong)
end

"""
    strength_graph(A, θ, ::AbsoluteStrength)

Absolute-value strength: |a_{i,j}| ≥ θ * max_{k≠i} |a_{i,k}|.
Uses a KA kernel for GPU compatibility. The result array type matches the
backend (GPU arrays for GPU backends, CPU arrays for CPU backends).
"""
function strength_graph(A::CSRMatrix{Tv, Ti}, θ::Real, ::AbsoluteStrength;
                        backend=DEFAULT_BACKEND, block_size::Int=64,
                        is_strong=nothing) where {Tv, Ti}
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    be = _get_backend(nzv)
    if is_strong === nothing
        is_strong_buf = KernelAbstractions.zeros(be, Bool, nnz(A))
    else
        resize!(is_strong, nnz(A))
        fill!(is_strong, false)
        is_strong_buf = is_strong
    end
    kernel! = absolute_strength_kernel!(be, block_size)
    kernel!(is_strong_buf, nzv, cv, rp, Tv(θ); ndrange=n)
    _synchronize(be)
    return is_strong_buf
end

@kernel function absolute_strength_kernel!(is_strong, @Const(nzval), @Const(colval),
                                           @Const(rp), θ)
    row = @index(Global)
    @inbounds begin
        # Find maximum off-diagonal magnitude
        max_offdiag = zero(real(eltype(nzval)))
        for nz in rp[row]:(rp[row+1]-1)
            col = colval[nz]
            if col != row
                max_offdiag = max(max_offdiag, abs(nzval[nz]))
            end
        end
        threshold = θ * max_offdiag
        # Mark strong connections
        for nz in rp[row]:(rp[row+1]-1)
            col = colval[nz]
            if col != row
                is_strong[nz] = abs(nzval[nz]) >= threshold
            else
                is_strong[nz] = false
            end
        end
    end
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
Uses a KA kernel for GPU compatibility.
"""
function strength_graph(A::CSRMatrix{Tv, Ti}, θ::Real, ::SignedStrength;
                        backend=DEFAULT_BACKEND, block_size::Int=64,
                        is_strong=nothing) where {Tv, Ti}
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    be = _get_backend(nzv)
    if is_strong === nothing
        is_strong_buf = KernelAbstractions.zeros(be, Bool, nnz(A))
    else
        resize!(is_strong, nnz(A))
        fill!(is_strong, false)
        is_strong_buf = is_strong
    end
    kernel! = signed_strength_kernel!(be, block_size)
    kernel!(is_strong_buf, nzv, cv, rp, Tv(θ); ndrange=n)
    _synchronize(be)
    return is_strong_buf
end

@kernel function signed_strength_kernel!(is_strong, @Const(nzval), @Const(colval),
                                         @Const(rp), θ)
    row = @index(Global)
    @inbounds begin
        # Find diagonal
        a_ii = zero(eltype(nzval))
        for nz in rp[row]:(rp[row+1]-1)
            if colval[nz] == row
                a_ii = nzval[nz]
                break
            end
        end
        diag_sign = sign(real(a_ii))
        if diag_sign == 0
            diag_sign = one(real(eltype(nzval)))
        end
        # Find max magnitude among opposite-sign off-diagonals
        max_opposite = zero(real(eltype(nzval)))
        has_opposite = false
        for nz in rp[row]:(rp[row+1]-1)
            col = colval[nz]
            if col != row && sign(real(nzval[nz])) != diag_sign
                max_opposite = max(max_opposite, abs(nzval[nz]))
                has_opposite = true
            end
        end
        if has_opposite
            threshold = θ * max_opposite
            for nz in rp[row]:(rp[row+1]-1)
                col = colval[nz]
                if col != row && sign(real(nzval[nz])) != diag_sign
                    is_strong[nz] = abs(nzval[nz]) >= threshold
                else
                    is_strong[nz] = false
                end
            end
        else
            # Fallback: no opposite-sign connections, use absolute value
            max_offdiag = zero(real(eltype(nzval)))
            for nz in rp[row]:(rp[row+1]-1)
                col = colval[nz]
                if col != row
                    max_offdiag = max(max_offdiag, abs(nzval[nz]))
                end
            end
            threshold = θ * max_offdiag
            for nz in rp[row]:(rp[row+1]-1)
                col = colval[nz]
                if col != row
                    is_strong[nz] = abs(nzval[nz]) >= threshold
                else
                    is_strong[nz] = false
                end
            end
        end
    end
end

"""
    strength_graph(A, θ, config::AMGConfig)

Dispatch strength computation based on config's strength_type.
"""
function strength_graph(A::CSRMatrix, θ::Real, config::AMGConfig;
                        backend=DEFAULT_BACKEND, block_size::Int=64,
                        is_strong=nothing)
    return strength_graph(A, θ, config.strength_type; backend=backend, block_size=block_size, is_strong=is_strong)
end

"""
    strong_neighbors(A, is_strong, row)

Return an iterator of strongly connected column indices for a given row.
"""
function strong_neighbors(A::CSRMatrix, is_strong::AbstractVector{Bool}, row::Integer)
    cv = colvals(A)
    return (cv[nz] for nz in nzrange(A, row) if is_strong[nz])
end

"""
    _apply_max_row_sum(A, threshold)

Apply max row sum dependency weakening (as in hypre). For rows where
|row_sum| > |a_{i,i}| * threshold (and threshold < 1), all off-diagonal
entries are zeroed out, making all dependencies weak. Here row_sum is the
algebraic sum of all entries in the row (diagonal + off-diagonal).

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
        diag = zero(Tv)
        row_sum = zero(Tv)
        for nz in rp[i]:(rp[i+1]-1)
            j = cv[nz]
            row_sum += nzv_old[nz]
            if j == i
                diag = nzv_old[nz]
            end
        end
        abs_diag = abs(diag)
        abs_diag < eps(real(Tv)) && continue
        # Check if |row_sum| > |diag| * threshold and threshold < 1.0 (as in hypre)
        if abs(row_sum) > abs_diag * threshold && threshold < one(real(Tv))
            # Make all dependencies weak: zero out off-diagonal entries
            for nz in rp[i]:(rp[i+1]-1)
                j = cv[nz]
                if j != i
                    nzv_new[nz] = zero(Tv)
                end
            end
        end
    end
    return CSRMatrix(collect(rp), collect(cv), nzv_new, size(A, 1), size(A, 2))
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
