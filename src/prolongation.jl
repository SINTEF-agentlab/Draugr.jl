"""
    build_prolongation(A, agg, n_coarse)

Build a piecewise-constant prolongation operator from the aggregation map.
Each fine node i is interpolated from aggregate `agg[i]` with weight 1.
Returns a `ProlongationOp`.
"""
function build_prolongation(A::StaticSparsityMatrixCSR{Tv, Ti}, agg::Vector{Int},
                            n_coarse::Int) where {Tv, Ti}
    n_fine = size(A, 1)
    # P is n_fine × n_coarse, with exactly one nonzero per row (aggregation-based)
    rowptr = Vector{Ti}(undef, n_fine + 1)
    colval = Vector{Ti}(undef, n_fine)
    nzval = Vector{Tv}(undef, n_fine)
    @inbounds for i in 1:n_fine
        rowptr[i] = Ti(i)
        colval[i] = Ti(agg[i])
        nzval[i] = one(Tv)
    end
    rowptr[n_fine + 1] = Ti(n_fine + 1)
    return ProlongationOp{Ti, Tv}(rowptr, colval, nzval, n_fine, n_coarse)
end

# ══════════════════════════════════════════════════════════════════════════════
# Classical interpolation methods for CF-splitting based coarsening
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_cf_prolongation(A, cf, coarse_map, n_coarse, interp)

Build a prolongation operator from a CF-splitting using the specified interpolation method.
- `cf[i] == 1` → coarse point, `cf[i] == -1` → fine point
- `coarse_map[i]` → coarse-grid index for coarse points
"""
function build_cf_prolongation(A::StaticSparsityMatrixCSR{Tv, Ti}, cf::Vector{Int},
                               coarse_map::Vector{Int}, n_coarse::Int,
                               interp::InterpolationType) where {Tv, Ti}
    return _build_interpolation(A, cf, coarse_map, n_coarse, interp)
end

# ── Direct interpolation ─────────────────────────────────────────────────────

"""
    _build_interpolation(A, cf, coarse_map, n_coarse, ::DirectInterpolation)

Direct interpolation: for each fine point i, interpolate only from directly
connected strong coarse neighbors. Weak and fine connections are lumped
into the diagonal.

P[i, coarse_map[i]] = 1 for coarse points.
P[i, coarse_map[j]] = -a_{i,j} / d_i for fine points, where j ∈ C_i^s and
d_i = a_{i,i} + Σ_{k ∈ weak ∪ F_i^s} a_{i,k}.
"""
function _build_interpolation(A::StaticSparsityMatrixCSR{Tv, Ti}, cf::Vector{Int},
                              coarse_map::Vector{Int}, n_coarse::Int,
                              ::DirectInterpolation) where {Tv, Ti}
    n_fine = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)
    is_strong = strength_graph(A, zero(Float64))  # recompute, but any entry is "strong" here
    # Actually, we need the real strength. Let's recompute with a standard threshold.
    # We use the threshold from the strength graph used during coarsening.
    # For simplicity, use θ=0.25 (standard). But actually we don't need strength here;
    # we classify connections based on the CF-splitting and use all coarse connections.
    # Direct interpolation uses strong coarse neighbors. Recompute strength:
    is_strong = strength_graph(A, 0.25)

    # First pass: count entries per row
    row_counts = zeros(Int, n_fine)
    @inbounds for i in 1:n_fine
        if cf[i] == 1
            row_counts[i] = 1  # coarse point: identity mapping
        else
            # Fine point: count strong coarse neighbors
            for nz in nzrange(A, i)
                j = cv[nz]
                j == i && continue
                if is_strong[nz] && cf[j] == 1
                    row_counts[i] += 1
                end
            end
            if row_counts[i] == 0
                row_counts[i] = 1  # fallback: map to nearest coarse point
            end
        end
    end

    # Build rowptr
    total_nnz = sum(row_counts)
    rp = Vector{Ti}(undef, n_fine + 1)
    rp[1] = Ti(1)
    for i in 1:n_fine
        rp[i+1] = rp[i] + Ti(row_counts[i])
    end

    cval = Vector{Ti}(undef, total_nnz)
    nzv_p = Vector{Tv}(undef, total_nnz)

    # Second pass: fill entries
    @inbounds for i in 1:n_fine
        pos = rp[i]
        if cf[i] == 1
            cval[pos] = Ti(coarse_map[i])
            nzv_p[pos] = one(Tv)
        else
            # Compute diagonal correction: d_i = a_{i,i} + Σ weak/fine connections
            a_ii = zero(Tv)
            sum_nonC = zero(Tv)
            strong_coarse_cols = Ti[]
            strong_coarse_vals = Tv[]
            for nz in nzrange(A, i)
                j = cv[nz]
                if j == i
                    a_ii = nzv[nz]
                elseif is_strong[nz] && cf[j] == 1
                    push!(strong_coarse_cols, Ti(coarse_map[j]))
                    push!(strong_coarse_vals, nzv[nz])
                else
                    sum_nonC += nzv[nz]
                end
            end
            d_i = a_ii + sum_nonC
            if isempty(strong_coarse_cols)
                # Fallback: assign to nearest coarse neighbor (any connection)
                best_j = 0
                best_v = zero(real(Tv))
                for nz in nzrange(A, i)
                    j = cv[nz]
                    j == i && continue
                    if cf[j] == 1 && abs(nzv[nz]) > best_v
                        best_v = abs(nzv[nz])
                        best_j = coarse_map[j]
                    end
                end
                if best_j == 0
                    best_j = 1  # absolute fallback
                end
                cval[pos] = Ti(best_j)
                nzv_p[pos] = one(Tv)
            else
                for k in eachindex(strong_coarse_cols)
                    cval[pos] = strong_coarse_cols[k]
                    nzv_p[pos] = abs(d_i) > eps(Tv) ? -strong_coarse_vals[k] / d_i : zero(Tv)
                    pos += 1
                end
            end
        end
    end

    return ProlongationOp{Ti, Tv}(rp, cval, nzv_p, n_fine, n_coarse)
end

# ── Standard (Classical) interpolation ───────────────────────────────────────

"""
    _build_interpolation(A, cf, coarse_map, n_coarse, ::StandardInterpolation)

Standard (classical Ruge-Stüben) interpolation. For each fine point i:
- Strong coarse neighbors contribute directly
- Strong fine neighbors contribute indirectly through their coarse connections

w_j = -(a_{i,j} + Σ_{k∈F_i^s} a_{i,k} * a_{k,j} / Σ_{m∈C_i} a_{k,m}) / d_i
where d_i = a_{i,i} + Σ_{k∈weak} a_{i,k}
"""
function _build_interpolation(A::StaticSparsityMatrixCSR{Tv, Ti}, cf::Vector{Int},
                              coarse_map::Vector{Int}, n_coarse::Int,
                              ::StandardInterpolation) where {Tv, Ti}
    n_fine = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)
    is_strong = strength_graph(A, 0.25)

    # Build P using COO format, then convert to CSR
    I_p = Ti[]
    J_p = Ti[]
    V_p = Tv[]

    @inbounds for i in 1:n_fine
        if cf[i] == 1
            push!(I_p, Ti(i)); push!(J_p, Ti(coarse_map[i])); push!(V_p, one(Tv))
            continue
        end
        # Classify connections of fine point i
        a_ii = zero(Tv)
        sum_weak = zero(Tv)
        strong_coarse = Dict{Int, Tv}()  # coarse_map[j] → a_{i,j}
        strong_fine = Tuple{Int, Tv}[]   # (fine node k, a_{i,k})
        for nz in nzrange(A, i)
            j = cv[nz]
            if j == i
                a_ii = nzv[nz]
            elseif is_strong[nz] && cf[j] == 1
                cm = coarse_map[j]
                strong_coarse[cm] = get(strong_coarse, cm, zero(Tv)) + nzv[nz]
            elseif is_strong[nz] && cf[j] == -1
                push!(strong_fine, (j, nzv[nz]))
            else
                sum_weak += nzv[nz]
            end
        end
        d_i = a_ii + sum_weak
        # Add indirect contributions from strong fine neighbors
        contributions = Dict{Int, Tv}()
        for (cm, a_ij) in strong_coarse
            contributions[cm] = a_ij
        end
        for (k, a_ik) in strong_fine
            # Find coarse connections of fine point k
            sum_C_k = zero(Tv)
            coarse_vals_k = Dict{Int, Tv}()
            for nz2 in nzrange(A, k)
                j2 = cv[nz2]
                if j2 != k && cf[j2] == 1
                    cm2 = coarse_map[j2]
                    # Only distribute to coarse points in C_i
                    if haskey(strong_coarse, cm2)
                        coarse_vals_k[cm2] = get(coarse_vals_k, cm2, zero(Tv)) + nzv[nz2]
                        sum_C_k += nzv[nz2]
                    end
                end
            end
            if abs(sum_C_k) > eps(Tv)
                for (cm2, a_kj) in coarse_vals_k
                    indirect = a_ik * a_kj / sum_C_k
                    contributions[cm2] = get(contributions, cm2, zero(Tv)) + indirect
                end
            else
                # Lump into diagonal
                d_i += a_ik
            end
        end
        if isempty(contributions)
            # Fallback: map to nearest coarse point
            best_j = _find_nearest_coarse(A, i, cf, coarse_map)
            push!(I_p, Ti(i)); push!(J_p, Ti(best_j)); push!(V_p, one(Tv))
        else
            for (cm, val) in contributions
                w = abs(d_i) > eps(Tv) ? -val / d_i : zero(Tv)
                push!(I_p, Ti(i)); push!(J_p, Ti(cm)); push!(V_p, w)
            end
        end
    end

    return _coo_to_prolongation(I_p, J_p, V_p, n_fine, n_coarse)
end

# ── Extended+i interpolation ─────────────────────────────────────────────────

"""
    _build_interpolation(A, cf, coarse_map, n_coarse, ::ExtendedIInterpolation)

Extended+i interpolation. Extends standard interpolation by including distance-2
coarse points (coarse points connected through fine neighbors) as direct
interpolation targets, resulting in a larger but more accurate interpolation stencil.
"""
function _build_interpolation(A::StaticSparsityMatrixCSR{Tv, Ti}, cf::Vector{Int},
                              coarse_map::Vector{Int}, n_coarse::Int,
                              ::ExtendedIInterpolation) where {Tv, Ti}
    n_fine = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)
    is_strong = strength_graph(A, 0.25)

    I_p = Ti[]
    J_p = Ti[]
    V_p = Tv[]

    @inbounds for i in 1:n_fine
        if cf[i] == 1
            push!(I_p, Ti(i)); push!(J_p, Ti(coarse_map[i])); push!(V_p, one(Tv))
            continue
        end
        # Classify connections
        a_ii = zero(Tv)
        sum_weak = zero(Tv)
        strong_coarse_direct = Dict{Int, Tv}()
        strong_fine = Tuple{Int, Tv}[]
        for nz in nzrange(A, i)
            j = cv[nz]
            if j == i
                a_ii = nzv[nz]
            elseif is_strong[nz] && cf[j] == 1
                cm = coarse_map[j]
                strong_coarse_direct[cm] = get(strong_coarse_direct, cm, zero(Tv)) + nzv[nz]
            elseif is_strong[nz] && cf[j] == -1
                push!(strong_fine, (j, nzv[nz]))
            else
                sum_weak += nzv[nz]
            end
        end
        d_i = a_ii + sum_weak
        # Extended set: include distance-2 coarse points through strong fine neighbors
        extended_coarse = Dict{Int, Tv}()
        for (cm, val) in strong_coarse_direct
            extended_coarse[cm] = val
        end
        for (k, a_ik) in strong_fine
            # Add coarse connections of fine point k (distance-2 coarse)
            sum_C_k = zero(Tv)
            coarse_vals_k = Dict{Int, Tv}()
            for nz2 in nzrange(A, k)
                j2 = cv[nz2]
                if j2 != k && cf[j2] == 1
                    cm2 = coarse_map[j2]
                    coarse_vals_k[cm2] = get(coarse_vals_k, cm2, zero(Tv)) + nzv[nz2]
                    sum_C_k += nzv[nz2]
                end
            end
            if abs(sum_C_k) > eps(Tv)
                for (cm2, a_kj) in coarse_vals_k
                    indirect = a_ik * a_kj / sum_C_k
                    extended_coarse[cm2] = get(extended_coarse, cm2, zero(Tv)) + indirect
                end
            else
                d_i += a_ik
            end
        end
        if isempty(extended_coarse)
            best_j = _find_nearest_coarse(A, i, cf, coarse_map)
            push!(I_p, Ti(i)); push!(J_p, Ti(best_j)); push!(V_p, one(Tv))
        else
            for (cm, val) in extended_coarse
                w = abs(d_i) > eps(Tv) ? -val / d_i : zero(Tv)
                push!(I_p, Ti(i)); push!(J_p, Ti(cm)); push!(V_p, w)
            end
        end
    end

    return _coo_to_prolongation(I_p, J_p, V_p, n_fine, n_coarse)
end

# ── Helpers ──────────────────────────────────────────────────────────────────

"""Find nearest coarse point for fallback interpolation."""
function _find_nearest_coarse(A::StaticSparsityMatrixCSR{Tv, Ti}, i::Int,
                              cf::Vector{Int}, coarse_map::Vector{Int}) where {Tv, Ti}
    cv = colvals(A)
    nzv = nonzeros(A)
    best_j = 0
    best_v = zero(real(Tv))
    for nz in nzrange(A, i)
        j = cv[nz]
        j == i && continue
        if cf[j] == 1 && abs(nzv[nz]) > best_v
            best_v = abs(nzv[nz])
            best_j = coarse_map[j]
        end
    end
    return best_j > 0 ? best_j : 1
end

"""Convert COO format to ProlongationOp (CSR)."""
function _coo_to_prolongation(I_p::Vector{Ti}, J_p::Vector{Ti}, V_p::Vector{Tv},
                              n_fine::Int, n_coarse::Int) where {Ti, Tv}
    # Sort by (row, col)
    perm = sortperm(collect(zip(I_p, J_p)))
    I_s = I_p[perm]
    J_s = J_p[perm]
    V_s = V_p[perm]
    nnz_p = length(I_s)
    rp = Vector{Ti}(undef, n_fine + 1)
    fill!(rp, Ti(0))
    for k in 1:nnz_p
        rp[I_s[k]] += Ti(1)
    end
    cumsum = Ti(1)
    for i in 1:n_fine
        count = rp[i]
        rp[i] = cumsum
        cumsum += count
    end
    rp[n_fine + 1] = cumsum
    return ProlongationOp{Ti, Tv}(rp, J_s, V_s, n_fine, n_coarse)
end

"""
    prolongate!(x_fine, P, x_coarse)

Apply prolongation: x_fine += P * x_coarse.
Uses KernelAbstractions for parallel execution over fine rows.
"""
function prolongate!(x_fine::AbstractVector, P::ProlongationOp, x_coarse::AbstractVector;
                     backend=CPU())
    kernel! = prolongate_kernel!(backend, 64)
    kernel!(x_fine, P.rowptr, P.colval, P.nzval, x_coarse; ndrange=P.nrow)
    KernelAbstractions.synchronize(backend)
    return x_fine
end

@kernel function prolongate_kernel!(x_fine, @Const(P_rowptr), @Const(P_colval),
                                    @Const(P_nzval), @Const(x_coarse))
    i = @index(Global)
    @inbounds begin
        for nz in P_rowptr[i]:(P_rowptr[i+1]-1)
            j = P_colval[nz]
            x_fine[i] += P_nzval[nz] * x_coarse[j]
        end
    end
end

"""
    restrict!(b_coarse, P, r_fine)

Apply restriction (P^T): b_coarse = P^T * r_fine.
For aggregation-based P (one nonzero per row), this is race-free when
parallelized over fine rows using atomics.
"""
function restrict!(b_coarse::AbstractVector, P::ProlongationOp, r_fine::AbstractVector;
                   backend=CPU())
    fill!(b_coarse, zero(eltype(b_coarse)))
    kernel! = restrict_kernel!(backend, 64)
    kernel!(b_coarse, P.rowptr, P.colval, P.nzval, r_fine; ndrange=P.nrow)
    KernelAbstractions.synchronize(backend)
    return b_coarse
end

@kernel function restrict_kernel!(b_coarse, @Const(P_rowptr), @Const(P_colval),
                                  @Const(P_nzval), @Const(r_fine))
    i = @index(Global)
    @inbounds begin
        for nz in P_rowptr[i]:(P_rowptr[i+1]-1)
            j = P_colval[nz]
            Atomix.@atomic b_coarse[j] += P_nzval[nz] * r_fine[i]
        end
    end
end
