"""
    build_prolongation(A, agg, n_coarse)

Build a piecewise-constant prolongation operator from the aggregation map.
Each fine node i is interpolated from aggregate `agg[i]` with weight 1.
Returns a `ProlongationOp`.
"""
function build_prolongation(A::CSRMatrix{Tv, Ti}, agg::Vector{Int},
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
# Smoothed aggregation: P_smooth = (I - ω D⁻¹ A) P_tent
# ══════════════════════════════════════════════════════════════════════════════

"""
    _smooth_prolongation(A, P_tent, ω)

Smooth a tentative (piecewise-constant) prolongation operator using a damped
Jacobi step: P = (I - ω D⁻¹ A) P_tent.

The result has the sparsity pattern of A * P_tent (union of P_tent sparsity and
one ring of neighbors through A). Uses bounded per-row storage to avoid excessive
memory use on large models.
"""
function _smooth_prolongation(A::CSRMatrix{Tv, Ti},
                              P_tent::ProlongationOp{Ti, Tv},
                              ω::Real) where {Tv, Ti}
    # Convert to CPU for scalar indexing operations
    A_cpu = csr_to_cpu(A)
    n_fine = P_tent.nrow
    n_coarse = P_tent.ncol
    cv_a = colvals(A_cpu)
    nzv_a = nonzeros(A_cpu)
    rp_a = rowptr(A_cpu)

    # Compute inverse diagonal of A
    invdiag = Vector{Tv}(undef, n_fine)
    @inbounds for i in 1:n_fine
        d = zero(Tv)
        for nz in rp_a[i]:(rp_a[i+1]-1)
            if cv_a[nz] == i
                d = nzv_a[nz]
                break
            end
        end
        invdiag[i] = _safe_inv_diag(d, abs(d))
    end

    # Build the smoothed P in COO format
    # P_smooth[i, J] = P_tent[i, J] - ω * invdiag[i] * Σ_j a_{i,j} * P_tent[j, J]
    I_p = Ti[]
    J_p = Ti[]
    V_p = Tv[]

    # Pre-compute the set of coarse columns reachable from each fine row
    # to avoid Dict overhead on repeated lookups
    @inbounds for i in 1:n_fine
        # Collect contributions for row i using a sorted-keys approach
        row_entries = Dict{Int, Tv}()

        # Term 1: P_tent[i, :]
        for pnz in P_tent.rowptr[i]:(P_tent.rowptr[i+1]-1)
            J = Int(P_tent.colval[pnz])
            row_entries[J] = get(row_entries, J, zero(Tv)) + P_tent.nzval[pnz]
        end

        # Term 2: -ω * invdiag[i] * Σ_j a_{i,j} * P_tent[j, :]
        factor = -Tv(ω) * invdiag[i]
        for anz in rp_a[i]:(rp_a[i+1]-1)
            j = cv_a[anz]
            # Bounds check: j must be valid row of P_tent
            (j < 1 || j > P_tent.nrow) && continue
            a_ij = nzv_a[anz]
            w = factor * a_ij
            for pnz in P_tent.rowptr[j]:(P_tent.rowptr[j+1]-1)
                J = Int(P_tent.colval[pnz])
                row_entries[J] = get(row_entries, J, zero(Tv)) + w * P_tent.nzval[pnz]
            end
        end

        for (J, val) in row_entries
            # Drop near-zero entries to control sparsity on large models
            if abs(val) > eps(real(Tv))
                push!(I_p, Ti(i))
                push!(J_p, Ti(J))
                push!(V_p, val)
            end
        end
        # If entire row was dropped, preserve at least one entry from P_tent
        found_i = false
        for k in max(1, length(I_p) - length(row_entries) + 1):length(I_p)
            if I_p[k] == Ti(i)
                found_i = true
                break
            end
        end
        if !found_i && P_tent.rowptr[i] <= P_tent.rowptr[i+1] - 1
            # Fallback: keep the tent entry
            pnz = P_tent.rowptr[i]
            push!(I_p, Ti(i))
            push!(J_p, P_tent.colval[pnz])
            push!(V_p, P_tent.nzval[pnz])
        end
    end

    return _coo_to_prolongation(I_p, J_p, V_p, n_fine, n_coarse)
end

# ══════════════════════════════════════════════════════════════════════════════
# Prolongation filtering
# ══════════════════════════════════════════════════════════════════════════════

"""
    _filter_prolongation(P, tol)

Filter (drop) small entries from the prolongation operator P. For each row i,
entries with |p_{i,j}| < tol * max_j |p_{i,j}| are dropped. Remaining entries
are rescaled so each row sums to 1 (for tentative/aggregation P) or preserves
coarse-point identity mappings.
"""
function _filter_prolongation(P::ProlongationOp{Ti, Tv}, tol::Real) where {Ti, Tv}
    n_fine = P.nrow
    n_coarse = P.ncol

    I_p = Ti[]
    J_p = Ti[]
    V_p = Tv[]

    @inbounds for i in 1:n_fine
        rstart = P.rowptr[i]
        rend = P.rowptr[i+1] - 1
        rstart > rend && continue

        # Find max absolute value in this row
        max_val = zero(real(Tv))
        for nz in rstart:rend
            max_val = max(max_val, abs(P.nzval[nz]))
        end
        threshold = Tv(tol) * max_val

        # Collect entries above threshold
        row_count = 0
        for nz in rstart:rend
            if abs(P.nzval[nz]) >= threshold
                push!(I_p, Ti(i))
                push!(J_p, P.colval[nz])
                push!(V_p, P.nzval[nz])
                row_count += 1
            end
        end

        # If all entries were dropped, keep the largest
        if row_count == 0
            best_nz = rstart
            best_val = zero(real(Tv))
            for nz in rstart:rend
                if abs(P.nzval[nz]) > best_val
                    best_val = abs(P.nzval[nz])
                    best_nz = nz
                end
            end
            push!(I_p, Ti(i))
            push!(J_p, P.colval[best_nz])
            push!(V_p, P.nzval[best_nz])
        end
    end

    return _coo_to_prolongation(I_p, J_p, V_p, n_fine, n_coarse)
end

# ══════════════════════════════════════════════════════════════════════════════
# Classical interpolation methods for CF-splitting based coarsening
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_cf_prolongation(A, cf, coarse_map, n_coarse, interp, θ; build_update_map=false)

Build a prolongation operator from a CF-splitting using the specified interpolation method.
- `cf[i] == 1` → coarse point, `cf[i] == -1` → fine point
- `coarse_map[i]` → coarse-grid index for coarse points
- `θ` → strength threshold used consistently for interpolation stencil selection
- `build_update_map` → if true, also returns a `ProlongationUpdateMap` for in-place value update

Returns `(P, P_update_map)` where P_update_map is `nothing` if `build_update_map=false`.
"""
function build_cf_prolongation(A::CSRMatrix{Tv, Ti}, cf::Vector{Int},
                               coarse_map::Vector{Int}, n_coarse::Int,
                               interp::InterpolationType, θ::Real=0.25;
                               backend=DEFAULT_BACKEND, block_size::Int=64,
                               setup_workspace=nothing,
                               build_update_map::Bool=false) where {Tv, Ti}
    return _build_interpolation(A, cf, coarse_map, n_coarse, interp, θ; backend=backend, block_size=block_size, setup_workspace=setup_workspace, build_update_map=build_update_map)
end

# ── Direct interpolation ─────────────────────────────────────────────────────

"""
    _build_interpolation(A, cf, coarse_map, n_coarse, ::DirectInterpolation)

Direct interpolation: for each fine point i, interpolate only from directly
connected strong coarse neighbors. Weak and fine connections are lumped
into the diagonal.

Handles "wrong"-sign off-diagonals (positive off-diags when diagonal is positive):
such connections are treated as weak and lumped into the diagonal correction.

P[i, coarse_map[i]] = 1 for coarse points.
P[i, coarse_map[j]] = -a_{i,j} / d_i for fine points, where j ∈ C_i^s and
d_i = a_{i,i} + Σ_{k ∈ weak ∪ F_i^s ∪ same_sign} a_{i,k}.

When `build_update_map=true`, also builds a `ProlongationUpdateMap` that stores
all index mappings needed for in-place P value update with KernelAbstractions.
"""
function _build_interpolation(A_in::CSRMatrix{Tv, Ti}, cf::Vector{Int},
                              coarse_map::Vector{Int}, n_coarse::Int,
                              ::DirectInterpolation, θ::Real=0.25;
                              backend=DEFAULT_BACKEND, block_size::Int=64,
                              setup_workspace=nothing,
                              build_update_map::Bool=false) where {Tv, Ti}
    # Compute strength on GPU if available, then convert to CPU for graph algorithms
    is_strong_raw = strength_graph(A_in, θ; backend=backend, block_size=block_size,
        is_strong=setup_workspace !== nothing ? setup_workspace.is_strong : nothing)
    is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    A = csr_to_cpu(A_in)
    n_fine = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)
    rp_A = rowptr(A)

    # First pass: count entries per row and determine sign classification
    row_counts = zeros(Int, n_fine)
    diag_positive = Vector{Bool}(undef, n_fine)
    diag_nz_idx = Vector{Ti}(undef, n_fine)
    
    @inbounds for i in 1:n_fine
        diag_nz_idx[i] = Ti(0)
        if cf[i] == 1
            row_counts[i] = 1  # coarse point: identity mapping
            diag_positive[i] = true
            for nz in nzrange(A, i)
                if cv[nz] == i
                    diag_nz_idx[i] = Ti(nz)
                    break
                end
            end
        else
            # Find diagonal sign and index
            a_ii = zero(Tv)
            for nz in nzrange(A, i)
                if cv[nz] == i
                    a_ii = nzv[nz]
                    diag_nz_idx[i] = Ti(nz)
                    break
                end
            end
            diag_positive[i] = real(a_ii) >= 0
            diag_sign = sign(real(a_ii))
            # Fine point: count strong coarse neighbors with opposite sign
            for nz in nzrange(A, i)
                j = cv[nz]
                j == i && continue
                if is_strong[nz] && cf[j] == 1
                    # Only interpolate from opposite-sign connections
                    if diag_sign == 0 || sign(real(nzv[nz])) != diag_sign
                        row_counts[i] += 1
                    end
                end
            end
            if row_counts[i] == 0
                row_counts[i] = 1  # fallback: map to nearest coarse point
            end
        end
    end

    # Build rowptr — reuse old_P arrays when available
    total_nnz = sum(row_counts)
    old_P_reuse = setup_workspace !== nothing ? setup_workspace.old_P : nothing
    if old_P_reuse !== nothing && old_P_reuse.colval isa Vector
        rp = resize!(old_P_reuse.rowptr, n_fine + 1)
        cval = resize!(old_P_reuse.colval, total_nnz)
        nzv_p = resize!(old_P_reuse.nzval, total_nnz)
    else
        rp = Vector{Ti}(undef, n_fine + 1)
        cval = Vector{Ti}(undef, total_nnz)
        nzv_p = Vector{Tv}(undef, total_nnz)
    end
    rp[1] = Ti(1)
    for i in 1:n_fine
        rp[i+1] = rp[i] + Ti(row_counts[i])
    end

    # Prepare update map arrays if requested
    if build_update_map
        entry_type = Vector{Ti}(undef, total_nnz)
        numer_idx = Vector{Ti}(undef, total_nnz)
        denom_offsets = Vector{Ti}(undef, total_nnz + 1)
        denom_entries_list = Ti[]
        # For Standard/Extended+i: strong neighbor structure
        strong_nbrs_offsets = Vector{Ti}(undef, n_fine + 1)
        strong_nbrs_cols_list = Ti[]
        strong_nbrs_nz_list = Ti[]
    end

    # Second pass: fill entries and build update map
    @inbounds for i in 1:n_fine
        pos = rp[i]
        
        # Build strong neighbor list if update map requested
        if build_update_map
            strong_nbrs_offsets[i] = Ti(length(strong_nbrs_cols_list) + 1)
            for nz in nzrange(A, i)
                j = cv[nz]
                if j != i && is_strong[nz]
                    push!(strong_nbrs_cols_list, Ti(j))
                    push!(strong_nbrs_nz_list, Ti(nz))
                end
            end
        end
        
        if cf[i] == 1
            cval[pos] = Ti(coarse_map[i])
            nzv_p[pos] = one(Tv)
            if build_update_map
                entry_type[pos] = Ti(0)  # coarse point: P=1
                numer_idx[pos] = Ti(0)
                denom_offsets[pos] = Ti(length(denom_entries_list) + 1)
            end
        else
            diag_sign = diag_positive[i] ? 1 : -1
            # Collect strong coarse columns and denominator entries
            strong_coarse_cols = Vector{Ti}()
            strong_coarse_nz_idx = Vector{Ti}()  # A.nzval indices for numerators
            denom_nz_idx = Vector{Ti}()          # A.nzval indices for denominator
            
            for nz in nzrange(A, i)
                j = cv[nz]
                if j == i
                    # Diagonal always in denominator
                    push!(denom_nz_idx, Ti(nz))
                else
                    is_interp_coarse = is_strong[nz] && cf[j] == 1 &&
                        sign(real(nzv[nz])) != diag_sign
                    if is_interp_coarse
                        push!(strong_coarse_cols, Ti(coarse_map[j]))
                        push!(strong_coarse_nz_idx, Ti(nz))
                    else
                        # Weak/fine connections go to denominator
                        push!(denom_nz_idx, Ti(nz))
                    end
                end
            end
            
            # Compute d_i for this row
            d_i = zero(Tv)
            for nz_idx in denom_nz_idx
                d_i += nzv[nz_idx]
            end
            
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
                if build_update_map
                    entry_type[pos] = Ti(0)  # fallback = P=1
                    numer_idx[pos] = Ti(0)
                    denom_offsets[pos] = Ti(length(denom_entries_list) + 1)
                end
            else
                for k in eachindex(strong_coarse_cols)
                    cval[pos] = strong_coarse_cols[k]
                    abs_d_i = abs(d_i)
                    nzv_p[pos] = abs_d_i > _safe_threshold(Tv, abs_d_i) ? -nzv[strong_coarse_nz_idx[k]] / d_i : zero(Tv)
                    if build_update_map
                        entry_type[pos] = Ti(1)  # fine point: compute from A
                        numer_idx[pos] = strong_coarse_nz_idx[k]
                        # Denominator is the same for all P entries in this row
                        denom_offsets[pos] = Ti(length(denom_entries_list) + 1)
                        append!(denom_entries_list, denom_nz_idx)
                    end
                    pos += 1
                end
            end
        end
    end

    # Finalize update map
    P_update_map = nothing
    if build_update_map
        denom_offsets[total_nnz + 1] = Ti(length(denom_entries_list) + 1)
        strong_nbrs_offsets[n_fine + 1] = Ti(length(strong_nbrs_cols_list) + 1)
        denom_entries = Vector{Ti}(denom_entries_list)
        strong_nbrs_cols = Vector{Ti}(strong_nbrs_cols_list)
        strong_nbrs_nz = Vector{Ti}(strong_nbrs_nz_list)
        # Direct interpolation doesn't need workspace, but we include empty buffers
        P_update_map = ProlongationUpdateMap{Ti, Tv}(
            1,  # interp_type = Direct
            copy(is_strong),
            copy(cf),
            copy(coarse_map),
            diag_nz_idx,
            entry_type,
            numer_idx,
            denom_offsets,
            denom_entries,
            strong_nbrs_offsets,
            strong_nbrs_cols,
            strong_nbrs_nz,
            Int[],  # P_marker (not used for Direct)
            Int[],  # chat_indices (not used for Direct)
            Tv[]    # P_data (not used for Direct)
        )
    end

    if old_P_reuse !== nothing && old_P_reuse.colval isa Vector
        old_P_reuse.nrow = n_fine
        old_P_reuse.ncol = n_coarse
        old_P_reuse.trunc_scaling = nothing
        if setup_workspace !== nothing
            setup_workspace.old_P = nothing
        end
        return old_P_reuse, P_update_map
    else
        return ProlongationOp{Ti, Tv}(rp, cval, nzv_p, n_fine, n_coarse), P_update_map
    end
end

# ── Standard (Classical) interpolation ───────────────────────────────────────

"""
    _build_interpolation(A, cf, coarse_map, n_coarse, ::StandardInterpolation)

Standard (classical Ruge-Stüben) interpolation. For each fine point i:
- Strong coarse neighbors contribute directly
- Strong fine neighbors contribute indirectly through their coarse connections

w_j = -(a_{i,j} + Σ_{k∈F_i^s} a_{i,k} * a_{k,j} / Σ_{m∈C_i} a_{k,m}) / d_i
where d_i = a_{i,i} + Σ_{k∈weak} a_{i,k}

When `build_update_map=true`, stores the complete graph structure (is_strong, cf, 
coarse_map, strong neighbors) to enable efficient P value recomputation via
KernelAbstractions kernels.
"""
function _build_interpolation(A_in::CSRMatrix{Tv, Ti}, cf::Vector{Int},
                              coarse_map::Vector{Int}, n_coarse::Int,
                              ::StandardInterpolation, θ::Real=0.25;
                              backend=DEFAULT_BACKEND, block_size::Int=64,
                              setup_workspace=nothing,
                              build_update_map::Bool=false) where {Tv, Ti}
    # Compute strength on GPU if available, then convert to CPU for graph algorithms
    is_strong_raw = strength_graph(A_in, θ; backend=backend, block_size=block_size,
        is_strong=setup_workspace !== nothing ? setup_workspace.is_strong : nothing)
    is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    A = csr_to_cpu(A_in)
    n_fine = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)

    # Build strong neighbor structure for update map
    diag_nz_idx = Vector{Ti}(undef, n_fine)
    if build_update_map
        strong_nbrs_offsets = Vector{Ti}(undef, n_fine + 1)
        strong_nbrs_cols_list = Ti[]
        strong_nbrs_nz_list = Ti[]
    end
    
    # First pass: find diagonal indices and build strong neighbor structure
    @inbounds for i in 1:n_fine
        diag_nz_idx[i] = Ti(0)
        for nz in nzrange(A, i)
            if cv[nz] == i
                diag_nz_idx[i] = Ti(nz)
                break
            end
        end
        if build_update_map
            strong_nbrs_offsets[i] = Ti(length(strong_nbrs_cols_list) + 1)
            for nz in nzrange(A, i)
                j = cv[nz]
                if j != i && is_strong[nz]
                    push!(strong_nbrs_cols_list, Ti(j))
                    push!(strong_nbrs_nz_list, Ti(nz))
                end
            end
        end
    end
    if build_update_map
        strong_nbrs_offsets[n_fine + 1] = Ti(length(strong_nbrs_cols_list) + 1)
    end

    # Build P using COO format, then convert to CSR
    nnz_hint = nnz(A)
    if setup_workspace !== nothing
        I_p = setup_workspace.I_p
        J_p = setup_workspace.J_p
        V_p = setup_workspace.V_p
        empty!(I_p); empty!(J_p); empty!(V_p)
        sizehint!(I_p, nnz_hint)
        sizehint!(J_p, nnz_hint)
        sizehint!(V_p, nnz_hint)
    else
        I_p = Ti[]
        J_p = Ti[]
        V_p = Tv[]
        sizehint!(I_p, nnz_hint)
        sizehint!(J_p, nnz_hint)
        sizehint!(V_p, nnz_hint)
    end

    # For update map: store entry types and per-entry numerator/denominator info
    if build_update_map
        entry_types_list = Ti[]
        numer_idx_list = Ti[]  # Single A.nzval index for direct entries
        denom_offsets_list = Ti[1]
        denom_entries_list = Ti[]
    end

    @inbounds for i in 1:n_fine
        if cf[i] == 1
            push!(I_p, Ti(i)); push!(J_p, Ti(coarse_map[i])); push!(V_p, one(Tv))
            if build_update_map
                push!(entry_types_list, Ti(0))  # coarse = P=1
                push!(numer_idx_list, Ti(0))
                push!(denom_offsets_list, Ti(length(denom_entries_list) + 1))
            end
            continue
        end
        # Classify connections of fine point i
        a_ii = zero(Tv)
        diag_nz = diag_nz_idx[i]
        if diag_nz > 0
            a_ii = nzv[diag_nz]
        end
        sum_weak = zero(Tv)
        weak_nz_indices = Ti[]
        strong_coarse = Dict{Int, Tv}()       # coarse_map[j] → a_{i,j}
        strong_coarse_nz = Dict{Int, Ti}()    # coarse_map[j] → A.nzval index
        strong_fine = Tuple{Int, Tv, Ti}[]    # (fine node k, a_{i,k}, nz index)
        
        for nz in nzrange(A, i)
            j = cv[nz]
            if j == i
                continue  # already handled
            elseif is_strong[nz] && cf[j] == 1
                cm = coarse_map[j]
                strong_coarse[cm] = get(strong_coarse, cm, zero(Tv)) + nzv[nz]
                # Store A.nzval index for this direct contribution
                if !haskey(strong_coarse_nz, cm)
                    strong_coarse_nz[cm] = Ti(nz)
                end
            elseif is_strong[nz] && cf[j] == -1
                push!(strong_fine, (j, nzv[nz], Ti(nz)))
            else
                sum_weak += nzv[nz]
                push!(weak_nz_indices, Ti(nz))
            end
        end
        d_i = a_ii + sum_weak
        
        # Add indirect contributions from strong fine neighbors
        contributions = Dict{Int, Tv}()
        for (cm, a_ij) in strong_coarse
            contributions[cm] = a_ij
        end
        for (k, a_ik, nz_ik) in strong_fine
            # Find coarse connections of fine point k
            diag_k = zero(Tv)
            for nz2 in nzrange(A, k)
                if cv[nz2] == k
                    diag_k = nzv[nz2]
                    break
                end
            end
            sgn = real(diag_k) < 0 ? -1 : 1
            sum_C_k = zero(Tv)
            coarse_vals_k = Dict{Int, Tv}()
            for nz2 in nzrange(A, k)
                j2 = cv[nz2]
                j2 == k && continue
                a_kj = nzv[nz2]
                if cf[j2] == 1
                    cm2 = coarse_map[j2]
                    if haskey(strong_coarse, cm2) && sgn * real(a_kj) < 0
                        coarse_vals_k[cm2] = get(coarse_vals_k, cm2, zero(Tv)) + a_kj
                        sum_C_k += a_kj
                    end
                end
                if j2 == i && sgn * real(a_kj) < 0
                    sum_C_k += a_kj
                end
            end
            if abs(sum_C_k) > eps(real(Tv))
                distribute = a_ik / sum_C_k
                for (cm2, a_kj) in coarse_vals_k
                    contributions[cm2] = get(contributions, cm2, zero(Tv)) + distribute * a_kj
                end
                for nz2 in nzrange(A, k)
                    if cv[nz2] == i && sgn * real(nzv[nz2]) < 0
                        d_i += distribute * nzv[nz2]
                        break
                    end
                end
            else
                d_i += a_ik
            end
        end
        
        if isempty(contributions)
            # Fallback: map to nearest coarse point
            best_j = _find_nearest_coarse(A, i, cf, coarse_map)
            push!(I_p, Ti(i)); push!(J_p, Ti(best_j)); push!(V_p, one(Tv))
            if build_update_map
                push!(entry_types_list, Ti(0))  # fallback = P=1
                push!(numer_idx_list, Ti(0))
                push!(denom_offsets_list, Ti(length(denom_entries_list) + 1))
            end
        else
            for (cm, val) in contributions
                w = abs(d_i) > eps(real(Tv)) ? -val / d_i : zero(Tv)
                push!(I_p, Ti(i)); push!(J_p, Ti(cm)); push!(V_p, w)
                if build_update_map
                    # For Standard interpolation, mark as type 2 (needs full recomputation)
                    push!(entry_types_list, Ti(2))
                    # Store the direct A index for this coarse column if available
                    numer_nz = get(strong_coarse_nz, cm, Ti(0))
                    push!(numer_idx_list, numer_nz)
                    # Store denominator indices (diagonal + weak)
                    push!(denom_offsets_list, Ti(length(denom_entries_list) + 1))
                    if diag_nz > 0
                        push!(denom_entries_list, diag_nz)
                    end
                    append!(denom_entries_list, weak_nz_indices)
                end
            end
        end
    end

    old_P_reuse = setup_workspace !== nothing ? setup_workspace.old_P : nothing
    P = _coo_to_prolongation(I_p, J_p, V_p, n_fine, n_coarse;
        old_P=old_P_reuse,
        sort_perm=setup_workspace !== nothing ? setup_workspace.sort_perm : nothing)
    if setup_workspace !== nothing
        setup_workspace.old_P = nothing
    end
    
    # Build update map with full graph structure for Standard interpolation
    P_update_map = nothing
    if build_update_map
        entry_type = Vector{Ti}(entry_types_list)
        numer_idx = Vector{Ti}(numer_idx_list)
        denom_offsets = Vector{Ti}(denom_offsets_list)
        denom_entries = Vector{Ti}(denom_entries_list)
        strong_nbrs_cols = Vector{Ti}(strong_nbrs_cols_list)
        strong_nbrs_nz = Vector{Ti}(strong_nbrs_nz_list)
        
        # Allocate workspace for Standard interpolation update
        # Estimate max P entries per row as max strong neighbors
        max_strong = 0
        for i in 1:n_fine
            num_strong = Int(strong_nbrs_offsets[i+1] - strong_nbrs_offsets[i])
            max_strong = max(max_strong, num_strong)
        end
        
        P_update_map = ProlongationUpdateMap{Ti, Tv}(
            2,  # interp_type = Standard
            copy(is_strong),
            copy(cf),
            copy(coarse_map),
            diag_nz_idx,
            entry_type,
            numer_idx,
            denom_offsets,
            denom_entries,
            strong_nbrs_offsets,
            strong_nbrs_cols,
            strong_nbrs_nz,
            fill(-1, n_fine),         # P_marker
            Vector{Int}(undef, max_strong + 1),  # chat_indices buffer
            Vector{Tv}(undef, max_strong + 1)    # P_data buffer
        )
    end
    
    return P, P_update_map
end

# ── Extended+i interpolation ─────────────────────────────────────────────────

"""
    _build_interpolation(A, cf, coarse_map, n_coarse, ::ExtendedIInterpolation)

Extended+i interpolation. Extends standard interpolation by including distance-2
coarse points (coarse points connected through fine neighbors) as direct
interpolation targets, resulting in a larger but more accurate interpolation stencil.
"""
function _build_interpolation(A_in::CSRMatrix{Tv, Ti}, cf::Vector{Int},
                              coarse_map::Vector{Int}, n_coarse::Int,
                              interp::ExtendedIInterpolation, θ::Real=0.25;
                              backend=DEFAULT_BACKEND, block_size::Int=64,
                              setup_workspace=nothing,
                              build_update_map::Bool=false) where {Tv, Ti}
    # Compute strength on GPU if available, then convert to CPU for graph algorithms
    is_strong_raw = strength_graph(A_in, θ; backend=backend, block_size=block_size,
        is_strong=setup_workspace !== nothing ? setup_workspace.is_strong : nothing)
    is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    A = csr_to_cpu(A_in)
    n_fine = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)

    trunc_factor = interp.trunc_factor
    max_elements = interp.max_elements
    norm_p = interp.norm_p
    do_rescale = interp.rescale

    # Build strength-based CSR structure for determining C-hat.
    # Use flat CSR representation (offsets + data) to avoid allocating Vector{Vector{Int}}.
    if setup_workspace !== nothing
        sn_offsets = setup_workspace.strong_nbrs_offsets
        sn_data = setup_workspace.strong_nbrs_data
    else
        sn_offsets = Vector{Int}(undef, n_fine + 1)
        sn_data = Int[]
    end
    _ws_resize!(sn_offsets, n_fine + 1)
    # Count strong neighbors per row
    @inbounds begin
        for i in 1:n_fine
            cnt = 0
            for nz in nzrange(A, i)
                j = cv[nz]
                if j != i && is_strong[nz]
                    cnt += 1
                end
            end
            sn_offsets[i] = cnt
        end
    end
    # Cumulative sum to build offsets
    total_sn = 0
    @inbounds for i in 1:n_fine
        cnt = sn_offsets[i]
        sn_offsets[i] = total_sn + 1
        total_sn += cnt
    end
    sn_offsets[n_fine + 1] = total_sn + 1
    _ws_resize!(sn_data, total_sn)
    # Fill data
    @inbounds begin
        pos = 0
        for i in 1:n_fine
            for nz in nzrange(A, i)
                j = cv[nz]
                if j != i && is_strong[nz]
                    pos += 1
                    sn_data[pos] = j
                end
            end
        end
    end

    # P_marker tracks which coarse points are in C-hat for current row
    if setup_workspace !== nothing
        P_marker = _ws_resize!(setup_workspace.P_marker, n_fine)
        fill!(P_marker, -1)
    else
        P_marker = fill(-1, n_fine)
    end
    strong_f_marker = -2

    nnz_hint = nnz(A)
    if setup_workspace !== nothing
        I_p = setup_workspace.I_p
        J_p = setup_workspace.J_p
        V_p = setup_workspace.V_p
        empty!(I_p); empty!(J_p); empty!(V_p)
        sizehint!(I_p, nnz_hint)
        sizehint!(J_p, nnz_hint)
        sizehint!(V_p, nnz_hint)
    else
        I_p = Ti[]
        J_p = Ti[]
        V_p = Tv[]
        sizehint!(I_p, nnz_hint)
        sizehint!(J_p, nnz_hint)
        sizehint!(V_p, nnz_hint)
    end
    S_p = do_rescale ? Tv[] : nothing  # per-entry scaling factors
    if do_rescale
        sizehint!(S_p, nnz_hint)
    end

    @inbounds for i in 1:n_fine
        if cf[i] == 1
            push!(I_p, Ti(i)); push!(J_p, Ti(coarse_map[i])); push!(V_p, one(Tv))
            if do_rescale; push!(S_p, one(Tv)); end
            continue
        end

        # ── Phase 1: Determine C-hat (extended coarse interpolation set) ──
        # C-hat = strong C neighbors of i ∪ strong C neighbors of strong F neighbors of i
        # Also mark strong F neighbors with strong_f_marker
        chat_indices = Int[]  # indices into P arrays for C-hat points

        for si in sn_offsets[i]:(sn_offsets[i + 1] - 1)
            j = sn_data[si]
            if cf[j] == 1
                # j is a strong C neighbor of i
                if P_marker[j] < 0
                    P_marker[j] = length(chat_indices)
                    push!(chat_indices, j)
                end
            elseif cf[j] == -1
                # j is a strong F neighbor of i — mark it and add its C neighbors
                P_marker[j] = strong_f_marker
                for sj in sn_offsets[j]:(sn_offsets[j + 1] - 1)
                    k = sn_data[sj]
                    if cf[k] == 1 && P_marker[k] < 0
                        P_marker[k] = length(chat_indices)
                        push!(chat_indices, k)
                    end
                end
            end
        end

        n_chat = length(chat_indices)
        if n_chat == 0
            # No C-hat: fallback to nearest coarse point
            best_j = _find_nearest_coarse(A, i, cf, coarse_map)
            push!(I_p, Ti(i)); push!(J_p, Ti(best_j)); push!(V_p, one(Tv))
            if do_rescale; push!(S_p, one(Tv)); end
            # Reset markers
            for si in sn_offsets[i]:(sn_offsets[i + 1] - 1)
                j = sn_data[si]
                P_marker[j] = -1
                if cf[j] == -1
                    for sj in sn_offsets[j]:(sn_offsets[j + 1] - 1)
                        P_marker[sn_data[sj]] = -1
                    end
                end
            end
            strong_f_marker -= 1
            continue
        end

        # ── Phase 2: Compute weights (matching hypre's ExtPI formula) ──
        # Initialize P_data for C-hat points to zero, and diagonal
        P_data = zeros(Tv, n_chat)
        diagonal = zero(Tv)

        for nz in nzrange(A, i)
            j = cv[nz]
            a_ij = nzv[nz]

            if j == i
                diagonal += a_ij
                continue
            end

            p_idx = P_marker[j]
            if p_idx >= 0
                # j is a C-point in C-hat: accumulate a_{i,j}
                P_data[p_idx + 1] += a_ij
            elseif p_idx == strong_f_marker
                # j is a strong F-neighbor: distribute through row of A[j,:]
                # Compute sum = Σ a_{j,m} for m in (C-hat ∪ {i}) with sgn*a_{j,m} < 0
                diag_j = zero(Tv)
                for nz3 in nzrange(A, j)
                    if cv[nz3] == j
                        diag_j = nzv[nz3]
                        break
                    end
                end
                sgn = real(diag_j) < 0 ? -1 : 1

                sum_val = zero(Tv)
                for nz2 in nzrange(A, j)
                    m = cv[nz2]
                    m == j && continue
                    a_jm = nzv[nz2]
                    if sgn * real(a_jm) < 0
                        if P_marker[m] >= 0 || m == i
                            sum_val += a_jm
                        end
                    end
                end

                if abs(sum_val) > eps(real(Tv))
                    distribute = a_ij / sum_val
                    for nz2 in nzrange(A, j)
                        m = cv[nz2]
                        m == j && continue
                        a_jm = nzv[nz2]
                        if sgn * real(a_jm) < 0
                            p_idx_m = P_marker[m]
                            if p_idx_m >= 0
                                P_data[p_idx_m + 1] += distribute * a_jm
                            elseif m == i
                                diagonal += distribute * a_jm
                            end
                        end
                    end
                else
                    # Can't distribute: lump into diagonal
                    diagonal += a_ij
                end
            else
                # Weak connection or non-C-hat C-point: lump into diagonal
                diagonal += a_ij
            end
        end

        # ── Phase 3: Finalize weights: P[j] = P_data[j] / (-diagonal) ──
        if abs(diagonal) > eps(real(Tv))
            for idx in 1:n_chat
                P_data[idx] /= -diagonal
            end
        end

        # ── Phase 4: Truncation (trunc_factor + max_elements limit) ──
        # First, apply trunc_factor to determine which entries survive
        keep = collect(1:n_chat)
        if trunc_factor > 0 && n_chat > 0
            max_w = zero(real(Tv))
            for idx in 1:n_chat
                max_w = max(max_w, abs(P_data[idx])^norm_p)
            end
            threshold = trunc_factor * max_w
            keep = [idx for idx in 1:n_chat if abs(P_data[idx])^norm_p >= threshold]
        end
        # Then, apply max_elements limit: keep only the strongest entries
        if max_elements > 0 && length(keep) > max_elements
            sort!(keep; by = idx -> abs(P_data[idx]), rev = true)
            resize!(keep, max_elements)
        end
        # Compute rescaling factor if enabled
        row_scale = one(Tv)
        if do_rescale && length(keep) < n_chat
            sum_removed = zero(Tv)
            kept_set = Set(keep)
            for idx in 1:n_chat
                if !(idx in kept_set)
                    sum_removed += P_data[idx]
                end
            end
            denom = one(Tv) - sum_removed
            if abs(denom) > Tv(1e-12)
                row_scale = one(Tv) / denom
            end
        end
        for idx in keep
            push!(I_p, Ti(i)); push!(J_p, Ti(coarse_map[chat_indices[idx]])); push!(V_p, P_data[idx] * row_scale)
            if do_rescale; push!(S_p, row_scale); end
        end

        # ── Reset markers ──
        for j in chat_indices
            P_marker[j] = -1
        end
        for si in sn_offsets[i]:(sn_offsets[i + 1] - 1)
            j = sn_data[si]
            P_marker[j] = -1
            if cf[j] == -1
                for sj in sn_offsets[j]:(sn_offsets[j + 1] - 1)
                    P_marker[sn_data[sj]] = -1
                end
            end
        end
        strong_f_marker -= 1
    end

    _sort_perm = setup_workspace !== nothing ? setup_workspace.sort_perm : nothing
    old_P_reuse = setup_workspace !== nothing ? setup_workspace.old_P : nothing
    if do_rescale
        P = _coo_to_prolongation(I_p, J_p, V_p, n_fine, n_coarse;
            old_P=old_P_reuse, S_p=S_p, sort_perm=_sort_perm)
    else
        P = _coo_to_prolongation(I_p, J_p, V_p, n_fine, n_coarse;
            old_P=old_P_reuse, sort_perm=_sort_perm)
    end
    if setup_workspace !== nothing
        setup_workspace.old_P = nothing
    end
    
    # Build update map with full graph structure for Extended+i interpolation
    P_update_map = nothing
    if build_update_map
        # We need to store the complete graph structure for Extended+i
        # The sn_offsets and sn_data already contain strong neighbor info
        # Build diagonal indices
        diag_nz_idx = Vector{Ti}(undef, n_fine)
        strong_nbrs_nz_list = Ti[]
        @inbounds for i in 1:n_fine
            diag_nz_idx[i] = Ti(0)
            for nz in nzrange(A, i)
                if cv[nz] == i
                    diag_nz_idx[i] = Ti(nz)
                    break
                end
            end
        end
        
        # Build strong neighbor nz indices
        @inbounds for i in 1:n_fine
            for nz in nzrange(A, i)
                j = cv[nz]
                if j != i && is_strong[nz]
                    push!(strong_nbrs_nz_list, Ti(nz))
                end
            end
        end
        
        # For Extended+i, all entries need full recomputation (entry_type=3)
        nnz_P = length(P.nzval)
        entry_type = fill(Ti(3), nnz_P)
        numer_idx = zeros(Ti, nnz_P)
        denom_offsets = Vector{Ti}(undef, nnz_P + 1)
        denom_entries = Ti[]
        
        # Mark coarse points as type 0
        for i in 1:n_fine
            if cf[i] == 1
                for p_nz in P.rowptr[i]:(P.rowptr[i+1]-1)
                    entry_type[p_nz] = Ti(0)
                end
            end
        end
        
        # Simple denom_offsets (not used for type 3, but needs valid structure)
        for k in 1:(nnz_P + 1)
            denom_offsets[k] = Ti(1)
        end
        
        strong_nbrs_cols = Vector{Ti}(sn_data)
        strong_nbrs_nz = Vector{Ti}(strong_nbrs_nz_list)
        strong_nbrs_offsets = Vector{Ti}(sn_offsets)
        
        # Allocate workspace for Extended+i interpolation update
        # Estimate max P entries per row (including distance-2 coarse)
        max_chat = 0
        for i in 1:n_fine
            if cf[i] == -1
                # Count distance-1 coarse + distance-2 through fine
                count = 0
                for si in sn_offsets[i]:(sn_offsets[i+1]-1)
                    j = sn_data[si]
                    if cf[j] == 1
                        count += 1
                    elseif cf[j] == -1
                        for sj in sn_offsets[j]:(sn_offsets[j+1]-1)
                            if cf[sn_data[sj]] == 1
                                count += 1
                            end
                        end
                    end
                end
                max_chat = max(max_chat, count)
            end
        end
        max_chat = max(max_chat, 1)  # at least 1
        
        P_update_map = ProlongationUpdateMap{Ti, Tv}(
            3,  # interp_type = Extended+i
            copy(is_strong),
            copy(cf),
            copy(coarse_map),
            diag_nz_idx,
            entry_type,
            numer_idx,
            denom_offsets,
            denom_entries,
            strong_nbrs_offsets,
            strong_nbrs_cols,
            strong_nbrs_nz,
            fill(-1, n_fine),              # P_marker
            Vector{Int}(undef, max_chat),  # chat_indices buffer
            Vector{Tv}(undef, max_chat)    # P_data buffer
        )
    end
    
    return P, P_update_map
end

# ── Helpers ──────────────────────────────────────────────────────────────────

"""Find nearest coarse point for fallback interpolation."""
function _find_nearest_coarse(A::CSRMatrix{Tv, Ti}, i::Int,
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

"""Scatter COO entries into CSR arrays using counting sort positions."""
@inline function _scatter_coo_to_csr!(colval, nzval, I_p, J_p, V_p, pos, nnz_p::Int)
    @inbounds for k in 1:nnz_p
        row = I_p[k]
        p = Int(pos[row])
        colval[p] = J_p[k]
        nzval[p] = V_p[k]
        pos[row] += one(eltype(pos))
    end
end

"""Scatter COO entries + trunc_scaling into CSR arrays using counting sort positions."""
@inline function _scatter_coo_to_csr!(colval, nzval, trunc, I_p, J_p, V_p, S_p, pos, nnz_p::Int)
    @inbounds for k in 1:nnz_p
        row = I_p[k]
        p = Int(pos[row])
        colval[p] = J_p[k]
        nzval[p] = V_p[k]
        trunc[p] = S_p[k]
        pos[row] += one(eltype(pos))
    end
end

"""In-place insertion sort of CSR row entries by column index."""
function _sort_csr_row!(colval::AbstractVector, nzval::AbstractVector,
                         rs::Int, re::Int)
    @inbounds for j in (rs+1):re
        key_c = colval[j]
        key_v = nzval[j]
        k = j - 1
        while k >= rs && colval[k] > key_c
            colval[k+1] = colval[k]
            nzval[k+1] = nzval[k]
            k -= 1
        end
        colval[k+1] = key_c
        nzval[k+1] = key_v
    end
end

"""In-place insertion sort of CSR row entries by column index, also permuting trunc_scaling."""
function _sort_csr_row!(colval::AbstractVector, nzval::AbstractVector,
                         trunc::AbstractVector, rs::Int, re::Int)
    @inbounds for j in (rs+1):re
        key_c = colval[j]
        key_v = nzval[j]
        key_t = trunc[j]
        k = j - 1
        while k >= rs && colval[k] > key_c
            colval[k+1] = colval[k]
            nzval[k+1] = nzval[k]
            trunc[k+1] = trunc[k]
            k -= 1
        end
        colval[k+1] = key_c
        nzval[k+1] = key_v
        trunc[k+1] = key_t
    end
end

"""Convert COO format to ProlongationOp (CSR) using counting sort by row
followed by per-row insertion sort by column. Much faster than global
sortperm for typical prolongation operators with few entries per row.

When `old_P` is provided, its arrays are resized and reused instead of
allocating new ones. When `S_p` is provided, it is reordered into CSR
order and stored as `trunc_scaling`. `sort_perm` is used as a temporary
position buffer during the counting sort."""
function _coo_to_prolongation(I_p::Vector{Ti}, J_p::Vector{Ti}, V_p::Vector{Tv},
                              n_fine::Int, n_coarse::Int;
                              old_P::Union{Nothing, ProlongationOp}=nothing,
                              S_p::Union{Nothing, Vector}=nothing,
                              sort_perm::Union{Nothing,Vector{Int}}=nothing) where {Ti, Tv}
    nnz_p = length(I_p)

    # Get or create output arrays, reusing old_P when available
    if old_P !== nothing && old_P.colval isa Vector
        rp = resize!(old_P.rowptr, n_fine + 1)
        colval = resize!(old_P.colval, nnz_p)
        nzval = resize!(old_P.nzval, nnz_p)
    else
        rp = Vector{Ti}(undef, n_fine + 1)
        colval = Vector{Ti}(undef, nnz_p)
        nzval = Vector{Tv}(undef, nnz_p)
    end

    # Handle trunc_scaling: resize from old_P or allocate
    if S_p !== nothing
        if old_P !== nothing && old_P.trunc_scaling isa Vector
            trunc = resize!(old_P.trunc_scaling, nnz_p)
        else
            trunc = Vector{Tv}(undef, nnz_p)
        end
    else
        trunc = nothing
    end

    # 1. Count entries per row
    fill!(rp, Ti(0))
    @inbounds for k in 1:nnz_p
        rp[I_p[k]] += Ti(1)
    end

    # 2. Build rowptr (cumulative sum)
    cumsum_val = Ti(1)
    @inbounds for i in 1:n_fine
        count = rp[i]
        rp[i] = cumsum_val
        cumsum_val += count
    end
    rp[n_fine + 1] = cumsum_val

    # 3. Counting sort: distribute COO entries into CSR positions
    if sort_perm !== nothing
        _ws_resize!(sort_perm, n_fine)
        pos = sort_perm
    else
        pos = Vector{Int}(undef, n_fine)
    end
    copyto!(pos, 1, rp, 1, n_fine)

    if trunc !== nothing
        _scatter_coo_to_csr!(colval, nzval, trunc, I_p, J_p, V_p, S_p, pos, nnz_p)
    else
        _scatter_coo_to_csr!(colval, nzval, I_p, J_p, V_p, pos, nnz_p)
    end

    # 4. Sort each row by column (insertion sort — rows are typically small)
    if trunc !== nothing
        @inbounds for i in 1:n_fine
            _sort_csr_row!(colval, nzval, trunc, Int(rp[i]), Int(rp[i+1]) - 1)
        end
    else
        @inbounds for i in 1:n_fine
            _sort_csr_row!(colval, nzval, Int(rp[i]), Int(rp[i+1]) - 1)
        end
    end

    # Return ProlongationOp (mutate old_P or create new)
    if old_P !== nothing && old_P.colval isa Vector
        old_P.nrow = n_fine
        old_P.ncol = n_coarse
        old_P.trunc_scaling = trunc
        return old_P
    elseif trunc !== nothing
        return ProlongationOp{Ti, Tv, Vector{Ti}, Vector{Tv}}(rp, colval, nzval, n_fine, n_coarse, trunc)
    else
        return ProlongationOp{Ti, Tv}(rp, colval, nzval, n_fine, n_coarse)
    end
end

"""
    prolongate!(x_fine, P, x_coarse)

Apply prolongation: x_fine += P * x_coarse.
Uses KernelAbstractions for parallel execution over fine rows.
"""
function prolongate!(x_fine::AbstractVector, P::ProlongationOp, x_coarse::AbstractVector;
                     backend=DEFAULT_BACKEND, block_size::Int=64)
    kernel! = prolongate_kernel!(backend, block_size)
    kernel!(x_fine, P.rowptr, P.colval, P.nzval, x_coarse; ndrange=P.nrow)
    _synchronize(backend)
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
    build_transpose_map(P) -> TransposeMap

Build a transpose structure for prolongation operator P, mapping each coarse
column J to its contributing fine rows. Enables atomic-free restriction.
"""
function build_transpose_map(P::ProlongationOp{Ti, Tv}) where {Ti, Tv}
    n_fine = P.nrow
    n_coarse = P.ncol
    # Count entries per coarse column
    col_counts = zeros(Int, n_coarse)
    @inbounds for i in 1:n_fine
        for nz in P.rowptr[i]:(P.rowptr[i+1]-1)
            col_counts[P.colval[nz]] += 1
        end
    end
    # Build offsets
    offsets = Vector{Ti}(undef, n_coarse + 1)
    offsets[1] = Ti(1)
    for j in 1:n_coarse
        offsets[j+1] = offsets[j] + Ti(col_counts[j])
    end
    total = offsets[n_coarse + 1] - Ti(1)
    fine_rows = Vector{Ti}(undef, total)
    p_nz_idx = Vector{Ti}(undef, total)
    # Fill entries
    pos = copy(offsets[1:n_coarse])
    @inbounds for i in 1:n_fine
        for nz in P.rowptr[i]:(P.rowptr[i+1]-1)
            J = P.colval[nz]
            fine_rows[pos[J]] = Ti(i)
            p_nz_idx[pos[J]] = Ti(nz)
            pos[J] += Ti(1)
        end
    end
    return TransposeMap(offsets, fine_rows, p_nz_idx)
end

"""
    restrict!(b_coarse, Pt_map, P, r_fine)

Apply restriction (P^T): b_coarse = P^T * r_fine.
Uses the pre-computed TransposeMap to parallelize over coarse rows without atomics.
"""
function restrict!(b_coarse::AbstractVector, Pt_map::TransposeMap,
                   P::ProlongationOp, r_fine::AbstractVector;
                   backend=DEFAULT_BACKEND, block_size::Int=64)
    n_coarse = P.ncol
    kernel! = restrict_kernel!(backend, block_size)
    kernel!(b_coarse, Pt_map.offsets, Pt_map.fine_rows,
            Pt_map.p_nz_idx, P.nzval, r_fine; ndrange=n_coarse)
    _synchronize(backend)
    return b_coarse
end

@kernel function restrict_kernel!(b_coarse, @Const(offsets), @Const(fine_rows),
                                  @Const(p_nz_idx), @Const(P_nzval), @Const(r_fine))
    J = @index(Global)
    @inbounds begin
        acc = zero(eltype(b_coarse))
        for k in offsets[J]:(offsets[J+1]-1)
            i = fine_rows[k]
            acc += P_nzval[p_nz_idx[k]] * r_fine[i]
        end
        b_coarse[J] = acc
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# In-place prolongation value update for resetup
# ══════════════════════════════════════════════════════════════════════════════

"""
    _update_prolongation_values!(level, A; backend, block_size)

Update the prolongation operator values in-place based on new matrix A.
The sparsity pattern of P is preserved (same rowptr, colval), but nzval
is recomputed using the stored graph structure in level.P_update_map.

This uses GPU-compatible KernelAbstractions kernels. The precomputed map
stores all classification decisions (strength graph, CF-split) from setup,
so P value updates use the same graph structure with new A values.

Dispatch based on interpolation type:
- Direct (type 1): Simple formula P[k] = -A[numer_idx[k]] / d_i
- Standard (type 2): Full row-by-row recomputation with indirect contributions
- Extended+i (type 3): Full row-by-row recomputation with extended stencil
"""
function _update_prolongation_values!(level::AMGLevel{Tv, Ti}, A::CSRMatrix{Tv, Ti};
                                      backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    P_update_map = level.P_update_map
    P_update_map === nothing && return level.P
    
    interp_type = P_update_map.interp_type
    if interp_type == 1
        # Direct interpolation: use simple kernel
        _update_P_direct_kernel!(level.P, A, P_update_map; backend=backend, block_size=block_size)
    elseif interp_type == 2
        # Standard interpolation: full row recomputation
        _update_P_standard!(level.P, A, P_update_map)
    elseif interp_type == 3
        # Extended+i interpolation: full row recomputation
        _update_P_extendedi!(level.P, A, P_update_map)
    end
    
    return level.P
end

"""
    _update_P_direct_kernel!(P, A, P_update_map; backend, block_size)

GPU-compatible kernel for Direct interpolation P value update.

For each P entry k:
- If entry_type[k] == 0: P[k] = 1 (coarse point or fallback)
- If entry_type[k] == 1: P[k] = -A[numer_idx[k]] / d_i where d_i = Σ A[denom_entries[j]]
"""
function _update_P_direct_kernel!(P::ProlongationOp, A::CSRMatrix{Tv, Ti},
                                  P_update_map::ProlongationUpdateMap;
                                  backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    be = _get_backend(P.nzval)
    A_nzval = nonzeros(A)
    P_nzval = P.nzval
    entry_type = P_update_map.entry_type
    numer_idx = P_update_map.numer_idx
    denom_offsets = P_update_map.denom_offsets
    denom_entries = P_update_map.denom_entries
    nnz_P = length(P_nzval)
    
    kernel! = _p_direct_update_kernel!(be, block_size)
    kernel!(P_nzval, A_nzval, entry_type, numer_idx, denom_offsets, denom_entries; ndrange=nnz_P)
    _synchronize(be)
    
    return P
end

@kernel function _p_direct_update_kernel!(P_nzval, @Const(A_nzval), @Const(entry_type),
                                          @Const(numer_idx), @Const(denom_offsets), @Const(denom_entries))
    k = @index(Global)
    @inbounds begin
        if entry_type[k] == 0
            # Coarse point or fallback: P value = 1
            P_nzval[k] = one(eltype(P_nzval))
        else
            # Direct formula: P[k] = -A[numer] / d_i
            numer = numer_idx[k]
            d_i = zero(eltype(A_nzval))
            for j in denom_offsets[k]:(denom_offsets[k+1]-1)
                d_i += A_nzval[denom_entries[j]]
            end
            abs_d_i = abs(d_i)
            threshold = eps(real(eltype(A_nzval))) * max(one(real(eltype(A_nzval))), abs_d_i)
            if abs_d_i > threshold
                P_nzval[k] = -A_nzval[numer] / d_i
            else
                P_nzval[k] = zero(eltype(P_nzval))
            end
        end
    end
end

"""
    _update_P_standard!(P, A, P_update_map)

CPU-based Standard interpolation P value update.
Uses stored graph structure to recompute interpolation weights.
"""
function _update_P_standard!(P::ProlongationOp{Ti, Tv}, A::CSRMatrix{Tv, Ti},
                             P_update_map::ProlongationUpdateMap{Ti, Tv2}) where {Tv, Ti, Tv2}
    A_cpu = csr_to_cpu(A)
    P_nzval = P.nzval isa Array ? P.nzval : Array(P.nzval)
    P_rowptr = P.rowptr isa Array ? P.rowptr : Array(P.rowptr)
    P_colval = P.colval isa Array ? P.colval : Array(P.colval)
    
    is_strong = P_update_map.is_strong
    cf = P_update_map.cf
    coarse_map = P_update_map.coarse_map
    strong_nbrs_offsets = P_update_map.strong_nbrs_offsets
    strong_nbrs_cols = P_update_map.strong_nbrs_cols
    strong_nbrs_nz = P_update_map.strong_nbrs_nz
    
    n_fine = length(cf)
    cv = colvals(A_cpu)
    nzv = nonzeros(A_cpu)
    
    @inbounds for i in 1:n_fine
        p_start = Int(P_rowptr[i])
        p_end = Int(P_rowptr[i+1]) - 1
        
        if cf[i] == 1
            # Coarse point: P = 1
            if p_start <= p_end
                P_nzval[p_start] = one(Tv)
            end
            continue
        end
        
        # Classify connections using stored strength graph
        a_ii = zero(Tv)
        sum_weak = zero(Tv)
        strong_coarse = Dict{Int, Tv}()
        strong_fine = Tuple{Int, Tv}[]
        
        for nz in nzrange(A_cpu, i)
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
        
        # Indirect contributions from strong fine neighbors
        contributions = Dict{Int, Tv}()
        for (cm, a_ij) in strong_coarse
            contributions[cm] = a_ij
        end
        
        for (k, a_ik) in strong_fine
            diag_k = zero(Tv)
            for nz2 in nzrange(A_cpu, k)
                if cv[nz2] == k
                    diag_k = nzv[nz2]
                    break
                end
            end
            sgn = real(diag_k) < 0 ? -1 : 1
            sum_C_k = zero(Tv)
            coarse_vals_k = Dict{Int, Tv}()
            for nz2 in nzrange(A_cpu, k)
                j2 = cv[nz2]
                j2 == k && continue
                a_kj = nzv[nz2]
                if cf[j2] == 1
                    cm2 = coarse_map[j2]
                    if haskey(strong_coarse, cm2) && sgn * real(a_kj) < 0
                        coarse_vals_k[cm2] = get(coarse_vals_k, cm2, zero(Tv)) + a_kj
                        sum_C_k += a_kj
                    end
                end
                if j2 == i && sgn * real(a_kj) < 0
                    sum_C_k += a_kj
                end
            end
            if abs(sum_C_k) > eps(real(Tv))
                distribute = a_ik / sum_C_k
                for (cm2, a_kj) in coarse_vals_k
                    contributions[cm2] = get(contributions, cm2, zero(Tv)) + distribute * a_kj
                end
                for nz2 in nzrange(A_cpu, k)
                    if cv[nz2] == i && sgn * real(nzv[nz2]) < 0
                        d_i += distribute * nzv[nz2]
                        break
                    end
                end
            else
                d_i += a_ik
            end
        end
        
        # Update P values for this row
        if isempty(contributions)
            for p_nz in p_start:p_end
                P_nzval[p_nz] = one(Tv)
            end
        else
            for p_nz in p_start:p_end
                coarse_col = Int(P_colval[p_nz])
                if haskey(contributions, coarse_col)
                    val = contributions[coarse_col]
                    P_nzval[p_nz] = abs(d_i) > eps(real(Tv)) ? -val / d_i : zero(Tv)
                else
                    P_nzval[p_nz] = one(Tv)  # fallback
                end
            end
        end
    end
    
    # Copy back to device if needed
    if !(P.nzval isa Array)
        copyto!(P.nzval, P_nzval)
    end
    
    return P
end

"""
    _update_P_extendedi!(P, A, P_update_map)

CPU-based Extended+i interpolation P value update.
Uses stored graph structure to recompute interpolation weights.
Uses workspace buffers from P_update_map to avoid per-row allocations.
"""
function _update_P_extendedi!(P::ProlongationOp{Ti, Tv}, A::CSRMatrix{Tv, Ti},
                              P_update_map::ProlongationUpdateMap{Ti, Tv2}) where {Tv, Ti, Tv2}
    A_cpu = csr_to_cpu(A)
    P_nzval = P.nzval isa Array ? P.nzval : Array(P.nzval)
    P_rowptr = P.rowptr isa Array ? P.rowptr : Array(P.rowptr)
    P_colval = P.colval isa Array ? P.colval : Array(P.colval)
    
    is_strong = P_update_map.is_strong
    cf = P_update_map.cf
    coarse_map = P_update_map.coarse_map
    sn_offsets = P_update_map.strong_nbrs_offsets
    sn_data = P_update_map.strong_nbrs_cols
    
    n_fine = length(cf)
    cv = colvals(A_cpu)
    nzv = nonzeros(A_cpu)
    
    # Use workspace from P_update_map to avoid allocations
    P_marker = P_update_map.P_marker
    chat_indices_buf = P_update_map.chat_indices
    P_data_buf = P_update_map.P_data
    
    # Reset P_marker to -1 at start
    fill!(P_marker, -1)
    strong_f_marker = -2
    
    @inbounds for i in 1:n_fine
        p_start = Int(P_rowptr[i])
        p_end = Int(P_rowptr[i+1]) - 1
        
        if cf[i] == 1
            if p_start <= p_end
                P_nzval[p_start] = one(Tv)
            end
            continue
        end
        
        # Phase 1: Determine C-hat using workspace buffer
        n_chat = 0
        for si in sn_offsets[i]:(sn_offsets[i+1]-1)
            j = sn_data[si]
            if cf[j] == 1
                if P_marker[j] < 0
                    n_chat += 1
                    # Resize buffer if needed
                    if n_chat > length(chat_indices_buf)
                        resize!(chat_indices_buf, max(n_chat, 2 * length(chat_indices_buf)))
                        resize!(P_data_buf, length(chat_indices_buf))
                    end
                    chat_indices_buf[n_chat] = j
                    P_marker[j] = n_chat - 1  # 0-based index into P_data
                end
            elseif cf[j] == -1
                P_marker[j] = strong_f_marker
                for sj in sn_offsets[j]:(sn_offsets[j+1]-1)
                    k = sn_data[sj]
                    if cf[k] == 1 && P_marker[k] < 0
                        n_chat += 1
                        # Resize buffer if needed
                        if n_chat > length(chat_indices_buf)
                            resize!(chat_indices_buf, max(n_chat, 2 * length(chat_indices_buf)))
                            resize!(P_data_buf, length(chat_indices_buf))
                        end
                        chat_indices_buf[n_chat] = k
                        P_marker[k] = n_chat - 1
                    end
                end
            end
        end
        
        if n_chat == 0
            for p_nz in p_start:p_end
                P_nzval[p_nz] = one(Tv)
            end
            # Reset markers
            for si in sn_offsets[i]:(sn_offsets[i+1]-1)
                j = sn_data[si]
                P_marker[j] = -1
                if cf[j] == -1
                    for sj in sn_offsets[j]:(sn_offsets[j+1]-1)
                        P_marker[sn_data[sj]] = -1
                    end
                end
            end
            strong_f_marker -= 1
            continue
        end
        
        # Phase 2: Compute weights - zero out P_data workspace
        for idx in 1:n_chat
            P_data_buf[idx] = zero(Tv)
        end
        diagonal = zero(Tv)
        
        for nz in nzrange(A_cpu, i)
            j = cv[nz]
            a_ij = nzv[nz]
            
            if j == i
                diagonal += a_ij
                continue
            end
            
            p_idx = P_marker[j]
            if p_idx >= 0
                P_data_buf[p_idx + 1] += a_ij
            elseif p_idx == strong_f_marker
                diag_j = zero(Tv)
                for nz3 in nzrange(A_cpu, j)
                    if cv[nz3] == j
                        diag_j = nzv[nz3]
                        break
                    end
                end
                sgn = real(diag_j) < 0 ? -1 : 1
                
                sum_val = zero(Tv)
                for nz2 in nzrange(A_cpu, j)
                    m = cv[nz2]
                    m == j && continue
                    a_jm = nzv[nz2]
                    if sgn * real(a_jm) < 0
                        if P_marker[m] >= 0 || m == i
                            sum_val += a_jm
                        end
                    end
                end
                
                if abs(sum_val) > eps(real(Tv))
                    distribute = a_ij / sum_val
                    for nz2 in nzrange(A_cpu, j)
                        m = cv[nz2]
                        m == j && continue
                        a_jm = nzv[nz2]
                        if sgn * real(a_jm) < 0
                            p_idx_m = P_marker[m]
                            if p_idx_m >= 0
                                P_data_buf[p_idx_m + 1] += distribute * a_jm
                            elseif m == i
                                diagonal += distribute * a_jm
                            end
                        end
                    end
                else
                    diagonal += a_ij
                end
            else
                diagonal += a_ij
            end
        end
        
        # Phase 3: Finalize weights
        if abs(diagonal) > eps(real(Tv))
            for idx in 1:n_chat
                P_data_buf[idx] /= -diagonal
            end
        end
        
        # Update P values for this row
        for p_nz in p_start:p_end
            coarse_col = Int(P_colval[p_nz])
            found = false
            for idx in 1:n_chat
                if coarse_map[chat_indices_buf[idx]] == coarse_col
                    P_nzval[p_nz] = P_data_buf[idx]
                    found = true
                    break
                end
            end
            if !found
                P_nzval[p_nz] = one(Tv)
            end
        end
        
        # Reset markers for this row's C-hat
        for idx in 1:n_chat
            P_marker[chat_indices_buf[idx]] = -1
        end
        for si in sn_offsets[i]:(sn_offsets[i+1]-1)
            j = sn_data[si]
            P_marker[j] = -1
            if cf[j] == -1
                for sj in sn_offsets[j]:(sn_offsets[j+1]-1)
                    P_marker[sn_data[sj]] = -1
                end
            end
        end
        strong_f_marker -= 1
    end
    
    # Copy back to device if needed
    if !(P.nzval isa Array)
        copyto!(P.nzval, P_nzval)
    end
    
    return P
end
