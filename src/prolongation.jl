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
    build_cf_prolongation(A, cf, coarse_map, n_coarse, interp, θ)

Build a prolongation operator from a CF-splitting using the specified interpolation method.
- `cf[i] == 1` → coarse point, `cf[i] == -1` → fine point
- `coarse_map[i]` → coarse-grid index for coarse points
- `θ` → strength threshold used consistently for interpolation stencil selection
"""
function build_cf_prolongation(A::CSRMatrix{Tv, Ti}, cf::Vector{Int},
                               coarse_map::Vector{Int}, n_coarse::Int,
                               interp::InterpolationType, θ::Real=0.25;
                               backend=DEFAULT_BACKEND, block_size::Int=64,
                               setup_workspace=nothing) where {Tv, Ti}
    return _build_interpolation(A, cf, coarse_map, n_coarse, interp, θ; backend=backend, block_size=block_size, setup_workspace=setup_workspace)
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
"""
function _build_interpolation(A_in::CSRMatrix{Tv, Ti}, cf::Vector{Int},
                              coarse_map::Vector{Int}, n_coarse::Int,
                              ::DirectInterpolation, θ::Real=0.25;
                              backend=DEFAULT_BACKEND, block_size::Int=64,
                              setup_workspace=nothing) where {Tv, Ti}
    # Compute strength on GPU if available, then convert to CPU for graph algorithms
    is_strong_raw = strength_graph(A_in, θ; backend=backend, block_size=block_size,
        is_strong=setup_workspace !== nothing ? setup_workspace.is_strong : nothing)
    is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    A = csr_to_cpu(A_in)
    n_fine = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)

    # First pass: count entries per row
    row_counts = zeros(Int, n_fine)
    @inbounds for i in 1:n_fine
        if cf[i] == 1
            row_counts[i] = 1  # coarse point: identity mapping
        else
            # Find diagonal sign
            a_ii = zero(Tv)
            for nz in nzrange(A, i)
                if cv[nz] == i
                    a_ii = nzv[nz]
                    break
                end
            end
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
            # Compute diagonal correction: d_i = a_{i,i} + Σ weak/fine/same-sign connections
            a_ii = zero(Tv)
            sum_nonC = zero(Tv)
            strong_coarse_cols = Vector{Ti}()
            strong_coarse_vals = Vector{Tv}()
            diag_sign = zero(real(Tv))
            for nz in nzrange(A, i)
                j = cv[nz]
                if j == i
                    a_ii = nzv[nz]
                    diag_sign = sign(real(a_ii))
                end
            end
            for nz in nzrange(A, i)
                j = cv[nz]
                j == i && continue
                is_interp_coarse = is_strong[nz] && cf[j] == 1 &&
                    (diag_sign == 0 || sign(real(nzv[nz])) != diag_sign)
                if is_interp_coarse
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
                    nzv_p[pos] = abs(d_i) > _safe_threshold(Tv, abs(d_i)) ? -strong_coarse_vals[k] / d_i : zero(Tv)
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
function _build_interpolation(A_in::CSRMatrix{Tv, Ti}, cf::Vector{Int},
                              coarse_map::Vector{Int}, n_coarse::Int,
                              ::StandardInterpolation, θ::Real=0.25;
                              backend=DEFAULT_BACKEND, block_size::Int=64,
                              setup_workspace=nothing) where {Tv, Ti}
    # Compute strength on GPU if available, then convert to CPU for graph algorithms
    is_strong_raw = strength_graph(A_in, θ; backend=backend, block_size=block_size,
        is_strong=setup_workspace !== nothing ? setup_workspace.is_strong : nothing)
    is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    A = csr_to_cpu(A_in)
    n_fine = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)

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
            # hypre formula: distribute = a_{i,k} / sum where
            # sum = Σ a_{k,m} for m in (C_i ∪ {i}) with opposite sign to a_{k,k}
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
                    # Only distribute to coarse points in C_i
                    if haskey(strong_coarse, cm2) && sgn * real(a_kj) < 0
                        coarse_vals_k[cm2] = get(coarse_vals_k, cm2, zero(Tv)) + a_kj
                        sum_C_k += a_kj
                    end
                end
                # Also include connection back to i in the sum
                if j2 == i && sgn * real(a_kj) < 0
                    sum_C_k += a_kj
                end
            end
            if abs(sum_C_k) > eps(real(Tv))
                distribute = a_ik / sum_C_k
                for (cm2, a_kj) in coarse_vals_k
                    contributions[cm2] = get(contributions, cm2, zero(Tv)) + distribute * a_kj
                end
                # Diagonal contribution: a_{k,i} * distribute
                for nz2 in nzrange(A, k)
                    if cv[nz2] == i && sgn * real(nzv[nz2]) < 0
                        d_i += distribute * nzv[nz2]
                        break
                    end
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
                w = abs(d_i) > eps(real(Tv)) ? -val / d_i : zero(Tv)
                push!(I_p, Ti(i)); push!(J_p, Ti(cm)); push!(V_p, w)
            end
        end
    end

    return _coo_to_prolongation(I_p, J_p, V_p, n_fine, n_coarse;
        sort_perm=setup_workspace !== nothing ? setup_workspace.sort_perm : nothing)
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
                              setup_workspace=nothing) where {Tv, Ti}
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
    resize!(sn_offsets, n_fine + 1)
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
    resize!(sn_data, total_sn)
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
        P_marker = resize!(setup_workspace.P_marker, n_fine)
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
    if do_rescale
        # Compute sort permutation before _coo_to_prolongation mutates I_p/J_p
        nnz_p = length(I_p)
        if _sort_perm !== nothing
            resize!(_sort_perm, nnz_p)
            perm = _sort_perm
        else
            perm = Vector{Int}(undef, nnz_p)
        end
        sortperm!(perm, 1:nnz_p; by=k -> (I_p[k], J_p[k]))
        permute!(S_p, perm)
        P = _coo_to_prolongation(I_p, J_p, V_p, n_fine, n_coarse; sort_perm=_sort_perm)
        P.trunc_scaling = copy(S_p)
    else
        P = _coo_to_prolongation(I_p, J_p, V_p, n_fine, n_coarse; sort_perm=_sort_perm)
    end
    return P
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

"""Convert COO format to ProlongationOp (CSR). The input arrays are sorted
in-place as workspace; the returned ProlongationOp owns independent copies.
`sort_perm` is an optional pre-allocated buffer for sortperm!."""
function _coo_to_prolongation(I_p::Vector{Ti}, J_p::Vector{Ti}, V_p::Vector{Tv},
                              n_fine::Int, n_coarse::Int;
                              sort_perm::Union{Nothing,Vector{Int}}=nothing) where {Ti, Tv}
    # Sort by (row, col) using buffer to avoid allocating tuples
    nnz_p = length(I_p)
    if sort_perm !== nothing
        resize!(sort_perm, nnz_p)
        perm = sort_perm
    else
        perm = Vector{Int}(undef, nnz_p)
    end
    sortperm!(perm, 1:nnz_p; by=k -> (I_p[k], J_p[k]))
    permute!(I_p, perm)
    permute!(J_p, perm)
    permute!(V_p, perm)
    rp = Vector{Ti}(undef, n_fine + 1)
    fill!(rp, Ti(0))
    for k in 1:nnz_p
        rp[I_p[k]] += Ti(1)
    end
    cumsum = Ti(1)
    for i in 1:n_fine
        count = rp[i]
        rp[i] = cumsum
        cumsum += count
    end
    rp[n_fine + 1] = cumsum
    # Copy colval/nzval so the ProlongationOp doesn't alias workspace arrays
    return ProlongationOp{Ti, Tv}(rp, copy(J_p), copy(V_p), n_fine, n_coarse)
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
