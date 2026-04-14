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
                               build_update_map::Bool=false,
                               coarsening_is_strong::Union{Nothing, AbstractVector{Bool}}=nothing) where {Tv, Ti}
    return _build_interpolation(A, cf, coarse_map, n_coarse, interp, θ; backend=backend, block_size=block_size, setup_workspace=setup_workspace, build_update_map=build_update_map, coarsening_is_strong=coarsening_is_strong)
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

Uses a **two-phase approach** when `build_update_map=true`:
1. **Phase 1**: Build sparsity pattern (rowptr, colval) and update map (index mappings)
2. **Phase 2**: Call `_update_P_direct_kernel!` to compute nzval using the same code path as resetup

This ensures that initial setup and `update_P=true` resetup produce **identical** results.
"""
function _build_interpolation(A_in::CSRMatrix{Tv, Ti}, cf::Vector{Int},
                              coarse_map::Vector{Int}, n_coarse::Int,
                              ::DirectInterpolation, θ::Real=0.25;
                              backend=DEFAULT_BACKEND, block_size::Int=64,
                              setup_workspace=nothing,
                              build_update_map::Bool=false,
                              coarsening_is_strong::Union{Nothing, AbstractVector{Bool}}=nothing) where {Tv, Ti}
    # Use the coarsening strength graph if provided (ensures consistency when max_row_sum < 1.0),
    # otherwise recompute from A_in.
    if coarsening_is_strong !== nothing
        is_strong = coarsening_is_strong isa Array ? coarsening_is_strong : Array(coarsening_is_strong)
    else
        is_strong_raw = strength_graph(A_in, θ; backend=backend, block_size=block_size,
            is_strong=setup_workspace !== nothing ? setup_workspace.is_strong : nothing)
        is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    end
    A = csr_to_cpu(A_in)
    n_fine = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)
    
    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Build sparsity pattern and update map (graph structure only)
    # ══════════════════════════════════════════════════════════════════════════
    
    # First pass: count entries per row
    row_counts = zeros(Int, n_fine)
    diag_nz_idx = Vector{Ti}(undef, n_fine)
    
    @inbounds for i in 1:n_fine
        diag_nz_idx[i] = Ti(0)
        if cf[i] == 1
            row_counts[i] = 1  # coarse point: identity mapping
            for nz in nzrange(A, i)
                if cv[nz] == i
                    diag_nz_idx[i] = Ti(nz)
                    break
                end
            end
        else
            # Find diagonal index
            for nz in nzrange(A, i)
                if cv[nz] == i
                    diag_nz_idx[i] = Ti(nz)
                    break
                end
            end
            # Fine point: count strong coarse neighbors (no sign filtering —
            # matching hypre's Direct interpolation where all strong C-connections
            # are used for interpolation regardless of sign)
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

    # Prepare update map arrays
    # Always build update map structure (needed for two-phase approach)
    entry_type = Vector{Ti}(undef, total_nnz)
    numer_idx = Vector{Ti}(undef, total_nnz)
    denom_offsets = Vector{Ti}(undef, total_nnz + 1)
    denom_entries_list = Ti[]
    sizehint!(denom_entries_list, total_nnz * 4; shrink=false)  # estimate 4 denom entries per P entry
    
    # For Standard/Extended+i: strong neighbor structure (only if storing update map)
    if build_update_map
        strong_nbrs_offsets = Vector{Ti}(undef, n_fine + 1)
        strong_nbrs_cols_list = Ti[]
        strong_nbrs_nz_list = Ti[]
        sizehint!(strong_nbrs_cols_list, nnz(A); shrink=false)
        sizehint!(strong_nbrs_nz_list, nnz(A); shrink=false)
    end

    # Pre-allocate workspace buffers for inner loop (avoid per-row allocations)
    # Estimate max row size from A's max row length
    max_row_nnz = 0
    for i in 1:n_fine
        row_nnz = Int(rowptr(A)[i+1] - rowptr(A)[i])
        max_row_nnz = max(max_row_nnz, row_nnz)
    end
    strong_coarse_cols = Ti[]
    strong_coarse_nz_idx = Ti[]
    denom_nz_idx = Ti[]
    sizehint!(strong_coarse_cols, max_row_nnz; shrink=false)
    sizehint!(strong_coarse_nz_idx, max_row_nnz; shrink=false)
    sizehint!(denom_nz_idx, max_row_nnz; shrink=false)

    # Second pass: fill sparsity pattern (colval) and build index mappings
    # NOTE: We do NOT compute nzv_p values here - Phase 2 will do that
    @inbounds for i in 1:n_fine
        pos = rp[i]
        
        # Build strong neighbor list if storing update map
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
            # Coarse point: identity mapping, P=1
            cval[pos] = Ti(coarse_map[i])
            entry_type[pos] = Ti(0)  # type 0 = coarse point (P=1)
            numer_idx[pos] = Ti(0)
            denom_offsets[pos] = Ti(length(denom_entries_list) + 1)
        else
            # Clear workspace buffers (reuse allocations)
            empty!(strong_coarse_cols)
            empty!(strong_coarse_nz_idx)
            empty!(denom_nz_idx)
            
            for nz in nzrange(A, i)
                j = cv[nz]
                if j == i
                    # Diagonal always in denominator
                    push!(denom_nz_idx, Ti(nz))
                else
                    # Match hypre: all strong C-connections are interpolation
                    # targets, non-strong/fine connections go to denominator
                    is_interp_coarse = is_strong[nz] && cf[j] == 1
                    if is_interp_coarse
                        push!(strong_coarse_cols, Ti(coarse_map[j]))
                        push!(strong_coarse_nz_idx, Ti(nz))
                    else
                        # Weak/fine connections go to denominator
                        push!(denom_nz_idx, Ti(nz))
                    end
                end
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
                entry_type[pos] = Ti(0)  # fallback = P=1
                numer_idx[pos] = Ti(0)
                denom_offsets[pos] = Ti(length(denom_entries_list) + 1)
            else
                # Fill colval and update map for each interpolation entry
                for k in eachindex(strong_coarse_cols)
                    cval[pos] = strong_coarse_cols[k]
                    entry_type[pos] = Ti(1)  # type 1 = compute from formula
                    numer_idx[pos] = strong_coarse_nz_idx[k]
                    # Denominator is the same for all P entries in this row
                    denom_offsets[pos] = Ti(length(denom_entries_list) + 1)
                    append!(denom_entries_list, denom_nz_idx)
                    pos += 1
                end
            end
        end
    end

    # Finalize update map
    denom_offsets[total_nnz + 1] = Ti(length(denom_entries_list) + 1)
    denom_entries = Vector{Ti}(denom_entries_list)
    
    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Compute P values using the update kernel
    # ══════════════════════════════════════════════════════════════════════════
    # This is the SAME code path used by update_P=true resetup,
    # ensuring initial setup and resetup produce identical results.
    
    # Build P operator (values will be filled by kernel)
    P = if old_P_reuse !== nothing && old_P_reuse.colval isa Vector
        old_P_reuse.nrow = n_fine
        old_P_reuse.ncol = n_coarse
        old_P_reuse.trunc_scaling = nothing
        if setup_workspace !== nothing
            setup_workspace.old_P = nothing
        end
        old_P_reuse
    else
        ProlongationOp{Ti, Tv}(rp, cval, nzv_p, n_fine, n_coarse)
    end
    
    # Use kernel to compute P values (same as resetup)
    # For CPU backend, this is a simple loop; for GPU, it will be a kernel launch
    _direct_interp_compute_values!(P.nzval, nonzeros(A_in), entry_type, numer_idx, 
                                   denom_offsets, denom_entries;
                                   backend=backend, block_size=block_size)
    
    # Build and return update map if requested
    P_update_map = nothing
    if build_update_map
        strong_nbrs_offsets[n_fine + 1] = Ti(length(strong_nbrs_cols_list) + 1)
        strong_nbrs_cols = Vector{Ti}(strong_nbrs_cols_list)
        strong_nbrs_nz = Vector{Ti}(strong_nbrs_nz_list)
        # IMPORTANT: Copy only valid portions of workspace arrays to avoid
        # out-of-bounds access during P update (workspace arrays may be larger
        # than needed for this level due to grow-only resizing)
        nnz_A = nnz(A)
        is_strong_copy = is_strong[1:nnz_A]
        cf_copy = cf[1:n_fine]
        coarse_map_copy = coarse_map[1:n_fine]
        # Direct interpolation doesn't need workspace, but we include empty buffers
        P_update_map = ProlongationUpdateMap{Ti, Tv}(
            1,  # interp_type = Direct
            is_strong_copy,
            cf_copy,
            coarse_map_copy,
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
            Tv[],   # P_data (not used for Direct)
            # GPU kernel data for Standard (10 fields, empty for Direct)
            Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[],
            # GPU kernel data for Extended+i (13 fields, empty for Direct)
            Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[]
        )
    end
    
    return P, P_update_map
end

"""
    _direct_interp_compute_values!(P_nzval, A_nzval, entry_type, numer_idx, denom_offsets, denom_entries)

Compute Direct interpolation P values using KernelAbstractions.
This is the same kernel used for both initial setup and update_P=true resetup.
"""
function _direct_interp_compute_values!(P_nzval::AbstractVector{Tv}, A_nzval::AbstractVector{Tv},
                                        entry_type::AbstractVector{Ti}, numer_idx::AbstractVector{Ti},
                                        denom_offsets::AbstractVector{Ti}, denom_entries::AbstractVector{Ti};
                                        backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    nnz_P = length(P_nzval)
    nnz_P == 0 && return P_nzval
    
    # Determine if we're doing GPU computation (P.nzval on GPU)
    P_on_gpu = !(P_nzval isa Array)
    A_on_gpu = !(A_nzval isa Array)
    
    if P_on_gpu
        # P is on GPU: convert update map arrays to GPU and use GPU kernel
        be = _get_backend(P_nzval)
        # Convert A values to GPU if needed
        if A_on_gpu
            A_nzval_gpu = A_nzval
        else
            A_nzval_gpu = similar(P_nzval, eltype(A_nzval), length(A_nzval))
            copyto!(A_nzval_gpu, A_nzval)
        end
        # Convert update map arrays to GPU
        entry_type_gpu = similar(P_nzval, Ti, length(entry_type))
        numer_idx_gpu = similar(P_nzval, Ti, length(numer_idx))
        denom_offsets_gpu = similar(P_nzval, Ti, length(denom_offsets))
        denom_entries_gpu = similar(P_nzval, Ti, length(denom_entries))
        copyto!(entry_type_gpu, entry_type)
        copyto!(numer_idx_gpu, numer_idx)
        copyto!(denom_offsets_gpu, denom_offsets)
        copyto!(denom_entries_gpu, denom_entries)
        
        kernel! = _p_direct_update_kernel!(be, block_size)
        kernel!(P_nzval, A_nzval_gpu, entry_type_gpu, numer_idx_gpu, denom_offsets_gpu, denom_entries_gpu; ndrange=nnz_P)
        _synchronize(be)
    else
        # P is on CPU: use CPU computation
        be = _get_backend(P_nzval)
        # Convert A values to CPU if needed
        A_nzval_cpu = A_on_gpu ? Array(A_nzval) : A_nzval
        kernel! = _p_direct_update_kernel!(be, block_size)
        kernel!(P_nzval, A_nzval_cpu, entry_type, numer_idx, denom_offsets, denom_entries; ndrange=nnz_P)
        _synchronize(be)
    end
    
    return P_nzval
end

# ── Standard (Classical) interpolation ───────────────────────────────────────

"""
    _build_interpolation(A, cf, coarse_map, n_coarse, ::StandardInterpolation)

Standard (classical Ruge-Stüben) interpolation. For each fine point i:
- Strong coarse neighbors contribute directly
- Strong fine neighbors contribute indirectly through their coarse connections

w_j = -(a_{i,j} + Σ_{k∈F_i^s} a_{i,k} * a_{k,j} / Σ_{m∈C_i} a_{k,m}) / d_i
where d_i = a_{i,i} + Σ_{k∈weak} a_{i,k}

Uses a **two-phase approach** when `build_update_map=true`:
1. **Phase 1**: Build sparsity pattern (which coarse columns each row interpolates from)
   and update map (graph structure for recomputation)
2. **Phase 2**: Call `_update_P_standard!` to compute values using the same code path as resetup

This ensures that initial setup and `update_P=true` resetup produce **identical** results.
"""
function _build_interpolation(A_in::CSRMatrix{Tv, Ti}, cf::Vector{Int},
                              coarse_map::Vector{Int}, n_coarse::Int,
                              ::StandardInterpolation, θ::Real=0.25;
                              backend=DEFAULT_BACKEND, block_size::Int=64,
                              setup_workspace=nothing,
                              build_update_map::Bool=false,
                              coarsening_is_strong::Union{Nothing, AbstractVector{Bool}}=nothing) where {Tv, Ti}
    # Use the coarsening strength graph if provided (ensures consistency when max_row_sum < 1.0),
    # otherwise recompute from A_in.
    if coarsening_is_strong !== nothing
        is_strong = coarsening_is_strong isa Array ? coarsening_is_strong : Array(coarsening_is_strong)
    else
        is_strong_raw = strength_graph(A_in, θ; backend=backend, block_size=block_size,
            is_strong=setup_workspace !== nothing ? setup_workspace.is_strong : nothing)
        is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    end
    A = csr_to_cpu(A_in)
    n_fine = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Build sparsity pattern and update map (graph structure)
    # ══════════════════════════════════════════════════════════════════════════
    
    # Build strong neighbor structure for update map
    diag_nz_idx = Vector{Ti}(undef, n_fine)
    strong_nbrs_offsets = Vector{Ti}(undef, n_fine + 1)
    strong_nbrs_cols_list = Ti[]
    strong_nbrs_nz_list = Ti[]
    sizehint!(strong_nbrs_cols_list, nnz(A); shrink=false)
    sizehint!(strong_nbrs_nz_list, nnz(A); shrink=false)
    
    # First pass: find diagonal indices and build strong neighbor structure
    @inbounds for i in 1:n_fine
        diag_nz_idx[i] = Ti(0)
        for nz in nzrange(A, i)
            if cv[nz] == i
                diag_nz_idx[i] = Ti(nz)
                break
            end
        end
        strong_nbrs_offsets[i] = Ti(length(strong_nbrs_cols_list) + 1)
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && is_strong[nz]
                push!(strong_nbrs_cols_list, Ti(j))
                push!(strong_nbrs_nz_list, Ti(nz))
            end
        end
    end
    strong_nbrs_offsets[n_fine + 1] = Ti(length(strong_nbrs_cols_list) + 1)

    # Build P sparsity using COO format
    # We need to determine which coarse columns each row interpolates from
    nnz_hint = nnz(A)
    if setup_workspace !== nothing
        I_p = setup_workspace.I_p
        J_p = setup_workspace.J_p
        V_p = setup_workspace.V_p
        empty!(I_p); empty!(J_p); empty!(V_p)
        sizehint!(I_p, nnz_hint; shrink=false)
        sizehint!(J_p, nnz_hint; shrink=false)
        sizehint!(V_p, nnz_hint; shrink=false)
    else
        I_p = Ti[]
        J_p = Ti[]
        V_p = Tv[]
        sizehint!(I_p, nnz_hint; shrink=false)
        sizehint!(J_p, nnz_hint; shrink=false)
        sizehint!(V_p, nnz_hint; shrink=false)
    end

    # For update map: store entry types (needed for coarse vs fine row distinction)
    entry_types_list = Ti[]
    numer_idx_list = Ti[]
    denom_offsets_list = Ti[1]
    denom_entries_list = Ti[]
    sizehint!(entry_types_list, nnz_hint; shrink=false)
    sizehint!(numer_idx_list, nnz_hint; shrink=false)
    sizehint!(denom_entries_list, nnz_hint * 4; shrink=false)

    # ═══════════════════════════════════════════════════════════════════════════
    # GPU kernel data for Standard interpolation
    # ═══════════════════════════════════════════════════════════════════════════
    # Per P entry: direct contribution index, fine neighbor contributions
    std_direct_numer_idx_list = Ti[]
    std_fine_offsets_list = Ti[1]
    std_a_ik_list = Ti[]
    std_a_kJ_list = Ti[]
    std_diag_k_list = Ti[]
    std_a_ki_list = Ti[]
    std_sum_offsets_list = Ti[1]
    std_sum_indices_list = Ti[]
    std_d_base_offsets_list = Ti[1]
    std_d_base_entries_list = Ti[]
    sizehint!(std_direct_numer_idx_list, nnz_hint; shrink=false)
    sizehint!(std_a_ik_list, nnz_hint * 2; shrink=false)
    sizehint!(std_a_kJ_list, nnz_hint * 2; shrink=false)
    sizehint!(std_diag_k_list, nnz_hint * 2; shrink=false)
    sizehint!(std_a_ki_list, nnz_hint * 2; shrink=false)
    sizehint!(std_sum_indices_list, nnz_hint * 4; shrink=false)
    sizehint!(std_d_base_entries_list, nnz_hint * 4; shrink=false)

    # Pre-allocate workspace buffers for inner loop (avoid per-row allocations)
    max_row_nnz = 0
    for i in 1:n_fine
        row_nnz = Int(rowptr(A)[i+1] - rowptr(A)[i])
        max_row_nnz = max(max_row_nnz, row_nnz)
    end
    weak_nz_indices = Ti[]
    strong_fine = Tuple{Int, Tv, Ti}[]
    sizehint!(weak_nz_indices, max_row_nnz; shrink=false)
    sizehint!(strong_fine, max_row_nnz; shrink=false)
    
    # Additional workspace for GPU kernel data building
    # For each fine neighbor, store: (fine_idx, a_ik_idx, diag_k_idx, a_ki_idx, sum_C_k_indices)
    fine_nbr_info = Tuple{Int, Ti, Ti, Ti, Vector{Ti}}[]
    sizehint!(fine_nbr_info, max_row_nnz; shrink=false)

    # Determine sparsity pattern by computing which coarse columns contribute
    # NOTE: We still compute values here for sparsity determination, but
    # Phase 2 will recompute using the same function as resetup
    @inbounds for i in 1:n_fine
        if cf[i] == 1
            # Coarse point: single entry to self
            push!(I_p, Ti(i)); push!(J_p, Ti(coarse_map[i])); push!(V_p, one(Tv))
            push!(entry_types_list, Ti(0))  # coarse = P=1
            push!(numer_idx_list, Ti(0))
            push!(denom_offsets_list, Ti(length(denom_entries_list) + 1))
            # GPU kernel data: coarse point has entry_type=0, so no fine contributions
            push!(std_direct_numer_idx_list, Ti(0))
            push!(std_fine_offsets_list, Ti(length(std_a_ik_list) + 1))
            push!(std_d_base_offsets_list, Ti(length(std_d_base_entries_list) + 1))
            continue
        end
        
        # Fine point: determine which coarse columns get entries
        a_ii = zero(Tv)
        diag_nz = diag_nz_idx[i]
        if diag_nz > 0
            a_ii = nzv[diag_nz]
        end
        sum_weak = zero(Tv)
        empty!(weak_nz_indices)  # reuse allocation
        strong_coarse = Dict{Int, Tv}()       # coarse_map[j] → a_{i,j}
        strong_coarse_nz = Dict{Int, Ti}()    # coarse_map[j] → A.nzval index
        empty!(strong_fine)  # reuse allocation
        
        for nz in nzrange(A, i)
            j = cv[nz]
            if j == i
                continue  # already handled
            elseif is_strong[nz] && cf[j] == 1
                cm = coarse_map[j]
                strong_coarse[cm] = get(strong_coarse, cm, zero(Tv)) + nzv[nz]
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
        
        # ═══════════════════════════════════════════════════════════════════════
        # Build GPU kernel data: process fine neighbors and collect index info
        # ═══════════════════════════════════════════════════════════════════════
        # For each fine neighbor, we need to record:
        # - Which coarse columns it contributes to (via a_{k,c} where c is in strong_coarse)
        # - The A.nzval indices for a_{i,k}, a_{k,c}, diag_k, a_{k,i}, and sum_C_k
        
        empty!(fine_nbr_info)
        contributions = Dict{Int, Tv}()
        # Maps: coarse_map -> list of (fine_nbr_idx_in_fine_nbr_info, a_kJ_idx) 
        fine_contribs_per_cm = Dict{Int, Vector{Tuple{Int, Ti}}}()
        
        for (cm, a_ij) in strong_coarse
            contributions[cm] = a_ij
        end
        
        for (k, a_ik, nz_ik) in strong_fine
            # Find diagonal of k
            diag_k_nz = Ti(0)
            for nz2 in nzrange(A, k)
                if cv[nz2] == k
                    diag_k_nz = Ti(nz2)
                    break
                end
            end
            
            # Find a_{k,i} index (no sign filtering — matching hypre's Standard
            # interpolation where all connections participate in redistribution)
            a_ki_nz = Ti(0)
            for nz2 in nzrange(A, k)
                if cv[nz2] == i
                    a_ki_nz = Ti(nz2)
                    break
                end
            end
            
            # Build sum_C_k indices and collect coarse contributions.
            # Include ALL a_{k,c} for c in C-hat(i) and a_{k,i} without sign
            # filtering, matching hypre's Standard interpolation formula.
            sum_C_k_indices = Ti[]
            sum_C_k = zero(Tv)
            coarse_vals_k = Dict{Int, Tuple{Tv, Ti}}()  # cm -> (a_kc, nz_idx)
            
            for nz2 in nzrange(A, k)
                j2 = cv[nz2]
                j2 == k && continue
                a_kj = nzv[nz2]
                if cf[j2] == 1
                    cm2 = coarse_map[j2]
                    if haskey(strong_coarse, cm2)
                        old = get(coarse_vals_k, cm2, (zero(Tv), Ti(0)))
                        coarse_vals_k[cm2] = (old[1] + a_kj, Ti(nz2))
                        sum_C_k += a_kj
                        push!(sum_C_k_indices, Ti(nz2))
                    end
                end
                if j2 == i
                    sum_C_k += a_kj
                    push!(sum_C_k_indices, Ti(nz2))
                end
            end
            
            # Store fine neighbor info for GPU kernel
            fine_nbr_idx = length(fine_nbr_info) + 1
            push!(fine_nbr_info, (k, nz_ik, diag_k_nz, a_ki_nz, sum_C_k_indices))
            
            # Record which coarse columns this fine neighbor contributes to
            for (cm2, (a_kj, nz_idx)) in coarse_vals_k
                if !haskey(fine_contribs_per_cm, cm2)
                    fine_contribs_per_cm[cm2] = Tuple{Int, Ti}[]
                end
                push!(fine_contribs_per_cm[cm2], (fine_nbr_idx, nz_idx))
            end
            
            # Update contributions (keep original logic for value computation)
            if abs(sum_C_k) > eps(real(Tv))
                distribute = a_ik / sum_C_k
                for (cm2, (a_kj, _)) in coarse_vals_k
                    contributions[cm2] = get(contributions, cm2, zero(Tv)) + distribute * a_kj
                end
                if a_ki_nz > 0
                    d_i += distribute * nzv[a_ki_nz]
                end
            else
                d_i += a_ik
            end
        end
        
        if isempty(contributions)
            # Fallback: map to nearest coarse point
            best_j = _find_nearest_coarse(A, i, cf, coarse_map)
            push!(I_p, Ti(i)); push!(J_p, Ti(best_j)); push!(V_p, one(Tv))
            push!(entry_types_list, Ti(0))  # fallback = P=1
            push!(numer_idx_list, Ti(0))
            push!(denom_offsets_list, Ti(length(denom_entries_list) + 1))
            # GPU kernel data: fallback has entry_type=0
            push!(std_direct_numer_idx_list, Ti(0))
            push!(std_fine_offsets_list, Ti(length(std_a_ik_list) + 1))
            push!(std_d_base_offsets_list, Ti(length(std_d_base_entries_list) + 1))
        else
            # Add entries for all contributing coarse columns
            for (cm, val) in contributions
                w = abs(d_i) > eps(real(Tv)) ? -val / d_i : zero(Tv)
                push!(I_p, Ti(i)); push!(J_p, Ti(cm)); push!(V_p, w)
                # For Standard interpolation, mark as type 2 (needs full recomputation)
                push!(entry_types_list, Ti(2))
                numer_nz = get(strong_coarse_nz, cm, Ti(0))
                push!(numer_idx_list, numer_nz)
                push!(denom_offsets_list, Ti(length(denom_entries_list) + 1))
                if diag_nz > 0
                    push!(denom_entries_list, diag_nz)
                end
                append!(denom_entries_list, weak_nz_indices)
                
                # ═══════════════════════════════════════════════════════════════
                # Build GPU kernel data for this P entry (row i, coarse column cm)
                # ═══════════════════════════════════════════════════════════════
                
                # Direct contribution (if any)
                direct_idx = get(strong_coarse_nz, cm, Ti(0))
                push!(std_direct_numer_idx_list, direct_idx)
                
                # Fine neighbor contributions to this coarse column
                fine_contribs = get(fine_contribs_per_cm, cm, Tuple{Int, Ti}[])
                push!(std_fine_offsets_list, Ti(length(std_a_ik_list) + 1))
                for (fnbr_idx, a_kJ_idx) in fine_contribs
                    (_, nz_ik, diag_k_nz, a_ki_nz, sum_C_k_indices) = fine_nbr_info[fnbr_idx]
                    push!(std_a_ik_list, nz_ik)
                    push!(std_a_kJ_list, a_kJ_idx)
                    push!(std_diag_k_list, diag_k_nz)
                    push!(std_a_ki_list, a_ki_nz)
                    push!(std_sum_offsets_list, Ti(length(std_sum_indices_list) + 1))
                    append!(std_sum_indices_list, sum_C_k_indices)
                end
                
                # Base d_i indices (diagonal + weak)
                push!(std_d_base_offsets_list, Ti(length(std_d_base_entries_list) + 1))
                if diag_nz > 0
                    push!(std_d_base_entries_list, diag_nz)
                end
                append!(std_d_base_entries_list, weak_nz_indices)
            end
        end
    end

    # Convert COO to CSR (with placeholder values that will be overwritten in Phase 2)
    old_P_reuse = setup_workspace !== nothing ? setup_workspace.old_P : nothing
    P = _coo_to_prolongation(I_p, J_p, V_p, n_fine, n_coarse;
        old_P=old_P_reuse,
        sort_perm=setup_workspace !== nothing ? setup_workspace.sort_perm : nothing)
    if setup_workspace !== nothing
        setup_workspace.old_P = nothing
    end
    
    # Build update map with full graph structure
    entry_type = Vector{Ti}(entry_types_list)
    numer_idx = Vector{Ti}(numer_idx_list)
    denom_offsets = Vector{Ti}(denom_offsets_list)
    denom_entries = Vector{Ti}(denom_entries_list)
    strong_nbrs_cols = Vector{Ti}(strong_nbrs_cols_list)
    strong_nbrs_nz = Vector{Ti}(strong_nbrs_nz_list)
    
    # Build GPU kernel data arrays
    std_direct_numer_idx = Vector{Ti}(std_direct_numer_idx_list)
    std_fine_offsets = Vector{Ti}(std_fine_offsets_list)
    std_a_ik = Vector{Ti}(std_a_ik_list)
    std_a_kJ = Vector{Ti}(std_a_kJ_list)
    std_diag_k = Vector{Ti}(std_diag_k_list)
    std_a_ki = Vector{Ti}(std_a_ki_list)
    std_sum_offsets = Vector{Ti}(std_sum_offsets_list)
    std_sum_indices = Vector{Ti}(std_sum_indices_list)
    std_d_base_offsets = Vector{Ti}(std_d_base_offsets_list)
    std_d_base_entries = Vector{Ti}(std_d_base_entries_list)
    
    # Allocate workspace for Standard interpolation update
    max_strong = 0
    for i in 1:n_fine
        num_strong = Int(strong_nbrs_offsets[i+1] - strong_nbrs_offsets[i])
        max_strong = max(max_strong, num_strong)
    end
    
    # IMPORTANT: Copy only valid portions of workspace arrays
    nnz_A = nnz(A)
    is_strong_copy = is_strong[1:nnz_A]
    cf_copy = cf[1:n_fine]
    coarse_map_copy = coarse_map[1:n_fine]
    
    P_update_map = ProlongationUpdateMap{Ti, Tv}(
        2,  # interp_type = Standard
        is_strong_copy,
        cf_copy,
        coarse_map_copy,
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
        Vector{Tv}(undef, max_strong + 1),   # P_data buffer
        # GPU kernel data for Standard (10 fields)
        std_direct_numer_idx,
        std_fine_offsets,
        std_a_ik,
        std_a_kJ,
        std_diag_k,
        std_a_ki,
        std_sum_offsets,
        std_sum_indices,
        std_d_base_offsets,
        std_d_base_entries,
        # GPU kernel data for Extended+i (13 fields, empty for Standard)
        Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[]
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Compute P values using the update function
    # ══════════════════════════════════════════════════════════════════════════
    # This uses the SAME code path as update_P=true resetup,
    # ensuring initial setup and resetup produce identical results.
    
    if build_update_map
        # Use the update function to compute P values
        _update_P_standard!(P, A_in, P_update_map)
    end
    # If not building update map, values are already computed in Phase 1
    
    # Return update map only if requested
    return P, build_update_map ? P_update_map : nothing
end

# ── Extended+i interpolation ─────────────────────────────────────────────────

"""
    _build_interpolation(A, cf, coarse_map, n_coarse, ::ExtendedIInterpolation)

Extended+i interpolation. Extends standard interpolation by including distance-2
coarse points (coarse points connected through fine neighbors) as direct
interpolation targets, resulting in a larger but more accurate interpolation stencil.

Uses a **two-phase approach** when `build_update_map=true`:
1. **Phase 1**: Build sparsity pattern (which coarse columns each row interpolates from,
   including truncation decisions) and update map (graph structure for recomputation)
2. **Phase 2**: Call `_update_P_extendedi!` to compute values using the same code path as resetup

This ensures that initial setup and `update_P=true` resetup produce **identical** results.
"""
function _build_interpolation(A_in::CSRMatrix{Tv, Ti}, cf::Vector{Int},
                              coarse_map::Vector{Int}, n_coarse::Int,
                              interp::ExtendedIInterpolation, θ::Real=0.25;
                              backend=DEFAULT_BACKEND, block_size::Int=64,
                              setup_workspace=nothing,
                              build_update_map::Bool=false,
                              coarsening_is_strong::Union{Nothing, AbstractVector{Bool}}=nothing) where {Tv, Ti}
    # Use the coarsening strength graph if provided (ensures consistency when max_row_sum < 1.0),
    # otherwise recompute from A_in.
    if coarsening_is_strong !== nothing
        is_strong = coarsening_is_strong isa Array ? coarsening_is_strong : Array(coarsening_is_strong)
    else
        is_strong_raw = strength_graph(A_in, θ; backend=backend, block_size=block_size,
            is_strong=setup_workspace !== nothing ? setup_workspace.is_strong : nothing)
        is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    end
    A = csr_to_cpu(A_in)
    n_fine = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)

    trunc_factor = interp.trunc_factor
    max_elements = interp.max_elements
    norm_p = interp.norm_p
    do_rescale = interp.rescale

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Build sparsity pattern and update map (graph structure)
    # ══════════════════════════════════════════════════════════════════════════

    # Build strength-based CSR structure for determining C-hat.
    sn_offsets = Vector{Int}(undef, n_fine + 1)
    sn_data = Int[]
    sizehint!(sn_data, nnz(A); shrink=false)
    
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
        P_marker = _ws_resize!(setup_workspace.P_marker, n_fine)
        fill!(P_marker, -1)
    else
        P_marker = fill(-1, n_fine)
    end
    strong_f_marker = -2

    nnz_hint = nnz(A)
    I_p = Ti[]
    J_p = Ti[]
    V_p = Tv[]
    sizehint!(I_p, nnz_hint; shrink=false)
    sizehint!(J_p, nnz_hint; shrink=false)
    sizehint!(V_p, nnz_hint; shrink=false)
    
    S_p = do_rescale ? Tv[] : nothing  # per-entry scaling factors
    if do_rescale
        sizehint!(S_p, nnz_hint; shrink=false)
    end

    # Pre-allocate workspace for C-hat indices (avoid per-row allocations)
    # Estimate max C-hat size from max strong neighbors * 2 (distance-2)
    max_chat_est = 0
    for i in 1:n_fine
        if cf[i] == -1
            count = Int(sn_offsets[i + 1] - sn_offsets[i]) * 2
            max_chat_est = max(max_chat_est, count)
        end
    end
    max_chat_est = max(max_chat_est, 1)
    chat_indices = Int[]
    sizehint!(chat_indices, max_chat_est; shrink=false)

    # Pre-allocate workspace arrays for weight computation (avoid per-row allocations)
    P_data_ws = zeros(Tv, max_chat_est)
    keep_ws = Vector{Int}(undef, max_chat_est)
    kept_flags = Vector{Bool}(undef, max_chat_est)

    # Build sparsity pattern and compute initial values
    @inbounds for i in 1:n_fine
        if cf[i] == 1
            push!(I_p, Ti(i)); push!(J_p, Ti(coarse_map[i])); push!(V_p, one(Tv))
            if do_rescale; push!(S_p, one(Tv)); end
            continue
        end

        # Determine C-hat (extended coarse interpolation set)
        empty!(chat_indices)  # reuse allocation

        for si in sn_offsets[i]:(sn_offsets[i + 1] - 1)
            j = sn_data[si]
            if cf[j] == 1
                if P_marker[j] < 0
                    P_marker[j] = length(chat_indices)
                    push!(chat_indices, j)
                end
            elseif cf[j] == -1
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

        # Compute weights (matching hypre's ExtPI formula)
        # Grow pre-allocated workspace if needed
        if n_chat > length(P_data_ws)
            resize!(P_data_ws, n_chat)
            resize!(keep_ws, n_chat)
            resize!(kept_flags, n_chat)
        end
        for idx in 1:n_chat
            P_data_ws[idx] = zero(Tv)
        end
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
                P_data_ws[p_idx + 1] += a_ij
            elseif p_idx == strong_f_marker
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
                                P_data_ws[p_idx_m + 1] += distribute * a_jm
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

        # Finalize weights: P[j] = P_data_ws[j] / (-diagonal)
        if abs(diagonal) > eps(real(Tv))
            for idx in 1:n_chat
                P_data_ws[idx] /= -diagonal
            end
        end

        # Truncation (trunc_factor + max_elements limit)
        n_keep = n_chat
        for idx in 1:n_chat
            keep_ws[idx] = idx
        end
        if trunc_factor > 0 && n_chat > 0
            max_w = zero(real(Tv))
            for idx in 1:n_chat
                max_w = max(max_w, abs(P_data_ws[idx])^norm_p)
            end
            threshold = trunc_factor * max_w
            n_keep = 0
            for idx in 1:n_chat
                if abs(P_data_ws[idx])^norm_p >= threshold
                    n_keep += 1
                    keep_ws[n_keep] = idx
                end
            end
        end
        if max_elements > 0 && n_keep > max_elements
            sort!(view(keep_ws, 1:n_keep); by = idx -> abs(P_data_ws[idx]), rev = true)
            n_keep = max_elements
        end
        
        # Compute rescaling factor if enabled
        row_scale = one(Tv)
        if do_rescale && n_keep < n_chat
            # Mark which indices are kept
            for idx in 1:n_chat
                kept_flags[idx] = false
            end
            for k in 1:n_keep
                kept_flags[keep_ws[k]] = true
            end
            sum_removed = zero(Tv)
            for idx in 1:n_chat
                if !kept_flags[idx]
                    sum_removed += P_data_ws[idx]
                end
            end
            denom = one(Tv) - sum_removed
            if abs(denom) > Tv(1e-12)
                row_scale = one(Tv) / denom
            end
        end
        for k in 1:n_keep
            idx = keep_ws[k]
            push!(I_p, Ti(i)); push!(J_p, Ti(coarse_map[chat_indices[idx]])); push!(V_p, P_data_ws[idx] * row_scale)
            if do_rescale; push!(S_p, row_scale); end
        end

        # Reset markers
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

    # Convert COO to CSR
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
        # Build diagonal indices
        diag_nz_idx = Vector{Ti}(undef, n_fine)
        strong_nbrs_nz_list = Ti[]
        sizehint!(strong_nbrs_nz_list, nnz(A); shrink=false)
        
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
        
        # Copy valid portions of arrays
        total_strong_nbrs = sn_offsets[n_fine + 1] - 1
        strong_nbrs_offsets = Vector{Ti}(sn_offsets[1:n_fine+1])
        strong_nbrs_cols = Vector{Ti}(sn_data[1:total_strong_nbrs])
        strong_nbrs_nz = Vector{Ti}(strong_nbrs_nz_list)
        nnz_A = nnz(A)
        is_strong_copy = is_strong[1:nnz_A]
        cf_copy = cf[1:n_fine]
        coarse_map_copy = coarse_map[1:n_fine]
        
        # Allocate workspace for Extended+i interpolation update
        max_chat = 0
        for i in 1:n_fine
            if cf_copy[i] == -1
                count = 0
                for si in strong_nbrs_offsets[i]:(strong_nbrs_offsets[i+1]-1)
                    j = strong_nbrs_cols[si]
                    if cf_copy[j] == 1
                        count += 1
                    elseif cf_copy[j] == -1
                        for sj in strong_nbrs_offsets[j]:(strong_nbrs_offsets[j+1]-1)
                            if cf_copy[strong_nbrs_cols[sj]] == 1
                                count += 1
                            end
                        end
                    end
                end
                max_chat = max(max_chat, count)
            end
        end
        max_chat = max(max_chat, 1)
        
        # ══════════════════════════════════════════════════════════════════════
        # Build GPU kernel data for Extended+i interpolation
        # ══════════════════════════════════════════════════════════════════════
        # For each P entry k (row i, P column J), we pre-compute:
        # - extd_p_col[k]: target P column J
        # - extd_direct_a_idx[k]: A.nzval index for direct contribution
        # - Fine neighbor contributions (extd_fine_offsets, extd_a_ik, etc.)
        # - Base d_i entries (diagonal + weak connections)
        
        # Reuse extd arrays from old P_update_map (set per-level in _build_levels!)
        # to avoid allocating fresh arrays on every resetup call.
        _old_pmap = setup_workspace !== nothing ? setup_workspace.old_P_update_map : nothing
        if _old_pmap !== nothing
            extd_entry_row_list = _old_pmap.extd_entry_row;   empty!(extd_entry_row_list)
            extd_p_col_list = _old_pmap.extd_p_col;           empty!(extd_p_col_list)
            extd_direct_a_idx_list = _old_pmap.extd_direct_a_idx; empty!(extd_direct_a_idx_list)
            extd_fine_offsets_list = _old_pmap.extd_fine_offsets; empty!(extd_fine_offsets_list)
            extd_a_ik_list = _old_pmap.extd_a_ik;             empty!(extd_a_ik_list)
            extd_diag_k_list = _old_pmap.extd_diag_k;         empty!(extd_diag_k_list)
            extd_sum_offsets_list = _old_pmap.extd_sum_offsets; empty!(extd_sum_offsets_list)
            extd_sum_indices_list = _old_pmap.extd_sum_indices; empty!(extd_sum_indices_list)
            extd_contrib_offsets_list = _old_pmap.extd_contrib_offsets; empty!(extd_contrib_offsets_list)
            extd_contrib_a_idx_list = _old_pmap.extd_contrib_a_idx; empty!(extd_contrib_a_idx_list)
            extd_contrib_p_col_list = _old_pmap.extd_contrib_p_col; empty!(extd_contrib_p_col_list)
            extd_d_base_offsets_list = _old_pmap.extd_d_base_offsets; empty!(extd_d_base_offsets_list)
            extd_d_base_entries_list = _old_pmap.extd_d_base_entries; empty!(extd_d_base_entries_list)
        else
            extd_entry_row_list = Ti[]
            extd_p_col_list = Ti[]
            extd_direct_a_idx_list = Ti[]
            extd_fine_offsets_list = Ti[]      # NO initial value
            extd_a_ik_list = Ti[]
            extd_diag_k_list = Ti[]
            extd_sum_offsets_list = Ti[]       # NO initial value
            extd_sum_indices_list = Ti[]
            extd_contrib_offsets_list = Ti[]   # NO initial value
            extd_contrib_a_idx_list = Ti[]
            extd_contrib_p_col_list = Ti[]  # 0 = contributes to d_i, otherwise = P column
            extd_d_base_offsets_list = Ti[]    # NO initial value
            extd_d_base_entries_list = Ti[]
        end
        
        # Size hints
        sizehint!(extd_entry_row_list, nnz_P; shrink=false)
        sizehint!(extd_p_col_list, nnz_P; shrink=false)
        sizehint!(extd_direct_a_idx_list, nnz_P; shrink=false)
        sizehint!(extd_a_ik_list, nnz_P * 2; shrink=false)
        sizehint!(extd_diag_k_list, nnz_P * 2; shrink=false)
        sizehint!(extd_sum_indices_list, nnz_P * 4; shrink=false)
        sizehint!(extd_contrib_a_idx_list, nnz_P * 4; shrink=false)
        sizehint!(extd_contrib_p_col_list, nnz_P * 4; shrink=false)
        sizehint!(extd_d_base_entries_list, nnz_P * 4; shrink=false)
        
        # P_marker2: tracks which coarse points are in current row's C-hat
        P_marker2 = fill(-1, n_fine)
        strong_f_marker2 = -2
        
        # Build lookup: for each row i, map from coarse column to A.nzval index (if direct neighbor)
        # and also build P column to P.nzval index mapping
        P_rowptr = P.rowptr isa Array ? P.rowptr : Array(P.rowptr)
        P_colval = P.colval isa Array ? P.colval : Array(P.colval)
        
        @inbounds for i in 1:n_fine
            p_start = Int(P_rowptr[i])
            p_end = Int(P_rowptr[i+1]) - 1
            
            if cf_copy[i] == 1
                # Coarse point: P entry = 1, entry_type = 0 (already set)
                for p_nz in p_start:p_end
                    push!(extd_entry_row_list, Ti(i))
                    push!(extd_p_col_list, Ti(P_colval[p_nz]))
                    push!(extd_direct_a_idx_list, Ti(0))
                    push!(extd_fine_offsets_list, Ti(length(extd_a_ik_list) + 1))
                    push!(extd_d_base_offsets_list, Ti(length(extd_d_base_entries_list) + 1))
                end
                continue
            end
            
            # Fine point: determine C-hat and build GPU kernel data
            # Step 1: Determine C-hat (extended coarse interpolation set)
            empty!(chat_indices)  # reuse allocation from sparsity pattern building
            
            for si in strong_nbrs_offsets[i]:(strong_nbrs_offsets[i+1]-1)
                j = strong_nbrs_cols[si]
                if cf_copy[j] == 1
                    if P_marker2[j] < 0
                        P_marker2[j] = length(chat_indices)
                        push!(chat_indices, j)
                    end
                elseif cf_copy[j] == -1
                    P_marker2[j] = strong_f_marker2
                    for sj in strong_nbrs_offsets[j]:(strong_nbrs_offsets[j+1]-1)
                        k = strong_nbrs_cols[sj]
                        if cf_copy[k] == 1 && P_marker2[k] < 0
                            P_marker2[k] = length(chat_indices)
                            push!(chat_indices, k)
                        end
                    end
                end
            end
            
            n_chat = length(chat_indices)
            
            if n_chat == 0
                # No C-hat: fallback entry
                for p_nz in p_start:p_end
                    push!(extd_entry_row_list, Ti(i))
                    push!(extd_p_col_list, Ti(P_colval[p_nz]))
                    push!(extd_direct_a_idx_list, Ti(0))
                    push!(extd_fine_offsets_list, Ti(length(extd_a_ik_list) + 1))
                    push!(extd_d_base_offsets_list, Ti(length(extd_d_base_entries_list) + 1))
                    entry_type[p_nz] = Ti(0)  # treat as fallback
                end
                # Reset markers
                for si in strong_nbrs_offsets[i]:(strong_nbrs_offsets[i+1]-1)
                    j = strong_nbrs_cols[si]
                    P_marker2[j] = -1
                    if cf_copy[j] == -1
                        for sj in strong_nbrs_offsets[j]:(strong_nbrs_offsets[j+1]-1)
                            P_marker2[strong_nbrs_cols[sj]] = -1
                        end
                    end
                end
                strong_f_marker2 -= 1
                continue
            end
            
            # Step 2: For each P entry in this row, build GPU kernel data
            for p_nz in p_start:p_end
                target_col = Int(P_colval[p_nz])  # coarse column J
                push!(extd_entry_row_list, Ti(i))
                push!(extd_p_col_list, Ti(target_col))
                
                # Find direct contribution: coarse neighbor c where coarse_map[c] == target_col
                direct_idx = Ti(0)
                for nz in nzrange(A, i)
                    j = cv[nz]
                    if j != i && cf_copy[j] == 1 && coarse_map_copy[j] == target_col
                        direct_idx = Ti(nz)
                        break
                    end
                end
                push!(extd_direct_a_idx_list, direct_idx)
                
                # Base d_i: push offset BEFORE adding entries
                push!(extd_d_base_offsets_list, Ti(length(extd_d_base_entries_list) + 1))
                # Add diagonal
                if diag_nz_idx[i] > 0
                    push!(extd_d_base_entries_list, diag_nz_idx[i])
                end
                # Add weak connections (not strong, not in C-hat)
                for nz in nzrange(A, i)
                    j = cv[nz]
                    if j != i && !is_strong_copy[nz]
                        push!(extd_d_base_entries_list, Ti(nz))
                    end
                end
                
                # Fine neighbor contributions: push offset BEFORE adding entries
                push!(extd_fine_offsets_list, Ti(length(extd_a_ik_list) + 1))
                
                for nz in nzrange(A, i)
                    k = cv[nz]
                    if k != i && cf_copy[k] == -1 && P_marker2[k] == strong_f_marker2
                        # k is a strong fine neighbor
                        # Find A[i,k] (which is nz)
                        nz_ik = nz
                        
                        # Find diag(k)
                        diag_k_nz = diag_nz_idx[k]
                        
                        push!(extd_a_ik_list, Ti(nz_ik))
                        push!(extd_diag_k_list, diag_k_nz)
                        
                        # sum_C_k and contrib: push offset BEFORE adding entries
                        push!(extd_sum_offsets_list, Ti(length(extd_sum_indices_list) + 1))
                        push!(extd_contrib_offsets_list, Ti(length(extd_contrib_a_idx_list) + 1))
                        
                        # Compute sum_C_k indices: A[k,m] where m in C-hat OR m == i
                        for nz2 in nzrange(A, k)
                            m = cv[nz2]
                            if m != k && (P_marker2[m] >= 0 || m == i)
                                push!(extd_sum_indices_list, Ti(nz2))
                            end
                        end
                        
                        # Contribution data: A[k,m] where m in C-hat OR m == i
                        for nz2 in nzrange(A, k)
                            m = cv[nz2]
                            if m != k
                                p_col_contrib = Ti(0)  # default = contributes to d_i if m == i
                                if P_marker2[m] >= 0
                                    # m is a coarse point in C-hat
                                    p_col_contrib = Ti(coarse_map_copy[m])
                                elseif m == i
                                    p_col_contrib = Ti(0)  # contribution to d_i
                                else
                                    continue  # not in C-hat and not i, skip
                                end
                                push!(extd_contrib_a_idx_list, Ti(nz2))
                                push!(extd_contrib_p_col_list, p_col_contrib)
                            end
                        end
                    end
                end
            end
            
            # Reset markers for this row
            for c in chat_indices
                P_marker2[c] = -1
            end
            for si in strong_nbrs_offsets[i]:(strong_nbrs_offsets[i+1]-1)
                j = strong_nbrs_cols[si]
                P_marker2[j] = -1
                if cf_copy[j] == -1
                    for sj in strong_nbrs_offsets[j]:(strong_nbrs_offsets[j+1]-1)
                        P_marker2[strong_nbrs_cols[sj]] = -1
                    end
                end
            end
            strong_f_marker2 -= 1
        end
        
        # Finalize offset arrays: add terminators
        # Each offset array needs a final entry pointing past the last data element
        push!(extd_fine_offsets_list, Ti(length(extd_a_ik_list) + 1))
        push!(extd_sum_offsets_list, Ti(length(extd_sum_indices_list) + 1))
        push!(extd_contrib_offsets_list, Ti(length(extd_contrib_a_idx_list) + 1))
        push!(extd_d_base_offsets_list, Ti(length(extd_d_base_entries_list) + 1))
        
        P_update_map = ProlongationUpdateMap{Ti, Tv}(
            3,  # interp_type = Extended+i
            is_strong_copy,
            cf_copy,
            coarse_map_copy,
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
            Vector{Tv}(undef, max_chat),   # P_data buffer
            # GPU kernel data for Standard (10 fields, empty for Extended+i)
            Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[],
            # GPU kernel data for Extended+i (13 fields) — reused arrays, no copy needed
            extd_entry_row_list,
            extd_p_col_list,
            extd_direct_a_idx_list,
            extd_fine_offsets_list,
            extd_a_ik_list,
            extd_diag_k_list,
            extd_sum_offsets_list,
            extd_sum_indices_list,
            extd_contrib_offsets_list,
            extd_contrib_a_idx_list,
            extd_contrib_p_col_list,
            extd_d_base_offsets_list,
            extd_d_base_entries_list
        )
        if setup_workspace !== nothing
            # Clear the reference — the extd arrays now live in P_update_map.
            # On the next level, _build_levels! will set old_P_update_map from the
            # new level before this function is called again (same pattern as old_P).
            setup_workspace.old_P_update_map = nothing
        end
        
        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2: Recompute P values using the update function
        # ══════════════════════════════════════════════════════════════════════
        # This uses the SAME code path as update_P=true resetup,
        # ensuring initial setup and resetup produce identical results.
        
        _update_P_extendedi!(P, A_in, P_update_map; backend=backend, block_size=block_size)
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
        _update_P_standard!(level.P, A, P_update_map; backend=backend, block_size=block_size)
    elseif interp_type == 3
        # Extended+i interpolation: full row recomputation
        _update_P_extendedi!(level.P, A, P_update_map; backend=backend, block_size=block_size)
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
    A_nzval = nonzeros(A)
    P_nzval = P.nzval
    entry_type = P_update_map.entry_type
    numer_idx = P_update_map.numer_idx
    denom_offsets = P_update_map.denom_offsets
    denom_entries = P_update_map.denom_entries
    
    # Use the same function that handles GPU/CPU mixing
    _direct_interp_compute_values!(P_nzval, A_nzval, entry_type, numer_idx, 
                                   denom_offsets, denom_entries;
                                   backend=backend, block_size=block_size)
    
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
Standard interpolation GPU kernel.

For each P entry k, computes:
  P[k] = -(direct_contrib + Σ indirect_contribs) / d_i

where:
- direct_contrib = A[std_direct_numer_idx[k]] (if non-zero)
- indirect_contrib = A[a_ik] * A[a_kJ] / sum_C_k
- d_i = Σ A[d_base_entries] + Σ (A[a_ik] * A[a_ki] / sum_C_k, if a_ki contributes)
"""
@kernel function _p_standard_update_kernel!(P_nzval, @Const(A_nzval), @Const(entry_type),
                                             @Const(std_direct_numer_idx), @Const(std_fine_offsets),
                                             @Const(std_a_ik), @Const(std_a_kJ), @Const(std_diag_k),
                                             @Const(std_a_ki), @Const(std_sum_offsets), @Const(std_sum_indices),
                                             @Const(std_d_base_offsets), @Const(std_d_base_entries))
    k = @index(Global)
    @inbounds begin
        if entry_type[k] == 0
            # Coarse point or fallback: P value = 1
            P_nzval[k] = one(eltype(P_nzval))
        else
            # Compute numerator (direct + indirect contributions)
            numerator = zero(eltype(A_nzval))
            
            # Direct contribution
            direct_idx = std_direct_numer_idx[k]
            if direct_idx > 0
                numerator += A_nzval[direct_idx]
            end
            
            # Compute base denominator (diagonal + weak)
            d_i = zero(eltype(A_nzval))
            for j in std_d_base_offsets[k]:(std_d_base_offsets[k+1]-1)
                d_i += A_nzval[std_d_base_entries[j]]
            end
            
            # Indirect contributions from fine neighbors
            for fnbr_idx in std_fine_offsets[k]:(std_fine_offsets[k+1]-1)
                # Get a_{i,k} value
                a_ik_idx = std_a_ik[fnbr_idx]
                a_ik_val = A_nzval[a_ik_idx]
                
                # Get diagonal of fine neighbor (for sign determination)
                diag_k_idx = std_diag_k[fnbr_idx]
                diag_k_val = diag_k_idx > 0 ? A_nzval[diag_k_idx] : zero(eltype(A_nzval))
                
                # Compute sum_C_k
                sum_C_k = zero(eltype(A_nzval))
                for j in std_sum_offsets[fnbr_idx]:(std_sum_offsets[fnbr_idx+1]-1)
                    sum_C_k += A_nzval[std_sum_indices[j]]
                end
                
                if abs(sum_C_k) > eps(real(eltype(A_nzval)))
                    distribute = a_ik_val / sum_C_k
                    
                    # Indirect contribution to numerator
                    a_kJ_idx = std_a_kJ[fnbr_idx]
                    if a_kJ_idx > 0
                        numerator += distribute * A_nzval[a_kJ_idx]
                    end
                    
                    # Contribution to d_i from a_{k,i}
                    a_ki_idx = std_a_ki[fnbr_idx]
                    if a_ki_idx > 0
                        d_i += distribute * A_nzval[a_ki_idx]
                    end
                else
                    # If sum_C_k is too small, add a_{i,k} to d_i
                    d_i += a_ik_val
                end
            end
            
            # Final P value
            abs_d_i = abs(d_i)
            threshold = eps(real(eltype(A_nzval))) * max(one(real(eltype(A_nzval))), abs_d_i)
            if abs_d_i > threshold
                P_nzval[k] = -numerator / d_i
            else
                P_nzval[k] = zero(eltype(P_nzval))
            end
        end
    end
end

"""
Compute Standard interpolation P values using a KA kernel.
The map arrays are assumed to already reside on the same device as `P_nzval`.
Returns true if kernel data was available, false to fall back to CPU implementation.
"""
function _standard_interp_compute_values!(P_nzval::AbstractVector{Tv}, A_nzval::AbstractVector{Tv},
                                          P_update_map::ProlongationUpdateMap{Ti, Tv2};
                                          backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti, Tv2}
    nnz_P = length(P_nzval)
    nnz_P == 0 && return true  # Empty P is considered success
    
    # Get kernel data (already on the correct device)
    entry_type = P_update_map.entry_type
    std_direct_numer_idx = P_update_map.std_direct_numer_idx
    std_fine_offsets = P_update_map.std_fine_offsets
    std_a_ik = P_update_map.std_a_ik
    std_a_kJ = P_update_map.std_a_kJ
    std_diag_k = P_update_map.std_diag_k
    std_a_ki = P_update_map.std_a_ki
    std_sum_offsets = P_update_map.std_sum_offsets
    std_sum_indices = P_update_map.std_sum_indices
    std_d_base_offsets = P_update_map.std_d_base_offsets
    std_d_base_entries = P_update_map.std_d_base_entries
    
    # Check if kernel data is available
    if isempty(std_direct_numer_idx)
        return false
    end
    
    # Ensure A values are on the same device as P
    A_nzval_dev = _match_device(P_nzval, A_nzval)
    
    be = _get_backend(P_nzval)
    kernel! = _p_standard_update_kernel!(be, block_size)
    kernel!(P_nzval, A_nzval_dev, entry_type,
            std_direct_numer_idx, std_fine_offsets,
            std_a_ik, std_a_kJ, std_diag_k,
            std_a_ki, std_sum_offsets, std_sum_indices,
            std_d_base_offsets, std_d_base_entries; ndrange=nnz_P)
    _synchronize(be)
    
    return true  # Success
end

# ══════════════════════════════════════════════════════════════════════════════
# Extended+i Interpolation GPU Kernel
# ══════════════════════════════════════════════════════════════════════════════

"""
Extended+i interpolation GPU kernel. Computes P values for Extended+i interpolation
using pre-computed index maps. The structure is similar to Standard but includes
distance-2 coarse points in the interpolation stencil.

For each P entry k:
- entry_type[k] == 0: Coarse point, P = 1
- entry_type[k] == 3: Fine point, compute full formula

The formula is: P[k] = -numerator / d_i where:
- numerator = direct_contrib + sum of indirect contributions through fine neighbors
- d_i = diagonal + weak + redistributed fine neighbor contributions to diagonal
"""
@kernel function _p_extendedi_update_kernel!(P_nzval, @Const(A_nzval), @Const(entry_type),
                                             @Const(extd_direct_a_idx), @Const(extd_fine_offsets),
                                             @Const(extd_a_ik), @Const(extd_diag_k),
                                             @Const(extd_sum_offsets), @Const(extd_sum_indices),
                                             @Const(extd_contrib_offsets), @Const(extd_contrib_a_idx),
                                             @Const(extd_contrib_p_col), @Const(target_p_col),
                                             @Const(extd_d_base_offsets), @Const(extd_d_base_entries))
    k = @index(Global)
    @inbounds begin
        if entry_type[k] == 0
            # Coarse point: P value = 1
            P_nzval[k] = one(eltype(P_nzval))
        else
            # Fine point: compute numerator and denominator
            numerator = zero(eltype(A_nzval))
            target_col = target_p_col[k]  # The coarse column this P entry interpolates to
            
            # Direct contribution: A[i,J] if J is a direct strong coarse neighbor
            direct_idx = extd_direct_a_idx[k]
            if direct_idx > 0
                numerator += A_nzval[direct_idx]
            end
            
            # Compute base denominator (diagonal + weak connections)
            d_i = zero(eltype(A_nzval))
            for j in extd_d_base_offsets[k]:(extd_d_base_offsets[k+1]-1)
                d_i += A_nzval[extd_d_base_entries[j]]
            end
            
            # Indirect contributions through fine neighbors
            for fnbr_idx in extd_fine_offsets[k]:(extd_fine_offsets[k+1]-1)
                # Get A[i,k] value
                a_ik_idx = extd_a_ik[fnbr_idx]
                a_ik_val = A_nzval[a_ik_idx]
                
                # Get diagonal of fine neighbor k (for sign determination in sum_C_k)
                diag_k_idx = extd_diag_k[fnbr_idx]
                diag_k_val = diag_k_idx > 0 ? A_nzval[diag_k_idx] : zero(eltype(A_nzval))
                
                # Compute sum_C_k (sum of connections from k to C-hat ∪ {i})
                sum_C_k = zero(eltype(A_nzval))
                for j in extd_sum_offsets[fnbr_idx]:(extd_sum_offsets[fnbr_idx+1]-1)
                    sum_C_k += A_nzval[extd_sum_indices[j]]
                end
                
                if abs(sum_C_k) > eps(real(eltype(A_nzval)))
                    distribute = a_ik_val / sum_C_k
                    
                    # Distribute contributions to C-hat and diagonal
                    for contrib_idx in extd_contrib_offsets[fnbr_idx]:(extd_contrib_offsets[fnbr_idx+1]-1)
                        a_km_idx = extd_contrib_a_idx[contrib_idx]
                        contrib_col = extd_contrib_p_col[contrib_idx]
                        a_km_val = A_nzval[a_km_idx]
                        
                        if contrib_col == target_col
                            # Contribution to this P entry's numerator
                            numerator += distribute * a_km_val
                        elseif contrib_col == 0
                            # Contribution to diagonal (contrib_col == 0 means this is A[k,i])
                            d_i += distribute * a_km_val
                        end
                        # Note: contributions to other P columns are handled by their respective P entries
                    end
                else
                    # If sum_C_k is too small, add A[i,k] to d_i
                    d_i += a_ik_val
                end
            end
            
            # Final P value: P = -numerator / d_i
            abs_d_i = abs(d_i)
            threshold = eps(real(eltype(A_nzval))) * max(one(real(eltype(A_nzval))), abs_d_i)
            if abs_d_i > threshold
                P_nzval[k] = -numerator / d_i
            else
                P_nzval[k] = zero(eltype(P_nzval))
            end
        end
    end
end

"""
Compute Extended+i interpolation P values using a KA kernel.
The map arrays are assumed to already reside on the same device as `P_nzval`.
Returns true if kernel data was available, false to fall back to CPU implementation.
"""
function _extendedi_interp_compute_values!(P_nzval::AbstractVector{Tv}, A_nzval::AbstractVector{Tv},
                                           P_update_map::ProlongationUpdateMap{Ti, Tv2};
                                           backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti, Tv2}
    nnz_P = length(P_nzval)
    nnz_P == 0 && return true  # Empty P is considered success
    
    # Get kernel data (already on the correct device)
    entry_type = P_update_map.entry_type
    extd_direct_a_idx = P_update_map.extd_direct_a_idx
    extd_fine_offsets = P_update_map.extd_fine_offsets
    extd_a_ik = P_update_map.extd_a_ik
    extd_diag_k = P_update_map.extd_diag_k
    extd_sum_offsets = P_update_map.extd_sum_offsets
    extd_sum_indices = P_update_map.extd_sum_indices
    extd_contrib_offsets = P_update_map.extd_contrib_offsets
    extd_contrib_a_idx = P_update_map.extd_contrib_a_idx
    extd_contrib_p_col = P_update_map.extd_contrib_p_col
    extd_p_col = P_update_map.extd_p_col  # target P column for each entry
    extd_d_base_offsets = P_update_map.extd_d_base_offsets
    extd_d_base_entries = P_update_map.extd_d_base_entries
    
    # Check if kernel data is available
    if isempty(extd_direct_a_idx) || isempty(extd_p_col)
        return false  # Fall back to CPU implementation
    end
    
    # Ensure A values are on the same device as P
    A_nzval_dev = _match_device(P_nzval, A_nzval)
    
    be = _get_backend(P_nzval)
    kernel! = _p_extendedi_update_kernel!(be, block_size)
    kernel!(P_nzval, A_nzval_dev, entry_type,
            extd_direct_a_idx, extd_fine_offsets,
            extd_a_ik, extd_diag_k,
            extd_sum_offsets, extd_sum_indices,
            extd_contrib_offsets, extd_contrib_a_idx,
            extd_contrib_p_col, extd_p_col,
            extd_d_base_offsets, extd_d_base_entries; ndrange=nnz_P)
    _synchronize(be)
    
    return true  # Success
end

"""
    _update_P_standard!(P, A, P_update_map; backend, block_size)

Standard interpolation P value update.
Uses GPU kernel when available, falls back to CPU implementation.
"""
function _update_P_standard!(P::ProlongationOp{Ti, Tv}, A::CSRMatrix{Tv, Ti},
                             P_update_map::ProlongationUpdateMap{Ti, Tv2};
                             backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti, Tv2}
    # Try GPU kernel first if data is available
    if !isempty(P_update_map.std_direct_numer_idx)
        success = _standard_interp_compute_values!(P.nzval, nonzeros(A), P_update_map;
                                                    backend=backend, block_size=block_size)
        if success
            return P
        end
    end
    
    # Fall back to CPU implementation
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
            sum_C_k = zero(Tv)
            coarse_vals_k = Dict{Int, Tv}()
            for nz2 in nzrange(A_cpu, k)
                j2 = cv[nz2]
                j2 == k && continue
                a_kj = nzv[nz2]
                if cf[j2] == 1
                    cm2 = coarse_map[j2]
                    if haskey(strong_coarse, cm2)
                        coarse_vals_k[cm2] = get(coarse_vals_k, cm2, zero(Tv)) + a_kj
                        sum_C_k += a_kj
                    end
                end
                if j2 == i
                    sum_C_k += a_kj
                end
            end
            if abs(sum_C_k) > eps(real(Tv))
                distribute = a_ik / sum_C_k
                for (cm2, a_kj) in coarse_vals_k
                    contributions[cm2] = get(contributions, cm2, zero(Tv)) + distribute * a_kj
                end
                for nz2 in nzrange(A_cpu, k)
                    if cv[nz2] == i
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
    _update_P_extendedi!(P, A, P_update_map; backend, block_size)

Extended+i interpolation P value update.
Uses GPU kernel when available, falls back to CPU implementation.
Uses workspace buffers from P_update_map to avoid per-row allocations.
"""
function _update_P_extendedi!(P::ProlongationOp{Ti, Tv}, A::CSRMatrix{Tv, Ti},
                              P_update_map::ProlongationUpdateMap{Ti, Tv2};
                              backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti, Tv2}
    # Try GPU kernel first if data is available
    if !isempty(P_update_map.extd_direct_a_idx)
        success = _extendedi_interp_compute_values!(P.nzval, nonzeros(A), P_update_map;
                                                     backend=backend, block_size=block_size)
        if success
            return P
        end
    end
    
    # Fall back to CPU implementation
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
