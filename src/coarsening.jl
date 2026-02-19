# ── Aggregation coarsening ────────────────────────────────────────────────────

"""
    coarsen_aggregation(A, θ)

Multi-pass aggregation coarsening inspired by MueLu/PyAMG. Uses a distance-2
MIS (maximal independent set) for seed selection to ensure well-separated seeds,
then assigns remaining nodes to neighboring aggregates, and finally merges small
aggregates to prevent singleton-dominated coarsening at deeper levels.

Returns `agg::Vector{Int}` where `agg[i]` is the aggregate index (1-based) that
node `i` belongs to, and `n_coarse` the number of aggregates.
"""
function coarsen_aggregation(A_in::CSRMatrix{Tv, Ti}, θ::Real;
                             backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    n = size(A_in, 1)
    # Edge case: trivial system
    if n <= 1
        return ones(Int, n), n
    end
    # Compute strength on GPU if available, then convert to CPU for graph algorithms
    is_strong_raw = strength_graph(A_in, θ; backend=backend, block_size=block_size)
    is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    A = csr_to_cpu(A_in)
    cv = colvals(A)
    nzv = nonzeros(A)
    agg = zeros(Int, n)  # 0 = unassigned
    n_coarse = 0

    # Compute number of strong neighbors for each node (used for seed priority)
    strong_count = zeros(Int, n)
    @inbounds for i in 1:n
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && is_strong[nz]
                strong_count[i] += 1
            end
        end
    end

    # Phase 1: distance-2 MIS seed selection
    # A node can be a seed only if no node within distance 2 in the strong
    # graph is already a seed. Process nodes in decreasing strong-count order
    # so high-connectivity nodes seed first, creating larger aggregates.
    state = zeros(Int8, n)  # 0=available, 1=seed, -1=neighbor-of-seed, -2=dist-2-of-seed
    order = sortperm(strong_count; rev=true)
    @inbounds for idx in 1:n
        i = order[idx]
        state[i] != 0 && continue
        # Check if any strong neighbor is already a seed or neighbor-of-seed
        can_seed = true
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && is_strong[nz] && (state[j] == 1 || state[j] == -1)
                can_seed = false
                break
            end
        end
        if !can_seed
            continue
        end
        # Make i a seed
        state[i] = 1
        n_coarse += 1
        agg[i] = n_coarse
        # Mark all strong neighbors of i: they join aggregate and are
        # marked as "neighbor-of-seed" so they won't become seeds
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && is_strong[nz] && state[j] == 0
                state[j] = -1
                agg[j] = n_coarse
                # Mark distance-2 neighbors (neighbors of j) to prevent
                # them from becoming seeds too close
                for nz2 in nzrange(A, j)
                    k = cv[nz2]
                    if k != j && is_strong[nz2] && state[k] == 0
                        state[k] = -2
                    end
                end
            end
        end
    end

    # Phase 2: assign unaggregated nodes to strongest neighbor's aggregate
    @inbounds for i in 1:n
        agg[i] != 0 && continue
        best_agg = 0
        best_val = zero(real(Tv))
        # Prefer strong connections
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && is_strong[nz] && agg[j] != 0 && abs(nzv[nz]) > best_val
                best_val = abs(nzv[nz])
                best_agg = agg[j]
            end
        end
        # Fall back to any connection
        if best_agg == 0
            for nz in nzrange(A, i)
                j = cv[nz]
                if j != i && agg[j] != 0 && abs(nzv[nz]) > best_val
                    best_val = abs(nzv[nz])
                    best_agg = agg[j]
                end
            end
        end
        if best_agg != 0
            agg[i] = best_agg
        else
            # Truly isolated node: create singleton aggregate
            n_coarse += 1
            agg[i] = n_coarse
        end
    end

    # Phase 3: merge small aggregates (size ≤ 1) into neighboring aggregates
    # This prevents the pathological case of many singleton aggregates at
    # coarser levels where the strong graph becomes sparse.
    # Sizes are tracked only at union-find roots for correctness.
    aggregate_sizes = zeros(Int, n_coarse)
    @inbounds for i in 1:n
        aggregate_sizes[agg[i]] += 1
    end
    # Build merge map using union-find
    merge_map = collect(1:n_coarse)
    @inbounds for i in 1:n
        my_root = find_root!(merge_map, agg[i])
        aggregate_sizes[my_root] > 1 && continue
        # Find the best neighboring aggregate to merge into
        best_target = 0
        best_val = zero(real(Tv))
        for nz in nzrange(A, i)
            j = cv[nz]
            j == i && continue
            j_root = find_root!(merge_map, agg[j])
            if j_root != my_root && abs(nzv[nz]) > best_val
                best_val = abs(nzv[nz])
                best_target = j_root
            end
        end
        if best_target != 0
            # Merge my_root into best_target; accumulate size at new root
            old_my_size = aggregate_sizes[my_root]
            old_tgt_size = aggregate_sizes[best_target]
            union_roots!(merge_map, best_target, my_root)
            new_root = find_root!(merge_map, best_target)
            aggregate_sizes[new_root] = old_my_size + old_tgt_size
        end
    end
    # Compact aggregate numbering after merges
    new_id = zeros(Int, n_coarse)
    new_count = 0
    for i in 1:n_coarse
        root = find_root!(merge_map, i)
        if new_id[root] == 0
            new_count += 1
            new_id[root] = new_count
        end
        new_id[i] = new_id[root]
    end
    @inbounds for i in 1:n
        agg[i] = new_id[find_root!(merge_map, agg[i])]
    end
    n_coarse = new_count

    return agg, n_coarse
end

# ── PMIS coarsening ──────────────────────────────────────────────────────────

"""
    _compute_strong_transpose_count(A, is_strong)

Compute for each node i the number of nodes j that strongly depend on i,
i.e., the number of strong connections in the TRANSPOSE graph. This is the
column-based measure used by hypre for better PMIS coarsening.
"""
function _compute_strong_transpose_count(A::CSRMatrix{Tv, Ti}, is_strong::AbstractVector{Bool};
                                         setup_workspace=nothing) where {Tv, Ti}
    n = size(A, 1)
    cv = colvals(A)
    if setup_workspace !== nothing
        st_count = resize!(setup_workspace.st_count, n)
        fill!(st_count, 0)
    else
        st_count = zeros(Int, n)
    end
    @inbounds for i in 1:n
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && is_strong[nz]
                st_count[j] += 1  # j is strongly depended upon by i
            end
        end
    end
    return st_count
end

"""
    coarsen_pmis(A, θ; rng=Random.default_rng())

Parallel Modified Independent Set (PMIS) coarsening. Uses column-based strength
count (number of points that depend on each node) for the measure, as in hypre.
Returns `cf::Vector{Int}` where `cf[i] = 1` for coarse points and `cf[i] = -1`
for fine points, `coarse_map::Vector{Int}`, and `n_coarse`.
"""
function coarsen_pmis(A_in::CSRMatrix{Tv, Ti}, θ::Real;
                      rng=Random.default_rng(),
                      backend=DEFAULT_BACKEND, block_size::Int=64,
                      setup_workspace=nothing) where {Tv, Ti}
    n = size(A_in, 1)
    if n <= 1
        return ones(Int, n), collect(1:n), n
    end
    # Compute strength on GPU if available, then convert to CPU for graph algorithms
    is_strong_raw = strength_graph(A_in, θ; backend=backend, block_size=block_size)
    is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    A = csr_to_cpu(A_in)
    cv = colvals(A)
    # Use column-based measure: how many nodes strongly depend on i
    st_count = _compute_strong_transpose_count(A, is_strong; setup_workspace=setup_workspace)
    if setup_workspace !== nothing
        measure = resize!(setup_workspace.measure, n)
    else
        measure = zeros(Float64, n)
    end
    @inbounds for i in 1:n
        measure[i] = Float64(st_count[i]) + rand(rng)
    end
    # Mark isolated nodes (no strong connections at all) immediately as coarse
    if setup_workspace !== nothing
        cf = resize!(setup_workspace.cf, n)
        fill!(cf, 0)
    else
        cf = zeros(Int, n)
    end
    @inbounds for i in 1:n
        has_strong = false
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && is_strong[nz]
                has_strong = true
                break
            end
        end
        if !has_strong && st_count[i] == 0
            cf[i] = 1  # isolated → coarse
        end
    end
    # CF splitting: iterative PMIS
    max_iter = n + 1
    for iter in 1:max_iter
        all_decided = true
        @inbounds for i in 1:n
            cf[i] != 0 && continue
            all_decided = false
            is_max = true
            for nz in nzrange(A, i)
                j = cv[nz]
                if is_strong[nz] && j != i && cf[j] != -1
                    if measure[j] > measure[i]
                        is_max = false
                        break
                    end
                end
            end
            if is_max
                cf[i] = 1  # coarse point
            end
        end
        @inbounds for i in 1:n
            cf[i] != 0 && continue
            for nz in nzrange(A, i)
                j = cv[nz]
                if is_strong[nz] && cf[j] == 1
                    cf[i] = -1  # fine point
                    break
                end
            end
        end
        all_decided && break
    end
    @inbounds for i in 1:n
        if cf[i] == 0
            cf[i] = 1
        end
    end
    # Build coarse map
    n_coarse = 0
    if setup_workspace !== nothing
        coarse_map = resize!(setup_workspace.coarse_map, n)
        fill!(coarse_map, 0)
    else
        coarse_map = zeros(Int, n)
    end
    @inbounds for i in 1:n
        if cf[i] == 1
            n_coarse += 1
            coarse_map[i] = n_coarse
        end
    end
    return cf, coarse_map, n_coarse
end

"""
    _ensure_fine_have_coarse_neighbor!(cf, A, is_strong)

Second pass: any F-point that has no strong C-neighbor is promoted to C.
This guarantees the strong-connection property needed for interpolation.
"""
function _ensure_fine_have_coarse_neighbor!(cf::Vector{Int}, A::CSRMatrix, is_strong::AbstractVector{Bool})
    n = size(A, 1)
    cv = colvals(A)
    changed = true
    while changed
        changed = false
        @inbounds for i in 1:n
            cf[i] != -1 && continue
            has_strong_C = false
            for nz in nzrange(A, i)
                j = cv[nz]
                if j != i && is_strong[nz] && cf[j] == 1
                    has_strong_C = true
                    break
                end
            end
            if !has_strong_C
                cf[i] = 1  # promote to coarse
                changed = true
            end
        end
    end
end

# ── HMIS coarsening ──────────────────────────────────────────────────────────

"""
    coarsen_hmis(A, θ; rng=Random.default_rng())

Hybrid Modified Independent Set (HMIS) coarsening matching hypre's implementation.
HMIS = Ruge-Stüben first pass (greedy, bucket-based) followed by PMIS on remaining
undecided points. The RS first pass does aggressive greedy coarsening using the
transpose strength measure, then PMIS finalizes the splitting for remaining
undecided points.

This matches hypre's `hypre_BoomerAMGCoarsenHMIS` which calls
`hypre_BoomerAMGCoarsenRuge(S, A, measure_type, 10, ...)` (first pass only,
with f_pnt=Z_PT) followed by `hypre_BoomerAMGCoarsenPMIS(S, A, 1, ...)`.
"""
function coarsen_hmis(A_in::CSRMatrix{Tv, Ti}, θ::Real;
                      rng=Random.default_rng(),
                      backend=DEFAULT_BACKEND, block_size::Int=64,
                      setup_workspace=nothing) where {Tv, Ti}
    n = size(A_in, 1)
    if n <= 1
        return ones(Int, n), collect(1:n), n
    end
    # Compute strength on GPU if available, then convert to CPU for graph algorithms
    is_strong_raw = strength_graph(A_in, θ; backend=backend, block_size=block_size)
    is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    A = csr_to_cpu(A_in)
    cv = colvals(A)

    # ── Phase 1: RS first pass (greedy bucket-based coarsening) ──
    cf = _rs_first_pass!(A, is_strong; use_zpt=true, setup_workspace=setup_workspace)

    # ── Phase 2: PMIS on remaining undecided points ──
    st_count_pmis = _compute_strong_transpose_count(A, is_strong; setup_workspace=setup_workspace)
    if setup_workspace !== nothing
        pmis_measure = resize!(setup_workspace.measure, n)
    else
        pmis_measure = zeros(Float64, n)
    end
    @inbounds for i in 1:n
        pmis_measure[i] = Float64(st_count_pmis[i]) + rand(rng)
    end

    # Re-evaluate Z_PT nodes (matching hypre's CF_init=1 logic)
    @inbounds for i in 1:n
        if cf[i] == -2  # Z_PT from RS first pass
            has_strong_diag = false
            for nz in nzrange(A, i)
                j = cv[nz]
                if j != i && is_strong[nz]
                    has_strong_diag = true
                    break
                end
            end
            if pmis_measure[i] >= 1.0 || has_strong_diag
                cf[i] = 0  # undecided, will be handled by PMIS
            else
                cf[i] = -1  # no influence, make final F
            end
        end
    end

    # PMIS iterations on remaining undecided nodes
    _pmis_on_undecided!(cf, A, is_strong, pmis_measure)

    n_coarse = 0
    if setup_workspace !== nothing
        coarse_map = resize!(setup_workspace.coarse_map, n)
        fill!(coarse_map, 0)
    else
        coarse_map = zeros(Int, n)
    end
    @inbounds for i in 1:n
        if cf[i] == 1
            n_coarse += 1
            coarse_map[i] = n_coarse
        end
    end
    return cf, coarse_map, n_coarse
end

"""
    _rs_first_pass!(A, is_strong; use_zpt=false)

Run the Ruge-Stüben first pass (greedy bucket-based coarsening).
When `use_zpt=true`, nodes that become fine during initialization (zero-measure
nodes) are marked as Z_PT=-2 instead of F_PT=-1. This is used by HMIS
(matching hypre's coarsen_type=10 which sets f_pnt=Z_PT).
Nodes marked fine in the main RS greedy loop are always marked as F_PT=-1.
Returns the cf array with states: 0=undecided (shouldn't remain), 1=C, -1=F, -2=Z_PT.
"""
function _rs_first_pass!(A::CSRMatrix{Tv, Ti}, is_strong::AbstractVector{Bool};
                          use_zpt::Bool=false,
                          setup_workspace=nothing) where {Tv, Ti}
    n = size(A, 1)
    cv = colvals(A)
    f_pnt = use_zpt ? -2 : -1  # Z_PT or F_PT for zero-measure initialization

    λ = _compute_strong_transpose_count(A, is_strong; setup_workspace=setup_workspace)
    st_offsets, st_sources = _build_strong_transpose_adj(A, is_strong; setup_workspace=setup_workspace)

    if setup_workspace !== nothing
        cf = resize!(setup_workspace.cf, n)
        fill!(cf, 0)
    else
        cf = zeros(Int, n)
    end
    # Mark isolated nodes (no strong connections at all)
    @inbounds for i in 1:n
        has_strong = false
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && is_strong[nz]
                has_strong = true
                break
            end
        end
        if !has_strong && λ[i] == 0
            cf[i] = 1  # isolated → coarse
        end
    end

    # Process zero-measure nodes first: mark them with f_pnt and
    # increment λ for their strong neighbors.
    @inbounds for i in 1:n
        cf[i] != 0 && continue
        if λ[i] == 0
            # Zero measure → mark with f_pnt (Z_PT for HMIS, F_PT for RS)
            cf[i] = f_pnt
            # When a node becomes F, increment λ for its strong neighbors
            for nz in nzrange(A, i)
                j = cv[nz]
                if j != i && is_strong[nz] && cf[j] == 0
                    λ[j] += 1
                end
            end
        end
    end

    # Build bucket structure for greedy RS first pass.
    # Done after zero-measure processing so λ values are final.
    max_λ_val = 0
    @inbounds for i in 1:n
        cf[i] != 0 && continue
        max_λ_val = max(max_λ_val, λ[i])
    end
    if setup_workspace !== nothing
        bucket_head = resize!(setup_workspace.bucket_head, max(max_λ_val + 1, 1))
        fill!(bucket_head, 0)
        bucket_next = resize!(setup_workspace.bucket_next, n)
        fill!(bucket_next, 0)
        bucket_prev = resize!(setup_workspace.bucket_prev, n)
        fill!(bucket_prev, 0)
    else
        bucket_head = fill(0, max(max_λ_val + 1, 1))
        bucket_next = zeros(Int, n)
        bucket_prev = zeros(Int, n)
    end
    @inbounds for i in 1:n
        cf[i] != 0 && continue
        k = λ[i] + 1
        old_head = bucket_head[k]
        bucket_head[k] = i
        bucket_next[i] = old_head
        bucket_prev[i] = 0
        if old_head != 0
            bucket_prev[old_head] = i
        end
    end
    top_bucket = max_λ_val

    @inline function _bkt_remove!(i)
        @inbounds begin
            k = λ[i] + 1
            p = bucket_prev[i]
            nx = bucket_next[i]
            if p != 0
                bucket_next[p] = nx
            else
                bucket_head[k] = nx
            end
            if nx != 0
                bucket_prev[nx] = p
            end
            bucket_next[i] = 0
            bucket_prev[i] = 0
        end
    end
    @inline function _bkt_update!(i, new_λ)
        @inbounds begin
            _bkt_remove!(i)
            λ[i] = new_λ
            k = new_λ + 1
            old_len = length(bucket_head)
            if k > old_len
                resize!(bucket_head, k)
                for idx in (old_len + 1):k
                    bucket_head[idx] = 0
                end
            end
            old_head = bucket_head[k]
            bucket_head[k] = i
            bucket_next[i] = old_head
            bucket_prev[i] = 0
            if old_head != 0
                bucket_prev[old_head] = i
            end
        end
    end

    # Main RS first pass greedy loop
    while true
        while top_bucket >= 0 && bucket_head[top_bucket + 1] == 0
            top_bucket -= 1
        end
        top_bucket < 0 && break
        best_i = bucket_head[top_bucket + 1]
        best_i == 0 && break
        _bkt_remove!(best_i)
        cf[best_i] = 1  # C-point

        # For each undecided node j that strongly depends on best_i (S^T neighbors):
        # mark as F-point (F_PT=-1, not Z_PT, matching hypre's main loop behavior)
        @inbounds for idx in st_offsets[best_i]:(st_offsets[best_i + 1] - 1)
            j = st_sources[idx]
            cf[j] != 0 && continue
            _bkt_remove!(j)
            cf[j] = -1  # F-point (permanent)
            # Increment λ for undecided nodes that j strongly depends on
            for nz2 in nzrange(A, j)
                k = cv[nz2]
                if k != j && is_strong[nz2] && cf[k] == 0
                    new_val = λ[k] + 1
                    _bkt_update!(k, new_val)
                    if new_val > top_bucket
                        top_bucket = new_val
                    end
                end
            end
        end
        # Decrement λ for undecided nodes that best_i strongly depends on
        @inbounds for nz in nzrange(A, best_i)
            j = cv[nz]
            if j != best_i && is_strong[nz] && cf[j] == 0
                new_val = max(0, λ[j] - 1)
                _bkt_update!(j, new_val)
                if new_val == 0
                    # Node has no more undecided dependents → mark with f_pnt
                    _bkt_remove!(j)
                    cf[j] = f_pnt  # Z_PT for HMIS, F_PT for RS
                    # Increment λ for its undecided strong neighbors
                    for nz2 in nzrange(A, j)
                        k = cv[nz2]
                        if k != j && is_strong[nz2] && cf[k] == 0
                            new_val2 = λ[k] + 1
                            _bkt_update!(k, new_val2)
                            if new_val2 > top_bucket
                                top_bucket = new_val2
                            end
                        end
                    end
                end
            end
        end
    end

    return cf
end

"""
    _pmis_on_undecided!(cf, A, is_strong, measure)

Run PMIS iterations on undecided nodes (cf[i] == 0). Nodes with cf[i] == 1 (C)
or cf[i] == -1 (F) are not modified. This is used as the second phase of HMIS.
"""
function _pmis_on_undecided!(cf::Vector{Int}, A::CSRMatrix{Tv, Ti},
                              is_strong::AbstractVector{Bool},
                              measure::Vector{Float64}) where {Tv, Ti}
    n = size(A, 1)
    cv = colvals(A)
    max_iter = n + 1
    for iter in 1:max_iter
        all_decided = true
        # Identify local maxima as candidate C-points
        @inbounds for i in 1:n
            cf[i] != 0 && continue
            all_decided = false
            if measure[i] < 1.0
                cf[i] = -1  # no influence → F
                continue
            end
            is_max = true
            for nz in nzrange(A, i)
                j = cv[nz]
                if j != i && is_strong[nz] && cf[j] != -1
                    if measure[j] > measure[i]
                        is_max = false
                        break
                    end
                end
            end
            if is_max
                cf[i] = 1  # C-point
            end
        end
        # Mark undecided nodes adjacent to C-points as F
        @inbounds for i in 1:n
            cf[i] != 0 && continue
            for nz in nzrange(A, i)
                j = cv[nz]
                if is_strong[nz] && cf[j] == 1
                    cf[i] = -1
                    break
                end
            end
        end
        all_decided && break
    end
    # Any remaining undecided → F
    @inbounds for i in 1:n
        if cf[i] == 0
            cf[i] = -1
        end
    end
end

"""
    _symmetrize_strength(A, is_strong)

Build a symmetrized strength indicator: is_strong_sym[nz] = true iff the
connection (i,j) is strong AND (j,i) is also strong in the CSR structure.
"""
function _symmetrize_strength(A::CSRMatrix{Tv, Ti}, is_strong::AbstractVector{Bool}) where {Tv, Ti}
    n = size(A, 1)
    cv = colvals(A)
    is_strong_sym = copy(is_strong)
    @inbounds for i in 1:n
        for nz in nzrange(A, i)
            if !is_strong[nz]
                continue
            end
            j = cv[nz]
            j == i && continue
            found_reverse = false
            for nz2 in nzrange(A, j)
                if cv[nz2] == i && is_strong[nz2]
                    found_reverse = true
                    break
                end
            end
            if !found_reverse
                is_strong_sym[nz] = false
            end
        end
    end
    return is_strong_sym
end

# ── Classical Ruge-Stüben (RS) coarsening ─────────────────────────────────────

"""
    coarsen_rs(A, θ; rng=Random.default_rng())

Classical Ruge-Stüben coarsening with first pass (greedy selection based on
column-weight measure) and second pass (ensuring every F-point has interpolation
from C-points through strong connections).

This produces reliable coarsening ratios and is the classical AMG standard.
Uses a precomputed transpose graph and bucket sorting for O(nnz) complexity
instead of the naive O(n²) greedy scan.
"""
function coarsen_rs(A_in::CSRMatrix{Tv, Ti}, θ::Real;
                    rng=Random.default_rng(),
                    backend=DEFAULT_BACKEND, block_size::Int=64,
                    setup_workspace=nothing) where {Tv, Ti}
    n = size(A_in, 1)
    if n <= 1
        return ones(Int, n), collect(1:n), n
    end
    # Compute strength on GPU if available, then convert to CPU for graph algorithms
    is_strong_raw = strength_graph(A_in, θ; backend=backend, block_size=block_size)
    is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    A = csr_to_cpu(A_in)
    cv = colvals(A)
    # Compute λ_i = number of points that strongly depend on i (transpose measure)
    λ = _compute_strong_transpose_count(A, is_strong; setup_workspace=setup_workspace)
    # Precompute transpose adjacency
    st_offsets, st_sources = _build_strong_transpose_adj(A, is_strong; setup_workspace=setup_workspace)
    # CF splitting
    if setup_workspace !== nothing
        cf = resize!(setup_workspace.cf, n)
        fill!(cf, 0)
    else
        cf = zeros(Int, n)
    end
    # Mark isolated nodes immediately
    @inbounds for i in 1:n
        has_strong = false
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && is_strong[nz]
                has_strong = true
                break
            end
        end
        if !has_strong && λ[i] == 0
            cf[i] = 1  # isolated → coarse
        end
    end
    # First pass: greedy selection with bucket sorting for O(nnz) complexity.
    # Maintain a set of linked-list buckets indexed by λ value. Each step
    # picks from the highest non-empty bucket.
    max_λ_val = maximum(λ; init=0)
    # Build bucket structure: bucket_head[k+1] = first node with λ==k
    # (shift by 1 so λ==0 maps to index 1)
    # Pre-allocate to n to avoid resizing during λ increments
    if setup_workspace !== nothing
        bucket_head = resize!(setup_workspace.bucket_head, max(max_λ_val + 1, n))
        fill!(bucket_head, 0)
        bucket_next = resize!(setup_workspace.bucket_next, n)
        fill!(bucket_next, 0)
        bucket_prev = resize!(setup_workspace.bucket_prev, n)
        fill!(bucket_prev, 0)
    else
        bucket_head = fill(0, max(max_λ_val + 1, n))
        bucket_next = zeros(Int, n)
        bucket_prev = zeros(Int, n)
    end
    @inbounds for i in 1:n
        cf[i] != 0 && continue
        k = λ[i] + 1
        old_head = bucket_head[k]
        bucket_head[k] = i
        bucket_next[i] = old_head
        bucket_prev[i] = 0
        if old_head != 0
            bucket_prev[old_head] = i
        end
    end
    top_bucket = max_λ_val
    # Helper: remove node i from its bucket
    @inline function _bucket_remove!(i)
        @inbounds begin
            k = λ[i] + 1
            p = bucket_prev[i]
            nx = bucket_next[i]
            if p != 0
                bucket_next[p] = nx
            else
                bucket_head[k] = nx
            end
            if nx != 0
                bucket_prev[nx] = p
            end
            bucket_next[i] = 0
            bucket_prev[i] = 0
        end
    end
    # Helper: move node i to new bucket for updated λ value
    @inline function _bucket_update!(i, new_λ)
        @inbounds begin
            _bucket_remove!(i)
            λ[i] = new_λ
            k = new_λ + 1
            old_len = length(bucket_head)
            if k > old_len
                resize!(bucket_head, k)
                for idx in (old_len + 1):k
                    bucket_head[idx] = 0
                end
            end
            old_head = bucket_head[k]
            bucket_head[k] = i
            bucket_next[i] = old_head
            bucket_prev[i] = 0
            if old_head != 0
                bucket_prev[old_head] = i
            end
        end
    end
    while true
        # Find highest non-empty bucket
        while top_bucket >= 0 && bucket_head[top_bucket + 1] == 0
            top_bucket -= 1
        end
        top_bucket < 0 && break
        best_i = bucket_head[top_bucket + 1]
        best_i == 0 && break
        # Remove best_i from bucket and make it coarse
        _bucket_remove!(best_i)
        cf[best_i] = 1
        # For each undecided node j that strongly depends on best_i, make j fine
        @inbounds for idx in st_offsets[best_i]:(st_offsets[best_i + 1] - 1)
            j = st_sources[idx]
            cf[j] != 0 && continue
            _bucket_remove!(j)
            cf[j] = -1  # j becomes fine
            # Increment λ for undecided nodes that j strongly depends on
            # (they gain influence because j is now F)
            for nz2 in nzrange(A, j)
                k = cv[nz2]
                if k != j && is_strong[nz2] && cf[k] == 0
                    new_val = λ[k] + 1
                    _bucket_update!(k, new_val)
                    if new_val > top_bucket
                        top_bucket = new_val
                    end
                end
            end
        end
        # Decrement λ for nodes that best_i strongly depends on
        @inbounds for nz in nzrange(A, best_i)
            j = cv[nz]
            if j != best_i && is_strong[nz] && cf[j] == 0
                new_val = max(0, λ[j] - 1)
                _bucket_update!(j, new_val)
            end
        end
    end
    # Second pass: ensure every F-point has at least one strong C-neighbor
    _ensure_fine_have_coarse_neighbor!(cf, A, is_strong)
    # Build coarse map
    n_coarse = 0
    if setup_workspace !== nothing
        coarse_map = resize!(setup_workspace.coarse_map, n)
        fill!(coarse_map, 0)
    else
        coarse_map = zeros(Int, n)
    end
    @inbounds for i in 1:n
        if cf[i] == 1
            n_coarse += 1
            coarse_map[i] = n_coarse
        end
    end
    return cf, coarse_map, n_coarse
end

"""
    _build_strong_transpose_adj(A, is_strong) -> (offsets, sources)

Build the transpose adjacency list for strong connections. For each node j,
`sources[offsets[j]:(offsets[j+1]-1)]` gives the list of nodes i such that
row i has a strong connection to column j (i.e., i strongly depends on j).
"""
function _build_strong_transpose_adj(A::CSRMatrix{Tv, Ti}, is_strong::AbstractVector{Bool};
                                     setup_workspace=nothing) where {Tv, Ti}
    n = size(A, 1)
    cv = colvals(A)
    # Count: how many nodes strongly depend on each j
    if setup_workspace !== nothing
        counts = resize!(setup_workspace.counts, n)
        fill!(counts, 0)
    else
        counts = zeros(Int, n)
    end
    @inbounds for i in 1:n
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && is_strong[nz]
                counts[j] += 1
            end
        end
    end
    # Build offsets (CSR-style)
    if setup_workspace !== nothing
        offsets = resize!(setup_workspace.offsets, n + 1)
    else
        offsets = Vector{Int}(undef, n + 1)
    end
    offsets[1] = 1
    @inbounds for j in 1:n
        offsets[j + 1] = offsets[j] + counts[j]
    end
    total = offsets[n + 1] - 1
    if setup_workspace !== nothing
        sources = resize!(setup_workspace.sources, total)
    else
        sources = Vector{Int}(undef, total)
    end
    # Fill sources
    if setup_workspace !== nothing
        pos = resize!(setup_workspace.pos, n)
        copyto!(pos, 1, offsets, 1, n)
    else
        pos = copy(offsets[1:n])
    end
    @inbounds for i in 1:n
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && is_strong[nz]
                sources[pos[j]] = i
                pos[j] += 1
            end
        end
    end
    return offsets, sources
end

"""
    coarsen_aggressive(A, θ; rng=Random.default_rng())

Aggressive coarsening: applies PMIS twice. First pass produces an intermediate CF
splitting, then the coarse grid's strength-of-connection graph is used for a second
PMIS pass, merging the results. Returns `agg, n_coarse` like aggregation.
"""
function coarsen_aggressive(A_in::CSRMatrix{Tv, Ti}, θ::Real;
                            rng=Random.default_rng(),
                            backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    n = size(A_in, 1)
    # First pass: standard PMIS (will convert to CPU internally)
    cf, coarse_map, nc1 = coarsen_pmis(A_in, θ; rng=rng, backend=backend, block_size=block_size)
    # Compute strength on GPU if available, then convert to CPU for graph algorithms
    is_strong_raw = strength_graph(A_in, θ; backend=backend, block_size=block_size)
    is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    A = csr_to_cpu(A_in)
    cv = colvals(A)
    agg = zeros(Int, n)
    agg_count = 0
    @inbounds for i in 1:n
        if cf[i] == 1
            agg_count += 1
            agg[i] = agg_count
        end
    end
    nzv = nonzeros(A)
    @inbounds for i in 1:n
        cf[i] == 1 && continue
        best_agg = 0
        best_val = zero(real(Tv))
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && cf[j] == 1 && abs(nzv[nz]) > best_val
                best_val = abs(nzv[nz])
                best_agg = agg[j]
            end
        end
        if best_agg != 0
            agg[i] = best_agg
        end
    end
    # Phase 3: Merge coarse aggregates through F-points
    merge_map = collect(1:agg_count)
    @inbounds for i in 1:n
        cf[i] == -1 || continue
        coarse_aggs = Int[]
        for nz in nzrange(A, i)
            j = cv[nz]
            if is_strong[nz] && cf[j] == 1
                push!(coarse_aggs, find_root!(merge_map, agg[j]))
            end
        end
        for k in 2:length(coarse_aggs)
            union_roots!(merge_map, coarse_aggs[1], coarse_aggs[k])
        end
    end
    for i in 1:agg_count
        find_root!(merge_map, i)
    end
    new_id = zeros(Int, agg_count)
    n_coarse = 0
    for i in 1:agg_count
        root = find_root!(merge_map, i)
        if new_id[root] == 0
            n_coarse += 1
            new_id[root] = n_coarse
        end
        new_id[i] = new_id[root]
    end
    @inbounds for i in 1:n
        if agg[i] > 0
            agg[i] = new_id[find_root!(merge_map, agg[i])]
        end
    end
    @inbounds for i in 1:n
        if agg[i] == 0
            best_agg = 0
            best_val = zero(real(Tv))
            for nz in nzrange(A, i)
                j = cv[nz]
                if j != i && agg[j] != 0 && abs(nzv[nz]) > best_val
                    best_val = abs(nzv[nz])
                    best_agg = agg[j]
                end
            end
            if best_agg != 0
                agg[i] = best_agg
            else
                n_coarse += 1
                agg[i] = n_coarse
            end
        end
    end
    return agg, n_coarse
end

# ── Helper: union-find ───────────────────────────────────────────────────────

function find_root!(parent::Vector{Int}, i::Int)
    while parent[i] != i
        parent[i] = parent[parent[i]]  # path compression
        i = parent[i]
    end
    return i
end

function union_roots!(parent::Vector{Int}, a::Int, b::Int)
    ra = find_root!(parent, a)
    rb = find_root!(parent, b)
    if ra != rb
        parent[rb] = ra
    end
end

# ── Aggressive CF-splitting coarsening (HYPRE-style) ─────────────────────────

"""
    coarsen_aggressive_cf(A, θ, base; config)

Two-pass CF-splitting aggressive coarsening (HYPRE-style). Performs a first pass
of CF splitting using the base algorithm (HMIS or PMIS), then a second pass
among the C-points using distance-2 strong connections to further coarsen.

This matches HYPRE's behavior when AggNumLevels > 0 with CoarsenType=10 (HMIS).
Returns `(cf, coarse_map, n_coarse)` where the CF-splitting assigns more points
as fine than standard HMIS/PMIS alone.
"""
function coarsen_aggressive_cf(A_in::CSRMatrix{Tv, Ti}, θ::Real, base::Symbol;
                               config::AMGConfig=AMGConfig(),
                               rng=Random.default_rng(),
                               backend=DEFAULT_BACKEND, block_size::Int=64,
                               setup_workspace=nothing) where {Tv, Ti}
    n = size(A_in, 1)
    if n <= 1
        return ones(Int, n), collect(1:n), n
    end
    # First pass: standard CF-splitting using base algorithm
    A_eff = config.max_row_sum < 1.0 ? _apply_max_row_sum(csr_to_cpu(A_in), config.max_row_sum) : A_in
    if base == :hmis
        cf1, _, nc1 = coarsen_hmis(A_eff, θ; rng=rng, backend=backend, block_size=block_size, setup_workspace=setup_workspace)
    else  # :pmis
        cf1, _, nc1 = coarsen_pmis(A_eff, θ; rng=rng, backend=backend, block_size=block_size, setup_workspace=setup_workspace)
    end
    # Second pass: among C-points from first pass, do another CF-splitting
    # using distance-2 strong connections to further reduce the coarse set.
    # Compute strength on GPU if available, then convert to CPU for graph algorithms
    is_strong_raw = strength_graph(A_eff, θ; backend=backend, block_size=block_size)
    is_strong = is_strong_raw isa Array ? is_strong_raw : Array(is_strong_raw)
    A_eff = csr_to_cpu(A_eff)
    cv = colvals(A_eff)

    # Build distance-2 strong connection graph among C-points:
    # Two C-points are distance-2 connected if they share a strong connection
    # through any point (F or C).
    c_indices = Int[]
    c_local = zeros(Int, n)  # map from original index to C-point index
    @inbounds for i in 1:n
        if cf1[i] == 1
            push!(c_indices, i)
            c_local[i] = length(c_indices)
        end
    end
    nc1_actual = length(c_indices)

    # For each C-point, find other C-points at distance 2 (through strong graph)
    # Build a neighbor set for each C-point in the C-subgraph
    c_neighbors = [Set{Int}() for _ in 1:nc1_actual]
    @inbounds for i in 1:n
        # Collect strong C-neighbors of i
        c_nbrs_of_i = Int[]
        for nz in nzrange(A_eff, i)
            j = cv[nz]
            if j != i && is_strong[nz] && cf1[j] == 1
                push!(c_nbrs_of_i, c_local[j])
            end
        end
        # All pairs of C-neighbors of i are distance-2 connected in C-graph
        for a in c_nbrs_of_i
            for b in c_nbrs_of_i
                if a != b
                    push!(c_neighbors[a], b)
                end
            end
            # Also add direct C-C connections
            if cf1[i] == 1
                ci = c_local[i]
                push!(c_neighbors[ci], a)
                push!(c_neighbors[a], ci)
            end
        end
    end

    # Now do PMIS-style splitting on the C-subgraph
    c_measure = zeros(Float64, nc1_actual)
    @inbounds for ci in 1:nc1_actual
        c_measure[ci] = Float64(length(c_neighbors[ci])) + rand(rng)
    end

    cf2 = zeros(Int, nc1_actual)  # CF-splitting of C-points: 1=coarse, -1=fine
    max_iter = nc1_actual + 1
    for iter in 1:max_iter
        all_decided = true
        @inbounds for ci in 1:nc1_actual
            cf2[ci] != 0 && continue
            all_decided = false
            is_max = true
            for cj in c_neighbors[ci]
                if cf2[cj] != -1 && c_measure[cj] > c_measure[ci]
                    is_max = false
                    break
                end
            end
            if is_max
                cf2[ci] = 1
            end
        end
        @inbounds for ci in 1:nc1_actual
            cf2[ci] != 0 && continue
            for cj in c_neighbors[ci]
                if cf2[cj] == 1
                    cf2[ci] = -1
                    break
                end
            end
        end
        all_decided && break
    end
    @inbounds for ci in 1:nc1_actual
        if cf2[ci] == 0
            cf2[ci] = 1
        end
    end

    # Combine: original F-points stay F; C-points that became F in second pass stay F
    cf_final = copy(cf1)
    @inbounds for ci in 1:nc1_actual
        if cf2[ci] == -1
            cf_final[c_indices[ci]] = -1  # demote to fine
        end
    end

    # Ensure every F-point has at least one strong C-neighbor
    _ensure_fine_have_coarse_neighbor!(cf_final, A_eff, is_strong)

    # Build coarse map
    n_coarse = 0
    coarse_map = zeros(Int, n)
    @inbounds for i in 1:n
        if cf_final[i] == 1
            n_coarse += 1
            coarse_map[i] = n_coarse
        end
    end
    return cf_final, coarse_map, n_coarse
end

# ── Dispatch coarsening by type ──────────────────────────────────────────────

function coarsen(A::CSRMatrix, alg::AggregationCoarsening,
                config::AMGConfig=AMGConfig();
                backend=DEFAULT_BACKEND, block_size::Int=64,
                setup_workspace=nothing)
    if config.max_row_sum < 1.0
        A_weak = _apply_max_row_sum(csr_to_cpu(A), config.max_row_sum)
        return coarsen_aggregation(A_weak, alg.θ; backend=backend, block_size=block_size)
    end
    return coarsen_aggregation(A, alg.θ; backend=backend, block_size=block_size)
end

function coarsen(A::CSRMatrix, alg::AggressiveCoarsening,
                config::AMGConfig=AMGConfig();
                backend=DEFAULT_BACKEND, block_size::Int=64,
                setup_workspace=nothing)
    if config.max_row_sum < 1.0
        A_weak = _apply_max_row_sum(csr_to_cpu(A), config.max_row_sum)
        return coarsen_aggressive(A_weak, alg.θ; backend=backend, block_size=block_size)
    end
    return coarsen_aggressive(A, alg.θ; backend=backend, block_size=block_size)
end

"""
    uses_cf_splitting(alg)

Return true if the coarsening algorithm produces a CF-splitting (rather than aggregation).
"""
uses_cf_splitting(::AggregationCoarsening) = false
uses_cf_splitting(::AggressiveCoarsening) = false
uses_cf_splitting(::SmoothedAggregationCoarsening) = false
uses_cf_splitting(::PMISCoarsening) = true
uses_cf_splitting(::HMISCoarsening) = true
uses_cf_splitting(::RSCoarsening) = true

"""
    coarsen_cf(A, alg, config)

Perform CF-splitting coarsening. Returns `(cf, coarse_map, n_coarse)`.
"""
function coarsen_cf(A::CSRMatrix, alg::PMISCoarsening,
                    config::AMGConfig=AMGConfig();
                    backend=DEFAULT_BACKEND, block_size::Int=64,
                    setup_workspace=nothing)
    if config.max_row_sum < 1.0
        A_weak = _apply_max_row_sum(csr_to_cpu(A), config.max_row_sum)
        return coarsen_pmis(A_weak, alg.θ; backend=backend, block_size=block_size, setup_workspace=setup_workspace)
    end
    return coarsen_pmis(A, alg.θ; backend=backend, block_size=block_size, setup_workspace=setup_workspace)
end

function coarsen_cf(A::CSRMatrix, alg::HMISCoarsening,
                    config::AMGConfig=AMGConfig();
                    backend=DEFAULT_BACKEND, block_size::Int=64,
                    setup_workspace=nothing)
    if config.max_row_sum < 1.0
        A_weak = _apply_max_row_sum(csr_to_cpu(A), config.max_row_sum)
        return coarsen_hmis(A_weak, alg.θ; backend=backend, block_size=block_size, setup_workspace=setup_workspace)
    end
    return coarsen_hmis(A, alg.θ; backend=backend, block_size=block_size, setup_workspace=setup_workspace)
end

function coarsen_cf(A::CSRMatrix, alg::RSCoarsening,
                    config::AMGConfig=AMGConfig();
                    backend=DEFAULT_BACKEND, block_size::Int=64,
                    setup_workspace=nothing)
    if config.max_row_sum < 1.0
        A_weak = _apply_max_row_sum(csr_to_cpu(A), config.max_row_sum)
        return coarsen_rs(A_weak, alg.θ; backend=backend, block_size=block_size, setup_workspace=setup_workspace)
    end
    return coarsen_rs(A, alg.θ; backend=backend, block_size=block_size, setup_workspace=setup_workspace)
end

"""
Convert a CF splitting to an aggregation vector (legacy fallback).
"""
function _cf_to_aggregation(A::CSRMatrix{Tv, Ti}, cf, coarse_map, n_coarse) where {Tv, Ti}
    n = size(A, 1)
    agg = zeros(Int, n)
    cv = colvals(A)
    nzv = nonzeros(A)
    @inbounds for i in 1:n
        if cf[i] == 1
            agg[i] = coarse_map[i]
        end
    end
    @inbounds for i in 1:n
        cf[i] == 1 && continue
        best_agg = 0
        best_val = zero(real(Tv))
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && cf[j] == 1 && abs(nzv[nz]) > best_val
                best_val = abs(nzv[nz])
                best_agg = coarse_map[j]
            end
        end
        if best_agg != 0
            agg[i] = best_agg
        else
            n_coarse += 1
            agg[i] = n_coarse
        end
    end
    return agg, n_coarse
end
