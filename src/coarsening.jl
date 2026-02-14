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
function coarsen_aggregation(A::CSRMatrix{Tv, Ti}, θ::Real) where {Tv, Ti}
    n = size(A, 1)
    # Edge case: trivial system
    if n <= 1
        return ones(Int, n), n
    end
    is_strong = strength_graph(A, θ)
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
function _compute_strong_transpose_count(A::CSRMatrix{Tv, Ti}, is_strong::AbstractVector{Bool}) where {Tv, Ti}
    n = size(A, 1)
    cv = colvals(A)
    st_count = zeros(Int, n)
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
function coarsen_pmis(A::CSRMatrix{Tv, Ti}, θ::Real;
                      rng=Random.default_rng()) where {Tv, Ti}
    n = size(A, 1)
    if n <= 1
        return ones(Int, n), collect(1:n), n
    end
    is_strong = strength_graph(A, θ)
    cv = colvals(A)
    # Use column-based measure: how many nodes strongly depend on i
    st_count = _compute_strong_transpose_count(A, is_strong)
    measure = zeros(Float64, n)
    @inbounds for i in 1:n
        measure[i] = Float64(st_count[i]) + rand(rng)
    end
    # Mark isolated nodes (no strong connections at all) immediately as coarse
    cf = zeros(Int, n)
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
    # Second pass: ensure every F-point has at least one strong C-neighbor
    _ensure_fine_have_coarse_neighbor!(cf, A, is_strong)
    # Build coarse map
    n_coarse = 0
    coarse_map = zeros(Int, n)
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

Hybrid Modified Independent Set (HMIS) coarsening. Like PMIS but uses the
symmetrized strength graph (intersection of S and S^T) for the independent set
selection. Uses column-based measure. Includes second pass for strong-connection
property.
"""
function coarsen_hmis(A::CSRMatrix{Tv, Ti}, θ::Real;
                      rng=Random.default_rng()) where {Tv, Ti}
    n = size(A, 1)
    if n <= 1
        return ones(Int, n), collect(1:n), n
    end
    is_strong = strength_graph(A, θ)
    cv = colvals(A)
    is_strong_sym = _symmetrize_strength(A, is_strong)
    # Column-based measure on the symmetric graph
    st_count = zeros(Int, n)
    @inbounds for i in 1:n
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && is_strong_sym[nz]
                st_count[j] += 1
            end
        end
    end
    measure = zeros(Float64, n)
    @inbounds for i in 1:n
        measure[i] = Float64(st_count[i]) + rand(rng)
    end
    # Mark isolated nodes
    cf = zeros(Int, n)
    @inbounds for i in 1:n
        has_strong = false
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && is_strong_sym[nz]
                has_strong = true
                break
            end
        end
        if !has_strong
            # Check if also isolated in unsymmetrized graph
            has_any_strong = false
            for nz in nzrange(A, i)
                j = cv[nz]
                if j != i && is_strong[nz]
                    has_any_strong = true
                    break
                end
            end
            if !has_any_strong && st_count[i] == 0
                cf[i] = 1  # isolated → coarse
            end
        end
    end
    # CF splitting using symmetrized strength for IS selection
    max_iter = n + 1
    for iter in 1:max_iter
        all_decided = true
        @inbounds for i in 1:n
            cf[i] != 0 && continue
            all_decided = false
            is_max = true
            for nz in nzrange(A, i)
                j = cv[nz]
                if is_strong_sym[nz] && j != i && cf[j] != -1
                    if measure[j] > measure[i]
                        is_max = false
                        break
                    end
                end
            end
            if is_max
                cf[i] = 1
            end
        end
        @inbounds for i in 1:n
            cf[i] != 0 && continue
            for nz in nzrange(A, i)
                j = cv[nz]
                if is_strong_sym[nz] && cf[j] == 1
                    cf[i] = -1
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
    # Second pass using the full (non-symmetric) strength graph
    _ensure_fine_have_coarse_neighbor!(cf, A, is_strong)
    n_coarse = 0
    coarse_map = zeros(Int, n)
    @inbounds for i in 1:n
        if cf[i] == 1
            n_coarse += 1
            coarse_map[i] = n_coarse
        end
    end
    return cf, coarse_map, n_coarse
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
"""
function coarsen_rs(A::CSRMatrix{Tv, Ti}, θ::Real;
                    rng=Random.default_rng()) where {Tv, Ti}
    n = size(A, 1)
    if n <= 1
        return ones(Int, n), collect(1:n), n
    end
    is_strong = strength_graph(A, θ)
    cv = colvals(A)
    # Compute λ_i = number of points that strongly depend on i (transpose measure)
    λ = _compute_strong_transpose_count(A, is_strong)
    # CF splitting
    cf = zeros(Int, n)  # 0 = undecided, 1 = C, -1 = F
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
    # First pass: greedy selection
    # Pick the undecided node with largest λ, make it C, make its strong
    # dependents F, and update λ for affected nodes.
    while true
        # Find undecided node with maximum λ
        best_i = 0
        best_λ = -1
        @inbounds for i in 1:n
            if cf[i] == 0 && λ[i] > best_λ
                best_λ = λ[i]
                best_i = i
            end
        end
        best_i == 0 && break  # all decided
        # Make best_i coarse
        cf[best_i] = 1
        # For each undecided node j that strongly depends on best_i, make j fine
        @inbounds for j in 1:n
            cf[j] != 0 && continue
            # Check if j→best_i is a strong connection
            for nz in nzrange(A, j)
                if cv[nz] == best_i && is_strong[nz]
                    cf[j] = -1  # j becomes fine
                    # Increment λ for undecided nodes that j depends on
                    # (they gain influence because j is now F)
                    for nz2 in nzrange(A, j)
                        k = cv[nz2]
                        if k != j && is_strong[nz2] && cf[k] == 0
                            λ[k] += 1
                        end
                    end
                    break
                end
            end
        end
        # Decrement λ for nodes that best_i depends on (they lose influence)
        @inbounds for nz in nzrange(A, best_i)
            j = cv[nz]
            if j != best_i && is_strong[nz] && cf[j] == 0
                λ[j] = max(0, λ[j] - 1)
            end
        end
    end
    # Second pass: ensure every F-point has at least one strong C-neighbor
    _ensure_fine_have_coarse_neighbor!(cf, A, is_strong)
    # Build coarse map
    n_coarse = 0
    coarse_map = zeros(Int, n)
    @inbounds for i in 1:n
        if cf[i] == 1
            n_coarse += 1
            coarse_map[i] = n_coarse
        end
    end
    return cf, coarse_map, n_coarse
end

"""
    coarsen_aggressive(A, θ; rng=Random.default_rng())

Aggressive coarsening: applies PMIS twice. First pass produces an intermediate CF
splitting, then the coarse grid's strength-of-connection graph is used for a second
PMIS pass, merging the results. Returns `agg, n_coarse` like aggregation.
"""
function coarsen_aggressive(A::CSRMatrix{Tv, Ti}, θ::Real;
                            rng=Random.default_rng()) where {Tv, Ti}
    n = size(A, 1)
    # First pass: standard PMIS
    cf, coarse_map, nc1 = coarsen_pmis(A, θ; rng=rng)
    is_strong = strength_graph(A, θ)
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
function coarsen_aggressive_cf(A::CSRMatrix{Tv, Ti}, θ::Real, base::Symbol;
                               config::AMGConfig=AMGConfig(),
                               rng=Random.default_rng()) where {Tv, Ti}
    n = size(A, 1)
    if n <= 1
        return ones(Int, n), collect(1:n), n
    end
    # First pass: standard CF-splitting using base algorithm
    A_eff = config.max_row_sum > 0 ? _apply_max_row_sum(A, config.max_row_sum) : A
    if base == :hmis
        cf1, _, nc1 = coarsen_hmis(A_eff, θ; rng=rng)
    else  # :pmis
        cf1, _, nc1 = coarsen_pmis(A_eff, θ; rng=rng)
    end
    # Second pass: among C-points from first pass, do another CF-splitting
    # using distance-2 strong connections to further reduce the coarse set.
    is_strong = strength_graph(A_eff, θ)
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
                config::AMGConfig=AMGConfig())
    if config.max_row_sum > 0
        A_weak = _apply_max_row_sum(A, config.max_row_sum)
        return coarsen_aggregation(A_weak, alg.θ)
    end
    return coarsen_aggregation(A, alg.θ)
end

function coarsen(A::CSRMatrix, alg::AggressiveCoarsening,
                config::AMGConfig=AMGConfig())
    if config.max_row_sum > 0
        A_weak = _apply_max_row_sum(A, config.max_row_sum)
        return coarsen_aggressive(A_weak, alg.θ)
    end
    return coarsen_aggressive(A, alg.θ)
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
                    config::AMGConfig=AMGConfig())
    if config.max_row_sum > 0
        A_weak = _apply_max_row_sum(A, config.max_row_sum)
        return coarsen_pmis(A_weak, alg.θ)
    end
    return coarsen_pmis(A, alg.θ)
end

function coarsen_cf(A::CSRMatrix, alg::HMISCoarsening,
                    config::AMGConfig=AMGConfig())
    if config.max_row_sum > 0
        A_weak = _apply_max_row_sum(A, config.max_row_sum)
        return coarsen_hmis(A_weak, alg.θ)
    end
    return coarsen_hmis(A, alg.θ)
end

function coarsen_cf(A::CSRMatrix, alg::RSCoarsening,
                    config::AMGConfig=AMGConfig())
    if config.max_row_sum > 0
        A_weak = _apply_max_row_sum(A, config.max_row_sum)
        return coarsen_rs(A_weak, alg.θ)
    end
    return coarsen_rs(A, alg.θ)
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
