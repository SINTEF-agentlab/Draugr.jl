# ── Aggregation coarsening ────────────────────────────────────────────────────

"""
    coarsen_aggregation(A, θ)

Greedy aggregation coarsening. Returns `agg::Vector{Int}` where `agg[i]` is the
aggregate index (1-based) that node `i` belongs to, and `n_coarse` the number of
aggregates.
"""
function coarsen_aggregation(A::StaticSparsityMatrixCSR{Tv, Ti}, θ::Real) where {Tv, Ti}
    n = size(A, 1)
    # Edge case: trivial system
    if n <= 1
        return ones(Int, n), n
    end
    is_strong = strength_graph(A, θ)
    cv = colvals(A)
    agg = zeros(Int, n)  # 0 = unassigned
    n_coarse = 0
    # Phase 1: form aggregates around seed nodes
    @inbounds for i in 1:n
        agg[i] != 0 && continue
        # i becomes a seed
        n_coarse += 1
        agg[i] = n_coarse
        for nz in nzrange(A, i)
            j = cv[nz]
            if is_strong[nz] && agg[j] == 0
                agg[j] = n_coarse
            end
        end
    end
    # Phase 2: assign any remaining unaggregated nodes to strongest neighbor's aggregate
    @inbounds for i in 1:n
        agg[i] != 0 && continue
        nzv = nonzeros(A)
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
    return agg, n_coarse
end

# ── PMIS coarsening ──────────────────────────────────────────────────────────

"""
    coarsen_pmis(A, θ; rng=Random.default_rng())

Parallel Modified Independent Set (PMIS) coarsening. Returns `cf::Vector{Int}`
where `cf[i] = 1` for coarse points and `cf[i] = 0` for fine points,
`coarse_map::Vector{Int}` mapping coarse-point indices to 1:n_coarse, and `n_coarse`.
"""
function coarsen_pmis(A::StaticSparsityMatrixCSR{Tv, Ti}, θ::Real;
                      rng=Random.default_rng()) where {Tv, Ti}
    n = size(A, 1)
    # Edge case: trivial system
    if n <= 1
        return ones(Int, n), collect(1:n), n
    end
    is_strong = strength_graph(A, θ)
    cv = colvals(A)
    # Compute measure: number of strong connections + random perturbation
    measure = zeros(Float64, n)
    @inbounds for i in 1:n
        count = 0
        for nz in nzrange(A, i)
            if is_strong[nz]
                count += 1
            end
        end
        measure[i] = Float64(count) + rand(rng)
    end
    # CF splitting: iterative PMIS
    # 0 = undecided, 1 = coarse, -1 = fine
    cf = zeros(Int, n)
    max_iter = n + 1
    for iter in 1:max_iter
        all_decided = true
        @inbounds for i in 1:n
            cf[i] != 0 && continue
            all_decided = false
            # Check if i has the largest measure among undecided strong neighbors
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
        # Mark fine points: any undecided node with a strong coarse neighbor
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
    # Any remaining undecided nodes become coarse
    @inbounds for i in 1:n
        if cf[i] == 0
            cf[i] = 1
        end
    end
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

# ── HMIS coarsening ──────────────────────────────────────────────────────────

"""
    coarsen_hmis(A, θ; rng=Random.default_rng())

Hybrid Modified Independent Set (HMIS) coarsening. Like PMIS but uses the
symmetrized strength graph (intersection of S and S^T) for the independent set
selection, producing generally less aggressive coarsening. Returns same format
as `coarsen_pmis`.
"""
function coarsen_hmis(A::StaticSparsityMatrixCSR{Tv, Ti}, θ::Real;
                      rng=Random.default_rng()) where {Tv, Ti}
    n = size(A, 1)
    # Edge case: trivial system
    if n <= 1
        return ones(Int, n), collect(1:n), n
    end
    is_strong = strength_graph(A, θ)
    cv = colvals(A)
    # Build symmetric strength graph: strong in both directions
    # is_strong_sym[nz] = true iff is_strong[nz] AND is_strong[reverse nz]
    is_strong_sym = _symmetrize_strength(A, is_strong)
    # Compute measure: number of symmetric strong connections + random perturbation
    measure = zeros(Float64, n)
    @inbounds for i in 1:n
        count = 0
        for nz in nzrange(A, i)
            if is_strong_sym[nz]
                count += 1
            end
        end
        measure[i] = Float64(count) + rand(rng)
    end
    # CF splitting using symmetrized strength for IS selection
    cf = zeros(Int, n)
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
    # Remaining undecided → coarse
    @inbounds for i in 1:n
        if cf[i] == 0
            cf[i] = 1
        end
    end
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
    _symmetrize_strength(A, is_strong)

Build a symmetrized strength indicator: is_strong_sym[nz] = true iff the
connection (i,j) is strong AND (j,i) is also strong in the CSR structure.
"""
function _symmetrize_strength(A::StaticSparsityMatrixCSR{Tv, Ti}, is_strong::Vector{Bool}) where {Tv, Ti}
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
            # Check if (j, i) is also strong
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

"""
    coarsen_aggressive(A, θ; rng=Random.default_rng())

Aggressive coarsening: applies PMIS twice. First pass produces an intermediate CF
splitting, then the coarse grid's strength-of-connection graph is used for a second
PMIS pass, merging the results. Returns `agg, n_coarse` like aggregation.
"""
function coarsen_aggressive(A::StaticSparsityMatrixCSR{Tv, Ti}, θ::Real;
                            rng=Random.default_rng()) where {Tv, Ti}
    n = size(A, 1)
    # First pass: standard PMIS
    cf, coarse_map, nc1 = coarsen_pmis(A, θ; rng=rng)
    # For aggressive coarsening, treat the first-pass C-points as nodes in a graph
    # and aggregate them using distance-2 strong connections through F-points
    is_strong = strength_graph(A, θ)
    cv = colvals(A)
    # Build aggregation: each fine point is assigned to the aggregate of its strongest
    # coarse neighbor. Coarse points form initial singleton aggregates, then merge
    # through F-point connections.
    agg = zeros(Int, n)
    # Phase 1: Each coarse point gets a unique aggregate
    agg_count = 0
    @inbounds for i in 1:n
        if cf[i] == 1
            agg_count += 1
            agg[i] = agg_count
        end
    end
    # Phase 2: Each fine point is assigned to its strongest coarse neighbor
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
    # Phase 3: Merge coarse aggregates that are connected through F-points
    # Build graph of coarse-coarse connections through fine points
    merge_map = collect(1:agg_count)  # union-find roots
    @inbounds for i in 1:n
        cf[i] == -1 || continue
        # Collect coarse neighbors
        coarse_aggs = Int[]
        for nz in nzrange(A, i)
            j = cv[nz]
            if is_strong[nz] && cf[j] == 1
                push!(coarse_aggs, find_root!(merge_map, agg[j]))
            end
        end
        # Merge pairs
        for k in 2:length(coarse_aggs)
            union_roots!(merge_map, coarse_aggs[1], coarse_aggs[k])
        end
    end
    # Flatten merge_map and renumber
    for i in 1:agg_count
        find_root!(merge_map, i)
    end
    # Renumber aggregates
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
    # Update aggregation
    @inbounds for i in 1:n
        if agg[i] > 0
            agg[i] = new_id[find_root!(merge_map, agg[i])]
        end
    end
    # Assign any remaining unassigned nodes
    @inbounds for i in 1:n
        if agg[i] == 0
            # Assign to nearest aggregate
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

# ── Dispatch coarsening by type ──────────────────────────────────────────────

function coarsen(A::StaticSparsityMatrixCSR, alg::AggregationCoarsening,
                config::AMGConfig=AMGConfig())
    if config.max_row_sum > 0
        A_weak = _apply_max_row_sum(A, config.max_row_sum)
        return coarsen_aggregation(A_weak, alg.θ)
    end
    return coarsen_aggregation(A, alg.θ)
end

function coarsen(A::StaticSparsityMatrixCSR, alg::AggressiveCoarsening,
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

"""
    coarsen_cf(A, alg, config)

Perform CF-splitting coarsening. Returns `(cf, coarse_map, n_coarse)`.
When max_row_sum is configured, strength is computed on a weakened matrix.
"""
function coarsen_cf(A::StaticSparsityMatrixCSR, alg::PMISCoarsening,
                    config::AMGConfig=AMGConfig())
    if config.max_row_sum > 0
        A_weak = _apply_max_row_sum(A, config.max_row_sum)
        return coarsen_pmis(A_weak, alg.θ)
    end
    return coarsen_pmis(A, alg.θ)
end

function coarsen_cf(A::StaticSparsityMatrixCSR, alg::HMISCoarsening,
                    config::AMGConfig=AMGConfig())
    if config.max_row_sum > 0
        A_weak = _apply_max_row_sum(A, config.max_row_sum)
        return coarsen_hmis(A_weak, alg.θ)
    end
    return coarsen_hmis(A, alg.θ)
end

"""
Convert a CF splitting to an aggregation vector (legacy fallback).
"""
function _cf_to_aggregation(A::StaticSparsityMatrixCSR{Tv, Ti}, cf, coarse_map, n_coarse) where {Tv, Ti}
    n = size(A, 1)
    agg = zeros(Int, n)
    cv = colvals(A)
    nzv = nonzeros(A)
    # Coarse points map directly
    @inbounds for i in 1:n
        if cf[i] == 1
            agg[i] = coarse_map[i]
        end
    end
    # Fine points map to strongest coarse neighbor
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
            # Isolated node: create singleton
            n_coarse += 1
            agg[i] = n_coarse
        end
    end
    return agg, n_coarse
end
