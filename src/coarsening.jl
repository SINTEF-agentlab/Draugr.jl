# ── Aggregation coarsening ────────────────────────────────────────────────────

"""
    coarsen_aggregation(A, θ)

Greedy aggregation coarsening. Returns `agg::Vector{Int}` where `agg[i]` is the
aggregate index (1-based) that node `i` belongs to, and `n_coarse` the number of
aggregates.
"""
function coarsen_aggregation(A::CSRMatrix{Tv, Ti}, θ::Real) where {Tv, Ti}
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
            # Isolated node: create singleton aggregate
            n_coarse += 1
            agg[i] = n_coarse
        end
    end
    return agg, n_coarse
end

# ── PMIS coarsening ──────────────────────────────────────────────────────────

"""
    _compute_strong_transpose_count(A, is_strong)

Compute for each node i the number of nodes j that strongly depend on i,
i.e., the number of strong connections in the TRANSPOSE graph. This is the
column-based measure used by hypre for better PMIS coarsening.
"""
function _compute_strong_transpose_count(A::CSRMatrix{Tv, Ti}, is_strong::Vector{Bool}) where {Tv, Ti}
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
function _ensure_fine_have_coarse_neighbor!(cf::Vector{Int}, A::CSRMatrix, is_strong::Vector{Bool})
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
function _symmetrize_strength(A::CSRMatrix{Tv, Ti}, is_strong::Vector{Bool}) where {Tv, Ti}
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
