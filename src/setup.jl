# ── Device conversion helpers for AMG structures ─────────────────────────────

"""
    _prolongation_to_device(ref, P) -> ProlongationOp

Copy a ProlongationOp to the same device as `ref`'s arrays.
"""
function _prolongation_to_device(ref::CSRMatrix, P::ProlongationOp)
    rp = _to_device(ref, P.rowptr)
    cv = _to_device(ref, P.colval)
    nzv = _to_device(ref, P.nzval)
    ts = P.trunc_scaling === nothing ? nothing : _to_device(ref, P.trunc_scaling)
    return ProlongationOp(rp, cv, nzv, P.nrow, P.ncol, ts)
end

"""
    _transpose_map_to_device(ref, Pt_map) -> TransposeMap

Copy a TransposeMap to the same device as `ref`'s arrays.
"""
function _transpose_map_to_device(ref::CSRMatrix, Pt_map::TransposeMap)
    offsets = _to_device(ref, Pt_map.offsets)
    fine_rows = _to_device(ref, Pt_map.fine_rows)
    p_nz_idx = _to_device(ref, Pt_map.p_nz_idx)
    return TransposeMap(offsets, fine_rows, p_nz_idx)
end

"""
    _restriction_map_to_device(ref, r_map) -> RestrictionMap

Copy a RestrictionMap to the same device as `ref`'s arrays.
"""
function _restriction_map_to_device(ref::CSRMatrix, r_map::RestrictionMap)
    nz_offsets = _to_device(ref, r_map.nz_offsets)
    triple_pi_idx = _to_device(ref, r_map.triple_pi_idx)
    triple_anz_idx = _to_device(ref, r_map.triple_anz_idx)
    triple_pj_idx = _to_device(ref, r_map.triple_pj_idx)
    return RestrictionMap(nz_offsets, triple_pi_idx, triple_anz_idx, triple_pj_idx)
end

"""
    _prolongation_update_map_to_device(ref, p_map) -> ProlongationUpdateMap

Copy a ProlongationUpdateMap to the same device as `ref`'s arrays.
"""
function _prolongation_update_map_to_device(ref::CSRMatrix, p_map::ProlongationUpdateMap)
    numer_idx = _to_device(ref, p_map.numer_idx)
    denom_offsets = _to_device(ref, p_map.denom_offsets)
    denom_entries = _to_device(ref, p_map.denom_entries)
    return ProlongationUpdateMap(numer_idx, denom_offsets, denom_entries)
end

"""
    amg_setup(A::SparseMatrixCSC, config; backend, block_size) -> AMGHierarchy

External API entry point: convert `SparseMatrixCSC` to `CSRMatrix` once
and forward to the general CSRMatrix-based setup.
"""
function amg_setup(A::SparseMatrixCSC{Tv, Ti}, config::AMGConfig=AMGConfig();
                   backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    return amg_setup(csr_from_csc(A), config; backend=backend, block_size=block_size)
end

"""
    amg_setup(A::CSRMatrix, config) -> AMGHierarchy

Perform the full AMG setup (analysis phase). This determines the coarsening at each
level, constructs prolongation operators, computes Galerkin products, and sets up
smoothers.

When `config.allow_partial_resetup` is `true` (the default), the sparsity structure
and restriction maps computed here are reused by `amg_resetup!` with `partial=true`
when matrix coefficients change but the pattern remains the same. When `false`,
these additional mappings are skipped for a faster setup; only `amg_resetup!` with
`partial=false` can be used afterwards.
"""
function amg_setup(A_csr::CSRMatrix{Tv, Ti}, config::AMGConfig=AMGConfig();
                   backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    t_setup = time()
    levels = AMGLevel{Tv, Ti}[]
    n_finest = size(A_csr, 1)
    galerkin_ws = GalerkinWorkspace{Tv, Ti}()
    setup_ws = SetupWorkspace{Tv, Ti}()
    A_coarsest = _build_levels!(levels, A_csr, config;
                                backend=backend, block_size=block_size,
                                galerkin_workspace=galerkin_ws,
                                setup_workspace=setup_ws)
    coarse_dense, coarse_factor, coarse_x, coarse_b, solve_r =
        _build_coarse_solver(A_coarsest, A_csr, n_finest, config;
                             backend=backend, block_size=block_size)
    hierarchy = AMGHierarchy{Tv, Ti}(levels, coarse_dense,
                                      coarse_factor, coarse_x, coarse_b, solve_r,
                                      backend, block_size, config.coarse_solve_on_cpu,
                                      galerkin_ws, setup_ws)
    t_setup = time() - t_setup
    if config.verbose >= 1
        _print_hierarchy_info(hierarchy, config, n_finest, t_setup)
    end
    return hierarchy
end

# ── Shared helpers for setup and full resetup ────────────────────────────────

"""
    _reuse_or_allocate_vector(old_level, field, A_ref, Tv, n)

Try to reuse a workspace vector from an old AMG level if its size matches.
Otherwise allocate a new vector on the same device as `A_ref`.
"""
function _reuse_or_allocate_vector(old_level, field::Symbol,
                                   A_ref::CSRMatrix, ::Type{Tv}, n::Int) where {Tv}
    if old_level !== nothing
        old_vec = getfield(old_level, field)
        if length(old_vec) == n
            return old_vec
        end
    end
    if A_ref.nzval isa Array
        return Vector{Tv}(undef, n)
    else
        return _allocate_vector(A_ref, Tv, n)
    end
end

"""
    _build_levels!(levels, A_input, config; backend, block_size, galerkin_workspace, setup_workspace, device_ref)

Build AMG levels by coarsening `A_input`. Populates `levels` in-place, reusing
workspace vectors from any pre-existing levels when sizes match. Returns the
coarsest-level CSR matrix (to be used for the direct solver).

When `device_ref` is provided (a GPU-backed CSRMatrix), it is used as the
reference for device allocation. When `nothing` (the default), `A_input` is
used as the reference.

The `galerkin_workspace` and `setup_workspace` are reused across all levels
and resetup calls.
"""
function _build_levels!(levels::Vector{AMGLevel{Tv, Ti}},
                        A_input::CSRMatrix{Tv, Ti},
                        config::AMGConfig;
                        backend=DEFAULT_BACKEND, block_size::Int=64,
                        galerkin_workspace::GalerkinWorkspace{Tv, Ti}=GalerkinWorkspace{Tv, Ti}(),
                        setup_workspace::SetupWorkspace{Tv, Ti}=SetupWorkspace{Tv, Ti}(),
                        device_ref::Union{Nothing, CSRMatrix}=nothing) where {Tv, Ti}
    A_ref = device_ref === nothing ? A_input : device_ref
    is_gpu = !(A_ref.nzval isa Array)
    old_levels = copy(levels)
    empty!(levels)
    A_current = A_input
    allow_partial_resetup = config.allow_partial_resetup
    for lvl in 1:(config.max_levels - 1)
        n = size(A_current, 1)
        n <= config.max_coarse_size && break
        coarsening_alg = _get_coarsening_for_level(config, lvl)
        # Set old_P in workspace for array reuse during prolongation building (CPU only)
        old_lvl = lvl <= length(old_levels) ? old_levels[lvl] : nothing
        if setup_workspace !== nothing
            if old_lvl !== nothing && !is_gpu && old_lvl.P.colval isa Vector
                setup_workspace.old_P = old_lvl.P
            else
                setup_workspace.old_P = nothing
            end
        end
        # Build P_update_map only for CF-splitting methods when allow_partial_resetup is enabled
        build_P_update_map = allow_partial_resetup && uses_cf_splitting(coarsening_alg)
        P, n_coarse, P_update_map = _coarsen_with_fallback(A_current, coarsening_alg, config; backend=backend, block_size=block_size, setup_workspace=setup_workspace, build_P_update_map=build_P_update_map)
        n_coarse >= n && break
        n_coarse == 0 && break
        A_cpu = csr_to_cpu(A_current)
        # Get old A_coarse for array reuse (stored as next level's A, CPU only)
        old_A_coarse = nothing
        if !is_gpu && lvl + 1 <= length(old_levels)
            old_A_c = old_levels[lvl + 1].A
            if old_A_c.nzval isa Vector
                old_A_coarse = old_A_c
            end
        end
        # Build transpose map first — needed by compute_coarse_sparsity to
        # iterate by coarse row (P^T structure)
        Pt_map = build_transpose_map(P)
        A_coarse, r_map = compute_coarse_sparsity(A_cpu, P, Pt_map, n_coarse; build_restriction_map=allow_partial_resetup, workspace=galerkin_workspace, old_A_coarse=old_A_coarse)
        # P_update_map is now returned directly from _coarsen_with_fallback
        if is_gpu
            A_dev = _csr_to_device(A_ref, A_cpu)
            P_dev = _prolongation_to_device(A_ref, P)
            Pt_map_dev = _transpose_map_to_device(A_ref, Pt_map)
            r_map_dev = r_map === nothing ? nothing : _restriction_map_to_device(A_ref, r_map)
            P_update_map_dev = P_update_map === nothing ? nothing : _prolongation_update_map_to_device(A_ref, P_update_map)
            smoother = build_smoother(A_dev, config.smoother, config.jacobi_omega; backend=backend, block_size=block_size)
            r = _reuse_or_allocate_vector(old_lvl, :r, A_ref, Tv, n)
            xc = _reuse_or_allocate_vector(old_lvl, :xc, A_ref, Tv, n_coarse)
            bc = _reuse_or_allocate_vector(old_lvl, :bc, A_ref, Tv, n_coarse)
            level = AMGLevel{Tv, Ti}(A_dev, P_dev, Pt_map_dev, r_map_dev, smoother, r, xc, bc, P_update_map_dev)
        else
            smoother = build_smoother(A_cpu, config.smoother, config.jacobi_omega; backend=backend, block_size=block_size)
            r = _reuse_or_allocate_vector(old_lvl, :r, A_ref, Tv, n)
            xc = _reuse_or_allocate_vector(old_lvl, :xc, A_ref, Tv, n_coarse)
            bc = _reuse_or_allocate_vector(old_lvl, :bc, A_ref, Tv, n_coarse)
            level = AMGLevel{Tv, Ti}(A_cpu, P, Pt_map, r_map, smoother, r, xc, bc, P_update_map)
        end
        push!(levels, level)
        A_current = A_coarse
    end
    return A_current
end

"""
    _build_coarse_solver(A_current, A_ref, n_finest, config; backend, block_size)

Build the direct solver for the coarsest level. Returns a tuple of
`(coarse_dense, coarse_factor, coarse_x, coarse_b, solve_r)`.

`A_ref` is used as the device reference for GPU allocation. `n_finest` is the
finest-level matrix size (for the residual workspace `solve_r`).
"""
function _build_coarse_solver(A_current::CSRMatrix{Tv}, A_ref::CSRMatrix,
                              n_finest::Int, config::AMGConfig;
                              backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv}
    is_gpu = !(A_ref.nzval isa Array)
    n_coarse = size(A_current, 1)
    if is_gpu
        A_cpu = csr_to_cpu(A_current)
        coarse_cpu = Matrix{Tv}(undef, n_coarse, n_coarse)
        _csr_to_dense!(coarse_cpu, A_cpu)
        if config.coarse_solve_on_cpu
            coarse_dense = coarse_cpu
            coarse_factor = lu(coarse_dense)
        else
            coarse_dev = _allocate_dense_matrix(A_ref, Tv, n_coarse, n_coarse)
            copyto!(coarse_dev, coarse_cpu)
            coarse_dense, coarse_factor = _build_coarse_lu(coarse_dev)
        end
        coarse_x = similar(coarse_dense, Tv, n_coarse)
        fill!(coarse_x, zero(Tv))
        coarse_b = similar(coarse_dense, Tv, n_coarse)
        fill!(coarse_b, zero(Tv))
        solve_r = _allocate_vector(A_ref, Tv, n_finest)
    else
        coarse_dense = Matrix{Tv}(undef, n_coarse, n_coarse)
        A_cpu = csr_to_cpu(A_current)
        _csr_to_dense!(coarse_dense, A_cpu)
        coarse_factor = lu(coarse_dense)
        coarse_x = Vector{Tv}(undef, n_coarse)
        coarse_b = Vector{Tv}(undef, n_coarse)
        solve_r = Vector{Tv}(undef, n_finest)
    end
    return coarse_dense, coarse_factor, coarse_x, coarse_b, solve_r
end

"""
    _coarsen_with_fallback(A, alg, config; build_P_update_map=false)

Attempt coarsening. If the result is poor (n_coarse/n > 0.8), retry with
progressively lower θ (halving each time, up to 3 attempts). This handles
the common case where coarser-level matrices have sparser strong connectivity
and the original θ is too aggressive.

When `build_P_update_map=true` and using CF-splitting coarsening, also returns
the data needed to recompute P values in-place during resetup.

Returns `(P, n_coarse, cf_info)` where `cf_info` is either `nothing` for
aggregation-based methods, or a NamedTuple with CF-split data for CF methods.
"""
function _coarsen_with_fallback(A::CSRMatrix, alg::CoarseningAlgorithm,
                                config::AMGConfig;
                                backend=DEFAULT_BACKEND, block_size::Int=64,
                                setup_workspace=nothing,
                                build_P_update_map::Bool=false)
    n = size(A, 1)
    P, n_coarse, cf_info = _coarsen_and_build_P(A, alg, config; backend=backend, block_size=block_size, setup_workspace=setup_workspace, build_P_update_map=build_P_update_map)
    # If coarsening is adequate, return
    (n_coarse < 0.8 * n || n <= config.max_coarse_size) && return P, n_coarse, cf_info
    # Try reducing θ
    for attempt in 1:3
        reduced_alg = _reduce_theta(alg, 0.5^attempt)
        reduced_alg === nothing && return P, n_coarse, cf_info  # no θ to reduce
        P2, nc2, cf_info2 = _coarsen_and_build_P(A, reduced_alg, config; backend=backend, block_size=block_size, setup_workspace=setup_workspace, build_P_update_map=build_P_update_map)
        if nc2 < n_coarse
            P, n_coarse, cf_info = P2, nc2, cf_info2
        end
        (n_coarse < 0.8 * n) && break
    end
    return P, n_coarse, cf_info
end

"""Reduce the θ parameter of a coarsening algorithm by a factor. Returns nothing
if the algorithm has no θ parameter."""
_reduce_theta(a::AggregationCoarsening, f) = AggregationCoarsening(a.θ * f, a.filtering, a.filter_tol)
_reduce_theta(a::SmoothedAggregationCoarsening, f) = SmoothedAggregationCoarsening(a.θ * f, a.ω, a.filtering, a.filter_tol)
_reduce_theta(a::PMISCoarsening, f) = PMISCoarsening(a.θ * f, a.interpolation)
_reduce_theta(a::HMISCoarsening, f) = HMISCoarsening(a.θ * f, a.interpolation)
_reduce_theta(a::RSCoarsening, f) = RSCoarsening(a.θ * f, a.interpolation)
_reduce_theta(a::AggressiveCoarsening, f) = AggressiveCoarsening(a.θ * f, a.base, a.interpolation)
_reduce_theta(::CoarseningAlgorithm, _) = nothing

"""
    _coarsen_and_build_P(A, alg, config; build_P_update_map=false)

Perform coarsening and build the prolongation operator. Dispatches based on
whether the algorithm uses CF-splitting or aggregation.
When max_row_sum is configured, strength computation uses a weakened matrix.

Returns `(P, n_coarse, cf_info)` where `cf_info` is:
- For CF-splitting methods with `build_P_update_map=true`: a NamedTuple
  `(cf=cf, coarse_map=coarse_map, θ=θ, interp_type=interp)`
- Otherwise: `nothing`
"""
function _coarsen_and_build_P(A::CSRMatrix, alg::CoarseningAlgorithm,
                              config::AMGConfig=AMGConfig();
                              backend=DEFAULT_BACKEND, block_size::Int=64,
                              setup_workspace=nothing,
                              build_P_update_map::Bool=false)
    if uses_cf_splitting(alg)
        cf, coarse_map, n_coarse = coarsen_cf(A, alg, config; backend=backend, block_size=block_size, setup_workspace=setup_workspace)
        P, P_update_map = build_cf_prolongation(A, cf, coarse_map, n_coarse, alg.interpolation, alg.θ; backend=backend, block_size=block_size, setup_workspace=setup_workspace, build_update_map=build_P_update_map)
        # Apply interpolation truncation if configured
        tf = _get_trunc_factor(alg.interpolation)
        if tf > 0
            P = _truncate_interpolation(P, tf)
        end
        return P, n_coarse, P_update_map
    else
        agg, n_coarse = coarsen(A, alg, config; backend=backend, block_size=block_size, setup_workspace=setup_workspace)
        P = build_prolongation(A, agg, n_coarse)
        # Apply filtering if requested
        if _has_filtering(alg) && alg.filtering
            P = _filter_prolongation(P, alg.filter_tol)
        end
        return P, n_coarse, nothing
    end
end

function _coarsen_and_build_P(A::CSRMatrix, alg::SmoothedAggregationCoarsening,
                              config::AMGConfig=AMGConfig();
                              backend=DEFAULT_BACKEND, block_size::Int=64,
                              setup_workspace=nothing,
                              build_P_update_map::Bool=false)
    agg, n_coarse = coarsen(A, AggregationCoarsening(alg.θ), config; backend=backend, block_size=block_size, setup_workspace=setup_workspace)
    P_tent = build_prolongation(A, agg, n_coarse)
    # Smooth: P = (I - ω D⁻¹ A) P_tent
    P = _smooth_prolongation(A, P_tent, alg.ω)
    # Apply filtering if requested
    if alg.filtering
        P = _filter_prolongation(P, alg.filter_tol)
    end
    return P, n_coarse, nothing
end

"""
    _coarsen_and_build_P(A, alg::AggressiveCoarsening, config; build_P_update_map=false)

Aggressive coarsening dispatch. When `base=:hmis` or `base=:pmis`, performs
two-pass CF-splitting (HYPRE-style aggressive coarsening) and builds
interpolation using the configured interpolation type.
When `base=:pmis` with no interpolation specified, falls back to aggregation-based
aggressive coarsening.
"""
function _coarsen_and_build_P(A::CSRMatrix, alg::AggressiveCoarsening,
                              config::AMGConfig=AMGConfig();
                              backend=DEFAULT_BACKEND, block_size::Int=64,
                              setup_workspace=nothing,
                              build_P_update_map::Bool=false)
    if alg.base == :hmis
        cf, coarse_map, n_coarse = coarsen_aggressive_cf(A, alg.θ, :hmis; config=config, backend=backend, block_size=block_size, setup_workspace=setup_workspace)
        P, P_update_map = build_cf_prolongation(A, cf, coarse_map, n_coarse, alg.interpolation, alg.θ; backend=backend, block_size=block_size, setup_workspace=setup_workspace, build_update_map=build_P_update_map)
        tf = _get_trunc_factor(alg.interpolation)
        if tf > 0
            P = _truncate_interpolation(P, tf)
        end
        return P, n_coarse, P_update_map
    elseif alg.base == :pmis
        cf, coarse_map, n_coarse = coarsen_aggressive_cf(A, alg.θ, :pmis; config=config, backend=backend, block_size=block_size, setup_workspace=setup_workspace)
        P, P_update_map = build_cf_prolongation(A, cf, coarse_map, n_coarse, alg.interpolation, alg.θ; backend=backend, block_size=block_size, setup_workspace=setup_workspace, build_update_map=build_P_update_map)
        tf = _get_trunc_factor(alg.interpolation)
        if tf > 0
            P = _truncate_interpolation(P, tf)
        end
        return P, n_coarse, P_update_map
    else
        # Legacy: aggregation-based aggressive coarsening
        agg, n_coarse = coarsen(A, alg, config; backend=backend, block_size=block_size, setup_workspace=setup_workspace)
        P = build_prolongation(A, agg, n_coarse)
        return P, n_coarse, nothing
    end
end

_has_filtering(::AggregationCoarsening) = true
_has_filtering(::SmoothedAggregationCoarsening) = true
_has_filtering(::CoarseningAlgorithm) = false

"""Get the truncation factor from an interpolation type (0 = no truncation)."""
_get_trunc_factor(i::DirectInterpolation) = i.trunc_factor
_get_trunc_factor(i::StandardInterpolation) = i.trunc_factor
_get_trunc_factor(i::ExtendedIInterpolation) = i.trunc_factor
_get_trunc_factor(::InterpolationType) = 0.0

"""
    _truncate_interpolation(P, trunc_factor)

Truncate interpolation weights: for each row, drop entries with
|w| < trunc_factor * max|w| and rescale remaining entries to preserve row sum.
This is equivalent to HYPRE's AggTruncFactor.
Preserves `trunc_scaling` from the input P when present, filtering it in
sync with the nzval entries.
"""
function _truncate_interpolation(P::ProlongationOp{Ti, Tv}, trunc_factor::Real) where {Ti, Tv}
    P_new = _filter_prolongation(P, trunc_factor)
    if P.trunc_scaling !== nothing
        # Re-filter trunc_scaling using same logic as _filter_prolongation
        S_new = Tv[]
        @inbounds for i in 1:P.nrow
            rstart = P.rowptr[i]
            rend = P.rowptr[i+1] - 1
            rstart > rend && continue
            max_val = zero(real(Tv))
            for nz in rstart:rend
                max_val = max(max_val, abs(P.nzval[nz]))
            end
            threshold = Tv(trunc_factor) * max_val
            row_count = 0
            for nz in rstart:rend
                if abs(P.nzval[nz]) >= threshold
                    push!(S_new, P.trunc_scaling[nz])
                    row_count += 1
                end
            end
            if row_count == 0
                best_nz = rstart
                best_val = zero(real(Tv))
                for nz in rstart:rend
                    if abs(P.nzval[nz]) > best_val
                        best_val = abs(P.nzval[nz])
                        best_nz = nz
                    end
                end
                push!(S_new, P.trunc_scaling[best_nz])
            end
        end
        P_new.trunc_scaling = S_new
    end
    return P_new
end

# ── Pretty-printing helpers ──────────────────────────────────────────────────

"""
    _coarsening_name(alg)

Return a human-readable string describing the coarsening algorithm and its parameters.
"""
_coarsening_name(a::AggregationCoarsening) = a.filtering ? "Aggregation(θ=$(a.θ), filtered)" : "Aggregation(θ=$(a.θ))"
_coarsening_name(a::PMISCoarsening) = "PMIS(θ=$(a.θ), $(typeof(a.interpolation).name.name))"
_coarsening_name(a::HMISCoarsening) = "HMIS(θ=$(a.θ), $(typeof(a.interpolation).name.name))"
_coarsening_name(a::AggressiveCoarsening) = "Aggressive(θ=$(a.θ), base=$(a.base), $(typeof(a.interpolation).name.name))"
_coarsening_name(a::RSCoarsening) = "RS(θ=$(a.θ), $(typeof(a.interpolation).name.name))"
_coarsening_name(a::SmoothedAggregationCoarsening) = a.filtering ? "SmoothedAgg(θ=$(a.θ), ω=$(round(a.ω; digits=3)), filtered)" : "SmoothedAgg(θ=$(a.θ), ω=$(round(a.ω; digits=3)))"

"""
    _print_hierarchy_info(hierarchy, config, n_finest, t_setup)

Print AMG hierarchy complexity information including coarsening details.
"""
function _print_hierarchy_info(hierarchy::AMGHierarchy, config::AMGConfig, n_finest::Int, t_setup::Float64)
    nlevels = length(hierarchy.levels)
    total_nnz = 0
    total_rows = 0
    println("+=============================================================================+")
    println("                         AMG Hierarchy Summary                                  ")
    println("+=============================================================================+")
    Printf.@printf("  Setup time:      %.4f s\n", t_setup)
    println("  Levels:          $(nlevels + 1) ($(nlevels) AMG + 1 coarse direct)")
    println("  Cycle type:      $(config.cycle_type)-cycle")
    println("  Backend:         $(_backend_name(hierarchy.backend))")
    println("  Block size:      $(hierarchy.block_size)")
    println("  Coarse solve:    $(hierarchy.coarse_solve_on_cpu ? "CPU (forced)" : "default")")
    println("  Strength:        $(typeof(config.strength_type).name.name)")
    # Coarsening info
    coarsening_str = _coarsening_name(config.coarsening)
    println("  Coarsening:      $coarsening_str")
    if config.initial_coarsening_levels > 0
        init_str = _coarsening_name(config.initial_coarsening)
        println("  Initial coars.:  $init_str (first $(config.initial_coarsening_levels) levels)")
    end
    println("  Smoother:        $(typeof(config.smoother).name.name)")
    if config.smoother isa JacobiSmootherType || config.smoother isa L1JacobiSmootherType
        Printf.@printf("    omega = %.3f\n", config.jacobi_omega)
    end
    Printf.@printf("  Pre-smooth:      %d steps, Post-smooth: %d steps\n",
                    config.pre_smoothing_steps, config.post_smoothing_steps)
    if config.max_row_sum < 1.0
        Printf.@printf("  Max row sum:     %.2f\n", config.max_row_sum)
    end
    println("+-------+----------+-----------+---------+------------------------------------+")
    println("  Level     Rows        NNZ      Ratio    Smoother                              ")
    println("+-------+----------+-----------+---------+------------------------------------+")
    for (i, lvl) in enumerate(hierarchy.levels)
        n = size(lvl.A, 1)
        nz = nnz(lvl.A)
        total_nnz += nz
        total_rows += n
        sname = _smoother_name(lvl.smoother)
        if i == 1
            Printf.@printf("  %5d   %7d    %8d        -   %s\n", i, n, nz, sname)
        else
            prev_n = size(hierarchy.levels[i-1].A, 1)
            ratio = n / prev_n
            Printf.@printf("  %5d   %7d    %8d    %5.3f   %s\n", i, n, nz, ratio, sname)
        end
    end
    # Coarsest level
    nc = size(hierarchy.coarse_A, 1)
    nc_nnz = count(!iszero, hierarchy.coarse_A)
    total_nnz += nc_nnz
    total_rows += nc
    if nlevels > 0
        prev_n = size(hierarchy.levels[end].A, 1)
        ratio = nc / prev_n
        Printf.@printf("  %5d   %7d    %8d    %5.3f   %s\n", nlevels + 1, nc, nc_nnz, ratio, "Direct (LU)")
    else
        Printf.@printf("  %5d   %7d    %8d         -   %s\n", nlevels + 1, nc, nc_nnz, "Direct (LU)")
    end
    println("+=============================================================================+")
    if nlevels > 0
        finest_nnz = nnz(hierarchy.levels[1].A)
        oc = total_nnz / finest_nnz
        gc = total_rows / n_finest
        Printf.@printf("  Operator complexity: %.3f\n", oc)
        Printf.@printf("  Grid complexity:     %.3f\n", gc)
    end
    println("+=============================================================================+")
end

_backend_name(b::CPU) = "CPU"
_backend_name(b) = string(typeof(b))

_smoother_name(::JacobiSmoother) = "Jacobi"
_smoother_name(::ColoredGaussSeidelSmoother) = "Colored GS"
_smoother_name(::L1ColoredGaussSeidelSmoother) = "l1-Colored GS"
_smoother_name(::SerialGaussSeidelSmoother) = "Serial GS"
_smoother_name(::L1SerialGaussSeidelSmoother) = "l1-Serial GS"
_smoother_name(::SPAI0Smoother) = "SPAI(0)"
_smoother_name(::SPAI1Smoother) = "SPAI(1)"
_smoother_name(::L1JacobiSmoother) = "l1-Jacobi"
_smoother_name(::ChebyshevSmoother) = "Chebyshev"
_smoother_name(::ILU0Smoother) = "ILU(0)"
_smoother_name(::SerialILU0Smoother) = "Serial ILU(0)"

"""
    _csr_to_dense!(M, A; backend=DEFAULT_BACKEND)

Convert a CSRMatrix to a dense matrix using a KA kernel.
"""
function _csr_to_dense!(M::AbstractMatrix{Tv}, A::CSRMatrix{Tv};
                        backend=_get_backend(nonzeros(A)), block_size::Int=64) where {Tv}
    fill!(M, zero(Tv))
    n = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)
    rp = rowptr(A)
    kernel! = csr_to_dense_kernel!(backend, block_size)
    kernel!(M, nzv, cv, rp; ndrange=n)
    _synchronize(backend)
    return M
end

@kernel function csr_to_dense_kernel!(M, @Const(nzval), @Const(colval), @Const(rp))
    i = @index(Global)
    @inbounds begin
        for nz in rp[i]:(rp[i+1]-1)
            j = colval[nz]
            M[i, j] = nzval[nz]
        end
    end
end

"""
    _build_coarse_lu(M::AbstractMatrix{Tv}) -> (dense_matrix, factorization)

Build an LU factorization of the dense coarse matrix `M`.
For GPU arrays that support `lu()` (e.g., CuArray), the factorization stays on device.
For GPU arrays without native `lu()` support, falls back to CPU.
Returns a tuple of (dense_matrix, factorization) where both are on the same device.
"""
function _build_coarse_lu(M::AbstractMatrix{Tv}) where {Tv}
    try
        return M, lu(M)
    catch
        # Fall back to CPU if the GPU backend doesn't support lu()
        M_cpu = Matrix{Tv}(M)
        return M_cpu, lu(M_cpu)
    end
end
