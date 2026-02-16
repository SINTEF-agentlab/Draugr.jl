# ── Device conversion helpers for AMG structures ─────────────────────────────

"""
    _prolongation_to_device(ref, P) -> ProlongationOp

Copy a ProlongationOp to the same device as `ref`'s arrays.
"""
function _prolongation_to_device(ref::CSRMatrix, P::ProlongationOp)
    rp = _to_device(ref, P.rowptr)
    cv = _to_device(ref, P.colval)
    nzv = _to_device(ref, P.nzval)
    return ProlongationOp(rp, cv, nzv, P.nrow, P.ncol)
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

The sparsity structure computed here is reused by `amg_resetup!` when matrix
coefficients change but the pattern remains the same.
"""
function amg_setup(A_csr::CSRMatrix{Tv, Ti}, config::AMGConfig=AMGConfig();
                   backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    t_setup = time()
    levels = AMGLevel{Tv, Ti}[]
    A_current = A_csr
    n_finest = size(A_csr, 1)
    # Determine if we need GPU-resident arrays: any non-CPU array type
    # (CuArray, JLArray, MtlArray, etc.) triggers GPU-resident hierarchy
    is_gpu = !(A_csr.nzval isa Array)
    for lvl in 1:(config.max_levels - 1)
        n = size(A_current, 1)
        n <= config.max_coarse_size && break
        # Select coarsening algorithm for this level
        coarsening_alg = _get_coarsening_for_level(config, lvl)
        # Coarsen and build prolongation (coarsening converts to CPU internally)
        P, n_coarse = _coarsen_with_fallback(A_current, coarsening_alg, config; backend=backend, block_size=block_size)
        n_coarse >= n && break  # no coarsening progress
        n_coarse == 0 && break  # degenerate case
        # Compute coarse operator via Galerkin product (CPU — inherently sequential)
        A_cpu = csr_to_cpu(A_current)
        A_coarse, r_map = compute_coarse_sparsity(A_cpu, P, n_coarse)
        # Build transpose map for atomic-free restriction
        Pt_map = build_transpose_map(P)
        if is_gpu
            # Copy level structures to GPU device so kernels can access them
            A_dev = _csr_to_device(A_csr, A_cpu)
            P_dev = _prolongation_to_device(A_csr, P)
            Pt_map_dev = _transpose_map_to_device(A_csr, Pt_map)
            r_map_dev = _restriction_map_to_device(A_csr, r_map)
            # Build smoother on device arrays
            smoother = build_smoother(A_dev, config.smoother, config.jacobi_omega; backend=backend, block_size=block_size)
            # Workspace on device
            r = _allocate_vector(A_csr, Tv, n)
            xc = _allocate_vector(A_csr, Tv, n_coarse)
            bc = _allocate_vector(A_csr, Tv, n_coarse)
            level = AMGLevel{Tv, Ti}(A_dev, P_dev, Pt_map_dev, r_map_dev, smoother, r, xc, bc)
        else
            # CPU path: use arrays as-is
            smoother = build_smoother(A_cpu, config.smoother, config.jacobi_omega; backend=backend, block_size=block_size)
            r = Vector{Tv}(undef, n)
            xc = Vector{Tv}(undef, n_coarse)
            bc = Vector{Tv}(undef, n_coarse)
            level = AMGLevel{Tv, Ti}(A_cpu, P, Pt_map, r_map, smoother, r, xc, bc)
        end
        push!(levels, level)
        A_current = A_coarse
    end
    # Set up direct solver at coarsest level using high-level lu()
    # For GPU: build dense matrix on CPU (A_current may be CPU from coarsening),
    # then copy to device so lu() dispatches to GPU (e.g., CUDA's cuSOLVER).
    # If the GPU backend doesn't support lu(), _build_coarse_lu falls back to CPU.
    n_coarse = size(A_current, 1)
    if is_gpu
        # Build dense on CPU first (A_current is CPU from compute_coarse_sparsity)
        A_cpu = csr_to_cpu(A_current)
        coarse_cpu = Matrix{Tv}(undef, n_coarse, n_coarse)
        _csr_to_dense!(coarse_cpu, A_cpu)
        # Copy to device and attempt GPU LU; falls back to CPU if unsupported
        coarse_dev = _allocate_dense_matrix(A_csr, Tv, n_coarse, n_coarse)
        copyto!(coarse_dev, coarse_cpu)
        coarse_dense, coarse_factor = _build_coarse_lu(coarse_dev)
        coarse_x = similar(coarse_dense, Tv, n_coarse)
        fill!(coarse_x, zero(Tv))
        coarse_b = similar(coarse_dense, Tv, n_coarse)
        fill!(coarse_b, zero(Tv))
        solve_r = _allocate_vector(A_csr, Tv, n_finest)
    else
        coarse_dense = Matrix{Tv}(undef, n_coarse, n_coarse)
        A_cpu = csr_to_cpu(A_current)
        _csr_to_dense!(coarse_dense, A_cpu)
        coarse_factor = lu(coarse_dense)
        coarse_x = Vector{Tv}(undef, n_coarse)
        coarse_b = Vector{Tv}(undef, n_coarse)
        solve_r = Vector{Tv}(undef, n_finest)
    end
    hierarchy = AMGHierarchy{Tv, Ti}(levels, coarse_dense,
                                      coarse_factor, coarse_x, coarse_b, solve_r,
                                      backend, block_size)
    t_setup = time() - t_setup
    if config.verbose >= 1
        _print_hierarchy_info(hierarchy, config, n_finest, t_setup)
    end
    return hierarchy
end

"""
    _coarsen_with_fallback(A, alg, config)

Attempt coarsening. If the result is poor (n_coarse/n > 0.8), retry with
progressively lower θ (halving each time, up to 3 attempts). This handles
the common case where coarser-level matrices have sparser strong connectivity
and the original θ is too aggressive.
"""
function _coarsen_with_fallback(A::CSRMatrix, alg::CoarseningAlgorithm,
                                config::AMGConfig;
                                backend=DEFAULT_BACKEND, block_size::Int=64)
    n = size(A, 1)
    P, n_coarse = _coarsen_and_build_P(A, alg, config; backend=backend, block_size=block_size)
    # If coarsening is adequate, return
    (n_coarse < 0.8 * n || n <= config.max_coarse_size) && return P, n_coarse
    # Try reducing θ
    for attempt in 1:3
        reduced_alg = _reduce_theta(alg, 0.5^attempt)
        reduced_alg === nothing && return P, n_coarse  # no θ to reduce
        P2, nc2 = _coarsen_and_build_P(A, reduced_alg, config; backend=backend, block_size=block_size)
        if nc2 < n_coarse
            P, n_coarse = P2, nc2
        end
        (n_coarse < 0.8 * n) && break
    end
    return P, n_coarse
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
    _coarsen_and_build_P(A, alg, config)

Perform coarsening and build the prolongation operator. Dispatches based on
whether the algorithm uses CF-splitting or aggregation.
When max_row_sum is configured, strength computation uses a weakened matrix.
"""
function _coarsen_and_build_P(A::CSRMatrix, alg::CoarseningAlgorithm,
                              config::AMGConfig=AMGConfig();
                              backend=DEFAULT_BACKEND, block_size::Int=64)
    if uses_cf_splitting(alg)
        cf, coarse_map, n_coarse = coarsen_cf(A, alg, config; backend=backend, block_size=block_size)
        P = build_cf_prolongation(A, cf, coarse_map, n_coarse, alg.interpolation, alg.θ; backend=backend, block_size=block_size)
        # Apply interpolation truncation if configured
        tf = _get_trunc_factor(alg.interpolation)
        if tf > 0
            P = _truncate_interpolation(P, tf)
        end
        return P, n_coarse
    else
        agg, n_coarse = coarsen(A, alg, config; backend=backend, block_size=block_size)
        P = build_prolongation(A, agg, n_coarse)
        # Apply filtering if requested
        if _has_filtering(alg) && alg.filtering
            P = _filter_prolongation(P, alg.filter_tol)
        end
        return P, n_coarse
    end
end

function _coarsen_and_build_P(A::CSRMatrix, alg::SmoothedAggregationCoarsening,
                              config::AMGConfig=AMGConfig();
                              backend=DEFAULT_BACKEND, block_size::Int=64)
    agg, n_coarse = coarsen(A, AggregationCoarsening(alg.θ), config; backend=backend, block_size=block_size)
    P_tent = build_prolongation(A, agg, n_coarse)
    # Smooth: P = (I - ω D⁻¹ A) P_tent
    P = _smooth_prolongation(A, P_tent, alg.ω)
    # Apply filtering if requested
    if alg.filtering
        P = _filter_prolongation(P, alg.filter_tol)
    end
    return P, n_coarse
end

"""
    _coarsen_and_build_P(A, alg::AggressiveCoarsening, config)

Aggressive coarsening dispatch. When `base=:hmis` or `base=:pmis`, performs
two-pass CF-splitting (HYPRE-style aggressive coarsening) and builds
interpolation using the configured interpolation type.
When `base=:pmis` with no interpolation specified, falls back to aggregation-based
aggressive coarsening.
"""
function _coarsen_and_build_P(A::CSRMatrix, alg::AggressiveCoarsening,
                              config::AMGConfig=AMGConfig();
                              backend=DEFAULT_BACKEND, block_size::Int=64)
    if alg.base == :hmis
        cf, coarse_map, n_coarse = coarsen_aggressive_cf(A, alg.θ, :hmis; config=config, backend=backend, block_size=block_size)
        P = build_cf_prolongation(A, cf, coarse_map, n_coarse, alg.interpolation, alg.θ; backend=backend, block_size=block_size)
        tf = _get_trunc_factor(alg.interpolation)
        if tf > 0
            P = _truncate_interpolation(P, tf)
        end
        return P, n_coarse
    elseif alg.base == :pmis
        cf, coarse_map, n_coarse = coarsen_aggressive_cf(A, alg.θ, :pmis; config=config, backend=backend, block_size=block_size)
        P = build_cf_prolongation(A, cf, coarse_map, n_coarse, alg.interpolation, alg.θ; backend=backend, block_size=block_size)
        tf = _get_trunc_factor(alg.interpolation)
        if tf > 0
            P = _truncate_interpolation(P, tf)
        end
        return P, n_coarse
    else
        # Legacy: aggregation-based aggressive coarsening
        agg, n_coarse = coarsen(A, alg, config; backend=backend, block_size=block_size)
        P = build_prolongation(A, agg, n_coarse)
        return P, n_coarse
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
"""
function _truncate_interpolation(P::ProlongationOp{Ti, Tv}, trunc_factor::Real) where {Ti, Tv}
    return _filter_prolongation(P, trunc_factor)
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
    println("╔══════════════════════════════════════════════════════════════════════════════╗")
    println("║                        AMG Hierarchy Summary                                ║")
    println("╠══════════════════════════════════════════════════════════════════════════════╣")
    Printf.@printf("║  Setup time:      %.4f s\n", t_setup)
    println("║  Levels:          $(nlevels + 1) ($(nlevels) AMG + 1 coarse direct)")
    println("║  Cycle type:      $(config.cycle_type)-cycle")
    println("║  Strength:        $(typeof(config.strength_type).name.name)")
    # Coarsening info
    coarsening_str = _coarsening_name(config.coarsening)
    println("║  Coarsening:      $coarsening_str")
    if config.initial_coarsening_levels > 0
        init_str = _coarsening_name(config.initial_coarsening)
        println("║  Initial coars.:  $init_str (first $(config.initial_coarsening_levels) levels)")
    end
    println("║  Smoother:        $(typeof(config.smoother).name.name)")
    if config.smoother isa JacobiSmootherType || config.smoother isa L1JacobiSmootherType
        Printf.@printf("║    ω = %.3f\n", config.jacobi_omega)
    end
    Printf.@printf("║  Pre-smooth:      %d steps, Post-smooth: %d steps\n",
                    config.pre_smoothing_steps, config.post_smoothing_steps)
    if config.max_row_sum > 0
        Printf.@printf("║  Max row sum:     %.2f\n", config.max_row_sum)
    end
    println("╠══════════════════════════════════════════════════════════════════════════════╣")
    println("║  Level │    Rows │      NNZ │   Ratio │ Smoother                            ")
    println("╠────────┼─────────┼──────────┼─────────┼─────────────────────────────────────╣")
    for (i, lvl) in enumerate(hierarchy.levels)
        n = size(lvl.A, 1)
        nz = nnz(lvl.A)
        total_nnz += nz
        total_rows += n
        sname = _smoother_name(lvl.smoother)
        if i == 1
            Printf.@printf("║  %5d │ %7d │ %8d │       — │ %s\n", i, n, nz, sname)
        else
            prev_n = size(hierarchy.levels[i-1].A, 1)
            ratio = n / prev_n
            Printf.@printf("║  %5d │ %7d │ %8d │  %5.3f │ %s\n", i, n, nz, ratio, sname)
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
        Printf.@printf("║  %5d │ %7d │ %8d │  %5.3f │ %s\n", nlevels + 1, nc, nc_nnz, ratio, "Direct (LU)")
    else
        Printf.@printf("║  %5d │ %7d │ %8d │       — │ %s\n", nlevels + 1, nc, nc_nnz, "Direct (LU)")
    end
    println("╠══════════════════════════════════════════════════════════════════════════════╣")
    if nlevels > 0
        finest_nnz = nnz(hierarchy.levels[1].A)
        oc = total_nnz / finest_nnz
        gc = total_rows / n_finest
        Printf.@printf("║  Operator complexity: %.3f\n", oc)
        Printf.@printf("║  Grid complexity:     %.3f\n", gc)
    end
    println("╚══════════════════════════════════════════════════════════════════════════════╝")
end

_smoother_name(::JacobiSmoother) = "Jacobi"
_smoother_name(::ColoredGaussSeidelSmoother) = "Colored GS"
_smoother_name(::SerialGaussSeidelSmoother) = "Serial GS"
_smoother_name(::SPAI0Smoother) = "SPAI(0)"
_smoother_name(::SPAI1Smoother) = "SPAI(1)"
_smoother_name(::L1JacobiSmoother) = "l1-Jacobi"
_smoother_name(::ChebyshevSmoother) = "Chebyshev"
_smoother_name(::ILU0Smoother) = "ILU(0)"

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
