"""
    amg_setup(A, config) -> AMGHierarchy

Perform the full AMG setup (analysis phase). This determines the coarsening at each
level, constructs prolongation operators, computes Galerkin products, and sets up
smoothers.

The sparsity structure computed here is reused by `amg_resetup!` when matrix
coefficients change but the pattern remains the same.

The external API accepts `StaticSparsityMatrixCSR`; internally the hierarchy stores
lightweight `CSRMatrix` objects (raw CSR vectors).
"""
function amg_setup(A::StaticSparsityMatrixCSR{Tv, Ti}, config::AMGConfig=AMGConfig();
                   backend=DEFAULT_BACKEND) where {Tv, Ti}
    t_setup = time()
    levels = AMGLevel{Tv, Ti}[]
    A_csr = csr_from_static(A)
    A_current = A_csr
    n_finest = size(A, 1)
    consecutive_stalls = 0
    for lvl in 1:(config.max_levels - 1)
        n = size(A_current, 1)
        n <= config.max_coarse_size && break
        # Select coarsening algorithm for this level
        coarsening_alg = _get_coarsening_for_level(config, lvl)
        # Coarsen and build prolongation
        P, n_coarse = _coarsen_and_build_P(A_current, coarsening_alg, config)
        n_coarse >= n && break  # no coarsening progress
        # Check minimum coarsening ratio to avoid stalling
        coarse_ratio = n_coarse / n
        if coarse_ratio > config.min_coarse_ratio && n_coarse > config.max_coarse_size
            consecutive_stalls += 1
            if consecutive_stalls >= config.max_stall_levels
                config.verbose && println("Coarsening stalled at level $lvl: ratio=$(round(coarse_ratio, digits=3)), $consecutive_stalls consecutive stalls, stopping.")
                break
            else
                config.verbose && println("Warning: poor coarsening at level $lvl (ratio=$(round(coarse_ratio, digits=3))), stall $(consecutive_stalls)/$(config.max_stall_levels).")
            end
        else
            consecutive_stalls = 0
        end
        # Compute coarse operator via Galerkin product
        A_coarse, r_map = compute_coarse_sparsity(A_current, P, n_coarse)
        # Build smoother
        smoother = build_smoother(A_current, config.smoother, config.jacobi_omega; backend=backend)
        # Workspace
        r = KernelAbstractions.zeros(backend, Tv, n)
        xc = KernelAbstractions.zeros(backend, Tv, n_coarse)
        bc = KernelAbstractions.zeros(backend, Tv, n_coarse)
        level = AMGLevel{Tv, Ti}(A_current, P, r_map, smoother, r, xc, bc)
        push!(levels, level)
        A_current = A_coarse
    end
    # Set up direct solver at coarsest level with in-place LU buffer
    n_coarse = size(A_current, 1)
    coarse_dense = Matrix{Tv}(undef, n_coarse, n_coarse)
    _csr_to_dense!(coarse_dense, A_current)
    coarse_lu = copy(coarse_dense)
    coarse_ipiv = Vector{LinearAlgebra.BlasInt}(undef, n_coarse)
    LinearAlgebra.LAPACK.getrf!(coarse_lu, coarse_ipiv)
    coarse_factor = LU(coarse_lu, coarse_ipiv, 0)  # 0 = successful factorization info
    coarse_x = KernelAbstractions.zeros(backend, Tv, n_coarse)
    coarse_b = KernelAbstractions.zeros(backend, Tv, n_coarse)
    # Pre-allocate residual buffer for amg_solve! at finest level size
    solve_r = KernelAbstractions.zeros(backend, Tv, n_finest)
    hierarchy = AMGHierarchy{Tv, Ti}(levels, coarse_dense, coarse_lu, coarse_ipiv,
                                      coarse_factor, coarse_x, coarse_b, solve_r)
    t_setup = time() - t_setup
    if config.verbose
        _print_hierarchy_info(hierarchy, config, n_finest, t_setup)
    end
    return hierarchy
end

"""
    _coarsen_and_build_P(A, alg, config)

Perform coarsening and build the prolongation operator. Dispatches based on
whether the algorithm uses CF-splitting or aggregation.
When max_row_sum is configured, strength computation uses a weakened matrix.
"""
function _coarsen_and_build_P(A::CSRMatrix, alg::CoarseningAlgorithm,
                              config::AMGConfig=AMGConfig())
    if uses_cf_splitting(alg)
        cf, coarse_map, n_coarse = coarsen_cf(A, alg, config)
        P = build_cf_prolongation(A, cf, coarse_map, n_coarse, alg.interpolation)
        return P, n_coarse
    else
        agg, n_coarse = coarsen(A, alg, config)
        P = build_prolongation(A, agg, n_coarse)
        # Apply filtering if requested
        if _has_filtering(alg) && alg.filtering
            P = _filter_prolongation(P, alg.filter_tol)
        end
        return P, n_coarse
    end
end

function _coarsen_and_build_P(A::CSRMatrix, alg::SmoothedAggregationCoarsening,
                              config::AMGConfig=AMGConfig())
    agg, n_coarse = coarsen(A, AggregationCoarsening(alg.θ), config)
    P_tent = build_prolongation(A, agg, n_coarse)
    # Smooth: P = (I - ω D⁻¹ A) P_tent
    P = _smooth_prolongation(A, P_tent, alg.ω)
    # Apply filtering if requested
    if alg.filtering
        P = _filter_prolongation(P, alg.filter_tol)
    end
    return P, n_coarse
end

_has_filtering(::AggregationCoarsening) = true
_has_filtering(::SmoothedAggregationCoarsening) = true
_has_filtering(::CoarseningAlgorithm) = false

# ── Pretty-printing helpers ──────────────────────────────────────────────────

"""
    _coarsening_name(alg)

Return a human-readable string describing the coarsening algorithm and its parameters.
"""
_coarsening_name(a::AggregationCoarsening) = a.filtering ? "Aggregation(θ=$(a.θ), filtered)" : "Aggregation(θ=$(a.θ))"
_coarsening_name(a::PMISCoarsening) = "PMIS(θ=$(a.θ), $(typeof(a.interpolation).name.name))"
_coarsening_name(a::HMISCoarsening) = "HMIS(θ=$(a.θ), $(typeof(a.interpolation).name.name))"
_coarsening_name(a::AggressiveCoarsening) = "Aggressive(θ=$(a.θ))"
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
_smoother_name(::SPAI0Smoother) = "SPAI(0)"
_smoother_name(::SPAI1Smoother) = "SPAI(1)"
_smoother_name(::L1JacobiSmoother) = "l1-Jacobi"
_smoother_name(::ChebyshevSmoother) = "Chebyshev"
_smoother_name(::ILU0Smoother) = "ILU(0)"

"""
    _csr_to_dense!(M, A; backend=DEFAULT_BACKEND)

Convert a CSRMatrix to a dense matrix using a KA kernel.
"""
function _csr_to_dense!(M::Matrix{Tv}, A::CSRMatrix{Tv, Ti};
                        backend=DEFAULT_BACKEND) where {Tv, Ti}
    fill!(M, zero(Tv))
    n = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)
    rp = rowptr(A)
    kernel! = csr_to_dense_kernel!(backend, 64)
    kernel!(M, nzv, cv, rp; ndrange=n)
    KernelAbstractions.synchronize(backend)
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
