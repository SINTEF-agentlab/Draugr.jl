"""
    amg_setup(A, config) -> AMGHierarchy

Perform the full AMG setup (analysis phase). This determines the coarsening at each
level, constructs prolongation operators, computes Galerkin products, and sets up
smoothers.

The sparsity structure computed here is reused by `amg_resetup!` when matrix
coefficients change but the pattern remains the same.
"""
function amg_setup(A::StaticSparsityMatrixCSR{Tv, Ti}, config::AMGConfig=AMGConfig()) where {Tv, Ti}
    t_setup = time()
    levels = AMGLevel{Tv, Ti}[]
    A_current = A
    n_finest = size(A, 1)
    for lvl in 1:(config.max_levels - 1)
        n = size(A_current, 1)
        n <= config.max_coarse_size && break
        # Select coarsening algorithm for this level
        coarsening_alg = _get_coarsening_for_level(config, lvl)
        # Coarsen and build prolongation
        P, n_coarse = _coarsen_and_build_P(A_current, coarsening_alg)
        n_coarse >= n && break  # no coarsening progress
        # Compute coarse operator via Galerkin product
        A_coarse, r_map = compute_coarse_sparsity(A_current, P, n_coarse)
        # Build smoother
        smoother = build_smoother(A_current, config.smoother, config.jacobi_omega)
        # Workspace
        r = zeros(Tv, n)
        xc = zeros(Tv, n_coarse)
        bc = zeros(Tv, n_coarse)
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
    coarse_x = zeros(Tv, n_coarse)
    coarse_b = zeros(Tv, n_coarse)
    # Pre-allocate residual buffer for amg_solve! at finest level size
    solve_r = zeros(Tv, n_finest)
    hierarchy = AMGHierarchy{Tv, Ti}(levels, coarse_dense, coarse_lu, coarse_ipiv,
                                      coarse_factor, coarse_x, coarse_b, solve_r)
    t_setup = time() - t_setup
    if config.verbose
        _print_hierarchy_info(hierarchy, n_finest, t_setup)
    end
    return hierarchy
end

"""
    _coarsen_and_build_P(A, alg)

Perform coarsening and build the prolongation operator. Dispatches based on
whether the algorithm uses CF-splitting or aggregation.
"""
function _coarsen_and_build_P(A::StaticSparsityMatrixCSR, alg::CoarseningAlgorithm)
    if uses_cf_splitting(alg)
        cf, coarse_map, n_coarse = coarsen_cf(A, alg)
        P = build_cf_prolongation(A, cf, coarse_map, n_coarse, alg.interpolation)
        return P, n_coarse
    else
        agg, n_coarse = coarsen(A, alg)
        P = build_prolongation(A, agg, n_coarse)
        return P, n_coarse
    end
end

"""
    _print_hierarchy_info(hierarchy, n_finest, t_setup)

Print AMG hierarchy complexity information.
"""
function _print_hierarchy_info(hierarchy::AMGHierarchy, n_finest::Int, t_setup::Float64)
    nlevels = length(hierarchy.levels)
    total_nnz = 0
    total_rows = 0
    println("╔══════════════════════════════════════════════════════════╗")
    println("║            AMG Hierarchy Summary                       ║")
    println("╠══════════════════════════════════════════════════════════╣")
    Printf.@printf("║  Setup time: %.4f s                                   \n", t_setup)
    println("║  Levels: $(nlevels + 1) ($(nlevels) AMG + 1 coarse direct)")
    println("╠══════════════════════════════════════════════════════════╣")
    println("║  Level │    Rows │      NNZ │ Smoother                  ")
    println("╠────────┼─────────┼──────────┼───────────────────────────╣")
    for (i, lvl) in enumerate(hierarchy.levels)
        n = size(lvl.A, 1)
        nz = nnz(lvl.A)
        total_nnz += nz
        total_rows += n
        sname = _smoother_name(lvl.smoother)
        Printf.@printf("║  %5d │ %7d │ %8d │ %s\n", i, n, nz, sname)
    end
    # Coarsest level
    nc = size(hierarchy.coarse_A, 1)
    nc_nnz = count(!iszero, hierarchy.coarse_A)
    total_nnz += nc_nnz
    total_rows += nc
    Printf.@printf("║  %5d │ %7d │ %8d │ %s\n", nlevels + 1, nc, nc_nnz, "Direct (LU)")
    println("╠══════════════════════════════════════════════════════════╣")
    if nlevels > 0
        finest_nnz = nnz(hierarchy.levels[1].A)
        oc = total_nnz / finest_nnz
        gc = total_rows / n_finest
        Printf.@printf("║  Operator complexity: %.3f                            \n", oc)
        Printf.@printf("║  Grid complexity:     %.3f                            \n", gc)
    end
    println("╚══════════════════════════════════════════════════════════╝")
end

_smoother_name(::JacobiSmoother) = "Jacobi"
_smoother_name(::ColoredGaussSeidelSmoother) = "Colored GS"
_smoother_name(::SPAI0Smoother) = "SPAI(0)"
_smoother_name(::SPAI1Smoother) = "SPAI(1)"

"""
    _csr_to_dense!(M, A; backend=CPU())

Convert a StaticSparsityMatrixCSR to a dense matrix using a KA kernel.
"""
function _csr_to_dense!(M::Matrix{Tv}, A::StaticSparsityMatrixCSR{Tv, Ti};
                        backend=CPU()) where {Tv, Ti}
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
