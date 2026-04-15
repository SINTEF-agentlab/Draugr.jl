@testset "AMG Setup - Aggregation" begin
    A = poisson2d_csr(10)
    config = AMGConfig(coarsening=AggregationCoarsening())
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) > 0
    # Check that coarsening reduces size at each level
    for lvl in 1:length(hierarchy.levels)
        level = hierarchy.levels[lvl]
        @test level.P.ncol < level.P.nrow
    end
end

@testset "AMG Setup - PMIS" begin
    A = poisson2d_csr(10)
    config = AMGConfig(coarsening=PMISCoarsening())
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) > 0
end

@testset "AMG Setup - Aggressive" begin
    A = poisson2d_csr(10)
    config = AMGConfig(coarsening=AggressiveCoarsening())
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) > 0
end

@testset "AMG Setup - HMIS" begin
    A = poisson2d_csr(10)
    config = AMGConfig(coarsening=HMISCoarsening())
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) > 0
end

@testset "AMG Cycle Convergence - Aggregation" begin
    n = 20
    A = poisson2d_csr(n)
    N = n*n
    b = ones(N)
    x = zeros(N)
    config = AMGConfig(coarsening=AggregationCoarsening(),
                       pre_smoothing_steps=2, post_smoothing_steps=2)
    hierarchy = amg_setup(A, config)
    # Apply a few cycles and check that residual decreases
    r_prev = norm(b)
    for _ in 1:5
        amg_cycle!(x, b, hierarchy, config)
    end
    r_vec = similar(x)
    mul!(r_vec, A, x)
    r_vec .= b .- r_vec
    @test norm(r_vec) < r_prev
end

@testset "AMG Solve - Aggregation" begin
    test_amg_convergence(AMGConfig(coarsening=AggregationCoarsening()))
end

@testset "AMG Solve - PMIS" begin
    test_amg_convergence(AMGConfig(coarsening=PMISCoarsening()))
end

@testset "AMG Solve - Aggressive" begin
    test_amg_convergence(AMGConfig(coarsening=AggressiveCoarsening()))
end

@testset "AMG Solve - coarse_solve_on_cpu" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    b = rand(N)
    x = zeros(N)
    config = AMGConfig(coarse_solve_on_cpu=true)
    hierarchy = amg_setup(A, config)
    @test hierarchy.coarse_solve_on_cpu == true
    @test hierarchy.coarse_A isa Matrix
    @test hierarchy.coarse_x isa Vector
    @test hierarchy.coarse_b isa Vector
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
    @test niter < 200
end

@testset "AMG Resetup" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    config = AMGConfig(coarsening=AggregationCoarsening())
    hierarchy = amg_setup(A, config)
    # Solve with original matrix
    b = rand(N)
    x1 = zeros(N)
    x1, _ = amg_solve!(x1, b, hierarchy, config; tol=1e-8, maxiter=200)
    r1 = b - sparse(A.At') * x1
    @test norm(r1) / norm(b) < 1e-8
    # Scale matrix values by 2 (same pattern)
    nonzeros(A) .*= 2.0
    # Resetup
    amg_resetup!(hierarchy, A, config)
    # Solve with updated matrix
    x2 = zeros(N)
    x2, _ = amg_solve!(x2, b, hierarchy, config; tol=1e-8, maxiter=200)
    r2 = b - sparse(A.At') * x2
    @test norm(r2) / norm(b) < 1e-8
    # Solutions should be different (A changed)
    @test !isapprox(x1, x2, atol=1e-6)
end

@testset "AMG Resetup Preserves Pattern" begin
    n = 8
    A = poisson2d_csr(n)
    N = n*n
    config = AMGConfig(coarsening=AggregationCoarsening())
    hierarchy = amg_setup(A, config)
    # Record sparsity pattern
    patterns = [(copy(Draugr.colvals(lvl.A)), copy(rowptr(lvl.A)))
                 for lvl in hierarchy.levels]
    # Scale and resetup
    nonzeros(A) .*= 3.0
    amg_resetup!(hierarchy, A, config)
    # Verify sparsity patterns are unchanged
    for (i, (cv, rp)) in enumerate(patterns)
        @test Draugr.colvals(hierarchy.levels[i].A) == cv
        @test rowptr(hierarchy.levels[i].A) == rp
    end
end

@testset "AMG Setup - allow_partial_resetup=false" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    config = AMGConfig(coarsening=AggregationCoarsening(), allow_partial_resetup=false)
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) > 0
    # R_map should be nothing when allow_partial_resetup=false
    for lvl in hierarchy.levels
        @test lvl.R_map === nothing
    end
    # Solve should still work
    b = rand(N)
    x = zeros(N)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
    @test niter < 200
end

@testset "AMG Setup - allow_partial_resetup=true has R_map" begin
    n = 10
    A = poisson2d_csr(n)
    config = AMGConfig(coarsening=AggregationCoarsening(), allow_partial_resetup=true)
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) > 0
    for lvl in hierarchy.levels
        @test lvl.R_map !== nothing
    end
end

@testset "AMG Resetup - partial=false" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    config = AMGConfig(coarsening=AggregationCoarsening())
    hierarchy = amg_setup(A, config)
    # Solve with original matrix
    b = rand(N)
    x1 = zeros(N)
    x1, _ = amg_solve!(x1, b, hierarchy, config; tol=1e-8, maxiter=200)
    r1 = b - sparse(A.At') * x1
    @test norm(r1) / norm(b) < 1e-8
    # Scale matrix and do full resetup
    nonzeros(A) .*= 2.0
    amg_resetup!(hierarchy, A, config; partial=false)
    x2 = zeros(N)
    x2, _ = amg_solve!(x2, b, hierarchy, config; tol=1e-8, maxiter=200)
    r2 = b - sparse(A.At') * x2
    @test norm(r2) / norm(b) < 1e-8
    @test !isapprox(x1, x2, atol=1e-6)
end

@testset "AMG Resetup - partial=false with allow_partial_resetup" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    config_no_partial = AMGConfig(coarsening=AggregationCoarsening(), allow_partial_resetup=false)
    hierarchy = amg_setup(A, config_no_partial)
    # R_map should be nothing
    for lvl in hierarchy.levels
        @test lvl.R_map === nothing
    end
    # Full resetup with allow_partial_resetup=true config should populate R_map
    config_with_partial = AMGConfig(coarsening=AggregationCoarsening(), allow_partial_resetup=true)
    amg_resetup!(hierarchy, A, config_with_partial; partial=false)
    for lvl in hierarchy.levels
        @test lvl.R_map !== nothing
    end
    # Partial resetup should now work
    nonzeros(A) .*= 2.0
    amg_resetup!(hierarchy, A, config_with_partial; partial=true)
    b = rand(N)
    x = zeros(N)
    x, niter = amg_solve!(x, b, hierarchy, config_with_partial; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
end

@testset "AMG Resetup - partial=false, allow_partial_resetup=false" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    config = AMGConfig(coarsening=AggregationCoarsening())
    hierarchy = amg_setup(A, config)
    # Full resetup without restriction maps
    config_no_partial = AMGConfig(coarsening=AggregationCoarsening(), allow_partial_resetup=false)
    amg_resetup!(hierarchy, A, config_no_partial; partial=false)
    for lvl in hierarchy.levels
        @test lvl.R_map === nothing
    end
    b = rand(N)
    x = zeros(N)
    x, niter = amg_solve!(x, b, hierarchy, config_no_partial; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
end

@testset "AMG Resetup - update_P=true with HMIS coarsening" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    config = AMGConfig(coarsening=HMISCoarsening(0.5, DirectInterpolation()))
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) > 0
    # P_update_map should be present for CF-splitting methods
    for lvl in hierarchy.levels
        @test lvl.P_update_map !== nothing
    end
    # Solve with original matrix
    b = rand(N)
    x1 = zeros(N)
    x1, niter1 = amg_solve!(x1, b, hierarchy, config; tol=1e-8, maxiter=200)
    r1 = b - sparse(A.At') * x1
    @test norm(r1) / norm(b) < 1e-8
    # Scale matrix and resetup with update_P=true
    nonzeros(A) .*= 2.0
    amg_resetup!(hierarchy, A, config; partial=true, update_P=true)
    # Solve with updated matrix
    x2 = zeros(N)
    x2, niter2 = amg_solve!(x2, b, hierarchy, config; tol=1e-8, maxiter=200)
    r2 = b - sparse(A.At') * x2
    @test norm(r2) / norm(b) < 1e-8
end

@testset "AMG Resetup - update_P=true with PMIS coarsening" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    config = AMGConfig(coarsening=PMISCoarsening(0.5, DirectInterpolation()))
    hierarchy = amg_setup(A, config)
    if length(hierarchy.levels) > 0
        # P_update_map should be present
        for lvl in hierarchy.levels
            @test lvl.P_update_map !== nothing
        end
        # Scale and resetup with update_P=true
        nonzeros(A) .*= 2.0
        amg_resetup!(hierarchy, A, config; partial=true, update_P=true)
        b = rand(N)
        x = zeros(N)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
    end
end

@testset "AMG Resetup - update_P=true with RS coarsening" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    config = AMGConfig(coarsening=RSCoarsening(0.5, DirectInterpolation()))
    hierarchy = amg_setup(A, config)
    if length(hierarchy.levels) > 0
        # P_update_map should be present
        for lvl in hierarchy.levels
            @test lvl.P_update_map !== nothing
        end
        # Scale and resetup with update_P=true
        nonzeros(A) .*= 2.0
        amg_resetup!(hierarchy, A, config; partial=true, update_P=true)
        b = rand(N)
        x = zeros(N)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
    end
end

@testset "AMG Resetup - update_P=true with StandardInterpolation" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    config = AMGConfig(coarsening=HMISCoarsening(0.5, StandardInterpolation()))
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) > 0
    # Standard interpolation now supports update_P (interp_type=2)
    for lvl in hierarchy.levels
        @test lvl.P_update_map !== nothing
        @test lvl.P_update_map.interp_type == 2
    end
    nonzeros(A) .*= 2.0
    amg_resetup!(hierarchy, A, config; partial=true, update_P=true)
    b = rand(N)
    x = zeros(N)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
end

@testset "AMG Resetup - update_P=true with ExtendedIInterpolation" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    config = AMGConfig(coarsening=HMISCoarsening(0.5, ExtendedIInterpolation()))
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) > 0
    # Extended+i interpolation now supports update_P (interp_type=3)
    for lvl in hierarchy.levels
        @test lvl.P_update_map !== nothing
        @test lvl.P_update_map.interp_type == 3
    end
    nonzeros(A) .*= 2.0
    amg_resetup!(hierarchy, A, config; partial=true, update_P=true)
    b = rand(N)
    x = zeros(N)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
end

@testset "AMG Resetup - update_P does not impact Aggregation" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    config = AMGConfig(coarsening=AggregationCoarsening())
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) > 0
    # P_update_map should be nothing for aggregation-based methods
    for lvl in hierarchy.levels
        @test lvl.P_update_map === nothing
    end
    # update_P=true should still work (just does nothing for aggregation)
    nonzeros(A) .*= 2.0
    amg_resetup!(hierarchy, A, config; partial=true, update_P=true)
    b = rand(N)
    x = zeros(N)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
end

@testset "Small System Direct Solve" begin
    # System small enough to be solved directly
    A = poisson1d_csr(5)
    config = AMGConfig(max_coarse_size=10)  # force direct solve
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) == 0
    b = ones(5)
    x = zeros(5)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-10)
    r = b - sparse(A.At') * x
    @test norm(r) < 1e-10
end

@testset "In-place LU Refactorization" begin
    n = 8
    A = poisson2d_csr(n)
    N = n*n
    config = AMGConfig(coarsening=AggregationCoarsening())
    hierarchy = amg_setup(A, config)
    # Check that coarse_factor is a valid factorization
    @test hierarchy.coarse_factor isa Factorization
    # Solve, then resetup with scaled matrix and solve again
    b = rand(N)
    x1 = zeros(N)
    x1, _ = amg_solve!(x1, b, hierarchy, config; tol=1e-8, maxiter=200)
    r1 = b - sparse(A.At') * x1
    @test norm(r1) / norm(b) < 1e-8
    # Resetup
    nonzeros(A) .*= 2.0
    amg_resetup!(hierarchy, A, config)
    # Verify factorization was updated
    @test hierarchy.coarse_factor isa Factorization
    # Solve with updated matrix
    x2 = zeros(N)
    x2, _ = amg_solve!(x2, b, hierarchy, config; tol=1e-8, maxiter=200)
    r2 = b - sparse(A.At') * x2
    @test norm(r2) / norm(b) < 1e-8
end

@testset "Verbose Output" begin
    A = poisson2d_csr(8)
    N = 64
    b = rand(N)
    x = zeros(N)
    config = AMGConfig(verbose=true)
    # Capture stdout using mktemp
    output = mktempdir() do dir
        path = joinpath(dir, "out.txt")
        open(path, "w") do f
            redirect_stdout(f) do
                hierarchy = amg_setup(A, config)
                x, _ = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=100)
            end
        end
        read(path, String)
    end
    @test contains(output, "AMG Hierarchy Summary")
    @test contains(output, "Operator complexity")
    @test contains(output, "AMG solve converged")
    @test contains(output, "Backend")
    @test contains(output, "Block size")
end

# ══════════════════════════════════════════════════════════════════════════
# W-Cycle
# ══════════════════════════════════════════════════════════════════════════

@testset "W-Cycle Config" begin
    config = AMGConfig(cycle_type=:W)
    @test config.cycle_type == :W
    config_v = AMGConfig()
    @test config_v.cycle_type == :V
end

@testset "W-Cycle Solve" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    b = rand(N)

    # V-cycle baseline
    x_v = zeros(N)
    config_v = AMGConfig(cycle_type=:V, pre_smoothing_steps=2, post_smoothing_steps=2)
    h = amg_setup(A, config_v)
    x_v, niter_v = amg_solve!(x_v, b, h, config_v; tol=1e-8, maxiter=200)
    r_v = b - sparse(A.At') * x_v
    @test norm(r_v) / norm(b) < 1e-8

    # W-cycle should also converge
    x_w = zeros(N)
    config_w = AMGConfig(cycle_type=:W, pre_smoothing_steps=2, post_smoothing_steps=2)
    h_w = amg_setup(A, config_w)
    x_w, niter_w = amg_solve!(x_w, b, h_w, config_w; tol=1e-8, maxiter=200)
    r_w = b - sparse(A.At') * x_w
    @test norm(r_w) / norm(b) < 1e-8
    # W-cycle should converge in fewer or equal iterations than V-cycle
    @test niter_w <= niter_v
end

# ══════════════════════════════════════════════════════════════════════════
# Initial Coarsening Configuration
# ══════════════════════════════════════════════════════════════════════════

@testset "Initial Coarsening - Default" begin
    config = AMGConfig(coarsening=AggregationCoarsening())
    # Default: initial_coarsening == coarsening, initial_coarsening_levels == 0
    @test config.initial_coarsening isa AggregationCoarsening
    @test config.initial_coarsening_levels == 0
    # _get_coarsening_for_level always returns main coarsening when levels=0
    @test Draugr._get_coarsening_for_level(config, 1) isa AggregationCoarsening
    @test Draugr._get_coarsening_for_level(config, 5) isa AggregationCoarsening
end

@testset "Initial Coarsening - Custom" begin
    config = AMGConfig(
        coarsening=AggregationCoarsening(),
        initial_coarsening=PMISCoarsening(0.25, DirectInterpolation()),
        initial_coarsening_levels=2,
    )
    @test config.initial_coarsening isa PMISCoarsening
    @test config.initial_coarsening_levels == 2
    # Levels 1-2 use initial_coarsening, level 3+ uses main
    @test Draugr._get_coarsening_for_level(config, 1) isa PMISCoarsening
    @test Draugr._get_coarsening_for_level(config, 2) isa PMISCoarsening
    @test Draugr._get_coarsening_for_level(config, 3) isa AggregationCoarsening
end

@testset "Initial Coarsening - Solve" begin
    # Use aggressive coarsening for first 2 levels, then aggregation
    config = AMGConfig(
        coarsening=AggregationCoarsening(),
        initial_coarsening=PMISCoarsening(0.25, DirectInterpolation()),
        initial_coarsening_levels=1,
        pre_smoothing_steps=2,
        post_smoothing_steps=2,
    )
    hierarchy, = test_amg_convergence(config)
    @test length(hierarchy.levels) > 0
end

# ══════════════════════════════════════════════════════════════════════════
# Interpolation type tests
# ══════════════════════════════════════════════════════════════════════════

@testset "AMG Solve - HMIS Direct" begin
    test_amg_convergence(
        AMGConfig(coarsening=HMISCoarsening(0.25, DirectInterpolation()),
                  pre_smoothing_steps=2, post_smoothing_steps=2))
end

@testset "PMIS - Direct Interpolation" begin
    test_amg_convergence(
        AMGConfig(coarsening=PMISCoarsening(0.25, DirectInterpolation()),
                  pre_smoothing_steps=2, post_smoothing_steps=2))
end

@testset "PMIS - Standard Interpolation" begin
    test_amg_convergence(
        AMGConfig(coarsening=PMISCoarsening(0.25, StandardInterpolation()),
                  pre_smoothing_steps=2, post_smoothing_steps=2))
end

@testset "PMIS - Extended+i Interpolation" begin
    test_amg_convergence(
        AMGConfig(coarsening=PMISCoarsening(0.25, ExtendedIInterpolation()),
                  pre_smoothing_steps=2, post_smoothing_steps=2))
end

@testset "HMIS - Standard Interpolation" begin
    test_amg_convergence(
        AMGConfig(coarsening=HMISCoarsening(0.25, StandardInterpolation()),
                  pre_smoothing_steps=2, post_smoothing_steps=2))
end

@testset "HMIS - Extended+i Interpolation" begin
    test_amg_convergence(
        AMGConfig(coarsening=HMISCoarsening(0.25, ExtendedIInterpolation()),
                  pre_smoothing_steps=2, post_smoothing_steps=2))
end

@testset "RS Coarsening - Solve" begin
    test_amg_convergence(
        AMGConfig(coarsening=RSCoarsening(0.25, DirectInterpolation()),
                  pre_smoothing_steps=2, post_smoothing_steps=2))
end

@testset "RS Coarsening - Standard Interpolation" begin
    test_amg_convergence(
        AMGConfig(coarsening=RSCoarsening(0.25, StandardInterpolation()),
                  pre_smoothing_steps=2, post_smoothing_steps=2))
end

@testset "ExtendedI Convergence - PMIS" begin
    test_amg_convergence(
        AMGConfig(coarsening=PMISCoarsening(0.25, ExtendedIInterpolation()),
                  pre_smoothing_steps=2, post_smoothing_steps=2);
        n=15)
end

@testset "ExtendedI Convergence - HMIS" begin
    test_amg_convergence(
        AMGConfig(coarsening=HMISCoarsening(0.25, ExtendedIInterpolation()),
                  pre_smoothing_steps=2, post_smoothing_steps=2);
        n=15)
end

# ══════════════════════════════════════════════════════════════════════════
# Robustness tests
# ══════════════════════════════════════════════════════════════════════════

@testset "Isolated diagonal-only rows" begin
    # Matrix with disconnected diagonal-only rows
    I = [1,1,2,3,3,4,5]
    J = [1,2,2,3,4,4,5]
    V = [4.0,-1.0,5.0,-1.0,4.0,-1.0,3.0]  # rows 2 and 5 are diagonal-only
    A = static_sparsity_sparse(I, J, V, 5, 5)
    config = AMGConfig()
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) >= 0  # might go straight to direct solve
    b = [1.0, 2.0, 3.0, 4.0, 5.0]
    x = zeros(5)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-10, maxiter=100)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
end

@testset "Block diagonal matrix" begin
    # Two disconnected 2x2 blocks + one isolated node
    I = [1,1,2,2,3,3,4,4,5]
    J = [1,2,1,2,3,4,3,4,5]
    V = [4.0,-1.0,-1.0,4.0,4.0,-1.0,-1.0,4.0,3.0]
    A = static_sparsity_sparse(I, J, V, 5, 5)
    config = AMGConfig()
    hierarchy = amg_setup(A, config)
    b = [1.0, 1.0, 1.0, 1.0, 1.0]
    x = zeros(5)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-10, maxiter=100)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
end

@testset "Isolated nodes in PMIS/HMIS" begin
    # Matrix with some nodes having no strong connections (small off-diags)
    n = 10
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:n
        push!(I, i); push!(J, i); push!(V, 100.0)  # strong diagonal
        if i > 1
            # Very weak connection (won't be strong)
            push!(I, i); push!(J, i-1); push!(V, -1e-10)
        end
        if i < n
            push!(I, i); push!(J, i+1); push!(V, -1e-10)
        end
    end
    A = static_sparsity_sparse(I, J, V, n, n)
    # All off-diags are negligible → all nodes isolated
    for coarsening_alg in [PMISCoarsening(), HMISCoarsening(), RSCoarsening()]
        config = AMGConfig(coarsening=coarsening_alg)
        hierarchy = amg_setup(A, config)
        b = rand(n)
        x = zeros(n)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
    end
end

@testset "Safe Diagonal - Zero diagonal row" begin
    # Matrix with near-zero diagonal in one row
    I = [1,1,2,2,2,3,3]
    J = [1,2,1,2,3,2,3]
    V = [2.0,-1.0,-1.0,1e-20,-1.0,-1.0,2.0]  # row 2 has near-zero diagonal
    A = static_sparsity_sparse(I, J, V, 3, 3)
    Ac = to_csr(A)
    smoother = Draugr.build_jacobi_smoother(Ac, 2.0/3.0)
    # invdiag should be safe (zero, not Inf)
    @test isfinite(smoother.invdiag[1])
    @test isfinite(smoother.invdiag[2])
    @test isfinite(smoother.invdiag[3])
end

@testset "Small/Trivial Systems" begin
    # 1x1 system
    A1 = static_sparsity_sparse([1], [1], [5.0], 1, 1)
    config = AMGConfig(max_coarse_size=10)
    h = amg_setup(A1, config)
    x = zeros(1)
    x, niter = amg_solve!(x, [3.0], h, config; tol=1e-10)
    @test x[1] ≈ 0.6 atol=1e-8

    # 2x2 system
    A2 = static_sparsity_sparse([1,1,2,2], [1,2,1,2], [4.0,-1.0,-1.0,4.0], 2, 2)
    config2 = AMGConfig(max_coarse_size=10)
    h2 = amg_setup(A2, config2)
    b = [1.0, 1.0]
    x = zeros(2)
    x, niter = amg_solve!(x, b, h2, config2; tol=1e-10)
    @test norm(b - sparse(A2.At') * x) / norm(b) < 1e-10
end

@testset "SignedStrength - Solve" begin
    A = reservoir_like_csr(50)
    N = 50
    b = rand(N)
    x = zeros(N)
    config = AMGConfig(strength_type=SignedStrength())
    hierarchy = amg_setup(A, config)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
    @test niter < 200
end

@testset "Positive Off-Diags - All smoothers converge" begin
    A = reservoir_like_csr(50)
    N = 50
    b = rand(N)
    for (name, cfg) in [
        ("Jacobi", AMGConfig()),
        ("l1-Jacobi", AMGConfig(smoother=L1JacobiSmootherType())),
        ("Colored GS", AMGConfig(smoother=ColoredGaussSeidelType())),
        ("l1-Colored GS", AMGConfig(smoother=L1ColoredGaussSeidelType())),
        ("l1-Serial GS", AMGConfig(smoother=L1SerialGaussSeidelType())),
        ("SPAI0", AMGConfig(smoother=SPAI0SmootherType())),
        ("ILU0", AMGConfig(smoother=ILU0SmootherType())),
    ]
        x = zeros(N)
        hierarchy = amg_setup(A, cfg)
        x, niter = amg_solve!(x, b, hierarchy, cfg; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
    end
end

@testset "Anisotropic Matrix - Convergence" begin
    A = anisotropic_csr(8, 8)
    N = 64
    b = rand(N)
    for (name, cfg) in [
        ("Default", AMGConfig(pre_smoothing_steps=2, post_smoothing_steps=2)),
        ("l1-Jacobi", AMGConfig(smoother=L1JacobiSmootherType(), pre_smoothing_steps=2, post_smoothing_steps=2)),
        ("ILU0", AMGConfig(smoother=ILU0SmootherType())),
    ]
        x = zeros(N)
        hierarchy = amg_setup(A, cfg)
        x, niter = amg_solve!(x, b, hierarchy, cfg; tol=1e-8, maxiter=300)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
    end
end

@testset "Block-aware helpers" begin
    # Test _frobenius_norm2 for scalars
    @test Draugr._frobenius_norm2(3.0) ≈ 9.0
    @test Draugr._frobenius_norm2(-2.0) ≈ 4.0
    # Test _entry_norm for scalars
    @test Draugr._entry_norm(3.0) ≈ 3.0
    @test Draugr._entry_norm(-2.0) ≈ 2.0
    # Test _is_finite_entry for scalars
    @test Draugr._is_finite_entry(1.0) == true
    @test Draugr._is_finite_entry(Inf) == false
    @test Draugr._is_finite_entry(NaN) == false
    # Test block-aware helpers with small matrices (simulating SMatrix behavior)
    M = [1.0 2.0; 3.0 4.0]
    @test Draugr._frobenius_norm2(M) ≈ 1.0 + 4.0 + 9.0 + 16.0  # sum of squares
    @test Draugr._entry_norm(M) ≈ sqrt(30.0)
    @test Draugr._is_finite_entry(M) == true
    M_inf = [1.0 Inf; 0.0 1.0]
    @test Draugr._is_finite_entry(M_inf) == false
end

# ══════════════════════════════════════════════════════════════════════════
# Smoothed Aggregation
# ══════════════════════════════════════════════════════════════════════════

@testset "Smoothed Aggregation - Config" begin
    alg = SmoothedAggregationCoarsening()
    @test alg.θ ≈ 0.25
    @test alg.ω ≈ 2/3
    @test alg.filtering == false
    alg2 = SmoothedAggregationCoarsening(0.3, 0.5)
    @test alg2.θ ≈ 0.3
    @test alg2.ω ≈ 0.5
end

@testset "Smoothed Aggregation - Setup" begin
    A = poisson2d_csr(10)
    config = AMGConfig(coarsening=SmoothedAggregationCoarsening())
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) >= 1
    # SA produces denser coarse matrices than plain aggregation
    @test nnz(hierarchy.levels[1].A) > 0
end

@testset "Smoothed Aggregation - Solve" begin
    test_amg_convergence(
        AMGConfig(coarsening=SmoothedAggregationCoarsening(),
                  pre_smoothing_steps=2, post_smoothing_steps=2))
end

@testset "Smoothed Aggregation - Better than Plain Aggregation" begin
    n = 15
    A = poisson2d_csr(n)
    N = n*n
    b = rand(N)

    # Plain aggregation
    x1 = zeros(N)
    config1 = AMGConfig(coarsening=AggregationCoarsening(),
                        pre_smoothing_steps=2, post_smoothing_steps=2)
    hierarchy1 = amg_setup(A, config1)
    x1, niter1 = amg_solve!(x1, b, hierarchy1, config1; tol=1e-8, maxiter=200)

    # Smoothed aggregation
    x2 = zeros(N)
    config2 = AMGConfig(coarsening=SmoothedAggregationCoarsening(),
                        pre_smoothing_steps=2, post_smoothing_steps=2)
    hierarchy2 = amg_setup(A, config2)
    x2, niter2 = amg_solve!(x2, b, hierarchy2, config2; tol=1e-8, maxiter=200)

    # SA should converge in fewer iterations
    @test niter2 <= niter1
end

@testset "Smoothed Aggregation - With Filtering" begin
    test_amg_convergence(
        AMGConfig(coarsening=SmoothedAggregationCoarsening(0.25, 2/3, true, 0.1),
                  pre_smoothing_steps=2, post_smoothing_steps=2))
end

@testset "Smoothed Aggregation - Resetup" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    b = rand(N)
    config = AMGConfig(coarsening=SmoothedAggregationCoarsening(),
                       pre_smoothing_steps=2, post_smoothing_steps=2)
    hierarchy = amg_setup(A, config)
    x1 = zeros(N)
    x1, niter1 = amg_solve!(x1, b, hierarchy, config; tol=1e-8, maxiter=200)
    @test niter1 < 200
    # Resetup with modified coefficients
    nonzeros(A) .*= 2.0
    amg_resetup!(hierarchy, A, config)
    x2 = zeros(N)
    x2, niter2 = amg_solve!(x2, b, hierarchy, config; tol=1e-8, maxiter=200)
    r2 = b - sparse(A.At') * x2
    @test norm(r2) / norm(b) < 1e-8
end

# ══════════════════════════════════════════════════════════════════════════
# Aggregation Filtering configuration
# ══════════════════════════════════════════════════════════════════════════

@testset "Aggregation Filtering - Config" begin
    alg = AggregationCoarsening(0.25, true, 0.2)
    @test alg.filtering == true
    @test alg.filter_tol ≈ 0.2
    alg2 = AggregationCoarsening()
    @test alg2.filtering == false
end

@testset "Aggregation Filtering - Solve" begin
    test_amg_convergence(AMGConfig(coarsening=AggregationCoarsening(0.25, true, 0.1)))
end

# ══════════════════════════════════════════════════════════════════════════
# Max Row Sum Threshold
# ══════════════════════════════════════════════════════════════════════════

@testset "Max Row Sum - Config" begin
    config = AMGConfig(max_row_sum=0.9)
    @test config.max_row_sum ≈ 0.9
    config2 = AMGConfig()
    @test config2.max_row_sum ≈ 1.0  # disabled by default
end

@testset "Max Row Sum - Weakening Function" begin
    A = poisson2d_csr(5)
    Ac = to_csr(A)
    # For Poisson 2D (5-point stencil):
    #   diagonal = 4, off-diagonals = -1
    #   Corner row (2 neighbors): row_sum = 4 - 2 = 2, |row_sum|/|diag| = 0.5
    #   Edge row (3 neighbors): row_sum = 4 - 3 = 1, |row_sum|/|diag| = 0.25
    #   Interior row (4 neighbors): row_sum = 4 - 4 = 0, |row_sum|/|diag| = 0.0
    # With threshold=0.3, corner rows (ratio 0.5 > 0.3) should be zeroed
    # but edge rows (ratio 0.25 < 0.3) should not
    A_weak = Draugr._apply_max_row_sum(Ac, 0.3)
    # The weakened matrix should have same size and structure
    @test size(A_weak) == size(Ac)
    @test nnz(A_weak) == nnz(Ac)
    cv = colvals(Ac)
    nzv_orig = nonzeros(Ac)
    nzv_weak = nonzeros(A_weak)
    rp = rowptr(Ac)
    # Row 1 (corner, 2 neighbors): |row_sum|/|diag| = 0.5 > 0.3, all off-diag should be zeroed
    for nz in rp[1]:(rp[1+1]-1)
        j = cv[nz]
        if j != 1
            @test abs(nzv_weak[nz]) < 1e-14
        end
    end
    # Interior row 13 (4 neighbors): |row_sum|/|diag| = 0.0 < 0.3, should NOT be affected
    row13_unchanged = true
    for nz in rp[13]:(rp[13+1]-1)
        if abs(nzv_weak[nz] - nzv_orig[nz]) > 1e-14
            row13_unchanged = false
        end
    end
    @test row13_unchanged
end

@testset "Max Row Sum - Solve" begin
    test_amg_convergence(AMGConfig(max_row_sum=0.9))
end

# ══════════════════════════════════════════════════════════════════════════
# HYPRE-equivalent configuration tests
# ══════════════════════════════════════════════════════════════════════════

@testset "hypre_default_config construction" begin
    config = hypre_default_config()
    @test config.coarsening isa HMISCoarsening
    @test config.coarsening.θ == 0.5
    @test config.coarsening.interpolation isa ExtendedIInterpolation
    @test config.coarsening.interpolation.trunc_factor == 0.3
    @test config.initial_coarsening isa AggressiveCoarsening
    @test config.initial_coarsening.θ == 0.5
    @test config.initial_coarsening.base == :hmis
    @test config.initial_coarsening.interpolation isa ExtendedIInterpolation
    @test config.initial_coarsening.interpolation.trunc_factor == 0.3
    @test config.initial_coarsening_levels == 1
end

@testset "hypre_default_config with custom params" begin
    config = hypre_default_config(θ=0.3, agg_num_levels=2, agg_trunc_factor=0.5,
                                   verbose=false, smoother=ColoredGaussSeidelType())
    @test config.coarsening.θ == 0.3
    @test config.initial_coarsening.θ == 0.3
    @test config.initial_coarsening_levels == 2
    @test config.coarsening.interpolation.trunc_factor == 0.5
    @test config.smoother isa ColoredGaussSeidelType
end

@testset "hypre_default_config solve - 2D Poisson" begin
    n = 12
    A = poisson2d_csr(n)
    N = n * n
    b = rand(N)
    x = zeros(N)
    config = hypre_default_config(verbose=false)
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) >= 1
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-6
    @test niter < 200
end

@testset "Interpolation trunc_factor" begin
    # Test that trunc_factor fields work correctly
    d = DirectInterpolation(0.3)
    @test d.trunc_factor == 0.3
    d0 = DirectInterpolation()
    @test d0.trunc_factor == 0.0

    s = StandardInterpolation(0.5)
    @test s.trunc_factor == 0.5
    s0 = StandardInterpolation()
    @test s0.trunc_factor == 0.0

    e = ExtendedIInterpolation(0.3)
    @test e.trunc_factor == 0.3
    e0 = ExtendedIInterpolation()
    @test e0.trunc_factor == 0.0
end

@testset "AggressiveCoarsening with HMIS base" begin
    config = AMGConfig(
        coarsening = HMISCoarsening(0.5, ExtendedIInterpolation()),
        initial_coarsening = AggressiveCoarsening(0.5, :hmis, ExtendedIInterpolation(0.3)),
        initial_coarsening_levels = 1,
        pre_smoothing_steps = 2,
        post_smoothing_steps = 2,
    )
    test_amg_convergence(config; tol=1e-6)
end

@testset "AggressiveCoarsening with PMIS base" begin
    config = AMGConfig(
        coarsening = PMISCoarsening(0.5, ExtendedIInterpolation()),
        initial_coarsening = AggressiveCoarsening(0.5, :pmis, ExtendedIInterpolation(0.3)),
        initial_coarsening_levels = 1,
        pre_smoothing_steps = 2,
        post_smoothing_steps = 2,
    )
    test_amg_convergence(config; tol=1e-6)
end

@testset "Consistent θ in interpolation" begin
    # Verify that the coarsening's θ is passed to interpolation (not hardcoded 0.25)
    n = 8
    A = poisson2d_csr(n)
    # Use θ=0.5 with HMIS + Direct: should still converge
    config05 = AMGConfig(coarsening=HMISCoarsening(0.5, DirectInterpolation()),
                         pre_smoothing_steps=2, post_smoothing_steps=2)
    h05 = amg_setup(A, config05)
    @test length(h05.levels) >= 1
    x = zeros(n*n)
    b = rand(n*n)
    x, niter = amg_solve!(x, b, h05, config05; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-6
end

@testset "HMIS with ExtendedI and trunc_factor" begin
    test_amg_convergence(
        AMGConfig(coarsening=HMISCoarsening(0.5, ExtendedIInterpolation(0.3)),
                  pre_smoothing_steps=2, post_smoothing_steps=2);
        tol=1e-6)
end

@testset "ExtendedI max_elements" begin
    # Test that max_elements limits the number of interpolation points per row
    @test ExtendedIInterpolation().max_elements == 0
    @test ExtendedIInterpolation(0.3).max_elements == 0
    @test ExtendedIInterpolation(0.0, 8).max_elements == 8
    # Test solving with different max_elements values
    n = 10
    A = poisson2d_csr(n)
    N = n * n
    b = rand(N)
    for max_elems in [2, 4, 8, 0]
        x = zeros(N)
        config = AMGConfig(
            coarsening = HMISCoarsening(0.5, ExtendedIInterpolation(0.0, max_elems)),
            pre_smoothing_steps = 2,
            post_smoothing_steps = 2,
        )
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) >= 1
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-6, maxiter=300)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-6
    end
end

@testset "ExtendedI norm_p and rescale fields" begin
    # Default constructor
    e = ExtendedIInterpolation()
    @test e.norm_p == 1
    @test e.rescale == false
    # Single-arg constructor
    e1 = ExtendedIInterpolation(0.3)
    @test e1.norm_p == 1
    @test e1.rescale == false
    # Two-arg constructor
    e2 = ExtendedIInterpolation(0.3, 4)
    @test e2.norm_p == 1
    @test e2.rescale == false
    # Full constructor
    e3 = ExtendedIInterpolation(0.3, 4, 2, true)
    @test e3.trunc_factor == 0.3
    @test e3.max_elements == 4
    @test e3.norm_p == 2
    @test e3.rescale == true
end

@testset "ExtendedI norm_p truncation" begin
    # Verify that norm_p changes truncation behavior and solver converges
    n = 10
    A = poisson2d_csr(n)
    N = n * n
    b = rand(N)
    for np in [1, 2]
        x = zeros(N)
        config = AMGConfig(
            coarsening = HMISCoarsening(0.5, ExtendedIInterpolation(0.3, 0, np, false)),
            pre_smoothing_steps = 2,
            post_smoothing_steps = 2,
        )
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) >= 1
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-6, maxiter=300)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-6
    end
end

@testset "ExtendedI rescale" begin
    # Verify rescaling option works and solver converges
    n = 10
    A = poisson2d_csr(n)
    N = n * n
    b = rand(N)
    for do_rescale in [false, true]
        x = zeros(N)
        config = AMGConfig(
            coarsening = HMISCoarsening(0.5, ExtendedIInterpolation(0.3, 0, 1, do_rescale)),
            pre_smoothing_steps = 2,
            post_smoothing_steps = 2,
        )
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) >= 1
        # When rescale is true and truncation occurs, trunc_scaling should be stored
        if do_rescale
            for lvl in hierarchy.levels
                P = lvl.P
                @test P.trunc_scaling !== nothing
            end
        else
            for lvl in hierarchy.levels
                P = lvl.P
                @test P.trunc_scaling === nothing
            end
        end
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-6, maxiter=300)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-6
    end
end

# ══════════════════════════════════════════════════════════════════════════
# HYPRE comparison tests
# ══════════════════════════════════════════════════════════════════════════

@testset "HYPRE comparison - hierarchy structure" begin
    using Krylov
    using HYPRE: BoomerAMGPreconditioner

    N = 100
    A = poisson2d_csr(N, N)
    n = N * N
    b = ones(n)

    # Draugr setup matching HYPRE defaults (no aggressive coarsening, no truncation)
    coarsen = HMISCoarsening(0.5, ExtendedIInterpolation(0.0, 4, 2, true))
    config = AMGConfig(coarsening=coarsen,
        smoother = SerialGaussSeidelType(),
        verbose = false,
        max_row_sum = 0.9,
        strength_type = SignedStrength()
    )
    hierarchy = amg_setup(A, config)

    # Level 0 → Level 1 coarsening should be ~0.5 for 2D 5-point Poisson
    ratio_0 = hierarchy.levels[2].A.nrow / hierarchy.levels[1].A.nrow
    @test ratio_0 ≈ 0.5 atol=0.05

    # Level 1 → Level 2 should also be ~0.5 (not 0.25!)
    # This is the key test: the strength comparison (strict >) ensures that
    # the 9-point Galerkin product at Level 1 has only 4 strong connections
    # (not 8), giving 0.5 coarsening ratio matching HYPRE
    if length(hierarchy.levels) >= 3
        ratio_1 = hierarchy.levels[3].A.nrow / hierarchy.levels[2].A.nrow
        @test ratio_1 ≈ 0.5 atol=0.05
    end
end

@testset "HYPRE comparison - GMRES iteration count" begin
    using Krylov
    using HYPRE: BoomerAMGPreconditioner

    N = 100
    A = poisson2d_csr(N, N)
    n = N * N
    b = ones(n)

    # HYPRE solve
    prec_hypre = BoomerAMGPreconditioner(PrintLevel = 0, AggNumLevels = 0, AggTruncFactor = 0.0)
    Jutul.update_preconditioner!(prec_hypre, A, b, missing, missing)
    op_hypre = Jutul.linear_operator(prec_hypre)
    _, stats_h = gmres(A, b; M = op_hypre, rtol = 1e-8, itmax=100)

    # Draugr solve
    coarsen = HMISCoarsening(0.5, ExtendedIInterpolation(0.0, 4, 2, true))
    config = AMGConfig(coarsening=coarsen,
        smoother = SerialGaussSeidelType(),
        verbose = false,
        max_row_sum = 0.9,
        strength_type = SignedStrength()
    )
    hierarchy = amg_setup(A, config)
    M = DraugrPreconditioner(config, hierarchy, size(A))
    _, stats_d = gmres(A, b; M = M, rtol = 1e-8, itmax=100, ldiv = true)

    # Draugr should be within 3x of HYPRE's iteration count
    @test stats_d.niter <= 3 * stats_h.niter
    # Both should converge
    @test stats_h.solved
    @test stats_d.solved
end
