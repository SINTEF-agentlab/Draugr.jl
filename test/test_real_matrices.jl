@testset "Real Matrices - AMG Setup and Level Count" begin
    TEST_MATRICES_DIR = joinpath(@__DIR__, "test_matrices")
    for name in ["egg", "norne", "olympus_1", "spe10_tarbert", "spe10_ness", "spe10_full"]
        @testset "$name" begin
            A = read_mtx(joinpath(TEST_MATRICES_DIR, "$name.mtx"))
            config = AMGConfig()
            hierarchy = amg_setup(A, config)
            nlevels = length(hierarchy.levels)
            @test nlevels > 0
            @test nlevels <= 25
            # Levels must be strictly coarser from fine to coarse
            for i in 1:(nlevels - 1)
                @test size(hierarchy.levels[i + 1].A, 1) < size(hierarchy.levels[i].A, 1)
            end
        end
    end
end

@testset "Real Matrices - Level Counts per Coarsening" begin
    TEST_MATRICES_DIR = joinpath(@__DIR__, "test_matrices")
    for matname in ["egg", "norne"]
        A = read_mtx(joinpath(TEST_MATRICES_DIR, "$matname.mtx"))
        @testset "$matname" begin
            for coarsening in [
                    AggregationCoarsening(),
                    PMISCoarsening(),
                    HMISCoarsening(),
                    RSCoarsening(),
                    AggressiveCoarsening(),
                    SmoothedAggregationCoarsening(),
                ]
                @testset "$(typeof(coarsening).name.name)" begin
                    config = AMGConfig(coarsening=coarsening)
                    hierarchy = amg_setup(A, config)
                    nlevels = length(hierarchy.levels)
                    @test nlevels > 0
                    @test nlevels <= 25
                    # Each coarse level must be strictly smaller
                    for i in 1:(nlevels - 1)
                        @test size(hierarchy.levels[i + 1].A, 1) < size(hierarchy.levels[i].A, 1)
                    end
                end
            end
        end
    end
end

@testset "Real Matrices - Solve with Different Coarsenings (egg)" begin
    TEST_MATRICES_DIR = joinpath(@__DIR__, "test_matrices")
    A = read_mtx(joinpath(TEST_MATRICES_DIR, "egg.mtx"))
    N = size(A, 1)
    b = ones(N)
    for (cname, coarsening) in [
            ("Aggregation",          AggregationCoarsening()),
            ("PMIS",                 PMISCoarsening()),
            ("HMIS",                 HMISCoarsening()),
            ("RS",                   RSCoarsening()),
            ("Aggressive",           AggressiveCoarsening()),
            ("SmoothedAggregation",  SmoothedAggregationCoarsening()),
        ]
        @testset "$cname" begin
            config = AMGConfig(coarsening=coarsening)
            hierarchy = amg_setup(A, config)
            @test length(hierarchy.levels) > 0
            x = zeros(N)
            x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-6, maxiter=500)
            r = b - A * x
            @test norm(r) / norm(b) < 1e-6
            @test niter < 500
        end
    end
end

@testset "Real Matrices - Resetup Options (egg)" begin
    TEST_MATRICES_DIR = joinpath(@__DIR__, "test_matrices")
    A = read_mtx(joinpath(TEST_MATRICES_DIR, "egg.mtx"))
    N = size(A, 1)
    b = ones(N)

    @testset "Partial resetup (R_map, coefficient-only)" begin
        config = AMGConfig()
        hierarchy = amg_setup(A, config)
        # Verify R_map is present by default
        for lvl in hierarchy.levels
            @test lvl.R_map !== nothing
        end
        x = zeros(N)
        x, _ = amg_solve!(x, b, hierarchy, config; tol=1e-6, maxiter=500)
        @test norm(b - A * x) / norm(b) < 1e-6
        # Scale coefficients and do partial resetup
        A2 = copy(A)
        nonzeros(A2) .*= 1.5
        amg_resetup!(hierarchy, A2, config; partial=true)
        x2 = zeros(N)
        x2, _ = amg_solve!(x2, b, hierarchy, config; tol=1e-6, maxiter=500)
        @test norm(b - A2 * x2) / norm(b) < 1e-6
    end

    @testset "Full resetup (partial=false)" begin
        config = AMGConfig()
        hierarchy = amg_setup(A, config)
        A2 = copy(A)
        nonzeros(A2) .*= 1.5
        amg_resetup!(hierarchy, A2, config; partial=false)
        x2 = zeros(N)
        x2, _ = amg_solve!(x2, b, hierarchy, config; tol=1e-6, maxiter=500)
        @test norm(b - A2 * x2) / norm(b) < 1e-6
    end

    @testset "Resetup without allow_partial_resetup (full rebuild only)" begin
        config = AMGConfig(allow_partial_resetup=false)
        hierarchy = amg_setup(A, config)
        for lvl in hierarchy.levels
            @test lvl.R_map === nothing
        end
        A2 = copy(A)
        nonzeros(A2) .*= 1.5
        amg_resetup!(hierarchy, A2, config; partial=false)
        x2 = zeros(N)
        x2, _ = amg_solve!(x2, b, hierarchy, config; tol=1e-6, maxiter=500)
        @test norm(b - A2 * x2) / norm(b) < 1e-6
    end

    @testset "Resetup update_P=true (HMIS + DirectInterpolation)" begin
        config = AMGConfig(coarsening=HMISCoarsening(0.5, DirectInterpolation()))
        hierarchy = amg_setup(A, config)
        for lvl in hierarchy.levels
            @test lvl.P_update_map !== nothing
        end
        x = zeros(N)
        x, _ = amg_solve!(x, b, hierarchy, config; tol=1e-6, maxiter=500)
        @test norm(b - A * x) / norm(b) < 1e-6
        A2 = copy(A)
        nonzeros(A2) .*= 1.5
        amg_resetup!(hierarchy, A2, config; partial=true, update_P=true)
        x2 = zeros(N)
        x2, _ = amg_solve!(x2, b, hierarchy, config; tol=1e-6, maxiter=500)
        @test norm(b - A2 * x2) / norm(b) < 1e-6
    end

    @testset "Resetup update_P=true (HMIS + ExtendedIInterpolation)" begin
        config = AMGConfig(coarsening=HMISCoarsening(0.5, ExtendedIInterpolation()))
        hierarchy = amg_setup(A, config)
        for lvl in hierarchy.levels
            @test lvl.P_update_map !== nothing
            @test lvl.P_update_map.interp_type == 3
        end
        A2 = copy(A)
        nonzeros(A2) .*= 1.5
        amg_resetup!(hierarchy, A2, config; partial=true, update_P=true)
        x2 = zeros(N)
        x2, _ = amg_solve!(x2, b, hierarchy, config; tol=1e-6, maxiter=500)
        @test norm(b - A2 * x2) / norm(b) < 1e-6
    end

    @testset "Resetup preserves sparsity pattern" begin
        config = AMGConfig()
        hierarchy = amg_setup(A, config)
        patterns = [(copy(colvals(lvl.A)), copy(rowptr(lvl.A))) for lvl in hierarchy.levels]
        A2 = copy(A)
        nonzeros(A2) .*= 2.0
        amg_resetup!(hierarchy, A2, config; partial=true)
        for (i, (cv, rp)) in enumerate(patterns)
            @test colvals(hierarchy.levels[i].A) == cv
            @test rowptr(hierarchy.levels[i].A) == rp
        end
    end
end

@testset "Real Matrices - Resetup Options (norne)" begin
    TEST_MATRICES_DIR = joinpath(@__DIR__, "test_matrices")
    A = read_mtx(joinpath(TEST_MATRICES_DIR, "norne.mtx"))
    N = size(A, 1)
    b = ones(N)
    config = AMGConfig()

    @testset "Partial resetup convergence" begin
        hierarchy = amg_setup(A, config)
        nlevels_init = length(hierarchy.levels)
        A2 = copy(A)
        nonzeros(A2) .*= 1.2
        amg_resetup!(hierarchy, A2, config; partial=true)
        @test length(hierarchy.levels) == nlevels_init
        x = zeros(N)
        x, _ = amg_solve!(x, b, hierarchy, config; tol=1e-6, maxiter=500)
        @test norm(b - A2 * x) / norm(b) < 1e-6
    end

    @testset "Full resetup convergence" begin
        hierarchy = amg_setup(A, config)
        A2 = copy(A)
        nonzeros(A2) .*= 1.5
        amg_resetup!(hierarchy, A2, config; partial=false)
        x = zeros(N)
        x, _ = amg_solve!(x, b, hierarchy, config; tol=1e-6, maxiter=500)
        @test norm(b - A2 * x) / norm(b) < 1e-6
    end
end
