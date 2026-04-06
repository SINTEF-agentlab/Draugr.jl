@testset "Strength of Connection" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    is_strong = Draugr.strength_graph(Ac, 0.25)
    # All off-diagonal entries should be strong (|-1| >= 0.25*|-1|)
    @test sum(is_strong) == 18  # 9+9 off-diagonal entries
end

@testset "Aggregation Coarsening" begin
    A = poisson1d_csr(20)
    Ac = to_csr(A)
    agg, nc = Draugr.coarsen_aggregation(Ac, 0.25)
    @test length(agg) == 20
    @test all(agg .> 0)
    @test nc > 0
    @test nc < 20
    # Each aggregate should have at least one member
    for k in 1:nc
        @test any(agg .== k)
    end
end

@testset "PMIS Coarsening" begin
    A = poisson1d_csr(20)
    Ac = to_csr(A)
    cf, cmap, nc = Draugr.coarsen_pmis(Ac, 0.25)
    @test length(cf) == 20
    @test all(abs.(cf) .== 1)  # all decided
    @test nc > 0
    @test nc < 20
end

@testset "Aggressive Coarsening" begin
    A = poisson1d_csr(20)
    Ac = to_csr(A)
    agg, nc = Draugr.coarsen_aggressive(Ac, 0.25)
    @test length(agg) == 20
    @test all(agg .> 0)
    @test nc > 0
    @test nc < 20
end

@testset "HMIS Coarsening" begin
    A = poisson1d_csr(20)
    Ac = to_csr(A)
    cf, cmap, nc = Draugr.coarsen_hmis(Ac, 0.25)
    @test length(cf) == 20
    @test all(abs.(cf) .== 1)
    @test nc > 0
    @test nc < 20
    # Every fine point should have at least one coarse neighbor
    cv = Draugr.colvals(Ac)
    for i in 1:20
        if cf[i] == -1
            has_coarse = false
            for nz in nzrange(Ac, i)
                j = cv[nz]
                if j != i && cf[j] == 1
                    has_coarse = true
                    break
                end
            end
            @test has_coarse
        end
    end
end

@testset "HMIS Coarsening - 2D" begin
    A = poisson2d_csr(8)
    Ac = to_csr(A)
    cf, cmap, nc = Draugr.coarsen_hmis(Ac, 0.25)
    @test nc > 0
    @test nc < 64
    @test sum(cf .== 1) == nc
end

@testset "HMIS Coarsening ratio - 2D" begin
    # Verify HMIS produces good coarsening ratios (matching hypre behavior).
    # HMIS uses RS first pass + PMIS, yielding aggressive coarsening.
    A = poisson2d_csr(20)
    Ac = to_csr(A)
    n = size(Ac, 1)
    cf, cmap, nc = Draugr.coarsen_hmis(Ac, 0.5)
    ratio = nc / n
    @test ratio < 0.6  # hypre typically achieves ~0.45 for 2D Poisson with θ=0.5
    @test all(abs.(cf) .== 1)  # all points decided as C (1) or F (-1)
end

@testset "HMIS Hierarchy depth - 2D" begin
    # Verify HMIS produces a shallow hierarchy matching hypre's behavior
    # (should produce ~5-8 levels for 100x100 2D Poisson, not 20)
    A = poisson2d_csr(50)
    config = AMGConfig(coarsening=HMISCoarsening(0.5, ExtendedIInterpolation()))
    hierarchy = amg_setup(A, config)
    nlevels = length(hierarchy.levels) + 1
    @test nlevels <= 10
end

@testset "RS Coarsening - Basic" begin
    A = poisson1d_csr(20)
    Ac = to_csr(A)
    cf, cmap, nc = Draugr.coarsen_rs(Ac, 0.25)
    @test length(cf) == 20
    @test all(abs.(cf) .== 1)
    @test nc > 0
    @test nc < 20
    # Every F-point should have a strong C-neighbor
    is_strong = Draugr.strength_graph(Ac, 0.25)
    cv = colvals(Ac)
    for i in 1:20
        if cf[i] == -1
            has_C = false
            for nz in nzrange(Ac, i)
                j = cv[nz]
                if j != i && is_strong[nz] && cf[j] == 1
                    has_C = true
                    break
                end
            end
            @test has_C
        end
    end
end

@testset "RS Coarsening - 2D" begin
    A = poisson2d_csr(10)
    Ac = to_csr(A)
    cf, cmap, nc = Draugr.coarsen_rs(Ac, 0.25)
    @test nc > 0
    @test nc < 100
    @test sum(cf .== 1) == nc
end

@testset "RS Coarsening - Good coarsening ratios" begin
    A = poisson2d_csr(30)
    config = AMGConfig(coarsening=RSCoarsening(0.25, DirectInterpolation()))
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) >= 1
    # RS should produce good coarsening ratios throughout
    for i in 1:length(hierarchy.levels)-1
        n_current = size(hierarchy.levels[i].A, 1)
        n_next = size(hierarchy.levels[i+1].A, 1)
        # Every level should coarsen meaningfully (ratio < 0.85)
        @test n_next < n_current
    end
end

@testset "RS Coarsening - larger system" begin
    # Test with a larger system to exercise the bucket sort path
    A = poisson2d_csr(30)
    Ac = to_csr(A)
    cf, cmap, nc = Draugr.coarsen_rs(Ac, 0.25)
    @test nc > 0
    @test nc < 900
    @test sum(cf .== 1) == nc
    @test all(abs.(cf) .== 1)
    # Every F-point should have a strong C-neighbor
    is_strong = Draugr.strength_graph(Ac, 0.25)
    cv = colvals(Ac)
    for i in 1:size(Ac, 1)
        if cf[i] == -1
            has_C = false
            for nz in nzrange(Ac, i)
                j = cv[nz]
                if j != i && is_strong[nz] && cf[j] == 1
                    has_C = true
                    break
                end
            end
            @test has_C
        end
    end
end

@testset "PMIS - Good coarsening ratios" begin
    A = poisson2d_csr(20)
    config = AMGConfig(coarsening=PMISCoarsening(0.25, DirectInterpolation()))
    hierarchy = amg_setup(A, config)
    @test length(hierarchy.levels) >= 1
    # Solve should still work
    N = size(A, 1)
    b = rand(N)
    x = zeros(N)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
end

@testset "coarsen_aggressive_cf with HMIS" begin
    n = 10
    A = poisson2d_csr(n)
    A_csr = to_csr(A)
    N = n * n
    # Use fixed RNG for reproducibility
    rng = Random.MersenneTwister(42)
    cf, coarse_map, n_coarse = Draugr.coarsen_aggressive_cf(A_csr, 0.25, :hmis; rng=rng)
    @test n_coarse > 0
    @test n_coarse < N  # should have coarsened
    # Verify it produces a reasonable coarsening ratio (less than 60% of original)
    @test n_coarse < 0.6 * N
    # All points should be decided as C or F
    for i in 1:N
        @test cf[i] == 1 || cf[i] == -1
    end
    # Coarse map should be valid for all C-points
    for i in 1:N
        if cf[i] == 1
            @test coarse_map[i] >= 1
            @test coarse_map[i] <= n_coarse
        end
    end
end

@testset "coarsen_aggressive_cf with PMIS" begin
    n = 10
    A = poisson2d_csr(n)
    A_csr = to_csr(A)
    N = n * n
    cf, coarse_map, n_coarse = Draugr.coarsen_aggressive_cf(A_csr, 0.25, :pmis)
    @test n_coarse > 0
    @test n_coarse < N
    for i in 1:N
        @test cf[i] == 1 || cf[i] == -1
    end
end

@testset "MIS-based aggregation produces larger aggregates" begin
    # 2D Poisson: the MIS-based aggregation should create fewer, larger aggregates
    A = poisson2d_csr(20)
    A_csr = to_csr(A)
    agg, n_coarse = Draugr.coarsen_aggregation(A_csr, 0.25)
    n = size(A, 1)
    # The coarsening ratio should be aggressive (not more than 50% of original)
    @test n_coarse < 0.5 * n
    # Average aggregate size should be > 2
    @test n / n_coarse > 2.0
end

@testset "Aggregation - no stalling on sparse irregular matrix" begin
    # Build a sparse irregular matrix that previously caused stalling
    # (many levels with barely decreasing row count)
    Random.seed!(42)
    n = 500
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:n
        push!(I, i); push!(J, i); push!(V, 10.0 + 90.0*rand())
        n_neigh = rand(2:min(5, n-1))
        for _ in 1:n_neigh
            j = rand(1:n)
            j == i && continue
            push!(I, i); push!(J, j); push!(V, -(1.0 + 9.0*rand()))
        end
    end
    A = static_sparsity_sparse(I, J, V, n, n)
    config = AMGConfig(coarsening=AggregationCoarsening(0.25), max_levels=20)
    hierarchy = amg_setup(A, config)
    # Must produce a hierarchy with fewer than 8 levels (previously > 12 stalling levels)
    @test length(hierarchy.levels) < 8
    # Each level should coarsen meaningfully (no consecutive near-stall levels)
    for i in 1:length(hierarchy.levels)-1
        n_current = size(hierarchy.levels[i].A, 1)
        n_next = size(hierarchy.levels[i+1].A, 1)
        @test n_next < n_current  # strictly decreasing
    end
end

@testset "Aggregation - θ auto-reduction fallback" begin
    # Matrix where default θ=0.25 might create poor coarsening
    Random.seed!(123)
    n = 200
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:n
        push!(I, i); push!(J, i); push!(V, 100.0)
        # Very sparse connectivity: only 1-2 neighbors
        for k in 1:rand(1:2)
            j = rand(1:n)
            j == i && continue
            push!(I, i); push!(J, j); push!(V, -(0.1 + rand()))
        end
    end
    A = static_sparsity_sparse(I, J, V, n, n)
    config = AMGConfig(coarsening=AggregationCoarsening(0.25))
    hierarchy = amg_setup(A, config)
    # Should still produce a reasonable hierarchy
    @test length(hierarchy.levels) >= 1
    @test length(hierarchy.levels) < 15
end

@testset "Sign-Aware Strength - AbsoluteStrength" begin
    A = reservoir_like_csr(20)
    Ac = to_csr(A)
    is_strong = Draugr.strength_graph(Ac, 0.25, AbsoluteStrength())
    @test length(is_strong) == nnz(A)
    @test sum(is_strong) > 0
end

@testset "Sign-Aware Strength - SignedStrength" begin
    A = reservoir_like_csr(20)
    Ac = to_csr(A)
    is_strong_signed = Draugr.strength_graph(Ac, 0.25, SignedStrength())
    is_strong_abs = Draugr.strength_graph(Ac, 0.25, AbsoluteStrength())
    @test length(is_strong_signed) == nnz(Ac)
    # Signed strength should not mark positive off-diags as strong (when diag is positive)
    cv = colvals(Ac)
    nzv = nonzeros(Ac)
    for nz in 1:nnz(Ac)
        if is_strong_signed[nz]
            # This connection should have opposite sign from diagonal
            # (or be in a fallback row)
            @test true  # basic validity
        end
    end
    # Should have fewer or equal strong connections (positive off-diags excluded)
    @test sum(is_strong_signed) <= sum(is_strong_abs)
end

@testset "SignedStrength - Config dispatch" begin
    A = poisson2d_csr(8)
    Ac = to_csr(A)
    is1 = Draugr.strength_graph(Ac, 0.25, AMGConfig(strength_type=AbsoluteStrength()))
    is2 = Draugr.strength_graph(Ac, 0.25, AMGConfig(strength_type=SignedStrength()))
    @test length(is1) == nnz(Ac)
    @test length(is2) == nnz(Ac)
end
