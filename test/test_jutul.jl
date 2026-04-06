@testset "Jutul Interface - DraugrPreconditioner" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    b = rand(N)

    # Create preconditioner via solver dispatch
    prec = DraugrPreconditioner(solver=:jutul)
    @test prec isa Jutul.JutulPreconditioner
    @test isnothing(prec.hierarchy)
    @test Jutul.operator_nrows(prec) == 0

    # Update preconditioner (first call = setup)
    ctx = Jutul.DefaultContext()
    Jutul.update_preconditioner!(prec, A, b, ctx, nothing)
    @test !isnothing(prec.hierarchy)
    @test Jutul.operator_nrows(prec) == N

    # Apply preconditioner (one V-cycle)
    x = zeros(N)
    Jutul.apply!(x, prec, b)
    @test norm(x) > 0  # not zero

    # Update again (resetup)
    nonzeros(A) .*= 2.0
    Jutul.update_preconditioner!(prec, A, b, ctx, nothing)
    @test Jutul.operator_nrows(prec) == N
end

@testset "Jutul Interface - Convergence" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    b = rand(N)

    prec = DraugrPreconditioner(solver=:jutul)
    ctx = Jutul.DefaultContext()
    Jutul.update_preconditioner!(prec, A, b, ctx, nothing)

    # Use the preconditioner iteratively (manual Krylov-like iteration)
    x = zeros(N)
    for _ in 1:100
        r = b - sparse(A.At') * x
        dx = zeros(N)
        Jutul.apply!(dx, prec, r)
        x .+= dx
    end
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-6
end

@testset "Jutul Interface - Custom Config" begin
    prec = DraugrPreconditioner(
        solver=:jutul,
        smoother=ColoredGaussSeidelType(),
        coarsening=PMISCoarsening(),
        pre_smoothing_steps=2,
        post_smoothing_steps=2
    )
    @test prec.config.smoother isa ColoredGaussSeidelType
    @test prec.config.coarsening isa PMISCoarsening
    @test prec.config.pre_smoothing_steps == 2

    A = poisson2d_csr(10)
    N = 100
    b = rand(N)
    ctx = Jutul.DefaultContext()
    Jutul.update_preconditioner!(prec, A, b, ctx, nothing)
    @test Jutul.operator_nrows(prec) == N
end

@testset "Jutul Interface - Smart resetup (update_P when available)" begin
    # With HMIS+ExtendedI (default), P_update_map is built → update_P=true path
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    b = rand(N)
    prec = DraugrPreconditioner(solver=:jutul, coarsening=HMISCoarsening(0.5, DirectInterpolation()))
    ctx = Jutul.DefaultContext()

    # First call: full setup
    Jutul.update_preconditioner!(prec, A, b, ctx, nothing)
    @test !isnothing(prec.hierarchy)
    @test Jutul.operator_nrows(prec) == N

    # Second call (resetup): should use smartest available path
    nonzeros(A) .*= 2.0
    Jutul.update_preconditioner!(prec, A, b, ctx, nothing)
    @test Jutul.operator_nrows(prec) == N

    # Result should still converge
    x = zeros(N)
    Jutul.apply!(x, prec, b)
    @test norm(x) > 0
end

@testset "Jutul Interface - :jutul_partial preconditioner" begin
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    b = rand(N)

    prec = DraugrPreconditioner(solver=:jutul_partial)
    @test prec isa Jutul.JutulPreconditioner
    @test isnothing(prec.hierarchy)
    @test Jutul.operator_nrows(prec) == 0

    ctx = Jutul.DefaultContext()
    # First call: full setup
    Jutul.update_preconditioner!(prec, A, b, ctx, nothing)
    @test !isnothing(prec.hierarchy)
    @test Jutul.operator_nrows(prec) == N

    # Apply
    x = zeros(N)
    Jutul.apply!(x, prec, b)
    @test norm(x) > 0

    # Second call (resetup): uses smart strategy
    nonzeros(A) .*= 2.0
    Jutul.update_preconditioner!(prec, A, b, ctx, nothing)
    @test Jutul.operator_nrows(prec) == N
end

@testset "Jutul Interface - :jutul_partial with AggregationCoarsening (no P_update_map)" begin
    # Aggregation coarsening → no P_update_map → should fall back to partial=true
    n = 10
    A = poisson2d_csr(n)
    N = n*n
    b = rand(N)

    prec = DraugrPreconditioner(solver=:jutul_partial,
                                coarsening=AggregationCoarsening())
    ctx = Jutul.DefaultContext()
    Jutul.update_preconditioner!(prec, A, b, ctx, nothing)
    @test !isnothing(prec.hierarchy)
    # Aggregation doesn't build P_update_map
    for lvl in prec.hierarchy.levels
        @test lvl.P_update_map === nothing
    end
    # But R_map should be present (allow_partial_resetup=true by default)
    @test prec.hierarchy.levels[1].R_map !== nothing

    # Second call: falls back to partial=true (no update_P)
    nonzeros(A) .*= 3.0
    Jutul.update_preconditioner!(prec, A, b, ctx, nothing)
    @test Jutul.operator_nrows(prec) == N

    # Should still converge
    x = zeros(N)
    for _ in 1:100
        r = b - sparse(A.At') * x
        dx = zeros(N)
        Jutul.apply!(dx, prec, r)
        x .+= dx
    end
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-6
end

@testset "Jutul Interface - :jutul_partial with allow_partial_resetup=false (full rebuild)" begin
    # No restriction maps → should fall back to partial=false
    n = 8
    A = poisson2d_csr(n)
    N = n*n
    b = rand(N)

    prec = DraugrPreconditioner(solver=:jutul_partial,
                                coarsening=AggregationCoarsening(),
                                allow_partial_resetup=false)
    ctx = Jutul.DefaultContext()
    Jutul.update_preconditioner!(prec, A, b, ctx, nothing)
    @test !isnothing(prec.hierarchy)
    # No R_map with allow_partial_resetup=false
    for lvl in prec.hierarchy.levels
        @test lvl.R_map === nothing
    end

    # Second call: should do full rebuild (partial=false)
    nonzeros(A) .*= 2.0
    Jutul.update_preconditioner!(prec, A, b, ctx, nothing)
    @test Jutul.operator_nrows(prec) == N
end
