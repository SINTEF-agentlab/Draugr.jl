using Jutul, HYPRE, Draugr, Krylov, Test
using Jutul.StaticCSR: StaticSparsityMatrixCSR, static_sparsity_sparse

include("helpers.jl")


# Small test script to compare HYPRE and Draugr on a matrix with gmres.
function solve_draugr(A, b, config)
    tol = 1e-8
    maxit = 100

    t_setup_initial = @elapsed hierarchy = amg_setup(A, config);
    t_setup = @elapsed amg_resetup!(hierarchy, A, config);
    M = DraugrPreconditioner(config, hierarchy, size(A))
    t_solve = @elapsed x, stats = gmres(A, b; M = M, rtol = tol, itmax=maxit, verbose = 1, ldiv = true)
    println("DRAUGR setup time: $t_setup seconds (initial setup: $t_setup_initial seconds)")
    println("DRAUGR solve time: $t_solve seconds")
    return (x, Dict(:t_setup => t_setup, :t_solve => t_solve, :t_setup_initial => t_setup_initial, :nits => stats.niter))
end

function solve_hypre(A, b, ; kwarg...)
    tol = 1e-8
    maxit = 100

    prec_hypre = Jutul.BoomerAMGPreconditioner(PrintLevel = 1, AggNumLevels = 0; kwarg...)
    t_setup_hypre = @elapsed Jutul.update_preconditioner!(prec_hypre, A, b, missing, missing)
    op_hypre = Jutul.linear_operator(prec_hypre)
    t_solve_hypre = @elapsed x, stats = gmres(A, b; M = op_hypre, rtol = tol, itmax=maxit, verbose = 1)
    println("HYPRE setup time: $t_setup_hypre seconds")
    println("HYPRE solve time: $t_solve_hypre seconds")
    return (x, Dict(:t_setup => t_setup_hypre, :t_solve => t_solve_hypre, :nits => stats.niter))
end

function test_hmis(A, b)
    # Example on running hypre as bench
    x_h, stats_h = solve_hypre(A, b, AggTruncFactor = 0.0, StrongThreshold=0.25)
    # Matching HYPRE defaults: StrongThreshold=0.25, max_elmts=4, no trunc_factor, max_row_sum=0.9
    coarsen = HMISCoarsening(0.25, ExtendedIInterpolation(0.0, 4, 2, true))
    s = L1SerialGaussSeidelType()

    config = AMGConfig(coarsening=coarsen,
        smoother = s,
        verbose = true,
        max_row_sum = 0.9,
        strength_type = SignedStrength()
    )
    x, stats = solve_draugr(A, b, config);

    @test stats[:nits] <= 1.1 * stats_h[:nits]
end

function test_classical(A, b)
    # Example on running hypre as bench
    x_h, stats_h = solve_hypre(A, b, CoarsenType = 1, InterpType = 0, StrongThreshold=0.25)
    coarsen = RSCoarsening(0.25)
    s = L1SerialGaussSeidelType()

    config = AMGConfig(coarsening=coarsen,
        smoother = s,
        verbose = true,
        max_row_sum = 1.0,
        strength_type = SignedStrength()
    )
    x, stats = solve_draugr(A, b, config);

    @test stats[:nits] <= 1.1 * stats_h[:nits]
end

@testset "poisson_2d 1000x1000" begin
    N = 1000
    A = poisson2d_csr(N, N)
    b = rand(N*N)

    @testset "HMIS + Ext+i" begin
        test_hmis(A, b)
    end
    @testset "classical RS" begin
        test_classical(A, b)
    end
end

@testset "anisotropic_csr 1000x1000" begin
    N = 1000
    A = anisotropic_csr(N, N)
    b = rand(N*N)

    @testset "HMIS + Ext+i" begin
        test_hmis(A, b)
    end
    @testset "classical RS" begin
        test_classical(A, b)
    end
end


