using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR

function _cuda_poisson1d(n::Int)
    I = vcat(collect(1:n), collect(1:n - 1), collect(2:n))
    J = vcat(collect(1:n), collect(2:n), collect(1:n - 1))
    V = vcat(fill(2.0, n), fill(-1.0, 2n - 2))
    return CuSparseMatrixCSR(sparse(I, J, V, n, n))
end

@testset "CUDA native AMG setup" begin
    if !CUDA.functional()
        @test_skip CUDA.functional()
    else
        n = 128
        A = _cuda_poisson1d(n)
        config = AMGConfig(
            coarsening=PMISCoarsening(0.25, DirectInterpolation()),
            smoother=JacobiSmootherType(),
            max_coarse_size=8,
            max_row_sum=0.9,
        )
        hierarchy = amg_setup(A, config)

        @test !isempty(hierarchy.levels)
        @test all(level -> level.A.nzval isa CuArray &&
                           level.P.nzval isa CuArray &&
                           level.Pt_map.offsets isa CuArray &&
                           level.R_map === nothing,
                  hierarchy.levels)
        @test hierarchy.coarse_A isa CuArray

        b = CUDA.ones(Float64, n)
        x = CUDA.zeros(Float64, n)
        _, niter = amg_solve!(x, b, hierarchy, config; tol=1e-6, maxiter=100)
        @test niter < 100

        # A native hierarchy has no CPU triple map; resetup uses the same
        # full device rebuild and preserves device-backed P and transpose maps.
        amg_resetup!(hierarchy, _cuda_poisson1d(n), config)
        @test all(level -> level.A.nzval isa CuArray &&
                           level.P.nzval isa CuArray &&
                           level.Pt_map.offsets isa CuArray,
                  hierarchy.levels)
    end
end
