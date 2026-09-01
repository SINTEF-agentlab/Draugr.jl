using CUDA
using Draugr
using SparseArrays
using Test
using CUDA.CUSPARSE: CuSparseMatrixCSR

function poisson1d_cuda(n::Int)
    I = vcat(collect(1:n), collect(1:n-1), collect(2:n))
    J = vcat(collect(1:n), collect(2:n), collect(1:n-1))
    V = vcat(fill(2.0, n), fill(-1.0, 2n - 2))
    return CuSparseMatrixCSR(sparse(I, J, V, n, n))
end

@test CUDA.functional()

@testset "CUDA AMG setup" begin
    n = 128
    A = poisson1d_cuda(n)
    A_csr = csr_from_gpu(A)

    # PMIS graph analysis is executed through KernelAbstractions on the GPU.
    cf, coarse_map, n_coarse = Draugr.coarsen_pmis(
        A_csr, 0.25;
        backend = CUDA.CUDABackend(),
        strength_type = SignedStrength(),
    )
    @test n_coarse == count(==(1), cf)
    @test all(i -> cf[i] == 1 ? coarse_map[i] > 0 : coarse_map[i] == 0,
              eachindex(cf))

    config = AMGConfig(
        coarsening = PMISCoarsening(0.25, DirectInterpolation()),
        max_coarse_size = 8,
    )
    hierarchy = amg_setup(A, config)
    @test !isempty(hierarchy.levels)
    @test hierarchy.levels[1].A.nzval isa CuArray

    b = CUDA.ones(Float64, n)
    x = CUDA.zeros(Float64, n)
    _, niter = amg_solve!(x, b, hierarchy, config; tol = 1e-6, maxiter = 100)
    @test niter < 100
end
