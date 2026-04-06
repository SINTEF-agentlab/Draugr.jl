@testset "JLArrays GPU Backend" begin
    using JLArrays
    using JLArrays: JLSparseMatrixCSR
    using KernelAbstractions

    # Helper: build a CSR matrix on JLArrays backend from a CPU SparseMatrixCSC
    function poisson1d_jl(n)
        I = Int[]; J = Int[]; V = Float64[]
        for i in 1:n
            push!(I, i); push!(J, i); push!(V, 2.0)
            if i > 1
                push!(I, i); push!(J, i-1); push!(V, -1.0)
            end
            if i < n
                push!(I, i); push!(J, i+1); push!(V, -1.0)
            end
        end
        A_csc = sparse(I, J, V, n, n)
        return JLSparseMatrixCSR(A_csc)
    end

    function poisson2d_jl(nx, ny=nx)
        n = nx * ny
        I = Int[]; J = Int[]; V = Float64[]
        for j in 1:ny, i in 1:nx
            idx = (j-1)*nx + i
            push!(I, idx); push!(J, idx); push!(V, 4.0)
            if i > 1
                push!(I, idx); push!(J, idx-1); push!(V, -1.0)
            end
            if i < nx
                push!(I, idx); push!(J, idx+1); push!(V, -1.0)
            end
            if j > 1
                push!(I, idx); push!(J, idx-nx); push!(V, -1.0)
            end
            if j < ny
                push!(I, idx); push!(J, idx+nx); push!(V, -1.0)
            end
        end
        A_csc = sparse(I, J, V, n, n)
        return JLSparseMatrixCSR(A_csc)
    end

    # Helper: compute residual norm on CPU for a JLArrays solution
    function jl_residual_norm(A_jl, x, b)
        x_cpu = Array(x)
        b_cpu = Array(b)
        A_cpu = csr_to_cpu(csr_from_gpu(A_jl))
        n = length(b_cpu)
        r = zeros(n)
        rp = A_cpu.rowptr; cv = A_cpu.colval; nzv = A_cpu.nzval
        for i in 1:n
            Ax_i = 0.0
            for nz in rp[i]:(rp[i+1]-1)
                Ax_i += nzv[nz] * x_cpu[cv[nz]]
            end
            r[i] = b_cpu[i] - Ax_i
        end
        return norm(r) / norm(b_cpu)
    end

    @testset "csr_from_gpu with JLSparseMatrixCSR" begin
        A_jl = poisson1d_jl(10)
        A_csr = csr_from_gpu(A_jl)
        @test A_csr isa CSRMatrix
        @test size(A_csr) == (10, 10)
        @test A_csr.rowptr isa JLArray
        @test A_csr.colval isa JLArray
        @test A_csr.nzval isa JLArray
    end

    @testset "AMG setup with JLSparseMatrixCSR" begin
        A_jl = poisson1d_jl(100)
        config = AMGConfig()
        hierarchy = amg_setup(A_jl, config)
        @test length(hierarchy.levels) >= 1
    end

    @testset "AMG solve with JLSparseMatrixCSR - 1D Poisson" begin
        n = 100
        A_jl = poisson1d_jl(n)
        config = AMGConfig()
        hierarchy = amg_setup(A_jl, config)
        b = JLArray(ones(n))
        x = JLArray(zeros(n))
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=100)
        @test jl_residual_norm(A_jl, x, b) < 1e-6
        @test niter < 100
    end

    @testset "AMG solve with JLSparseMatrixCSR - 2D Poisson" begin
        nx = 10
        A_jl = poisson2d_jl(nx)
        n = nx * nx
        config = AMGConfig()
        hierarchy = amg_setup(A_jl, config)
        b = JLArray(ones(n))
        x = JLArray(zeros(n))
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=100)
        @test jl_residual_norm(A_jl, x, b) < 1e-6
        @test niter < 100
    end

    @testset "AMG cycle with JLSparseMatrixCSR" begin
        A_jl = poisson1d_jl(100)
        config = AMGConfig()
        hierarchy = amg_setup(A_jl, config)
        n = 100
        b = JLArray(ones(n))
        x = JLArray(zeros(n))
        # Apply a single cycle (backend from hierarchy)
        amg_cycle!(x, b, hierarchy, config)
        @test !all(Array(x) .== 0.0)  # something changed
    end

    @testset "Backend stored in hierarchy" begin
        A_jl = poisson1d_jl(100)
        config = AMGConfig()
        hierarchy = amg_setup(A_jl, config)
        @test length(hierarchy.levels) >= 1
        @test hierarchy.backend isa JLBackend
        @test hierarchy.block_size == 64
    end

    @testset "AMG with all smoothers on JLArrays" begin
        A_jl = poisson1d_jl(100)
        all_smoothers = [
            JacobiSmootherType(),
            SPAI0SmootherType(),
            SPAI1SmootherType(),
            L1JacobiSmootherType(),
            ChebyshevSmootherType(),
            ColoredGaussSeidelType(),
            L1ColoredGaussSeidelType(),
            ILU0SmootherType(),
        ]
        for smoother in all_smoothers
            @testset "Smoother: $(typeof(smoother).name.name)" begin
                config = AMGConfig(smoother=smoother)
                hierarchy = amg_setup(A_jl, config)
                @test length(hierarchy.levels) >= 1
                b = JLArray(ones(100))
                x = JLArray(zeros(100))
                x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
                @test niter < 200
            end
        end
    end

    @testset "AMG with all coarsening types on JLArrays" begin
        A_jl = poisson2d_jl(10)
        n = 100
        all_coarsenings = [
            AggregationCoarsening(),
            PMISCoarsening(),
            HMISCoarsening(),
            RSCoarsening(),
            AggressiveCoarsening(),
            SmoothedAggregationCoarsening(),
        ]
        for coarsening in all_coarsenings
            @testset "Coarsening: $(typeof(coarsening).name.name)" begin
                config = AMGConfig(coarsening=coarsening)
                hierarchy = amg_setup(A_jl, config)
                @test length(hierarchy.levels) >= 1
                b = JLArray(ones(n))
                x = JLArray(zeros(n))
                x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-6, maxiter=200)
                @test niter < 200
            end
        end
    end

    @testset "AMG with CF interpolation types on JLArrays" begin
        A_jl = poisson2d_jl(10)
        n = 100
        interp_types = [
            DirectInterpolation(),
            StandardInterpolation(),
            ExtendedIInterpolation(),
        ]
        for interp in interp_types
            @testset "Interpolation: $(typeof(interp).name.name)" begin
                config = AMGConfig(coarsening=PMISCoarsening(0.25, interp))
                hierarchy = amg_setup(A_jl, config)
                @test length(hierarchy.levels) >= 1
                b = JLArray(ones(n))
                x = JLArray(zeros(n))
                x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-6, maxiter=200)
                @test niter < 200
            end
        end
    end

    @testset "AMG resetup with JLSparseMatrixCSR" begin
        nx = 10
        A_jl = poisson2d_jl(nx)
        n = nx * nx
        config = AMGConfig()
        hierarchy = amg_setup(A_jl, config)
        @test length(hierarchy.levels) >= 1
        # Resetup with the same matrix (same sparsity, same values)
        amg_resetup!(hierarchy, A_jl, config)
        # Solve after resetup (backend from hierarchy)
        b = JLArray(ones(n))
        x = JLArray(zeros(n))
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=100)
        @test niter < 100
    end

    @testset "Verbosity levels" begin
        A_jl = poisson2d_jl(10)
        n = 100
        # verbosity 0 = silent
        config0 = AMGConfig(verbose=0)
        hierarchy0 = amg_setup(A_jl, config0)
        @test length(hierarchy0.levels) >= 1
        # verbosity 1 = hierarchy + solve summary
        config1 = AMGConfig(verbose=1)
        hierarchy1 = amg_setup(A_jl, config1)
        @test length(hierarchy1.levels) >= 1
        # verbosity 2 = per-iteration output
        config2 = AMGConfig(verbose=2)
        hierarchy2 = amg_setup(A_jl, config2)
        @test length(hierarchy2.levels) >= 1
        # Bool backward compat
        config_bool = AMGConfig(verbose=true)
        @test config_bool.verbose == 1
        config_bool_f = AMGConfig(verbose=false)
        @test config_bool_f.verbose == 0
    end

    @testset "csr_to_cpu conversion" begin
        A_jl = poisson1d_jl(10)
        A_gpu = csr_from_gpu(A_jl)
        A_cpu = csr_to_cpu(A_gpu)
        @test A_cpu.rowptr isa Vector
        @test A_cpu.colval isa Vector
        @test A_cpu.nzval isa Vector
        @test size(A_cpu) == size(A_gpu)
    end

    @testset "coarse_solve_on_cpu with JLArrays" begin
        A_jl = poisson1d_jl(100)
        config = AMGConfig(coarse_solve_on_cpu=true)
        hierarchy = amg_setup(A_jl, config)
        @test hierarchy.coarse_solve_on_cpu == true
        # coarse_A and coarse_x/coarse_b should be CPU arrays
        @test hierarchy.coarse_A isa Matrix
        @test hierarchy.coarse_x isa Vector
        @test hierarchy.coarse_b isa Vector
        # Solve should still converge
        n = 100
        b = JLArray(ones(n))
        x = JLArray(zeros(n))
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=100)
        @test jl_residual_norm(A_jl, x, b) < 1e-6
        @test niter < 100
        # Resetup should also work with coarse_solve_on_cpu
        amg_resetup!(hierarchy, A_jl, config)
        @test hierarchy.coarse_A isa Matrix
        @test hierarchy.coarse_x isa Vector
        @test hierarchy.coarse_b isa Vector
        x2 = JLArray(zeros(n))
        x2, niter2 = amg_solve!(x2, b, hierarchy, config; tol=1e-8, maxiter=100)
        @test niter2 < 100
    end
end
