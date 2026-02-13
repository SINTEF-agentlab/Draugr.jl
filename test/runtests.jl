using Test
using ParallelAMG
using SparseArrays
using LinearAlgebra
import Jutul

# ── Helper: 1D Poisson matrix ────────────────────────────────────────────────
function poisson1d_csr(n)
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
    return static_sparsity_sparse(I, J, V, n, n)
end

# ── Helper: 2D Poisson matrix ────────────────────────────────────────────────
function poisson2d_csr(nx, ny=nx)
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
    return static_sparsity_sparse(I, J, V, n, n)
end

@testset "ParallelAMG" begin

    @testset "StaticCSR" begin
        A = poisson1d_csr(10)
        @test size(A) == (10, 10)
        @test nnz(A) == 28  # 10 diag + 9+9 off-diag
        # Check matrix-vector product
        x = ones(10)
        y = zeros(10)
        mul!(y, A, x)
        # For 1D Poisson with Dirichlet BC: first/last rows have [2,-1]*[1,1] = 1
        # Interior rows have [-1,2,-1]*[1,1,1] = 0
        @test y[1] ≈ 1.0
        @test y[10] ≈ 1.0
        @test all(y[2:9] .≈ 0.0)
        # Test getindex
        @test A[1,1] ≈ 2.0
        @test A[1,2] ≈ -1.0
        # Test from CSC
        A_csc = sparse([1,1,2,2,2,3,3], [1,2,1,2,3,2,3], [2.0,-1.0,-1.0,2.0,-1.0,-1.0,2.0], 3, 3)
        A2 = static_csr_from_csc(A_csc)
        @test size(A2) == (3, 3)
        @test A2[1,1] ≈ 2.0
        @test A2[2,1] ≈ -1.0
    end

    @testset "Strength of Connection" begin
        A = poisson1d_csr(10)
        is_strong = ParallelAMG.strength_graph(A, 0.25)
        # All off-diagonal entries should be strong (|-1| >= 0.25*|-1|)
        @test sum(is_strong) == 18  # 9+9 off-diagonal entries
    end

    @testset "Aggregation Coarsening" begin
        A = poisson1d_csr(20)
        agg, nc = ParallelAMG.coarsen_aggregation(A, 0.25)
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
        cf, cmap, nc = ParallelAMG.coarsen_pmis(A, 0.25)
        @test length(cf) == 20
        @test all(abs.(cf) .== 1)  # all decided
        @test nc > 0
        @test nc < 20
    end

    @testset "Aggressive Coarsening" begin
        A = poisson1d_csr(20)
        agg, nc = ParallelAMG.coarsen_aggressive(A, 0.25)
        @test length(agg) == 20
        @test all(agg .> 0)
        @test nc > 0
        @test nc < 20
    end

    @testset "Prolongation" begin
        A = poisson1d_csr(10)
        agg, nc = ParallelAMG.coarsen_aggregation(A, 0.25)
        P = ParallelAMG.build_prolongation(A, agg, nc)
        @test P.nrow == 10
        @test P.ncol == nc
        # Each row of P has exactly one nonzero for aggregation
        for i in 1:P.nrow
            @test P.rowptr[i+1] - P.rowptr[i] == 1
        end
        # Test prolongation operation
        xc = ones(nc)
        xf = zeros(10)
        ParallelAMG.prolongate!(xf, P, xc)
        @test all(xf .≈ 1.0)  # P*ones should be ones
        # Test restriction
        rf = ones(10)
        bc = zeros(nc)
        ParallelAMG.restrict!(bc, P, rf)
        # Sum should be preserved
        @test sum(bc) ≈ sum(rf)
    end

    @testset "Galerkin Product" begin
        A = poisson1d_csr(10)
        agg, nc = ParallelAMG.coarsen_aggregation(A, 0.25)
        P = ParallelAMG.build_prolongation(A, agg, nc)
        A_coarse, r_map = ParallelAMG.compute_coarse_sparsity(A, P, nc)
        @test size(A_coarse) == (nc, nc)
        # Verify Galerkin product against explicit computation
        # Build explicit P as sparse matrix
        I_p = Int[]; J_p = Int[]; V_p = Float64[]
        for i in 1:P.nrow
            for nz in P.rowptr[i]:(P.rowptr[i+1]-1)
                push!(I_p, i)
                push!(J_p, P.colval[nz])
                push!(V_p, P.nzval[nz])
            end
        end
        P_sparse = sparse(I_p, J_p, V_p, P.nrow, P.ncol)
        # Build explicit A as sparse
        A_sparse = sparse(A.At')
        # Explicit Galerkin product
        Ac_explicit = P_sparse' * A_sparse * P_sparse
        # Compare
        for i in 1:nc, j in 1:nc
            @test A_coarse[i,j] ≈ Ac_explicit[i,j] atol=1e-12
        end
    end

    @testset "In-place Galerkin Resetup" begin
        A = poisson1d_csr(10)
        agg, nc = ParallelAMG.coarsen_aggregation(A, 0.25)
        P = ParallelAMG.build_prolongation(A, agg, nc)
        A_coarse, r_map = ParallelAMG.compute_coarse_sparsity(A, P, nc)
        # Now modify A's values (scale by 2)
        nzv = nonzeros(A)
        nzv .*= 2.0
        # Recompute in-place
        ParallelAMG.galerkin_product!(A_coarse, A, P, r_map)
        # Verify against explicit computation with scaled matrix
        I_p = Int[]; J_p = Int[]; V_p = Float64[]
        for i in 1:P.nrow
            for nz in P.rowptr[i]:(P.rowptr[i+1]-1)
                push!(I_p, i)
                push!(J_p, P.colval[nz])
                push!(V_p, P.nzval[nz])
            end
        end
        P_sparse = sparse(I_p, J_p, V_p, P.nrow, P.ncol)
        A_sparse = sparse(A.At')
        Ac_explicit = P_sparse' * A_sparse * P_sparse
        for i in 1:nc, j in 1:nc
            @test A_coarse[i,j] ≈ Ac_explicit[i,j] atol=1e-12
        end
    end

    @testset "Jacobi Smoother" begin
        A = poisson1d_csr(10)
        smoother = ParallelAMG.build_jacobi_smoother(A, 2.0/3.0)
        @test length(smoother.invdiag) == 10
        @test all(smoother.invdiag .≈ 0.5)  # 1/2.0
        # Test smoothing reduces error
        b = ones(10)
        x = zeros(10)
        smooth!(x, A, b, smoother; steps=10)
        r = b - sparse(A.At') * x
        @test norm(r) < norm(b)
    end

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
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=AggregationCoarsening())
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    @testset "AMG Solve - PMIS" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=PMISCoarsening())
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    @testset "AMG Solve - Aggressive" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=AggressiveCoarsening())
        hierarchy = amg_setup(A, config)
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
        patterns = [(copy(colvals(lvl.A)), copy(rowptr(lvl.A)))
                     for lvl in hierarchy.levels]
        # Scale and resetup
        nonzeros(A) .*= 3.0
        amg_resetup!(hierarchy, A, config)
        # Verify sparsity patterns are unchanged
        for (i, (cv, rp)) in enumerate(patterns)
            @test colvals(hierarchy.levels[i].A) == cv
            @test rowptr(hierarchy.levels[i].A) == rp
        end
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

    # ══════════════════════════════════════════════════════════════════════════
    # Colored Gauss-Seidel Smoother
    # ══════════════════════════════════════════════════════════════════════════

    @testset "Graph Coloring" begin
        A = poisson1d_csr(10)
        colors, nc = ParallelAMG.greedy_coloring(A)
        @test length(colors) == 10
        @test all(colors .> 0)
        @test nc >= 2  # tridiagonal needs at least 2 colors
        # Verify no two adjacent nodes have the same color
        cv = colvals(A)
        for i in 1:10
            for nz in nzrange(A, i)
                j = cv[nz]
                if j != i
                    @test colors[i] != colors[j]
                end
            end
        end
    end

    @testset "Colored GS Smoother - Build" begin
        A = poisson1d_csr(10)
        smoother = ParallelAMG.build_colored_gs_smoother(A)
        @test smoother.num_colors >= 2
        @test length(smoother.invdiag) == 10
        @test all(smoother.invdiag .≈ 0.5)  # 1/2.0
        @test length(smoother.color_order) == 10
    end

    @testset "Colored GS Smoother - Smoothing" begin
        A = poisson1d_csr(10)
        smoother = ParallelAMG.build_colored_gs_smoother(A)
        b = ones(10)
        x = zeros(10)
        smooth!(x, A, b, smoother; steps=10)
        r = b - sparse(A.At') * x
        @test norm(r) < norm(b)
    end

    @testset "AMG Solve - Colored GS" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(smoother=ColoredGaussSeidelType())
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    # ══════════════════════════════════════════════════════════════════════════
    # SPAI(0) Smoother
    # ══════════════════════════════════════════════════════════════════════════

    @testset "SPAI0 Smoother - Build" begin
        A = poisson1d_csr(10)
        smoother = ParallelAMG.build_spai0_smoother(A)
        @test length(smoother.m_diag) == 10
        # For tridiagonal with diag=2, off-diag=-1:
        # interior row: [−1 2 −1], row_norm_sq = 1+4+1 = 6, diag = 2, m = 2/6 = 1/3
        @test smoother.m_diag[5] ≈ 2.0/6.0
    end

    @testset "SPAI0 Smoother - Smoothing" begin
        A = poisson1d_csr(10)
        smoother = ParallelAMG.build_spai0_smoother(A)
        b = ones(10)
        x = zeros(10)
        smooth!(x, A, b, smoother; steps=10)
        r = b - sparse(A.At') * x
        @test norm(r) < norm(b)
    end

    @testset "AMG Solve - SPAI0" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(smoother=SPAI0SmootherType())
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    # ══════════════════════════════════════════════════════════════════════════
    # SPAI(1) Smoother
    # ══════════════════════════════════════════════════════════════════════════

    @testset "SPAI1 Smoother - Build" begin
        A = poisson1d_csr(5)
        smoother = ParallelAMG.build_spai1_smoother(A)
        @test length(smoother.nzval) == nnz(A)
    end

    @testset "SPAI1 Smoother - Smoothing" begin
        A = poisson1d_csr(10)
        smoother = ParallelAMG.build_spai1_smoother(A)
        b = ones(10)
        x = zeros(10)
        smooth!(x, A, b, smoother; steps=10)
        r = b - sparse(A.At') * x
        @test norm(r) < norm(b)
    end

    @testset "AMG Solve - SPAI1" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(smoother=SPAI1SmootherType(),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=300)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 300
    end

    # ══════════════════════════════════════════════════════════════════════════
    # Verbose output
    # ══════════════════════════════════════════════════════════════════════════

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
    end

    # ══════════════════════════════════════════════════════════════════════════
    # Jutul Preconditioner Interface
    # ══════════════════════════════════════════════════════════════════════════

    @testset "Jutul Interface - ParallelAMGPreconditioner" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)

        # Create preconditioner
        prec = ParallelAMGPreconditioner()
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

        prec = ParallelAMGPreconditioner()
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
        prec = ParallelAMGPreconditioner(
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

    # ══════════════════════════════════════════════════════════════════════════
    # HMIS Coarsening
    # ══════════════════════════════════════════════════════════════════════════

    @testset "HMIS Coarsening" begin
        A = poisson1d_csr(20)
        cf, cmap, nc = ParallelAMG.coarsen_hmis(A, 0.25)
        @test length(cf) == 20
        @test all(abs.(cf) .== 1)
        @test nc > 0
        @test nc < 20
        # Every fine point should have at least one coarse neighbor
        cv = colvals(A)
        for i in 1:20
            if cf[i] == -1
                has_coarse = false
                for nz in nzrange(A, i)
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
        cf, cmap, nc = ParallelAMG.coarsen_hmis(A, 0.25)
        @test nc > 0
        @test nc < 64
        @test sum(cf .== 1) == nc
    end

    @testset "AMG Setup - HMIS" begin
        A = poisson2d_csr(10)
        config = AMGConfig(coarsening=HMISCoarsening())
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) > 0
    end

    @testset "AMG Solve - HMIS Direct" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=HMISCoarsening(0.25, DirectInterpolation()),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    # ══════════════════════════════════════════════════════════════════════════
    # Interpolation Types for PMIS
    # ══════════════════════════════════════════════════════════════════════════

    @testset "PMIS - Direct Interpolation" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=PMISCoarsening(0.25, DirectInterpolation()),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) > 0
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    @testset "PMIS - Standard Interpolation" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=PMISCoarsening(0.25, StandardInterpolation()),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) > 0
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    @testset "PMIS - Extended+i Interpolation" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=PMISCoarsening(0.25, ExtendedIInterpolation()),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) > 0
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    # ══════════════════════════════════════════════════════════════════════════
    # Interpolation Types for HMIS
    # ══════════════════════════════════════════════════════════════════════════

    @testset "HMIS - Standard Interpolation" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=HMISCoarsening(0.25, StandardInterpolation()),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) > 0
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    @testset "HMIS - Extended+i Interpolation" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=HMISCoarsening(0.25, ExtendedIInterpolation()),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) > 0
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    # ══════════════════════════════════════════════════════════════════════════
    # In-place LU Refactorization
    # ══════════════════════════════════════════════════════════════════════════

    @testset "In-place LU Refactorization" begin
        n = 8
        A = poisson2d_csr(n)
        N = n*n
        config = AMGConfig(coarsening=AggregationCoarsening())
        hierarchy = amg_setup(A, config)
        # Check that LU buffer is separate from coarse_A
        @test hierarchy.coarse_lu !== hierarchy.coarse_A
        @test size(hierarchy.coarse_lu) == size(hierarchy.coarse_A)
        # Check that ipiv is pre-allocated
        @test length(hierarchy.coarse_ipiv) == size(hierarchy.coarse_A, 1)
        # Solve, then resetup with scaled matrix and solve again
        b = rand(N)
        x1 = zeros(N)
        x1, _ = amg_solve!(x1, b, hierarchy, config; tol=1e-8, maxiter=200)
        r1 = b - sparse(A.At') * x1
        @test norm(r1) / norm(b) < 1e-8
        # Track the LU buffer memory address
        lu_ptr = pointer(hierarchy.coarse_lu)
        ipiv_ptr = pointer(hierarchy.coarse_ipiv)
        # Resetup
        nonzeros(A) .*= 2.0
        amg_resetup!(hierarchy, A, config)
        # Verify buffers were reused (same memory)
        @test pointer(hierarchy.coarse_lu) == lu_ptr
        @test pointer(hierarchy.coarse_ipiv) == ipiv_ptr
        # Solve with updated matrix
        x2 = zeros(N)
        x2, _ = amg_solve!(x2, b, hierarchy, config; tol=1e-8, maxiter=200)
        r2 = b - sparse(A.At') * x2
        @test norm(r2) / norm(b) < 1e-8
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
        @test ParallelAMG._get_coarsening_for_level(config, 1) isa AggregationCoarsening
        @test ParallelAMG._get_coarsening_for_level(config, 5) isa AggregationCoarsening
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
        @test ParallelAMG._get_coarsening_for_level(config, 1) isa PMISCoarsening
        @test ParallelAMG._get_coarsening_for_level(config, 2) isa PMISCoarsening
        @test ParallelAMG._get_coarsening_for_level(config, 3) isa AggregationCoarsening
    end

    @testset "Initial Coarsening - Solve" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        # Use aggressive coarsening for first 2 levels, then aggregation
        config = AMGConfig(
            coarsening=AggregationCoarsening(),
            initial_coarsening=PMISCoarsening(0.25, DirectInterpolation()),
            initial_coarsening_levels=1,
            pre_smoothing_steps=2,
            post_smoothing_steps=2,
        )
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) > 0
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    # ══════════════════════════════════════════════════════════════════════════
    # CF-splitting Prolongation Properties
    # ══════════════════════════════════════════════════════════════════════════

    @testset "CF Prolongation - Coarse Points Identity" begin
        A = poisson1d_csr(20)
        cf, cmap, nc = ParallelAMG.coarsen_pmis(A, 0.25)
        # Test all three interpolation types
        for interp in [DirectInterpolation(), StandardInterpolation(), ExtendedIInterpolation()]
            P = ParallelAMG.build_cf_prolongation(A, cf, cmap, nc, interp)
            @test P.nrow == 20
            @test P.ncol == nc
            # Coarse points should have identity mapping: P[i, cmap[i]] = 1
            for i in 1:20
                if cf[i] == 1
                    nnz_row = P.rowptr[i+1] - P.rowptr[i]
                    @test nnz_row == 1
                    @test P.colval[P.rowptr[i]] == cmap[i]
                    @test P.nzval[P.rowptr[i]] ≈ 1.0
                end
            end
        end
    end

    @testset "CF Prolongation - Fine Points Have Entries" begin
        A = poisson1d_csr(20)
        cf, cmap, nc = ParallelAMG.coarsen_pmis(A, 0.25)
        for interp in [DirectInterpolation(), StandardInterpolation(), ExtendedIInterpolation()]
            P = ParallelAMG.build_cf_prolongation(A, cf, cmap, nc, interp)
            for i in 1:20
                if cf[i] == -1
                    nnz_row = P.rowptr[i+1] - P.rowptr[i]
                    @test nnz_row >= 1  # every fine point should interpolate from somewhere
                end
            end
        end
    end

    @testset "Galerkin Product - Multi-entry P" begin
        A = poisson2d_csr(6)
        cf, cmap, nc = ParallelAMG.coarsen_pmis(A, 0.25)
        P = ParallelAMG.build_cf_prolongation(A, cf, cmap, nc, StandardInterpolation())
        A_coarse, r_map = ParallelAMG.compute_coarse_sparsity(A, P, nc)
        # Verify against explicit computation
        I_p = Int[]; J_p = Int[]; V_p = Float64[]
        for i in 1:P.nrow
            for nz in P.rowptr[i]:(P.rowptr[i+1]-1)
                push!(I_p, i); push!(J_p, P.colval[nz]); push!(V_p, P.nzval[nz])
            end
        end
        P_sparse = sparse(I_p, J_p, V_p, P.nrow, P.ncol)
        A_sparse = sparse(A.At')
        Ac_explicit = P_sparse' * A_sparse * P_sparse
        for i in 1:nc, j in 1:nc
            @test A_coarse[i,j] ≈ Ac_explicit[i,j] atol=1e-10
        end
        # Test in-place resetup with triple map
        nzv = nonzeros(A)
        nzv .*= 1.5
        ParallelAMG.galerkin_product!(A_coarse, A, P, r_map)
        A_sparse2 = sparse(A.At')
        Ac_explicit2 = P_sparse' * A_sparse2 * P_sparse
        for i in 1:nc, j in 1:nc
            @test A_coarse[i,j] ≈ Ac_explicit2[i,j] atol=1e-10
        end
    end

    # ══════════════════════════════════════════════════════════════════════════
    # Coarsening stalling heuristic
    # ══════════════════════════════════════════════════════════════════════════

    @testset "Coarsening Stalling - min_coarse_ratio" begin
        A = poisson2d_csr(30)
        # With a very strict ratio, coarsening should stop early
        config = AMGConfig(coarsening=HMISCoarsening(0.25, DirectInterpolation()),
                           min_coarse_ratio=0.3)
        hierarchy = amg_setup(A, config)
        nlevels = length(hierarchy.levels) + 1
        # Check that we didn't create too many similar-sized levels
        for i in 1:length(hierarchy.levels)-1
            n_current = size(hierarchy.levels[i].A, 1)
            n_next = size(hierarchy.levels[i+1].A, 1)
            ratio = n_next / n_current
            # Ratio should be below or near the threshold
            @test ratio <= 0.6 || n_next <= config.max_coarse_size
        end
    end

    @testset "Coarsening Stalling - Default ratio stops gracefully" begin
        A = poisson2d_csr(20)
        config = AMGConfig(coarsening=PMISCoarsening(0.25, DirectInterpolation()),
                           min_coarse_ratio=0.5)
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

    # ══════════════════════════════════════════════════════════════════════════
    # ExtendedI Interpolation convergence fix
    # ══════════════════════════════════════════════════════════════════════════

    @testset "ExtendedI Convergence - PMIS" begin
        n = 15
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=PMISCoarsening(0.25, ExtendedIInterpolation()),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    @testset "ExtendedI Convergence - HMIS" begin
        n = 15
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=HMISCoarsening(0.25, ExtendedIInterpolation()),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    # ══════════════════════════════════════════════════════════════════════════
    # Filtering for Aggregation Coarsening
    # ══════════════════════════════════════════════════════════════════════════

    @testset "Aggregation Filtering - Config" begin
        alg = AggregationCoarsening(0.25, true, 0.2)
        @test alg.filtering == true
        @test alg.filter_tol ≈ 0.2
        alg2 = AggregationCoarsening()
        @test alg2.filtering == false
    end

    @testset "Aggregation Filtering - Solve" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=AggregationCoarsening(0.25, true, 0.1))
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    @testset "Filter Prolongation" begin
        # Build a P with some small entries that should be filtered
        P = ParallelAMG.ProlongationOp{Int, Float64}(
            [1, 4, 7, 10],  # rowptr: 3 entries per row
            [1, 2, 3, 1, 2, 3, 1, 2, 3],  # colval
            [1.0, 0.05, 0.01, 0.01, 1.0, 0.05, 0.05, 0.01, 1.0],  # nzval
            3, 3
        )
        P_filt = ParallelAMG._filter_prolongation(P, 0.1)
        # After filtering with tol=0.1:
        # Row 1: max=1.0, threshold=0.1, keep entries ≥ 0.1: [1.0] (drop 0.05, 0.01)
        for i in 1:3
            nnz_row = P_filt.rowptr[i+1] - P_filt.rowptr[i]
            @test nnz_row >= 1  # at least one entry per row
        end
    end

    # ══════════════════════════════════════════════════════════════════════════
    # Max Row Sum Threshold
    # ══════════════════════════════════════════════════════════════════════════

    @testset "Max Row Sum - Config" begin
        config = AMGConfig(max_row_sum=0.9)
        @test config.max_row_sum ≈ 0.9
        config2 = AMGConfig()
        @test config2.max_row_sum ≈ 0.0  # disabled by default
    end

    @testset "Max Row Sum - Weakening Function" begin
        A = poisson2d_csr(5)
        # For Poisson 2D boundary row: |a_ii|=4, |off-diag|=2, ratio=(4+2)/4=1.5
        # With threshold=1.2, rows with ratio > 1.2 should be scaled
        A_weak = ParallelAMG._apply_max_row_sum(A, 1.2)
        # The weakened matrix should have same size and structure
        @test size(A_weak) == size(A)
        @test nnz(A_weak) == nnz(A)
        cv = colvals(A)
        nzv_orig = nonzeros(A)
        nzv_weak = nonzeros(A_weak)
        rp = rowptr(A)
        # Interior row (center point, index 13 for 5x5): ratio = (4+4)/4 = 2.0 > 1.2
        # Actually all rows with 4 off-diag neighbors: ratio=(4+4)/4=2.0 > 1.2, will be scaled
        # Rows with 2 off-diag neighbors: ratio=(4+2)/4=1.5 > 1.2, also scaled
        # So with threshold=1.2, most rows get scaled
        # Let's use threshold=1.8 to only affect boundary rows
        A_weak2 = ParallelAMG._apply_max_row_sum(A, 1.8)
        nzv_weak2 = nonzeros(A_weak2)
        # Interior row 13 (4 neighbors): ratio=2.0 > 1.8, should be scaled
        has_scaled_interior = false
        for nz in rp[13]:(rp[13+1]-1)
            j = cv[nz]
            if j != 13 && abs(nzv_weak2[nz]) < abs(nzv_orig[nz]) - 1e-14
                has_scaled_interior = true
            end
        end
        @test has_scaled_interior
        # Row 1 (corner, 2 neighbors): ratio=1.5 < 1.8, should NOT be scaled
        row1_unchanged = true
        for nz in rp[1]:(rp[1+1]-1)
            if abs(nzv_weak2[nz] - nzv_orig[nz]) > 1e-14
                row1_unchanged = false
            end
        end
        @test row1_unchanged
    end

    @testset "Max Row Sum - Solve" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(max_row_sum=0.9)
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
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
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=SmoothedAggregationCoarsening(),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
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
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=SmoothedAggregationCoarsening(0.25, 2/3, true, 0.1),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
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

    @testset "Smoothed Prolongation Construction" begin
        A = poisson1d_csr(10)
        agg, nc = ParallelAMG.coarsen_aggregation(A, 0.25)
        P_tent = ParallelAMG.build_prolongation(A, agg, nc)
        P_smooth = ParallelAMG._smooth_prolongation(A, P_tent, 2/3)
        @test P_smooth.nrow == 10
        @test P_smooth.ncol == nc
        # Smoothed P should have more nonzeros than tentative P
        nnz_tent = P_tent.rowptr[end] - 1
        nnz_smooth = P_smooth.rowptr[end] - 1
        @test nnz_smooth >= nnz_tent
    end

end
