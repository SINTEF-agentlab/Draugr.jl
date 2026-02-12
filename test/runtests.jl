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

end
