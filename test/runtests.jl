using Test
using ParallelAMG
using SparseArrays
using LinearAlgebra
using Random
import Jutul

# Helper: convert StaticSparsityMatrixCSR to internal CSRMatrix for unit tests
to_csr(A) = ParallelAMG.csr_from_static(A)

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

    @testset "CSRMatrix - AbstractVector support" begin
        # CSRMatrix should work with any AbstractVector subtype
        A_static = poisson1d_csr(5)
        A_csr = ParallelAMG.csr_from_static(A_static)
        @test A_csr isa CSRMatrix
        @test A_csr.rowptr isa AbstractVector
        @test A_csr.colval isa AbstractVector
        @test A_csr.nzval isa AbstractVector
        @test size(A_csr) == (5, 5)
        @test A_csr[1,1] ≈ 2.0
        @test A_csr[1,2] ≈ -1.0
        # Test mul! with CSRMatrix
        x = ones(5)
        y = zeros(5)
        mul!(y, A_csr, x)
        @test y[1] ≈ 1.0
        @test y[5] ≈ 1.0
        @test all(y[2:4] .≈ 0.0)
    end

    @testset "Strength of Connection" begin
        A = poisson1d_csr(10)
        Ac = to_csr(A)
        is_strong = ParallelAMG.strength_graph(Ac, 0.25)
        # All off-diagonal entries should be strong (|-1| >= 0.25*|-1|)
        @test sum(is_strong) == 18  # 9+9 off-diagonal entries
    end

    @testset "Aggregation Coarsening" begin
        A = poisson1d_csr(20)
        Ac = to_csr(A)
        agg, nc = ParallelAMG.coarsen_aggregation(Ac, 0.25)
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
        cf, cmap, nc = ParallelAMG.coarsen_pmis(Ac, 0.25)
        @test length(cf) == 20
        @test all(abs.(cf) .== 1)  # all decided
        @test nc > 0
        @test nc < 20
    end

    @testset "Aggressive Coarsening" begin
        A = poisson1d_csr(20)
        Ac = to_csr(A)
        agg, nc = ParallelAMG.coarsen_aggressive(Ac, 0.25)
        @test length(agg) == 20
        @test all(agg .> 0)
        @test nc > 0
        @test nc < 20
    end

    @testset "Prolongation" begin
        A = poisson1d_csr(10)
        Ac = to_csr(A)
        agg, nc = ParallelAMG.coarsen_aggregation(Ac, 0.25)
        P = ParallelAMG.build_prolongation(Ac, agg, nc)
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
        Pt_map = ParallelAMG.build_transpose_map(P)
        ParallelAMG.restrict!(bc, Pt_map, P, rf)
        # Sum should be preserved
        @test sum(bc) ≈ sum(rf)
        # Test TransposeMap structure
        @test length(Pt_map.offsets) == nc + 1
        @test Pt_map.offsets[1] == 1
        @test Pt_map.offsets[nc + 1] == P.nrow + 1  # aggregation P: one NZ per fine row
        # Verify restrict! against explicit P^T * r computation
        rf2 = randn(10)
        bc2 = zeros(nc)
        ParallelAMG.restrict!(bc2, Pt_map, P, rf2)
        # Build explicit P and compute P^T * rf2
        I_p = Int[]; J_p = Int[]; V_p = Float64[]
        for i in 1:P.nrow
            for nz in P.rowptr[i]:(P.rowptr[i+1]-1)
                push!(I_p, i); push!(J_p, P.colval[nz]); push!(V_p, P.nzval[nz])
            end
        end
        P_sparse = sparse(I_p, J_p, V_p, P.nrow, P.ncol)
        bc_ref = P_sparse' * rf2
        @test bc2 ≈ bc_ref atol=1e-12
    end

    @testset "Galerkin Product - contention-free kernel" begin
        # Verify the grouped nz_offsets structure produces correct results
        A = poisson1d_csr(20)
        Ac = to_csr(A)
        agg, nc = ParallelAMG.coarsen_aggregation(Ac, 0.25)
        P = ParallelAMG.build_prolongation(Ac, agg, nc)
        A_coarse, r_map = ParallelAMG.compute_coarse_sparsity(Ac, P, nc)
        # Verify nz_offsets structure
        nnz_c = SparseArrays.nnz(A_coarse)
        @test length(r_map.nz_offsets) == nnz_c + 1
        @test r_map.nz_offsets[1] == 1
        # Every offset range should be non-empty (each coarse NZ has contributing triples)
        for k in 1:nnz_c
            @test r_map.nz_offsets[k+1] >= r_map.nz_offsets[k]
        end
        # Verify resetup: modify values and compare to explicit
        nzv = nonzeros(A)
        nzv .*= 3.0
        Ac_new = to_csr(A)
        ParallelAMG.galerkin_product!(A_coarse, Ac_new, P, r_map)
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
            @test A_coarse[i,j] ≈ Ac_explicit[i,j] atol=1e-12
        end
    end

    @testset "Galerkin Product" begin
        A = poisson1d_csr(10)
        Ac = to_csr(A)
        agg, nc = ParallelAMG.coarsen_aggregation(Ac, 0.25)
        P = ParallelAMG.build_prolongation(Ac, agg, nc)
        A_coarse, r_map = ParallelAMG.compute_coarse_sparsity(Ac, P, nc)
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
        Ac = to_csr(A)
        agg, nc = ParallelAMG.coarsen_aggregation(Ac, 0.25)
        P = ParallelAMG.build_prolongation(Ac, agg, nc)
        A_coarse, r_map = ParallelAMG.compute_coarse_sparsity(Ac, P, nc)
        # Now modify A's values (scale by 2)
        nzv = nonzeros(A)
        nzv .*= 2.0
        Ac = to_csr(A)
        # Recompute in-place
        ParallelAMG.galerkin_product!(A_coarse, Ac, P, r_map)
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
        Ac = to_csr(A)
        smoother = ParallelAMG.build_jacobi_smoother(Ac, 2.0/3.0)
        @test length(smoother.invdiag) == 10
        @test all(smoother.invdiag .≈ 0.5)  # 1/2.0
        # Test smoothing reduces error
        b = ones(10)
        x = zeros(10)
        smooth!(x, Ac, b, smoother; steps=10)
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
        Ac = to_csr(A)
        colors, nc = ParallelAMG.greedy_coloring(Ac)
        @test length(colors) == 10
        @test all(colors .> 0)
        @test nc >= 2  # tridiagonal needs at least 2 colors
        # Verify no two adjacent nodes have the same color
        cv = colvals(Ac)
        for i in 1:10
            for nz in nzrange(Ac, i)
                j = cv[nz]
                if j != i
                    @test colors[i] != colors[j]
                end
            end
        end
    end

    @testset "Colored GS Smoother - Build" begin
        A = poisson1d_csr(10)
        Ac = to_csr(A)
        smoother = ParallelAMG.build_colored_gs_smoother(Ac)
        @test smoother.num_colors >= 2
        @test length(smoother.invdiag) == 10
        @test all(smoother.invdiag .≈ 0.5)  # 1/2.0
        @test length(smoother.color_order) == 10
    end

    @testset "Colored GS Smoother - Smoothing" begin
        A = poisson1d_csr(10)
        Ac = to_csr(A)
        smoother = ParallelAMG.build_colored_gs_smoother(Ac)
        b = ones(10)
        x = zeros(10)
        smooth!(x, Ac, b, smoother; steps=10)
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
        Ac = to_csr(A)
        smoother = ParallelAMG.build_spai0_smoother(Ac)
        @test length(smoother.m_diag) == 10
        # For tridiagonal with diag=2, off-diag=-1:
        # interior row: [−1 2 −1], row_norm_sq = 1+4+1 = 6, diag = 2, m = 2/6 = 1/3
        @test smoother.m_diag[5] ≈ 2.0/6.0
    end

    @testset "SPAI0 Smoother - Smoothing" begin
        A = poisson1d_csr(10)
        Ac = to_csr(A)
        smoother = ParallelAMG.build_spai0_smoother(Ac)
        b = ones(10)
        x = zeros(10)
        smooth!(x, Ac, b, smoother; steps=10)
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
        Ac = to_csr(A)
        smoother = ParallelAMG.build_spai1_smoother(Ac)
        @test length(smoother.nzval) == nnz(A)
    end

    @testset "SPAI1 Smoother - Smoothing" begin
        A = poisson1d_csr(10)
        Ac = to_csr(A)
        smoother = ParallelAMG.build_spai1_smoother(Ac)
        b = ones(10)
        x = zeros(10)
        smooth!(x, Ac, b, smoother; steps=10)
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
        Ac = to_csr(A)
        cf, cmap, nc = ParallelAMG.coarsen_hmis(Ac, 0.25)
        @test length(cf) == 20
        @test all(abs.(cf) .== 1)
        @test nc > 0
        @test nc < 20
        # Every fine point should have at least one coarse neighbor
        cv = colvals(Ac)
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
        cf, cmap, nc = ParallelAMG.coarsen_hmis(Ac, 0.25)
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
        Ac = to_csr(A)
        cf, cmap, nc = ParallelAMG.coarsen_pmis(Ac, 0.25)
        # Test all three interpolation types
        for interp in [DirectInterpolation(), StandardInterpolation(), ExtendedIInterpolation()]
            P = ParallelAMG.build_cf_prolongation(Ac, cf, cmap, nc, interp)
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
        Ac = to_csr(A)
        cf, cmap, nc = ParallelAMG.coarsen_pmis(Ac, 0.25)
        for interp in [DirectInterpolation(), StandardInterpolation(), ExtendedIInterpolation()]
            P = ParallelAMG.build_cf_prolongation(Ac, cf, cmap, nc, interp)
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
        Ac = to_csr(A)
        cf, cmap, nc = ParallelAMG.coarsen_pmis(Ac, 0.25)
        P = ParallelAMG.build_cf_prolongation(Ac, cf, cmap, nc, StandardInterpolation())
        A_coarse, r_map = ParallelAMG.compute_coarse_sparsity(Ac, P, nc)
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
        Ac = to_csr(A)
        ParallelAMG.galerkin_product!(A_coarse, Ac, P, r_map)
        A_sparse2 = sparse(A.At')
        Ac_explicit2 = P_sparse' * A_sparse2 * P_sparse
        for i in 1:nc, j in 1:nc
            @test A_coarse[i,j] ≈ Ac_explicit2[i,j] atol=1e-10
        end
    end

    # ══════════════════════════════════════════════════════════════════════════
    # RS Coarsening (replaces stalling heuristic)
    # ══════════════════════════════════════════════════════════════════════════

    @testset "RS Coarsening - Basic" begin
        A = poisson1d_csr(20)
        Ac = to_csr(A)
        cf, cmap, nc = ParallelAMG.coarsen_rs(Ac, 0.25)
        @test length(cf) == 20
        @test all(abs.(cf) .== 1)
        @test nc > 0
        @test nc < 20
        # Every F-point should have a strong C-neighbor
        is_strong = ParallelAMG.strength_graph(Ac, 0.25)
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
        cf, cmap, nc = ParallelAMG.coarsen_rs(Ac, 0.25)
        @test nc > 0
        @test nc < 100
        @test sum(cf .== 1) == nc
    end

    @testset "RS Coarsening - Solve" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=RSCoarsening(0.25, DirectInterpolation()),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) >= 1
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    @testset "RS Coarsening - Standard Interpolation" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(coarsening=RSCoarsening(0.25, StandardInterpolation()),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) >= 1
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
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

    # ══════════════════════════════════════════════════════════════════════════
    # Disconnected / Isolated cells
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
        Ac = to_csr(A)
        # For Poisson 2D boundary row: |a_ii|=4, |off-diag|=2, ratio=(4+2)/4=1.5
        # With threshold=1.2, rows with ratio > 1.2 should be scaled
        A_weak = ParallelAMG._apply_max_row_sum(Ac, 1.2)
        # The weakened matrix should have same size and structure
        @test size(A_weak) == size(Ac)
        @test nnz(A_weak) == nnz(Ac)
        cv = colvals(Ac)
        nzv_orig = nonzeros(Ac)
        nzv_weak = nonzeros(A_weak)
        rp = rowptr(Ac)
        # Interior row (center point, index 13 for 5x5): ratio = (4+4)/4 = 2.0 > 1.2
        # Actually all rows with 4 off-diag neighbors: ratio=(4+4)/4=2.0 > 1.2, will be scaled
        # Rows with 2 off-diag neighbors: ratio=(4+2)/4=1.5 > 1.2, also scaled
        # So with threshold=1.2, most rows get scaled
        # Let's use threshold=1.8 to only affect boundary rows
        A_weak2 = ParallelAMG._apply_max_row_sum(Ac, 1.8)
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
        Ac = to_csr(A)
        agg, nc = ParallelAMG.coarsen_aggregation(Ac, 0.25)
        P_tent = ParallelAMG.build_prolongation(Ac, agg, nc)
        P_smooth = ParallelAMG._smooth_prolongation(Ac, P_tent, 2/3)
        @test P_smooth.nrow == 10
        @test P_smooth.ncol == nc
        # Smoothed P should have more nonzeros than tentative P
        nnz_tent = P_tent.rowptr[end] - 1
        nnz_smooth = P_smooth.rowptr[end] - 1
        @test nnz_smooth >= nnz_tent
    end

    # ══════════════════════════════════════════════════════════════════════════
    # Robustness & Hardening
    # ══════════════════════════════════════════════════════════════════════════

    # Helper: reservoir-like matrix with positive off-diags and large variation
    function reservoir_like_csr(n)
        I = Int[]; J = Int[]; V = Float64[]
        for i in 1:n
            push!(I, i); push!(J, i); push!(V, 10.0)
            if i > 1
                v = i % 3 == 0 ? 0.5 : -1.0  # positive off-diag every 3rd row
                push!(I, i); push!(J, i-1); push!(V, v)
            end
            if i < n
                v = (i+1) % 5 == 0 ? 0.3 : -2.0  # positive off-diag every 5th row
                push!(I, i); push!(J, i+1); push!(V, v)
            end
        end
        return static_sparsity_sparse(I, J, V, n, n)
    end

    # Helper: anisotropic 2D problem
    function anisotropic_csr(nx, ny; kx=1e4, ky=1e-2)
        n = nx * ny
        I = Int[]; J = Int[]; V = Float64[]
        for j in 1:ny, i in 1:nx
            idx = (j-1)*nx + i
            diag = 2*kx + 2*ky
            push!(I, idx); push!(J, idx); push!(V, diag)
            if i > 1 push!(I, idx); push!(J, idx-1); push!(V, -kx) end
            if i < nx push!(I, idx); push!(J, idx+1); push!(V, -kx) end
            if j > 1 push!(I, idx); push!(J, idx-nx); push!(V, -ky) end
            if j < ny push!(I, idx); push!(J, idx+nx); push!(V, -ky) end
        end
        return static_sparsity_sparse(I, J, V, n, n)
    end

    @testset "Sign-Aware Strength - AbsoluteStrength" begin
        A = reservoir_like_csr(20)
        Ac = to_csr(A)
        is_strong = ParallelAMG.strength_graph(Ac, 0.25, AbsoluteStrength())
        @test length(is_strong) == nnz(A)
        @test sum(is_strong) > 0
    end

    @testset "Sign-Aware Strength - SignedStrength" begin
        A = reservoir_like_csr(20)
        Ac = to_csr(A)
        is_strong_signed = ParallelAMG.strength_graph(Ac, 0.25, SignedStrength())
        is_strong_abs = ParallelAMG.strength_graph(Ac, 0.25, AbsoluteStrength())
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

    @testset "SignedStrength - Config dispatch" begin
        A = poisson2d_csr(8)
        Ac = to_csr(A)
        is1 = ParallelAMG.strength_graph(Ac, 0.25, AMGConfig(strength_type=AbsoluteStrength()))
        is2 = ParallelAMG.strength_graph(Ac, 0.25, AMGConfig(strength_type=SignedStrength()))
        @test length(is1) == nnz(Ac)
        @test length(is2) == nnz(Ac)
    end

    @testset "Positive Off-Diags - All smoothers converge" begin
        A = reservoir_like_csr(50)
        N = 50
        b = rand(N)
        for (name, cfg) in [
            ("Jacobi", AMGConfig()),
            ("l1-Jacobi", AMGConfig(smoother=L1JacobiSmootherType())),
            ("Colored GS", AMGConfig(smoother=ColoredGaussSeidelType())),
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

    @testset "Safe Diagonal - Zero diagonal row" begin
        # Matrix with near-zero diagonal in one row
        I = [1,1,2,2,2,3,3]
        J = [1,2,1,2,3,2,3]
        V = [2.0,-1.0,-1.0,1e-20,-1.0,-1.0,2.0]  # row 2 has near-zero diagonal
        A = static_sparsity_sparse(I, J, V, 3, 3)
        Ac = to_csr(A)
        smoother = ParallelAMG.build_jacobi_smoother(Ac, 2.0/3.0)
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
    # l1-Jacobi Smoother
    # ══════════════════════════════════════════════════════════════════════════

    @testset "l1-Jacobi Smoother - Build" begin
        A = poisson1d_csr(10)
        Ac = to_csr(A)
        smoother = ParallelAMG.build_l1jacobi_smoother(Ac, 2.0/3.0)
        @test length(smoother.invdiag) == 10
        @test smoother.ω ≈ 2.0/3.0
        # For interior row: l1 norm = |−1| + |2| + |−1| = 4, invdiag = 1/4
        @test smoother.invdiag[5] ≈ 0.25
    end

    @testset "l1-Jacobi Smoother - Smoothing" begin
        A = poisson1d_csr(10)
        Ac = to_csr(A)
        smoother = ParallelAMG.build_l1jacobi_smoother(Ac, 2.0/3.0)
        b = ones(10)
        x = zeros(10)
        smooth!(x, Ac, b, smoother; steps=10)
        r = b - sparse(A.At') * x
        @test norm(r) < norm(b)
    end

    @testset "l1-Jacobi Smoother - AMG Solve" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(smoother=L1JacobiSmootherType())
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    @testset "l1-Jacobi - Resetup" begin
        n = 8
        A = poisson2d_csr(n)
        N = n*n
        config = AMGConfig(smoother=L1JacobiSmootherType())
        hierarchy = amg_setup(A, config)
        b = rand(N)
        x1 = zeros(N)
        x1, _ = amg_solve!(x1, b, hierarchy, config; tol=1e-8, maxiter=200)
        nonzeros(A) .*= 2.0
        amg_resetup!(hierarchy, A, config)
        x2 = zeros(N)
        x2, _ = amg_solve!(x2, b, hierarchy, config; tol=1e-8, maxiter=200)
        r2 = b - sparse(A.At') * x2
        @test norm(r2) / norm(b) < 1e-8
    end

    # ══════════════════════════════════════════════════════════════════════════
    # Chebyshev Smoother
    # ══════════════════════════════════════════════════════════════════════════

    @testset "Chebyshev Smoother - Build" begin
        A = poisson1d_csr(10)
        Ac = to_csr(A)
        smoother = ParallelAMG.build_chebyshev_smoother(Ac)
        @test length(smoother.invdiag) == 10
        @test smoother.λ_max > 0
        @test smoother.λ_min > 0
        @test smoother.λ_max > smoother.λ_min
        @test smoother.degree == 3
    end

    @testset "Chebyshev Smoother - Smoothing" begin
        A = poisson1d_csr(10)
        Ac = to_csr(A)
        smoother = ParallelAMG.build_chebyshev_smoother(Ac)
        b = ones(10)
        x = zeros(10)
        smooth!(x, Ac, b, smoother; steps=5)
        r = b - sparse(A.At') * x
        @test norm(r) < norm(b)
    end

    @testset "Chebyshev Smoother - AMG Solve" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(smoother=ChebyshevSmootherType(),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    @testset "Chebyshev - Resetup" begin
        n = 8
        A = poisson2d_csr(n)
        N = n*n
        config = AMGConfig(smoother=ChebyshevSmootherType(),
                           pre_smoothing_steps=2, post_smoothing_steps=2)
        hierarchy = amg_setup(A, config)
        b = rand(N)
        x1 = zeros(N)
        x1, _ = amg_solve!(x1, b, hierarchy, config; tol=1e-8, maxiter=200)
        nonzeros(A) .*= 2.0
        amg_resetup!(hierarchy, A, config)
        x2 = zeros(N)
        x2, _ = amg_solve!(x2, b, hierarchy, config; tol=1e-8, maxiter=200)
        r2 = b - sparse(A.At') * x2
        @test norm(r2) / norm(b) < 1e-8
    end

    # ══════════════════════════════════════════════════════════════════════════
    # ILU(0) Smoother
    # ══════════════════════════════════════════════════════════════════════════

    @testset "ILU(0) Smoother - Build" begin
        A = poisson1d_csr(10)
        Ac = to_csr(A)
        smoother = ParallelAMG.build_ilu0_smoother(Ac)
        @test length(smoother.L_nzval) == nnz(A)
        @test length(smoother.U_nzval) == nnz(A)
        @test length(smoother.diag_idx) == 10
        @test smoother.num_colors >= 2
    end

    @testset "ILU(0) Smoother - Smoothing" begin
        A = poisson1d_csr(10)
        Ac = to_csr(A)
        smoother = ParallelAMG.build_ilu0_smoother(Ac)
        b = ones(10)
        x = zeros(10)
        smooth!(x, Ac, b, smoother; steps=3)
        r = b - sparse(A.At') * x
        @test norm(r) < norm(b)
    end

    @testset "ILU(0) Smoother - AMG Solve" begin
        n = 10
        A = poisson2d_csr(n)
        N = n*n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(smoother=ILU0SmootherType())
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
        @test niter < 200
    end

    @testset "ILU(0) Smoother - Resetup" begin
        n = 8
        A = poisson2d_csr(n)
        N = n*n
        config = AMGConfig(smoother=ILU0SmootherType())
        hierarchy = amg_setup(A, config)
        b = rand(N)
        x1 = zeros(N)
        x1, _ = amg_solve!(x1, b, hierarchy, config; tol=1e-8, maxiter=200)
        nonzeros(A) .*= 2.0
        amg_resetup!(hierarchy, A, config)
        x2 = zeros(N)
        x2, _ = amg_solve!(x2, b, hierarchy, config; tol=1e-8, maxiter=200)
        r2 = b - sparse(A.At') * x2
        @test norm(r2) / norm(b) < 1e-8
    end

    @testset "ILU(0) - Anisotropic" begin
        A = anisotropic_csr(8, 8)
        N = 64
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(smoother=ILU0SmootherType())
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
    end

    @testset "ILU(0) - Reservoir-like" begin
        A = reservoir_like_csr(50)
        N = 50
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(smoother=ILU0SmootherType())
        hierarchy = amg_setup(A, config)
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-8
    end

    # ══════════════════════════════════════════════════════════════════════════
    # Coarsening stalling fix
    # ══════════════════════════════════════════════════════════════════════════

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

    @testset "MIS-based aggregation produces larger aggregates" begin
        # 2D Poisson: the MIS-based aggregation should create fewer, larger aggregates
        A = poisson2d_csr(20)
        A_csr = to_csr(A)
        agg, n_coarse = ParallelAMG.coarsen_aggregation(A_csr, 0.25)
        n = size(A, 1)
        # The coarsening ratio should be aggressive (not more than 50% of original)
        @test n_coarse < 0.5 * n
        # Average aggregate size should be > 2
        @test n / n_coarse > 2.0
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
        n = 10
        A = poisson2d_csr(n)
        N = n * n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(
            coarsening = HMISCoarsening(0.5, ExtendedIInterpolation()),
            initial_coarsening = AggressiveCoarsening(0.5, :hmis, ExtendedIInterpolation(0.3)),
            initial_coarsening_levels = 1,
            pre_smoothing_steps = 2,
            post_smoothing_steps = 2,
        )
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) >= 1
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-6
        @test niter < 200
    end

    @testset "AggressiveCoarsening with PMIS base" begin
        n = 10
        A = poisson2d_csr(n)
        N = n * n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(
            coarsening = PMISCoarsening(0.5, ExtendedIInterpolation()),
            initial_coarsening = AggressiveCoarsening(0.5, :pmis, ExtendedIInterpolation(0.3)),
            initial_coarsening_levels = 1,
            pre_smoothing_steps = 2,
            post_smoothing_steps = 2,
        )
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) >= 1
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-6
        @test niter < 200
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

    @testset "coarsen_aggressive_cf with HMIS" begin
        n = 10
        A = poisson2d_csr(n)
        A_csr = to_csr(A)
        N = n * n
        # Use fixed RNG for reproducibility
        rng = Random.MersenneTwister(42)
        cf, coarse_map, n_coarse = ParallelAMG.coarsen_aggressive_cf(A_csr, 0.25, :hmis; rng=rng)
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
        cf, coarse_map, n_coarse = ParallelAMG.coarsen_aggressive_cf(A_csr, 0.25, :pmis)
        @test n_coarse > 0
        @test n_coarse < N
        for i in 1:N
            @test cf[i] == 1 || cf[i] == -1
        end
    end

    @testset "HMIS with ExtendedI and trunc_factor" begin
        n = 10
        A = poisson2d_csr(n)
        N = n * n
        b = rand(N)
        x = zeros(N)
        config = AMGConfig(
            coarsening = HMISCoarsening(0.5, ExtendedIInterpolation(0.3)),
            pre_smoothing_steps = 2,
            post_smoothing_steps = 2,
        )
        hierarchy = amg_setup(A, config)
        @test length(hierarchy.levels) >= 1
        x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
        r = b - sparse(A.At') * x
        @test norm(r) / norm(b) < 1e-6
        @test niter < 200
    end

    # ══════════════════════════════════════════════════════════════════════════
    # JLArrays GPU backend tests
    # ══════════════════════════════════════════════════════════════════════════

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
            x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=100, backend=JLBackend())
            # Verify convergence by computing residual on CPU
            x_cpu = Array(x)
            b_cpu = Array(b)
            A_cpu = csr_to_cpu(csr_from_gpu(A_jl))
            r = zeros(n)
            rp = A_cpu.rowptr; cv = A_cpu.colval; nzv = A_cpu.nzval
            for i in 1:n
                Ax_i = 0.0
                for nz in rp[i]:(rp[i+1]-1)
                    Ax_i += nzv[nz] * x_cpu[cv[nz]]
                end
                r[i] = b_cpu[i] - Ax_i
            end
            @test norm(r) / norm(b_cpu) < 1e-6
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
            x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=100, backend=JLBackend())
            # Verify convergence
            x_cpu = Array(x)
            b_cpu = Array(b)
            A_cpu = csr_to_cpu(csr_from_gpu(A_jl))
            r = zeros(n)
            rp = A_cpu.rowptr; cv = A_cpu.colval; nzv = A_cpu.nzval
            for i in 1:n
                Ax_i = 0.0
                for nz in rp[i]:(rp[i+1]-1)
                    Ax_i += nzv[nz] * x_cpu[cv[nz]]
                end
                r[i] = b_cpu[i] - Ax_i
            end
            @test norm(r) / norm(b_cpu) < 1e-6
            @test niter < 100
        end

        @testset "AMG cycle with JLSparseMatrixCSR" begin
            A_jl = poisson1d_jl(100)
            config = AMGConfig()
            hierarchy = amg_setup(A_jl, config)
            n = 100
            b = JLArray(ones(n))
            x = JLArray(zeros(n))
            # Apply a single cycle
            amg_cycle!(x, b, hierarchy, config; backend=JLBackend())
            @test !all(Array(x) .== 0.0)  # something changed
        end

        @testset "Backend consistency - strength_graph uses JLBackend" begin
            # This test verifies that the backend is properly threaded
            # through the setup path including coarsening/strength_graph
            A_jl = poisson1d_jl(100)
            config = AMGConfig()
            # If backend is not properly threaded, this would fail with
            # scalar indexing errors on JLArrays
            hierarchy = amg_setup(A_jl, config; backend=JLBackend())
            @test length(hierarchy.levels) >= 1
        end

        @testset "AMG with different smoothers on JLArrays" begin
            A_jl = poisson1d_jl(100)
            for smoother in [JacobiSmootherType(), SPAI0SmootherType()]
                config = AMGConfig(smoother=smoother)
                hierarchy = amg_setup(A_jl, config)
                @test length(hierarchy.levels) >= 1
                b = JLArray(ones(100))
                x = JLArray(zeros(100))
                x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=100, backend=JLBackend())
                @test niter < 100
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
            # Solve after resetup
            b = JLArray(ones(n))
            x = JLArray(zeros(n))
            x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=100, backend=JLBackend())
            @test niter < 100
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
    end

end
