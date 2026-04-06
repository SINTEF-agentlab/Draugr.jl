@testset "SparseMatricesCSR Extension" begin
    using SparseMatricesCSR

    function poisson1d_sparsecsr(n)
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
        return sparsecsr(I, J, V, n, n)
    end

    @testset "amg_setup from SparseMatrixCSR" begin
        A = poisson1d_sparsecsr(100)
        h = amg_setup(A)
        @test h isa AMGHierarchy
        @test length(h.levels) >= 1
    end

    @testset "amg_solve from SparseMatrixCSR" begin
        A = poisson1d_sparsecsr(100)
        h = amg_setup(A)
        b = ones(100)
        x = zeros(100)
        x, iter = amg_solve!(x, b, h)
        # Verify convergence against CSC reference
        n = 100
        A_csc = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
        r = b - A_csc * x
        @test norm(r) / norm(b) < 1e-6
    end

    @testset "amg_resetup from SparseMatrixCSR" begin
        A = poisson1d_sparsecsr(50)
        h = amg_setup(A)
        # Resetup with scaled values
        I = Int[]; J = Int[]; V = Float64[]
        for i in 1:50
            push!(I, i); push!(J, i); push!(V, 4.0)
            if i > 1
                push!(I, i); push!(J, i-1); push!(V, -2.0)
            end
            if i < 50
                push!(I, i); push!(J, i+1); push!(V, -2.0)
            end
        end
        A2 = sparsecsr(I, J, V, 50, 50)
        amg_resetup!(h, A2)
        b = ones(50)
        x = zeros(50)
        x, iter = amg_solve!(x, b, h)
        @test iter < 100
    end

    @testset "csr_from_sparse_csr do_collect" begin
        A_sparse = poisson1d_sparsecsr(10)
        ext = Base.get_extension(Draugr, :DraugrSparseMatricesCSRExt)
        # Default (do_collect=false, 1-based): shares underlying arrays
        A_csr = ext.csr_from_sparse_csr(A_sparse)
        @test A_csr isa CSRMatrix
        @test size(A_csr) == (10, 10)
        @test A_csr[1,1] ≈ 2.0
        @test A_csr.nzval === nonzeros(A_sparse)

        # do_collect=true: independent copy
        A_csr2 = ext.csr_from_sparse_csr(A_sparse; do_collect=true)
        @test A_csr2 isa CSRMatrix
        @test size(A_csr2) == (10, 10)
        @test A_csr2[1,1] ≈ 2.0
        @test A_csr2.nzval !== nonzeros(A_sparse)
        @test A_csr2.nzval == nonzeros(A_sparse)
    end
end

@testset "LinearSolve Extension" begin
    using LinearSolve

    @testset "DraugrPreconditioner with solver=:linearsolve" begin
        prec = DraugrPreconditioner(solver=:linearsolve)
        @test prec isa AbstractDraugrPreconditioner
    end

    @testset "ldiv! with LinearSolve preconditioner" begin
        prec = DraugrPreconditioner(solver=:linearsolve)
        n = 50
        A_csc = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
        ext = Base.get_extension(Draugr, :DraugrLinearSolveExt)
        ext.update!(prec, A_csc)
        b = ones(n)
        x = zeros(n)
        ldiv!(x, prec, b)
        @test norm(x) > 0
    end

    @testset "LinearSolve GMRES with AMG preconditioner" begin
        n = 50
        A_csc = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
        prec = DraugrPreconditioner(solver=:linearsolve)
        ext = Base.get_extension(Draugr, :DraugrLinearSolveExt)
        ext.update!(prec, A_csc)
        b = rand(n)
        prob = LinearProblem(A_csc, b)
        sol = solve(prob, KrylovJL_GMRES(), Pl=prec)
        @test norm(A_csc * sol.u - b) / norm(b) < 1e-4
    end
end

@testset "Base DraugrPreconditioner" begin
    @testset "Standalone preconditioner" begin
        prec = DraugrPreconditioner()
        @test prec isa AbstractDraugrPreconditioner
        @test prec isa DraugrPreconditioner
        @test isnothing(prec.hierarchy)
        @test Draugr.preconditioner_nrows(prec) == 0
    end

    @testset "preconditioner_update! and preconditioner_apply!" begin
        prec = DraugrPreconditioner()
        n = 50
        A_csc = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
        A_csr = csr_from_csc(A_csc)
        Draugr.preconditioner_update!(prec, A_csr)
        @test !isnothing(prec.hierarchy)
        @test Draugr.preconditioner_nrows(prec) == n

        b = ones(n)
        x = zeros(n)
        Draugr.preconditioner_apply!(x, prec, b)
        @test norm(x) > 0
    end

    @testset "ldiv! on base preconditioner" begin
        prec = DraugrPreconditioner()
        n = 50
        A_csr = csr_from_csc(spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1)))
        Draugr.preconditioner_update!(prec, A_csr)
        b = ones(n)
        x = zeros(n)
        ldiv!(x, prec, b)
        @test norm(x) > 0
    end

    @testset "setup_specific_preconditioner error for unknown solver" begin
        @test_throws ErrorException DraugrPreconditioner(solver=:unknown_solver)
    end
end

@testset "SparseMatrixCSC convenience" begin
    @testset "amg_setup from SparseMatrixCSC" begin
        n = 50
        A_csc = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
        h = amg_setup(A_csc)
        @test h isa AMGHierarchy
    end

    @testset "amg_solve from SparseMatrixCSC" begin
        n = 50
        A_csc = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
        h = amg_setup(A_csc)
        b = ones(n)
        x = zeros(n)
        x, iter = amg_solve!(x, b, h)
        r = b - A_csc * x
        @test norm(r) / norm(b) < 1e-6
    end

    @testset "amg_resetup from SparseMatrixCSC" begin
        n = 50
        A_csc = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
        h = amg_setup(A_csc)
        A_csc2 = spdiagm(0 => fill(4.0, n), -1 => fill(-2.0, n-1), 1 => fill(-2.0, n-1))
        amg_resetup!(h, A_csc2)
        b = ones(n)
        x = zeros(n)
        x, iter = amg_solve!(x, b, h)
        @test iter < 100
    end

    @testset "csr_from_csc" begin
        A_csc = sparse([1,1,2,2,2,3,3], [1,2,1,2,3,2,3], [2.0,-1.0,-1.0,2.0,-1.0,-1.0,2.0], 3, 3)
        A_csr = csr_from_csc(A_csc)
        @test A_csr isa CSRMatrix
        @test size(A_csr) == (3, 3)
        @test A_csr[1,1] ≈ 2.0
        @test A_csr[1,2] ≈ -1.0
        @test A_csr[2,1] ≈ -1.0
    end

    @testset "csr_from_csc do_collect" begin
        A_csc = sparse([1,1,2,2,2,3,3], [1,2,1,2,3,2,3], [2.0,-1.0,-1.0,2.0,-1.0,-1.0,2.0], 3, 3)
        # Default (do_collect=false): arrays are not independently copied
        A_csr = csr_from_csc(A_csc)
        @test A_csr isa CSRMatrix
        @test size(A_csr) == (3, 3)
        @test A_csr[1,1] ≈ 2.0
        @test A_csr[1,2] ≈ -1.0

        # do_collect=true: arrays are independent copies
        A_csr2 = csr_from_csc(A_csc; do_collect=true)
        @test A_csr2 isa CSRMatrix
        @test size(A_csr2) == (3, 3)
        @test A_csr2[1,1] ≈ 2.0
        @test A_csr2[1,2] ≈ -1.0
    end

    @testset "csr_from_static do_collect" begin
        A_static = poisson1d_csr(10)
        # Default (do_collect=false): shares underlying arrays
        A_csr = Draugr.csr_from_static(A_static)
        @test A_csr isa CSRMatrix
        @test size(A_csr) == (10, 10)
        @test A_csr[1,1] ≈ 2.0
        @test A_csr.nzval === nonzeros(A_static)

        # do_collect=true: independent copy
        A_csr2 = Draugr.csr_from_static(A_static; do_collect=true)
        @test A_csr2 isa CSRMatrix
        @test size(A_csr2) == (10, 10)
        @test A_csr2[1,1] ≈ 2.0
        @test A_csr2.nzval !== nonzeros(A_static)
        @test A_csr2.nzval == nonzeros(A_static)
    end
end

@testset "csr_from_raw" begin
    @testset "one-based (default)" begin
        # Build a small 3x3 tridiagonal matrix in 1-based CSR
        rowptr = Int32[1, 3, 6, 8]
        colval = Int32[1, 2, 1, 2, 3, 2, 3]
        nzval  = [2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0]
        A = csr_from_raw(rowptr, colval, nzval, 3, 3; index_base=1)
        @test A isa CSRMatrix
        @test size(A) == (3, 3)
        @test A[1,1] ≈ 2.0
        @test A[1,2] ≈ -1.0
        @test A[2,1] ≈ -1.0
        @test A[2,2] ≈ 2.0
        @test A[3,3] ≈ 2.0
    end

    @testset "zero-based indexing" begin
        # Same 3x3 matrix but with 0-based indices (C-style)
        rowptr = Int32[0, 2, 5, 7]
        colval = Int32[0, 1, 0, 1, 2, 1, 2]
        nzval  = [2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0]
        A = csr_from_raw(copy(rowptr), copy(colval), copy(nzval), 3, 3; index_base=0)
        @test A isa CSRMatrix
        @test size(A) == (3, 3)
        @test A[1,1] ≈ 2.0
        @test A[1,2] ≈ -1.0
        @test A[2,1] ≈ -1.0
        @test A[2,2] ≈ 2.0
        @test A[3,3] ≈ 2.0
        # Verify mul! works correctly after conversion
        x = ones(3)
        y = zeros(3)
        mul!(y, A, x)
        @test y[1] ≈ 1.0   # 2 - 1
        @test y[2] ≈ 0.0   # -1 + 2 - 1
        @test y[3] ≈ 1.0   # -1 + 2
    end

    @testset "zero-based full solve" begin
        # Build a 1D Poisson (n=50) in 0-based CSR and solve via AMG
        n = 50
        # Build in 1-based first, then convert to 0-based representation
        A_csc = spdiagm(0 => fill(2.0, n), -1 => fill(-1.0, n-1), 1 => fill(-1.0, n-1))
        A_ref = csr_from_csc(A_csc)
        # Convert to 0-based
        rp_0 = copy(A_ref.rowptr) .- Int32(1)
        cv_0 = copy(A_ref.colval) .- Int32(1)
        nzv_0 = copy(A_ref.nzval)
        A_zero = csr_from_raw(copy(rp_0), copy(cv_0), copy(nzv_0), n, n; index_base=0)
        # Verify entries match
        for i in 1:n, j in 1:n
            @test A_zero[i,j] ≈ A_ref[i,j]
        end
        # Run AMG setup and solve with zero-based constructed matrix
        h = amg_setup(A_zero)
        b = ones(n)
        x = zeros(n)
        x, niter = amg_solve!(x, b, h)
        r = b .- A_csc * x
        @test norm(r) / norm(b) < 1e-6
    end

    @testset "invalid index_base" begin
        rowptr = Int32[1, 3, 5]
        colval = Int32[1, 2, 1, 2]
        nzval  = [1.0, 2.0, 3.0, 4.0]
        @test_throws ArgumentError csr_from_raw(copy(rowptr), copy(colval), copy(nzval), 2, 2; index_base=2)
    end

    @testset "in-place mutation (no extra copy)" begin
        # Verify that csr_from_raw with index_base=0 mutates the passed arrays in-place
        rowptr = Int32[0, 2, 5, 7]
        colval = Int32[0, 1, 0, 1, 2, 1, 2]
        nzval  = [2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0]
        A = csr_from_raw(rowptr, colval, nzval, 3, 3; index_base=0)
        # After call, rowptr and colval should be 1-based (mutated in-place)
        @test rowptr[1] == 1
        @test colval[1] == 1
        # The CSRMatrix should reference the same arrays
        @test A.rowptr === rowptr
        @test A.colval === colval
        @test A.nzval === nzval
    end
end
