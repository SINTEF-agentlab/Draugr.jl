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
    A_csr = Draugr.csr_from_static(A_static)
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
