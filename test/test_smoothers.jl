@testset "Graph Coloring" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    colors, nc = Draugr.greedy_coloring(Ac)
    @test length(colors) == 10
    @test all(colors .> 0)
    @test nc >= 2  # tridiagonal needs at least 2 colors
    # Verify no two adjacent nodes have the same color
    cv = Draugr.colvals(Ac)
    for i in 1:10
        for nz in nzrange(Ac, i)
            j = cv[nz]
            if j != i
                @test colors[i] != colors[j]
            end
        end
    end
end

@testset "Jacobi Smoother" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_jacobi_smoother(Ac, 2.0/3.0)
    @test length(smoother.invdiag) == 10
    @test all(smoother.invdiag .≈ 0.5)  # 1/2.0
    # Test smoothing reduces error
    b = ones(10)
    x = zeros(10)
    smooth!(x, Ac, b, smoother; steps=10)
    r = b - sparse(A.At') * x
    @test norm(r) < norm(b)
end

# ══════════════════════════════════════════════════════════════════════════
# Colored Gauss-Seidel Smoother
# ══════════════════════════════════════════════════════════════════════════

@testset "Colored GS Smoother - Build" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_colored_gs_smoother(Ac)
    @test smoother.num_colors >= 2
    @test length(smoother.invdiag) == 10
    @test all(smoother.invdiag .≈ 0.5)  # 1/2.0
    @test length(smoother.color_order) == 10
end

@testset "Colored GS Smoother - Smoothing" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_colored_gs_smoother(Ac)
    test_smoother_smoothing(smoother, Ac, A)
end

@testset "AMG Solve - Colored GS" begin
    test_amg_convergence(AMGConfig(smoother=ColoredGaussSeidelType()))
end

# ══════════════════════════════════════════════════════════════════════════
# L1 Colored Gauss-Seidel Smoother
# ══════════════════════════════════════════════════════════════════════════

@testset "L1 Colored GS Smoother - Build" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_l1_colored_gs_smoother(Ac)
    @test smoother isa L1ColoredGaussSeidelSmoother
    @test smoother.num_colors >= 2
    @test length(smoother.invdiag) == 10
    # For interior row: hypre option 4 l1 norm = |2| + 0.5*(|−1|+|−1|) = 3, invdiag = 1/3
    @test smoother.invdiag[5] ≈ 1.0/3.0
    @test length(smoother.color_order) == 10
end

@testset "L1 Colored GS Smoother - Smoothing" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_l1_colored_gs_smoother(Ac)
    test_smoother_smoothing(smoother, Ac, A)
end

@testset "AMG Solve - L1 Colored GS" begin
    test_amg_convergence(AMGConfig(smoother=L1ColoredGaussSeidelType()))
end

# ══════════════════════════════════════════════════════════════════════════
# Serial Gauss-Seidel Smoother
# ══════════════════════════════════════════════════════════════════════════

@testset "Serial GS Smoother - Build" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_serial_gs_smoother(Ac)
    @test smoother isa SerialGaussSeidelSmoother
    @test length(smoother.invdiag) == 10
    @test all(smoother.invdiag .≈ 0.5)  # 1/2.0
end

@testset "Serial GS Smoother - Smoothing" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_serial_gs_smoother(Ac)
    test_smoother_smoothing(smoother, Ac, A)
end

@testset "Serial GS Smoother - AMG Solve" begin
    test_amg_convergence(AMGConfig(smoother=SerialGaussSeidelType()))
end

@testset "Serial GS Smoother - Update" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_serial_gs_smoother(Ac)
    @test all(smoother.invdiag .≈ 0.5)
    # Update with same matrix
    update_smoother!(smoother, Ac)
    @test all(smoother.invdiag .≈ 0.5)
end

# ══════════════════════════════════════════════════════════════════════════
# L1 Serial Gauss-Seidel Smoother
# ══════════════════════════════════════════════════════════════════════════

@testset "L1 Serial GS Smoother - Build" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_l1_serial_gs_smoother(Ac)
    @test smoother isa L1SerialGaussSeidelSmoother
    @test length(smoother.invdiag) == 10
    # For serial GS (matching hypre serial): l1 norm = |a_{i,i}| = |2| = 2, invdiag = 1/2
    @test smoother.invdiag[5] ≈ 0.5
    # For boundary row: l1 norm = |a_{i,i}| = |2| = 2, invdiag = 1/2
    @test smoother.invdiag[1] ≈ 0.5
end

@testset "L1 Serial GS Smoother - Smoothing" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_l1_serial_gs_smoother(Ac)
    test_smoother_smoothing(smoother, Ac, A)
end

@testset "L1 Serial GS Smoother - AMG Solve" begin
    test_amg_convergence(AMGConfig(smoother=L1SerialGaussSeidelType()))
end

@testset "L1 Serial GS Smoother - Update" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_l1_serial_gs_smoother(Ac)
    @test smoother.invdiag[5] ≈ 0.5
    # Update with same matrix
    update_smoother!(smoother, Ac)
    @test smoother.invdiag[5] ≈ 0.5
end

@testset "L1 Serial GS Smoother - build_smoother dispatch" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_smoother(Ac, L1SerialGaussSeidelType(), 1.0)
    @test smoother isa L1SerialGaussSeidelSmoother
end

# ══════════════════════════════════════════════════════════════════════════
# SPAI(0) Smoother
# ══════════════════════════════════════════════════════════════════════════

@testset "SPAI0 Smoother - Build" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_spai0_smoother(Ac)
    @test length(smoother.m_diag) == 10
    # For tridiagonal with diag=2, off-diag=-1:
    # interior row: [−1 2 −1], row_norm_sq = 1+4+1 = 6, diag = 2, m = 2/6 = 1/3
    @test smoother.m_diag[5] ≈ 2.0/6.0
end

@testset "SPAI0 Smoother - Smoothing" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_spai0_smoother(Ac)
    test_smoother_smoothing(smoother, Ac, A)
end

@testset "AMG Solve - SPAI0" begin
    test_amg_convergence(AMGConfig(smoother=SPAI0SmootherType()))
end

# ══════════════════════════════════════════════════════════════════════════
# SPAI(1) Smoother
# ══════════════════════════════════════════════════════════════════════════

@testset "SPAI1 Smoother - Build" begin
    A = poisson1d_csr(5)
    Ac = to_csr(A)
    smoother = Draugr.build_spai1_smoother(Ac)
    @test length(smoother.nzval) == nnz(A)
end

@testset "SPAI1 Smoother - Smoothing" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_spai1_smoother(Ac)
    test_smoother_smoothing(smoother, Ac, A)
end

@testset "AMG Solve - SPAI1" begin
    test_amg_convergence(
        AMGConfig(smoother=SPAI1SmootherType(), pre_smoothing_steps=2, post_smoothing_steps=2);
        maxiter=300)
end

# ══════════════════════════════════════════════════════════════════════════
# l1-Jacobi Smoother
# ══════════════════════════════════════════════════════════════════════════

@testset "l1-Jacobi Smoother - Build" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_l1jacobi_smoother(Ac, 2.0/3.0)
    @test length(smoother.invdiag) == 10
    @test smoother.ω ≈ 2.0/3.0
    # For interior row: l1 norm = |−1| + |2| + |−1| = 4, invdiag = 1/4
    @test smoother.invdiag[5] ≈ 0.25
end

@testset "l1-Jacobi Smoother - Smoothing" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_l1jacobi_smoother(Ac, 2.0/3.0)
    test_smoother_smoothing(smoother, Ac, A)
end

@testset "l1-Jacobi Smoother - AMG Solve" begin
    test_amg_convergence(AMGConfig(smoother=L1JacobiSmootherType()))
end

@testset "l1-Jacobi - Resetup" begin
    test_smoother_resetup(AMGConfig(smoother=L1JacobiSmootherType()))
end

# ══════════════════════════════════════════════════════════════════════════
# Chebyshev Smoother
# ══════════════════════════════════════════════════════════════════════════

@testset "Chebyshev Smoother - Build" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_chebyshev_smoother(Ac)
    @test length(smoother.invdiag) == 10
    @test smoother.λ_max > 0
    @test smoother.λ_min > 0
    @test smoother.λ_max > smoother.λ_min
    @test smoother.degree == 3
end

@testset "Chebyshev Smoother - Smoothing" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_chebyshev_smoother(Ac)
    test_smoother_smoothing(smoother, Ac, A; steps=5)
end

@testset "Chebyshev Smoother - AMG Solve" begin
    test_amg_convergence(
        AMGConfig(smoother=ChebyshevSmootherType(), pre_smoothing_steps=2, post_smoothing_steps=2))
end

@testset "Chebyshev - Resetup" begin
    test_smoother_resetup(
        AMGConfig(smoother=ChebyshevSmootherType(), pre_smoothing_steps=2, post_smoothing_steps=2))
end

# ══════════════════════════════════════════════════════════════════════════
# ILU(0) Smoother
# ══════════════════════════════════════════════════════════════════════════

@testset "ILU(0) Smoother - Build" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_ilu0_smoother(Ac)
    @test length(smoother.L_nzval) == nnz(A)
    @test length(smoother.U_nzval) == nnz(A)
    @test length(smoother.diag_idx) == 10
    @test smoother.num_fwd_levels >= 2
end

@testset "ILU(0) Smoother - Smoothing" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_ilu0_smoother(Ac)
    test_smoother_smoothing(smoother, Ac, A; steps=3)
end

@testset "ILU(0) Smoother - AMG Solve" begin
    test_amg_convergence(AMGConfig(smoother=ILU0SmootherType()))
end

@testset "ILU(0) Smoother - Resetup" begin
    test_smoother_resetup(AMGConfig(smoother=ILU0SmootherType()))
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
# Serial ILU(0) Smoother
# ══════════════════════════════════════════════════════════════════════════

@testset "Serial ILU(0) Smoother - Build" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_serial_ilu0_smoother(Ac)
    @test smoother isa SerialILU0Smoother
    @test length(smoother.L_nzval) == nnz(A)
    @test length(smoother.U_nzval) == nnz(A)
    @test length(smoother.diag_idx) == 10
end

@testset "Serial ILU(0) Smoother - Smoothing" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_serial_ilu0_smoother(Ac)
    test_smoother_smoothing(smoother, Ac, A; steps=3)
end

@testset "Serial ILU(0) Smoother - AMG Solve" begin
    test_amg_convergence(AMGConfig(smoother=SerialILU0SmootherType()))
end

@testset "Serial ILU(0) Smoother - Resetup" begin
    test_smoother_resetup(AMGConfig(smoother=SerialILU0SmootherType()))
end

@testset "Serial ILU(0) - Anisotropic" begin
    A = anisotropic_csr(8, 8)
    N = 64
    b = rand(N)
    x = zeros(N)
    config = AMGConfig(smoother=SerialILU0SmootherType())
    hierarchy = amg_setup(A, config)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
end

@testset "Serial ILU(0) - Reservoir-like" begin
    A = reservoir_like_csr(50)
    N = 50
    b = rand(N)
    x = zeros(N)
    config = AMGConfig(smoother=SerialILU0SmootherType())
    hierarchy = amg_setup(A, config)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
end

@testset "Parallel and Serial ILU(0) produce same factorization" begin
    A = poisson2d_csr(6)
    Ac = to_csr(A)
    s_par = Draugr.build_ilu0_smoother(Ac)
    s_ser = Draugr.build_serial_ilu0_smoother(Ac)
    @test s_par.L_nzval ≈ s_ser.L_nzval
    @test s_par.U_nzval ≈ s_ser.U_nzval
    @test s_par.diag_idx == s_ser.diag_idx
end

@testset "Serial ILU(0) build_smoother dispatch" begin
    A = poisson1d_csr(10)
    s = build_smoother(A, SerialILU0SmootherType())
    @test s isa SerialILU0Smoother
end

# ══════════════════════════════════════════════════════════════════════════
# GPU ILU(0) Smoother
# ══════════════════════════════════════════════════════════════════════════

@testset "GPU ILU(0) Smoother - Build" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_gpu_ilu0_smoother(Ac)
    @test smoother isa GPUILU0Smoother
    @test length(smoother.L_nzval) == nnz(A)
    @test length(smoother.U_nzval) == nnz(A)
    @test length(smoother.diag_idx) == 10
    @test smoother.num_fwd_levels >= 2
end

@testset "GPU ILU(0) Smoother - Smoothing" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_gpu_ilu0_smoother(Ac)
    test_smoother_smoothing(smoother, Ac, A; steps=3)
end

@testset "GPU ILU(0) Smoother - AMG Solve" begin
    test_amg_convergence(AMGConfig(smoother=GPUILU0SmootherType()))
end

@testset "GPU ILU(0) Smoother - Resetup" begin
    test_smoother_resetup(AMGConfig(smoother=GPUILU0SmootherType()))
end

@testset "GPU ILU(0) - Anisotropic" begin
    A = anisotropic_csr(8, 8)
    N = 64
    b = rand(N)
    x = zeros(N)
    config = AMGConfig(smoother=GPUILU0SmootherType())
    hierarchy = amg_setup(A, config)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
end

@testset "GPU ILU(0) matches CPU ILU(0) factorization" begin
    A = poisson2d_csr(6)
    Ac = to_csr(A)
    s_gpu = Draugr.build_gpu_ilu0_smoother(Ac)
    s_cpu = Draugr.build_ilu0_smoother(Ac)
    @test Array(s_gpu.L_nzval) ≈ s_cpu.L_nzval
    @test Array(s_gpu.U_nzval) ≈ s_cpu.U_nzval
    @test Array(s_gpu.diag_idx) == s_cpu.diag_idx
end

@testset "GPU ILU(0) build_smoother dispatch" begin
    A = poisson1d_csr(10)
    s = build_smoother(A, GPUILU0SmootherType())
    @test s isa GPUILU0Smoother
end

# ══════════════════════════════════════════════════════════════════════════
# DILU Smoother
# ══════════════════════════════════════════════════════════════════════════

@testset "DILU Smoother - Build" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_dilu_smoother(Ac)
    @test smoother isa DILUSmoother
    @test length(smoother.inv_diag) == 10
    @test length(smoother.diag_idx) == 10
    @test smoother.num_fwd_levels >= 2
end

@testset "DILU Smoother - Smoothing" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    smoother = Draugr.build_dilu_smoother(Ac)
    test_smoother_smoothing(smoother, Ac, A; steps=3)
end

@testset "DILU Smoother - AMG Solve" begin
    test_amg_convergence(AMGConfig(smoother=DILUSmootherType()))
end

@testset "DILU Smoother - Resetup" begin
    test_smoother_resetup(AMGConfig(smoother=DILUSmootherType()))
end

@testset "DILU - Anisotropic" begin
    A = anisotropic_csr(8, 8)
    N = 64
    b = rand(N)
    x = zeros(N)
    config = AMGConfig(smoother=DILUSmootherType())
    hierarchy = amg_setup(A, config)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
end

@testset "DILU - Reservoir-like" begin
    A = reservoir_like_csr(50)
    N = 50
    b = rand(N)
    x = zeros(N)
    config = AMGConfig(smoother=DILUSmootherType())
    hierarchy = amg_setup(A, config)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-8, maxiter=200)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < 1e-8
end

@testset "DILU build_smoother dispatch" begin
    A = poisson1d_csr(10)
    s = build_smoother(A, DILUSmootherType())
    @test s isa DILUSmoother
end

@testset "DILU diagonal correctness" begin
    # For a tridiagonal matrix A = [-1 2 -1], the DILU modified diagonal
    # should differ from the original since lower neighbors contribute.
    # d_i = a_{ii} - Σ_{j<i} a_{ij} * d_j^{-1} * a_{ji}
    A = poisson1d_csr(5)
    Ac = to_csr(A)
    smoother = Draugr.build_dilu_smoother(Ac)
    inv_diag = Array(smoother.inv_diag)
    # Row 1: no lower neighbors, d_1 = 2.0
    @test inv_diag[1] ≈ 1.0 / 2.0
    # Row 2: d_2 = 2.0 - (-1)*(1/2)*(-1) = 2.0 - 0.5 = 1.5
    @test inv_diag[2] ≈ 1.0 / 1.5
    # Row 3: d_3 = 2.0 - (-1)*(1/1.5)*(-1) = 2.0 - 2/3 ≈ 1.333...
    @test inv_diag[3] ≈ 1.0 / (2.0 - 1.0/1.5)
    # Row 4: d_4 = 2.0 - (-1)*inv_diag[3]*(-1) = 2.0 - 1/d_3
    d3 = 2.0 - 1.0/1.5
    @test inv_diag[4] ≈ 1.0 / (2.0 - 1.0/d3)
    # Row 5 (last, boundary): only one lower neighbor
    d4 = 2.0 - 1.0/d3
    @test inv_diag[5] ≈ 1.0 / (2.0 - 1.0/d4)
end

# ══════════════════════════════════════════════════════════════════════════
# Standalone Smoother API
# ══════════════════════════════════════════════════════════════════════════

@testset "Standalone Smoother API - build_smoother from StaticCSR" begin
    A = poisson1d_csr(10)
    # Jacobi
    s = build_smoother(A, JacobiSmootherType())
    @test s isa JacobiSmoother
    @test length(s.invdiag) == 10
    @test all(s.invdiag .≈ 0.5)
    # SPAI0
    s2 = build_smoother(A, SPAI0SmootherType())
    @test s2 isa SPAI0Smoother
    # l1-Jacobi
    s3 = build_smoother(A, L1JacobiSmootherType())
    @test s3 isa L1JacobiSmoother
    # Colored GS
    s4 = build_smoother(A, ColoredGaussSeidelType())
    @test s4 isa ColoredGaussSeidelSmoother
    # ILU0
    s5 = build_smoother(A, ILU0SmootherType())
    @test s5 isa ILU0Smoother
    # Chebyshev
    s6 = build_smoother(A, ChebyshevSmootherType())
    @test s6 isa ChebyshevSmoother
    # SPAI1
    s7 = build_smoother(A, SPAI1SmootherType())
    @test s7 isa SPAI1Smoother
    # Serial GS
    s8 = build_smoother(A, SerialGaussSeidelType())
    @test s8 isa SerialGaussSeidelSmoother
    # L1 Colored GS
    s9 = build_smoother(A, L1ColoredGaussSeidelType())
    @test s9 isa L1ColoredGaussSeidelSmoother
    # Serial ILU0
    s10 = build_smoother(A, SerialILU0SmootherType())
    @test s10 isa SerialILU0Smoother
    # L1 Serial GS
    s10 = build_smoother(A, L1SerialGaussSeidelType())
    @test s10 isa L1SerialGaussSeidelSmoother
    # GPU ILU0
    s11 = build_smoother(A, GPUILU0SmootherType())
    @test s11 isa GPUILU0Smoother
    # DILU
    s12 = build_smoother(A, DILUSmootherType())
    @test s12 isa DILUSmoother
end

@testset "Standalone Smoother API - smooth! with StaticCSR" begin
    A = poisson1d_csr(10)
    smoother = build_smoother(A, JacobiSmootherType())
    b = ones(10)
    x = zeros(10)
    # Apply smoother directly with StaticSparsityMatrixCSR
    smooth!(x, A, b, smoother; steps=10)
    r = b - sparse(A.At') * x
    @test norm(r) < norm(b)
end

@testset "Standalone Smoother API - update_smoother! with StaticCSR" begin
    A = poisson1d_csr(10)
    smoother = build_smoother(A, JacobiSmootherType())
    @test all(smoother.invdiag .≈ 0.5)
    # Update with same matrix
    update_smoother!(smoother, A)
    @test all(smoother.invdiag .≈ 0.5)
end

@testset "Standalone Smoother API - all types smooth correctly" begin
    A = poisson1d_csr(10)
    b = ones(10)
    for stype in [JacobiSmootherType(), SPAI0SmootherType(), L1JacobiSmootherType(),
                  ColoredGaussSeidelType(), L1ColoredGaussSeidelType(), ILU0SmootherType(), ChebyshevSmootherType(),
                  SPAI1SmootherType(), SerialGaussSeidelType(), SerialILU0SmootherType(), L1SerialGaussSeidelType(),
                  GPUILU0SmootherType(), DILUSmootherType()]
        smoother = build_smoother(A, stype)
        x = zeros(10)
        smooth!(x, A, b, smoother; steps=10)
        r = b - sparse(A.At') * x
        @test norm(r) < norm(b)
    end
end

# ══════════════════════════════════════════════════════════════════════════
# Pre-computed residual passing tests
# ══════════════════════════════════════════════════════════════════════════

@testset "Pre-computed residual - smooth! produces identical results" begin
    A = poisson2d_csr(8, 8)
    Ac = to_csr(A)
    A_sparse = sparse(A.At')
    b = rand(MersenneTwister(42), 64)
    x0 = rand(MersenneTwister(43), 64)

    # Smoother types that benefit from pre-computed residual (compute residual internally)
    residual_types = [
        JacobiSmootherType(), L1JacobiSmootherType(),
        SPAI0SmootherType(), SPAI1SmootherType(),
        ChebyshevSmootherType(), SerialILU0SmootherType(), ILU0SmootherType(),
    ]

    for stype in residual_types
        smoother = build_smoother(A, stype)

        # Test with steps=1: residual-based should match normal
        x_normal = copy(x0)
        smooth!(x_normal, Ac, b, smoother; steps=1)

        r = b - A_sparse * x0
        x_resid = copy(x0)
        smooth!(x_resid, Ac, b, smoother; steps=1, residual=r)

        @test x_normal ≈ x_resid atol=1e-12

        # Test with steps=2: only first iteration uses residual
        x_normal2 = copy(x0)
        smooth!(x_normal2, Ac, b, smoother; steps=2)

        x_resid2 = copy(x0)
        smooth!(x_resid2, Ac, b, smoother; steps=2, residual=r)

        @test x_normal2 ≈ x_resid2 atol=1e-12
    end

    # GS smoothers: residual kwarg is accepted but ignored (verify no error)
    gs_types = [
        ColoredGaussSeidelType(), L1ColoredGaussSeidelType(),
        SerialGaussSeidelType(), L1SerialGaussSeidelType(),
    ]
    for stype in gs_types
        smoother = build_smoother(A, stype)
        r = b - A_sparse * x0

        x_normal = copy(x0)
        smooth!(x_normal, Ac, b, smoother; steps=1)

        x_resid = copy(x0)
        smooth!(x_resid, Ac, b, smoother; steps=1, residual=r)

        @test x_normal ≈ x_resid atol=1e-12
    end
end

@testset "Pre-computed residual - AMG cycle with x=0 residual=b" begin
    A = poisson2d_csr(16, 16)
    b = ones(256)

    for stype in [JacobiSmootherType(), SPAI0SmootherType(), L1JacobiSmootherType(),
                  ChebyshevSmootherType(), L1ColoredGaussSeidelType()]
        config = AMGConfig(smoother=stype)
        hierarchy = amg_setup(A, config)

        # Without residual
        x1 = zeros(256)
        amg_cycle!(x1, b, hierarchy, config)

        # With residual=b (since x=0, residual = b - A*0 = b)
        x2 = zeros(256)
        amg_cycle!(x2, b, hierarchy, config; residual=b)

        @test x1 ≈ x2 atol=1e-12
    end
end

# ══════════════════════════════════════════════════════════════════════════
# Block smoother tests (SMatrix entries, SVector rhs)
# ══════════════════════════════════════════════════════════════════════════

"""
Build a 2×2 block version of the 1-D Poisson matrix on n (block-)nodes.
The entry type is SMatrix{2,2,Float64,4}; entries are block-diagonal:
  diag blocks = 2*I₂, off-diag blocks = -I₂.
Returns the CSRMatrix and a corresponding SparseMatrixCSC for residual checks.
"""
function block_poisson1d(n)
    B = 2
    T = Float64
    Tv = SMatrix{B, B, T, B*B}
    Ti = Int
    I_rows = Ti[]; J_cols = Ti[]; Vvals = Tv[]
    for i in 1:n
        push!(I_rows, i); push!(J_cols, i); push!(Vvals, Tv(2*one(Tv)))
        if i > 1
            push!(I_rows, i); push!(J_cols, i-1); push!(Vvals, Tv(-one(Tv)))
        end
        if i < n
            push!(I_rows, i); push!(J_cols, i+1); push!(Vvals, Tv(-one(Tv)))
        end
    end
    # Build CSRMatrix directly
    row_counts = zeros(Int, n)
    for r in I_rows; row_counts[r] += 1; end
    rptr = Vector{Ti}(undef, n+1)
    rptr[1] = 1
    for i in 1:n; rptr[i+1] = rptr[i] + row_counts[i]; end
    pos = copy(rptr[1:n])
    cval = Vector{Ti}(undef, length(Vvals))
    nzv  = Vector{Tv}(undef, length(Vvals))
    for k in eachindex(I_rows)
        r = I_rows[k]; p = pos[r]
        cval[p] = J_cols[k]; nzv[p] = Vvals[k]; pos[r] += 1
    end
    # Sort each row by column
    for i in 1:n
        rng = rptr[i]:(rptr[i+1]-1)
        perm = sortperm(cval[rng])
        cval[rng] .= cval[rng][perm]
        nzv[rng]  .= nzv[rng][perm]
    end
    return Draugr.CSRMatrix(rptr, cval, nzv, n, n)
end

@testset "Block Jacobi Smoother" begin
    n = 8
    B = 2; T = Float64
    Tv = SMatrix{B,B,T,B*B}; Tx = SVector{B,T}
    Ac = block_poisson1d(n)
    smoother = Draugr.build_jacobi_smoother(Ac, 2.0/3.0; x_eltype=Tx)
    @test smoother isa Draugr.JacobiSmoother
    @test eltype(smoother.invdiag) == Tv
    @test eltype(smoother.tmp)     == Tx
    b = [Tx(1.0, 0.0) for _ in 1:n]
    x = [Tx(0.0, 0.0) for _ in 1:n]
    smooth!(x, Ac, b, smoother; steps=20)
    # residual must shrink
    r_norm = norm([norm(b[i] - sum(Ac.nzval[nz]*x[Ac.colval[nz]] for nz in Ac.rowptr[i]:(Ac.rowptr[i+1]-1))) for i in 1:n])
    @test r_norm < norm(norm.(b))
end

@testset "Block L1-Jacobi Smoother" begin
    n = 8
    B = 2; T = Float64
    Tv = SMatrix{B,B,T,B*B}; Tx = SVector{B,T}
    Ac = block_poisson1d(n)
    smoother = Draugr.build_l1jacobi_smoother(Ac, 2.0/3.0; x_eltype=Tx)
    @test smoother isa Draugr.L1JacobiSmoother
    @test eltype(smoother.invdiag) == Tv
    @test eltype(smoother.tmp)     == Tx
    b = [Tx(1.0, 0.0) for _ in 1:n]
    x = [Tx(0.0, 0.0) for _ in 1:n]
    smooth!(x, Ac, b, smoother; steps=20)
    r_norm = norm([norm(b[i] - sum(Ac.nzval[nz]*x[Ac.colval[nz]] for nz in Ac.rowptr[i]:(Ac.rowptr[i+1]-1))) for i in 1:n])
    @test r_norm < norm(norm.(b))
end

@testset "Block SPAI0 Smoother" begin
    n = 8
    B = 2; T = Float64
    Tv = SMatrix{B,B,T,B*B}; Tx = SVector{B,T}
    Ac = block_poisson1d(n)
    smoother = Draugr.build_spai0_smoother(Ac; x_eltype=Tx)
    @test smoother isa Draugr.SPAI0Smoother
    @test eltype(smoother.m_diag) == Tv
    @test eltype(smoother.tmp)    == Tx
    b = [Tx(1.0, 0.0) for _ in 1:n]
    x = [Tx(0.0, 0.0) for _ in 1:n]
    smooth!(x, Ac, b, smoother; steps=20)
    r_norm = norm([norm(b[i] - sum(Ac.nzval[nz]*x[Ac.colval[nz]] for nz in Ac.rowptr[i]:(Ac.rowptr[i+1]-1))) for i in 1:n])
    @test r_norm < norm(norm.(b))
end

@testset "Block SPAI1 Smoother" begin
    n = 8
    B = 2; T = Float64
    Tv = SMatrix{B,B,T,B*B}; Tx = SVector{B,T}
    Ac = block_poisson1d(n)
    smoother = Draugr.build_spai1_smoother(Ac; x_eltype=Tx)
    @test smoother isa Draugr.SPAI1Smoother
    @test eltype(smoother.nzval) == Tv
    @test eltype(smoother.tmp)   == Tx
    b = [Tx(1.0, 0.0) for _ in 1:n]
    x = [Tx(0.0, 0.0) for _ in 1:n]
    smooth!(x, Ac, b, smoother; steps=5)
    r_norm = norm([norm(b[i] - sum(Ac.nzval[nz]*x[Ac.colval[nz]] for nz in Ac.rowptr[i]:(Ac.rowptr[i+1]-1))) for i in 1:n])
    @test r_norm < norm(norm.(b))
end

@testset "Block Chebyshev Smoother" begin
    n = 8
    B = 2; T = Float64
    Tv = SMatrix{B,B,T,B*B}; Tx = SVector{B,T}
    Ac = block_poisson1d(n)
    smoother = Draugr.build_chebyshev_smoother(Ac; x_eltype=Tx)
    @test smoother isa Draugr.ChebyshevSmoother
    @test eltype(smoother.invdiag) == Tv
    @test eltype(smoother.tmp1)    == Tx
    @test eltype(smoother.tmp2)    == Tx
    b = [Tx(1.0, 0.0) for _ in 1:n]
    x = [Tx(0.0, 0.0) for _ in 1:n]
    smooth!(x, Ac, b, smoother; steps=5)
    r_norm = norm([norm(b[i] - sum(Ac.nzval[nz]*x[Ac.colval[nz]] for nz in Ac.rowptr[i]:(Ac.rowptr[i+1]-1))) for i in 1:n])
    @test r_norm < norm(norm.(b))
end

@testset "Block Colored GS Smoother" begin
    n = 8
    B = 2; T = Float64
    Tv = SMatrix{B,B,T,B*B}; Tx = SVector{B,T}
    Ac = block_poisson1d(n)
    smoother = Draugr.build_colored_gs_smoother(Ac)
    @test smoother isa Draugr.ColoredGaussSeidelSmoother
    @test eltype(smoother.invdiag) == Tv
    b = [Tx(1.0, 0.0) for _ in 1:n]
    x = [Tx(0.0, 0.0) for _ in 1:n]
    smooth!(x, Ac, b, smoother; steps=20)
    r_norm = norm([norm(b[i] - sum(Ac.nzval[nz]*x[Ac.colval[nz]] for nz in Ac.rowptr[i]:(Ac.rowptr[i+1]-1))) for i in 1:n])
    @test r_norm < norm(norm.(b))
end

@testset "Block L1 Colored GS Smoother" begin
    n = 8
    B = 2; T = Float64
    Tv = SMatrix{B,B,T,B*B}; Tx = SVector{B,T}
    Ac = block_poisson1d(n)
    smoother = Draugr.build_l1_colored_gs_smoother(Ac)
    @test smoother isa Draugr.L1ColoredGaussSeidelSmoother
    @test eltype(smoother.invdiag) == Tv
    b = [Tx(1.0, 0.0) for _ in 1:n]
    x = [Tx(0.0, 0.0) for _ in 1:n]
    smooth!(x, Ac, b, smoother; steps=20)
    r_norm = norm([norm(b[i] - sum(Ac.nzval[nz]*x[Ac.colval[nz]] for nz in Ac.rowptr[i]:(Ac.rowptr[i+1]-1))) for i in 1:n])
    @test r_norm < norm(norm.(b))
end

@testset "Block Serial GS Smoother" begin
    n = 8
    B = 2; T = Float64
    Tv = SMatrix{B,B,T,B*B}; Tx = SVector{B,T}
    Ac = block_poisson1d(n)
    smoother = Draugr.build_serial_gs_smoother(Ac)
    @test smoother isa Draugr.SerialGaussSeidelSmoother
    @test eltype(smoother.invdiag) == Tv
    b = [Tx(1.0, 0.0) for _ in 1:n]
    x = [Tx(0.0, 0.0) for _ in 1:n]
    smooth!(x, Ac, b, smoother; steps=20)
    r_norm = norm([norm(b[i] - sum(Ac.nzval[nz]*x[Ac.colval[nz]] for nz in Ac.rowptr[i]:(Ac.rowptr[i+1]-1))) for i in 1:n])
    @test r_norm < norm(norm.(b))
end

@testset "Block L1 Serial GS Smoother" begin
    n = 8
    B = 2; T = Float64
    Tv = SMatrix{B,B,T,B*B}; Tx = SVector{B,T}
    Ac = block_poisson1d(n)
    smoother = Draugr.build_l1_serial_gs_smoother(Ac)
    @test smoother isa Draugr.L1SerialGaussSeidelSmoother
    @test eltype(smoother.invdiag) == Tv
    b = [Tx(1.0, 0.0) for _ in 1:n]
    x = [Tx(0.0, 0.0) for _ in 1:n]
    smooth!(x, Ac, b, smoother; steps=20)
    r_norm = norm([norm(b[i] - sum(Ac.nzval[nz]*x[Ac.colval[nz]] for nz in Ac.rowptr[i]:(Ac.rowptr[i+1]-1))) for i in 1:n])
    @test r_norm < norm(norm.(b))
end

@testset "Block ILU0 Smoother" begin
    n = 8
    B = 2; T = Float64
    Tv = SMatrix{B,B,T,B*B}; Tx = SVector{B,T}
    Ac = block_poisson1d(n)
    smoother = Draugr.build_ilu0_smoother(Ac; x_eltype=Tx)
    @test smoother isa Draugr.ILU0Smoother
    @test eltype(smoother.L_nzval) == Tv
    @test eltype(smoother.tmp)     == Tx
    b = [Tx(1.0, 0.0) for _ in 1:n]
    x = [Tx(0.0, 0.0) for _ in 1:n]
    smooth!(x, Ac, b, smoother; steps=3)
    r_norm = norm([norm(b[i] - sum(Ac.nzval[nz]*x[Ac.colval[nz]] for nz in Ac.rowptr[i]:(Ac.rowptr[i+1]-1))) for i in 1:n])
    @test r_norm < norm(norm.(b))
end

@testset "Block Serial ILU0 Smoother" begin
    n = 8
    B = 2; T = Float64
    Tv = SMatrix{B,B,T,B*B}; Tx = SVector{B,T}
    Ac = block_poisson1d(n)
    smoother = Draugr.build_serial_ilu0_smoother(Ac; x_eltype=Tx)
    @test smoother isa Draugr.SerialILU0Smoother
    @test eltype(smoother.L_nzval) == Tv
    @test eltype(smoother.tmp)     == Tx
    b = [Tx(1.0, 0.0) for _ in 1:n]
    x = [Tx(0.0, 0.0) for _ in 1:n]
    smooth!(x, Ac, b, smoother; steps=3)
    r_norm = norm([norm(b[i] - sum(Ac.nzval[nz]*x[Ac.colval[nz]] for nz in Ac.rowptr[i]:(Ac.rowptr[i+1]-1))) for i in 1:n])
    @test r_norm < norm(norm.(b))
end

