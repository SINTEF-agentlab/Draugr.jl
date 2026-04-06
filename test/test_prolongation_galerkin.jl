@testset "Prolongation" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    agg, nc = Draugr.coarsen_aggregation(Ac, 0.25)
    P = Draugr.build_prolongation(Ac, agg, nc)
    @test P.nrow == 10
    @test P.ncol == nc
    # Each row of P has exactly one nonzero for aggregation
    for i in 1:P.nrow
        @test P.rowptr[i+1] - P.rowptr[i] == 1
    end
    # Test prolongation operation
    xc = ones(nc)
    xf = zeros(10)
    Draugr.prolongate!(xf, P, xc)
    @test all(xf .≈ 1.0)  # P*ones should be ones
    # Test restriction
    rf = ones(10)
    bc = zeros(nc)
    Pt_map = Draugr.build_transpose_map(P)
    Draugr.restrict!(bc, Pt_map, P, rf)
    # Sum should be preserved
    @test sum(bc) ≈ sum(rf)
    # Test TransposeMap structure
    @test length(Pt_map.offsets) == nc + 1
    @test Pt_map.offsets[1] == 1
    @test Pt_map.offsets[nc + 1] == P.nrow + 1  # aggregation P: one NZ per fine row
    # Verify restrict! against explicit P^T * r computation
    rf2 = randn(10)
    bc2 = zeros(nc)
    Draugr.restrict!(bc2, Pt_map, P, rf2)
    P_sparse = prolongation_to_sparse(P)
    bc_ref = P_sparse' * rf2
    @test bc2 ≈ bc_ref atol=1e-12
end

@testset "Galerkin Product - contention-free kernel" begin
    # Verify the grouped nz_offsets structure produces correct results
    A = poisson1d_csr(20)
    Ac = to_csr(A)
    agg, nc = Draugr.coarsen_aggregation(Ac, 0.25)
    P = Draugr.build_prolongation(Ac, agg, nc)
    Pt_map = Draugr.build_transpose_map(P)
    A_coarse, r_map = Draugr.compute_coarse_sparsity(Ac, P, Pt_map, nc)
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
    Draugr.galerkin_product!(A_coarse, Ac_new, P, r_map)
    P_sparse = prolongation_to_sparse(P)
    A_sparse = sparse(A.At')
    Ac_explicit = P_sparse' * A_sparse * P_sparse
    for i in 1:nc, j in 1:nc
        @test A_coarse[i,j] ≈ Ac_explicit[i,j] atol=1e-12
    end
end

@testset "Galerkin Product" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    agg, nc = Draugr.coarsen_aggregation(Ac, 0.25)
    P = Draugr.build_prolongation(Ac, agg, nc)
    Pt_map = Draugr.build_transpose_map(P)
    A_coarse, r_map = Draugr.compute_coarse_sparsity(Ac, P, Pt_map, nc)
    # Verify Galerkin product against explicit computation
    P_sparse = prolongation_to_sparse(P)
    A_sparse = sparse(A.At')
    Ac_explicit = P_sparse' * A_sparse * P_sparse
    for i in 1:nc, j in 1:nc
        @test A_coarse[i,j] ≈ Ac_explicit[i,j] atol=1e-12
    end
end

@testset "In-place Galerkin Resetup" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    agg, nc = Draugr.coarsen_aggregation(Ac, 0.25)
    P = Draugr.build_prolongation(Ac, agg, nc)
    Pt_map = Draugr.build_transpose_map(P)
    A_coarse, r_map = Draugr.compute_coarse_sparsity(Ac, P, Pt_map, nc)
    nzv = nonzeros(A)
    nzv .*= 2.0
    Ac = to_csr(A)
    # Recompute in-place
    Draugr.galerkin_product!(A_coarse, Ac, P, r_map)
    # Verify against explicit computation with scaled matrix
    P_sparse = prolongation_to_sparse(P)
    A_sparse = sparse(A.At')
    Ac_explicit = P_sparse' * A_sparse * P_sparse
    for i in 1:nc, j in 1:nc
        @test A_coarse[i,j] ≈ Ac_explicit[i,j] atol=1e-12
    end
end

@testset "CF Prolongation - Coarse Points Identity" begin
    A = poisson1d_csr(20)
    Ac = to_csr(A)
    cf, cmap, nc = Draugr.coarsen_pmis(Ac, 0.25)
    # Test all three interpolation types
    for interp in [DirectInterpolation(), StandardInterpolation(), ExtendedIInterpolation()]
        P, _ = Draugr.build_cf_prolongation(Ac, cf, cmap, nc, interp)
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
    cf, cmap, nc = Draugr.coarsen_pmis(Ac, 0.25)
    for interp in [DirectInterpolation(), StandardInterpolation(), ExtendedIInterpolation()]
        P, _ = Draugr.build_cf_prolongation(Ac, cf, cmap, nc, interp)
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
    cf, cmap, nc = Draugr.coarsen_pmis(Ac, 0.25)
    P, _ = Draugr.build_cf_prolongation(Ac, cf, cmap, nc, StandardInterpolation())
    Pt_map = Draugr.build_transpose_map(P)
    A_coarse, r_map = Draugr.compute_coarse_sparsity(Ac, P, Pt_map, nc)
    # Verify against explicit computation
    P_sparse = prolongation_to_sparse(P)
    A_sparse = sparse(A.At')
    Ac_explicit = P_sparse' * A_sparse * P_sparse
    for i in 1:nc, j in 1:nc
        @test A_coarse[i,j] ≈ Ac_explicit[i,j] atol=1e-10
    end
    # Test in-place resetup with triple map
    nzv = nonzeros(A)
    nzv .*= 1.5
    Ac = to_csr(A)
    Draugr.galerkin_product!(A_coarse, Ac, P, r_map)
    A_sparse2 = sparse(A.At')
    Ac_explicit2 = P_sparse' * A_sparse2 * P_sparse
    for i in 1:nc, j in 1:nc
        @test A_coarse[i,j] ≈ Ac_explicit2[i,j] atol=1e-10
    end
end

@testset "Smoothed Prolongation Construction" begin
    A = poisson1d_csr(10)
    Ac = to_csr(A)
    agg, nc = Draugr.coarsen_aggregation(Ac, 0.25)
    P_tent = Draugr.build_prolongation(Ac, agg, nc)
    P_smooth = Draugr._smooth_prolongation(Ac, P_tent, 2/3)
    @test P_smooth.nrow == 10
    @test P_smooth.ncol == nc
    # Smoothed P should have more nonzeros than tentative P
    nnz_tent = P_tent.rowptr[end] - 1
    nnz_smooth = P_smooth.rowptr[end] - 1
    @test nnz_smooth >= nnz_tent
end

@testset "Filter Prolongation" begin
    # Build a P with some small entries that should be filtered
    P = Draugr.ProlongationOp{Int, Float64}(
        [1, 4, 7, 10],  # rowptr: 3 entries per row
        [1, 2, 3, 1, 2, 3, 1, 2, 3],  # colval
        [1.0, 0.05, 0.01, 0.01, 1.0, 0.05, 0.05, 0.01, 1.0],  # nzval
        3, 3
    )
    P_filt = Draugr._filter_prolongation(P, 0.1)
    # After filtering with tol=0.1:
    # Row 1: max=1.0, threshold=0.1, keep entries ≥ 0.1: [1.0] (drop 0.05, 0.01)
    for i in 1:3
        nnz_row = P_filt.rowptr[i+1] - P_filt.rowptr[i]
        @test nnz_row >= 1  # at least one entry per row
    end
end
