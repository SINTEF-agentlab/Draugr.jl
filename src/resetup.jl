"""
    amg_resetup!(hierarchy, A_new, config)

Re-setup the AMG hierarchy for new matrix coefficients, reusing the existing
sparsity pattern and prolongation structure. This updates:
1. The fine-grid matrix at the first level
2. Galerkin products at all levels (in-place, parallelized with KA kernels)
3. Smoothers at all levels
4. The direct solver at the coarsest level
"""
function amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                      A_new::StaticSparsityMatrixCSR{Tv, Ti},
                      config::AMGConfig=AMGConfig();
                      backend=DEFAULT_BACKEND) where {Tv, Ti}
    nlevels = length(hierarchy.levels)
    if nlevels == 0
        # Only one level (coarsest), update direct solver
        A_csr = csr_from_static(A_new)
        _update_coarse_solver!(hierarchy, A_csr; backend=backend)
        return hierarchy
    end
    # Update first level's matrix (copy values from A_new into existing CSRMatrix)
    level1 = hierarchy.levels[1]
    csr_copy_nzvals!(level1.A, A_new; backend=backend)
    update_smoother!(level1.smoother, level1.A; backend=backend)
    # Update subsequent levels via Galerkin products
    for lvl in 1:(nlevels - 1)
        level = hierarchy.levels[lvl]
        next_level = hierarchy.levels[lvl + 1]
        # Recompute A_coarse in-place
        galerkin_product!(next_level.A, level.A, level.P, level.R_map; backend=backend)
        # Update smoother
        update_smoother!(next_level.smoother, next_level.A; backend=backend)
    end
    # Recompute Galerkin product for the last level to get the coarsest matrix
    last_level = hierarchy.levels[nlevels]
    _recompute_coarsest_dense!(hierarchy, last_level; backend=backend)
    # In-place LU refactorization: copy dense values to LU buffer, then factorize
    copyto!(hierarchy.coarse_lu, hierarchy.coarse_A)
    LinearAlgebra.LAPACK.getrf!(hierarchy.coarse_lu, hierarchy.coarse_ipiv)
    hierarchy.coarse_factor = LU(hierarchy.coarse_lu, hierarchy.coarse_ipiv, 0)
    return hierarchy
end

"""
    _copy_nzvals!(dest, src; backend=DEFAULT_BACKEND)

Copy nonzero values from `src` CSRMatrix into `dest` (same sparsity pattern),
using a KA kernel for parallelism.
"""
function _copy_nzvals!(dest::CSRMatrix, src::CSRMatrix;
                       backend=DEFAULT_BACKEND)
    nzv_d = nonzeros(dest)
    nzv_s = nonzeros(src)
    n = length(nzv_d)
    kernel! = copy_kernel!(backend, 64)
    kernel!(nzv_d, nzv_s; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return dest
end

"""
    _recompute_coarsest_dense!(hierarchy, last_level; backend=DEFAULT_BACKEND)

Recompute the coarsest dense matrix from the last AMG level in-place,
writing directly into `hierarchy.coarse_A`. Avoids allocating a temporary
coarse CSR matrix. Modifies `hierarchy.coarse_A` as a side effect.
"""
function _recompute_coarsest_dense!(hierarchy::AMGHierarchy{Tv, Ti},
                                    level::AMGLevel{Tv, Ti};
                                    backend=DEFAULT_BACKEND) where {Tv, Ti}
    M = hierarchy.coarse_A
    fill!(M, zero(Tv))
    n_fine = size(level.A, 1)
    cv_a = colvals(level.A)
    nzv_a = nonzeros(level.A)
    P = level.P
    rp_a = rowptr(level.A)
    # Write directly into dense matrix - iterate over fine rows
    @inbounds for i in 1:n_fine
        for pnz_i in P.rowptr[i]:(P.rowptr[i+1]-1)
            I = P.colval[pnz_i]
            p_i = P.nzval[pnz_i]
            for anz in rp_a[i]:(rp_a[i+1]-1)
                j = cv_a[anz]
                a_ij = nzv_a[anz]
                for pnz_j in P.rowptr[j]:(P.rowptr[j+1]-1)
                    J = P.colval[pnz_j]
                    p_j = P.nzval[pnz_j]
                    M[I, J] += p_i * a_ij * p_j
                end
            end
        end
    end
    return M
end

"""
    _update_coarse_solver!(hierarchy, A; backend=DEFAULT_BACKEND)

Update the direct solver at the coarsest level using in-place LU refactorization.
"""
function _update_coarse_solver!(hierarchy::AMGHierarchy{Tv}, A::CSRMatrix{Tv};
                                backend=DEFAULT_BACKEND) where {Tv}
    _csr_to_dense!(hierarchy.coarse_A, A; backend=backend)
    copyto!(hierarchy.coarse_lu, hierarchy.coarse_A)
    LinearAlgebra.LAPACK.getrf!(hierarchy.coarse_lu, hierarchy.coarse_ipiv)
    hierarchy.coarse_factor = LU(hierarchy.coarse_lu, hierarchy.coarse_ipiv, 0)
    return hierarchy
end
