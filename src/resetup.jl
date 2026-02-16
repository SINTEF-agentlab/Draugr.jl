"""
    amg_resetup!(hierarchy, A_new::SparseMatrixCSC, config)

External API entry point: convert `SparseMatrixCSC` to `CSRMatrix` once
and forward to the general resetup.

The backend and block_size are taken from the hierarchy (set during `amg_setup`).
"""
function amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                      A_new::SparseMatrixCSC{Tv, Ti},
                      config::AMGConfig=AMGConfig()) where {Tv, Ti}
    A_csr = csr_from_csc(A_new)
    backend = hierarchy.backend
    block_size = hierarchy.block_size
    nlevels = length(hierarchy.levels)
    if nlevels == 0
        _update_coarse_solver!(hierarchy, A_csr; backend=backend, block_size=block_size)
        return hierarchy
    end
    level1 = hierarchy.levels[1]
    _copy_nzvals!(level1.A, A_csr; backend=backend, block_size=block_size)
    update_smoother!(level1.smoother, level1.A; backend=backend, block_size=block_size)
    for lvl in 1:(nlevels - 1)
        level = hierarchy.levels[lvl]
        next_level = hierarchy.levels[lvl + 1]
        galerkin_product!(next_level.A, level.A, level.P, level.R_map; backend=backend, block_size=block_size)
        update_smoother!(next_level.smoother, next_level.A; backend=backend, block_size=block_size)
    end
    last_level = hierarchy.levels[nlevels]
    _recompute_coarsest_dense!(hierarchy, last_level; backend=backend)
    hierarchy.coarse_factor = lu(hierarchy.coarse_A)
    return hierarchy
end

"""
    _copy_nzvals!(dest, src; backend, block_size)

Copy nonzero values from `src` CSRMatrix into `dest` (same sparsity pattern).
When arrays are on the same device, uses a KA kernel. When arrays are on different
devices, falls back to `copyto!`.
"""
function _copy_nzvals!(dest::CSRMatrix, src::CSRMatrix;
                       backend=_get_backend(nonzeros(dest)), block_size::Int=64)
    nzv_d = nonzeros(dest)
    nzv_s = nonzeros(src)
    n = length(nzv_d)
    # If source and dest are on different devices, use copyto! for cross-device transfer
    src_on_cpu = nzv_s isa Array
    dst_on_cpu = nzv_d isa Array
    if src_on_cpu != dst_on_cpu
        copyto!(nzv_d, src_on_cpu ? nzv_s : Array(nzv_s))
    else
        kernel! = copy_kernel!(backend, block_size)
        kernel!(nzv_d, nzv_s; ndrange=n)
        _synchronize(backend)
    end
    return dest
end

"""
    _recompute_coarsest_dense!(hierarchy, last_level; backend=DEFAULT_BACKEND)

Recompute the coarsest dense matrix from the last AMG level in-place,
writing directly into `hierarchy.coarse_A`. Avoids allocating a temporary
coarse CSR matrix. Modifies `hierarchy.coarse_A` as a side effect.

For GPU hierarchies, the Galerkin product is computed on CPU (requires scalar
indexing) and then copied back to the device dense matrix.
"""
function _recompute_coarsest_dense!(hierarchy::AMGHierarchy{Tv, Ti},
                                    level::AMGLevel{Tv, Ti};
                                    backend=DEFAULT_BACKEND) where {Tv, Ti}
    M = hierarchy.coarse_A
    n_coarse = size(M, 1)
    # Always compute on CPU (Galerkin product requires scalar indexing)
    M_cpu = Matrix{Tv}(undef, n_coarse, n_coarse)
    fill!(M_cpu, zero(Tv))
    A_cpu = csr_to_cpu(level.A)
    n_fine = size(A_cpu, 1)
    cv_a = colvals(A_cpu)
    nzv_a = nonzeros(A_cpu)
    rp_a = rowptr(A_cpu)
    P_rowptr = level.P.rowptr isa Array ? level.P.rowptr : Array(level.P.rowptr)
    P_colval = level.P.colval isa Array ? level.P.colval : Array(level.P.colval)
    P_nzval = level.P.nzval isa Array ? level.P.nzval : Array(level.P.nzval)
    # Write directly into dense matrix - iterate over fine rows
    @inbounds for i in 1:n_fine
        for pnz_i in P_rowptr[i]:(P_rowptr[i+1]-1)
            I = P_colval[pnz_i]
            p_i = P_nzval[pnz_i]
            for anz in rp_a[i]:(rp_a[i+1]-1)
                j = cv_a[anz]
                a_ij = nzv_a[anz]
                for pnz_j in P_rowptr[j]:(P_rowptr[j+1]-1)
                    J = P_colval[pnz_j]
                    p_j = P_nzval[pnz_j]
                    M_cpu[I, J] += p_i * a_ij * p_j
                end
            end
        end
    end
    # Copy back to device (no-op for CPU matrices)
    copyto!(M, M_cpu)
    return M
end

"""
    _update_coarse_solver!(hierarchy, A; backend=DEFAULT_BACKEND)

Update the direct solver at the coarsest level using high-level lu().
Handles cross-device scenarios where coarse_A may be CPU while A is on GPU.
"""
function _update_coarse_solver!(hierarchy::AMGHierarchy{Tv}, A::CSRMatrix{Tv};
                                backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv}
    M = hierarchy.coarse_A
    if M isa Matrix
        # Coarse system is on CPU — convert CSR to CPU for _csr_to_dense!
        A_cpu = csr_to_cpu(A)
        _csr_to_dense!(M, A_cpu; block_size=block_size)
    else
        # Coarse system is on device — use device CSR directly
        _csr_to_dense!(M, A; backend=backend, block_size=block_size)
    end
    hierarchy.coarse_factor = lu(hierarchy.coarse_A)
    return hierarchy
end
