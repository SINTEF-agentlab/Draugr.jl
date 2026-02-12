"""
    amg_resetup!(hierarchy, A_new, config)

Re-setup the AMG hierarchy for new matrix coefficients, reusing the existing
sparsity pattern and prolongation structure. This updates:
1. The fine-grid matrix at the first level
2. Galerkin products at all levels (in-place)
3. Smoothers at all levels
4. The direct solver at the coarsest level
"""
function amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                      A_new::StaticSparsityMatrixCSR{Tv, Ti},
                      config::AMGConfig=AMGConfig()) where {Tv, Ti}
    nlevels = length(hierarchy.levels)
    if nlevels == 0
        # Only one level (coarsest), update direct solver
        _update_coarse_solver!(hierarchy, A_new)
        return hierarchy
    end
    # Update first level's matrix (copy values from A_new into existing structure)
    level1 = hierarchy.levels[1]
    _copy_nzvals!(level1.A, A_new)
    update_smoother!(level1.smoother, level1.A)
    # Update subsequent levels via Galerkin products
    for lvl in 1:(nlevels - 1)
        level = hierarchy.levels[lvl]
        next_level = hierarchy.levels[lvl + 1]
        # Recompute A_coarse in-place
        galerkin_product!(next_level.A, level.A, level.P, level.R_map)
        # Update smoother
        update_smoother!(next_level.smoother, next_level.A)
    end
    # Recompute Galerkin product for the last level to get the coarsest matrix
    last_level = hierarchy.levels[nlevels]
    # The coarsest matrix (for direct solve) is one level below the last AMG level
    # We need to recompute it using the last level's P and R_map
    A_coarsest = _compute_coarsest_matrix(last_level)
    _csr_to_dense!(hierarchy.coarse_A, A_coarsest)
    hierarchy.coarse_factor = lu(hierarchy.coarse_A)
    return hierarchy
end

"""
    _copy_nzvals!(dest, src)

Copy nonzero values from `src` into `dest` (same sparsity pattern).
"""
function _copy_nzvals!(dest::StaticSparsityMatrixCSR, src::StaticSparsityMatrixCSR)
    copyto!(nonzeros(dest), nonzeros(src))
    return dest
end

"""
    _compute_coarsest_matrix(last_level)

Compute the coarsest matrix from the last AMG level's data.
Uses the Galerkin product through the stored maps.
"""
function _compute_coarsest_matrix(level::AMGLevel{Tv, Ti}) where {Tv, Ti}
    # Build a temporary coarse matrix with proper sparsity
    n_fine = size(level.A, 1)
    n_coarse = level.P.ncol
    cv_a = colvals(level.A)
    nzv_a = nonzeros(level.A)
    P = level.P
    coarse_entries = Dict{Tuple{Int,Int}, Tv}()
    @inbounds for i in 1:n_fine
        for pnz_i in P.rowptr[i]:(P.rowptr[i+1]-1)
            I = P.colval[pnz_i]
            p_i = P.nzval[pnz_i]
            for anz in nzrange(level.A, i)
                j = cv_a[anz]
                a_ij = nzv_a[anz]
                for pnz_j in P.rowptr[j]:(P.rowptr[j+1]-1)
                    J = P.colval[pnz_j]
                    p_j = P.nzval[pnz_j]
                    key = (I, J)
                    coarse_entries[key] = get(coarse_entries, key, zero(Tv)) + p_i * a_ij * p_j
                end
            end
        end
    end
    # Build CSR
    sorted_keys = sort!(collect(keys(coarse_entries)))
    nnz_c = length(sorted_keys)
    rowptr_c = Vector{Ti}(undef, n_coarse + 1)
    colval_c = Vector{Ti}(undef, nnz_c)
    nzval_c = Vector{Tv}(undef, nnz_c)
    fill!(rowptr_c, Ti(0))
    for (I, _) in sorted_keys
        rowptr_c[I] += 1
    end
    cumsum_val = Ti(1)
    for i in 1:n_coarse
        count = rowptr_c[i]
        rowptr_c[i] = cumsum_val
        cumsum_val += count
    end
    rowptr_c[n_coarse + 1] = cumsum_val
    pos = copy(rowptr_c[1:n_coarse])
    for (I, J) in sorted_keys
        idx = pos[I]
        colval_c[idx] = Ti(J)
        nzval_c[idx] = coarse_entries[(I, J)]
        pos[I] += 1
    end
    return StaticSparsityMatrixCSR(n_coarse, n_coarse, rowptr_c, colval_c, nzval_c)
end

"""
    _update_coarse_solver!(hierarchy, A)

Update the direct solver at the coarsest level.
"""
function _update_coarse_solver!(hierarchy::AMGHierarchy{Tv}, A::StaticSparsityMatrixCSR{Tv}) where {Tv}
    _csr_to_dense!(hierarchy.coarse_A, A)
    hierarchy.coarse_factor = lu(hierarchy.coarse_A)
    return hierarchy
end
