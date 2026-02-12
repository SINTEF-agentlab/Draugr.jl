"""
    compute_coarse_sparsity(A_fine, P, n_coarse)

Determine the sparsity pattern of the coarse grid operator A_c = P^T A_f P.
Returns a StaticSparsityMatrixCSR with the correct structure and zero values,
and a RestrictionMap for in-place updates.
"""
function compute_coarse_sparsity(A_fine::StaticSparsityMatrixCSR{Tv, Ti},
                                 P::ProlongationOp{Ti, Tv},
                                 n_coarse::Int) where {Tv, Ti}
    n_fine = size(A_fine, 1)
    cv_a = colvals(A_fine)
    nzv_a = nonzeros(A_fine)
    # Determine which (I,J) pairs exist in A_c
    # For each nonzero A_f[i,j], it contributes to A_c[agg_i, agg_j] for each
    # (agg_i in P-cols of row i) × (agg_j in P-cols of row j)
    # For piecewise-constant P, each row has exactly one column.
    # Build set of (I,J) pairs
    coarse_entries = Dict{Tuple{Int,Int}, Tv}()
    @inbounds for i in 1:n_fine
        for pnz_i in P.rowptr[i]:(P.rowptr[i+1]-1)
            I = P.colval[pnz_i]
            p_i = P.nzval[pnz_i]
            for anz in nzrange(A_fine, i)
                j = cv_a[anz]
                a_ij = nzv_a[anz]
                for pnz_j in P.rowptr[j]:(P.rowptr[j+1]-1)
                    J = P.colval[pnz_j]
                    p_j = P.nzval[pnz_j]
                    key = (I, J)
                    val = p_i * a_ij * p_j
                    coarse_entries[key] = get(coarse_entries, key, zero(Tv)) + val
                end
            end
        end
    end
    # Build CSR arrays for coarse matrix
    # Sort entries by (row, col)
    sorted_keys = sort!(collect(keys(coarse_entries)))
    nnz_c = length(sorted_keys)
    rowptr_c = Vector{Ti}(undef, n_coarse + 1)
    colval_c = Vector{Ti}(undef, nnz_c)
    nzval_c = Vector{Tv}(undef, nnz_c)
    fill!(rowptr_c, Ti(0))
    # Count entries per row
    for (I, J) in sorted_keys
        rowptr_c[I] += 1
    end
    # Cumulative sum
    running_sum = Ti(1)
    for i in 1:n_coarse
        count = rowptr_c[i]
        rowptr_c[i] = running_sum
        running_sum += count
    end
    rowptr_c[n_coarse + 1] = running_sum
    # Fill column indices and values
    pos = copy(rowptr_c[1:n_coarse])
    for (I, J) in sorted_keys
        idx = pos[I]
        colval_c[idx] = Ti(J)
        nzval_c[idx] = coarse_entries[(I, J)]
        pos[I] += 1
    end
    A_coarse = StaticSparsityMatrixCSR(n_coarse, n_coarse, rowptr_c, colval_c, nzval_c)
    # Build restriction map: for each nonzero in A_fine, find corresponding position in A_coarse
    fine_to_coarse_nz = Vector{Ti}(undef, nnz(A_fine))
    @inbounds for i in 1:n_fine
        for pnz_i in P.rowptr[i]:(P.rowptr[i+1]-1)
            I = P.colval[pnz_i]
            for anz in nzrange(A_fine, i)
                j = cv_a[anz]
                for pnz_j in P.rowptr[j]:(P.rowptr[j+1]-1)
                    J = P.colval[pnz_j]
                    # Find position of (I,J) in A_coarse
                    idx = find_nz_index(A_coarse, I, J)
                    fine_to_coarse_nz[anz] = Ti(idx)
                end
            end
        end
    end
    r_map = RestrictionMap{Ti}(fine_to_coarse_nz)
    return A_coarse, r_map
end

"""
    galerkin_product!(A_coarse, A_fine, P, r_map)

In-place Galerkin product: recompute A_coarse values from A_fine and P,
using the precomputed restriction map. This is used during resetup.
"""
function galerkin_product!(A_coarse::StaticSparsityMatrixCSR{Tv, Ti},
                           A_fine::StaticSparsityMatrixCSR{Tv, Ti},
                           P::ProlongationOp{Ti, Tv},
                           r_map::RestrictionMap{Ti}) where {Tv, Ti}
    nzv_c = nonzeros(A_coarse)
    nzv_f = nonzeros(A_fine)
    cv_f = colvals(A_fine)
    n_fine = size(A_fine, 1)
    # Zero out coarse matrix values
    fill!(nzv_c, zero(Tv))
    # Accumulate: for each nonzero in A_fine, add contribution to A_coarse
    @inbounds for i in 1:n_fine
        for pnz_i in P.rowptr[i]:(P.rowptr[i+1]-1)
            p_i = P.nzval[pnz_i]
            for anz in nzrange(A_fine, i)
                j = cv_f[anz]
                a_ij = nzv_f[anz]
                for pnz_j in P.rowptr[j]:(P.rowptr[j+1]-1)
                    p_j = P.nzval[pnz_j]
                    c_idx = r_map.fine_to_coarse_nz[anz]
                    nzv_c[c_idx] += p_i * a_ij * p_j
                end
            end
        end
    end
    return A_coarse
end
