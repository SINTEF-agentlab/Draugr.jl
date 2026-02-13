"""
    compute_coarse_sparsity(A_fine, P, n_coarse)

Determine the sparsity pattern of the coarse grid operator A_c = P^T A_f P.
Returns a StaticSparsityMatrixCSR with the correct structure and values,
and a RestrictionMap for in-place updates.
"""
function compute_coarse_sparsity(A_fine::StaticSparsityMatrixCSR{Tv, Ti},
                                 P::ProlongationOp{Ti, Tv},
                                 n_coarse::Int) where {Tv, Ti}
    n_fine = size(A_fine, 1)
    cv_a = colvals(A_fine)
    nzv_a = nonzeros(A_fine)
    # Determine which (I,J) pairs exist in A_c
    coarse_entries = Dict{Tuple{Int,Int}, Tv}()
    # Also build the triple map for resetup
    triple_ci = Ti[]  # coarse NZ index
    triple_pi = Ti[]  # P.nzval index for p_i
    triple_ai = Ti[]  # A.nzval index for a_ij
    triple_pj = Ti[]  # P.nzval index for p_j
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
                    # Store triple (coarse NZ index will be filled after we build the CSR)
                    push!(triple_pi, Ti(pnz_i))
                    push!(triple_ai, Ti(anz))
                    push!(triple_pj, Ti(pnz_j))
                    push!(triple_ci, Ti(0))  # placeholder
                end
            end
        end
    end
    # Build CSR arrays for coarse matrix
    sorted_keys = sort!(collect(keys(coarse_entries)))
    nnz_c = length(sorted_keys)
    rowptr_c = Vector{Ti}(undef, n_coarse + 1)
    colval_c = Vector{Ti}(undef, nnz_c)
    nzval_c = Vector{Tv}(undef, nnz_c)
    fill!(rowptr_c, Ti(0))
    for (I, J) in sorted_keys
        rowptr_c[I] += 1
    end
    running_sum = Ti(1)
    for i in 1:n_coarse
        count = rowptr_c[i]
        rowptr_c[i] = running_sum
        running_sum += count
    end
    rowptr_c[n_coarse + 1] = running_sum
    pos = copy(rowptr_c[1:n_coarse])
    # Build a (I,J) → nz_index map
    coarse_nz_map = Dict{Tuple{Int,Int}, Ti}()
    for (I, J) in sorted_keys
        idx = pos[I]
        colval_c[idx] = Ti(J)
        nzval_c[idx] = coarse_entries[(I, J)]
        coarse_nz_map[(I, J)] = idx
        pos[I] += 1
    end
    A_coarse = StaticSparsityMatrixCSR(n_coarse, n_coarse, rowptr_c, colval_c, nzval_c)
    # Fill coarse NZ indices in the triple map
    # Re-iterate over the same order to fill triple_ci
    k = 0
    @inbounds for i in 1:n_fine
        for pnz_i in P.rowptr[i]:(P.rowptr[i+1]-1)
            I = P.colval[pnz_i]
            for anz in nzrange(A_fine, i)
                j = cv_a[anz]
                for pnz_j in P.rowptr[j]:(P.rowptr[j+1]-1)
                    J = P.colval[pnz_j]
                    k += 1
                    triple_ci[k] = coarse_nz_map[(Int(I), Int(J))]
                end
            end
        end
    end
    r_map = RestrictionMap{Ti}(triple_ci, triple_pi, triple_ai, triple_pj)
    return A_coarse, r_map
end

"""
    galerkin_product!(A_coarse, A_fine, P, r_map)

In-place Galerkin product: recompute A_coarse values from A_fine and P,
using the precomputed restriction map. This is used during resetup.
Uses KernelAbstractions for parallel execution over all triples.
"""
function galerkin_product!(A_coarse::StaticSparsityMatrixCSR{Tv, Ti},
                           A_fine::StaticSparsityMatrixCSR{Tv, Ti},
                           P::ProlongationOp{Ti, Tv},
                           r_map::RestrictionMap{Ti};
                           backend=CPU()) where {Tv, Ti}
    nzv_c = nonzeros(A_coarse)
    nzv_f = nonzeros(A_fine)
    # Zero out coarse matrix values
    fill!(nzv_c, zero(Tv))
    # Accumulate using kernel over all triples
    ntriples = length(r_map.triple_coarse_nz)
    if ntriples > 0
        kernel! = galerkin_triple_kernel!(backend, 64)
        kernel!(nzv_c, nzv_f, P.nzval,
                r_map.triple_coarse_nz, r_map.triple_pi_idx,
                r_map.triple_anz_idx, r_map.triple_pj_idx; ndrange=ntriples)
        KernelAbstractions.synchronize(backend)
    end
    return A_coarse
end

@kernel function galerkin_triple_kernel!(nzv_c, @Const(nzv_f), @Const(P_nzval),
                                          @Const(ci), @Const(pi), @Const(ai), @Const(pj))
    k = @index(Global)
    @inbounds begin
        val = P_nzval[pi[k]] * nzv_f[ai[k]] * P_nzval[pj[k]]
        Atomix.@atomic nzv_c[ci[k]] += val
    end
end
