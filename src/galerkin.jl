"""
    compute_coarse_sparsity(A_fine, P, n_coarse)

Determine the sparsity pattern of the coarse grid operator A_c = P^T A_f P.
Returns a CSRMatrix with the correct structure and values,
and a RestrictionMap for in-place updates.

The RestrictionMap groups triples by their destination coarse NZ index so that
`galerkin_product!` can parallelize over coarse NZ entries without atomics.
"""
function compute_coarse_sparsity(A_fine::CSRMatrix{Tv, Ti},
                                 P::ProlongationOp{Ti, Tv},
                                 n_coarse::Int) where {Tv, Ti}
    n_fine = size(A_fine, 1)
    cv_a = colvals(A_fine)
    nzv_a = nonzeros(A_fine)
    # Determine which (I,J) pairs exist in A_c and collect triples
    coarse_entries = Dict{Tuple{Int,Int}, Tv}()
    # Estimate the number of triples for sizehint!:
    # Each triple comes from (pnz_i, anz, pnz_j) in the triple loop.
    # A rough estimate is nnz(A) * (nnz(P)/n_fine)^2, clamped to a reasonable range.
    nnz_a = length(nzv_a)
    nnz_p = length(P.nzval)
    avg_p_per_row = n_fine > 0 ? nnz_p / n_fine : 1.0
    est_triples = max(nnz_a, round(Int, nnz_a * avg_p_per_row * avg_p_per_row))
    raw_ci = Ti[]  # coarse NZ index (placeholder)
    raw_pi = Ti[]  # P.nzval index for p_i
    raw_ai = Ti[]  # A.nzval index for a_ij
    raw_pj = Ti[]  # P.nzval index for p_j
    sizehint!(raw_ci, est_triples)
    sizehint!(raw_pi, est_triples)
    sizehint!(raw_ai, est_triples)
    sizehint!(raw_pj, est_triples)
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
                    push!(raw_pi, Ti(pnz_i))
                    push!(raw_ai, Ti(anz))
                    push!(raw_pj, Ti(pnz_j))
                    push!(raw_ci, Ti(0))  # placeholder
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
    coarse_nz_map = Dict{Tuple{Int,Int}, Ti}()
    for (I, J) in sorted_keys
        idx = pos[I]
        colval_c[idx] = Ti(J)
        nzval_c[idx] = coarse_entries[(I, J)]
        coarse_nz_map[(I, J)] = idx
        pos[I] += 1
    end
    A_coarse = CSRMatrix(rowptr_c, colval_c, nzval_c, n_coarse, n_coarse)
    # Fill coarse NZ indices in the triple map
    k = 0
    @inbounds for i in 1:n_fine
        for pnz_i in P.rowptr[i]:(P.rowptr[i+1]-1)
            I = P.colval[pnz_i]
            for anz in nzrange(A_fine, i)
                j = cv_a[anz]
                for pnz_j in P.rowptr[j]:(P.rowptr[j+1]-1)
                    J = P.colval[pnz_j]
                    k += 1
                    raw_ci[k] = coarse_nz_map[(Int(I), Int(J))]
                end
            end
        end
    end
    # Group triples by coarse NZ destination for contention-free parallel resetup.
    # Sort triples by their coarse NZ index so each output entry owns a contiguous
    # range of contributing triples.
    ntriples = length(raw_ci)
    perm = sortperm(raw_ci)
    sorted_ci = raw_ci[perm]
    sorted_pi = raw_pi[perm]
    sorted_ai = raw_ai[perm]
    sorted_pj = raw_pj[perm]
    # Build an offset array: nz_offsets[k] to nz_offsets[k+1]-1 = triples for coarse NZ k
    nz_offsets = Vector{Ti}(undef, nnz_c + 1)
    fill!(nz_offsets, Ti(0))
    @inbounds for t in 1:ntriples
        nz_offsets[sorted_ci[t]] += Ti(1)
    end
    cumsum_val = Ti(1)
    for k in 1:nnz_c
        cnt = nz_offsets[k]
        nz_offsets[k] = cumsum_val
        cumsum_val += cnt
    end
    nz_offsets[nnz_c + 1] = cumsum_val
    r_map = RestrictionMap(nz_offsets, sorted_pi, sorted_ai, sorted_pj)
    return A_coarse, r_map
end

"""
    galerkin_product!(A_coarse, A_fine, P, r_map)

In-place Galerkin product: recompute A_coarse values from A_fine and P,
using the precomputed restriction map. This is used during resetup.

Parallelizes over coarse NZ entries (one thread per output entry), with each
thread summing its contributing triples. No atomics needed.
"""
function galerkin_product!(A_coarse::CSRMatrix{Tv, Ti},
                           A_fine::CSRMatrix{Tv, Ti},
                           P::ProlongationOp,
                           r_map::RestrictionMap;
                           backend=_get_backend(nonzeros(A_coarse)), block_size::Int=64) where {Tv, Ti}
    nzv_c = nonzeros(A_coarse)
    nzv_f = nonzeros(A_fine)
    nnz_c = length(nzv_c)
    if nnz_c > 0
        kernel! = galerkin_nz_kernel!(backend, block_size)
        kernel!(nzv_c, nzv_f, P.nzval,
                r_map.nz_offsets, r_map.triple_pi_idx,
                r_map.triple_anz_idx, r_map.triple_pj_idx; ndrange=nnz_c)
        _synchronize(backend)
    end
    return A_coarse
end

@kernel function galerkin_nz_kernel!(nzv_c, @Const(nzv_f), @Const(P_nzval),
                                     @Const(nz_offsets), @Const(pi), @Const(ai), @Const(pj))
    k = @index(Global)
    @inbounds begin
        acc = zero(eltype(nzv_c))
        for t in nz_offsets[k]:(nz_offsets[k+1]-1)
            acc += P_nzval[pi[t]] * nzv_f[ai[t]] * P_nzval[pj[t]]
        end
        nzv_c[k] = acc
    end
end
