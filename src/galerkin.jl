"""
    GalerkinWorkspace{Ti}

Pre-allocated workspace for `compute_coarse_sparsity` to avoid repeated memory
allocation during resetup.  Create once and pass via the `workspace` keyword.
"""
mutable struct GalerkinWorkspace{Ti<:Integer}
    raw_I::Vector{Ti}
    raw_J::Vector{Ti}
    raw_ci::Vector{Ti}
    perm::Vector{Ti}
    col_buf::Vector{Ti}
end

GalerkinWorkspace{Ti}() where {Ti} = GalerkinWorkspace{Ti}(Ti[], Ti[], Ti[], Ti[], Ti[])

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
                                 n_coarse::Int;
                                 build_restriction_map::Bool=true,
                                 workspace::Union{Nothing, GalerkinWorkspace{Ti}}=nothing) where {Tv, Ti}
    n_fine = size(A_fine, 1)
    cv_a = colvals(A_fine)
    nzv_a = nonzeros(A_fine)

    # Pre-compute exact triple count to avoid conservative sizehints and reallocations
    ntriples = 0
    @inbounds for i in 1:n_fine
        p_count_i = Int(P.rowptr[i+1] - P.rowptr[i])
        for anz in nzrange(A_fine, i)
            j = cv_a[anz]
            p_count_j = Int(P.rowptr[j+1] - P.rowptr[j])
            ntriples += p_count_i * p_count_j
        end
    end

    # Phase 1: Collect all (I, J) coarse row-column pairs and triple indices.
    # We store the coarse row I of each triple to build the CSR structure.
    # Reuse workspace arrays when available to avoid reallocation during resetup.
    if workspace !== nothing
        raw_I = resize!(workspace.raw_I, ntriples)
        raw_J = resize!(workspace.raw_J, ntriples)
        raw_ci = resize!(workspace.raw_ci, ntriples)
        perm = resize!(workspace.perm, ntriples)
        col_buf = workspace.col_buf
    else
        raw_I = Vector{Ti}(undef, ntriples)
        raw_J = Vector{Ti}(undef, ntriples)
        raw_ci = Vector{Ti}(undef, ntriples)
        perm = Vector{Ti}(undef, ntriples)
        col_buf = Ti[]
    end
    raw_pi = Vector{Ti}(undef, ntriples)
    raw_ai = Vector{Ti}(undef, ntriples)
    raw_pj = Vector{Ti}(undef, ntriples)
    idx = 0
    @inbounds for i in 1:n_fine
        for pnz_i in P.rowptr[i]:(P.rowptr[i+1]-1)
            I = P.colval[pnz_i]
            for anz in nzrange(A_fine, i)
                j = cv_a[anz]
                for pnz_j in P.rowptr[j]:(P.rowptr[j+1]-1)
                    J = P.colval[pnz_j]
                    idx += 1
                    raw_I[idx] = I
                    raw_J[idx] = J
                    raw_pi[idx] = Ti(pnz_i)
                    raw_ai[idx] = Ti(anz)
                    raw_pj[idx] = Ti(pnz_j)
                end
            end
        end
    end

    # Phase 2: Determine unique (I, J) pairs and build the coarse CSR structure.
    # First count how many triples fall into each coarse row.
    row_counts = zeros(Ti, n_coarse)
    @inbounds for t in 1:ntriples
        row_counts[raw_I[t]] += one(Ti)
    end

    # Build rowptr for a temporary row-sorted ordering
    rowptr_tmp = Vector{Ti}(undef, n_coarse + 1)
    rowptr_tmp[1] = one(Ti)
    @inbounds for i in 1:n_coarse
        rowptr_tmp[i+1] = rowptr_tmp[i] + row_counts[i]
    end

    # Sort triples by coarse row using a counting sort
    pos = copy(rowptr_tmp[1:n_coarse])
    @inbounds for t in 1:ntriples
        row = raw_I[t]
        perm[pos[row]] = Ti(t)
        pos[row] += one(Ti)
    end

    # For each coarse row, find unique columns by sorting columns within the row
    # and build the coarse CSR colval directly.
    # First pass: count unique columns per row
    unique_per_row = zeros(Ti, n_coarse)
    @inbounds for row in 1:n_coarse
        rs = rowptr_tmp[row]
        re = rowptr_tmp[row+1] - one(Ti)
        row_len = re - rs + one(Ti)
        row_len == zero(Ti) && continue
        # Gather columns for this row
        resize!(col_buf, row_len)
        for k in rs:re
            col_buf[k - rs + one(Ti)] = raw_J[perm[k]]
        end
        sort!(col_buf, alg=Base.Sort.InsertionSort)
        # Count unique
        nuniq = one(Ti)
        for k in 2:Ti(row_len)
            if col_buf[k] != col_buf[k - one(Ti)]
                nuniq += one(Ti)
            end
        end
        unique_per_row[row] = nuniq
    end

    # Build coarse CSR rowptr
    nnz_c = zero(Ti)
    @inbounds for row in 1:n_coarse
        nnz_c += unique_per_row[row]
    end
    rowptr_c = Vector{Ti}(undef, n_coarse + 1)
    rowptr_c[1] = one(Ti)
    @inbounds for i in 1:n_coarse
        rowptr_c[i+1] = rowptr_c[i] + unique_per_row[i]
    end

    # Build colval_c (unique sorted columns per row) and map each triple to its NZ index
    colval_c = Vector{Ti}(undef, nnz_c)
    nzval_c = zeros(Tv, nnz_c)

    @inbounds for row in 1:n_coarse
        rs = rowptr_tmp[row]
        re = rowptr_tmp[row+1] - one(Ti)
        row_len = re - rs + one(Ti)
        row_len == zero(Ti) && continue
        # Gather columns for this row (via perm)
        resize!(col_buf, row_len)
        for k in rs:re
            col_buf[k - rs + one(Ti)] = raw_J[perm[k]]
        end
        sort!(col_buf, alg=Base.Sort.InsertionSort)
        # Fill unique columns into colval_c
        csr_pos = rowptr_c[row]
        colval_c[csr_pos] = col_buf[1]
        for k in 2:Ti(row_len)
            if col_buf[k] != col_buf[k - one(Ti)]
                csr_pos += one(Ti)
                colval_c[csr_pos] = col_buf[k]
            end
        end
        # Now map each triple in this row to its NZ index using binary search
        csr_start = rowptr_c[row]
        csr_end = rowptr_c[row+1] - one(Ti)
        for k in rs:re
            t = perm[k]
            J = raw_J[t]
            nz_idx = _find_nz_in_row(colval_c, csr_start, csr_end, J)
            raw_ci[t] = nz_idx
            # Accumulate value
            nzval_c[nz_idx] += P.nzval[raw_pi[t]] * nzv_a[raw_ai[t]] * P.nzval[raw_pj[t]]
        end
    end

    A_coarse = CSRMatrix(rowptr_c, colval_c, nzval_c, n_coarse, n_coarse)

    if !build_restriction_map
        return A_coarse, nothing
    end

    # Phase 3: Group triples by coarse NZ destination for contention-free parallel resetup.
    # Sort triples by their coarse NZ index so each output entry owns a contiguous
    # range of contributing triples.
    # Reuse perm buffer for sortperm! to avoid extra allocation.
    sortperm!(perm, raw_ci)
    # Permute in-place instead of making copies by indexing.
    permute!(raw_ci, perm)
    permute!(raw_pi, perm)
    permute!(raw_ai, perm)
    permute!(raw_pj, perm)

    # Build an offset array: nz_offsets[k] to nz_offsets[k+1]-1 = triples for coarse NZ k
    nz_offsets = Vector{Ti}(undef, nnz_c + 1)
    fill!(nz_offsets, Ti(0))
    @inbounds for t in 1:ntriples
        nz_offsets[raw_ci[t]] += Ti(1)
    end
    cumsum_val = Ti(1)
    for k in 1:nnz_c
        cnt = nz_offsets[k]
        nz_offsets[k] = cumsum_val
        cumsum_val += cnt
    end
    nz_offsets[nnz_c + 1] = cumsum_val
    r_map = RestrictionMap(nz_offsets, raw_pi, raw_ai, raw_pj)
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
