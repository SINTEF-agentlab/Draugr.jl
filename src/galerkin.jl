"""
    GalerkinWorkspace{Ti}

Pre-allocated workspace for `compute_coarse_sparsity` to avoid repeated memory
allocation during resetup.  Create once and pass via the `workspace` keyword.
"""
mutable struct GalerkinWorkspace{Ti<:Integer}
    raw_I::Vector{Ti}
    raw_J::Vector{Ti}
    raw_ci::Vector{Ti}
    raw_pi::Vector{Ti}
    raw_ai::Vector{Ti}
    raw_pj::Vector{Ti}
    perm::Vector{Ti}
end

GalerkinWorkspace{Ti}() where {Ti} = GalerkinWorkspace{Ti}(Ti[], Ti[], Ti[], Ti[], Ti[], Ti[], Ti[])

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
        raw_pi = resize!(workspace.raw_pi, ntriples)
        raw_ai = resize!(workspace.raw_ai, ntriples)
        raw_pj = resize!(workspace.raw_pj, ntriples)
        perm = resize!(workspace.perm, ntriples)
        if build_restriction_map
            raw_ci = resize!(workspace.raw_ci, ntriples)
        end
    else
        raw_I = Vector{Ti}(undef, ntriples)
        raw_J = Vector{Ti}(undef, ntriples)
        raw_pi = Vector{Ti}(undef, ntriples)
        raw_ai = Vector{Ti}(undef, ntriples)
        raw_pj = Vector{Ti}(undef, ntriples)
        perm = Vector{Ti}(undef, ntriples)
        if build_restriction_map
            raw_ci = Vector{Ti}(undef, ntriples)
        end
    end
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

    # Phase 2: Build coarse CSR structure using a sparse row accumulator.
    # Instead of sorting all triples within each row by column (expensive when
    # triples >> unique columns), we use a generation-based marker to discover
    # unique columns and a dense accumulator for values.  Only the unique column
    # list (much smaller than the number of triples) needs to be sorted.
    col_marker = zeros(Ti, n_coarse)   # generation-based "seen" marker per column
    col_list = Vector{Ti}(undef, n_coarse)  # unique columns found in current row
    marker_gen = zero(Ti)

    # First pass: count unique columns per row
    unique_per_row = zeros(Ti, n_coarse)
    @inbounds for row in 1:n_coarse
        marker_gen += one(Ti)
        nuniq = zero(Ti)
        for k in rowptr_tmp[row]:(rowptr_tmp[row+1] - one(Ti))
            col = raw_J[perm[k]]
            if col_marker[col] != marker_gen
                col_marker[col] = marker_gen
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

    # Second pass: fill colval_c and nzval_c using sparse accumulator,
    # then optionally map triples to NZ indices for the restriction map.
    # NOTE: marker_gen continues from the first pass (not reset) so generation
    # values in col_marker from the first pass cannot collide with second-pass values.
    colval_c = Vector{Ti}(undef, nnz_c)
    nzval_c = zeros(Tv, nnz_c)
    val_acc = Vector{Tv}(undef, n_coarse)  # dense value accumulator (only touched entries are read)

    @inbounds for row in 1:n_coarse
        marker_gen += one(Ti)
        nuniq = zero(Ti)
        rs = rowptr_tmp[row]
        re = rowptr_tmp[row+1] - one(Ti)
        # Discover unique columns and accumulate values
        for k in rs:re
            t = perm[k]
            col = raw_J[t]
            if col_marker[col] != marker_gen
                col_marker[col] = marker_gen
                nuniq += one(Ti)
                col_list[nuniq] = col
                val_acc[col] = zero(Tv)
            end
            val_acc[col] += P.nzval[raw_pi[t]] * nzv_a[raw_ai[t]] * P.nzval[raw_pj[t]]
        end
        # Sort only the unique columns (much fewer than triples)
        sort!(view(col_list, 1:Int(nuniq)), alg=Base.Sort.InsertionSort)
        # Fill CSR row
        csr_pos = rowptr_c[row]
        for i in one(Ti):nuniq
            col = col_list[i]
            colval_c[csr_pos] = col
            nzval_c[csr_pos] = val_acc[col]
            csr_pos += one(Ti)
        end
        # Map each triple to its coarse NZ index via binary search (only for restriction map)
        if build_restriction_map
            csr_start = rowptr_c[row]
            csr_end = rowptr_c[row+1] - one(Ti)
            for k in rs:re
                t = perm[k]
                raw_ci[t] = _find_nz_in_row(colval_c, csr_start, csr_end, raw_J[t])
            end
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

    # Copy permuted triple indices for the RestrictionMap (the originals may be
    # workspace-owned and will be reused on the next call).
    map_pi = Vector{Ti}(undef, ntriples)
    map_ai = Vector{Ti}(undef, ntriples)
    map_pj = Vector{Ti}(undef, ntriples)
    @inbounds for i in 1:ntriples
        p = perm[i]
        map_pi[i] = raw_pi[p]
        map_ai[i] = raw_ai[p]
        map_pj[i] = raw_pj[p]
    end

    # Build an offset array: nz_offsets[k] to nz_offsets[k+1]-1 = triples for coarse NZ k
    nz_offsets = Vector{Ti}(undef, nnz_c + 1)
    fill!(nz_offsets, Ti(0))
    @inbounds for t in 1:ntriples
        nz_offsets[raw_ci[perm[t]]] += Ti(1)
    end
    cumsum_val = Ti(1)
    for k in 1:nnz_c
        cnt = nz_offsets[k]
        nz_offsets[k] = cumsum_val
        cumsum_val += cnt
    end
    nz_offsets[nnz_c + 1] = cumsum_val
    r_map = RestrictionMap(nz_offsets, map_pi, map_ai, map_pj)
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
