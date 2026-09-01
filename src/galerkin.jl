"""
    GalerkinWorkspace{Tv, Ti}

Pre-allocated workspace for `compute_coarse_sparsity` to avoid repeated memory
allocation during resetup.  Create once and pass via the `workspace` keyword.
All arrays use grow-only semantics (via `_ws_resize!`) so they remain at their
high-water mark across levels and resetup calls.
"""
mutable struct GalerkinWorkspace{Tv, Ti<:Integer}
    col_marker::Vector{Ti}       # generation-based column marker (n_coarse)
    col_list::Vector{Ti}         # unique columns in current row (n_coarse)
    val_acc::Vector{Tv}          # dense value accumulator (n_coarse)
    unique_per_row::Vector{Ti}   # unique column count per row (n_coarse)
    nz_counts::Vector{Ti}        # triple count per NZ (nnz_c, restriction map only)
    nz_pos::Vector{Ti}           # fill position per NZ (nnz_c, restriction map only)
end

GalerkinWorkspace{Tv, Ti}() where {Tv, Ti} = GalerkinWorkspace{Tv, Ti}(Ti[], Ti[], Tv[], Ti[], Ti[], Ti[])

"""
    compute_coarse_sparsity(A_fine, P, Pt_map, n_coarse)

Determine the sparsity pattern of the coarse grid operator A_c = P^T A_f P.
Returns a CSRMatrix with the correct structure and values,
and a RestrictionMap for in-place updates.

Uses the pre-computed TransposeMap (`Pt_map`) to iterate by coarse row,
avoiding flat triple arrays entirely (O(n_coarse) workspace instead of
O(ntriples)).

The RestrictionMap groups triples by their destination coarse NZ index so that
`galerkin_product!` can parallelize over coarse NZ entries without atomics.
"""
@kernel function _device_dense_csr_pattern_kernel!(rowptr_c, colval_c, n_coarse)
    I = @index(Global)
    @inbounds begin
        rowptr_c[I] = (I - 1) * n_coarse + one(eltype(rowptr_c))
        I == n_coarse && (rowptr_c[I + 1] = I * n_coarse + one(eltype(rowptr_c)))
        base = (I - 1) * n_coarse
        for J in 1:n_coarse
            colval_c[base + J] = J
        end
    end
end

@kernel function _device_galerkin_dense_kernel!(nzval_c, @Const(a_rowptr),
                                                 @Const(a_colval), @Const(a_nzval),
                                                 @Const(p_rowptr), @Const(p_colval),
                                                 @Const(p_nzval), @Const(pt_offsets),
                                                 @Const(pt_rows), @Const(pt_nz_idx),
                                                 n_coarse)
    I = @index(Global)
    @inbounds begin
        base = (I - 1) * n_coarse
        for J in 1:n_coarse
            nzval_c[base + J] = zero(eltype(nzval_c))
        end
        for ptr in pt_offsets[I]:(pt_offsets[I + 1] - 1)
            i = pt_rows[ptr]
            pi = p_nzval[pt_nz_idx[ptr]]
            for anz in a_rowptr[i]:(a_rowptr[i + 1] - 1)
                j = a_colval[anz]
                aij = a_nzval[anz]
                for pnz in p_rowptr[j]:(p_rowptr[j + 1] - 1)
                    J = p_colval[pnz]
                    nzval_c[base + J] += pi * aij * p_nzval[pnz]
                end
            end
        end
    end
end

"""
    _compute_coarse_sparsity_device(A_fine, P, Pt_map, n_coarse)

Backend-neutral device Galerkin fallback. It builds a dense CSR pattern and
forms each coarse row in a KA kernel, keeping all arrays on the selected
backend. Native sparse backends (CUDA cuSPARSE) specialize the public method
with a compact SpGEMM result instead. This fallback is primarily for portable
KA backends and correctness testing, not the preferred large-problem format.
"""
function _compute_coarse_sparsity_device(A_fine::CSRMatrix{Tv, Ti},
                                          P::ProlongationOp{Ti, Tv},
                                          Pt_map::TransposeMap,
                                          n_coarse::Int;
                                          backend=_get_backend(nonzeros(A_fine)),
                                          block_size::Int=64) where {Tv, Ti}
    nnz_c = n_coarse * n_coarse
    rowptr_c = KernelAbstractions.allocate(backend, Ti, n_coarse + 1)
    colval_c = KernelAbstractions.allocate(backend, Ti, nnz_c)
    nzval_c = KernelAbstractions.allocate(backend, Tv, nnz_c)
    _device_dense_csr_pattern_kernel!(backend, block_size)(rowptr_c, colval_c, n_coarse;
        ndrange=n_coarse)
    _device_galerkin_dense_kernel!(backend, block_size)(nzval_c, rowptr(A_fine), colvals(A_fine),
        nonzeros(A_fine), P.rowptr, P.colval, P.nzval, Pt_map.offsets,
        Pt_map.fine_rows, Pt_map.p_nz_idx, n_coarse; ndrange=n_coarse)
    _synchronize(backend)
    return CSRMatrix(rowptr_c, colval_c, nzval_c, n_coarse, n_coarse), nothing
end

function compute_coarse_sparsity(A_fine::CSRMatrix{Tv, Ti},
                                 P::ProlongationOp{Ti, Tv},
                                 Pt_map::TransposeMap,
                                 n_coarse::Int;
                                 build_restriction_map::Bool=true,
                                 workspace::Union{Nothing, GalerkinWorkspace{Tv, Ti}}=nothing,
                                 old_A_coarse::Union{Nothing, CSRMatrix}=nothing,
                                 old_r_map::Union{Nothing, RestrictionMap}=nothing) where {Tv, Ti}
    if !(P.nzval isa Array)
        return _compute_coarse_sparsity_device(A_fine, P, Pt_map, n_coarse;
            backend=_get_backend(nonzeros(A_fine)), block_size=A_fine.block_size)
    end
    n_fine = size(A_fine, 1)
    cv_a = colvals(A_fine)
    nzv_a = nonzeros(A_fine)

    # Workspace arrays (all n_coarse-sized, grow-only)
    if workspace !== nothing
        col_marker = _ws_resize!(workspace.col_marker, n_coarse)
        col_list = _ws_resize!(workspace.col_list, n_coarse)
        val_acc = _ws_resize!(workspace.val_acc, n_coarse)
        unique_per_row = _ws_resize!(workspace.unique_per_row, n_coarse)
    else
        col_marker = Vector{Ti}(undef, n_coarse)
        col_list = Vector{Ti}(undef, n_coarse)
        val_acc = Vector{Tv}(undef, n_coarse)
        unique_per_row = Vector{Ti}(undef, n_coarse)
    end
    @inbounds for k in 1:n_coarse
        col_marker[k] = zero(Ti)
    end
    marker_gen = zero(Ti)

    # Pass 1: Count unique columns per coarse row (iterate via Pt_map).
    @inbounds for I in 1:n_coarse
        marker_gen += one(Ti)
        nuniq = zero(Ti)
        for ptr in Pt_map.offsets[I]:(Pt_map.offsets[I+1]-1)
            i = Pt_map.fine_rows[ptr]
            for anz in nzrange(A_fine, i)
                j = cv_a[anz]
                for pnz_j in P.rowptr[j]:(P.rowptr[j+1]-one(Ti))
                    J = P.colval[pnz_j]
                    if col_marker[J] != marker_gen
                        col_marker[J] = marker_gen
                        nuniq += one(Ti)
                    end
                end
            end
        end
        unique_per_row[I] = nuniq
    end

    # Build coarse CSR rowptr — reuse old_A_coarse arrays when available
    nnz_c = zero(Ti)
    @inbounds for row in 1:n_coarse
        nnz_c += unique_per_row[row]
    end
    if old_A_coarse !== nothing && old_A_coarse.nzval isa Vector
        rowptr_c = resize!(old_A_coarse.rowptr, n_coarse + 1)
        colval_c = resize!(old_A_coarse.colval, Int(nnz_c))
        nzval_c = resize!(old_A_coarse.nzval, Int(nnz_c))
        fill!(nzval_c, zero(Tv))
    else
        rowptr_c = Vector{Ti}(undef, n_coarse + 1)
        colval_c = Vector{Ti}(undef, nnz_c)
        nzval_c = zeros(Tv, nnz_c)
    end
    rowptr_c[1] = one(Ti)
    @inbounds for i in 1:n_coarse
        rowptr_c[i+1] = rowptr_c[i] + unique_per_row[i]
    end

    # For restriction map: prepare per-NZ triple counts
    if build_restriction_map
        if workspace !== nothing
            nz_counts = _ws_resize!(workspace.nz_counts, Int(nnz_c))
        else
            nz_counts = Vector{Ti}(undef, Int(nnz_c))
        end
        @inbounds for k in 1:Int(nnz_c)
            nz_counts[k] = zero(Ti)
        end
    end

    # Pass 2: Fill colval_c and nzval_c using sparse row accumulator,
    # and count triples per NZ for the restriction map.
    @inbounds for I in 1:n_coarse
        marker_gen += one(Ti)
        nuniq = zero(Ti)
        for ptr in Pt_map.offsets[I]:(Pt_map.offsets[I+1]-1)
            i = Pt_map.fine_rows[ptr]
            pi_nz = Pt_map.p_nz_idx[ptr]
            for anz in nzrange(A_fine, i)
                j = cv_a[anz]
                for pnz_j in P.rowptr[j]:(P.rowptr[j+1]-one(Ti))
                    J = P.colval[pnz_j]
                    if col_marker[J] != marker_gen
                        col_marker[J] = marker_gen
                        nuniq += one(Ti)
                        col_list[nuniq] = J
                        val_acc[J] = zero(Tv)
                    end
                    val_acc[J] += P.nzval[pi_nz] * nzv_a[anz] * P.nzval[pnz_j]
                end
            end
        end
        # Sort only the unique columns (much fewer than triples)
        sort!(view(col_list, 1:Int(nuniq)), alg=Base.Sort.InsertionSort)
        # Fill CSR row
        csr_pos = rowptr_c[I]
        for k in one(Ti):nuniq
            col = col_list[k]
            colval_c[csr_pos] = col
            nzval_c[csr_pos] = val_acc[col]
            csr_pos += one(Ti)
        end
        # Count triples per NZ for restriction map (binary search per triple)
        if build_restriction_map
            csr_start = rowptr_c[I]
            csr_end = rowptr_c[I+1] - one(Ti)
            for ptr in Pt_map.offsets[I]:(Pt_map.offsets[I+1]-1)
                i = Pt_map.fine_rows[ptr]
                for anz in nzrange(A_fine, i)
                    j = cv_a[anz]
                    for pnz_j in P.rowptr[j]:(P.rowptr[j+1]-one(Ti))
                        J = P.colval[pnz_j]
                        nz_idx = _find_nz_in_row(colval_c, csr_start, csr_end, J)
                        nz_counts[nz_idx] += one(Ti)
                    end
                end
            end
        end
    end

    A_coarse = CSRMatrix(rowptr_c, colval_c, nzval_c, n_coarse, n_coarse)

    if !build_restriction_map
        return A_coarse, nothing
    end

    # Phase 3: Build RestrictionMap by filling triple arrays.
    # Build offset array from nz_counts.
    if old_r_map !== nothing && old_r_map.nz_offsets isa Vector
        nz_offsets = old_r_map.nz_offsets
        resize!(nz_offsets, Int(nnz_c) + 1)
    else
        nz_offsets = Vector{Ti}(undef, Int(nnz_c) + 1)
    end
    cumsum_val = one(Ti)
    for k in 1:Int(nnz_c)
        nz_offsets[k] = cumsum_val
        cumsum_val += nz_counts[k]
    end
    nz_offsets[Int(nnz_c) + 1] = cumsum_val
    ntriples = Int(cumsum_val) - 1

    # Allocate/reuse triples array
    if old_r_map !== nothing && old_r_map.triples isa Vector
        map_triples = old_r_map.triples
        resize!(map_triples, ntriples)
    else
        map_triples = Vector{NTuple{3,Ti}}(undef, ntriples)
    end

    # Fill position counters (reuse nz_counts or workspace)
    if workspace !== nothing
        nz_pos = _ws_resize!(workspace.nz_pos, Int(nnz_c))
    else
        nz_pos = Vector{Ti}(undef, Int(nnz_c))
    end
    @inbounds for k in 1:Int(nnz_c)
        nz_pos[k] = nz_offsets[k]
    end

    # Pass 3: Fill triple arrays (iterate by coarse row, binary search for NZ index)
    @inbounds for I in 1:n_coarse
        csr_start = rowptr_c[I]
        csr_end = rowptr_c[I+1] - one(Ti)
        for ptr in Pt_map.offsets[I]:(Pt_map.offsets[I+1]-1)
            i = Pt_map.fine_rows[ptr]
            pi_nz = Pt_map.p_nz_idx[ptr]
            for anz in nzrange(A_fine, i)
                j = cv_a[anz]
                for pnz_j in P.rowptr[j]:(P.rowptr[j+1]-one(Ti))
                    J = P.colval[pnz_j]
                    nz_idx = _find_nz_in_row(colval_c, csr_start, csr_end, J)
                    p = nz_pos[nz_idx]
                    map_triples[p] = (Ti(pi_nz), Ti(anz), Ti(pnz_j))
                    nz_pos[nz_idx] += one(Ti)
                end
            end
        end
    end

    r_map = RestrictionMap(nz_offsets, map_triples)
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
                r_map.nz_offsets, r_map.triples; ndrange=nnz_c)
        _synchronize(backend)
    end
    return A_coarse
end

@kernel function galerkin_nz_kernel!(nzv_c, @Const(nzv_f), @Const(P_nzval),
                                     @Const(nz_offsets), @Const(triples))
    k = @index(Global)
    @inbounds begin
        acc = zero(eltype(nzv_c))
        for t in nz_offsets[k]:(nz_offsets[k+1]-1)
            pi_idx, ai_idx, pj_idx = triples[t]
            acc += P_nzval[pi_idx] * nzv_f[ai_idx] * P_nzval[pj_idx]
        end
        nzv_c[k] = acc
    end
end
