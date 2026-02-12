"""
    build_jacobi_smoother(A, ω)

Build a weighted Jacobi smoother from matrix `A` with damping `ω`.
"""
function build_jacobi_smoother(A::StaticSparsityMatrixCSR{Tv, Ti}, ω::Real) where {Tv, Ti}
    n = size(A, 1)
    invdiag = Vector{Tv}(undef, n)
    compute_inverse_diagonal!(invdiag, A)
    tmp = zeros(Tv, n)
    return JacobiSmoother{Tv}(invdiag, tmp, Tv(ω))
end

"""
    compute_inverse_diagonal!(invdiag, A)

Compute inverse of diagonal entries of A using a KA kernel.
"""
function compute_inverse_diagonal!(invdiag::AbstractVector{Tv},
                                   A::StaticSparsityMatrixCSR{Tv, Ti};
                                   backend=CPU()) where {Tv, Ti}
    n = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)
    rp = rowptr(A)
    kernel! = invdiag_kernel!(backend, 64)
    kernel!(invdiag, nzv, cv, rp; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return invdiag
end

@kernel function invdiag_kernel!(invdiag, @Const(nzval), @Const(colval), @Const(rp))
    i = @index(Global)
    @inbounds begin
        diag_val = zero(eltype(invdiag))
        for nz in rp[i]:(rp[i+1]-1)
            if colval[nz] == i
                diag_val = nzval[nz]
                break
            end
        end
        invdiag[i] = inv(diag_val)
    end
end

"""
    update_smoother!(smoother, A)

Update the smoother for new matrix values (same sparsity pattern).
"""
function update_smoother!(smoother::JacobiSmoother, A::StaticSparsityMatrixCSR;
                          backend=CPU())
    compute_inverse_diagonal!(smoother.invdiag, A; backend=backend)
    return smoother
end

# ── KernelAbstractions-based parallel Jacobi kernel ──────────────────────────

@kernel function jacobi_kernel!(x_new, @Const(x), @Const(b),
                                @Const(nzval), @Const(colval), @Const(rowptr),
                                @Const(invdiag), ω)
    i = @index(Global)
    @inbounds begin
        # Compute residual r_i = b[i] - A[i,:]*x
        r_i = b[i]
        start = rowptr[i]
        stop = rowptr[i+1] - 1
        for nz in start:stop
            j = colval[nz]
            r_i -= nzval[nz] * x[j]
        end
        # Jacobi update: x_new = x + ω * D^{-1} * (b - A*x)
        x_new[i] = x[i] + ω * invdiag[i] * r_i
    end
end

"""
    smooth!(x, A, b, smoother::JacobiSmoother; steps=1)

Apply `steps` iterations of weighted Jacobi smoothing to solve `Ax = b`.
Uses KernelAbstractions for parallel execution.
"""
function smooth!(x::AbstractVector, A::StaticSparsityMatrixCSR, b::AbstractVector,
                 smoother::JacobiSmoother; steps::Int=1, backend=CPU())
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    tmp = smoother.tmp
    for _ in 1:steps
        kernel! = jacobi_kernel!(backend, 64)
        kernel!(tmp, x, b, nzv, cv, rp, smoother.invdiag, smoother.ω; ndrange=n)
        KernelAbstractions.synchronize(backend)
        copyto!(x, tmp)
    end
    return x
end

# ══════════════════════════════════════════════════════════════════════════════
# Parallel Colored Gauss-Seidel Smoother
# ══════════════════════════════════════════════════════════════════════════════

"""
    greedy_coloring(A)

Compute a greedy graph coloring of the adjacency graph of CSR matrix `A`.
Returns `(colors, num_colors)` where `colors[i]` is the color of node i.
"""
function greedy_coloring(A::StaticSparsityMatrixCSR{Tv, Ti}) where {Tv, Ti}
    n = size(A, 1)
    cv = colvals(A)
    colors = zeros(Ti, n)
    num_colors = zero(Ti)
    neighbor_colors = Set{Ti}()
    @inbounds for i in 1:n
        empty!(neighbor_colors)
        for nz in nzrange(A, i)
            j = cv[nz]
            if j != i && colors[j] > 0
                push!(neighbor_colors, colors[j])
            end
        end
        # Find smallest color not used by neighbors
        c = one(Ti)
        while c in neighbor_colors
            c += one(Ti)
        end
        colors[i] = c
        num_colors = max(num_colors, c)
    end
    return colors, Int(num_colors)
end

"""
    build_colored_gs_smoother(A)

Build a parallel colored Gauss-Seidel smoother.
"""
function build_colored_gs_smoother(A::StaticSparsityMatrixCSR{Tv, Ti}) where {Tv, Ti}
    n = size(A, 1)
    colors, num_colors = greedy_coloring(A)
    # Sort nodes by color for efficient parallel iteration
    color_counts = zeros(Int, num_colors)
    @inbounds for i in 1:n
        color_counts[colors[i]] += 1
    end
    color_offsets = Vector{Int}(undef, num_colors + 1)
    color_offsets[1] = 1
    for c in 1:num_colors
        color_offsets[c+1] = color_offsets[c] + color_counts[c]
    end
    color_order = Vector{Ti}(undef, n)
    pos = copy(color_offsets[1:num_colors])
    @inbounds for i in 1:n
        c = colors[i]
        color_order[pos[c]] = Ti(i)
        pos[c] += 1
    end
    invdiag = Vector{Tv}(undef, n)
    compute_inverse_diagonal!(invdiag, A)
    return ColoredGaussSeidelSmoother{Tv, Ti}(colors, color_offsets, color_order,
                                               num_colors, invdiag)
end

function update_smoother!(smoother::ColoredGaussSeidelSmoother, A::StaticSparsityMatrixCSR;
                          backend=CPU())
    compute_inverse_diagonal!(smoother.invdiag, A; backend=backend)
    return smoother
end

@kernel function gs_color_kernel!(x, @Const(b), @Const(nzval), @Const(colval), @Const(rp),
                                  @Const(invdiag), @Const(color_order), offset)
    idx = @index(Global)
    @inbounds begin
        i = color_order[offset + idx]
        # Compute residual r_i = b[i] - A[i,:]*x  (uses latest x values)
        r_i = b[i]
        for nz in rp[i]:(rp[i+1]-1)
            j = colval[nz]
            r_i -= nzval[nz] * x[j]
        end
        # GS update: x[i] += D[i,i]^{-1} * r_i
        x[i] += invdiag[i] * r_i
    end
end

"""
    smooth!(x, A, b, smoother::ColoredGaussSeidelSmoother; steps=1)

Apply parallel colored Gauss-Seidel smoothing.
"""
function smooth!(x::AbstractVector, A::StaticSparsityMatrixCSR, b::AbstractVector,
                 smoother::ColoredGaussSeidelSmoother; steps::Int=1, backend=CPU())
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    for _ in 1:steps
        for c in 1:smoother.num_colors
            start = smoother.color_offsets[c]
            count = smoother.color_offsets[c+1] - start
            count == 0 && continue
            kernel! = gs_color_kernel!(backend, 64)
            kernel!(x, b, nzv, cv, rp, smoother.invdiag,
                    smoother.color_order, start - 1; ndrange=count)
            KernelAbstractions.synchronize(backend)
        end
    end
    return x
end

# ══════════════════════════════════════════════════════════════════════════════
# SPAI(0) Smoother - Diagonal Sparse Approximate Inverse
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_spai0_smoother(A)

Build an SPAI(0) smoother.  For each row i, the diagonal entry is:
  m[i] = a[i,i] / ‖A[i,:]‖₂²
This minimizes ‖e_i - m[i]*A[i,:]‖₂.
"""
function build_spai0_smoother(A::StaticSparsityMatrixCSR{Tv, Ti}) where {Tv, Ti}
    n = size(A, 1)
    m_diag = Vector{Tv}(undef, n)
    _compute_spai0!(m_diag, A)
    tmp = zeros(Tv, n)
    return SPAI0Smoother{Tv}(m_diag, tmp)
end

function _compute_spai0!(m_diag::AbstractVector{Tv}, A::StaticSparsityMatrixCSR{Tv, Ti};
                         backend=CPU()) where {Tv, Ti}
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    kernel! = spai0_kernel!(backend, 64)
    kernel!(m_diag, nzv, cv, rp; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return m_diag
end

@kernel function spai0_kernel!(m_diag, @Const(nzval), @Const(colval), @Const(rp))
    i = @index(Global)
    @inbounds begin
        diag_val = zero(eltype(m_diag))
        row_norm_sq = zero(eltype(m_diag))
        for nz in rp[i]:(rp[i+1]-1)
            v = nzval[nz]
            row_norm_sq += v * v
            if colval[nz] == i
                diag_val = v
            end
        end
        m_diag[i] = row_norm_sq > zero(eltype(m_diag)) ? diag_val / row_norm_sq : zero(eltype(m_diag))
    end
end

function update_smoother!(smoother::SPAI0Smoother, A::StaticSparsityMatrixCSR;
                          backend=CPU())
    _compute_spai0!(smoother.m_diag, A; backend=backend)
    return smoother
end

@kernel function spai0_smooth_kernel!(x_new, @Const(x), @Const(b),
                                      @Const(nzval), @Const(colval), @Const(rp),
                                      @Const(m_diag))
    i = @index(Global)
    @inbounds begin
        r_i = b[i]
        for nz in rp[i]:(rp[i+1]-1)
            j = colval[nz]
            r_i -= nzval[nz] * x[j]
        end
        x_new[i] = x[i] + m_diag[i] * r_i
    end
end

"""
    smooth!(x, A, b, smoother::SPAI0Smoother; steps=1)

Apply SPAI(0) smoothing iterations.
"""
function smooth!(x::AbstractVector, A::StaticSparsityMatrixCSR, b::AbstractVector,
                 smoother::SPAI0Smoother; steps::Int=1, backend=CPU())
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    tmp = smoother.tmp
    for _ in 1:steps
        kernel! = spai0_smooth_kernel!(backend, 64)
        kernel!(tmp, x, b, nzv, cv, rp, smoother.m_diag; ndrange=n)
        KernelAbstractions.synchronize(backend)
        copyto!(x, tmp)
    end
    return x
end

# ══════════════════════════════════════════════════════════════════════════════
# SPAI(1) Smoother - Sparse Approximate Inverse with sparsity of A
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_spai1_smoother(A)

Build an SPAI(1) smoother. For each row i, computes the optimal sparse vector
m_i that minimizes ‖e_i - A^T * m_i‖₂ with sparsity(m_i) ⊆ sparsity(A[i,:]).

This is stored in the same CSR pattern as A but with modified values.
"""
function build_spai1_smoother(A::StaticSparsityMatrixCSR{Tv, Ti}) where {Tv, Ti}
    n = size(A, 1)
    nzval_m = Vector{Tv}(undef, nnz(A))
    _compute_spai1!(nzval_m, A)
    tmp = zeros(Tv, n)
    return SPAI1Smoother{Tv, Ti}(nzval_m, tmp)
end

"""
    _compute_spai1!(nzval_m, A)

Compute SPAI(1) values. For each row i, we solve the small least-squares problem:
  min_{m_i} ‖e_i - A^T m_i‖₂²
where m_i has support on the sparsity pattern of A[i,:].

For efficiency, we compute this row-by-row using the normal equations:
  (A_J^T A_J) m_i = A_J^T e_i = A[:,i][J]
where J = {columns in row i of A} and A_J = A[:, J] restricted to rows that
intersect with columns of row i.
"""
function _compute_spai1!(nzval_m::Vector{Tv}, A::StaticSparsityMatrixCSR{Tv, Ti}) where {Tv, Ti}
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    @inbounds for i in 1:n
        rng_i = rp[i]:(rp[i+1]-1)
        k = length(rng_i)  # number of nonzeros in row i
        if k == 0
            continue
        end
        # Get column indices for row i
        J = cv[rng_i]
        # Build the small k×k Gram matrix G = (A_J)^T (A_J)
        # where A_J are the columns of A indexed by J
        # G[p,q] = A[:,J[p]]' * A[:,J[q]] = sum_r A[r,J[p]] * A[r,J[q]]
        # For CSR, A[r,j] requires scanning row r for column j.
        # We compute G by iterating over rows of A that touch columns in J.
        G = zeros(Tv, k, k)
        rhs = zeros(Tv, k)
        # For each column j in J, we need to find which rows r have A[r,j] != 0
        # In CSR, this means finding rows where j appears in the column list.
        # For efficiency with CSR, compute G[p,q] = sum over rows r of A[r,J[p]]*A[r,J[q]]
        # We iterate over all rows and accumulate.
        # Build a map: column index -> local index in J
        col_to_local = Dict{Ti, Int}()
        for (p, j) in enumerate(J)
            col_to_local[j] = p
        end
        # Iterate over all rows
        for r in 1:n
            rng_r = rp[r]:(rp[r+1]-1)
            # Collect entries in this row that hit columns in J
            local_entries = Tuple{Int, Tv}[]
            for nz in rng_r
                c = cv[nz]
                if haskey(col_to_local, c)
                    push!(local_entries, (col_to_local[c], nzv[nz]))
                end
            end
            isempty(local_entries) && continue
            # Accumulate into G
            for (p, vp) in local_entries
                for (q, vq) in local_entries
                    G[p, q] += vp * vq
                end
                # RHS: A[:,i][J[p]] evaluated at row r, but we want (A^T e_i)[J[p]] = A[i, J[p]]
                # Actually, rhs[p] = (A^T e_i)_J[p] = A[i, J[p]]
            end
        end
        # RHS: for each local index p, rhs[p] = A[i, J[p]]
        for (p, j) in enumerate(J)
            for nz in rng_i
                if cv[nz] == j
                    rhs[p] = nzv[nz]
                    break
                end
            end
        end
        # Solve the small system G * m = rhs
        # Add small regularization for stability
        for p in 1:k
            G[p, p] += eps(Tv) * max(one(Tv), abs(G[p, p]))
        end
        m_local = G \ rhs
        # Store back
        for (p, nz) in enumerate(rng_i)
            nzval_m[nz] = m_local[p]
        end
    end
    return nzval_m
end

function update_smoother!(smoother::SPAI1Smoother, A::StaticSparsityMatrixCSR;
                          backend=CPU())
    _compute_spai1!(smoother.nzval, A)
    return smoother
end

@kernel function spai1_smooth_kernel!(x_new, @Const(x), @Const(b),
                                      @Const(A_nzval), @Const(A_colval), @Const(A_rp),
                                      @Const(M_nzval))
    i = @index(Global)
    @inbounds begin
        # Compute residual r = b - A*x
        r_i = b[i]
        for nz in A_rp[i]:(A_rp[i+1]-1)
            j = A_colval[nz]
            r_i -= A_nzval[nz] * x[j]
        end
        # Apply M[i,:] * r: but since M has same sparsity as A,
        # and we're doing x_new = x + M*(b-Ax), we need the full M*r.
        # However for a smoother we do: x_new[i] = x[i] + sum_j M[i,j] * r_j
        # But we only have r_i computed for row i. We need the full residual.
        # This means we need a two-pass approach.
        # Store residual temporarily
        x_new[i] = r_i
    end
end

@kernel function spai1_apply_kernel!(x, @Const(r), @Const(M_nzval), @Const(colval), @Const(rp))
    i = @index(Global)
    @inbounds begin
        v = zero(eltype(x))
        for nz in rp[i]:(rp[i+1]-1)
            j = colval[nz]
            v += M_nzval[nz] * r[j]
        end
        x[i] += v
    end
end

"""
    smooth!(x, A, b, smoother::SPAI1Smoother; steps=1)

Apply SPAI(1) smoothing: x <- x + M*(b - A*x) where M ≈ A⁻¹.
Two-pass: first compute residual into tmp, then apply M.
"""
function smooth!(x::AbstractVector, A::StaticSparsityMatrixCSR, b::AbstractVector,
                 smoother::SPAI1Smoother; steps::Int=1, backend=CPU())
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    tmp = smoother.tmp
    for _ in 1:steps
        # Pass 1: compute residual r = b - A*x into tmp
        kernel1! = spai1_smooth_kernel!(backend, 64)
        kernel1!(tmp, x, b, nzv, cv, rp, smoother.nzval; ndrange=n)
        KernelAbstractions.synchronize(backend)
        # Pass 2: x += M * r
        kernel2! = spai1_apply_kernel!(backend, 64)
        kernel2!(x, tmp, smoother.nzval, cv, rp; ndrange=n)
        KernelAbstractions.synchronize(backend)
    end
    return x
end

# ══════════════════════════════════════════════════════════════════════════════
# Smoother dispatch based on SmootherType config
# ══════════════════════════════════════════════════════════════════════════════

function build_smoother(A::StaticSparsityMatrixCSR, ::JacobiSmootherType, ω::Real)
    return build_jacobi_smoother(A, ω)
end

function build_smoother(A::StaticSparsityMatrixCSR, ::ColoredGaussSeidelType, ω::Real)
    return build_colored_gs_smoother(A)
end

function build_smoother(A::StaticSparsityMatrixCSR, ::SPAI0SmootherType, ω::Real)
    return build_spai0_smoother(A)
end

function build_smoother(A::StaticSparsityMatrixCSR, ::SPAI1SmootherType, ω::Real)
    return build_spai1_smoother(A)
end
