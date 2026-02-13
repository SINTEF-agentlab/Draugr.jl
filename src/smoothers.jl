"""
    build_jacobi_smoother(A, ω)

Build a weighted Jacobi smoother from matrix `A` with damping `ω`.
"""
function build_jacobi_smoother(A::CSRMatrix{Tv, Ti}, ω::Real) where {Tv, Ti}
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
                                   A::CSRMatrix{Tv, Ti};
                                   backend=DEFAULT_BACKEND) where {Tv, Ti}
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
        row_norm = zero(real(eltype(invdiag)))
        for nz in rp[i]:(rp[i+1]-1)
            row_norm += abs(nzval[nz])
            if colval[nz] == i
                diag_val = nzval[nz]
            end
        end
        # Safe inverse: avoid Inf/NaN for zero or near-zero diagonals
        abs_d = abs(diag_val)
        threshold = eps(real(eltype(invdiag))) * max(one(real(eltype(invdiag))), row_norm)
        invdiag[i] = abs_d > threshold ? inv(diag_val) : zero(eltype(invdiag))
    end
end

"""
    update_smoother!(smoother, A)

Update the smoother for new matrix values (same sparsity pattern).
"""
function update_smoother!(smoother::JacobiSmoother, A::CSRMatrix;
                          backend=DEFAULT_BACKEND)
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
Uses KernelAbstractions for parallel execution. Alternates read/write buffers
to avoid an extra copy per step; only copies back on odd step counts.
"""
function smooth!(x::AbstractVector, A::CSRMatrix, b::AbstractVector,
                 smoother::JacobiSmoother; steps::Int=1, backend=DEFAULT_BACKEND)
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    tmp = smoother.tmp
    src = x
    dst = tmp
    for _ in 1:steps
        kernel! = jacobi_kernel!(backend, 64)
        kernel!(dst, src, b, nzv, cv, rp, smoother.invdiag, smoother.ω; ndrange=n)
        KernelAbstractions.synchronize(backend)
        src, dst = dst, src
    end
    # After the loop, src holds the latest result.
    # If steps is odd, src == tmp, so copy result back to x.
    if isodd(steps)
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
function greedy_coloring(A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
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
function build_colored_gs_smoother(A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
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

function update_smoother!(smoother::ColoredGaussSeidelSmoother, A::CSRMatrix;
                          backend=DEFAULT_BACKEND)
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
function smooth!(x::AbstractVector, A::CSRMatrix, b::AbstractVector,
                 smoother::ColoredGaussSeidelSmoother; steps::Int=1, backend=DEFAULT_BACKEND)
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
function build_spai0_smoother(A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    n = size(A, 1)
    m_diag = Vector{Tv}(undef, n)
    _compute_spai0!(m_diag, A)
    tmp = zeros(Tv, n)
    return SPAI0Smoother{Tv}(m_diag, tmp)
end

function _compute_spai0!(m_diag::AbstractVector{Tv}, A::CSRMatrix{Tv, Ti};
                         backend=DEFAULT_BACKEND) where {Tv, Ti}
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

function update_smoother!(smoother::SPAI0Smoother, A::CSRMatrix;
                          backend=DEFAULT_BACKEND)
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

Apply SPAI(0) smoothing iterations. Alternates buffers to avoid extra copies.
"""
function smooth!(x::AbstractVector, A::CSRMatrix, b::AbstractVector,
                 smoother::SPAI0Smoother; steps::Int=1, backend=DEFAULT_BACKEND)
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    tmp = smoother.tmp
    src = x
    dst = tmp
    for _ in 1:steps
        kernel! = spai0_smooth_kernel!(backend, 64)
        kernel!(dst, src, b, nzv, cv, rp, smoother.m_diag; ndrange=n)
        KernelAbstractions.synchronize(backend)
        src, dst = dst, src
    end
    if isodd(steps)
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
function build_spai1_smoother(A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
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
function _compute_spai1!(nzval_m::Vector{Tv}, A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
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

function update_smoother!(smoother::SPAI1Smoother, A::CSRMatrix;
                          backend=DEFAULT_BACKEND)
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
function smooth!(x::AbstractVector, A::CSRMatrix, b::AbstractVector,
                 smoother::SPAI1Smoother; steps::Int=1, backend=DEFAULT_BACKEND)
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

function build_smoother(A::CSRMatrix, ::JacobiSmootherType, ω::Real; backend=DEFAULT_BACKEND)
    return build_jacobi_smoother(A, ω)
end

function build_smoother(A::CSRMatrix, ::ColoredGaussSeidelType, ω::Real; backend=DEFAULT_BACKEND)
    return build_colored_gs_smoother(A)
end

function build_smoother(A::CSRMatrix, ::SPAI0SmootherType, ω::Real; backend=DEFAULT_BACKEND)
    return build_spai0_smoother(A)
end

function build_smoother(A::CSRMatrix, ::SPAI1SmootherType, ω::Real; backend=DEFAULT_BACKEND)
    return build_spai1_smoother(A)
end

# ══════════════════════════════════════════════════════════════════════════════
# l1-Jacobi Smoother
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_l1jacobi_smoother(A, ω)

Build an l1-Jacobi smoother. Uses l1 row norms for diagonal scaling:
m[i] = ω / (|a_{i,i}| + Σ_{j≠i} |a_{i,j}|)

More robust than standard Jacobi for matrices with large off-diagonal entries,
near-zero diagonals, or wrong-sign off-diagonals.
"""
function build_l1jacobi_smoother(A::CSRMatrix{Tv, Ti}, ω::Real) where {Tv, Ti}
    n = size(A, 1)
    invdiag = Vector{Tv}(undef, n)
    _compute_l1_invdiag!(invdiag, A)
    tmp = zeros(Tv, n)
    return L1JacobiSmoother{Tv}(invdiag, tmp, Tv(ω))
end

function _compute_l1_invdiag!(invdiag::AbstractVector{Tv},
                               A::CSRMatrix{Tv, Ti};
                               backend=DEFAULT_BACKEND) where {Tv, Ti}
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    kernel! = l1_invdiag_kernel!(backend, 64)
    kernel!(invdiag, nzv, cv, rp; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return invdiag
end

@kernel function l1_invdiag_kernel!(invdiag, @Const(nzval), @Const(colval), @Const(rp))
    i = @index(Global)
    @inbounds begin
        l1_norm = zero(real(eltype(invdiag)))
        for nz in rp[i]:(rp[i+1]-1)
            l1_norm += abs(nzval[nz])
        end
        # Safe inverse
        invdiag[i] = l1_norm > eps(real(eltype(invdiag))) ? inv(l1_norm) : zero(eltype(invdiag))
    end
end

function update_smoother!(smoother::L1JacobiSmoother, A::CSRMatrix;
                          backend=DEFAULT_BACKEND)
    _compute_l1_invdiag!(smoother.invdiag, A; backend=backend)
    return smoother
end

function smooth!(x::AbstractVector, A::CSRMatrix, b::AbstractVector,
                 smoother::L1JacobiSmoother; steps::Int=1, backend=DEFAULT_BACKEND)
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    tmp = smoother.tmp
    src = x
    dst = tmp
    for _ in 1:steps
        kernel! = jacobi_kernel!(backend, 64)
        kernel!(dst, src, b, nzv, cv, rp, smoother.invdiag, smoother.ω; ndrange=n)
        KernelAbstractions.synchronize(backend)
        src, dst = dst, src
    end
    if isodd(steps)
        copyto!(x, tmp)
    end
    return x
end

function build_smoother(A::CSRMatrix, ::L1JacobiSmootherType, ω::Real; backend=DEFAULT_BACKEND)
    return build_l1jacobi_smoother(A, ω)
end

# ══════════════════════════════════════════════════════════════════════════════
# Chebyshev Polynomial Smoother
# ══════════════════════════════════════════════════════════════════════════════

"""
    _estimate_spectral_radius(A, invdiag; niter=10)

Estimate the spectral radius of D⁻¹A using power iteration.
"""
function _estimate_spectral_radius(A::CSRMatrix{Tv, Ti},
                                   invdiag::Vector{Tv}; niter::Int=10) where {Tv, Ti}
    n = size(A, 1)
    v = randn(Tv, n)
    v ./= norm(v)
    w = similar(v)
    λ = one(Tv)
    for _ in 1:niter
        mul!(w, A, v)
        @inbounds for i in 1:n
            w[i] *= invdiag[i]
        end
        λ = norm(w)
        if λ > eps(Tv)
            v .= w ./ λ
        end
    end
    return real(λ)
end

"""
    build_chebyshev_smoother(A; degree=3)

Build a Chebyshev polynomial smoother. Estimates eigenvalues of D⁻¹A and
constructs a degree-`degree` Chebyshev iteration.
"""
function build_chebyshev_smoother(A::CSRMatrix{Tv, Ti};
                                  degree::Int=3) where {Tv, Ti}
    n = size(A, 1)
    invdiag = Vector{Tv}(undef, n)
    compute_inverse_diagonal!(invdiag, A)
    ρ = _estimate_spectral_radius(A, invdiag)
    # Standard Chebyshev bounds for SPD: [ρ/30, 1.1*ρ]
    λ_max = Tv(1.1) * ρ
    λ_min = λ_max / Tv(30.0)
    tmp1 = zeros(Tv, n)
    tmp2 = zeros(Tv, n)
    return ChebyshevSmoother{Tv}(invdiag, tmp1, tmp2, λ_min, λ_max, degree)
end

function update_smoother!(smoother::ChebyshevSmoother, A::CSRMatrix;
                          backend=DEFAULT_BACKEND)
    compute_inverse_diagonal!(smoother.invdiag, A; backend=backend)
    ρ = _estimate_spectral_radius(A, smoother.invdiag)
    smoother.λ_max = eltype(smoother.invdiag)(1.1) * ρ
    smoother.λ_min = smoother.λ_max / eltype(smoother.invdiag)(30.0)
    return smoother
end

@kernel function chebyshev_apply_kernel!(x, @Const(r), @Const(invdiag), scale)
    i = @index(Global)
    @inbounds x[i] += scale * invdiag[i] * r[i]
end

"""
    smooth!(x, A, b, smoother::ChebyshevSmoother; steps=1)

Apply Chebyshev polynomial smoothing. Each step applies the full polynomial
of the configured degree using the standard three-term recurrence.
"""
function smooth!(x::AbstractVector, A::CSRMatrix, b::AbstractVector,
                 smoother::ChebyshevSmoother; steps::Int=1, backend=DEFAULT_BACKEND)
    n = size(A, 1)
    Tv = eltype(x)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    r = smoother.tmp1
    d = smoother.tmp2

    θ = (smoother.λ_max + smoother.λ_min) / 2
    δ = (smoother.λ_max - smoother.λ_min) / 2

    for _ in 1:steps
        # Iteration 0: r = b - A*x, d = (1/θ) * D⁻¹ * r, x += d
        rkernel! = residual_kernel_smoother!(backend, 64)
        rkernel!(r, b, x, nzv, cv, rp; ndrange=n)
        KernelAbstractions.synchronize(backend)

        @inbounds for i in 1:n
            d[i] = smoother.invdiag[i] * r[i] / θ
            x[i] += d[i]
        end

        # Iterations 1..degree-1 using three-term recurrence
        σ_old = θ / δ
        for k in 1:(smoother.degree - 1)
            rkernel!(r, b, x, nzv, cv, rp; ndrange=n)
            KernelAbstractions.synchronize(backend)

            σ_new = one(Tv) / (Tv(2) * θ / δ - σ_old)
            @inbounds for i in 1:n
                d[i] = Tv(2) * σ_new / δ * smoother.invdiag[i] * r[i] + σ_new * σ_old * d[i]
                x[i] += d[i]
            end
            σ_old = σ_new
        end
    end
    return x
end

@kernel function residual_kernel_smoother!(r, @Const(b), @Const(x),
                                           @Const(nzval), @Const(colval), @Const(rp))
    i = @index(Global)
    @inbounds begin
        Ax_i = zero(eltype(r))
        for nz in rp[i]:(rp[i+1]-1)
            j = colval[nz]
            Ax_i += nzval[nz] * x[j]
        end
        r[i] = b[i] - Ax_i
    end
end

function build_smoother(A::CSRMatrix, ::ChebyshevSmootherType, ω::Real; backend=DEFAULT_BACKEND)
    return build_chebyshev_smoother(A)
end

# ══════════════════════════════════════════════════════════════════════════════
# ILU(0) Smoother
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_ilu0_smoother(A)

Build a parallel ILU(0) smoother. Computes an incomplete LU factorization with
the same sparsity pattern as A, using graph coloring for parallel forward/backward
substitution.
"""
function build_ilu0_smoother(A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    n = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)
    rp = rowptr(A)

    # Compute coloring for parallel triangular solves
    colors, num_colors = greedy_coloring(A)
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

    # Find diagonal indices
    diag_idx = Vector{Ti}(undef, n)
    @inbounds for i in 1:n
        for nz in rp[i]:(rp[i+1]-1)
            if cv[nz] == i
                diag_idx[i] = Ti(nz)
                break
            end
        end
    end

    # ILU(0) factorization
    L_nzval = zeros(Tv, nnz(A))
    U_nzval = copy(nzv)
    _ilu0_factorize!(L_nzval, U_nzval, diag_idx, A)

    tmp = zeros(Tv, n)
    return ILU0Smoother{Tv, Ti}(L_nzval, U_nzval, diag_idx, colors,
                                 color_offsets, color_order, num_colors, tmp)
end

"""
    _ilu0_factorize!(L_nzval, U_nzval, diag_idx, A)

Compute ILU(0) factorization: A ≈ L*U where L,U have the same sparsity as A.
L has 1 on the diagonal, L_nzval stores strictly lower triangle.
U_nzval stores upper triangle + diagonal.
"""
function _ilu0_factorize!(L_nzval::Vector{Tv}, U_nzval::Vector{Tv},
                          diag_idx::Vector{Ti},
                          A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    n = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)
    rp = rowptr(A)

    # Copy A values into U
    copyto!(U_nzval, nzv)
    fill!(L_nzval, zero(Tv))

    # Maximum factor growth for ILU entries
    const_max_ilu_factor = Tv(1e8)

    @inbounds for i in 1:n
        # Process row i: for each k < i in row i's lower triangle
        for nz in rp[i]:(diag_idx[i]-1)
            k = cv[nz]
            # L[i,k] = U[i,k] / U[k,k]
            u_kk = U_nzval[diag_idx[k]]
            # Use original row k's norm as reference scale for zero check
            row_k_norm = zero(real(Tv))
            for nz_k in rp[k]:(rp[k+1]-1)
                row_k_norm += abs(nzv[nz_k])
            end
            if abs(u_kk) < _safe_threshold(Tv, row_k_norm)
                L_nzval[nz] = zero(Tv)
                U_nzval[nz] = zero(Tv)
                continue
            end
            l_ik = U_nzval[nz] / u_kk
            # Clamp to prevent growth
            if abs(l_ik) > const_max_ilu_factor
                l_ik = sign(l_ik) * const_max_ilu_factor
            end
            L_nzval[nz] = l_ik
            U_nzval[nz] = zero(Tv)  # Clear lower triangle in U

            # Update row i: for each j in row k with j > k, if (i,j) exists
            for nz_k in (diag_idx[k]+1):(rp[k+1]-1)
                j = cv[nz_k]
                # Find (i,j) in row i
                nz_ij = _find_nz_in_row(cv, rp[i], rp[i+1]-1, j)
                if nz_ij > 0
                    U_nzval[nz_ij] -= l_ik * U_nzval[nz_k]
                end
            end
        end
        # Diagonal safeguard: if U[i,i] became zero or near-zero, perturb it
        u_ii = U_nzval[diag_idx[i]]
        row_norm = zero(real(Tv))
        for nz in rp[i]:(rp[i+1]-1)
            row_norm += abs(nzv[nz])
        end
        safe_thresh = _safe_threshold(Tv, row_norm)
        if abs(u_ii) < safe_thresh
            U_nzval[diag_idx[i]] = safe_thresh
        end
    end
    return nothing
end

"""Find column `col` in row range [start, stop] of colval array."""
function _find_nz_in_row(cv::AbstractVector{Ti}, start::Ti, stop::Ti, col::Ti) where Ti
    # Binary search since columns are sorted in CSR
    lo, hi = Int(start), Int(stop)
    while lo <= hi
        mid = (lo + hi) >> 1
        @inbounds c = cv[mid]
        if c == col
            return Ti(mid)
        elseif c < col
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return Ti(0)
end

function update_smoother!(smoother::ILU0Smoother, A::CSRMatrix;
                          backend=DEFAULT_BACKEND)
    _ilu0_factorize!(smoother.L_nzval, smoother.U_nzval, smoother.diag_idx, A)
    return smoother
end

"""
    smooth!(x, A, b, smoother::ILU0Smoother; steps=1)

Apply ILU(0) smoothing: x += (LU)⁻¹ (b - Ax).
Uses sequential forward/backward substitution for robustness.
"""
function smooth!(x::AbstractVector, A::CSRMatrix, b::AbstractVector,
                 smoother::ILU0Smoother; steps::Int=1, backend=DEFAULT_BACKEND)
    n = size(A, 1)
    Tv = eltype(x)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    tmp = smoother.tmp  # residual / solve workspace

    for _ in 1:steps
        # Compute residual: tmp = b - A*x
        rkernel! = residual_kernel_smoother!(backend, 64)
        rkernel!(tmp, b, x, nzv, cv, rp; ndrange=n)
        KernelAbstractions.synchronize(backend)

        # Forward substitution: L * z = tmp  (z stored in tmp, natural row order)
        @inbounds for i in 1:n
            for nz in rp[i]:(smoother.diag_idx[i]-1)
                j = cv[nz]
                tmp[i] -= smoother.L_nzval[nz] * tmp[j]
            end
        end

        # Backward substitution: U * dx = z  (dx stored in tmp, reverse row order)
        @inbounds for i in n:-1:1
            for nz in (smoother.diag_idx[i]+1):(rp[i+1]-1)
                j = cv[nz]
                tmp[i] -= smoother.U_nzval[nz] * tmp[j]
            end
            u_ii = smoother.U_nzval[smoother.diag_idx[i]]
            tmp[i] = abs(u_ii) > eps(real(Tv)) ? tmp[i] / u_ii : zero(Tv)
        end

        # Update: x += dx (with NaN protection)
        @inbounds for i in 1:n
            v = tmp[i]
            if isfinite(v)
                x[i] += v
            end
        end
    end
    return x
end

function build_smoother(A::CSRMatrix, ::ILU0SmootherType, ω::Real; backend=DEFAULT_BACKEND)
    return build_ilu0_smoother(A)
end
