# ── Block-aware helpers ───────────────────────────────────────────────────────
# For scalars, _frobenius_norm2 is just abs2(v). For block matrices (SMatrix),
# it computes the squared Frobenius norm: sum of abs2 of all elements = tr(v' * v).
_frobenius_norm2(v::Number) = abs2(v)
_frobenius_norm2(v) = real(LinearAlgebra.dot(v, v))

# For scalars, _entry_norm is abs(v). For block matrices, it returns the
# Frobenius norm (a scalar). Used for threshold comparisons.
_entry_norm(v::Number) = abs(v)
_entry_norm(v) = sqrt(real(LinearAlgebra.dot(v, v)))

# For scalars, isfinite check. For block types, checks all elements are finite.
_is_finite_entry(v::Number) = isfinite(v)
_is_finite_entry(v) = all(isfinite, v)

"""
    build_jacobi_smoother(A, ω)

Build a weighted Jacobi smoother from matrix `A` with damping `ω`.
"""
function build_jacobi_smoother(A::CSRMatrix{Tv, Ti}, ω::Real) where {Tv, Ti}
    n = size(A, 1)
    invdiag = _allocate_undef_vector(A, Tv, n)
    compute_inverse_diagonal!(invdiag, A)
    tmp = _allocate_vector(A, Tv, n)
    return JacobiSmoother(invdiag, tmp, Tv(ω))
end

"""
    compute_inverse_diagonal!(invdiag, A)

Compute inverse of diagonal entries of A using a KA kernel.
"""
function compute_inverse_diagonal!(invdiag::AbstractVector{Tv},
                                   A::CSRMatrix{Tv, Ti};
                                   backend=_get_backend(nonzeros(A)), block_size::Int=64) where {Tv, Ti}
    n = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)
    rp = rowptr(A)
    kernel! = invdiag_kernel!(backend, block_size)
    kernel!(invdiag, nzv, cv, rp; ndrange=n)
    _synchronize(backend)
    return invdiag
end

@kernel function invdiag_kernel!(invdiag, @Const(nzval), @Const(colval), @Const(rp))
    i = @index(Global)
    @inbounds begin
        diag_val = zero(eltype(invdiag))
        row_norm = zero(real(eltype(invdiag)))
        for nz in rp[i]:(rp[i+1]-1)
            row_norm += _entry_norm(nzval[nz])
            if colval[nz] == i
                diag_val = nzval[nz]
            end
        end
        # Safe inverse: avoid Inf/NaN for zero or near-zero diagonals
        abs_d = _entry_norm(diag_val)
        threshold = eps(real(eltype(invdiag))) * max(one(real(eltype(invdiag))), row_norm)
        invdiag[i] = abs_d > threshold ? inv(diag_val) : zero(eltype(invdiag))
    end
end

"""
    update_smoother!(smoother, A)

Update the smoother for new matrix values (same sparsity pattern).
"""
function update_smoother!(smoother::JacobiSmoother, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64)
    compute_inverse_diagonal!(smoother.invdiag, A; backend=backend, block_size=block_size)
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
                 smoother::JacobiSmoother; steps::Int=1, backend=DEFAULT_BACKEND, block_size::Int=64)
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    tmp = smoother.tmp
    src = x
    dst = tmp
    kernel! = jacobi_kernel!(backend, block_size)
    for _ in 1:steps
        kernel!(dst, src, b, nzv, cv, rp, smoother.invdiag, smoother.ω; ndrange=n)
        _synchronize(backend)
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
Graph coloring is performed on CPU, then color_order and invdiag are
copied to the same device as A.
"""
function build_colored_gs_smoother(A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    n = size(A, 1)
    A_cpu = csr_to_cpu(A)
    colors, num_colors = greedy_coloring(A_cpu)
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
    color_order_cpu = Vector{Ti}(undef, n)
    pos = copy(color_offsets[1:num_colors])
    @inbounds for i in 1:n
        c = colors[i]
        color_order_cpu[pos[c]] = Ti(i)
        pos[c] += 1
    end
    invdiag = _allocate_undef_vector(A, Tv, n)
    compute_inverse_diagonal!(invdiag, A)
    # Copy color_order to device
    color_order_dev = A.nzval isa Array ? color_order_cpu : _to_device(A, color_order_cpu)
    return ColoredGaussSeidelSmoother(colors, color_offsets, color_order_dev,
                                       num_colors, invdiag)
end

function update_smoother!(smoother::ColoredGaussSeidelSmoother, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64)
    compute_inverse_diagonal!(smoother.invdiag, A; backend=backend, block_size=block_size)
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
                 smoother::ColoredGaussSeidelSmoother; steps::Int=1, backend=DEFAULT_BACKEND, block_size::Int=64)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    kernel! = gs_color_kernel!(backend, block_size)
    for _ in 1:steps
        for c in 1:smoother.num_colors
            start = smoother.color_offsets[c]
            count = smoother.color_offsets[c+1] - start
            count == 0 && continue
            kernel!(x, b, nzv, cv, rp, smoother.invdiag,
                    smoother.color_order, start - 1; ndrange=count)
            _synchronize(backend)
        end
    end
    return x
end

# ══════════════════════════════════════════════════════════════════════════════
# L1 Colored Gauss-Seidel Smoother
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_l1_colored_gs_smoother(A)

Build an L1 variant of the parallel colored Gauss-Seidel smoother.
Uses l1 row norms for diagonal scaling instead of just the diagonal entry,
providing more robust smoothing for difficult problems.
Graph coloring is performed on CPU, then color_order and invdiag are
copied to the same device as A.
"""
function build_l1_colored_gs_smoother(A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    n = size(A, 1)
    A_cpu = csr_to_cpu(A)
    colors, num_colors = greedy_coloring(A_cpu)
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
    color_order_cpu = Vector{Ti}(undef, n)
    pos = copy(color_offsets[1:num_colors])
    @inbounds for i in 1:n
        c = colors[i]
        color_order_cpu[pos[c]] = Ti(i)
        pos[c] += 1
    end
    invdiag = _allocate_undef_vector(A, Tv, n)
    _compute_l1_invdiag!(invdiag, A)
    # Copy color_order to device
    color_order_dev = A.nzval isa Array ? color_order_cpu : _to_device(A, color_order_cpu)
    return L1ColoredGaussSeidelSmoother(colors, color_offsets, color_order_dev,
                                         num_colors, invdiag)
end

function update_smoother!(smoother::L1ColoredGaussSeidelSmoother, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64)
    _compute_l1_invdiag!(smoother.invdiag, A; backend=backend, block_size=block_size)
    return smoother
end

"""
    smooth!(x, A, b, smoother::L1ColoredGaussSeidelSmoother; steps=1)

Apply L1 colored Gauss-Seidel smoothing. Uses l1 row norms for diagonal scaling.
"""
function smooth!(x::AbstractVector, A::CSRMatrix, b::AbstractVector,
                 smoother::L1ColoredGaussSeidelSmoother; steps::Int=1, backend=DEFAULT_BACKEND, block_size::Int=64)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    kernel! = gs_color_kernel!(backend, block_size)
    for _ in 1:steps
        for c in 1:smoother.num_colors
            start = smoother.color_offsets[c]
            count = smoother.color_offsets[c+1] - start
            count == 0 && continue
            kernel!(x, b, nzv, cv, rp, smoother.invdiag,
                    smoother.color_order, start - 1; ndrange=count)
            _synchronize(backend)
        end
    end
    return x
end

function build_smoother(A::CSRMatrix, ::L1ColoredGaussSeidelType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64)
    return build_l1_colored_gs_smoother(A)
end

# ══════════════════════════════════════════════════════════════════════════════
# Serial (non-threaded) Gauss-Seidel Smoother
# ══════════════════════════════════════════════════════════════════════════════

"""
    _serial_gs_compute_invdiag!(invdiag, nzv, cv, rp, n)

Function barrier for computing inverse diagonal. Takes concrete array types
as arguments to ensure type stability in the inner loop.
"""
function _serial_gs_compute_invdiag!(invdiag::Vector{Tv}, nzv, cv, rp, n::Int) where {Tv}
    @inbounds for i in 1:n
        d = zero(Tv)
        for nz in rp[i]:(rp[i+1]-1)
            if cv[nz] == i
                d = nzv[nz]
                break
            end
        end
        abs_d = _entry_norm(d)
        invdiag[i] = abs_d > eps(real(Tv)) ? inv(d) : zero(Tv)
    end
    return invdiag
end

"""
    _serial_gs_sweep!(x, b, nzv, cv, rp, invdiag, n, steps)

Function barrier for the Gauss-Seidel forward sweep. Takes concrete array types
as arguments to ensure type stability in the inner loop.
"""
function _serial_gs_sweep!(x, b, nzv, cv, rp, invdiag, n::Int, steps::Int)
    for _ in 1:steps
        @inbounds for i in 1:n
            r_i = b[i]
            for nz in rp[i]:(rp[i+1]-1)
                j = cv[nz]
                r_i -= nzv[nz] * x[j]
            end
            x[i] += invdiag[i] * r_i
        end
    end
    return x
end

"""
    build_serial_gs_smoother(A)

Build a serial Gauss-Seidel smoother. All data is stored on CPU.
No graph coloring, threading, or KernelAbstractions are used.
"""
function build_serial_gs_smoother(A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    A_cpu = csr_to_cpu(A)
    n = size(A_cpu, 1)
    invdiag = Vector{Tv}(undef, n)
    _serial_gs_compute_invdiag!(invdiag, nonzeros(A_cpu), colvals(A_cpu), rowptr(A_cpu), n)
    return SerialGaussSeidelSmoother{Tv, Ti}(invdiag, A_cpu)
end

function update_smoother!(smoother::SerialGaussSeidelSmoother{Tv, Ti}, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64) where {Tv, Ti}
    A_cpu = csr_to_cpu(A)
    copyto!(smoother.A_cpu.nzval, A_cpu.nzval)
    n = size(A_cpu, 1)
    _serial_gs_compute_invdiag!(smoother.invdiag, nonzeros(smoother.A_cpu), colvals(smoother.A_cpu), rowptr(smoother.A_cpu), n)
    return smoother
end

"""
    smooth!(x, A, b, smoother::SerialGaussSeidelSmoother; steps=1)

Apply serial Gauss-Seidel smoothing. Performs a sequential forward sweep
over all rows without threading or KernelAbstractions. For GPU arrays,
copies data to CPU, applies GS, and copies back.
"""
function smooth!(x::AbstractVector, A::CSRMatrix{Tv, Ti}, b::AbstractVector,
                 smoother::SerialGaussSeidelSmoother; steps::Int=1, backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    n = size(A, 1)
    is_gpu = !(x isa Array)
    if is_gpu
        x_cpu = Array(x)
        b_cpu = Array(b)
    else
        x_cpu = x
        b_cpu = b
    end
    _serial_gs_sweep!(x_cpu, b_cpu, nonzeros(smoother.A_cpu), colvals(smoother.A_cpu),
                      rowptr(smoother.A_cpu), smoother.invdiag, n, steps)
    if is_gpu
        copyto!(x, x_cpu)
    end
    return x
end

function build_smoother(A::CSRMatrix, ::SerialGaussSeidelType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64)
    return build_serial_gs_smoother(A)
end

# ══════════════════════════════════════════════════════════════════════════════
# L1 Serial (non-threaded) Gauss-Seidel Smoother
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_l1_serial_gs_smoother(A)

Build a serial L1 Gauss-Seidel smoother matching hypre's default l1-GS relaxation.
Uses l1 row norms for diagonal scaling. All data is stored on CPU.
No graph coloring, threading, or KernelAbstractions are used.
"""
function build_l1_serial_gs_smoother(A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    A_cpu = csr_to_cpu(A)
    n = size(A_cpu, 1)
    invdiag = Vector{Tv}(undef, n)
    _serial_l1_gs_compute_invdiag!(invdiag, nonzeros(A_cpu), colvals(A_cpu), rowptr(A_cpu), n)
    return L1SerialGaussSeidelSmoother{Tv, Ti}(invdiag, A_cpu)
end

"""
    _serial_l1_gs_compute_invdiag!(invdiag, nzv, cv, rp, n)

Compute inverse l1 row norms for serial L1 GS smoother.
"""
function _serial_l1_gs_compute_invdiag!(invdiag::Vector{Tv}, nzv, cv, rp, n::Int) where {Tv}
    @inbounds for i in 1:n
        l1_norm = zero(real(Tv))
        for nz in rp[i]:(rp[i+1]-1)
            l1_norm += _entry_norm(nzv[nz])
        end
        invdiag[i] = l1_norm > eps(real(Tv)) ? inv(l1_norm) : zero(Tv)
    end
    return invdiag
end

function update_smoother!(smoother::L1SerialGaussSeidelSmoother{Tv, Ti}, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64) where {Tv, Ti}
    A_cpu = csr_to_cpu(A)
    copyto!(smoother.A_cpu.nzval, A_cpu.nzval)
    n = size(A_cpu, 1)
    _serial_l1_gs_compute_invdiag!(smoother.invdiag, nonzeros(smoother.A_cpu), colvals(smoother.A_cpu), rowptr(smoother.A_cpu), n)
    return smoother
end

"""
    smooth!(x, A, b, smoother::L1SerialGaussSeidelSmoother; steps=1)

Apply serial L1 Gauss-Seidel smoothing. Performs a sequential forward sweep
over all rows using l1 row norms for diagonal scaling. For GPU arrays,
copies data to CPU, applies GS, and copies back.
"""
function smooth!(x::AbstractVector, A::CSRMatrix{Tv, Ti}, b::AbstractVector,
                 smoother::L1SerialGaussSeidelSmoother; steps::Int=1, backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    n = size(A, 1)
    is_gpu = !(x isa Array)
    if is_gpu
        x_cpu = Array(x)
        b_cpu = Array(b)
    else
        x_cpu = x
        b_cpu = b
    end
    _serial_gs_sweep!(x_cpu, b_cpu, nonzeros(smoother.A_cpu), colvals(smoother.A_cpu),
                      rowptr(smoother.A_cpu), smoother.invdiag, n, steps)
    if is_gpu
        copyto!(x, x_cpu)
    end
    return x
end

function build_smoother(A::CSRMatrix, ::L1SerialGaussSeidelType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64)
    return build_l1_serial_gs_smoother(A)
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
    m_diag = _allocate_undef_vector(A, Tv, n)
    _compute_spai0!(m_diag, A)
    tmp = _allocate_vector(A, Tv, n)
    return SPAI0Smoother(m_diag, tmp)
end

function _compute_spai0!(m_diag::AbstractVector{Tv}, A::CSRMatrix{Tv, Ti};
                         backend=_get_backend(nonzeros(A)), block_size::Int=64) where {Tv, Ti}
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    kernel! = spai0_kernel!(backend, block_size)
    kernel!(m_diag, nzv, cv, rp; ndrange=n)
    _synchronize(backend)
    return m_diag
end

@kernel function spai0_kernel!(m_diag, @Const(nzval), @Const(colval), @Const(rp))
    i = @index(Global)
    @inbounds begin
        diag_val = zero(eltype(m_diag))
        row_norm_sq = zero(real(eltype(nzval)))
        for nz in rp[i]:(rp[i+1]-1)
            v = nzval[nz]
            row_norm_sq += _frobenius_norm2(v)
            if colval[nz] == i
                diag_val = v
            end
        end
        m_diag[i] = row_norm_sq > zero(row_norm_sq) ? diag_val / row_norm_sq : zero(eltype(m_diag))
    end
end

function update_smoother!(smoother::SPAI0Smoother, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64)
    _compute_spai0!(smoother.m_diag, A; backend=backend, block_size=block_size)
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
                 smoother::SPAI0Smoother; steps::Int=1, backend=DEFAULT_BACKEND, block_size::Int=64)
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    tmp = smoother.tmp
    src = x
    dst = tmp
    kernel! = spai0_smooth_kernel!(backend, block_size)
    for _ in 1:steps
        kernel!(dst, src, b, nzv, cv, rp, smoother.m_diag; ndrange=n)
        _synchronize(backend)
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
    A_cpu = csr_to_cpu(A)
    nzval_m = Vector{Tv}(undef, nnz(A))
    _compute_spai1!(nzval_m, A_cpu)
    # Copy nzval to device if needed
    nzval_dev = A.nzval isa Array ? nzval_m : _to_device(A, nzval_m)
    tmp = _allocate_vector(A, Tv, n)
    return SPAI1Smoother{Tv, Ti, typeof(nzval_dev)}(nzval_dev, tmp)
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
                          backend=_get_backend(nonzeros(A)), block_size::Int=64)
    A_cpu = csr_to_cpu(A)
    nzval_cpu = Vector{eltype(smoother.nzval)}(undef, nnz(A))
    _compute_spai1!(nzval_cpu, A_cpu)
    copyto!(smoother.nzval, nzval_cpu)
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
                 smoother::SPAI1Smoother; steps::Int=1, backend=DEFAULT_BACKEND, block_size::Int=64)
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    tmp = smoother.tmp
    kernel1! = spai1_smooth_kernel!(backend, block_size)
    kernel2! = spai1_apply_kernel!(backend, block_size)
    for _ in 1:steps
        # Pass 1: compute residual r = b - A*x into tmp
        kernel1!(tmp, x, b, nzv, cv, rp, smoother.nzval; ndrange=n)
        _synchronize(backend)
        # Pass 2: x += M * r
        kernel2!(x, tmp, smoother.nzval, cv, rp; ndrange=n)
        _synchronize(backend)
    end
    return x
end

# ══════════════════════════════════════════════════════════════════════════════
# Smoother dispatch based on SmootherType config
# ══════════════════════════════════════════════════════════════════════════════

function build_smoother(A::CSRMatrix, ::JacobiSmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64)
    return build_jacobi_smoother(A, ω)
end

function build_smoother(A::CSRMatrix, ::ColoredGaussSeidelType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64)
    return build_colored_gs_smoother(A)
end

function build_smoother(A::CSRMatrix, ::SPAI0SmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64)
    return build_spai0_smoother(A)
end

function build_smoother(A::CSRMatrix, ::SPAI1SmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64)
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
    invdiag = _allocate_undef_vector(A, Tv, n)
    _compute_l1_invdiag!(invdiag, A)
    tmp = _allocate_vector(A, Tv, n)
    return L1JacobiSmoother(invdiag, tmp, Tv(ω))
end

function _compute_l1_invdiag!(invdiag::AbstractVector{Tv},
                               A::CSRMatrix{Tv, Ti};
                               backend=_get_backend(nonzeros(A)), block_size::Int=64) where {Tv, Ti}
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    kernel! = l1_invdiag_kernel!(backend, block_size)
    kernel!(invdiag, nzv, cv, rp; ndrange=n)
    _synchronize(backend)
    return invdiag
end

@kernel function l1_invdiag_kernel!(invdiag, @Const(nzval), @Const(colval), @Const(rp))
    i = @index(Global)
    @inbounds begin
        l1_norm = zero(real(eltype(invdiag)))
        for nz in rp[i]:(rp[i+1]-1)
            l1_norm += _entry_norm(nzval[nz])
        end
        # Safe inverse: for scalars inv(l1_norm) is 1/l1_norm,
        # for block systems this gives a scalar inverse which scales the identity.
        invdiag[i] = l1_norm > eps(real(eltype(invdiag))) ? inv(l1_norm) : zero(eltype(invdiag))
    end
end

function update_smoother!(smoother::L1JacobiSmoother, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64)
    _compute_l1_invdiag!(smoother.invdiag, A; backend=backend, block_size=block_size)
    return smoother
end

function smooth!(x::AbstractVector, A::CSRMatrix, b::AbstractVector,
                 smoother::L1JacobiSmoother; steps::Int=1, backend=DEFAULT_BACKEND, block_size::Int=64)
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    tmp = smoother.tmp
    src = x
    dst = tmp
    kernel! = jacobi_kernel!(backend, block_size)
    for _ in 1:steps
        kernel!(dst, src, b, nzv, cv, rp, smoother.invdiag, smoother.ω; ndrange=n)
        _synchronize(backend)
        src, dst = dst, src
    end
    if isodd(steps)
        copyto!(x, tmp)
    end
    return x
end

function build_smoother(A::CSRMatrix, ::L1JacobiSmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64)
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
    invdiag = _allocate_undef_vector(A, Tv, n)
    compute_inverse_diagonal!(invdiag, A)
    # Spectral radius estimation uses scalar indexing and mul!, which require CPU arrays
    invdiag_cpu = invdiag isa Array ? invdiag : Array(invdiag)
    A_cpu = csr_to_cpu(A)
    ρ = _estimate_spectral_radius(A_cpu, invdiag_cpu)
    # Standard Chebyshev bounds for SPD: [ρ/30, 1.1*ρ]
    λ_max = Tv(1.1) * ρ
    λ_min = λ_max / Tv(30.0)
    tmp1 = _allocate_vector(A, Tv, n)
    tmp2 = _allocate_vector(A, Tv, n)
    return ChebyshevSmoother(invdiag, tmp1, tmp2, λ_min, λ_max, degree)
end

function update_smoother!(smoother::ChebyshevSmoother, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64)
    compute_inverse_diagonal!(smoother.invdiag, A; backend=backend, block_size=block_size)
    # Spectral radius estimation requires CPU arrays
    invdiag_cpu = smoother.invdiag isa Array ? smoother.invdiag : Array(smoother.invdiag)
    A_cpu = csr_to_cpu(A)
    ρ = _estimate_spectral_radius(A_cpu, invdiag_cpu)
    smoother.λ_max = eltype(smoother.invdiag)(1.1) * ρ
    smoother.λ_min = smoother.λ_max / eltype(smoother.invdiag)(30.0)
    return smoother
end

@kernel function chebyshev_init_kernel!(d, x, @Const(invdiag), @Const(r), inv_θ)
    i = @index(Global)
    @inbounds begin
        d[i] = invdiag[i] * r[i] * inv_θ
        x[i] += d[i]
    end
end

@kernel function chebyshev_iter_kernel!(d, x, @Const(invdiag), @Const(r), scale_r, scale_d)
    i = @index(Global)
    @inbounds begin
        d[i] = scale_r * invdiag[i] * r[i] + scale_d * d[i]
        x[i] += d[i]
    end
end

"""
    smooth!(x, A, b, smoother::ChebyshevSmoother; steps=1)

Apply Chebyshev polynomial smoothing. Each step applies the full polynomial
of the configured degree using the standard three-term recurrence.
Uses KA kernels for GPU compatibility.
"""
function smooth!(x::AbstractVector, A::CSRMatrix, b::AbstractVector,
                 smoother::ChebyshevSmoother; steps::Int=1, backend=DEFAULT_BACKEND, block_size::Int=64)
    n = size(A, 1)
    Tv = eltype(x)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    r = smoother.tmp1
    d = smoother.tmp2

    θ = (smoother.λ_max + smoother.λ_min) / 2
    δ = (smoother.λ_max - smoother.λ_min) / 2

    rkernel! = residual_kernel_smoother!(backend, block_size)
    init_kernel! = chebyshev_init_kernel!(backend, block_size)
    iter_kernel! = chebyshev_iter_kernel!(backend, block_size)
    for _ in 1:steps
        # Iteration 0: r = b - A*x, d = (1/θ) * D⁻¹ * r, x += d
        rkernel!(r, b, x, nzv, cv, rp; ndrange=n)
        _synchronize(backend)

        init_kernel!(d, x, smoother.invdiag, r, Tv(1) / θ; ndrange=n)
        _synchronize(backend)

        # Iterations 1..degree-1 using three-term recurrence
        σ_old = θ / δ
        for k in 1:(smoother.degree - 1)
            rkernel!(r, b, x, nzv, cv, rp; ndrange=n)
            _synchronize(backend)

            σ_new = one(Tv) / (Tv(2) * θ / δ - σ_old)
            scale_r = Tv(2) * σ_new / δ
            scale_d = σ_new * σ_old
            iter_kernel!(d, x, smoother.invdiag, r, scale_r, scale_d; ndrange=n)
            _synchronize(backend)
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

function build_smoother(A::CSRMatrix, ::ChebyshevSmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64)
    return build_chebyshev_smoother(A)
end

# ══════════════════════════════════════════════════════════════════════════════
# ILU(0) Smoother
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_ilu0_smoother(A)

Build a parallel ILU(0) smoother. Computes an incomplete LU factorization with
the same sparsity pattern as A, using graph coloring for parallel forward/backward
substitution. All factorization data is stored on CPU; GPU matrices are
automatically converted.
"""
function build_ilu0_smoother(A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    A_cpu = csr_to_cpu(A)
    n = size(A_cpu, 1)
    cv = colvals(A_cpu)
    nzv = nonzeros(A_cpu)
    rp = rowptr(A_cpu)
    ti_one = one(Ti)

    # Compute coloring for parallel triangular solves (on CPU)
    colors, num_colors = greedy_coloring(A_cpu)
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
        for nz in rp[i]:(rp[i+ti_one]-ti_one)
            if cv[nz] == i
                diag_idx[i] = Ti(nz)
                break
            end
        end
    end

    # ILU(0) factorization (on CPU)
    L_nzval = zeros(Tv, nnz(A_cpu))
    U_nzval = copy(nzv)
    _ilu0_factorize!(L_nzval, U_nzval, diag_idx, A_cpu)

    tmp = zeros(Tv, n)
    return ILU0Smoother{Tv, Ti}(L_nzval, U_nzval, diag_idx, colors,
                                 color_offsets, color_order, num_colors, tmp, A_cpu)
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
    ti_one = one(Ti)

    # Copy A values into U
    copyto!(U_nzval, nzv)
    fill!(L_nzval, zero(Tv))

    # Maximum factor growth for ILU entries
    const_max_ilu_factor = Tv(1e8)

    # Precompute row norms to avoid redundant recomputation in inner loop
    row_norms = Vector{real(Tv)}(undef, n)
    @inbounds for i in 1:n
        s = zero(real(Tv))
        for nz in rp[i]:(rp[i+ti_one]-ti_one)
            s += _entry_norm(nzv[nz])
        end
        row_norms[i] = s
    end

    @inbounds for i in 1:n
        # Process row i: for each k < i in row i's lower triangle
        for nz in rp[i]:(diag_idx[i]-ti_one)
            k = cv[nz]
            # L[i,k] = U[i,k] * U[k,k]⁻¹ (right division for block compatibility)
            u_kk = U_nzval[diag_idx[k]]
            if _entry_norm(u_kk) < _safe_threshold(Tv, row_norms[k])
                L_nzval[nz] = zero(Tv)
                U_nzval[nz] = zero(Tv)
                continue
            end
            l_ik = U_nzval[nz] / u_kk
            # Clamp to prevent growth
            l_ik_norm = _entry_norm(l_ik)
            if l_ik_norm > const_max_ilu_factor
                l_ik = l_ik * (const_max_ilu_factor / l_ik_norm)
            end
            L_nzval[nz] = l_ik
            U_nzval[nz] = zero(Tv)  # Clear lower triangle in U

            # Update row i: for each j in row k with j > k, if (i,j) exists
            for nz_k in (diag_idx[k]+ti_one):(rp[k+ti_one]-ti_one)
                j = cv[nz_k]
                # Find (i,j) in row i
                nz_ij = _find_nz_in_row(cv, rp[i], rp[i+ti_one]-ti_one, j)
                if nz_ij > 0
                    U_nzval[nz_ij] -= l_ik * U_nzval[nz_k]
                end
            end
        end
        # Diagonal safeguard: if U[i,i] became zero or near-zero, perturb it
        u_ii = U_nzval[diag_idx[i]]
        safe_thresh = _safe_threshold(Tv, row_norms[i])
        if _entry_norm(u_ii) < safe_thresh
            U_nzval[diag_idx[i]] = safe_thresh
        end
    end
    return nothing
end

"""Find column `col` in row range [start, stop] of colval array."""
function _find_nz_in_row(cv::AbstractVector{Ti}, start, stop, col) where Ti
    # Binary search since columns are sorted in CSR
    lo, hi = Ti(start), Ti(stop)
    while lo <= hi
        mid = (lo + hi) >> 1
        @inbounds c = cv[mid]
        if c == col
            return mid
        elseif c < col
            lo = mid + one(Ti)
        else
            hi = mid - one(Ti)
        end
    end
    return zero(Ti)
end

function update_smoother!(smoother::ILU0Smoother, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64)
    A_cpu = csr_to_cpu(A)
    copyto!(smoother.A_cpu.nzval, A_cpu.nzval)
    _ilu0_factorize!(smoother.L_nzval, smoother.U_nzval, smoother.diag_idx, smoother.A_cpu)
    return smoother
end

"""
    smooth!(x, A, b, smoother::ILU0Smoother; steps=1)

Apply ILU(0) smoothing: x += (LU)⁻¹ (b - Ax).
Uses sequential forward/backward substitution on CPU.
For GPU arrays, copies data to CPU, applies ILU, and copies back.
"""
function smooth!(x::AbstractVector, A::CSRMatrix{Tv, Ti}, b::AbstractVector,
                 smoother::ILU0Smoother; steps::Int=1, backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    n = size(A, 1)
    ti_one = one(Ti)

    # Use CPU arrays for sequential ILU solve
    is_gpu = !(x isa Array)
    if is_gpu
        x_cpu = Array(x)
        b_cpu = Array(b)
        A_cpu = smoother.A_cpu
    else
        x_cpu = x
        b_cpu = b
        A_cpu = A
    end

    nzv = nonzeros(A_cpu)
    cv = colvals(A_cpu)
    rp = rowptr(A_cpu)
    tmp = smoother.tmp  # always CPU

    for _ in 1:steps
        # Compute residual: tmp = b - A*x (on CPU)
        @inbounds for i in 1:n
            Ax_i = zero(Tv)
            for nz in rp[i]:(rp[i+ti_one]-ti_one)
                j = cv[nz]
                Ax_i += nzv[nz] * x_cpu[j]
            end
            tmp[i] = b_cpu[i] - Ax_i
        end

        # Forward substitution: L * z = tmp  (z stored in tmp, natural row order)
        @inbounds for i in 1:n
            for nz in rp[i]:(smoother.diag_idx[i]-ti_one)
                j = cv[nz]
                tmp[i] -= smoother.L_nzval[nz] * tmp[j]
            end
        end

        # Backward substitution: U * dx = z  (dx stored in tmp, reverse row order)
        @inbounds for i in n:-1:1
            for nz in (smoother.diag_idx[i]+ti_one):(rp[i+ti_one]-ti_one)
                j = cv[nz]
                tmp[i] -= smoother.U_nzval[nz] * tmp[j]
            end
            u_ii = smoother.U_nzval[smoother.diag_idx[i]]
            tmp[i] = _entry_norm(u_ii) > eps(real(Tv)) ? u_ii \ tmp[i] : zero(Tv)
        end

        # Update: x += dx (with NaN protection)
        @inbounds for i in 1:n
            v = tmp[i]
            if _is_finite_entry(v)
                x_cpu[i] += v
            end
        end
    end

    # Copy result back to GPU if needed
    if is_gpu
        copyto!(x, x_cpu)
    end
    return x
end

function build_smoother(A::CSRMatrix, ::ILU0SmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64)
    return build_ilu0_smoother(A)
end

# ══════════════════════════════════════════════════════════════════════════════
# Standalone smoother API
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_smoother(A::SparseMatrixCSC, smoother_type::SmootherType; ω=2/3, backend, block_size)

Build a smoother from a `SparseMatrixCSC` matrix. This is the public API
for using smoothers independently of the AMG hierarchy.

# Arguments
- `A`: The matrix to build the smoother for
- `smoother_type`: Type tag selecting the smoother algorithm
- `ω`: Damping factor (used by Jacobi and l1-Jacobi smoothers, default: 2/3)
- `backend`: KernelAbstractions backend (default: CPU)
- `block_size`: Kernel launch block size (default: 64)
"""
function build_smoother(A::SparseMatrixCSC, smoother_type::SmootherType;
                        ω::Real=2.0/3.0, backend=DEFAULT_BACKEND, block_size::Int=64)
    A_csr = csr_from_csc(A)
    return build_smoother(A_csr, smoother_type, ω; backend=backend, block_size=block_size)
end

"""
    update_smoother!(smoother::AbstractSmoother, A::SparseMatrixCSC; backend, block_size)

Update the smoother for new matrix values (same sparsity pattern). This is
the public API for updating smoothers with `SparseMatrixCSC` matrices.
"""
function update_smoother!(smoother::AbstractSmoother, A::SparseMatrixCSC;
                          backend=DEFAULT_BACKEND, block_size::Int=64)
    A_csr = csr_from_csc(A)
    return update_smoother!(smoother, A_csr; backend=backend, block_size=block_size)
end

"""
    smooth!(x, A::SparseMatrixCSC, b, smoother; steps=1, backend, block_size)

Apply smoother iterations to solve `Ax = b` using a `SparseMatrixCSC` matrix.
This is the public API for applying smoothers with `SparseMatrixCSC` matrices.
"""
function smooth!(x::AbstractVector, A::SparseMatrixCSC, b::AbstractVector,
                 smoother::AbstractSmoother; steps::Int=1, backend=DEFAULT_BACKEND, block_size::Int=64)
    A_csr = csr_from_csc(A)
    return smooth!(x, A_csr, b, smoother; steps=steps, backend=backend, block_size=block_size)
end
