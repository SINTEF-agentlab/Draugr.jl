# Minimum number of rows in a level-scheduled ILU level before spawning threads.
# Below this threshold, the threading overhead from Threads.@threads outweighs
# the parallel benefit, so levels are processed with a plain serial loop.
const _ILU0_MIN_PARALLEL_ROWS = 64

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

# Return the scalar real floating-point type underlying Tv.
# For scalars (e.g. Float64, ComplexF64): real(eltype(Float64)) = Float64.
# For block entries (e.g. SMatrix{2,2,Float64,4}): eltype gives Float64, real(Float64) = Float64.
@inline _scalar_real_type(::Type{T}) where T = real(eltype(T))

# Return the block size (number of rows) of a single entry of type Tv.
# For scalars, block size is 1 (size(0.0, 1) == 1 in Julia).
# For SMatrix{B,B,...}, block size is B.
@inline _block_size(::Type{T}) where T = size(zero(T), 1)

"""
    build_jacobi_smoother(A, ω)

Build a weighted Jacobi smoother from matrix `A` with damping `ω`.
"""
function build_jacobi_smoother(A::CSRMatrix{Tv, Ti}, ω::Real;
                               x_eltype::Type{Tx}=Tv) where {Tv, Ti, Tx}
    n = size(A, 1)
    invdiag = _allocate_undef_vector(A, Tv, n)
    compute_inverse_diagonal!(invdiag, A)
    tmp = _allocate_vector(A, Tx, n)
    return JacobiSmoother(invdiag, tmp, ω)
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
        Tv = eltype(invdiag)
        Ts = _scalar_real_type(Tv)
        diag_val = zero(Tv)
        row_norm = zero(Ts)
        for nz in rp[i]:(rp[i+1]-1)
            row_norm += _entry_norm(nzval[nz])
            if colval[nz] == i
                diag_val = nzval[nz]
            end
        end
        # Safe inverse: avoid Inf/NaN for zero or near-zero diagonals
        abs_d = _entry_norm(diag_val)
        threshold = eps(Ts) * max(one(Ts), row_norm)
        invdiag[i] = abs_d > threshold ? inv(diag_val) : zero(Tv)
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
                 smoother::JacobiSmoother; steps::Int=1, reverse::Bool=false, backend=DEFAULT_BACKEND, block_size::Int=64)
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
                 smoother::ColoredGaussSeidelSmoother; steps::Int=1, reverse::Bool=false, backend=DEFAULT_BACKEND, block_size::Int=64)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    kernel! = gs_color_kernel!(backend, block_size)
    nc = smoother.num_colors
    for _ in 1:steps
        color_range = reverse ? (nc:-1:1) : (1:nc)
        for c in color_range
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
                 smoother::L1ColoredGaussSeidelSmoother; steps::Int=1, reverse::Bool=false, backend=DEFAULT_BACKEND, block_size::Int=64)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    kernel! = gs_color_kernel!(backend, block_size)
    nc = smoother.num_colors
    for _ in 1:steps
        color_range = reverse ? (nc:-1:1) : (1:nc)
        for c in color_range
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
    Ts = _scalar_real_type(Tv)
    @inbounds for i in 1:n
        d = zero(Tv)
        for nz in rp[i]:(rp[i+1]-1)
            if cv[nz] == i
                d = nzv[nz]
                break
            end
        end
        abs_d = _entry_norm(d)
        invdiag[i] = abs_d > eps(Ts) ? inv(d) : zero(Tv)
    end
    return invdiag
end

"""
    _serial_gs_sweep!(x, b, nzv, cv, rp, invdiag, n, steps; reverse=false)

Function barrier for the Gauss-Seidel sweep. Takes concrete array types
as arguments to ensure type stability in the inner loop.
When `reverse=true`, performs a backward sweep (rows n to 1) instead of
forward (rows 1 to n). Using forward for pre-smoothing and backward for
post-smoothing matches HYPRE's default l1-GS relaxation and produces a
more effective AMG preconditioner.
"""
function _serial_gs_sweep!(x, b, nzv, cv, rp, invdiag, n::Int, steps::Int; reverse::Bool=false)
    for _ in 1:steps
        if reverse
            @inbounds for i in n:-1:1
                r_i = b[i]
                for nz in rp[i]:(rp[i+1]-1)
                    j = cv[nz]
                    r_i -= nzv[nz] * x[j]
                end
                x[i] += invdiag[i] * r_i
            end
        else
            @inbounds for i in 1:n
                r_i = b[i]
                for nz in rp[i]:(rp[i+1]-1)
                    j = cv[nz]
                    r_i -= nzv[nz] * x[j]
                end
                x[i] += invdiag[i] * r_i
            end
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
                 smoother::SerialGaussSeidelSmoother; steps::Int=1, reverse::Bool=false, backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
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
                      rowptr(smoother.A_cpu), smoother.invdiag, n, steps; reverse=reverse)
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
Matches hypre's `ComputeL1Norms` option 4 for serial execution:

    l1_norm = |a_{i,i}| + 0.5 * Σ_{j ∈ A_offd} |a_{i,j}|

In serial (single-process) HYPRE the off-diagonal block A_offd is empty,
so the formula reduces to `l1_norm = |a_{i,i}|`.  This makes serial L1 GS
equivalent to standard serial GS — the L1 damping only adds extra weight
for inter-process entries that remain frozen during a parallel sweep.
"""
function _serial_l1_gs_compute_invdiag!(invdiag::Vector{Tv}, nzv, cv, rp, n::Int) where {Tv}
    Ts = _scalar_real_type(Tv)
    @inbounds for i in 1:n
        abs_diag = zero(Ts)
        for nz in rp[i]:(rp[i+1]-1)
            if cv[nz] == i
                abs_diag = _entry_norm(nzv[nz])
                break
            end
        end
        # For serial GS, all connections are updated sequentially, so
        # there are no "frozen" off-processor entries. This matches
        # hypre's option 4 for a single-processor run: l1_norm = |a_{i,i}|.
        l1_norm = abs_diag
        invdiag[i] = l1_norm > eps(Ts) ? one(Tv) / l1_norm : zero(Tv)
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
                 smoother::L1SerialGaussSeidelSmoother; steps::Int=1, reverse::Bool=false, backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti}
    n = size(A, 1)
    is_gpu = !(x isa Array)
    if is_gpu
        x_cpu = Array(x)
        b_cpu = Array(b)
    else
        x_cpu = x
        b_cpu = b
    end
    # Reuses the same forward sweep as standard serial GS; the L1 variant
    # differs only in how invdiag is computed (l1 row norms vs diagonal entries).
    _serial_gs_sweep!(x_cpu, b_cpu, nonzeros(smoother.A_cpu), colvals(smoother.A_cpu),
                      rowptr(smoother.A_cpu), smoother.invdiag, n, steps; reverse=reverse)
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
function build_spai0_smoother(A::CSRMatrix{Tv, Ti};
                              x_eltype::Type{Tx}=Tv) where {Tv, Ti, Tx}
    n = size(A, 1)
    m_diag = _allocate_undef_vector(A, Tv, n)
    _compute_spai0!(m_diag, A)
    tmp = _allocate_vector(A, Tx, n)
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
        Tv = eltype(m_diag)
        Ts = _scalar_real_type(Tv)
        diag_val = zero(Tv)
        row_norm_sq = zero(Ts)
        for nz in rp[i]:(rp[i+1]-1)
            v = nzval[nz]
            row_norm_sq += _frobenius_norm2(v)
            if colval[nz] == i
                diag_val = v
            end
        end
        m_diag[i] = row_norm_sq > zero(Ts) ? diag_val / row_norm_sq : zero(Tv)
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
                 smoother::SPAI0Smoother; steps::Int=1, reverse::Bool=false, backend=DEFAULT_BACKEND, block_size::Int=64)
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
function build_spai1_smoother(A::CSRMatrix{Tv, Ti};
                              x_eltype::Type{Tx}=Tv) where {Tv, Ti, Tx}
    n = size(A, 1)
    A_cpu = csr_to_cpu(A)
    nzval_m = Vector{Tv}(undef, nnz(A))
    _compute_spai1!(nzval_m, A_cpu)
    # Copy nzval to device if needed
    nzval_dev = A.nzval isa Array ? nzval_m : _to_device(A, nzval_m)
    tmp = _allocate_vector(A, Tx, n)
    return SPAI1Smoother{Tv, Ti, Tx, typeof(nzval_dev), typeof(tmp)}(nzval_dev, tmp)
end

"""
    _compute_spai1!(nzval_m, A)

Compute SPAI(1) values. For scalar matrices (`Tv <: Number`), solves a small
k×k least-squares system per row. For block matrices (e.g. `SMatrix` entries),
reformulates as a kB×kB scalar system and solves B right-hand sides at once,
where B is the block size.
"""
function _compute_spai1!(nzval_m::Vector{Tv}, A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    B = _block_size(Tv)
    if B == 1
        _compute_spai1_scalar!(nzval_m, A)
    else
        _compute_spai1_block!(nzval_m, A, B)
    end
    return nzval_m
end

"""
    _compute_spai1_scalar!(nzval_m, A)

SPAI(1) for scalar (non-block) entry types. For each row i, solves the small
least-squares problem:
  min_{m_i} ‖e_i - A^T m_i‖₂²
where m_i has support on the sparsity pattern of A[i,:].
"""
function _compute_spai1_scalar!(nzval_m::Vector{Tv}, A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    Ts = _scalar_real_type(Tv)
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
        # G[p,q] = A[:,J[p]]' * A[:,J[q]] = sum_r A[r,J[p]] * conj(A[r,J[q]])
        G = zeros(Tv, k, k)
        rhs = zeros(Tv, k)
        # Build a map: column index -> local index in J
        col_to_local = Dict{Ti, Int}()
        for (p, j) in enumerate(J)
            col_to_local[j] = p
        end
        # Iterate over all rows
        for r in 1:n
            rng_r = rp[r]:(rp[r+1]-1)
            local_entries = Tuple{Int, Tv}[]
            for nz in rng_r
                c = cv[nz]
                if haskey(col_to_local, c)
                    push!(local_entries, (col_to_local[c], nzv[nz]))
                end
            end
            isempty(local_entries) && continue
            for (p, vp) in local_entries
                for (q, vq) in local_entries
                    G[p, q] += vp * vq
                end
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
            G[p, p] += eps(Ts) * max(one(Ts), abs(G[p, p]))
        end
        m_local = G \ rhs
        # Store back
        for (p, nz) in enumerate(rng_i)
            nzval_m[nz] = m_local[p]
        end
    end
    return nzval_m
end

"""
    _compute_spai1_block!(nzval_m, A, B)

SPAI(1) for block entry types (e.g. `SMatrix{B,B,T}`). Reformulates the
least-squares problem as a scalar kB×kB system with B right-hand sides.

The block SPAI(1) minimises ‖I_B - A^T M_i‖_F² where M_i is a block row
of B×B matrices. This is equivalent to solving B independent kB-vector problems,
assembled here as a kB×B matrix solve.
"""
function _compute_spai1_block!(nzval_m::Vector{Tv}, A::CSRMatrix{Tv, Ti}, B::Int) where {Tv, Ti}
    T = eltype(Tv)   # scalar float type
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    @inbounds for i in 1:n
        rng_i = rp[i]:(rp[i+1]-1)
        k = length(rng_i)
        k == 0 && continue
        J = cv[rng_i]
        kB = k * B

        col_to_local = Dict{Ti, Int}()
        for (p, j) in enumerate(J)
            col_to_local[j] = p
        end

        # G_flat[kB×kB]: block Gram matrix (scalar)
        # G_flat[(p-1)*B+1:p*B, (q-1)*B+1:q*B] = Σ_r A[r,J[p]]' * A[r,J[q]]
        G_flat = zeros(T, kB, kB)
        # rhs_flat[kB×B]: rhs_flat[(p-1)*B+1:p*B, :] = A[i, J[p]] (a B×B block)
        rhs_flat = zeros(T, kB, B)

        # Accumulate Gram matrix over all rows
        for r in 1:n
            rng_r = rp[r]:(rp[r+1]-1)
            local_entries = Tuple{Int, Matrix{T}}[]
            for nz in rng_r
                c = cv[nz]
                if haskey(col_to_local, c)
                    push!(local_entries, (col_to_local[c], Matrix{T}(Array(nzv[nz]))))
                end
            end
            isempty(local_entries) && continue
            for (p, Mp) in local_entries
                for (q, Mq) in local_entries
                    # G[(p-1)*B+1:p*B, (q-1)*B+1:q*B] += Mp' * Mq
                    mul!(view(G_flat, (p-1)*B+1:p*B, (q-1)*B+1:q*B), Mp', Mq, one(T), one(T))
                end
            end
        end

        # RHS: for each local index p, rhs_flat[(p-1)*B+1:p*B, :] = A[i, J[p]]^T
        for (p, j) in enumerate(J)
            for nz in rng_i
                if cv[nz] == j
                    rhs_flat[(p-1)*B+1:p*B, :] = Matrix{T}(Array(nzv[nz]))'
                    break
                end
            end
        end

        # Add small regularization for stability (scalar diagonal)
        for α in 1:kB
            G_flat[α, α] += eps(T) * max(one(T), abs(G_flat[α, α]))
        end

        # Solve: G_flat * M_local = rhs_flat  →  M_local is kB × B
        M_local = G_flat \ rhs_flat

        # Store back: the p-th block is M_local[(p-1)*B+1:p*B, :] (a B×B matrix)
        for (p, nz) in enumerate(rng_i)
            nzval_m[nz] = Tv(M_local[(p-1)*B+1:p*B, :])
        end
    end
    return nzval_m
end

function update_smoother!(smoother::SPAI1Smoother{Tv}, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64) where Tv
    A_cpu = csr_to_cpu(A)
    nzval_cpu = Vector{Tv}(undef, nnz(A))
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
                 smoother::SPAI1Smoother; steps::Int=1, reverse::Bool=false, backend=DEFAULT_BACKEND, block_size::Int=64)
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

function build_smoother(A::CSRMatrix{Tv, Ti}, ::JacobiSmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64,
                        x_eltype::Type=Tv) where {Tv, Ti}
    return build_jacobi_smoother(A, ω; x_eltype=x_eltype)
end

function build_smoother(A::CSRMatrix, ::ColoredGaussSeidelType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64)
    return build_colored_gs_smoother(A)
end

function build_smoother(A::CSRMatrix{Tv, Ti}, ::SPAI0SmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64,
                        x_eltype::Type=Tv) where {Tv, Ti}
    return build_spai0_smoother(A; x_eltype=x_eltype)
end

function build_smoother(A::CSRMatrix{Tv, Ti}, ::SPAI1SmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64,
                        x_eltype::Type=Tv) where {Tv, Ti}
    return build_spai1_smoother(A; x_eltype=x_eltype)
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
function build_l1jacobi_smoother(A::CSRMatrix{Tv, Ti}, ω::Real;
                                 x_eltype::Type{Tx}=Tv) where {Tv, Ti, Tx}
    n = size(A, 1)
    invdiag = _allocate_undef_vector(A, Tv, n)
    _compute_l1_jacobi_invdiag!(invdiag, A)
    tmp = _allocate_vector(A, Tx, n)
    return L1JacobiSmoother(invdiag, tmp, ω)
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
        Tv = eltype(invdiag)
        Ts = _scalar_real_type(Tv)
        # Compute L1 norms matching hypre's option 4 (Remark 6.2 in
        # "Multigrid Smoothers for Ultra-Parallel Computing", Baker et al. 2011).
        # For colored GS the non-sequentially-updated connections are the
        # off-diagonal entries (all neighbors have a different color), so:
        #   l1_norm = |a_{i,i}| + 0.5 * Σ_{j≠i} |a_{i,j}|
        # with truncation: if l1_norm ≤ 4/3 * |a_{i,i}|, use |a_{i,i}|.
        abs_diag = zero(Ts)
        offdiag_sum = zero(Ts)
        for nz in rp[i]:(rp[i+1]-1)
            if colval[nz] == i
                abs_diag = _entry_norm(nzval[nz])
            else
                offdiag_sum += _entry_norm(nzval[nz])
            end
        end
        l1_norm = abs_diag + Ts(0.5) * offdiag_sum
        # Truncation: when off-diagonal is small, fall back to diagonal
        four_thirds = Ts(4) / Ts(3)
        if l1_norm <= four_thirds * abs_diag
            l1_norm = abs_diag
        end
        # For block matrices, store (1/l1_norm)*I so that invdiag[i]*r gives (1/l1_norm)*r.
        # For scalars, one(Tv)/l1_norm == 1/l1_norm (a scalar).
        invdiag[i] = l1_norm > eps(Ts) ? one(Tv) / l1_norm : zero(Tv)
    end
end

"""
    _compute_l1_jacobi_invdiag!(invdiag, A)

Compute inverse l1 row norms for L1-Jacobi smoother using the full row sum
(matching hypre's option 1): l1_norm = Σ_j |a_{i,j}|.
This is the correct formula for Jacobi where ALL off-diagonal entries use
frozen (previous-iteration) values.
"""
function _compute_l1_jacobi_invdiag!(invdiag::AbstractVector{Tv},
                                      A::CSRMatrix{Tv, Ti};
                                      backend=_get_backend(nonzeros(A)), block_size::Int=64) where {Tv, Ti}
    n = size(A, 1)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    kernel! = l1_jacobi_invdiag_kernel!(backend, block_size)
    kernel!(invdiag, nzv, rp; ndrange=n)
    _synchronize(backend)
    return invdiag
end

@kernel function l1_jacobi_invdiag_kernel!(invdiag, @Const(nzval), @Const(rp))
    i = @index(Global)
    @inbounds begin
        Tv = eltype(invdiag)
        Ts = _scalar_real_type(Tv)
        l1_norm = zero(Ts)
        for nz in rp[i]:(rp[i+1]-1)
            l1_norm += _entry_norm(nzval[nz])
        end
        invdiag[i] = l1_norm > eps(Ts) ? one(Tv) / l1_norm : zero(Tv)
    end
end

function update_smoother!(smoother::L1JacobiSmoother, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64)
    _compute_l1_jacobi_invdiag!(smoother.invdiag, A; backend=backend, block_size=block_size)
    return smoother
end

function smooth!(x::AbstractVector, A::CSRMatrix, b::AbstractVector,
                 smoother::L1JacobiSmoother; steps::Int=1, reverse::Bool=false, backend=DEFAULT_BACKEND, block_size::Int=64)
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

function build_smoother(A::CSRMatrix{Tv, Ti}, ::L1JacobiSmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64,
                        x_eltype::Type=Tv) where {Tv, Ti}
    return build_l1jacobi_smoother(A, ω; x_eltype=x_eltype)
end

# ══════════════════════════════════════════════════════════════════════════════
# Chebyshev Polynomial Smoother
# ══════════════════════════════════════════════════════════════════════════════

"""
    _estimate_spectral_radius(A, invdiag; niter=10)

Estimate the spectral radius of D⁻¹A using power iteration for scalar matrices,
or a Gershgorin row-sum upper bound for block matrices.

For scalar matrices (`Tv <: Number`): uses power iteration with a random scalar
vector. The multiplication order `invdiag[i] * w[i]` is used explicitly to handle
block types where multiplication is non-commutative (for scalars this is equivalent
to the previous `w[i] * invdiag[i]`).

For block matrices: uses max_i (‖D_i^{-1}‖_F · Σ_j ‖A_{i,j}‖_F) as a
Gershgorin-like upper bound on the spectral radius.
"""
function _estimate_spectral_radius(A::CSRMatrix{Tv, Ti},
                                   invdiag::Vector{Tv}; niter::Int=10) where {Tv, Ti}
    Ts = _scalar_real_type(Tv)
    if Tv <: Number
        # Scalar path: standard power iteration
        n = size(A, 1)
        v = randn(Tv, n)
        v ./= norm(v)
        w = similar(v)
        λ = one(Ts)
        for _ in 1:niter
            mul!(w, A, v)
            @inbounds for i in 1:n
                # Use left-multiplication so the order is correct for block types.
                # For scalars this is identical to w[i] *= invdiag[i].
                w[i] = invdiag[i] * w[i]
            end
            λ = norm(w)
            if λ > eps(Ts)
                v .= w ./ λ
            end
        end
        return real(λ)
    else
        # Block path: Gershgorin row-sum upper bound
        n = size(A, 1)
        nzv = nonzeros(A)
        cv = colvals(A)
        rp = rowptr(A)
        ρ = zero(Ts)
        @inbounds for i in 1:n
            d_inv_norm = _entry_norm(invdiag[i])
            row_sum = zero(Ts)
            for nz in rp[i]:(rp[i+1]-1)
                row_sum += _entry_norm(nzv[nz])
            end
            ρ = max(ρ, d_inv_norm * row_sum)
        end
        return ρ
    end
end

"""
    build_chebyshev_smoother(A; degree=3)

Build a Chebyshev polynomial smoother. Estimates eigenvalues of D⁻¹A and
constructs a degree-`degree` Chebyshev iteration.
"""
function build_chebyshev_smoother(A::CSRMatrix{Tv, Ti};
                                  degree::Int=3,
                                  x_eltype::Type{Tx}=Tv) where {Tv, Ti, Tx}
    n = size(A, 1)
    invdiag = _allocate_undef_vector(A, Tv, n)
    compute_inverse_diagonal!(invdiag, A)
    # Spectral radius estimation uses scalar indexing and mul!, which require CPU arrays
    invdiag_cpu = invdiag isa Array ? invdiag : Array(invdiag)
    A_cpu = csr_to_cpu(A)
    ρ = _estimate_spectral_radius(A_cpu, invdiag_cpu)
    # Standard Chebyshev bounds for SPD: [ρ/30, 1.1*ρ]
    Tλ = _scalar_real_type(Tv)
    λ_max = Tλ(1.1) * ρ
    λ_min = λ_max / Tλ(30.0)
    tmp1 = _allocate_vector(A, Tx, n)
    tmp2 = _allocate_vector(A, Tx, n)
    return ChebyshevSmoother(invdiag, tmp1, tmp2, λ_min, λ_max, degree)
end

function update_smoother!(smoother::ChebyshevSmoother, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64)
    compute_inverse_diagonal!(smoother.invdiag, A; backend=backend, block_size=block_size)
    # Spectral radius estimation requires CPU arrays
    invdiag_cpu = smoother.invdiag isa Array ? smoother.invdiag : Array(smoother.invdiag)
    A_cpu = csr_to_cpu(A)
    ρ = _estimate_spectral_radius(A_cpu, invdiag_cpu)
    Tλ = typeof(smoother.λ_max)
    smoother.λ_max = Tλ(1.1) * ρ
    smoother.λ_min = smoother.λ_max / Tλ(30.0)
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
                 smoother::ChebyshevSmoother; steps::Int=1, reverse::Bool=false, backend=DEFAULT_BACKEND, block_size::Int=64)
    n = size(A, 1)
    # Tλ is always a real scalar type (Float64, etc.), independent of the block structure.
    Tλ = typeof(smoother.λ_max)
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

        init_kernel!(d, x, smoother.invdiag, r, one(Tλ) / θ; ndrange=n)
        _synchronize(backend)

        # Iterations 1..degree-1 using three-term recurrence
        σ_old = θ / δ
        for k in 1:(smoother.degree - 1)
            rkernel!(r, b, x, nzv, cv, rp; ndrange=n)
            _synchronize(backend)

            σ_new = one(Tλ) / (Tλ(2) * θ / δ - σ_old)
            scale_r = Tλ(2) * σ_new / δ
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

function build_smoother(A::CSRMatrix{Tv, Ti}, ::ChebyshevSmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64,
                        x_eltype::Type=Tv) where {Tv, Ti}
    return build_chebyshev_smoother(A; x_eltype=x_eltype)
end

# ══════════════════════════════════════════════════════════════════════════════
# ILU(0) Smoother — shared helpers
# ══════════════════════════════════════════════════════════════════════════════

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
    Ts = _scalar_real_type(Tv)

    # Copy A values into U
    copyto!(U_nzval, nzv)
    fill!(L_nzval, zero(Tv))

    # Maximum factor growth for ILU entries
    const_max_ilu_factor = Ts(1e8)

    # Precompute row norms to avoid redundant recomputation in inner loop
    row_norms = Vector{Ts}(undef, n)
    @inbounds for i in 1:n
        s = zero(Ts)
        for nz in rp[i]:(rp[i+ti_one]-ti_one)
            s += _entry_norm(nzv[nz])
        end
        row_norms[i] = s
    end

    @inbounds for i in 1:n
        # Process row i: for each k < i in row i's lower triangle
        for nz in rp[i]:(diag_idx[i]-ti_one)
            k = cv[nz]
            # L[i,k] = U[i,k] * U[k,k]^{-1}  (right division: U[i,k] / U[k,k])
            # This is the correct formula for row-oriented ILU;
            # for block matrices, `/` computes the right factor: A * inv(B).
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
            U_nzval[diag_idx[i]] = safe_thresh * one(Tv)
        end
    end
    return nothing
end

"""Compute diagonal indices for ILU(0) factorization."""
function _ilu0_diag_indices(A_cpu::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    n = size(A_cpu, 1)
    cv = colvals(A_cpu)
    rp = rowptr(A_cpu)
    ti_one = one(Ti)
    diag_idx = Vector{Ti}(undef, n)
    @inbounds for i in 1:n
        for nz in rp[i]:(rp[i+ti_one]-ti_one)
            if cv[nz] == i
                diag_idx[i] = Ti(nz)
                break
            end
        end
    end
    return diag_idx
end

# ══════════════════════════════════════════════════════════════════════════════
# Serial ILU(0) Smoother
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_serial_ilu0_smoother(A)

Build a serial ILU(0) smoother. Computes an incomplete LU factorization with
the same sparsity pattern as A, using plain sequential forward/backward
substitution (no graph coloring). All factorization data is stored on CPU;
GPU matrices are automatically converted.
"""
function build_serial_ilu0_smoother(A::CSRMatrix{Tv, Ti};
                                    x_eltype::Type{Tx}=Tv) where {Tv, Ti, Tx}
    A_cpu = csr_to_cpu(A)
    n = size(A_cpu, 1)
    nzv = nonzeros(A_cpu)

    diag_idx = _ilu0_diag_indices(A_cpu)

    # ILU(0) factorization (on CPU)
    L_nzval = zeros(Tv, nnz(A_cpu))
    U_nzval = copy(nzv)
    _ilu0_factorize!(L_nzval, U_nzval, diag_idx, A_cpu)

    tmp = zeros(Tx, n)
    return SerialILU0Smoother{Tv, Ti, Tx}(L_nzval, U_nzval, diag_idx, tmp, A_cpu)
end

function update_smoother!(smoother::SerialILU0Smoother, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64)
    A_cpu = csr_to_cpu(A)
    copyto!(smoother.A_cpu.nzval, A_cpu.nzval)
    _ilu0_factorize!(smoother.L_nzval, smoother.U_nzval, smoother.diag_idx, smoother.A_cpu)
    return smoother
end

"""
    smooth!(x, A, b, smoother::SerialILU0Smoother; steps=1)

Apply serial ILU(0) smoothing: x += (LU)⁻¹ (b - Ax).
Uses plain sequential forward/backward substitution on CPU.
For GPU arrays, copies data to CPU, applies ILU, and copies back.
"""
function smooth!(x::AbstractVector, A::CSRMatrix{Tv, Ti}, b::AbstractVector,
                 smoother::SerialILU0Smoother{Tv, Ti, Tx}; steps::Int=1, reverse::Bool=false, backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti, Tx}
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
            Ax_i = zero(Tx)
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
            tmp[i] = _entry_norm(u_ii) > eps(_scalar_real_type(Tv)) ? u_ii \ tmp[i] : zero(Tx)
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

function build_smoother(A::CSRMatrix{Tv, Ti}, ::SerialILU0SmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64,
                        x_eltype::Type=Tv) where {Tv, Ti}
    return build_serial_ilu0_smoother(A; x_eltype=x_eltype)
end

# ══════════════════════════════════════════════════════════════════════════════
# Parallel ILU(0) Smoother (with coloring)
# ══════════════════════════════════════════════════════════════════════════════

"""
    _compute_ilu_levels(A_cpu, diag_idx)

Compute level scheduling for parallel triangular solves. Returns level
assignments and ordering for both forward and backward substitution.

For forward substitution (L solve), row i depends on rows j < i where
L[i,j] ≠ 0. The level of row i is 1 + max(level(j)) over all such j.

For backward substitution (U solve), row i depends on rows j > i where
U[i,j] ≠ 0. The level of row i is 1 + max(level(j)) over all such j.

Returns (fwd_order, fwd_offsets, num_fwd_levels, bwd_order, bwd_offsets, num_bwd_levels).
"""
function _compute_ilu_levels(A_cpu::CSRMatrix{Tv, Ti}, diag_idx::Vector{Ti}) where {Tv, Ti}
    n = size(A_cpu, 1)
    cv = colvals(A_cpu)
    rp = rowptr(A_cpu)
    ti_one = one(Ti)

    # Forward solve levels: row i depends on j < i in L's sparsity
    fwd_level = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        max_dep = 0
        for nz in rp[i]:(diag_idx[i]-ti_one)
            j = cv[nz]
            max_dep = max(max_dep, fwd_level[j])
        end
        fwd_level[i] = max_dep + 1
    end
    num_fwd_levels = n > 0 ? maximum(fwd_level) : 0

    # Backward solve levels: row i depends on j > i in U's sparsity
    bwd_level = Vector{Int}(undef, n)
    @inbounds for i in n:-1:1
        max_dep = 0
        for nz in (diag_idx[i]+ti_one):(rp[i+ti_one]-ti_one)
            j = cv[nz]
            max_dep = max(max_dep, bwd_level[j])
        end
        bwd_level[i] = max_dep + 1
    end
    num_bwd_levels = n > 0 ? maximum(bwd_level) : 0

    # Build forward level ordering
    fwd_counts = zeros(Int, num_fwd_levels)
    @inbounds for i in 1:n
        fwd_counts[fwd_level[i]] += 1
    end
    fwd_offsets = Vector{Int}(undef, num_fwd_levels + 1)
    fwd_offsets[1] = 1
    for c in 1:num_fwd_levels
        fwd_offsets[c+1] = fwd_offsets[c] + fwd_counts[c]
    end
    fwd_order = Vector{Ti}(undef, n)
    pos = copy(fwd_offsets[1:num_fwd_levels])
    @inbounds for i in 1:n
        c = fwd_level[i]
        fwd_order[pos[c]] = Ti(i)
        pos[c] += 1
    end

    # Build backward level ordering
    bwd_counts = zeros(Int, num_bwd_levels)
    @inbounds for i in 1:n
        bwd_counts[bwd_level[i]] += 1
    end
    bwd_offsets = Vector{Int}(undef, num_bwd_levels + 1)
    bwd_offsets[1] = 1
    for c in 1:num_bwd_levels
        bwd_offsets[c+1] = bwd_offsets[c] + bwd_counts[c]
    end
    bwd_order = Vector{Ti}(undef, n)
    pos = copy(bwd_offsets[1:num_bwd_levels])
    @inbounds for i in 1:n
        c = bwd_level[i]
        bwd_order[pos[c]] = Ti(i)
        pos[c] += 1
    end

    return fwd_order, fwd_offsets, num_fwd_levels, bwd_order, bwd_offsets, num_bwd_levels
end

"""
    build_ilu0_smoother(A)

Build a parallel ILU(0) smoother. Computes an incomplete LU factorization with
the same sparsity pattern as A, using level scheduling for parallel forward/backward
substitution. All factorization data is stored on CPU; GPU matrices are
automatically converted.
"""
function build_ilu0_smoother(A::CSRMatrix{Tv, Ti};
                             x_eltype::Type{Tx}=Tv) where {Tv, Ti, Tx}
    A_cpu = csr_to_cpu(A)
    n = size(A_cpu, 1)
    nzv = nonzeros(A_cpu)

    diag_idx = _ilu0_diag_indices(A_cpu)

    # Compute level scheduling for parallel triangular solves
    fwd_order, fwd_offsets, num_fwd_levels, bwd_order, bwd_offsets, num_bwd_levels =
        _compute_ilu_levels(A_cpu, diag_idx)

    # ILU(0) factorization (on CPU)
    L_nzval = zeros(Tv, nnz(A_cpu))
    U_nzval = copy(nzv)
    _ilu0_factorize!(L_nzval, U_nzval, diag_idx, A_cpu)

    tmp = zeros(Tx, n)
    # Concatenate forward and backward level offsets for compact storage
    combined_offsets = vcat(fwd_offsets, bwd_offsets)
    return ILU0Smoother{Tv, Ti, Tx}(L_nzval, U_nzval, diag_idx, bwd_order,
                                     combined_offsets, fwd_order, num_fwd_levels, tmp, A_cpu)
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

Apply parallel ILU(0) smoothing: x += (LU)⁻¹ (b - Ax).
Uses level scheduling to process independent rows in parallel during
the forward/backward substitution phases.
For GPU arrays, copies data to CPU, applies ILU, and copies back.
"""
function smooth!(x::AbstractVector, A::CSRMatrix{Tv, Ti}, b::AbstractVector,
                 smoother::ILU0Smoother{Tv, Ti, Tx}; steps::Int=1, reverse::Bool=false, backend=DEFAULT_BACKEND, block_size::Int=64) where {Tv, Ti, Tx}
    n = size(A, 1)
    ti_one = one(Ti)

    # Use CPU arrays for ILU solve
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

    fwd_order = smoother.fwd_order
    num_fwd_levels = smoother.num_fwd_levels
    # Forward offsets are in level_offsets[1 : num_fwd_levels+1]
    combined_offsets = smoother.level_offsets
    bwd_order = smoother.bwd_order
    # Backward offsets start after the forward offsets (at index num_fwd_levels + 2)
    bwd_offset_start = num_fwd_levels + 2
    num_bwd_levels = length(combined_offsets) - bwd_offset_start

    L_nzval = smoother.L_nzval
    U_nzval = smoother.U_nzval
    diag_idx = smoother.diag_idx

    for _ in 1:steps
        # Compute residual: tmp = b - A*x (on CPU)
        @inbounds for i in 1:n
            Ax_i = zero(Tx)
            for nz in rp[i]:(rp[i+ti_one]-ti_one)
                j = cv[nz]
                Ax_i += nzv[nz] * x_cpu[j]
            end
            tmp[i] = b_cpu[i] - Ax_i
        end

        # Forward substitution: L * z = tmp, using level scheduling.
        # Within each level, rows are independent and can be processed in parallel.
        # Only spawn threads when a level is large enough to offset the overhead.
        @inbounds for lev in 1:num_fwd_levels
            lev_start = combined_offsets[lev]
            lev_end = combined_offsets[lev+1] - 1
            if lev_end - lev_start + 1 >= _ILU0_MIN_PARALLEL_ROWS
                Threads.@threads for idx in lev_start:lev_end
                    i = fwd_order[idx]
                    for nz in rp[i]:(diag_idx[i]-ti_one)
                        j = cv[nz]
                        tmp[i] -= L_nzval[nz] * tmp[j]
                    end
                end
            else
                for idx in lev_start:lev_end
                    i = fwd_order[idx]
                    for nz in rp[i]:(diag_idx[i]-ti_one)
                        j = cv[nz]
                        tmp[i] -= L_nzval[nz] * tmp[j]
                    end
                end
            end
        end

        # Backward substitution: U * dx = z, using level scheduling.
        # Within each level, rows are independent and can be processed in parallel.
        # Only spawn threads when a level is large enough to offset the overhead.
        @inbounds for lev in 1:num_bwd_levels
            lev_start = combined_offsets[bwd_offset_start + lev - 1]
            lev_end = combined_offsets[bwd_offset_start + lev] - 1
            if lev_end - lev_start + 1 >= _ILU0_MIN_PARALLEL_ROWS
                Threads.@threads for idx in lev_start:lev_end
                    i = bwd_order[idx]
                    for nz in (diag_idx[i]+ti_one):(rp[i+ti_one]-ti_one)
                        j = cv[nz]
                        tmp[i] -= U_nzval[nz] * tmp[j]
                    end
                    u_ii = U_nzval[diag_idx[i]]
                    tmp[i] = _entry_norm(u_ii) > eps(_scalar_real_type(Tv)) ? u_ii \ tmp[i] : zero(Tx)
                end
            else
                for idx in lev_start:lev_end
                    i = bwd_order[idx]
                    for nz in (diag_idx[i]+ti_one):(rp[i+ti_one]-ti_one)
                        j = cv[nz]
                        tmp[i] -= U_nzval[nz] * tmp[j]
                    end
                    u_ii = U_nzval[diag_idx[i]]
                    tmp[i] = _entry_norm(u_ii) > eps(_scalar_real_type(Tv)) ? u_ii \ tmp[i] : zero(Tx)
                end
            end
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

function build_smoother(A::CSRMatrix{Tv, Ti}, ::ILU0SmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64,
                        x_eltype::Type=Tv) where {Tv, Ti}
    return build_ilu0_smoother(A; x_eltype=x_eltype)
end

# ══════════════════════════════════════════════════════════════════════════════
# GPU ILU(0) Smoother (level-scheduled KA kernels)
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_gpu_ilu0_smoother(A; x_eltype=Tv)

Build a GPU-native ILU(0) smoother. The factorization is computed on CPU,
then L/U values and level-scheduling data are copied to the same device as `A`.
Triangular solves run entirely on-device using KernelAbstractions kernels.
"""
function build_gpu_ilu0_smoother(A::CSRMatrix{Tv, Ti};
                                 x_eltype::Type{Tx}=Tv) where {Tv, Ti, Tx}
    A_cpu = csr_to_cpu(A)
    n = size(A_cpu, 1)

    diag_idx = _ilu0_diag_indices(A_cpu)

    # Compute level scheduling for parallel triangular solves (reuse from ILU0)
    fwd_order, fwd_offsets, num_fwd_levels, bwd_order, bwd_offsets, num_bwd_levels =
        _compute_ilu_levels(A_cpu, diag_idx)

    # ILU(0) factorization (on CPU)
    L_nzval = zeros(Tv, nnz(A_cpu))
    U_nzval = copy(nonzeros(A_cpu))
    _ilu0_factorize!(L_nzval, U_nzval, diag_idx, A_cpu)

    # Copy factorization data to device
    L_nzval_dev = A.nzval isa Array ? L_nzval : _to_device(A, L_nzval)
    U_nzval_dev = A.nzval isa Array ? U_nzval : _to_device(A, U_nzval)
    diag_idx_dev = A.nzval isa Array ? diag_idx : _to_device(A, diag_idx)
    fwd_order_dev = A.nzval isa Array ? fwd_order : _to_device(A, fwd_order)
    bwd_order_dev = A.nzval isa Array ? bwd_order : _to_device(A, bwd_order)
    tmp = _allocate_vector(A, Tx, n)

    combined_offsets = vcat(fwd_offsets, bwd_offsets)
    return GPUILU0Smoother(L_nzval_dev, U_nzval_dev, diag_idx_dev,
                           fwd_order_dev, bwd_order_dev,
                           combined_offsets, num_fwd_levels, tmp, A_cpu)
end

function update_smoother!(smoother::GPUILU0Smoother{Tv, Ti}, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64) where {Tv, Ti}
    A_cpu = csr_to_cpu(A)
    copyto!(smoother.A_cpu.nzval, A_cpu.nzval)
    # Re-factorize on CPU
    L_cpu = Vector{Tv}(undef, length(smoother.A_cpu.nzval))
    U_cpu = copy(nonzeros(smoother.A_cpu))
    fill!(L_cpu, zero(Tv))
    diag_idx_cpu = Array(smoother.diag_idx)
    _ilu0_factorize!(L_cpu, U_cpu, diag_idx_cpu, smoother.A_cpu)
    # Copy updated values to device
    copyto!(smoother.L_nzval, L_cpu)
    copyto!(smoother.U_nzval, U_cpu)
    return smoother
end

# ── KA kernels for GPU ILU(0) triangular solves ──────────────────────────────

@kernel function _gpu_ilu0_residual_kernel!(tmp, @Const(b), @Const(x),
                                            @Const(nzval), @Const(colval), @Const(rp))
    i = @index(Global)
    @inbounds begin
        Ax_i = zero(eltype(tmp))
        for nz in rp[i]:(rp[i+1]-one(eltype(rp)))
            j = colval[nz]
            Ax_i += nzval[nz] * x[j]
        end
        tmp[i] = b[i] - Ax_i
    end
end

@kernel function _gpu_ilu0_fwd_kernel!(tmp, @Const(L_nzval), @Const(colval), @Const(rp),
                                       @Const(diag_idx), @Const(fwd_order), level_offset)
    idx = @index(Global)
    @inbounds begin
        i = fwd_order[level_offset + idx]
        ti_one = one(eltype(rp))
        for nz in rp[i]:(diag_idx[i]-ti_one)
            j = colval[nz]
            tmp[i] -= L_nzval[nz] * tmp[j]
        end
    end
end

@kernel function _gpu_ilu0_bwd_kernel!(tmp, @Const(U_nzval), @Const(colval), @Const(rp),
                                       @Const(diag_idx), @Const(bwd_order), level_offset)
    idx = @index(Global)
    @inbounds begin
        Tv = eltype(U_nzval)
        Tx = eltype(tmp)
        Ts = _scalar_real_type(Tv)
        i = bwd_order[level_offset + idx]
        ti_one = one(eltype(rp))
        for nz in (diag_idx[i]+ti_one):(rp[i+ti_one]-ti_one)
            j = colval[nz]
            tmp[i] -= U_nzval[nz] * tmp[j]
        end
        u_ii = U_nzval[diag_idx[i]]
        tmp[i] = _entry_norm(u_ii) > eps(Ts) ? u_ii \ tmp[i] : zero(Tx)
    end
end

@kernel function _gpu_ilu0_update_kernel!(x, @Const(tmp))
    i = @index(Global)
    @inbounds begin
        v = tmp[i]
        if _is_finite_entry(v)
            x[i] += v
        end
    end
end

"""
    smooth!(x, A, b, smoother::GPUILU0Smoother; steps=1)

Apply GPU-native ILU(0) smoothing: x += (LU)⁻¹ (b - Ax).
Uses KA kernels with level scheduling for the triangular solves.
All computation stays on-device — no CPU↔GPU copies per step.
"""
function smooth!(x::AbstractVector, A::CSRMatrix{Tv, Ti}, b::AbstractVector,
                 smoother::GPUILU0Smoother{Tv, Ti, Tx}; steps::Int=1, reverse::Bool=false,
                 backend=_get_backend(nonzeros(A)), block_size::Int=A.block_size) where {Tv, Ti, Tx}
    n = size(A, 1)
    n == 0 && return x
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    tmp = smoother.tmp

    fwd_order = smoother.fwd_order
    bwd_order = smoother.bwd_order
    num_fwd_levels = smoother.num_fwd_levels
    combined_offsets = smoother.level_offsets
    bwd_offset_start = num_fwd_levels + 2
    num_bwd_levels = length(combined_offsets) - bwd_offset_start

    L_nzval = smoother.L_nzval
    U_nzval = smoother.U_nzval
    diag_idx = smoother.diag_idx

    res_kernel! = _gpu_ilu0_residual_kernel!(backend, block_size)
    fwd_kernel! = _gpu_ilu0_fwd_kernel!(backend, block_size)
    bwd_kernel! = _gpu_ilu0_bwd_kernel!(backend, block_size)
    upd_kernel! = _gpu_ilu0_update_kernel!(backend, block_size)

    for _ in 1:steps
        # Compute residual: tmp = b - A*x
        res_kernel!(tmp, b, x, nzv, cv, rp; ndrange=n)
        _synchronize(backend)

        # Forward substitution: L * z = tmp, level by level
        for lev in 1:num_fwd_levels
            lev_start = combined_offsets[lev]
            lev_end = combined_offsets[lev+1] - 1
            count = lev_end - lev_start + 1
            count == 0 && continue
            fwd_kernel!(tmp, L_nzval, cv, rp, diag_idx, fwd_order, lev_start - 1; ndrange=count)
            _synchronize(backend)
        end

        # Backward substitution: U * dx = z, level by level
        for lev in 1:num_bwd_levels
            lev_start = combined_offsets[bwd_offset_start + lev - 1]
            lev_end = combined_offsets[bwd_offset_start + lev] - 1
            count = lev_end - lev_start + 1
            count == 0 && continue
            bwd_kernel!(tmp, U_nzval, cv, rp, diag_idx, bwd_order, lev_start - 1; ndrange=count)
            _synchronize(backend)
        end

        # Update: x += dx
        upd_kernel!(x, tmp; ndrange=n)
        _synchronize(backend)
    end
    return x
end

function build_smoother(A::CSRMatrix{Tv, Ti}, ::GPUILU0SmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64,
                        x_eltype::Type=Tv) where {Tv, Ti}
    return build_gpu_ilu0_smoother(A; x_eltype=x_eltype)
end

# ══════════════════════════════════════════════════════════════════════════════
# DILU (Diagonal ILU) Smoother
# ══════════════════════════════════════════════════════════════════════════════

"""
    _dilu_factorize!(inv_diag, diag_idx, A)

Compute the DILU modified diagonal and its inverse.
DILU approximates A ≈ (D+L)*D⁻¹*(D+U) where D is a modified diagonal and
L, U are the strict lower/upper parts of A with their original values.

The modified diagonal is computed row by row:
  d_i = a_{ii} - Σ_{j<i, (i,j)∈nz} a_{ij} * d_j⁻¹ * a_{ji}
"""
function _dilu_factorize!(inv_diag::Vector{Tv}, diag_idx::Vector{Ti},
                          A::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    n = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)
    rp = rowptr(A)
    ti_one = one(Ti)
    Ts = _scalar_real_type(Tv)

    # Precompute row norms for safeguards
    row_norms = Vector{Ts}(undef, n)
    @inbounds for i in 1:n
        s = zero(Ts)
        for nz in rp[i]:(rp[i+ti_one]-ti_one)
            s += _entry_norm(nzv[nz])
        end
        row_norms[i] = s
    end

    # d stores the modified diagonal values (forward pass, sequential)
    d = Vector{Tv}(undef, n)
    @inbounds for i in 1:n
        d_i = nzv[diag_idx[i]]  # start with a_{ii}
        # Subtract contributions from lower triangle: a_{ij} * d_j^{-1} * a_{ji}
        for nz in rp[i]:(diag_idx[i]-ti_one)
            j = cv[nz]  # j < i
            a_ij = nzv[nz]
            # Find a_{ji} in row j (the transpose entry)
            nz_ji = _find_nz_in_row(cv, rp[j], rp[j+ti_one]-ti_one, Ti(i))
            if nz_ji > zero(Ti)
                a_ji = nzv[nz_ji]
                d_i -= a_ij * inv_diag[j] * a_ji
            end
        end
        # Safeguard against zero/near-zero diagonal
        safe_thresh = _safe_threshold(Tv, row_norms[i])
        if _entry_norm(d_i) < safe_thresh
            d_i = safe_thresh * one(Tv)
        end
        d[i] = d_i
        inv_diag[i] = inv(d_i)
    end
    return nothing
end

"""
    build_dilu_smoother(A; x_eltype=Tv)

Build a DILU smoother. Computes a modified diagonal on CPU, then copies the
inverse diagonal and level-scheduling data to the same device as `A`.
Triangular solves run on-device using KernelAbstractions kernels.
"""
function build_dilu_smoother(A::CSRMatrix{Tv, Ti};
                             x_eltype::Type{Tx}=Tv) where {Tv, Ti, Tx}
    A_cpu = csr_to_cpu(A)
    n = size(A_cpu, 1)

    diag_idx = _ilu0_diag_indices(A_cpu)

    # Compute level scheduling (reuse from ILU0 — same dependency structure)
    fwd_order, fwd_offsets, num_fwd_levels, bwd_order, bwd_offsets, num_bwd_levels =
        _compute_ilu_levels(A_cpu, diag_idx)

    # DILU factorization (compute modified diagonal on CPU)
    inv_diag = Vector{Tv}(undef, n)
    _dilu_factorize!(inv_diag, diag_idx, A_cpu)

    # Copy data to device
    inv_diag_dev = A.nzval isa Array ? inv_diag : _to_device(A, inv_diag)
    diag_idx_dev = A.nzval isa Array ? diag_idx : _to_device(A, diag_idx)
    fwd_order_dev = A.nzval isa Array ? fwd_order : _to_device(A, fwd_order)
    bwd_order_dev = A.nzval isa Array ? bwd_order : _to_device(A, bwd_order)
    tmp = _allocate_vector(A, Tx, n)

    combined_offsets = vcat(fwd_offsets, bwd_offsets)
    return DILUSmoother(inv_diag_dev, diag_idx_dev,
                        fwd_order_dev, bwd_order_dev,
                        combined_offsets, num_fwd_levels, tmp, A_cpu)
end

function update_smoother!(smoother::DILUSmoother{Tv, Ti}, A::CSRMatrix;
                          backend=_get_backend(nonzeros(A)), block_size::Int=64) where {Tv, Ti}
    A_cpu = csr_to_cpu(A)
    copyto!(smoother.A_cpu.nzval, A_cpu.nzval)
    # Re-factorize on CPU
    n = size(smoother.A_cpu, 1)
    inv_diag_cpu = Vector{Tv}(undef, n)
    diag_idx_cpu = Array(smoother.diag_idx)
    _dilu_factorize!(inv_diag_cpu, diag_idx_cpu, smoother.A_cpu)
    # Copy updated diagonal to device
    copyto!(smoother.inv_diag, inv_diag_cpu)
    return smoother
end

# ── KA kernels for DILU triangular solves ─────────────────────────────────────

# Forward solve: (D + L) y = r
# y_i = inv_diag_i * (r_i - Σ_{j<i} a_{ij} * y_j)
@kernel function _dilu_fwd_kernel!(tmp, @Const(inv_diag), @Const(nzval), @Const(colval),
                                   @Const(rp), @Const(diag_idx), @Const(fwd_order), level_offset)
    idx = @index(Global)
    @inbounds begin
        i = fwd_order[level_offset + idx]
        ti_one = one(eltype(rp))
        s = zero(eltype(tmp))
        for nz in rp[i]:(diag_idx[i]-ti_one)
            j = colval[nz]
            s += nzval[nz] * tmp[j]
        end
        tmp[i] = inv_diag[i] * (tmp[i] - s)
    end
end

# Backward solve: (D + U) z = D y  ⟹  z_i = y_i - inv_diag_i * Σ_{j>i} a_{ij} * z_j
@kernel function _dilu_bwd_kernel!(tmp, @Const(inv_diag), @Const(nzval), @Const(colval),
                                   @Const(rp), @Const(diag_idx), @Const(bwd_order), level_offset)
    idx = @index(Global)
    @inbounds begin
        i = bwd_order[level_offset + idx]
        ti_one = one(eltype(rp))
        s = zero(eltype(tmp))
        for nz in (diag_idx[i]+ti_one):(rp[i+ti_one]-ti_one)
            j = colval[nz]
            s += nzval[nz] * tmp[j]
        end
        tmp[i] -= inv_diag[i] * s
    end
end

"""
    smooth!(x, A, b, smoother::DILUSmoother; steps=1)

Apply DILU smoothing: x += M⁻¹ (b - Ax) where M = (D+L)*D⁻¹*(D+U).
Uses KA kernels with level scheduling. All computation stays on-device.
"""
function smooth!(x::AbstractVector, A::CSRMatrix{Tv, Ti}, b::AbstractVector,
                 smoother::DILUSmoother{Tv, Ti, Tx}; steps::Int=1, reverse::Bool=false,
                 backend=_get_backend(nonzeros(A)), block_size::Int=A.block_size) where {Tv, Ti, Tx}
    n = size(A, 1)
    n == 0 && return x
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = rowptr(A)
    tmp = smoother.tmp

    inv_diag = smoother.inv_diag
    diag_idx = smoother.diag_idx
    fwd_order = smoother.fwd_order
    bwd_order = smoother.bwd_order
    num_fwd_levels = smoother.num_fwd_levels
    combined_offsets = smoother.level_offsets
    bwd_offset_start = num_fwd_levels + 2
    num_bwd_levels = length(combined_offsets) - bwd_offset_start

    # Reuse the residual kernel from GPU ILU(0) — same operation
    res_kernel! = _gpu_ilu0_residual_kernel!(backend, block_size)
    fwd_kernel! = _dilu_fwd_kernel!(backend, block_size)
    bwd_kernel! = _dilu_bwd_kernel!(backend, block_size)
    upd_kernel! = _gpu_ilu0_update_kernel!(backend, block_size)

    for _ in 1:steps
        # Compute residual: tmp = b - A*x
        res_kernel!(tmp, b, x, nzv, cv, rp; ndrange=n)
        _synchronize(backend)

        # Forward substitution: (D+L) y = tmp, level by level
        for lev in 1:num_fwd_levels
            lev_start = combined_offsets[lev]
            lev_end = combined_offsets[lev+1] - 1
            count = lev_end - lev_start + 1
            count == 0 && continue
            fwd_kernel!(tmp, inv_diag, nzv, cv, rp, diag_idx, fwd_order, lev_start - 1; ndrange=count)
            _synchronize(backend)
        end

        # Backward substitution: (D+U) z = D*y, level by level
        for lev in 1:num_bwd_levels
            lev_start = combined_offsets[bwd_offset_start + lev - 1]
            lev_end = combined_offsets[bwd_offset_start + lev] - 1
            count = lev_end - lev_start + 1
            count == 0 && continue
            bwd_kernel!(tmp, inv_diag, nzv, cv, rp, diag_idx, bwd_order, lev_start - 1; ndrange=count)
            _synchronize(backend)
        end

        # Update: x += dx
        upd_kernel!(x, tmp; ndrange=n)
        _synchronize(backend)
    end
    return x
end

function build_smoother(A::CSRMatrix{Tv, Ti}, ::DILUSmootherType, ω::Real; backend=DEFAULT_BACKEND, block_size::Int=64,
                        x_eltype::Type=Tv) where {Tv, Ti}
    return build_dilu_smoother(A; x_eltype=x_eltype)
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
                 smoother::AbstractSmoother; steps::Int=1, reverse::Bool=false, backend=DEFAULT_BACKEND, block_size::Int=64)
    A_csr = csr_from_csc(A)
    return smooth!(x, A_csr, b, smoother; steps=steps, reverse=reverse, backend=backend, block_size=block_size)
end
