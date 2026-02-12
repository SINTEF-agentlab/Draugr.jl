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
    smooth!(x, A, b, smoother; steps=1)

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
