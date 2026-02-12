"""
    build_prolongation(A, agg, n_coarse)

Build a piecewise-constant prolongation operator from the aggregation map.
Each fine node i is interpolated from aggregate `agg[i]` with weight 1.
Returns a `ProlongationOp`.
"""
function build_prolongation(A::StaticSparsityMatrixCSR{Tv, Ti}, agg::Vector{Int},
                            n_coarse::Int) where {Tv, Ti}
    n_fine = size(A, 1)
    # P is n_fine × n_coarse, with exactly one nonzero per row (aggregation-based)
    rowptr = Vector{Ti}(undef, n_fine + 1)
    colval = Vector{Ti}(undef, n_fine)
    nzval = Vector{Tv}(undef, n_fine)
    @inbounds for i in 1:n_fine
        rowptr[i] = Ti(i)
        colval[i] = Ti(agg[i])
        nzval[i] = one(Tv)
    end
    rowptr[n_fine + 1] = Ti(n_fine + 1)
    return ProlongationOp{Ti, Tv}(rowptr, colval, nzval, n_fine, n_coarse)
end

"""
    prolongate!(x_fine, P, x_coarse)

Apply prolongation: x_fine += P * x_coarse.
Uses KernelAbstractions for parallel execution over fine rows.
"""
function prolongate!(x_fine::AbstractVector, P::ProlongationOp, x_coarse::AbstractVector;
                     backend=CPU())
    kernel! = prolongate_kernel!(backend, 64)
    kernel!(x_fine, P.rowptr, P.colval, P.nzval, x_coarse; ndrange=P.nrow)
    KernelAbstractions.synchronize(backend)
    return x_fine
end

@kernel function prolongate_kernel!(x_fine, @Const(P_rowptr), @Const(P_colval),
                                    @Const(P_nzval), @Const(x_coarse))
    i = @index(Global)
    @inbounds begin
        for nz in P_rowptr[i]:(P_rowptr[i+1]-1)
            j = P_colval[nz]
            x_fine[i] += P_nzval[nz] * x_coarse[j]
        end
    end
end

"""
    restrict!(b_coarse, P, r_fine)

Apply restriction (P^T): b_coarse = P^T * r_fine.
For aggregation-based P (one nonzero per row), this is race-free when
parallelized over fine rows using atomics.
"""
function restrict!(b_coarse::AbstractVector, P::ProlongationOp, r_fine::AbstractVector;
                   backend=CPU())
    fill!(b_coarse, zero(eltype(b_coarse)))
    kernel! = restrict_kernel!(backend, 64)
    kernel!(b_coarse, P.rowptr, P.colval, P.nzval, r_fine; ndrange=P.nrow)
    KernelAbstractions.synchronize(backend)
    return b_coarse
end

@kernel function restrict_kernel!(b_coarse, @Const(P_rowptr), @Const(P_colval),
                                  @Const(P_nzval), @Const(r_fine))
    i = @index(Global)
    @inbounds begin
        for nz in P_rowptr[i]:(P_rowptr[i+1]-1)
            j = P_colval[nz]
            Atomix.@atomic b_coarse[j] += P_nzval[nz] * r_fine[i]
        end
    end
end
