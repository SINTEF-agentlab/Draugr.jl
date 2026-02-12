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

Apply prolongation: x_fine += P * x_coarse
"""
function prolongate!(x_fine::AbstractVector, P::ProlongationOp, x_coarse::AbstractVector)
    @inbounds for i in 1:P.nrow
        for nz in P.rowptr[i]:(P.rowptr[i+1]-1)
            j = P.colval[nz]
            x_fine[i] += P.nzval[nz] * x_coarse[j]
        end
    end
    return x_fine
end

"""
    restrict!(b_coarse, P, r_fine)

Apply restriction (P^T): b_coarse = P^T * r_fine
"""
function restrict!(b_coarse::AbstractVector, P::ProlongationOp, r_fine::AbstractVector)
    fill!(b_coarse, zero(eltype(b_coarse)))
    @inbounds for i in 1:P.nrow
        for nz in P.rowptr[i]:(P.rowptr[i+1]-1)
            j = P.colval[nz]
            b_coarse[j] += P.nzval[nz] * r_fine[i]
        end
    end
    return b_coarse
end
