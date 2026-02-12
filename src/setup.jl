"""
    amg_setup(A, config) -> AMGHierarchy

Perform the full AMG setup (analysis phase). This determines the coarsening at each
level, constructs prolongation operators, computes Galerkin products, and sets up
smoothers.

The sparsity structure computed here is reused by `amg_resetup!` when matrix
coefficients change but the pattern remains the same.
"""
function amg_setup(A::StaticSparsityMatrixCSR{Tv, Ti}, config::AMGConfig=AMGConfig()) where {Tv, Ti}
    levels = AMGLevel{Tv, Ti}[]
    A_current = A
    n_finest = size(A, 1)
    for lvl in 1:(config.max_levels - 1)
        n = size(A_current, 1)
        n <= config.max_coarse_size && break
        # Coarsen
        agg, n_coarse = coarsen(A_current, config.coarsening)
        n_coarse >= n && break  # no coarsening progress
        # Build prolongation
        P = build_prolongation(A_current, agg, n_coarse)
        # Compute coarse operator via Galerkin product
        A_coarse, r_map = compute_coarse_sparsity(A_current, P, n_coarse)
        # Build smoother
        smoother = build_jacobi_smoother(A_current, config.jacobi_omega)
        # Workspace
        r = zeros(Tv, n)
        xc = zeros(Tv, n_coarse)
        bc = zeros(Tv, n_coarse)
        level = AMGLevel{Tv, Ti}(A_current, P, r_map, smoother, r, xc, bc)
        push!(levels, level)
        A_current = A_coarse
    end
    # Set up direct solver at coarsest level
    n_coarse = size(A_current, 1)
    coarse_dense = Matrix{Tv}(undef, n_coarse, n_coarse)
    _csr_to_dense!(coarse_dense, A_current)
    coarse_factor = lu(coarse_dense)
    coarse_x = zeros(Tv, n_coarse)
    coarse_b = zeros(Tv, n_coarse)
    # Pre-allocate residual buffer for amg_solve! at finest level size
    solve_r = zeros(Tv, n_finest)
    return AMGHierarchy{Tv, Ti}(levels, coarse_dense, coarse_factor, coarse_x, coarse_b, solve_r)
end

"""
    _csr_to_dense!(M, A; backend=CPU())

Convert a StaticSparsityMatrixCSR to a dense matrix using a KA kernel.
"""
function _csr_to_dense!(M::Matrix{Tv}, A::StaticSparsityMatrixCSR{Tv, Ti};
                        backend=CPU()) where {Tv, Ti}
    fill!(M, zero(Tv))
    n = size(A, 1)
    cv = colvals(A)
    nzv = nonzeros(A)
    rp = rowptr(A)
    kernel! = csr_to_dense_kernel!(backend, 64)
    kernel!(M, nzv, cv, rp; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return M
end

@kernel function csr_to_dense_kernel!(M, @Const(nzval), @Const(colval), @Const(rp))
    i = @index(Global)
    @inbounds begin
        for nz in rp[i]:(rp[i+1]-1)
            j = colval[nz]
            M[i, j] = nzval[nz]
        end
    end
end
