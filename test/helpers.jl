# ── Matrix builders ──────────────────────────────────────────────────────────

# Helper: convert StaticSparsityMatrixCSR to internal CSRMatrix for unit tests
to_csr(A) = Draugr.csr_from_static(A)

function poisson1d_csr(n)
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:n
        push!(I, i); push!(J, i); push!(V, 2.0)
        if i > 1
            push!(I, i); push!(J, i-1); push!(V, -1.0)
        end
        if i < n
            push!(I, i); push!(J, i+1); push!(V, -1.0)
        end
    end
    return static_sparsity_sparse(I, J, V, n, n)
end

function poisson2d_csr(nx, ny=nx)
    n = nx * ny
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:ny, i in 1:nx
        idx = (j-1)*nx + i
        push!(I, idx); push!(J, idx); push!(V, 4.0)
        if i > 1
            push!(I, idx); push!(J, idx-1); push!(V, -1.0)
        end
        if i < nx
            push!(I, idx); push!(J, idx+1); push!(V, -1.0)
        end
        if j > 1
            push!(I, idx); push!(J, idx-nx); push!(V, -1.0)
        end
        if j < ny
            push!(I, idx); push!(J, idx+nx); push!(V, -1.0)
        end
    end
    return static_sparsity_sparse(I, J, V, n, n)
end

function reservoir_like_csr(n)
    I = Int[]; J = Int[]; V = Float64[]
    for i in 1:n
        push!(I, i); push!(J, i); push!(V, 10.0)
        if i > 1
            v = i % 3 == 0 ? 0.5 : -1.0  # positive off-diag every 3rd row
            push!(I, i); push!(J, i-1); push!(V, v)
        end
        if i < n
            v = (i+1) % 5 == 0 ? 0.3 : -2.0  # positive off-diag every 5th row
            push!(I, i); push!(J, i+1); push!(V, v)
        end
    end
    return static_sparsity_sparse(I, J, V, n, n)
end

function anisotropic_csr(nx, ny; kx=1e4, ky=1e-2)
    n = nx * ny
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:ny, i in 1:nx
        idx = (j-1)*nx + i
        diag = 2*kx + 2*ky
        push!(I, idx); push!(J, idx); push!(V, diag)
        if i > 1 push!(I, idx); push!(J, idx-1); push!(V, -kx) end
        if i < nx push!(I, idx); push!(J, idx+1); push!(V, -kx) end
        if j > 1 push!(I, idx); push!(J, idx-nx); push!(V, -ky) end
        if j < ny push!(I, idx); push!(J, idx+nx); push!(V, -ky) end
    end
    return static_sparsity_sparse(I, J, V, n, n)
end

"""
    reservoir_spe10_like(nx, ny, nz; perm_range=1e6, seed=42)

Build a 3D 7-point stencil matrix with log-normally distributed permeability,
mimicking the SPE10 benchmark. The `perm_range` controls the range of
permeability values (ratio of max to min). Larger values (e.g., 1e8) produce
more challenging matrices with near-singular rows that trigger max_row_sum
weakening.
"""
function reservoir_spe10_like(nx, ny, nz; perm_range=1e6, seed=42)
    Random.seed!(seed)
    n = nx * ny * nz
    logK = randn(nx, ny, nz) .* (log10(perm_range)/2)
    K = 10 .^ logK
    I = Int[]; J = Int[]; V = Float64[]
    for iz in 1:nz, iy in 1:ny, ix in 1:nx
        idx = (iz-1)*nx*ny + (iy-1)*nx + ix
        diag = 0.0
        for (dx,dy,dz,scale) in [(-1,0,0,1.0),(1,0,0,1.0),(0,-1,0,1.0),(0,1,0,1.0),(0,0,-1,0.01),(0,0,1,0.01)]
            jx,jy,jz = ix+dx,iy+dy,iz+dz
            (jx < 1 || jx > nx || jy < 1 || jy > ny || jz < 1 || jz > nz) && continue
            j = (jz-1)*nx*ny + (jy-1)*nx + jx
            T = scale * 2.0 / (1.0/K[ix,iy,iz] + 1.0/K[jx,jy,jz])
            push!(I, idx); push!(J, j); push!(V, -T)
            diag += T
        end
        c_i = K[ix, iy, iz] * 1e-6
        diag += c_i
        push!(I, idx); push!(J, idx); push!(V, diag)
    end
    return static_sparsity_sparse(I, J, V, n, n)
end

# Helper: read a Matrix Market (.mtx) file and return a SparseMatrixCSC
function read_mtx(path)
    open(path) do f
        local header
        while true
            line = readline(f)
            startswith(line, '%') && continue
            header = strip(line)
            break
        end
        parts = split(header)
        nrows = parse(Int, parts[1])
        ncols = parse(Int, parts[2])
        nz    = parse(Int, parts[3])
        I_arr = Vector{Int}(undef, nz)
        J_arr = Vector{Int}(undef, nz)
        V_arr = Vector{Float64}(undef, nz)
        for k in 1:nz
            p = split(strip(readline(f)))
            I_arr[k] = parse(Int, p[1])
            J_arr[k] = parse(Int, p[2])
            V_arr[k] = parse(Float64, p[3])
        end
        return sparse(I_arr, J_arr, V_arr, nrows, ncols)
    end
end

# ── Test helper functions ─────────────────────────────────────────────────────

# Convert a ProlongationOp to a SparseMatrixCSC for reference comparisons.
function prolongation_to_sparse(P)
    I_p = Int[]; J_p = Int[]; V_p = Float64[]
    for i in 1:P.nrow
        for nz in P.rowptr[i]:(P.rowptr[i+1]-1)
            push!(I_p, i); push!(J_p, P.colval[nz]); push!(V_p, P.nzval[nz])
        end
    end
    return sparse(I_p, J_p, V_p, P.nrow, P.ncol)
end

# Run a full AMG solve on an n×n 2-D Poisson problem and assert convergence.
# Returns (hierarchy, A, b, x, niter).
function test_amg_convergence(config; n=10, tol=1e-8, maxiter=200)
    A = poisson2d_csr(n)
    N = n * n
    b = rand(N)
    x = zeros(N)
    hierarchy = amg_setup(A, config)
    x, niter = amg_solve!(x, b, hierarchy, config; tol=tol, maxiter=maxiter)
    r = b - sparse(A.At') * x
    @test norm(r) / norm(b) < tol
    @test niter < maxiter
    return hierarchy, A, b, x, niter
end

# Build a smoother on a 1-D Poisson matrix, apply it, and assert the residual
# decreases.
function test_smoother_smoothing(smoother, Ac, A; steps=10)
    b = ones(size(Ac, 1))
    x = zeros(size(Ac, 1))
    smooth!(x, Ac, b, smoother; steps=steps)
    r = b - sparse(A.At') * x
    @test norm(r) < norm(b)
end

# Set up an AMG hierarchy, solve, scale the matrix by 2, resetup, solve again,
# and assert convergence after the resetup.
function test_smoother_resetup(config; n=8)
    A = poisson2d_csr(n)
    N = n * n
    hierarchy = amg_setup(A, config)
    b = rand(N)
    x1 = zeros(N)
    x1, _ = amg_solve!(x1, b, hierarchy, config; tol=1e-8, maxiter=200)
    nonzeros(A) .*= 2.0
    amg_resetup!(hierarchy, A, config)
    x2 = zeros(N)
    x2, _ = amg_solve!(x2, b, hierarchy, config; tol=1e-8, maxiter=200)
    r2 = b - sparse(A.At') * x2
    @test norm(r2) / norm(b) < 1e-8
end
