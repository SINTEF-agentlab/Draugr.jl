# Draugr.jl
> [!NOTE]
> This code is part of a research project on how agents can write and maintain targeted, complex scientific code under expert guidance. A part from minor changes (e.g. tweaking default parameters or options), all code has been written using agents. If you have feedback or questions, [the issue page is manned by humans](https://github.com/SINTEF-agentlab/Draugr.jl/issues).

A parallel Algebraic Multigrid (AMG) solver for Julia with GPU support via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).
Supports NVIDIA (CUDA), AMD (AMDGPU/ROCm), and Apple (Metal) GPUs as well as
CPU execution.  Built as a research project at the SINTEF AgentLab. The initial public release of the code is entirely coded by Claude Opus 4.6 with manual testing and feedback. The code is heavily based on the feature set of [hypre's BoomerAMG](https://github.com/hypre-space/hypre) (for classical AMG variants) and [amgcl](https://github.com/ddemidov/amgcl) (for aggregation variants).

Main design considerations:

- Use KernelAbstractions.jl for portable GPU support across vendors.
- Solution of "tough" scalar problems with high variation in coefficients in a shared memory environment.
- Allow fast resetup (reuse of the AMG hierarchy when coefficients change but sparsity does not) by recomputing Galerkin products and smoothers.
- Support both CSC and CSR input formats (with internal conversion as needed), but rely on CSR internally. You should use a CSR format for best performance and to avoid copies.
- Provide a flexible configuration system to enable a wide range of AMG variants (coarsening, interpolation, smoothers, etc.) while maintaining good defaults for typical use cases.
- Provide a C-callable interface for use from other languages.

If you want to test the capabilities of this code, we encourage you to report your findings. For human-written AMG solvers in Julia, you can have a look at some of the alternatives: [HYPRE](https://github.com/fredrikekre/HYPRE.jl), [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl), [AMGCLWrap.jl](https://github.com/j-fu/AMGCLWrap.jl).

## Installation

This package is not registered in the General registry yet, so you need to install it directly from the GitHub repository:

```julia
using Pkg
Pkg.add(url="https://github.com/moyner/Draugr.jl")
```

A potential future registration in the Julia registry is pending a validation performed by humans.

For GPU backends, install the corresponding package:

```julia
Pkg.add("CUDA")    # NVIDIA GPUs
Pkg.add("AMDGPU")  # AMD GPUs (ROCm)
Pkg.add("Metal")   # Apple GPUs
```

## Quick Start

```julia
using Draugr, SparseArrays

# Build a standard Julia CSC sparse matrix
I = [1,1,2,2,2,3,3]; J = [1,2,1,2,3,2,3]
V = [2.0,-1.0,-1.0,2.0,-1.0,-1.0,2.0]
A_csc = sparse(I, J, V, 3, 3)

# Setup the AMG hierarchy (CSC matrices are converted to CSR internally)
config    = AMGConfig()
hierarchy = amg_setup(A_csc, config)

# Or use a CSR matrix directly via SparseMatricesCSR.jl
using SparseMatricesCSR
A_csr = SparseMatrixCSR(A_csc)
hierarchy = amg_setup(A_csr, config)

# Solve Ax = b
b = [1.0, 0.0, 1.0]
x = zeros(3)
x, niter = amg_solve!(x, b, hierarchy, config; tol=1e-10, maxiter=100)

# Or apply a single AMG cycle (e.g. as a preconditioner)
fill!(x, 0.0)
amg_cycle!(x, b, hierarchy, config)
```

## AMG Configuration

`AMGConfig` controls every aspect of the solver.  All fields have sensible
defaults:

```julia
config = AMGConfig(;
    coarsening           = AggregationCoarsening(),  # see Coarsening below
    smoother             = JacobiSmootherType(),      # see Smoothers below
    max_levels           = 20,
    max_coarse_size      = 50,
    pre_smoothing_steps  = 1,
    post_smoothing_steps = 1,
    jacobi_omega         = 2/3,
    cycle_type           = :V,          # :V or :W
    strength_type        = AbsoluteStrength(), # or SignedStrength()
    verbose              = 0,           # 0=silent, 1=summary, 2=per-iteration
)
```

A HYPRE-like default for challenging 3D problems is also provided:

```julia
config = hypre_default_config()  # HMIS + aggressive coarsening + ext+i interpolation
```

### Coarsening Algorithms

| Type                            | Description                                    |
|---------------------------------|------------------------------------------------|
| `AggregationCoarsening(θ)`      | Greedy aggregation (default, θ=0.25)           |
| `SmoothedAggregationCoarsening` | Smoothed aggregation (Jacobi-smoothed P)       |
| `PMISCoarsening(θ, interp)`     | Parallel Modified Independent Set              |
| `HMISCoarsening(θ, interp)`     | Hybrid MIS (less aggressive than PMIS)         |
| `RSCoarsening(θ, interp)`       | Classical Ruge-Stüben                          |
| `AggressiveCoarsening(θ, base)` | Two-pass aggressive (`:pmis` or `:hmis` base)  |

### Interpolation Types

| Type                       | Description                              |
|----------------------------|------------------------------------------|
| `DirectInterpolation()`    | Direct interpolation from C-points       |
| `StandardInterpolation()`  | Classical RS interpolation               |
| `ExtendedIInterpolation()` | Extended+i (recommended with HMIS/aggressive) |

All interpolation types accept an optional `trunc_factor` for weight truncation
(matching HYPRE's `AggTruncFactor`).

### Smoothers

| Type                        | Description                                 |
|-----------------------------|---------------------------------------------|
| `JacobiSmootherType()`      | Weighted Jacobi (default, GPU-friendly)     |
| `L1JacobiSmootherType()`    | l1-Jacobi (more robust for difficult cases) |
| `ColoredGaussSeidelType()`  | Parallel multicolor Gauss-Seidel            |
| `L1ColoredGaussSeidelType()`| L1 multicolor Gauss-Seidel (more robust)    |
| `SerialGaussSeidelType()`   | Sequential Gauss-Seidel (CPU only)          |
| `SPAI0SmootherType()`       | Diagonal sparse approximate inverse         |
| `SPAI1SmootherType()`       | Sparse approximate inverse (pattern of A)   |
| `ChebyshevSmootherType()`   | Chebyshev polynomial smoother               |
| `ILU0SmootherType()`        | Incomplete LU(0)                            |

## Resetup (Updating Coefficients)

When the matrix sparsity stays the same but coefficients change (e.g. in a
nonlinear iteration), use `amg_resetup!` to avoid redoing the symbolic phase:

```julia
# A_new has the same sparsity pattern as A
amg_resetup!(hierarchy, A_new, config)
```

## GPU Usage

Draugr works on any GPU supported by KernelAbstractions.jl.  The setup
phase (coarsening, prolongation construction) runs on CPU; the hierarchy
data is then copied to the GPU for the solve/cycle phase.

### NVIDIA GPUs (CUDA)

```julia
using CUDA, Draugr

A_gpu = CuSparseMatrixCSR(A_csc)          # or build from CuVectors
hierarchy = amg_setup(A_gpu, config)       # auto-selects CUDABackend()

x = CUDA.zeros(Float64, n)
b = CuArray(b_cpu)
amg_solve!(x, b, hierarchy, config)
```

### AMD GPUs (AMDGPU / ROCm)

```julia
using AMDGPU, Draugr

A_gpu = ROCSparseMatrixCSR(A_csc)          # or build from ROCVectors
hierarchy = amg_setup(A_gpu, config)       # auto-selects ROCBackend()

x = AMDGPU.zeros(Float64, n)
b = ROCArray(b_cpu)
amg_solve!(x, b, hierarchy, config)
```

### Apple GPUs (Metal)

Metal does not have a native sparse CSR type.  Construct a `CSRMatrix` from
`MtlVector`s directly:

```julia
using Metal, Draugr

rp  = MtlVector(rowptr_cpu)
cv  = MtlVector(colval_cpu)
nzv = MtlVector(nzval_cpu)
A   = CSRMatrix(rp, cv, nzv, n, n)
hierarchy = amg_setup(A, config)           # auto-selects MetalBackend()
```

## Jutul Integration

Draugr provides a `DraugrPreconditioner` that implements the Jutul
preconditioner interface, so it can be used as a drop-in preconditioner in
Jutul's linear solvers:

```julia
using Jutul, Draugr

precond = DraugrPreconditioner(;
    coarsening = HMISCoarsening(0.5, ExtendedIInterpolation(0.3)),
    smoother   = JacobiSmootherType(),
    verbose    = 1,
)
# Pass `precond` to Jutul's solver setup
```

## C-Callable Interface

Draugr provides a set of `@cfunction`-compatible routines so the solver
can be called from C, C++, Fortran, or any language that can call C functions.
Integer enums are used to select algorithms.

### Enums

| Category        | Values                                                                     |
|-----------------|----------------------------------------------------------------------------|
| `CoarseningEnum`| `COARSENING_AGGREGATION(0)`, `COARSENING_PMIS(1)`, `COARSENING_HMIS(2)`, `COARSENING_RS(3)`, `COARSENING_AGGRESSIVE_PMIS(4)`, `COARSENING_AGGRESSIVE_HMIS(5)`, `COARSENING_SMOOTHED_AGGREGATION(6)` |
| `SmootherEnum`  | `SMOOTHER_JACOBI(0)`, `SMOOTHER_COLORED_GS(1)`, `SMOOTHER_SERIAL_GS(2)`, `SMOOTHER_SPAI0(3)`, `SMOOTHER_SPAI1(4)`, `SMOOTHER_L1_JACOBI(5)`, `SMOOTHER_CHEBYSHEV(6)`, `SMOOTHER_ILU0(7)`, `SMOOTHER_L1_COLORED_GS(8)` |
| `InterpolationEnum` | `INTERPOLATION_DIRECT(0)`, `INTERPOLATION_STANDARD(1)`, `INTERPOLATION_EXTENDED_I(2)` |
| `CycleEnum`     | `CYCLE_V(0)`, `CYCLE_W(1)` |
| `StrengthEnum`  | `STRENGTH_ABSOLUTE(0)`, `STRENGTH_SIGNED(1)` |

### Workflow from Julia

```julia
using Draugr

# 1. Create a config handle
cfg = amg_c_config_create(
    Int32(0),   # coarsening: AGGREGATION
    Int32(0),   # smoother:   JACOBI
    Int32(0),   # interpolation: DIRECT
    Int32(0),   # strength:   ABSOLUTE
    Int32(0),   # cycle:      V
    0.25,       # θ
    0.0,        # trunc_factor
    2/3,        # jacobi_omega
    Int32(20),  # max_levels
    Int32(50),  # max_coarse_size
    Int32(1),   # pre_smoothing_steps
    Int32(1),   # post_smoothing_steps
    Int32(0),   # verbose
)

# 2. Setup (rowptr, colval, nzval are 1-based Int32/Float64 arrays)
h = amg_c_setup(Int32(n), Int32(nnz), pointer(rowptr), pointer(colval), pointer(nzval), cfg)

# 3. Solve
niter = amg_c_solve!(h, Int32(n), pointer(x), pointer(b), cfg, 1e-10, Int32(100))

# 4. Apply a single cycle
amg_c_cycle!(h, Int32(n), pointer(x), pointer(b), cfg)

# 5. Resetup with new coefficients (same sparsity)
amg_c_resetup!(h, Int32(n), Int32(nnz), pointer(rowptr), pointer(colval), pointer(nzval_new), cfg)

# 6. Cleanup
amg_c_free!(h)
amg_c_config_free!(cfg)
```

### Getting `@cfunction` Pointers

To pass function pointers to a C library:

```julia
cfuncs = amg_c_get_cfunctions()
# cfuncs.setup, cfuncs.solve, cfuncs.cycle, etc. are Ptr{Cvoid}
```

## License

This module is MIT licensed.
