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

Draugr provides a set of `Base.@ccallable` routines so the solver can be
called from C, C++, Fortran, or any language that can call C functions.
All exported symbols use the `draugr_amg_` prefix.

Configuration is passed as a **JSON string** via `draugr_amg_config_from_json()`.

### JSON Configuration

All parameters are optional. Omitted keys use sensible defaults (HMIS
coarsening, Extended+i interpolation, L1 Colored GS smoother, V-cycle).

| Parameter               | Type            | Default          | Description                              |
|-------------------------|-----------------|------------------|------------------------------------------|
| `coarsening`            | string          | `"hmis"`         | `"aggregation"`, `"pmis"`, `"hmis"`, `"rs"`, `"aggressive_pmis"`, `"aggressive_hmis"`, `"smoothed_aggregation"` |
| `interpolation`         | string or object| `"extended_i"`   | `"direct"`, `"standard"`, `"extended_i"` |
| `smoother`              | string or object| `"l1_colored_gs"`| `"jacobi"`, `"colored_gs"`, `"serial_gs"`, `"spai0"`, `"spai1"`, `"l1_jacobi"`, `"l1_colored_gs"`, `"chebyshev"`, `"ilu0"` |
| `strength`              | string          | `"absolute"`     | `"absolute"` or `"signed"`               |
| `cycle`                 | string          | `"v"`            | `"v"` or `"w"`                           |
| `theta`                 | double          | `0.5`            | Strength-of-connection threshold         |
| `max_levels`            | int             | `20`             |                                          |
| `max_coarse_size`       | int             | `50`             |                                          |
| `pre_smoothing_steps`   | int             | `1`              |                                          |
| `post_smoothing_steps`  | int             | `1`              |                                          |
| `verbose`               | int             | `0`              | 0=silent, 1=summary, 2=per-iteration    |
| `jacobi_omega`          | double          | `0.667`          | Jacobi relaxation weight                 |
| `max_row_sum`           | double          | `1.0`            | Dependency weakening (1.0 = disabled)    |
| `coarse_solve_on_cpu`   | bool            | `false`          |                                          |
| `initial_coarsening`    | string          | same as coarsening| Coarsening for first N levels           |
| `initial_coarsening_levels` | int         | `0`              |                                          |

When `interpolation` or `smoother` is a plain string, defaults for that type
are used. Use the object form for type-specific sub-parameters:

```json
"interpolation": {
    "type": "extended_i",
    "trunc_factor": 0.3,
    "max_elements": 5,
    "norm_p": 2,
    "rescale": true
}
```

```json
"smoother": {"type": "jacobi", "omega": 0.667}
```

Unknown keys are silently ignored, so host applications can include their own
keys in the same JSON object.

### Error Handling

On failure, all `draugr_amg_*` functions return `-1`. Call
`draugr_amg_last_error()` to get the error message as a C string.

### Workflow from Julia

```julia
using Draugr

# 1. Create a config handle from JSON
json = """{
    "coarsening": "hmis",
    "interpolation": {"type": "extended_i", "trunc_factor": 0.3},
    "smoother": "l1_colored_gs",
    "theta": 0.5
}"""
cfg = draugr_amg_config_from_json(json)

# 2. Setup (pass index_base=0 for 0-based arrays)
h = draugr_amg_setup(Int32(n), Int32(nnz),
        pointer(rowptr), pointer(colval), pointer(nzval),
        cfg, Int32(0))

# 3. Solve
niter = draugr_amg_solve(h, Int32(n), pointer(x), pointer(b), cfg, 1e-10, Int32(100))

# 4. Apply a single cycle
draugr_amg_cycle(h, Int32(n), pointer(x), pointer(b), cfg)

# 5. Resetup with new coefficients (same sparsity)
draugr_amg_resetup(h, Int32(n), Int32(nnz),
        pointer(rowptr), pointer(colval), pointer(nzval_new),
        cfg, Int32(0))

# 6. Cleanup
draugr_amg_free(h)
draugr_amg_config_free(cfg)
```

### Workflow from C/C++

```c
#include "draugr_amg.h"

// Initialize Julia runtime (once)
init_julia(0, NULL);

// 1. Create config from JSON
const char* json = "{"
    "\"coarsening\": \"hmis\","
    "\"interpolation\": {\"type\": \"extended_i\", \"trunc_factor\": 0.3},"
    "\"smoother\": \"l1_colored_gs\","
    "\"theta\": 0.5"
    "}";
int32_t cfg = draugr_amg_config_from_json(json);
if (cfg < 0) {
    fprintf(stderr, "Config error: %s\n", draugr_amg_last_error());
    return 1;
}

// 2. Setup with 0-based CSR arrays
int32_t h = draugr_amg_setup(n, nnz, rowptr, colval, nzval, cfg, 0);

// 3. Solve or cycle
draugr_amg_solve(h, n, x, b, cfg, 1e-10, 100);
draugr_amg_cycle(h, n, x, b, cfg);

// 4. Resetup (same sparsity, new coefficients)
draugr_amg_resetup(h, n, nnz, rowptr, colval, nzval_new, cfg, 0);

// 5. Cleanup
draugr_amg_free(h);
draugr_amg_config_free(cfg);

// Shutdown Julia runtime (once)
shutdown_julia(0);
```

### Getting `@cfunction` Pointers

To pass function pointers to a C library:

```julia
cfuncs = draugr_amg_get_cfunctions()
# cfuncs.config_from_json, cfuncs.setup, cfuncs.solve, etc. are Ptr{Cvoid}
```

## License

This module is MIT licensed.
