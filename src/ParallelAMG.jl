module ParallelAMG

using SparseArrays
using LinearAlgebra
using Random
using KernelAbstractions
using Atomix

# Import StaticCSR types from Jutul
using Jutul.StaticCSR: StaticSparsityMatrixCSR, colvals, static_sparsity_sparse,
                       nthreads, minbatch

# Local helpers for StaticCSR (not provided by Jutul)
include("static_csr_helpers.jl")

# Type definitions
include("types.jl")

# Strength of connection
include("strength.jl")

# Coarsening algorithms
include("coarsening.jl")

# Prolongation and restriction operators
include("prolongation.jl")

# Galerkin triple product
include("galerkin.jl")

# Parallel Jacobi smoother
include("smoothers.jl")

# AMG setup (analysis phase)
include("setup.jl")

# AMG resetup (same pattern, new coefficients)
include("resetup.jl")

# AMG cycling (V-cycle, solver)
include("cycle.jl")

# Public API
export StaticSparsityMatrixCSR, static_sparsity_sparse, static_csr_from_csc
export colvals, rowptr
export AggregationCoarsening, PMISCoarsening, AggressiveCoarsening
export AMGConfig, AMGHierarchy, AMGLevel
export amg_setup, amg_resetup!, amg_cycle!, amg_solve!
export JacobiSmoother, smooth!

end # module
