module ParallelAMG

using SparseArrays
using LinearAlgebra
using Random
using KernelAbstractions
using Atomix
using Printf

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

# Smoothers (Jacobi, Colored GS, SPAI0, SPAI1)
include("smoothers.jl")

# AMG setup (analysis phase)
include("setup.jl")

# AMG resetup (same pattern, new coefficients)
include("resetup.jl")

# AMG cycling (V-cycle, solver)
include("cycle.jl")

# Jutul preconditioner interface
include("jutul_interface.jl")

# Public API
export StaticSparsityMatrixCSR, static_sparsity_sparse, static_csr_from_csc
export colvals, rowptr
export AggregationCoarsening, PMISCoarsening, HMISCoarsening, AggressiveCoarsening
export DirectInterpolation, StandardInterpolation, ExtendedIInterpolation
export AMGConfig, AMGHierarchy, AMGLevel
export amg_setup, amg_resetup!, amg_cycle!, amg_solve!
export JacobiSmoother, ColoredGaussSeidelSmoother, SPAI0Smoother, SPAI1Smoother
export JacobiSmootherType, ColoredGaussSeidelType, SPAI0SmootherType, SPAI1SmootherType
export smooth!
export ParallelAMGPreconditioner

end # module
