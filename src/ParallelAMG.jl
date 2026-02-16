module ParallelAMG

using SparseArrays
using LinearAlgebra
using Random
using KernelAbstractions
using Printf

# Import StaticCSR types from Jutul
using Jutul.StaticCSR: StaticSparsityMatrixCSR, static_sparsity_sparse,
                       nthreads, minbatch
import Jutul.StaticCSR: colvals

# Default backend for KernelAbstractions
const DEFAULT_BACKEND = CPU(; static = true)

# Local helpers for StaticCSR (not provided by Jutul)
include("static_csr_helpers.jl")

# Internal CSR matrix type (decoupled from Jutul)
include("csr_matrix.jl")

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

"""
    csr_from_gpu(A)

Convert a GPU sparse CSR matrix to the internal `CSRMatrix` representation.
This is a generic function extended by GPU backend extensions (CUDA, Metal).
"""
function csr_from_gpu end

# Public API
export StaticSparsityMatrixCSR, static_sparsity_sparse, static_csr_from_csc
export colvals, rowptr
export CSRMatrix, csr_from_gpu, csr_to_cpu
export AggregationCoarsening, PMISCoarsening, HMISCoarsening, AggressiveCoarsening
export SmoothedAggregationCoarsening, RSCoarsening
export DirectInterpolation, StandardInterpolation, ExtendedIInterpolation
export AbsoluteStrength, SignedStrength
export AMGConfig, AMGHierarchy, AMGLevel
export amg_setup, amg_resetup!, amg_cycle!, amg_solve!
export hypre_default_config
export AbstractSmoother
export JacobiSmoother, ColoredGaussSeidelSmoother, SerialGaussSeidelSmoother
export SPAI0Smoother, SPAI1Smoother
export L1JacobiSmoother, ChebyshevSmoother, ILU0Smoother
export JacobiSmootherType, ColoredGaussSeidelType, SerialGaussSeidelType
export SPAI0SmootherType, SPAI1SmootherType
export L1JacobiSmootherType, ChebyshevSmootherType, ILU0SmootherType
export build_smoother, update_smoother!, smooth!
export ParallelAMGPreconditioner

end # module
