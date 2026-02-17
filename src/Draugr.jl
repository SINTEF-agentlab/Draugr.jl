module Draugr

using SparseArrays
using LinearAlgebra
using Random
using KernelAbstractions
using Printf

# Default backend for KernelAbstractions
const DEFAULT_BACKEND = CPU(; static = true)

# Internal CSR matrix type
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

# Preconditioner base types
include("preconditioner.jl")

# C-callable function interface
include("cfunction_interface.jl")

"""
    csr_from_gpu(A)

Convert a GPU sparse CSR matrix to the internal `CSRMatrix` representation.
This is a generic function extended by GPU backend extensions (CUDA, Metal).
"""
function csr_from_gpu end

"""
    csr_from_static(A)

Convert a `StaticSparsityMatrixCSR` (Jutul) to the internal `CSRMatrix`.
Extended by the Jutul extension.
"""
function csr_from_static end

"""
    csr_copy_nzvals!(dest, src)

Copy nonzero values from a source sparse matrix into an existing `CSRMatrix`.
Extended by the Jutul extension for `StaticSparsityMatrixCSR`.
"""
function csr_copy_nzvals! end

"""
    static_csr_from_csc(A)

Create a `StaticSparsityMatrixCSR` from a `SparseMatrixCSC`.
Extended by the Jutul extension.
"""
function static_csr_from_csc end

"""
    find_nz_index(A, row, col)

Find the index in the nonzero array for entry (row, col). Returns 0 if not found.
Extended by the Jutul extension for `StaticSparsityMatrixCSR`.
"""
function find_nz_index end

# Public API
export colvals, rowptr
export CSRMatrix, csr_from_gpu, csr_to_cpu, csr_from_csc, csr_from_static, csr_from_raw
export static_csr_from_csc, find_nz_index, csr_copy_nzvals!
export AggregationCoarsening, PMISCoarsening, HMISCoarsening, AggressiveCoarsening
export SmoothedAggregationCoarsening, RSCoarsening
export DirectInterpolation, StandardInterpolation, ExtendedIInterpolation
export AbsoluteStrength, SignedStrength
export AMGConfig, AMGHierarchy, AMGLevel
export amg_setup, amg_resetup!, amg_cycle!, amg_solve!
export hypre_default_config
export AbstractSmoother
export JacobiSmoother, ColoredGaussSeidelSmoother, L1ColoredGaussSeidelSmoother, SerialGaussSeidelSmoother
export SPAI0Smoother, SPAI1Smoother
export L1JacobiSmoother, ChebyshevSmoother, ILU0Smoother
export JacobiSmootherType, ColoredGaussSeidelType, SerialGaussSeidelType
export SPAI0SmootherType, SPAI1SmootherType
export L1JacobiSmootherType, L1ColoredGaussSeidelType, ChebyshevSmootherType, ILU0SmootherType
export build_smoother, update_smoother!, smooth!
export AbstractDraugrPreconditioner, DraugrPreconditioner
export setup_specific_preconditioner
# C-callable interface
export CoarseningEnum, SmootherEnum, InterpolationEnum, CycleEnum, StrengthEnum
export COARSENING_AGGREGATION, COARSENING_PMIS, COARSENING_HMIS, COARSENING_RS
export COARSENING_AGGRESSIVE_PMIS, COARSENING_AGGRESSIVE_HMIS, COARSENING_SMOOTHED_AGGREGATION
export SMOOTHER_JACOBI, SMOOTHER_COLORED_GS, SMOOTHER_SERIAL_GS
export SMOOTHER_SPAI0, SMOOTHER_SPAI1, SMOOTHER_L1_JACOBI, SMOOTHER_CHEBYSHEV, SMOOTHER_ILU0
export SMOOTHER_L1_COLORED_GS
export INTERPOLATION_DIRECT, INTERPOLATION_STANDARD, INTERPOLATION_EXTENDED_I
export CYCLE_V, CYCLE_W
export STRENGTH_ABSOLUTE, STRENGTH_SIGNED
export amg_c_config_create, amg_c_setup, amg_c_resetup!, amg_c_solve!, amg_c_cycle!
export amg_c_free!, amg_c_config_free!, amg_c_get_cfunctions

end # module
