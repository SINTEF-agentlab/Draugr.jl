# ── Grow-only resize for workspace arrays ─────────────────────────────────────
"""
    _ws_resize!(v::Vector, n::Integer) -> v

Resize workspace vector `v` to at least `n` elements, but never shrink.
This keeps arrays at their high-water mark across levels and resetup calls,
trading memory for speed by avoiding repeated grow/shrink reallocations.
"""
@inline function _ws_resize!(v::Vector, n::Integer)
    n > length(v) && resize!(v, n)
    return v
end

# ── Interpolation type tags ───────────────────────────────────────────────────
abstract type InterpolationType end

"""
    DirectInterpolation(; trunc_factor=0.0)

Direct interpolation: interpolate only from directly connected coarse points.
`trunc_factor`: entries with |w| < trunc_factor * max|w| per row are dropped
(0 = no truncation). Maps to HYPRE's `AggTruncFactor`.
"""
struct DirectInterpolation <: InterpolationType
    trunc_factor::Float64
end
DirectInterpolation() = DirectInterpolation(0.0)

"""
    StandardInterpolation(; trunc_factor=0.0)

Standard (classical Ruge-Stüben) interpolation: includes indirect contributions
through strong fine neighbors.
`trunc_factor`: entries with |w| < trunc_factor * max|w| per row are dropped
(0 = no truncation). Maps to HYPRE's `AggTruncFactor`.
"""
struct StandardInterpolation <: InterpolationType
    trunc_factor::Float64
end
StandardInterpolation() = StandardInterpolation(0.0)

"""
    ExtendedIInterpolation(; trunc_factor=0.0, max_elements=0, norm_p=1, rescale=false)

Extended+i interpolation (HYPRE InterpType=6): extends standard by including
distance-2 coarse points (coarse points reachable through strong fine neighbors).
Recommended for use with HMIS coarsening for challenging 3D problems.
`trunc_factor`: entries with |w| < trunc_factor * max|w| per row are dropped
(0 = no truncation). Maps to HYPRE's `AggTruncFactor`.
`max_elements`: maximum number of interpolation entries per row (0 = no limit).
When the number of entries exceeds this limit, only the strongest entries are kept
and surviving entries are rescaled to preserve the original row sum (matching
HYPRE's default truncation behavior). HYPRE defaults to PMaxElmts=4; set to 4
when using Gauss-Seidel-type smoothers for best match. Default: 0 (no limit).
`norm_p`: norm exponent used for measuring entry magnitude during truncation.
Default: 1 (absolute value). Higher values (e.g. 2) penalize small entries less
aggressively relative to the maximum.
`rescale`: if true, after truncation the surviving entries are rescaled so that
the original row sum is preserved (scale = row_sum / sum_kept), matching HYPRE's
default truncation behavior. The per-entry scaling factors are stored in the
prolongation operator so that they can be reused during resetup without
recomputing the truncated elements. Default: false.
"""
struct ExtendedIInterpolation <: InterpolationType
    trunc_factor::Float64
    max_elements::Int
    norm_p::Int
    rescale::Bool
end
ExtendedIInterpolation() = ExtendedIInterpolation(0.0, 0, 1, false)
ExtendedIInterpolation(trunc_factor::Real) = ExtendedIInterpolation(Float64(trunc_factor), 0, 1, false)
ExtendedIInterpolation(trunc_factor::Real, max_elements::Integer) = ExtendedIInterpolation(Float64(trunc_factor), Int(max_elements), 1, false)

# ── Coarsening type tags ──────────────────────────────────────────────────────
abstract type CoarseningAlgorithm end

"""Greedy aggregation coarsening.

Fields:
- `θ`: Strength threshold (default: 0.25)
- `filtering`: If true, filter (drop) small entries from P to improve sparsity (default: false)
- `filter_tol`: Tolerance for filtering; entries with |p_ij| < filter_tol * max|p_i,:| are dropped (default: 0.1)
"""
struct AggregationCoarsening <: CoarseningAlgorithm
    θ::Float64   # strength threshold
    filtering::Bool
    filter_tol::Float64
end
AggregationCoarsening() = AggregationCoarsening(0.25, false, 0.1)
AggregationCoarsening(θ::Real) = AggregationCoarsening(θ, false, 0.1)
AggregationCoarsening(θ::Real, filtering::Bool) = AggregationCoarsening(θ, filtering, 0.1)

"""Parallel Modified Independent Set coarsening with classical interpolation."""
struct PMISCoarsening <: CoarseningAlgorithm
    θ::Float64
    interpolation::InterpolationType
end
PMISCoarsening() = PMISCoarsening(0.25, DirectInterpolation())
PMISCoarsening(θ::Real) = PMISCoarsening(θ, DirectInterpolation())

"""Hybrid Modified Independent Set coarsening. Performs an RS first pass
(greedy bucket-based coarsening) followed by PMIS on remaining undecided
points, matching hypre's `hypre_BoomerAMGCoarsenHMIS`."""
struct HMISCoarsening <: CoarseningAlgorithm
    θ::Float64
    interpolation::InterpolationType
end
HMISCoarsening() = HMISCoarsening(0.25, DirectInterpolation())
HMISCoarsening(θ::Real) = HMISCoarsening(θ, DirectInterpolation())

"""Classical Ruge-Stüben (RS) coarsening with first/second pass to guarantee
good coarsening ratios and the strong-connection property.

Fields:
- `θ`: Strength threshold (default: 0.25)
- `interpolation`: Interpolation type (default: DirectInterpolation())
"""
struct RSCoarsening <: CoarseningAlgorithm
    θ::Float64
    interpolation::InterpolationType
end
RSCoarsening() = RSCoarsening(0.25, DirectInterpolation())
RSCoarsening(θ::Real) = RSCoarsening(θ, DirectInterpolation())

"""
    AggressiveCoarsening(θ=0.25; base=:pmis, interpolation=ExtendedIInterpolation())

Aggressive coarsening with configurable base algorithm.

In HYPRE, aggressive coarsening performs two passes of C/F splitting: the first
pass uses the base coarsening algorithm (HMIS or PMIS), and the second pass further
coarsens among C-points using distance-2 strong connections. The result is a much
coarser grid, requiring long-range interpolation (ext+i recommended).

Fields:
- `θ`: Strength threshold (default: 0.25)
- `base`: Base coarsening algorithm (`:pmis` or `:hmis`, default: `:pmis`)
- `interpolation`: Interpolation type for CF-based aggressive coarsening
  (default: `ExtendedIInterpolation()`). Only used when `base` is `:hmis` or `:pmis`.

When `base=:hmis`, this matches HYPRE's CoarsenType=10 + AggNumLevels>0.
"""
struct AggressiveCoarsening <: CoarseningAlgorithm
    θ::Float64
    base::Symbol
    interpolation::InterpolationType
end
AggressiveCoarsening() = AggressiveCoarsening(0.25, :pmis, ExtendedIInterpolation())
AggressiveCoarsening(θ::Real) = AggressiveCoarsening(θ, :pmis, ExtendedIInterpolation())
AggressiveCoarsening(θ::Real, base::Symbol) = AggressiveCoarsening(θ, base, ExtendedIInterpolation())

"""Smoothed aggregation coarsening. Builds a tentative prolongation from aggregation,
then smooths it with a damped Jacobi step: P = (I - ω D⁻¹ A) P_tent.

Fields:
- `θ`: Strength threshold (default: 0.25)
- `ω`: Damping factor for the Jacobi smoothing step (default: 2/3)
- `filtering`: If true, filter small entries from the smoothed P (default: false)
- `filter_tol`: Tolerance for filtering (default: 0.1)
"""
struct SmoothedAggregationCoarsening <: CoarseningAlgorithm
    θ::Float64
    ω::Float64
    filtering::Bool
    filter_tol::Float64
end
SmoothedAggregationCoarsening() = SmoothedAggregationCoarsening(0.25, 2.0/3.0, false, 0.1)
SmoothedAggregationCoarsening(θ::Real) = SmoothedAggregationCoarsening(θ, 2.0/3.0, false, 0.1)
SmoothedAggregationCoarsening(θ::Real, ω::Real) = SmoothedAggregationCoarsening(θ, ω, false, 0.1)

# ── Strength of connection type tags ──────────────────────────────────────────
abstract type StrengthType end

"""Default absolute-value strength: |a_{i,j}| ≥ θ * max_{k≠i} |a_{i,k}|."""
struct AbsoluteStrength <: StrengthType end

"""Sign-aware (classical RS) strength for non-M-matrices.
A connection (i,j) is strong if a_{i,j} has opposite sign from a_{i,i}
and |a_{i,j}| ≥ θ * max_{k: sign(a_{i,k})≠sign(a_{i,i})} |a_{i,k}|.
Positive off-diagonals (same sign as diagonal) are treated as weak."""
struct SignedStrength <: StrengthType end

# ── Smoother type tags ────────────────────────────────────────────────────────
abstract type SmootherType end
struct JacobiSmootherType <: SmootherType end
struct ColoredGaussSeidelType <: SmootherType end
struct SerialGaussSeidelType <: SmootherType end
struct SPAI0SmootherType <: SmootherType end
struct SPAI1SmootherType <: SmootherType end
struct L1JacobiSmootherType <: SmootherType end
struct L1ColoredGaussSeidelType <: SmootherType end
struct L1SerialGaussSeidelType <: SmootherType end
struct ChebyshevSmootherType <: SmootherType end
struct ILU0SmootherType <: SmootherType end
struct SerialILU0SmootherType <: SmootherType end
struct GPUILU0SmootherType <: SmootherType end
struct DILUSmootherType <: SmootherType end

# ── Abstract smoother ─────────────────────────────────────────────────────────
abstract type AbstractSmoother end

# ── Smoother types ────────────────────────────────────────────────────────────
"""
    JacobiSmoother{Tv, Tx, Tω, Vi, Vx}

Weighted Jacobi smoother.  `Tv` is the matrix entry type (e.g. `Float64` for
scalars, `SMatrix{B,B,T}` for block systems).  `Tx` is the solution-vector
element type (same as `Tv` for scalars, `SVector{B,T}` for block systems).
`Vi` and `Vx` are the matching AbstractVector subtypes (device-aware).
"""
mutable struct JacobiSmoother{Tv, Tx, Tω<:Real,
                               Vi<:AbstractVector{Tv},
                               Vx<:AbstractVector{Tx}} <: AbstractSmoother
    invdiag::Vi   # inv(diag(A)): element type Tv (SMatrix for block)
    tmp::Vx       # double-buffer for x: element type Tx (SVector for block)
    ω::Tω         # damping factor (always a real scalar)
end

"""
    ColoredGaussSeidelSmoother{Tv, Ti, V, Vi}

Parallel multicolor Gauss-Seidel smoother. Nodes are colored such that same-color
nodes have no direct connections, enabling parallel updates within each color.
The `color_order` and `invdiag` arrays are stored on the same device as the matrix.
The `color_offsets` are always on CPU since they are used for loop control.
"""
mutable struct ColoredGaussSeidelSmoother{Tv, Ti, V<:AbstractVector{Tv}, Vi<:AbstractVector{Ti}} <: AbstractSmoother
    colors::Vector{Ti}          # color[i] = color index for node i (CPU, used for setup only)
    color_offsets::Vector{Int}  # color_offsets[c]:color_offsets[c+1]-1 = nodes of color c (CPU)
    color_order::Vi             # nodes sorted by color (device)
    num_colors::Int
    invdiag::V                  # inverse diagonal (device)
end

"""
    L1ColoredGaussSeidelSmoother{Tv, Ti, V, Vi}

L1 variant of the parallel multicolor Gauss-Seidel smoother. Uses l1 row norms
for diagonal scaling instead of just the diagonal entry, providing more robust
smoothing for difficult problems. Same coloring and parallelization strategy as
`ColoredGaussSeidelSmoother`.
"""
mutable struct L1ColoredGaussSeidelSmoother{Tv, Ti, V<:AbstractVector{Tv}, Vi<:AbstractVector{Ti}} <: AbstractSmoother
    colors::Vector{Ti}          # color[i] = color index for node i (CPU, used for setup only)
    color_offsets::Vector{Int}  # color_offsets[c]:color_offsets[c+1]-1 = nodes of color c (CPU)
    color_order::Vi             # nodes sorted by color (device)
    num_colors::Int
    invdiag::V                  # 1 / l1_row_norm (device)
end

"""
    SerialGaussSeidelSmoother{Tv, Ti}

Serial (non-threaded, non-KA) Gauss-Seidel smoother. Performs a classic
sequential forward sweep over all rows. Does not require graph coloring,
threading, or KernelAbstractions.  Useful for small problems, debugging,
or environments where parallelism overhead exceeds the benefit.

All data is stored on CPU.
"""
mutable struct SerialGaussSeidelSmoother{Tv, Ti} <: AbstractSmoother
    invdiag::Vector{Tv}         # inverse diagonal (CPU)
    A_cpu::CSRMatrix{Tv, Ti}    # CPU copy of A for sequential access
end

"""
    L1SerialGaussSeidelSmoother{Tv, Ti}

Serial (non-threaded, non-KA) L1 Gauss-Seidel smoother matching hypre's default
l1-GS relaxation. Uses l1 row norms for diagonal scaling instead of just the
diagonal entry. Performs a sequential forward sweep over all rows.

All data is stored on CPU.
"""
mutable struct L1SerialGaussSeidelSmoother{Tv, Ti} <: AbstractSmoother
    invdiag::Vector{Tv}         # 1 / l1_row_norm (CPU)
    A_cpu::CSRMatrix{Tv, Ti}    # CPU copy of A for sequential access
end

"""
    SPAI0Smoother{Tv, Tx, Vm, Vx}

SPAI(0) smoother: diagonal sparse approximate inverse. M ≈ diag(A)⁻¹ where
M[i,i] = A[i,i] / (A[i,:] ⋅ A[:,i]).  This is the minimizer of ‖I - M*A‖_F
restricted to diagonal M.

`Tv` is the matrix/diagonal entry type; `Tx` is the solution-vector element type.
"""
mutable struct SPAI0Smoother{Tv, Tx,
                              Vm<:AbstractVector{Tv},
                              Vx<:AbstractVector{Tx}} <: AbstractSmoother
    m_diag::Vm        # diagonal of sparse approximate inverse (element type Tv)
    tmp::Vx           # workspace / double-buffer for x (element type Tx)
end

"""
    SPAI1Smoother{Tv, Ti, Vnz, Vx}

SPAI(1) smoother: sparse approximate inverse using the sparsity pattern of A.
For each row i, computes the least-squares optimal sparse vector m_i such that
‖e_i - A * m_i‖₂ is minimized subject to sparsity(m_i) ⊆ sparsity(A[i,:]).
The result is stored in CSR format matching A's sparsity.

`Tv` is the matrix entry type; `Tx` is the solution-vector element type.
The `nzval` array is on the same device as the matrix; `tmp` matches `x`/`b`.
"""
mutable struct SPAI1Smoother{Tv, Ti, Tx,
                              Vnz<:AbstractVector{Tv},
                              Vx<:AbstractVector{Tx}} <: AbstractSmoother
    nzval::Vnz        # nonzero values of the approximate inverse (same pattern as A)
    tmp::Vx           # workspace / double-buffer for residual (element type Tx)
end

"""
    L1JacobiSmoother{Tv, Tx, Tω, Vi, Vx}

l1-Jacobi smoother: uses l1 row norms for diagonal scaling instead of just the
diagonal entry.  More robust for matrices with large off-diagonal entries.
m[i] = ω / (|a_{i,i}| + Σ_{j≠i} |a_{i,j}|)

`Tv` is the matrix entry type; `Tx` is the solution-vector element type.
"""
mutable struct L1JacobiSmoother{Tv, Tx, Tω<:Real,
                                 Vi<:AbstractVector{Tv},
                                 Vx<:AbstractVector{Tx}} <: AbstractSmoother
    invdiag::Vi   # (1/l1_norm)*I: element type Tv (SMatrix for block)
    tmp::Vx       # double-buffer for x: element type Tx (SVector for block)
    ω::Tω         # damping factor (always a real scalar)
end

"""
    ChebyshevSmoother{Tv, Tx, Tλ, Vi, Vx}

Chebyshev polynomial smoother. Uses eigenvalue estimates to construct an optimal
polynomial iteration. Good for SPD problems. Does not require explicit diagonal info.

`Tv` is the matrix entry type; `Tx` is the solution-vector element type.
`invdiag` has element type `Tv`; `tmp1`/`tmp2` have element type `Tx`.
"""
mutable struct ChebyshevSmoother{Tv, Tx, Tλ<:Real,
                                  Vi<:AbstractVector{Tv},
                                  Vx<:AbstractVector{Tx}} <: AbstractSmoother
    invdiag::Vi       # inverse diagonal: element type Tv (SMatrix for block)
    tmp1::Vx          # workspace 1: element type Tx (SVector for block)
    tmp2::Vx          # workspace 2: element type Tx (SVector for block)
    λ_min::Tλ         # estimated min eigenvalue (always a real scalar)
    λ_max::Tλ         # estimated max eigenvalue (always a real scalar)
    degree::Int       # polynomial degree
end

"""
    ILU0Smoother{Tv, Ti, Tx}

Parallel ILU(0) smoother. Computes an incomplete LU factorization with the same
sparsity pattern as A, then applies forward/backward substitution using level
scheduling for parallelism.

`Tv` is the matrix/factorization entry type; `Tx` is the solution-vector element type.
The factorization data is always stored on CPU since ILU factorization and
triangular solves require sequential scalar indexing. The apply step copies
vectors to/from CPU as needed for GPU matrices.
"""
mutable struct ILU0Smoother{Tv, Ti, Tx} <: AbstractSmoother
    L_nzval::Vector{Tv}       # strictly lower triangle values (same pattern positions as A)
    U_nzval::Vector{Tv}       # upper triangle + diagonal values
    diag_idx::Vector{Ti}      # index of diagonal in each row's nzrange
    bwd_order::Vector{Ti}     # rows sorted by backward level
    level_offsets::Vector{Int} # [fwd_offsets..., bwd_offsets...] concatenated
    fwd_order::Vector{Ti}     # rows sorted by forward level
    num_fwd_levels::Int       # number of forward levels
    tmp::Vector{Tx}           # workspace: element type Tx (SVector for block)
    A_cpu::CSRMatrix{Tv, Ti}  # CPU copy of A's structure for sequential triangular solves
end

"""
    SerialILU0Smoother{Tv, Ti, Tx}

Serial ILU(0) smoother. Computes an incomplete LU factorization with the same
sparsity pattern as A, then applies plain sequential forward/backward
substitution without graph coloring or parallelism.

`Tv` is the matrix/factorization entry type; `Tx` is the solution-vector element type.
All data is stored on CPU. For GPU arrays, copies data to/from CPU as needed.
"""
mutable struct SerialILU0Smoother{Tv, Ti, Tx} <: AbstractSmoother
    L_nzval::Vector{Tv}       # strictly lower triangle values (same pattern positions as A)
    U_nzval::Vector{Tv}       # upper triangle + diagonal values
    diag_idx::Vector{Ti}      # index of diagonal in each row's nzrange
    tmp::Vector{Tx}           # workspace: element type Tx (SVector for block)
    A_cpu::CSRMatrix{Tv, Ti}  # CPU copy of A's structure for sequential triangular solves
end

"""
    GPUILU0Smoother{Tv, Ti, Tx, Vnz, Vi, Vx}

GPU-native level-scheduled ILU(0) smoother. Factorization and level scheduling
are computed on CPU during setup; all arrays are then transferred to the device.
The `smooth!` step executes entirely on device using KernelAbstractions kernels
(no CPU↔GPU copies per solve).

`Tv` is the matrix entry type; `Ti` the integer index type; `Tx` the solution
vector element type. `Vnz`, `Vi`, `Vx` are the concrete vector types that may
live on GPU.
"""
mutable struct GPUILU0Smoother{Tv, Ti, Tx,
                                Vnz<:AbstractVector{Tv},
                                Vi<:AbstractVector{Ti},
                                Vx<:AbstractVector{Tx}} <: AbstractSmoother
    L_nzval::Vnz              # strictly lower triangle values
    U_nzval::Vnz              # upper triangle + diagonal values
    diag_idx::Vi              # index of diagonal in each row's nzrange
    fwd_order::Vi             # rows sorted by forward level
    bwd_order::Vi             # rows sorted by backward level
    level_offsets::Vector{Int}# [fwd_offsets..., bwd_offsets...] concatenated (CPU)
    num_fwd_levels::Int       # number of forward levels
    tmp::Vx                   # workspace on device
    row_norms::Vnz            # precomputed row norms on device (for safeguarding during factorization)
end

"""
    DILUSmoother{Tv, Ti, Tx, Vnz, Vi, Vx}

GPU-native diagonal ILU (DILU) smoother. Only the modified diagonal is stored
(not the full L/U factors), making this more memory-efficient than ILU(0).
Level scheduling and diagonal computation happen on CPU during setup; all
arrays are then transferred to the device. The `smooth!` step executes
entirely on device using KernelAbstractions kernels.

The DILU factorization defines:
    d_i = a_{ii} - Σ_{j<i, (i,j)∈S} a_{ij} * d_j⁻¹ * a_{ji}

and the preconditioner is M = (D + L) D⁻¹ (D + U) where D = diag(d_i),
L is the strict lower triangle of A, and U is the strict upper triangle.
"""
mutable struct DILUSmoother{Tv, Ti, Tx,
                             Vnz<:AbstractVector{Tv},
                             Vi<:AbstractVector{Ti},
                             Vx<:AbstractVector{Tx}} <: AbstractSmoother
    inv_diag::Vnz             # d_i⁻¹ (inverted DILU diagonal)
    diag_idx::Vi              # index of diagonal in each row's nzrange
    fwd_order::Vi             # rows sorted by forward level
    bwd_order::Vi             # rows sorted by backward level
    level_offsets::Vector{Int}# [fwd_offsets..., bwd_offsets...] concatenated (CPU)
    num_fwd_levels::Int       # number of forward levels
    tmp::Vx                   # workspace on device
    lower_transpose_nz::Vi    # for each lower-triangle nz (i,j), the nz-index of (j,i)
end

# ── Prolongation info (stored implicitly) ─────────────────────────────────────
"""
    ProlongationOp{Ti, Tv, Vi, Vv}

Stores the prolongation operator implicitly.
- `rowptr`, `colval`, `nzval` define the sparse P in CSR layout.
- `nrow` and `ncol` are the dimensions (n_fine × n_coarse).
- `trunc_scaling`: optional per-entry scaling factors from truncation with
  rescaling enabled. When not `nothing`, entry `k` of `nzval` was multiplied
  by `trunc_scaling[k]` during setup. During resetup the same factor is
  reapplied so that truncated elements need not be recomputed.
Vector types are parameterized to support GPU arrays.
"""
mutable struct ProlongationOp{Ti<:Integer, Tv, Vi<:AbstractVector{Ti}, Vv<:AbstractVector{Tv}}
    rowptr::Vi
    colval::Vi
    nzval::Vv
    nrow::Int
    ncol::Int
    trunc_scaling::Union{Nothing, Vv}
end

# Convenience constructor for CPU vectors
# Convenience constructor for CPU vectors (used during setup which runs on CPU)
function ProlongationOp{Ti, Tv}(rowptr::Vector{Ti}, colval::Vector{Ti}, nzval::Vector{Tv}, nrow::Int, ncol::Int) where {Ti, Tv}
    return ProlongationOp{Ti, Tv, Vector{Ti}, Vector{Tv}}(rowptr, colval, nzval, nrow, ncol, nothing)
end

"""
    TransposeMap{Ti, Vi}

Pre-computed transpose structure for P, mapping coarse rows to fine rows.
Enables atomic-free restriction (P^T * r) by parallelizing over coarse rows.

- `offsets[J]` to `offsets[J+1]-1` gives the range of fine rows i that have
  P[i, J] != 0 (i.e., fine rows that interpolate from coarse column J).
- `fine_rows[k]` is the fine row index i where P[i, J] is nonzero.
- `p_nz_idx[k]` is the index into P.nzval for the weight P[fine_rows[k], J].
"""
struct TransposeMap{Ti<:Integer, Vi<:AbstractVector{Ti}}
    offsets::Vi    # n_coarse + 1 entries
    fine_rows::Vi  # which fine rows map to each coarse row
    p_nz_idx::Vi   # index into P.nzval for the weight
end

"""
    RestrictionMap{Ti, Vi, Vt}

Maps the Galerkin product triples to coarse matrix nonzero entries for in-place
computation during resetup. Triples are grouped by their destination coarse NZ
index so that `galerkin_product!` can parallelize over coarse NZ entries (one
thread per output entry) without atomics.

- `nz_offsets[k]` to `nz_offsets[k+1]-1` gives the range of triples that
  contribute to coarse NZ entry `k`.
- Each triple `t` is a 3-tuple `(pi_idx, anz_idx, pj_idx)` representing the
  contribution:
  `P.nzval[pi_idx] * A.nzval[anz_idx] * P.nzval[pj_idx]`
"""
struct RestrictionMap{Ti<:Integer, Vi<:AbstractVector{Ti}, Vt<:AbstractVector{NTuple{3,Ti}}}
    nz_offsets::Vi        # offset array: nnz_c + 1 entries
    triples::Vt           # (pi_idx, anz_idx, pj_idx) tuples sorted by dest NZ
end

"""
    ProlongationUpdateMap

Stores precomputed index mappings for efficient in-place update of prolongation
operator values during resetup with `update_P=true`. All graph structure
decisions (strength, CF-split, etc.) are fixed at setup time.

## Design Philosophy

For Direct interpolation: P[i,c] = -A[i,c] / d_i where d_i = diagonal + weak sum
- Simple linear formula with fixed coefficient 1.0 on numerator A entry

For Standard/Extended+i interpolation: The formula involves indirect contributions
where weights themselves depend on A values. These use a more complex structure
that stores the full graph connectivity to enable recomputation.

## Fields

- `interp_type`: 1=Direct, 2=Standard, 3=Extended+i
- `is_strong`: Boolean array marking strong connections in A
- `cf`: Coarse/fine split (cf[i]=1 for coarse, -1 for fine)
- `coarse_map`: Maps fine indices to coarse indices for coarse points
- `diag_nz_idx`: Diagonal A.nzval index for each row

Per-entry formula data:
- `entry_type`: 0=coarse point (P=1), 1=Direct formula, 2=Standard, 3=Extended+i
- `numer_idx`: A.nzval index for numerator term (0 for coarse)
- `denom_offsets`, `denom_entries`: A.nzval indices for denominator sum

Strong neighbor structure (for Standard/Extended+i row recomputation):
- `strong_nbrs_offsets`: CSR offset array (n_fine + 1)
- `strong_nbrs_cols`: column indices of strong neighbors
- `strong_nbrs_nz`: A.nzval indices of strong neighbors

Workspace for Standard/Extended+i (to avoid allocations during resetup):
- `P_marker`: Scratch array for tracking visited nodes
- `chat_indices`: Reusable buffer for C-hat indices
- `P_data`: Reusable buffer for P values
"""
mutable struct ProlongationUpdateMap{Ti<:Integer, Tv<:Number}
    interp_type::Int                    # 1=Direct, 2=Standard, 3=Extended+i
    is_strong::Vector{Bool}             # strong connection mask (nnz_A)
    cf::Vector{Int}                     # coarse/fine split (n_fine)
    coarse_map::Vector{Int}             # fine-to-coarse mapping (n_fine)
    diag_nz_idx::Vector{Ti}             # diagonal A.nzval index for each row
    # Per-entry formula data (used by Direct interpolation kernel)
    entry_type::Vector{Ti}              # 0=coarse (P=1), 1+=compute formula
    numer_idx::Vector{Ti}               # A.nzval index for numerator
    denom_offsets::Vector{Ti}           # offset array for denominator
    denom_entries::Vector{Ti}           # A.nzval indices for denominator
    # Direct interpolation alfa/beta data: per P-entry, stores the A.nzval indices
    # needed to compute sum_N_neg/pos and sum_P_neg/pos for the alfa/beta formula.
    dir_diag_idx::AbstractVector{Ti}        # per P-entry: diagonal A.nzval index
    dir_all_offsets::AbstractVector{Ti}     # per P-entry: offset into all off-diag indices
    dir_all_entries::AbstractVector{Ti}     # all off-diagonal A.nzval indices for row
    dir_sc_offsets::AbstractVector{Ti}      # per P-entry: offset into strong-C indices
    dir_sc_entries::AbstractVector{Ti}      # strong C-neighbor A.nzval indices for row
    # Strong neighbor structure for Standard/Extended+i row recomputation
    strong_nbrs_offsets::Vector{Ti}     # offset array (n_fine + 1)
    strong_nbrs_cols::Vector{Ti}        # column indices of strong neighbors
    strong_nbrs_nz::Vector{Ti}          # A.nzval indices of strong neighbors
    # Workspace for Standard/Extended+i (to avoid allocations during resetup)
    P_marker::Vector{Int}               # marker array for C-hat tracking
    chat_indices::Vector{Int}           # reusable buffer for C-hat indices
    P_data::Vector{Tv}                  # reusable buffer for P values
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GPU kernel data for Standard interpolation (interp_type == 2)
    # ═══════════════════════════════════════════════════════════════════════════
    # Per P entry k, the formula is:
    #   P[k] = -(direct_contrib + Σ indirect_contribs) / d_i
    # 
    # Direct contribution: A[direct_numer_idx[k]] (or 0 if no direct connection)
    # Indirect contributions through fine neighbors:
    #   Each fine neighbor contributes: A[a_ik] * A[a_kJ] / sum_C_k
    #   where sum_C_k = Σ A[sum_indices] (with sign check based on diag_k sign)
    # Denominator d_i = Σ A[d_base_indices] + Σ (A[a_ik] * A[a_ki] / sum_C_k)
    #
    # Data layout (CSR-like):
    # - std_direct_numer_idx[k]: A.nzval index for direct a_{i,J} (0=none)
    # - std_fine_offsets[k]: offset into fine neighbor data
    # - For each fine neighbor j (from std_fine_offsets[k] to std_fine_offsets[k+1]-1):
    #   - std_a_ik[j]: A.nzval index for a_{i,fine_j}
    #   - std_a_kJ[j]: A.nzval index for a_{fine_j, coarse_J}
    #   - std_diag_k[j]: A.nzval index for diagonal of fine neighbor
    #   - std_a_ki[j]: A.nzval index for a_{fine_j, i} (for d_i contrib, 0=none)
    #   - std_sum_offsets[j]: offset into sum_C_k indices
    #   - std_sum_indices[...]: A.nzval indices for computing sum_C_k
    # - std_d_base_offsets[k]: offset into base denominator indices
    # - std_d_base_entries[...]: A.nzval indices for a_{i,i} + weak neighbors
    std_direct_numer_idx::AbstractVector{Ti}
    std_fine_offsets::AbstractVector{Ti}
    std_a_ik::AbstractVector{Ti}
    std_a_kJ::AbstractVector{Ti}
    std_diag_k::AbstractVector{Ti}
    std_a_ki::AbstractVector{Ti}
    std_sum_offsets::AbstractVector{Ti}
    std_sum_indices::AbstractVector{Ti}
    std_d_base_offsets::AbstractVector{Ti}
    std_d_base_entries::AbstractVector{Ti}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Kernel data for Extended+i interpolation (interp_type == 3)
    # ═══════════════════════════════════════════════════════════════════════════
    # Similar structure but C-hat includes distance-2 coarse points.
    # For each P entry k at row i to coarse column J:
    # - extd_entry_row[k]: row index i (needed for C-hat computation)
    # - extd_p_col[k]: P column index J (what we're interpolating to)
    # - For each C-hat point contributing to this P entry:
    #   - extd_chat_offsets[k]: offset into C-hat data
    #   - extd_chat_cols[...]: fine node indices c where coarse_map[c] = J
    # - For fine neighbors (contribute indirectly):
    #   - extd_fine_offsets[k]: offset into fine neighbor contribution data
    #   - extd_fine_data[...]: (similar structure to Standard)
    extd_entry_row::AbstractVector{Ti}
    extd_p_col::AbstractVector{Ti}
    extd_direct_a_idx::AbstractVector{Ti}       # A.nzval index for direct contribution
    extd_fine_offsets::AbstractVector{Ti}
    extd_a_ik::AbstractVector{Ti}
    extd_diag_k::AbstractVector{Ti}
    extd_sum_offsets::AbstractVector{Ti}
    extd_sum_indices::AbstractVector{Ti}
    extd_contrib_offsets::AbstractVector{Ti}
    extd_contrib_a_idx::AbstractVector{Ti}      # A.nzval indices that contribute to P entry
    extd_contrib_p_col::AbstractVector{Ti}      # P column for each contribution
    extd_d_base_offsets::AbstractVector{Ti}
    extd_d_base_entries::AbstractVector{Ti}
end

# ── AMG Level ─────────────────────────────────────────────────────────────────
"""
    AMGLevel{Tv, Ti}

One level of the AMG hierarchy. The matrix `A` is stored internally as a
`CSRMatrix` (raw CSR vectors). Conversion from external sparse CSR formats
happens at the API boundary in `amg_setup` and `amg_resetup!`.

Workspace vectors (`r`, `xc`, `bc`) are allocated on the same device as the
matrix arrays to avoid host/device memory mixing in GPU kernels.

When `allow_partial_resetup=true` and using CF-splitting based coarsening,
the `P_update_map` field stores the coarse-fine split and mapping data
needed for in-place P value update with `update_P=true`.
"""
mutable struct AMGLevel{Tv, Ti<:Integer}
    A::CSRMatrix{Tv, Ti}
    P::ProlongationOp
    Pt_map::TransposeMap
    R_map::Union{Nothing, RestrictionMap}
    smoother::AbstractSmoother
    r::AbstractVector{Tv}      # residual workspace
    xc::AbstractVector{Tv}     # coarse solution workspace
    bc::AbstractVector{Tv}     # coarse RHS workspace
    P_update_map::Union{Nothing, ProlongationUpdateMap}  # for update_P=true resetup
end

# ── AMG Hierarchy ─────────────────────────────────────────────────────────────
"""
    AMGHierarchy{Tv, Ti}

Complete AMG hierarchy with multiple levels and a direct solver at the coarsest level.
The coarse LU factorization uses high-level `lu` / `lu!` so that GPU backends
(CUDA, Metal) can dispatch to their own implementations.

The coarsest-level workspace (`coarse_x`, `coarse_b`) lives on the same device as
`coarse_A`. When `coarse_solve_on_cpu` is `true`, these are always on CPU.
Level workspace and smoother arrays are allocated on the same device as the input matrix.

The `backend` and `block_size` are stored in the hierarchy so that cycle/solve/resetup
functions automatically use the correct backend without requiring explicit kwargs.
"""
mutable struct AMGHierarchy{Tv, Ti<:Integer}
    levels::Vector{AMGLevel{Tv, Ti}}
    coarse_A::AbstractMatrix{Tv}       # dense coarse matrix (values recomputed each resetup)
    coarse_factor::Factorization{Tv}   # LU (or other) factorization of coarse_A
    coarse_x::AbstractVector{Tv}       # workspace for coarsest level direct solve
    coarse_b::AbstractVector{Tv}       # workspace for coarsest level direct solve
    solve_r::AbstractVector{Tv}        # residual buffer for amg_solve! (finest level size)
    backend::Any               # KernelAbstractions backend (CPU, CUDABackend, etc.)
    block_size::Int            # block size for KA kernel launches
    coarse_solve_on_cpu::Bool  # if true, coarse LU solve is always on CPU
    galerkin_workspace::Any    # GalerkinWorkspace, reused across setup/resetup calls
    setup_workspace::Any       # SetupWorkspace, reused across setup/resetup calls
end

"""
    SetupWorkspace{Tv, Ti}

Pre-allocated workspace for coarsening and prolongation building. Stored in the
hierarchy and reused across setup/resetup calls to avoid repeated allocations in
hot loops. Arrays are `resize!`'d as needed (they only grow, never shrink).
"""
mutable struct SetupWorkspace{Tv, Ti<:Integer}
    # Coarsening workspace (size n per level)
    cf::Vector{Int}
    coarse_map::Vector{Int}
    measure::Vector{Float64}
    st_count::Vector{Int}
    # _build_strong_transpose_adj workspace
    counts::Vector{Int}
    offsets::Vector{Int}
    sources::Vector{Int}
    pos::Vector{Int}
    # Bucket sort workspace (RS / HMIS first pass)
    bucket_head::Vector{Int}
    bucket_tail::Vector{Int}
    bucket_next::Vector{Int}
    bucket_prev::Vector{Int}
    # COO accumulation workspace for prolongation building
    I_p::Vector{Ti}
    J_p::Vector{Ti}
    V_p::Vector{Tv}
    # Extended interpolation workspace
    P_marker::Vector{Int}
    strong_nbrs_offsets::Vector{Int}
    strong_nbrs_data::Vector{Int}
    # Strength graph buffer (reused across calls)
    is_strong::Vector{Bool}
    # Sort permutation buffer (reused as counting-sort position buffer)
    sort_perm::Vector{Int}
    # Old ProlongationOp reference for array reuse during resetup (set per-level in _build_levels!)
    old_P::Any   # Union{Nothing, ProlongationOp}
    # Old ProlongationUpdateMap reference for extd array reuse during resetup (set per-level in _build_levels!)
    old_P_update_map::Any   # Union{Nothing, ProlongationUpdateMap}
end

function SetupWorkspace{Tv, Ti}() where {Tv, Ti}
    SetupWorkspace{Tv, Ti}(
        Int[], Int[], Float64[], Int[],
        Int[], Int[], Int[], Int[],
        Int[], Int[], Int[], Int[],
        Ti[], Ti[], Tv[],
        Int[], Int[], Int[],
        Bool[], Int[],
        nothing,
        nothing,
    )
end

# ── AMG Configuration ─────────────────────────────────────────────────────────
"""
    AMGConfig

Configuration for AMG setup.

Fields:
- `coarsening`: Main coarsening algorithm used at each level (default: `HMISCoarsening(0.25, ExtendedIInterpolation())`)
- `smoother`: Smoother type (default: `JacobiSmootherType()`)
- `max_levels`, `max_coarse_size`: Hierarchy limits
- `pre_smoothing_steps`, `post_smoothing_steps`: Smoothing counts
- `jacobi_omega`: Damping factor for Jacobi smoother
- `verbose`: Verbosity level as an integer:
  - 0: Silent
  - 1: Print hierarchy summary after setup and convergence summary after solve
  - 2: Additionally print iteration counter and residual norm at each cycle during solve
- `initial_coarsening`: Optional alternative coarsening for the first N levels (defaults to `coarsening`)
- `initial_coarsening_levels`: Number of levels to use `initial_coarsening` for (default: 0)
- `max_row_sum`: Maximum row sum threshold for dependency weakening (default: 1.0, disabled;
  HYPRE defaults to 0.9). When < 1.0, rows where |row_sum| > |a_ii| * max_row_sum have all
  off-diagonal entries zeroed out (all dependencies made weak), matching the hypre definition.
- `cycle_type`: AMG cycle type, `:V` for V-cycle or `:W` for W-cycle (default: `:V`)
- `strength_type`: Strength of connection algorithm (default: `SignedStrength()`).
  Matches hypre's default signed strength, which only marks opposite-sign off-diagonals
  as strong. Use `AbsoluteStrength()` for sign-agnostic strength based on magnitudes.
- `coarse_solve_on_cpu`: If `true`, the coarsest-level LU factorization and direct
  solve are performed on CPU even when using a GPU backend. Required for backends
  that do not support `lu` on device (e.g., Apple Metal). Default: `false`.
- `allow_partial_resetup`: If `true` (the default), restriction maps are built
  during setup so that `amg_resetup!(…; partial=true)` can update values in-place
  without re-coarsening. Set to `false` for a faster initial setup when only full
  resetup will be used.
- `reverse_post_smooth`: If `true` (the default), post-smoothing uses the backward
  sweep direction (rows n→1 for serial GS, reversed color order for colored GS).
  This matches HYPRE's default of forward pre-smoothing and backward post-smoothing,
  producing a more effective AMG preconditioner. Set to `false` to use the same
  forward direction for both pre- and post-smoothing.
"""
struct AMGConfig
    coarsening::CoarseningAlgorithm
    smoother::SmootherType
    max_levels::Int
    max_coarse_size::Int
    pre_smoothing_steps::Int
    post_smoothing_steps::Int
    jacobi_omega::Float64
    verbose::Int
    initial_coarsening::CoarseningAlgorithm
    initial_coarsening_levels::Int
    max_row_sum::Float64
    cycle_type::Symbol
    strength_type::StrengthType
    coarse_solve_on_cpu::Bool
    allow_partial_resetup::Bool
    reverse_post_smooth::Bool
end

function AMGConfig(;
    coarsening::CoarseningAlgorithm = HMISCoarsening(0.25, ExtendedIInterpolation()),
    smoother::SmootherType = L1ColoredGaussSeidelType(),
    max_levels::Int = 20,
    max_coarse_size::Int = 50,
    pre_smoothing_steps::Int = 1,
    post_smoothing_steps::Int = 1,
    jacobi_omega::Float64 = 2.0/3.0,
    verbose::Union{Bool, Int} = 0,
    initial_coarsening::CoarseningAlgorithm = coarsening,
    initial_coarsening_levels::Int = 0,
    max_row_sum::Float64 = 1.0,
    cycle_type::Symbol = :V,
    strength_type::StrengthType = SignedStrength(),
    coarse_solve_on_cpu::Bool = false,
    allow_partial_resetup::Bool = true,
    reverse_post_smooth::Bool = true,
)
    @assert cycle_type in (:V, :W) "cycle_type must be :V or :W"
    verbose_int = verbose isa Bool ? Int(verbose) : verbose
    return AMGConfig(coarsening, smoother, max_levels, max_coarse_size,
                     pre_smoothing_steps, post_smoothing_steps, jacobi_omega, verbose_int,
                     initial_coarsening, initial_coarsening_levels,
                     max_row_sum, cycle_type, strength_type, coarse_solve_on_cpu,
                     allow_partial_resetup, reverse_post_smooth)
end

"""
    _get_coarsening_for_level(config, lvl)

Return the coarsening algorithm to use at level `lvl`, accounting for
the `initial_coarsening` / `initial_coarsening_levels` configuration.
"""
function _get_coarsening_for_level(config::AMGConfig, lvl::Int)
    if lvl <= config.initial_coarsening_levels
        return config.initial_coarsening
    end
    return config.coarsening
end

"""
    hypre_default_config(; kwargs...)

Create an AMGConfig matching a typical HYPRE BoomerAMG setup for challenging 3D problems:

    CoarsenType = 10       → HMIS coarsening
    StrongThreshold = 0.5  → θ = 0.5
    AggNumLevels = 1       → Aggressive coarsening for first level
    AggTruncFactor = 0.3   → Truncation factor for interpolation weights
    InterpType = 6         → Extended+i interpolation

The resulting config uses:
- `HMISCoarsening(0.5, ExtendedIInterpolation(0.3))` as main coarsening
- `AggressiveCoarsening(0.5, :hmis, ExtendedIInterpolation(0.3))` for the first level
- `initial_coarsening_levels = 1`

Additional keyword arguments are forwarded to `AMGConfig`.
"""
function hypre_default_config(;
    θ::Float64 = 0.5,
    agg_num_levels::Int = 1,
    agg_trunc_factor::Float64 = 0.3,
    kwargs...
)
    interp = ExtendedIInterpolation(agg_trunc_factor)
    main_coarsening = HMISCoarsening(θ, interp)
    agg_coarsening = AggressiveCoarsening(θ, :hmis, interp)
    return AMGConfig(;
        coarsening = main_coarsening,
        initial_coarsening = agg_coarsening,
        initial_coarsening_levels = agg_num_levels,
        kwargs...
    )
end
