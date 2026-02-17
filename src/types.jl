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
    ExtendedIInterpolation(; trunc_factor=0.0, max_elements=4)

Extended+i interpolation (HYPRE InterpType=6): extends standard by including
distance-2 coarse points (coarse points reachable through strong fine neighbors).
Recommended for use with HMIS coarsening for challenging 3D problems.
`trunc_factor`: entries with |w| < trunc_factor * max|w| per row are dropped
(0 = no truncation). Maps to HYPRE's `AggTruncFactor`.
`max_elements`: maximum number of interpolation entries per row (0 = no limit).
When the number of entries exceeds this limit, only the strongest entries are kept.
Default: 0.
"""
struct ExtendedIInterpolation <: InterpolationType
    trunc_factor::Float64
    max_elements::Int
end
ExtendedIInterpolation() = ExtendedIInterpolation(0.0, 0)
ExtendedIInterpolation(trunc_factor::Real) = ExtendedIInterpolation(Float64(trunc_factor), 0)

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

"""Hybrid Modified Independent Set coarsening. Uses the symmetrized strength graph
(intersection of S and S^T) for independent set selection, producing generally
less aggressive coarsening than PMIS."""
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
struct ChebyshevSmootherType <: SmootherType end
struct ILU0SmootherType <: SmootherType end

# ── Abstract smoother ─────────────────────────────────────────────────────────
abstract type AbstractSmoother end

# ── Smoother types ────────────────────────────────────────────────────────────
"""
    JacobiSmoother{Tv, V}

Weighted Jacobi smoother.  Stores the inverse diagonal and a workspace vector.
Vector type `V` matches the device (CPU `Vector` or GPU array type).
"""
mutable struct JacobiSmoother{Tv, V<:AbstractVector{Tv}} <: AbstractSmoother
    invdiag::V
    tmp::V
    ω::Tv      # damping factor
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
    SPAI0Smoother{Tv, V}

SPAI(0) smoother: diagonal sparse approximate inverse. M ≈ diag(A)⁻¹ where
M[i,i] = A[i,i] / (A[i,:] ⋅ A[:,i]).  This is the minimizer of ‖I - M*A‖_F
restricted to diagonal M.
"""
mutable struct SPAI0Smoother{Tv, V<:AbstractVector{Tv}} <: AbstractSmoother
    m_diag::V         # diagonal of sparse approximate inverse
    tmp::V            # workspace
end

"""
    SPAI1Smoother{Tv, Ti}

SPAI(1) smoother: sparse approximate inverse using the sparsity pattern of A.
For each row i, computes the least-squares optimal sparse vector m_i such that
‖e_i - A * m_i‖₂ is minimized subject to sparsity(m_i) ⊆ sparsity(A[i,:]).
The result is stored in CSR format matching A's sparsity.

The `nzval` and `tmp` arrays are stored on the same device as the matrix so
that the apply kernels can run on GPU without host/device mixing.
"""
mutable struct SPAI1Smoother{Tv, Ti, V<:AbstractVector{Tv}} <: AbstractSmoother
    nzval::V          # nonzero values of the approximate inverse (same pattern as A)
    tmp::V            # workspace
end

"""
    L1JacobiSmoother{Tv, V}

l1-Jacobi smoother: uses l1 row norms for diagonal scaling instead of just the
diagonal entry.  More robust for matrices with large off-diagonal entries.
m[i] = ω / (|a_{i,i}| + Σ_{j≠i} |a_{i,j}|)
"""
mutable struct L1JacobiSmoother{Tv, V<:AbstractVector{Tv}} <: AbstractSmoother
    invdiag::V        # 1 / l1_row_norm
    tmp::V
    ω::Tv
end

"""
    ChebyshevSmoother{Tv, V}

Chebyshev polynomial smoother. Uses eigenvalue estimates to construct an optimal
polynomial iteration. Good for SPD problems. Does not require explicit diagonal info.
"""
mutable struct ChebyshevSmoother{Tv, V<:AbstractVector{Tv}} <: AbstractSmoother
    invdiag::V       # inverse diagonal (for preconditioning)
    tmp1::V
    tmp2::V
    λ_min::Tv                 # estimated min eigenvalue
    λ_max::Tv                 # estimated max eigenvalue
    degree::Int               # polynomial degree
end

"""
    ILU0Smoother{Tv, Ti}

Parallel ILU(0) smoother. Computes an incomplete LU factorization with the same
sparsity pattern as A, then applies forward/backward substitution using graph
coloring for parallelism.

The factorization data is always stored on CPU since ILU factorization and
triangular solves require sequential scalar indexing. The apply step copies
vectors to/from CPU as needed for GPU matrices.
"""
mutable struct ILU0Smoother{Tv, Ti} <: AbstractSmoother
    L_nzval::Vector{Tv}       # strictly lower triangle values (same pattern positions as A)
    U_nzval::Vector{Tv}       # upper triangle + diagonal values
    diag_idx::Vector{Ti}      # index of diagonal in each row's nzrange
    colors::Vector{Ti}
    color_offsets::Vector{Int}
    color_order::Vector{Ti}
    num_colors::Int
    tmp::Vector{Tv}
    A_cpu::CSRMatrix{Tv, Ti}  # CPU copy of A's structure for sequential triangular solves
end

# ── Prolongation info (stored implicitly) ─────────────────────────────────────
"""
    ProlongationOp{Ti, Tv, Vi, Vv}

Stores the prolongation operator implicitly.
- `rowptr`, `colval`, `nzval` define the sparse P in CSR layout.
- `nrow` and `ncol` are the dimensions (n_fine × n_coarse).
Vector types are parameterized to support GPU arrays.
"""
mutable struct ProlongationOp{Ti<:Integer, Tv, Vi<:AbstractVector{Ti}, Vv<:AbstractVector{Tv}}
    rowptr::Vi
    colval::Vi
    nzval::Vv
    nrow::Int
    ncol::Int
end

# Convenience constructor for CPU vectors
# Convenience constructor for CPU vectors (used during setup which runs on CPU)
function ProlongationOp{Ti, Tv}(rowptr::Vector{Ti}, colval::Vector{Ti}, nzval::Vector{Tv}, nrow::Int, ncol::Int) where {Ti, Tv}
    return ProlongationOp{Ti, Tv, Vector{Ti}, Vector{Tv}}(rowptr, colval, nzval, nrow, ncol)
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
    RestrictionMap{Ti, Vi}

Maps the Galerkin product triples to coarse matrix nonzero entries for in-place
computation during resetup. Triples are grouped by their destination coarse NZ
index so that `galerkin_product!` can parallelize over coarse NZ entries (one
thread per output entry) without atomics.

- `nz_offsets[k]` to `nz_offsets[k+1]-1` gives the range of triples that
  contribute to coarse NZ entry `k`.
- Each triple `t` represents the contribution:
  `P.nzval[triple_pi_idx[t]] * A.nzval[triple_anz_idx[t]] * P.nzval[triple_pj_idx[t]]`
"""
struct RestrictionMap{Ti<:Integer, Vi<:AbstractVector{Ti}}
    nz_offsets::Vi        # offset array: nnz_c + 1 entries
    triple_pi_idx::Vi     # P.nzval index for p_i weight (sorted by dest NZ)
    triple_anz_idx::Vi    # A.nzval index for a_ij value (sorted by dest NZ)
    triple_pj_idx::Vi     # P.nzval index for p_j weight (sorted by dest NZ)
end

# ── AMG Level ─────────────────────────────────────────────────────────────────
"""
    AMGLevel{Tv, Ti}

One level of the AMG hierarchy. The matrix `A` is stored internally as a
`CSRMatrix` (raw CSR vectors). Conversion from external sparse CSR formats
happens at the API boundary in `amg_setup` and `amg_resetup!`.

Workspace vectors (`r`, `xc`, `bc`) are allocated on the same device as the
matrix arrays to avoid host/device memory mixing in GPU kernels.
"""
mutable struct AMGLevel{Tv, Ti<:Integer}
    A::CSRMatrix{Tv, Ti}
    P::ProlongationOp
    Pt_map::TransposeMap
    R_map::RestrictionMap
    smoother::AbstractSmoother
    r::AbstractVector{Tv}      # residual workspace
    xc::AbstractVector{Tv}     # coarse solution workspace
    bc::AbstractVector{Tv}     # coarse RHS workspace
end

# ── AMG Hierarchy ─────────────────────────────────────────────────────────────
"""
    AMGHierarchy{Tv, Ti}

Complete AMG hierarchy with multiple levels and a direct solver at the coarsest level.
The coarse LU factorization uses high-level `lu` / `lu!` so that GPU backends
(CUDA, Metal) can dispatch to their own implementations.

The coarsest-level workspace (`coarse_x`, `coarse_b`) is always on CPU since
LU direct solves use LAPACK. Level workspace and smoother arrays are allocated
on the same device as the input matrix.

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
end

# ── AMG Configuration ─────────────────────────────────────────────────────────
"""
    AMGConfig

Configuration for AMG setup.

Fields:
- `coarsening`: Main coarsening algorithm used at each level (default: `AggregationCoarsening()`)
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
- `max_row_sum`: Maximum row sum threshold for dependency weakening (default: 1.0, disabled).
  When < 1.0, rows where |row_sum| > |a_ii| * max_row_sum have all off-diagonal entries
  zeroed out (all dependencies made weak), matching the hypre definition.
- `cycle_type`: AMG cycle type, `:V` for V-cycle or `:W` for W-cycle (default: `:V`)
- `strength_type`: Strength of connection algorithm (default: `AbsoluteStrength()`).
  Use `SignedStrength()` for non-M-matrices with positive off-diagonals.
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
end

function AMGConfig(;
    coarsening::CoarseningAlgorithm = HMISCoarsening(0.5, ExtendedIInterpolation()),
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
    strength_type::StrengthType = AbsoluteStrength(),
)
    @assert cycle_type in (:V, :W) "cycle_type must be :V or :W"
    verbose_int = verbose isa Bool ? Int(verbose) : verbose
    return AMGConfig(coarsening, smoother, max_levels, max_coarse_size,
                     pre_smoothing_steps, post_smoothing_steps, jacobi_omega, verbose_int,
                     initial_coarsening, initial_coarsening_levels,
                     max_row_sum, cycle_type, strength_type)
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
