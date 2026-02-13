# ── Interpolation type tags ───────────────────────────────────────────────────
abstract type InterpolationType end

"""Direct interpolation: interpolate only from directly connected coarse points."""
struct DirectInterpolation <: InterpolationType end

"""Standard (classical Ruge-Stüben) interpolation: includes indirect contributions
through strong fine neighbors."""
struct StandardInterpolation <: InterpolationType end

"""Extended+i interpolation: extends standard by including distance-2 coarse points."""
struct ExtendedIInterpolation <: InterpolationType end

# ── Coarsening type tags ──────────────────────────────────────────────────────
abstract type CoarseningAlgorithm end

"""Greedy aggregation coarsening."""
struct AggregationCoarsening <: CoarseningAlgorithm
    θ::Float64   # strength threshold
end
AggregationCoarsening() = AggregationCoarsening(0.25)

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

"""Aggressive coarsening (two passes of PMIS-based coarsening)."""
struct AggressiveCoarsening <: CoarseningAlgorithm
    θ::Float64
end
AggressiveCoarsening() = AggressiveCoarsening(0.25)

# ── Smoother type tags ────────────────────────────────────────────────────────
abstract type SmootherType end
struct JacobiSmootherType <: SmootherType end
struct ColoredGaussSeidelType <: SmootherType end
struct SPAI0SmootherType <: SmootherType end
struct SPAI1SmootherType <: SmootherType end

# ── Abstract smoother ─────────────────────────────────────────────────────────
abstract type AbstractSmoother end

# ── Smoother types ────────────────────────────────────────────────────────────
"""
    JacobiSmoother{Tv}

Weighted Jacobi smoother.  Stores the inverse diagonal and a workspace vector.
"""
mutable struct JacobiSmoother{Tv} <: AbstractSmoother
    invdiag::Vector{Tv}
    tmp::Vector{Tv}
    ω::Tv      # damping factor
end

"""
    ColoredGaussSeidelSmoother{Tv, Ti}

Parallel multicolor Gauss-Seidel smoother. Nodes are colored such that same-color
nodes have no direct connections, enabling parallel updates within each color.
"""
mutable struct ColoredGaussSeidelSmoother{Tv, Ti} <: AbstractSmoother
    colors::Vector{Ti}          # color[i] = color index for node i
    color_offsets::Vector{Int}  # color_offsets[c]:color_offsets[c+1]-1 = nodes of color c
    color_order::Vector{Ti}     # nodes sorted by color
    num_colors::Int
    invdiag::Vector{Tv}         # inverse diagonal
end

"""
    SPAI0Smoother{Tv}

SPAI(0) smoother: diagonal sparse approximate inverse. M ≈ diag(A)⁻¹ where
M[i,i] = A[i,i] / (A[i,:] ⋅ A[:,i]).  This is the minimizer of ‖I - M*A‖_F
restricted to diagonal M.
"""
mutable struct SPAI0Smoother{Tv} <: AbstractSmoother
    m_diag::Vector{Tv}         # diagonal of sparse approximate inverse
    tmp::Vector{Tv}            # workspace
end

"""
    SPAI1Smoother{Tv, Ti}

SPAI(1) smoother: sparse approximate inverse using the sparsity pattern of A.
For each row i, computes the least-squares optimal sparse vector m_i such that
‖e_i - A * m_i‖₂ is minimized subject to sparsity(m_i) ⊆ sparsity(A[i,:]).
The result is stored in CSR format matching A's sparsity.
"""
mutable struct SPAI1Smoother{Tv, Ti} <: AbstractSmoother
    nzval::Vector{Tv}          # nonzero values of the approximate inverse (same pattern as A)
    tmp::Vector{Tv}            # workspace
end

# ── Prolongation info (stored implicitly) ─────────────────────────────────────
"""
    ProlongationOp{Ti, Tv}

Stores the prolongation operator implicitly.
- `rowptr`, `colval`, `nzval` define the sparse P in CSR layout.
- `nrow` and `ncol` are the dimensions (n_fine × n_coarse).
"""
mutable struct ProlongationOp{Ti<:Integer, Tv}
    rowptr::Vector{Ti}
    colval::Vector{Ti}
    nzval::Vector{Tv}
    nrow::Int
    ncol::Int
end

"""
    RestrictionMap{Ti}

Maps the Galerkin product triples (P[i,I], A[i,j], P[j,J]) to coarse matrix
nonzero indices for in-place Galerkin product computation during resetup.
Each entry k represents a contribution p_i * a_ij * p_j → A_c[I,J].
"""
struct RestrictionMap{Ti<:Integer}
    triple_coarse_nz::Vector{Ti}  # coarse NZ index to accumulate into
    triple_pi_idx::Vector{Ti}     # P.nzval index for p_i weight
    triple_anz_idx::Vector{Ti}    # A.nzval index for a_ij value
    triple_pj_idx::Vector{Ti}     # P.nzval index for p_j weight
end

# ── AMG Level ─────────────────────────────────────────────────────────────────
"""
    AMGLevel{Tv, Ti}

One level of the AMG hierarchy.
"""
mutable struct AMGLevel{Tv, Ti<:Integer}
    A::StaticSparsityMatrixCSR{Tv, Ti}
    P::ProlongationOp{Ti, Tv}
    R_map::RestrictionMap{Ti}
    smoother::AbstractSmoother
    r::Vector{Tv}      # residual workspace
    xc::Vector{Tv}     # coarse solution workspace
    bc::Vector{Tv}     # coarse RHS workspace
end

# ── AMG Hierarchy ─────────────────────────────────────────────────────────────
"""
    AMGHierarchy{Tv, Ti}

Complete AMG hierarchy with multiple levels and a direct solver at the coarsest level.
The coarse LU factorization uses a pre-allocated buffer and pivot vector for
in-place refactorization during resetup.
"""
mutable struct AMGHierarchy{Tv, Ti<:Integer}
    levels::Vector{AMGLevel{Tv, Ti}}
    coarse_A::Matrix{Tv}       # dense coarse matrix (values recomputed each resetup)
    coarse_lu::Matrix{Tv}      # separate buffer for LU factorization (overwritten by getrf!)
    coarse_ipiv::Vector{LinearAlgebra.BlasInt}  # pivot vector (reused across resetups)
    coarse_factor::LU{Tv, Matrix{Tv}, Vector{LinearAlgebra.BlasInt}}  # LU wrapper (references lu/ipiv)
    coarse_x::Vector{Tv}       # workspace for coarsest level
    coarse_b::Vector{Tv}       # workspace for coarsest level
    solve_r::Vector{Tv}        # residual buffer for amg_solve! (finest level size)
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
- `verbose`: Print hierarchy information and solve diagnostics
- `initial_coarsening`: Optional alternative coarsening for the first N levels (defaults to `coarsening`)
- `initial_coarsening_levels`: Number of levels to use `initial_coarsening` for (default: 0)
"""
struct AMGConfig
    coarsening::CoarseningAlgorithm
    smoother::SmootherType
    max_levels::Int
    max_coarse_size::Int
    pre_smoothing_steps::Int
    post_smoothing_steps::Int
    jacobi_omega::Float64
    verbose::Bool
    initial_coarsening::CoarseningAlgorithm
    initial_coarsening_levels::Int
end

function AMGConfig(;
    coarsening::CoarseningAlgorithm = AggregationCoarsening(),
    smoother::SmootherType = JacobiSmootherType(),
    max_levels::Int = 20,
    max_coarse_size::Int = 50,
    pre_smoothing_steps::Int = 1,
    post_smoothing_steps::Int = 1,
    jacobi_omega::Float64 = 2.0/3.0,
    verbose::Bool = false,
    initial_coarsening::CoarseningAlgorithm = coarsening,
    initial_coarsening_levels::Int = 0,
)
    return AMGConfig(coarsening, smoother, max_levels, max_coarse_size,
                     pre_smoothing_steps, post_smoothing_steps, jacobi_omega, verbose,
                     initial_coarsening, initial_coarsening_levels)
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
