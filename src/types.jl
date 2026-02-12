# ── Coarsening type tags ──────────────────────────────────────────────────────
abstract type CoarseningAlgorithm end

"""Greedy aggregation coarsening."""
struct AggregationCoarsening <: CoarseningAlgorithm
    θ::Float64   # strength threshold
end
AggregationCoarsening() = AggregationCoarsening(0.25)

"""Parallel Modified Independent Set coarsening."""
struct PMISCoarsening <: CoarseningAlgorithm
    θ::Float64
end
PMISCoarsening() = PMISCoarsening(0.25)

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

Maps each nonzero of A_fine to the corresponding nonzero index in A_coarse,
enabling in-place Galerkin product computation during resetup.
"""
struct RestrictionMap{Ti<:Integer}
    fine_to_coarse_nz::Vector{Ti}
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
"""
mutable struct AMGHierarchy{Tv, Ti<:Integer}
    levels::Vector{AMGLevel{Tv, Ti}}
    coarse_A::Matrix{Tv}       # dense coarse matrix
    coarse_factor::Any         # LU factorization of coarse_A
    coarse_x::Vector{Tv}       # workspace for coarsest level
    coarse_b::Vector{Tv}       # workspace for coarsest level
    solve_r::Vector{Tv}        # residual buffer for amg_solve! (finest level size)
end

# ── AMG Configuration ─────────────────────────────────────────────────────────
"""
    AMGConfig

Configuration for AMG setup.
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
)
    return AMGConfig(coarsening, smoother, max_levels, max_coarse_size,
                     pre_smoothing_steps, post_smoothing_steps, jacobi_omega, verbose)
end
