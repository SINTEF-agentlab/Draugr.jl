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

# ── Smoother types ────────────────────────────────────────────────────────────
"""
    JacobiSmoother{Tv}

Weighted Jacobi smoother.  Stores the inverse diagonal and a workspace vector.
"""
mutable struct JacobiSmoother{Tv}
    invdiag::Vector{Tv}
    tmp::Vector{Tv}
    ω::Tv      # damping factor
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
    smoother::JacobiSmoother{Tv}
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
end

# ── AMG Configuration ─────────────────────────────────────────────────────────
"""
    AMGConfig

Configuration for AMG setup.
"""
struct AMGConfig
    coarsening::CoarseningAlgorithm
    max_levels::Int
    max_coarse_size::Int
    pre_smoothing_steps::Int
    post_smoothing_steps::Int
    jacobi_omega::Float64
end

function AMGConfig(;
    coarsening::CoarseningAlgorithm = AggregationCoarsening(),
    max_levels::Int = 20,
    max_coarse_size::Int = 50,
    pre_smoothing_steps::Int = 1,
    post_smoothing_steps::Int = 1,
    jacobi_omega::Float64 = 2.0/3.0,
)
    return AMGConfig(coarsening, max_levels, max_coarse_size,
                     pre_smoothing_steps, post_smoothing_steps, jacobi_omega)
end
