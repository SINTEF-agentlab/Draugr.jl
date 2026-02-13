# ══════════════════════════════════════════════════════════════════════════════
# Auto-conversion wrappers for StaticSparsityMatrixCSR → CSRMatrix
# ══════════════════════════════════════════════════════════════════════════════
#
# These thin wrappers allow all internal functions to be called with
# StaticSparsityMatrixCSR arguments. They convert to CSRMatrix and forward.

# ── Strength ─────────────────────────────────────────────────────────────────
strength_graph(A::StaticSparsityMatrixCSR, θ::Real) =
    strength_graph(_auto_convert(A), θ)
strength_graph(A::StaticSparsityMatrixCSR, θ::Real, st::StrengthType) =
    strength_graph(_auto_convert(A), θ, st)
strength_graph(A::StaticSparsityMatrixCSR, θ::Real, config::AMGConfig) =
    strength_graph(_auto_convert(A), θ, config)
strong_neighbors(A::StaticSparsityMatrixCSR, is_strong::Vector{Bool}, row::Integer) =
    strong_neighbors(_auto_convert(A), is_strong, row)
_apply_max_row_sum(A::StaticSparsityMatrixCSR, threshold::Real) =
    _apply_max_row_sum(_auto_convert(A), threshold)

# ── Coarsening ───────────────────────────────────────────────────────────────
coarsen_aggregation(A::StaticSparsityMatrixCSR, θ::Real) =
    coarsen_aggregation(_auto_convert(A), θ)
coarsen_pmis(A::StaticSparsityMatrixCSR, θ::Real; kwargs...) =
    coarsen_pmis(_auto_convert(A), θ; kwargs...)
coarsen_hmis(A::StaticSparsityMatrixCSR, θ::Real; kwargs...) =
    coarsen_hmis(_auto_convert(A), θ; kwargs...)
coarsen_aggressive(A::StaticSparsityMatrixCSR, θ::Real; kwargs...) =
    coarsen_aggressive(_auto_convert(A), θ; kwargs...)
coarsen(A::StaticSparsityMatrixCSR, alg::CoarseningAlgorithm, config::AMGConfig=AMGConfig()) =
    coarsen(_auto_convert(A), alg, config)
coarsen_cf(A::StaticSparsityMatrixCSR, alg::CoarseningAlgorithm, config::AMGConfig=AMGConfig()) =
    coarsen_cf(_auto_convert(A), alg, config)

# ── Prolongation ─────────────────────────────────────────────────────────────
build_prolongation(A::StaticSparsityMatrixCSR, agg::Vector{Int}, n_coarse::Int) =
    build_prolongation(_auto_convert(A), agg, n_coarse)
build_cf_prolongation(A::StaticSparsityMatrixCSR, cf::Vector{Int},
                      coarse_map::Vector{Int}, n_coarse::Int, interp::InterpolationType) =
    build_cf_prolongation(_auto_convert(A), cf, coarse_map, n_coarse, interp)
_smooth_prolongation(A::StaticSparsityMatrixCSR, P_tent::ProlongationOp, ω::Real) =
    _smooth_prolongation(_auto_convert(A), P_tent, ω)

# ── Galerkin product ─────────────────────────────────────────────────────────
compute_coarse_sparsity(A::StaticSparsityMatrixCSR, P::ProlongationOp, n_coarse::Int) =
    compute_coarse_sparsity(_auto_convert(A), P, n_coarse)
galerkin_product!(Ac::CSRMatrix, Af::StaticSparsityMatrixCSR, P::ProlongationOp,
                  r_map::RestrictionMap; kwargs...) =
    galerkin_product!(Ac, _auto_convert(Af), P, r_map; kwargs...)

# ── Smoothers (build) ────────────────────────────────────────────────────────
build_jacobi_smoother(A::StaticSparsityMatrixCSR, ω::Real) =
    build_jacobi_smoother(_auto_convert(A), ω)
build_colored_gs_smoother(A::StaticSparsityMatrixCSR) =
    build_colored_gs_smoother(_auto_convert(A))
build_spai0_smoother(A::StaticSparsityMatrixCSR) =
    build_spai0_smoother(_auto_convert(A))
build_spai1_smoother(A::StaticSparsityMatrixCSR) =
    build_spai1_smoother(_auto_convert(A))
build_l1jacobi_smoother(A::StaticSparsityMatrixCSR, ω::Real) =
    build_l1jacobi_smoother(_auto_convert(A), ω)
build_chebyshev_smoother(A::StaticSparsityMatrixCSR; kwargs...) =
    build_chebyshev_smoother(_auto_convert(A); kwargs...)
build_ilu0_smoother(A::StaticSparsityMatrixCSR) =
    build_ilu0_smoother(_auto_convert(A))
build_smoother(A::StaticSparsityMatrixCSR, st::SmootherType, ω::Real; kwargs...) =
    build_smoother(_auto_convert(A), st, ω; kwargs...)
greedy_coloring(A::StaticSparsityMatrixCSR) =
    greedy_coloring(_auto_convert(A))

# ── Smoothers (smooth!) ──────────────────────────────────────────────────────
smooth!(x::AbstractVector, A::StaticSparsityMatrixCSR, b::AbstractVector,
        smoother::AbstractSmoother; kwargs...) =
    smooth!(x, _auto_convert(A), b, smoother; kwargs...)

# ── Smoothers (update) ───────────────────────────────────────────────────────
update_smoother!(smoother::AbstractSmoother, A::StaticSparsityMatrixCSR; kwargs...) =
    update_smoother!(smoother, _auto_convert(A); kwargs...)

# ── Cycle ────────────────────────────────────────────────────────────────────
compute_residual!(r::AbstractVector, A::StaticSparsityMatrixCSR,
                  x::AbstractVector, b::AbstractVector; kwargs...) =
    compute_residual!(r, _auto_convert(A), x, b; kwargs...)
