# ══════════════════════════════════════════════════════════════════════════════
# C-callable interface for ParallelAMG
#
# Provides @cfunction-compatible wrappers around the core AMG routines
# (setup, resetup, solve, cycle) using integer enums to select coarsening,
# smoother, and interpolation options.  Hierarchies are stored in a
# global handle table so that C callers receive an opaque Int32 handle.
# ══════════════════════════════════════════════════════════════════════════════

# ── Enum definitions ─────────────────────────────────────────────────────────

"""
Integer codes for coarsening algorithms (passed from C).
"""
@enum CoarseningEnum::Int32 begin
    COARSENING_AGGREGATION          = 0
    COARSENING_PMIS                 = 1
    COARSENING_HMIS                 = 2
    COARSENING_RS                   = 3
    COARSENING_AGGRESSIVE_PMIS      = 4
    COARSENING_AGGRESSIVE_HMIS      = 5
    COARSENING_SMOOTHED_AGGREGATION = 6
end

"""
Integer codes for smoother types (passed from C).
"""
@enum SmootherEnum::Int32 begin
    SMOOTHER_JACOBI            = 0
    SMOOTHER_COLORED_GS        = 1
    SMOOTHER_SERIAL_GS         = 2
    SMOOTHER_SPAI0             = 3
    SMOOTHER_SPAI1             = 4
    SMOOTHER_L1_JACOBI         = 5
    SMOOTHER_CHEBYSHEV         = 6
    SMOOTHER_ILU0              = 7
    SMOOTHER_L1_COLORED_GS     = 8
end

"""
Integer codes for interpolation types (passed from C).
"""
@enum InterpolationEnum::Int32 begin
    INTERPOLATION_DIRECT     = 0
    INTERPOLATION_STANDARD   = 1
    INTERPOLATION_EXTENDED_I = 2
end

"""
Integer codes for cycle types (passed from C).
"""
@enum CycleEnum::Int32 begin
    CYCLE_V = 0
    CYCLE_W = 1
end

"""
Integer codes for strength of connection types (passed from C).
"""
@enum StrengthEnum::Int32 begin
    STRENGTH_ABSOLUTE = 0
    STRENGTH_SIGNED   = 1
end

# ── Enum → Julia type conversion helpers ─────────────────────────────────────

function _interpolation_from_enum(e::InterpolationEnum, trunc::Float64)
    if e == INTERPOLATION_DIRECT
        return DirectInterpolation(trunc)
    elseif e == INTERPOLATION_STANDARD
        return StandardInterpolation(trunc)
    elseif e == INTERPOLATION_EXTENDED_I
        return ExtendedIInterpolation(trunc)
    else
        return DirectInterpolation(trunc)
    end
end

function _smoother_from_enum(e::SmootherEnum)
    if e == SMOOTHER_JACOBI
        return JacobiSmootherType()
    elseif e == SMOOTHER_COLORED_GS
        return ColoredGaussSeidelType()
    elseif e == SMOOTHER_SERIAL_GS
        return SerialGaussSeidelType()
    elseif e == SMOOTHER_SPAI0
        return SPAI0SmootherType()
    elseif e == SMOOTHER_SPAI1
        return SPAI1SmootherType()
    elseif e == SMOOTHER_L1_JACOBI
        return L1JacobiSmootherType()
    elseif e == SMOOTHER_CHEBYSHEV
        return ChebyshevSmootherType()
    elseif e == SMOOTHER_ILU0
        return ILU0SmootherType()
    elseif e == SMOOTHER_L1_COLORED_GS
        return L1ColoredGaussSeidelType()
    else
        return JacobiSmootherType()
    end
end

function _coarsening_from_enum(e::CoarseningEnum, θ::Float64,
                               interp::InterpolationEnum, trunc::Float64)
    ip = _interpolation_from_enum(interp, trunc)
    if e == COARSENING_AGGREGATION
        return AggregationCoarsening(θ)
    elseif e == COARSENING_PMIS
        return PMISCoarsening(θ, ip)
    elseif e == COARSENING_HMIS
        return HMISCoarsening(θ, ip)
    elseif e == COARSENING_RS
        return RSCoarsening(θ, ip)
    elseif e == COARSENING_AGGRESSIVE_PMIS
        return AggressiveCoarsening(θ, :pmis, ip)
    elseif e == COARSENING_AGGRESSIVE_HMIS
        return AggressiveCoarsening(θ, :hmis, ip)
    elseif e == COARSENING_SMOOTHED_AGGREGATION
        return SmoothedAggregationCoarsening(θ)
    else
        return AggregationCoarsening(θ)
    end
end

function _cycle_from_enum(e::CycleEnum)
    return e == CYCLE_W ? :W : :V
end

function _strength_from_enum(e::StrengthEnum)
    return e == STRENGTH_SIGNED ? SignedStrength() : AbsoluteStrength()
end

# ── Handle table ─────────────────────────────────────────────────────────────

const _HANDLE_LOCK = ReentrantLock()
const _HIERARCHY_HANDLES = Dict{Int32, AMGHierarchy}()
const _CONFIG_HANDLES = Dict{Int32, AMGConfig}()
const _NEXT_HANDLE = Ref{Int32}(1)

function _new_handle()::Int32
    h = _NEXT_HANDLE[]
    _NEXT_HANDLE[] += Int32(1)
    return h
end

# ── Public C-callable functions ──────────────────────────────────────────────

"""
    amg_c_config_create(coarsening, smoother, interpolation, strength,
                        cycle, θ, trunc_factor, jacobi_omega,
                        max_levels, max_coarse_size,
                        pre_smoothing_steps, post_smoothing_steps,
                        verbose) -> Int32

Create an AMG configuration and return a handle.
All enum arguments are Int32 values matching the `CoarseningEnum`, etc.

Returns a config handle (> 0) on success, or -1 on error.
"""
function amg_c_config_create(coarsening::Int32, smoother::Int32,
                             interpolation::Int32, strength::Int32,
                             cycle::Int32, θ::Float64,
                             trunc_factor::Float64, jacobi_omega::Float64,
                             max_levels::Int32, max_coarse_size::Int32,
                             pre_smoothing_steps::Int32, post_smoothing_steps::Int32,
                             verbose::Int32)::Int32
    try
        config = AMGConfig(;
            coarsening = _coarsening_from_enum(CoarseningEnum(coarsening), θ,
                                               InterpolationEnum(interpolation), trunc_factor),
            smoother   = _smoother_from_enum(SmootherEnum(smoother)),
            strength_type = _strength_from_enum(StrengthEnum(strength)),
            cycle_type    = _cycle_from_enum(CycleEnum(cycle)),
            jacobi_omega  = jacobi_omega,
            max_levels    = Int(max_levels),
            max_coarse_size = Int(max_coarse_size),
            pre_smoothing_steps  = Int(pre_smoothing_steps),
            post_smoothing_steps = Int(post_smoothing_steps),
            verbose = Int(verbose),
        )
        lock(_HANDLE_LOCK) do
            h = _new_handle()
            _CONFIG_HANDLES[h] = config
            return h
        end
    catch e
        @error "amg_c_config_create failed" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

"""
    amg_c_setup(n, nnz, rowptr, colval, nzval, config_handle) -> Int32

Build an AMG hierarchy from CSR data and return a hierarchy handle.

Arguments:
- `n`:             Number of rows (== columns, square matrix)
- `nnz`:           Number of nonzeros (unused, length of colval/nzval)
- `rowptr`:        Ptr{Int32} to row-pointer array (length n+1, 1-based)
- `colval`:        Ptr{Int32} to column-index array (length nnz, 1-based)
- `nzval`:         Ptr{Float64} to values array  (length nnz)
- `config_handle`: Config handle from `amg_c_config_create`

Returns a hierarchy handle (> 0) on success, or -1 on error.
"""
function amg_c_setup(n::Int32, nnz_count::Int32,
                     rowptr::Ptr{Int32}, colval::Ptr{Int32}, nzval::Ptr{Float64},
                     config_handle::Int32)::Int32
    try
        rp = unsafe_wrap(Array, rowptr, Int(n) + 1)
        cv = unsafe_wrap(Array, colval, Int(nnz_count))
        nzv = unsafe_wrap(Array, nzval, Int(nnz_count))
        # Copy to owned arrays
        A = CSRMatrix(copy(rp), copy(cv), copy(nzv), Int(n), Int(n))
        config = lock(_HANDLE_LOCK) do
            get(_CONFIG_HANDLES, config_handle, nothing)
        end
        if config === nothing
            @error "Invalid config handle" config_handle
            return Int32(-1)
        end
        hierarchy = amg_setup(A, config)
        lock(_HANDLE_LOCK) do
            h = _new_handle()
            _HIERARCHY_HANDLES[h] = hierarchy
            return h
        end
    catch e
        @error "amg_c_setup failed" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

"""
    amg_c_resetup!(handle, n, nnz, rowptr, colval, nzval, config_handle) -> Int32

Update the AMG hierarchy with new matrix coefficients (same sparsity pattern).

Returns 0 on success, -1 on error.
"""
function amg_c_resetup!(handle::Int32, n::Int32, nnz_count::Int32,
                        rowptr::Ptr{Int32}, colval::Ptr{Int32}, nzval::Ptr{Float64},
                        config_handle::Int32)::Int32
    try
        rp = unsafe_wrap(Array, rowptr, Int(n) + 1)
        cv = unsafe_wrap(Array, colval, Int(nnz_count))
        nzv = unsafe_wrap(Array, nzval, Int(nnz_count))
        A_csr = CSRMatrix(copy(rp), copy(cv), copy(nzv), Int(n), Int(n))
        hierarchy, config = lock(_HANDLE_LOCK) do
            h = get(_HIERARCHY_HANDLES, handle, nothing)
            c = get(_CONFIG_HANDLES, config_handle, nothing)
            return h, c
        end
        if hierarchy === nothing || config === nothing
            @error "Invalid handle" handle config_handle
            return Int32(-1)
        end
        backend = hierarchy.backend
        block_size = hierarchy.block_size
        nlevels = length(hierarchy.levels)
        if nlevels == 0
            ParallelAMG._update_coarse_solver!(hierarchy, A_csr; block_size=block_size)
            return Int32(0)
        end
        level1 = hierarchy.levels[1]
        ParallelAMG._copy_nzvals!(level1.A, A_csr; block_size=block_size)
        ParallelAMG.update_smoother!(level1.smoother, level1.A; block_size=block_size)
        for lvl in 1:(nlevels - 1)
            level = hierarchy.levels[lvl]
            next_level = hierarchy.levels[lvl + 1]
            ParallelAMG.galerkin_product!(next_level.A, level.A, level.P, level.R_map; block_size=block_size)
            ParallelAMG.update_smoother!(next_level.smoother, next_level.A; block_size=block_size)
        end
        last_level = hierarchy.levels[nlevels]
        ParallelAMG._recompute_coarsest_dense!(hierarchy, last_level)
        hierarchy.coarse_factor = lu(hierarchy.coarse_A)
        return Int32(0)
    catch e
        @error "amg_c_resetup! failed" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

"""
    amg_c_solve!(handle, n, x, b, config_handle, tol, maxiter) -> Int32

Solve Ax = b using AMG. The solution is written into `x`.

Returns the number of iterations on success, or -1 on error.
"""
function amg_c_solve!(handle::Int32, n::Int32,
                      x::Ptr{Float64}, b::Ptr{Float64},
                      config_handle::Int32,
                      tol::Float64, maxiter::Int32)::Int32
    try
        xv = unsafe_wrap(Array, x, Int(n))
        bv = unsafe_wrap(Array, b, Int(n))
        hierarchy, config = lock(_HANDLE_LOCK) do
            h = get(_HIERARCHY_HANDLES, handle, nothing)
            c = get(_CONFIG_HANDLES, config_handle, nothing)
            return h, c
        end
        if hierarchy === nothing || config === nothing
            @error "Invalid handle" handle config_handle
            return Int32(-1)
        end
        _, niter = amg_solve!(xv, bv, hierarchy, config; tol=tol, maxiter=Int(maxiter))
        return Int32(niter)
    catch e
        @error "amg_c_solve! failed" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

"""
    amg_c_cycle!(handle, n, x, b, config_handle) -> Int32

Apply one AMG cycle (V or W, as configured) to improve x for Ax = b.

Returns 0 on success, -1 on error.
"""
function amg_c_cycle!(handle::Int32, n::Int32,
                      x::Ptr{Float64}, b::Ptr{Float64},
                      config_handle::Int32)::Int32
    try
        xv = unsafe_wrap(Array, x, Int(n))
        bv = unsafe_wrap(Array, b, Int(n))
        hierarchy, config = lock(_HANDLE_LOCK) do
            h = get(_HIERARCHY_HANDLES, handle, nothing)
            c = get(_CONFIG_HANDLES, config_handle, nothing)
            return h, c
        end
        if hierarchy === nothing || config === nothing
            @error "Invalid handle" handle config_handle
            return Int32(-1)
        end
        amg_cycle!(xv, bv, hierarchy, config)
        return Int32(0)
    catch e
        @error "amg_c_cycle! failed" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

"""
    amg_c_free!(handle) -> Int32

Free the AMG hierarchy associated with `handle`.

Returns 0 on success, -1 if handle not found.
"""
function amg_c_free!(handle::Int32)::Int32
    lock(_HANDLE_LOCK) do
        if haskey(_HIERARCHY_HANDLES, handle)
            delete!(_HIERARCHY_HANDLES, handle)
            return Int32(0)
        else
            return Int32(-1)
        end
    end
end

"""
    amg_c_config_free!(handle) -> Int32

Free the AMG configuration associated with `handle`.

Returns 0 on success, -1 if handle not found.
"""
function amg_c_config_free!(handle::Int32)::Int32
    lock(_HANDLE_LOCK) do
        if haskey(_CONFIG_HANDLES, handle)
            delete!(_CONFIG_HANDLES, handle)
            return Int32(0)
        else
            return Int32(-1)
        end
    end
end

# ── @cfunction pointers ─────────────────────────────────────────────────────

"""
    amg_c_get_cfunctions() -> NamedTuple

Return a named tuple of `@cfunction` pointers that can be passed to C code.

Fields:
- `config_create`: Ptr to `amg_c_config_create`
- `setup`:         Ptr to `amg_c_setup`
- `resetup`:       Ptr to `amg_c_resetup!`
- `solve`:         Ptr to `amg_c_solve!`
- `cycle`:         Ptr to `amg_c_cycle!`
- `free`:          Ptr to `amg_c_free!`
- `config_free`:   Ptr to `amg_c_config_free!`
"""
function amg_c_get_cfunctions()
    return (
        config_create = @cfunction(amg_c_config_create,
            Int32,
            (Int32, Int32, Int32, Int32, Int32,
             Float64, Float64, Float64,
             Int32, Int32, Int32, Int32, Int32)),
        setup = @cfunction(amg_c_setup,
            Int32,
            (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Int32)),
        resetup = @cfunction(amg_c_resetup!,
            Int32,
            (Int32, Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Int32)),
        solve = @cfunction(amg_c_solve!,
            Int32,
            (Int32, Int32, Ptr{Float64}, Ptr{Float64}, Int32, Float64, Int32)),
        cycle = @cfunction(amg_c_cycle!,
            Int32,
            (Int32, Int32, Ptr{Float64}, Ptr{Float64}, Int32)),
        free = @cfunction(amg_c_free!, Int32, (Int32,)),
        config_free = @cfunction(amg_c_config_free!, Int32, (Int32,)),
    )
end
