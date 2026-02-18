# ══════════════════════════════════════════════════════════════════════════════
# C-callable interface for Draugr
#
# Provides Base.@ccallable wrappers around the core AMG routines so that
# PackageCompiler can export them as C symbols in a compiled shared library.
# Also provides @cfunction pointers for the Julia-embedding use case.
#
# Configuration is passed as a JSON string via draugr_amg_config_from_json().
#
# Hierarchies and configs are stored in a global handle table so that C
# callers receive an opaque Int32 handle.
# ══════════════════════════════════════════════════════════════════════════════

using JSON3

# ── Error reporting ──────────────────────────────────────────────────────────

const _LAST_ERROR_BUF = Ref{Vector{UInt8}}(UInt8[0])

function _set_last_error(msg::String)
    buf = Vector{UInt8}(msg)
    push!(buf, 0x00)
    _LAST_ERROR_BUF[] = buf
end

Base.@ccallable function draugr_amg_last_error()::Ptr{UInt8}
    return pointer(_LAST_ERROR_BUF[])
end

# ── Type coercion helpers ────────────────────────────────────────────────────
# JSON values may arrive as strings (e.g. "0.5", "20", "true") rather than
# native JSON types. These helpers handle both forms transparently.

_to_float(v::Number) = Float64(v)
_to_float(v) = parse(Float64, string(v))

_to_int(v::Number) = Int(round(v))
_to_int(v) = parse(Int, string(v))

_to_bool(v::Bool) = v
_to_bool(v) = lowercase(string(v)) in ("true", "1")

_to_string(v) = lowercase(string(v))

# ── JSON ↔ Julia type tables ──────────────────────────────────────────────────

function _lookup(table, name, label)
    haskey(table, name) && return table[name]
    valid = join(sort(collect(keys(table))), ", ")
    error("Unknown $label: \"$name\". Valid: $valid")
end

# Smoother: name ↔ type
const _SMOOTHER_TYPES = Dict(
    "jacobi"        => JacobiSmootherType,
    "colored_gs"    => ColoredGaussSeidelType,
    "serial_gs"     => SerialGaussSeidelType,
    "spai0"         => SPAI0SmootherType,
    "spai1"         => SPAI1SmootherType,
    "l1_jacobi"     => L1JacobiSmootherType,
    "l1_colored_gs" => L1ColoredGaussSeidelType,
    "l1_serial_gs"  => L1SerialGaussSeidelType,
    "chebyshev"     => ChebyshevSmootherType,
    "ilu0"          => ILU0SmootherType,
    "serial_ilu0"   => SerialILU0SmootherType,
)
const _SMOOTHER_NAMES = Dict(v => k for (k, v) in _SMOOTHER_TYPES)

# Interpolation: name ↔ type, constructors with defaults from type instances
const _INTERPOLATION_NAMES = Dict(
    DirectInterpolation     => "direct",
    StandardInterpolation   => "standard",
    ExtendedIInterpolation  => "extended_i",
)

const _INTERPOLATION_CONSTRUCTORS = let
    dd = DirectInterpolation()
    sd = StandardInterpolation()
    ed = ExtendedIInterpolation()
    Dict(
        "direct"     => d -> DirectInterpolation(
            _to_float(get(d, "trunc_factor", dd.trunc_factor))),
        "standard"   => d -> StandardInterpolation(
            _to_float(get(d, "trunc_factor", sd.trunc_factor))),
        "extended_i" => d -> ExtendedIInterpolation(
            _to_float(get(d, "trunc_factor", ed.trunc_factor)),
            _to_int(get(d, "max_elements", ed.max_elements)),
            _to_int(get(d, "norm_p", ed.norm_p)),
            _to_bool(get(d, "rescale", ed.rescale))),
    )
end

# Coarsening: name ↔ type, constructors
const _COARSENING_NAMES = Dict(
    AggregationCoarsening          => "aggregation",
    PMISCoarsening                 => "pmis",
    HMISCoarsening                 => "hmis",
    RSCoarsening                   => "rs",
    AggressiveCoarsening           => "aggressive",
    SmoothedAggregationCoarsening  => "smoothed_aggregation",
)

const _COARSENING_CONSTRUCTORS = Dict(
    "aggregation"          => (θ, ip) -> AggregationCoarsening(θ),
    "pmis"                 => (θ, ip) -> PMISCoarsening(θ, ip),
    "hmis"                 => (θ, ip) -> HMISCoarsening(θ, ip),
    "rs"                   => (θ, ip) -> RSCoarsening(θ, ip),
    "aggressive_pmis"      => (θ, ip) -> AggressiveCoarsening(θ, :pmis, ip),
    "aggressive_hmis"      => (θ, ip) -> AggressiveCoarsening(θ, :hmis, ip),
    "smoothed_aggregation" => (θ, ip) -> SmoothedAggregationCoarsening(θ),
)

# Strength / cycle: name ↔ value
const _STRENGTH_MAP   = Dict("absolute" => AbsoluteStrength(), "signed" => SignedStrength())
const _STRENGTH_NAMES = Dict(typeof(v) => k for (k, v) in _STRENGTH_MAP)

const _CYCLE_MAP   = Dict("v" => :V, "w" => :W)
const _CYCLE_NAMES = Dict(v => k for (k, v) in _CYCLE_MAP)

# ── Parsers ───────────────────────────────────────────────────────────────────

function _parse_interpolation(v, default_name)
    if v isa AbstractString
        return _lookup(_INTERPOLATION_CONSTRUCTORS, lowercase(v), "interpolation")(Dict())
    elseif v isa AbstractDict
        name = _to_string(get(v, "type", default_name))
        return _lookup(_INTERPOLATION_CONSTRUCTORS, name, "interpolation")(v)
    else
        error("\"interpolation\" must be a string or object, got: $(typeof(v))")
    end
end

function _parse_smoother(v, default_name)
    if v isa AbstractString
        return (_lookup(_SMOOTHER_TYPES, lowercase(v), "smoother")(), nothing)
    elseif v isa AbstractDict
        name  = _to_string(get(v, "type", default_name))
        omega = haskey(v, "omega") ? _to_float(v["omega"]) : nothing
        return (_lookup(_SMOOTHER_TYPES, name, "smoother")(), omega)
    else
        error("\"smoother\" must be a string or object, got: $(typeof(v))")
    end
end

function _parse_coarsening(name_str, θ::Float64, interp::InterpolationType)
    name = _to_string(name_str)
    return _lookup(_COARSENING_CONSTRUCTORS, name, "coarsening")(θ, interp)
end

_parse_strength(v) = _lookup(_STRENGTH_MAP, _to_string(v), "strength")
_parse_cycle(v)    = _lookup(_CYCLE_MAP, _to_string(v), "cycle")

# ── Defaults (single source of truth: AMGConfig() in types.jl) ───────────────

const _DEFAULTS = AMGConfig()

const _DEFAULT_INTERP_NAME     = _INTERPOLATION_NAMES[typeof(_DEFAULTS.coarsening.interpolation)]
const _DEFAULT_COARSENING_NAME = _COARSENING_NAMES[typeof(_DEFAULTS.coarsening)]
const _DEFAULT_SMOOTHER_NAME   = _SMOOTHER_NAMES[typeof(_DEFAULTS.smoother)]
const _DEFAULT_STRENGTH_NAME   = _STRENGTH_NAMES[typeof(_DEFAULTS.strength_type)]
const _DEFAULT_CYCLE_NAME      = _CYCLE_NAMES[_DEFAULTS.cycle_type]

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

Base.@ccallable function draugr_amg_config_from_json(json_ptr::Ptr{UInt8})::Int32
    try
        _set_last_error("")
        json_str = (json_ptr == C_NULL) ? "{}" : unsafe_string(json_ptr)
        d = JSON3.read(json_str, Dict{String, Any})

        interp     = _parse_interpolation(get(d, "interpolation", _DEFAULT_INTERP_NAME), _DEFAULT_INTERP_NAME)
        θ          = _to_float(get(d, "theta", _DEFAULTS.coarsening.θ))
        coarsening = _parse_coarsening(get(d, "coarsening", _DEFAULT_COARSENING_NAME), θ, interp)

        ic_raw = get(d, "initial_coarsening", nothing)
        initial_coarsening = ic_raw === nothing ? coarsening :
            _parse_coarsening(ic_raw, θ, interp)

        smoother, omega_override = _parse_smoother(get(d, "smoother", _DEFAULT_SMOOTHER_NAME), _DEFAULT_SMOOTHER_NAME)
        jacobi_omega = something(omega_override,
            _to_float(get(d, "jacobi_omega", _DEFAULTS.jacobi_omega)))

        config = AMGConfig(;
            coarsening, smoother, jacobi_omega, initial_coarsening,
            strength_type         = _parse_strength(get(d, "strength", _DEFAULT_STRENGTH_NAME)),
            cycle_type            = _parse_cycle(get(d, "cycle", _DEFAULT_CYCLE_NAME)),
            max_levels            = _to_int(get(d, "max_levels", _DEFAULTS.max_levels)),
            max_coarse_size       = _to_int(get(d, "max_coarse_size", _DEFAULTS.max_coarse_size)),
            pre_smoothing_steps   = _to_int(get(d, "pre_smoothing_steps", _DEFAULTS.pre_smoothing_steps)),
            post_smoothing_steps  = _to_int(get(d, "post_smoothing_steps", _DEFAULTS.post_smoothing_steps)),
            verbose               = _to_int(get(d, "verbose", _DEFAULTS.verbose)),
            initial_coarsening_levels = _to_int(get(d, "initial_coarsening_levels", _DEFAULTS.initial_coarsening_levels)),
            max_row_sum           = _to_float(get(d, "max_row_sum", _DEFAULTS.max_row_sum)),
            coarse_solve_on_cpu   = _to_bool(get(d, "coarse_solve_on_cpu", _DEFAULTS.coarse_solve_on_cpu)),
        )

        lock(_HANDLE_LOCK) do
            h = _new_handle()
            _CONFIG_HANDLES[h] = config
            return h
        end
    catch e
        msg = sprint(showerror, e)
        _set_last_error(msg)
        @error "draugr_amg_config_from_json failed" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

@doc """
    draugr_amg_config_from_json(json_ptr) -> Int32

Create an AMG configuration from a JSON string and return a handle.
All keys are optional; defaults match `AMGConfig()` in types.jl.
Unknown keys are ignored.

Pass NULL or "{}" for all defaults.
Returns a config handle (> 0) on success, or -1 on error.
Call `draugr_amg_last_error()` for the error message on failure.

    draugr_amg_config_from_json(json::String) -> Int32

Convenience method for Julia callers — avoids manual pointer wrangling.
""" draugr_amg_config_from_json

draugr_amg_config_from_json(json::String) =
    GC.@preserve json draugr_amg_config_from_json(Base.unsafe_convert(Ptr{UInt8}, json))

Base.@ccallable function draugr_amg_setup(n::Int32, nnz_count::Int32,
                                          rowptr::Ptr{Int32}, colval::Ptr{Int32},
                                          nzval::Ptr{Float64},
                                          config_handle::Int32,
                                          index_base::Int32)::Int32
    try
        _set_last_error("")
        rp = unsafe_wrap(Array, rowptr, Int(n) + 1)
        cv = unsafe_wrap(Array, colval, Int(nnz_count))
        nzv = unsafe_wrap(Array, nzval, Int(nnz_count))
        A = csr_from_raw(copy(rp), copy(cv), copy(nzv), Int(n), Int(n); index_base=Int(index_base))
        config = lock(_HANDLE_LOCK) do
            get(_CONFIG_HANDLES, config_handle, nothing)
        end
        if config === nothing
            _set_last_error("Invalid config handle: $config_handle")
            return Int32(-1)
        end
        hierarchy = amg_setup(A, config)
        lock(_HANDLE_LOCK) do
            h = _new_handle()
            _HIERARCHY_HANDLES[h] = hierarchy
            return h
        end
    catch e
        msg = sprint(showerror, e)
        _set_last_error(msg)
        @error "draugr_amg_setup failed" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

@doc """
    draugr_amg_setup(n, nnz, rowptr, colval, nzval, config_handle, index_base) -> Int32

Build an AMG hierarchy from CSR data and return a hierarchy handle.

Arguments:
- `n`:             Number of rows (== columns, square matrix)
- `nnz`:           Number of nonzeros (length of colval/nzval)
- `rowptr`:        Ptr{Int32} to row-pointer array (length n+1)
- `colval`:        Ptr{Int32} to column-index array (length nnz)
- `nzval`:         Ptr{Float64} to values array  (length nnz)
- `config_handle`: Config handle from `draugr_amg_config_from_json`
- `index_base`:    Index base of incoming arrays: 0 for C-style zero-based,
                   1 for Fortran/Julia-style one-based.
                   When 0, rowptr and colval are converted to 1-based on
                   owned copies (no extra allocation).

Returns a hierarchy handle (> 0) on success, or -1 on error.
""" draugr_amg_setup

Base.@ccallable function draugr_amg_resetup(handle::Int32, n::Int32, nnz_count::Int32,
                                            rowptr::Ptr{Int32}, colval::Ptr{Int32},
                                            nzval::Ptr{Float64},
                                            config_handle::Int32,
                                            index_base::Int32)::Int32
    try
        _set_last_error("")
        rp = unsafe_wrap(Array, rowptr, Int(n) + 1)
        cv = unsafe_wrap(Array, colval, Int(nnz_count))
        nzv = unsafe_wrap(Array, nzval, Int(nnz_count))
        A_csr = csr_from_raw(copy(rp), copy(cv), copy(nzv), Int(n), Int(n); index_base=Int(index_base))
        hierarchy, config = lock(_HANDLE_LOCK) do
            h = get(_HIERARCHY_HANDLES, handle, nothing)
            c = get(_CONFIG_HANDLES, config_handle, nothing)
            return h, c
        end
        if hierarchy === nothing || config === nothing
            _set_last_error("Invalid handle (hierarchy=$handle, config=$config_handle)")
            return Int32(-1)
        end
        backend = hierarchy.backend
        block_size = hierarchy.block_size
        nlevels = length(hierarchy.levels)
        if nlevels == 0
            Draugr._update_coarse_solver!(hierarchy, A_csr; block_size=block_size)
            return Int32(0)
        end
        level1 = hierarchy.levels[1]
        Draugr._copy_nzvals!(level1.A, A_csr; block_size=block_size)
        Draugr.update_smoother!(level1.smoother, level1.A; block_size=block_size)
        for lvl in 1:(nlevels - 1)
            level = hierarchy.levels[lvl]
            next_level = hierarchy.levels[lvl + 1]
            Draugr.galerkin_product!(next_level.A, level.A, level.P, level.R_map; block_size=block_size)
            Draugr.update_smoother!(next_level.smoother, next_level.A; block_size=block_size)
        end
        last_level = hierarchy.levels[nlevels]
        Draugr._recompute_coarsest_dense!(hierarchy, last_level)
        hierarchy.coarse_factor = lu(hierarchy.coarse_A)
        return Int32(0)
    catch e
        msg = sprint(showerror, e)
        _set_last_error(msg)
        @error "draugr_amg_resetup failed" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

@doc """
    draugr_amg_resetup(handle, n, nnz, rowptr, colval, nzval, config_handle, index_base) -> Int32

Update the AMG hierarchy with new matrix coefficients (same sparsity pattern).

When `index_base=0`, the incoming rowptr/colval use zero-based indexing and are
converted to one-based on owned copies (no extra allocation).

Returns 0 on success, -1 on error.
""" draugr_amg_resetup

Base.@ccallable function draugr_amg_solve(handle::Int32, n::Int32,
                                          x::Ptr{Float64}, b::Ptr{Float64},
                                          config_handle::Int32,
                                          tol::Float64, maxiter::Int32)::Int32
    try
        _set_last_error("")
        xv = unsafe_wrap(Array, x, Int(n))
        bv = unsafe_wrap(Array, b, Int(n))
        hierarchy, config = lock(_HANDLE_LOCK) do
            h = get(_HIERARCHY_HANDLES, handle, nothing)
            c = get(_CONFIG_HANDLES, config_handle, nothing)
            return h, c
        end
        if hierarchy === nothing || config === nothing
            _set_last_error("Invalid handle (hierarchy=$handle, config=$config_handle)")
            return Int32(-1)
        end
        _, niter = amg_solve!(xv, bv, hierarchy, config; tol=tol, maxiter=Int(maxiter))
        return Int32(niter)
    catch e
        msg = sprint(showerror, e)
        _set_last_error(msg)
        @error "draugr_amg_solve failed" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

@doc """
    draugr_amg_solve(handle, n, x, b, config_handle, tol, maxiter) -> Int32

Solve Ax = b using AMG. The solution is written into `x`.

Returns the number of iterations on success, or -1 on error.
""" draugr_amg_solve

Base.@ccallable function draugr_amg_cycle(handle::Int32, n::Int32,
                                          x::Ptr{Float64}, b::Ptr{Float64},
                                          config_handle::Int32)::Int32
    try
        _set_last_error("")
        xv = unsafe_wrap(Array, x, Int(n))
        bv = unsafe_wrap(Array, b, Int(n))
        hierarchy, config = lock(_HANDLE_LOCK) do
            h = get(_HIERARCHY_HANDLES, handle, nothing)
            c = get(_CONFIG_HANDLES, config_handle, nothing)
            return h, c
        end
        if hierarchy === nothing || config === nothing
            _set_last_error("Invalid handle (hierarchy=$handle, config=$config_handle)")
            return Int32(-1)
        end
        amg_cycle!(xv, bv, hierarchy, config)
        return Int32(0)
    catch e
        msg = sprint(showerror, e)
        _set_last_error(msg)
        @error "draugr_amg_cycle failed" exception=(e, catch_backtrace())
        return Int32(-1)
    end
end

@doc """
    draugr_amg_cycle(handle, n, x, b, config_handle) -> Int32

Apply one AMG cycle (V or W, as configured) to improve x for Ax = b.

Returns 0 on success, -1 on error.
""" draugr_amg_cycle

Base.@ccallable function draugr_amg_free(handle::Int32)::Int32
    lock(_HANDLE_LOCK) do
        if haskey(_HIERARCHY_HANDLES, handle)
            delete!(_HIERARCHY_HANDLES, handle)
            return Int32(0)
        else
            return Int32(-1)
        end
    end
end

@doc """
    draugr_amg_free(handle) -> Int32

Free the AMG hierarchy associated with `handle`.

Returns 0 on success, -1 if handle not found.
""" draugr_amg_free

Base.@ccallable function draugr_amg_config_free(handle::Int32)::Int32
    lock(_HANDLE_LOCK) do
        if haskey(_CONFIG_HANDLES, handle)
            delete!(_CONFIG_HANDLES, handle)
            return Int32(0)
        else
            return Int32(-1)
        end
    end
end

@doc """
    draugr_amg_config_free(handle) -> Int32

Free the AMG configuration associated with `handle`.

Returns 0 on success, -1 if handle not found.
""" draugr_amg_config_free


# ── @cfunction pointers (for Julia-embedding use case) ───────────────────────

"""
    draugr_amg_get_cfunctions() -> NamedTuple

Return a named tuple of `@cfunction` pointers for the embedding use case.
"""
function draugr_amg_get_cfunctions()
    return (
        config_from_json = @cfunction(draugr_amg_config_from_json,
            Int32, (Ptr{UInt8},)),
        last_error = @cfunction(draugr_amg_last_error,
            Ptr{UInt8}, ()),
        setup = @cfunction(draugr_amg_setup,
            Int32,
            (Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Int32, Int32)),
        resetup = @cfunction(draugr_amg_resetup,
            Int32,
            (Int32, Int32, Int32, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Int32, Int32)),
        solve = @cfunction(draugr_amg_solve,
            Int32,
            (Int32, Int32, Ptr{Float64}, Ptr{Float64}, Int32, Float64, Int32)),
        cycle = @cfunction(draugr_amg_cycle,
            Int32,
            (Int32, Int32, Ptr{Float64}, Ptr{Float64}, Int32)),
        free = @cfunction(draugr_amg_free, Int32, (Int32,)),
        config_free = @cfunction(draugr_amg_config_free, Int32, (Int32,)),
    )
end
