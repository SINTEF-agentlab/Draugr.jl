# Warmup script for PackageCompiler precompilation.
# Exercises the main C-callable code paths so they are compiled into the
# sysimage and don't trigger JIT at runtime.

using Draugr

function _warmup()
    # Build a tiny 1D Laplacian in CSR format, 1-based.
    n = 20
    rowptr = Int32[]
    colval = Int32[]
    nzval  = Float64[]
    push!(rowptr, Int32(1))
    for i in 1:n
        if i > 1
            push!(colval, Int32(i - 1)); push!(nzval, -1.0)
        end
        push!(colval, Int32(i)); push!(nzval, 2.0)
        if i < n
            push!(colval, Int32(i + 1)); push!(nzval, -1.0)
        end
        push!(rowptr, Int32(length(colval) + 1))
    end
    nnz = Int32(length(nzval))

    # Exercise config creation — string-only path (non-default values)
    json = """{"coarsening": "pmis", "interpolation": "standard", "theta": 0.25}"""
    cfg = draugr_amg_config_from_json(json)

    # Exercise setup (1-based indexing, allow partial resetup)
    h = draugr_amg_setup(Int32(n), nnz, pointer(rowptr), pointer(colval),
                         pointer(nzval), cfg, Int32(1), Int32(1))

    # Exercise solve
    b = ones(Float64, n)
    x = zeros(Float64, n)
    draugr_amg_solve(h, Int32(n), pointer(x), pointer(b), cfg, 1e-6, Int32(50))

    # Exercise cycle
    fill!(x, 0.0)
    draugr_amg_cycle(h, Int32(n), pointer(x), pointer(b), cfg)

    # Exercise partial resetup (coefficient-only update)
    nzval2 = copy(nzval)
    nzval2 .*= 1.1
    draugr_amg_resetup(h, Int32(n), nnz, pointer(rowptr), pointer(colval),
                       pointer(nzval2), cfg, Int32(1), Int32(1), Int32(1))

    # Exercise full resetup (rebuild hierarchy, reuse workspace, keep restriction maps)
    nzval3 = copy(nzval)
    nzval3 .*= 0.9
    draugr_amg_resetup(h, Int32(n), nnz, pointer(rowptr), pointer(colval),
                       pointer(nzval3), cfg, Int32(1), Int32(0), Int32(1))

    # Exercise error reporting
    draugr_amg_last_error()

    # Exercise nested-object path (interpolation sub-params, smoother omega)
    cfg2 = draugr_amg_config_from_json("""{
        "interpolation": {"type": "extended_i", "trunc_factor": 0.3, "max_elements": 5},
        "smoother": {"type": "jacobi", "omega": 0.667}
    }""")
    draugr_amg_config_free(cfg2)

    # Cleanup
    draugr_amg_free(h)
    draugr_amg_config_free(cfg)
end

_warmup()
