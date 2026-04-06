@testset "JSON config parsing" begin

    function _config_from_json(json)
        h = draugr_amg_config_from_json(json)
        @assert h > 0 "Config creation failed: $(unsafe_string(draugr_amg_last_error()))"
        config = Draugr._CONFIG_HANDLES[h]
        draugr_amg_config_free(h)
        return config
    end

    @testset "empty JSON produces AMGConfig() defaults" begin
        c = _config_from_json("{}")
        d = AMGConfig()
        @test typeof(c.coarsening) == typeof(d.coarsening)
        @test c.coarsening.θ        == d.coarsening.θ
        @test typeof(c.smoother)    == typeof(d.smoother)
        @test c.jacobi_omega        ≈  d.jacobi_omega
        @test c.max_levels          == d.max_levels
        @test c.max_coarse_size     == d.max_coarse_size
        @test c.pre_smoothing_steps == d.pre_smoothing_steps
        @test c.post_smoothing_steps == d.post_smoothing_steps
        @test c.verbose             == d.verbose
        @test c.initial_coarsening_levels == d.initial_coarsening_levels
        @test c.max_row_sum         ≈  d.max_row_sum
        @test c.cycle_type          == d.cycle_type
        @test typeof(c.strength_type) == typeof(d.strength_type)
        @test c.coarse_solve_on_cpu == d.coarse_solve_on_cpu
    end

    @testset "scalar fields" begin
        c = _config_from_json("""{
            "theta": 0.3,
            "max_levels": 10,
            "max_coarse_size": 25,
            "pre_smoothing_steps": 2,
            "post_smoothing_steps": 3,
            "jacobi_omega": 0.8,
            "verbose": 2,
            "initial_coarsening_levels": 3,
            "max_row_sum": 0.9,
            "coarse_solve_on_cpu": true
        }""")
        @test c.coarsening.θ         ≈ 0.3
        @test c.max_levels           == 10
        @test c.max_coarse_size      == 25
        @test c.pre_smoothing_steps  == 2
        @test c.post_smoothing_steps == 3
        @test c.jacobi_omega         ≈ 0.8
        @test c.verbose              == 2
        @test c.initial_coarsening_levels == 3
        @test c.max_row_sum          ≈ 0.9
        @test c.coarse_solve_on_cpu  == true
    end

    @testset "string-encoded values" begin
        c = _config_from_json("""{
            "theta": "0.3",
            "max_levels": "10",
            "jacobi_omega": "0.8",
            "coarse_solve_on_cpu": "true",
            "verbose": "2"
        }""")
        @test c.coarsening.θ  ≈ 0.3
        @test c.max_levels   == 10
        @test c.jacobi_omega  ≈ 0.8
        @test c.coarse_solve_on_cpu == true
        @test c.verbose      == 2
    end

    @testset "coarsening types" begin
        for (name, T) in [
                ("hmis",                 HMISCoarsening),
                ("pmis",                 PMISCoarsening),
                ("rs",                   RSCoarsening),
                ("aggregation",          AggregationCoarsening),
                ("smoothed_aggregation", SmoothedAggregationCoarsening),
            ]
            c = _config_from_json("""{"coarsening": "$name"}""")
            @test typeof(c.coarsening) == T
        end
        for (name, base) in [("aggressive_pmis", :pmis), ("aggressive_hmis", :hmis)]
            c = _config_from_json("""{"coarsening": "$name"}""")
            @test typeof(c.coarsening) == AggressiveCoarsening
            @test c.coarsening.base == base
        end
    end

    @testset "smoother types" begin
        for (name, T) in [
                ("jacobi",        JacobiSmootherType),
                ("colored_gs",    ColoredGaussSeidelType),
                ("serial_gs",     SerialGaussSeidelType),
                ("spai0",         SPAI0SmootherType),
                ("spai1",         SPAI1SmootherType),
                ("l1_jacobi",     L1JacobiSmootherType),
                ("l1_colored_gs", L1ColoredGaussSeidelType),
                ("l1_serial_gs",  L1SerialGaussSeidelType),
                ("chebyshev",     ChebyshevSmootherType),
                ("ilu0",          ILU0SmootherType),
                ("serial_ilu0",   SerialILU0SmootherType),
            ]
            c = _config_from_json("""{"smoother": "$name"}""")
            @test typeof(c.smoother) == T
        end
    end

    @testset "smoother with omega override" begin
        c = _config_from_json("""{"smoother": {"type": "jacobi", "omega": 0.42}}""")
        @test typeof(c.smoother) == JacobiSmootherType
        @test c.jacobi_omega ≈ 0.42
    end

    @testset "interpolation — string names" begin
        for (name, T) in [
                ("direct",     DirectInterpolation),
                ("standard",   StandardInterpolation),
                ("extended_i", ExtendedIInterpolation),
            ]
            c = _config_from_json("""{"interpolation": "$name"}""")
            @test typeof(c.coarsening.interpolation) == T
        end
    end

    @testset "interpolation — nested extended_i" begin
        c = _config_from_json("""{
            "interpolation": {
                "type": "extended_i",
                "trunc_factor": 0.3,
                "max_elements": 5,
                "norm_p": 2,
                "rescale": true
            }
        }""")
        ip = c.coarsening.interpolation
        @test typeof(ip) == ExtendedIInterpolation
        @test ip.trunc_factor ≈ 0.3
        @test ip.max_elements == 5
        @test ip.norm_p       == 2
        @test ip.rescale      == true
    end

    @testset "interpolation — nested direct with trunc_factor" begin
        c = _config_from_json("""{"interpolation": {"type": "direct", "trunc_factor": 0.1}}""")
        ip = c.coarsening.interpolation
        @test typeof(ip) == DirectInterpolation
        @test ip.trunc_factor ≈ 0.1
    end

    @testset "interpolation — nested defaults from type constructors" begin
        c = _config_from_json("""{"interpolation": {"type": "extended_i"}}""")
        ip = c.coarsening.interpolation
        ed = ExtendedIInterpolation()
        @test ip.trunc_factor ≈ ed.trunc_factor
        @test ip.max_elements == ed.max_elements
        @test ip.norm_p       == ed.norm_p
        @test ip.rescale      == ed.rescale
    end

    @testset "strength types" begin
        c1 = _config_from_json("""{"strength": "absolute"}""")
        @test typeof(c1.strength_type) == AbsoluteStrength
        c2 = _config_from_json("""{"strength": "signed"}""")
        @test typeof(c2.strength_type) == SignedStrength
    end

    @testset "cycle types" begin
        c1 = _config_from_json("""{"cycle": "v"}""")
        @test c1.cycle_type == :V
        c2 = _config_from_json("""{"cycle": "w"}""")
        @test c2.cycle_type == :W
    end

    @testset "initial_coarsening" begin
        c = _config_from_json("""{
            "coarsening": "hmis",
            "initial_coarsening": "pmis",
            "initial_coarsening_levels": 2
        }""")
        @test typeof(c.coarsening) == HMISCoarsening
        @test typeof(c.initial_coarsening) == PMISCoarsening
        @test c.initial_coarsening_levels == 2
    end

    @testset "initial_coarsening defaults to coarsening" begin
        c = _config_from_json("""{"coarsening": "rs"}""")
        @test typeof(c.coarsening) == RSCoarsening
        @test typeof(c.initial_coarsening) == RSCoarsening
    end

    @testset "unknown keys are ignored" begin
        c = _config_from_json("""{"type": "draugr", "setup_frequency": 30, "theta": 0.4}""")
        @test c.coarsening.θ ≈ 0.4
    end

    @testset "invalid type names return -1" begin
        for json in [
                """{"coarsening": "bogus"}""",
                """{"smoother": "nope"}""",
                """{"interpolation": "bad"}""",
                """{"strength": "wrong"}""",
                """{"cycle": "z"}""",
            ]
            h = @test_logs (:error,) draugr_amg_config_from_json(json)
            @test h == -1
            err = unsafe_string(draugr_amg_last_error())
            @test length(err) > 0
        end
    end

    @testset "invalid JSON returns -1" begin
        h = @test_logs (:error,) draugr_amg_config_from_json("{not valid json")
        @test h == -1
    end

    @testset "handle lifecycle" begin
        h = draugr_amg_config_from_json("{}")
        @test h > 0
        @test haskey(Draugr._CONFIG_HANDLES, h)
        @test draugr_amg_config_free(h) == 0
        @test !haskey(Draugr._CONFIG_HANDLES, h)
        @test draugr_amg_config_free(h) == -1
    end

    @testset "case insensitive" begin
        c = _config_from_json("""{"coarsening": "HMIS", "smoother": "L1_Colored_GS", "cycle": "V"}""")
        @test typeof(c.coarsening) == HMISCoarsening
        @test typeof(c.smoother) == L1ColoredGaussSeidelType
        @test c.cycle_type == :V
    end

    @testset "full realistic config" begin
        c = _config_from_json("""{
            "coarsening": "hmis",
            "interpolation": {"type": "extended_i", "trunc_factor": 0.0, "max_elements": 5, "norm_p": 2, "rescale": true},
            "smoother": "serial_gs",
            "strength": "absolute",
            "cycle": "v",
            "theta": 0.5,
            "jacobi_omega": 0.6666666666666666,
            "max_levels": 20,
            "max_coarse_size": 50,
            "pre_smoothing_steps": 1,
            "post_smoothing_steps": 1,
            "initial_coarsening": "aggressive_pmis",
            "initial_coarsening_levels": 1,
            "max_row_sum": 1.0,
            "coarse_solve_on_cpu": false,
            "verbose": 0
        }""")
        @test typeof(c.coarsening)           == HMISCoarsening
        @test c.coarsening.θ                  ≈ 0.5
        ip = c.coarsening.interpolation
        @test typeof(ip)                     == ExtendedIInterpolation
        @test ip.max_elements                == 5
        @test ip.norm_p                      == 2
        @test ip.rescale                     == true
        @test typeof(c.smoother)             == SerialGaussSeidelType
        @test typeof(c.strength_type)        == AbsoluteStrength
        @test c.cycle_type                   == :V
        @test c.jacobi_omega                  ≈ 2/3
        @test typeof(c.initial_coarsening)   == AggressiveCoarsening
        @test c.initial_coarsening.base      == :pmis
        @test c.initial_coarsening_levels    == 1
    end

end

@testset "C API lifecycle" begin
    # Build a 1D Laplacian in raw CSR format (1-based)
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

    cfg = draugr_amg_config_from_json("{}")
    @test cfg > 0

    @testset "setup + solve + cycle + free" begin
        h = draugr_amg_setup(Int32(n), nnz, pointer(rowptr), pointer(colval),
                             pointer(nzval), cfg, Int32(1), Int32(1))
        @test h > 0

        b = ones(Float64, n)
        x = zeros(Float64, n)
        niter = draugr_amg_solve(h, Int32(n), pointer(x), pointer(b),
                                 cfg, 1e-10, Int32(200))
        @test niter > 0
        @test niter < 200

        fill!(x, 0.0)
        ret = draugr_amg_cycle(h, Int32(n), pointer(x), pointer(b), cfg)
        @test ret == 0
        @test norm(x) > 0

        @test draugr_amg_free(h) == 0
        @test draugr_amg_free(h) == -1
    end

    @testset "setup with allow_partial_resetup=0" begin
        h = draugr_amg_setup(Int32(n), nnz, pointer(rowptr), pointer(colval),
                             pointer(nzval), cfg, Int32(1), Int32(0))
        @test h > 0
        hierarchy = Draugr._HIERARCHY_HANDLES[h]
        for lvl in hierarchy.levels
            @test lvl.R_map === nothing
        end
        draugr_amg_free(h)
    end

    @testset "partial resetup (coefficient-only)" begin
        h = draugr_amg_setup(Int32(n), nnz, pointer(rowptr), pointer(colval),
                             pointer(nzval), cfg, Int32(1), Int32(1))
        @test h > 0
        nzval2 = copy(nzval)
        nzval2 .*= 1.5
        ret = draugr_amg_resetup(h, Int32(n), nnz, pointer(rowptr), pointer(colval),
                                 pointer(nzval2), cfg, Int32(1), Int32(1), Int32(1))
        @test ret == 0

        b = ones(Float64, n)
        x = zeros(Float64, n)
        niter = draugr_amg_solve(h, Int32(n), pointer(x), pointer(b),
                                 cfg, 1e-10, Int32(200))
        @test niter > 0
        @test niter < 200
        draugr_amg_free(h)
    end

    @testset "full resetup (partial=0)" begin
        h = draugr_amg_setup(Int32(n), nnz, pointer(rowptr), pointer(colval),
                             pointer(nzval), cfg, Int32(1), Int32(1))
        @test h > 0
        nzval2 = copy(nzval)
        nzval2 .*= 2.0
        ret = draugr_amg_resetup(h, Int32(n), nnz, pointer(rowptr), pointer(colval),
                                 pointer(nzval2), cfg, Int32(1), Int32(0), Int32(1))
        @test ret == 0
        hierarchy = Draugr._HIERARCHY_HANDLES[h]
        for lvl in hierarchy.levels
            @test lvl.R_map !== nothing
        end

        b = ones(Float64, n)
        x = zeros(Float64, n)
        niter = draugr_amg_solve(h, Int32(n), pointer(x), pointer(b),
                                 cfg, 1e-10, Int32(200))
        @test niter > 0
        @test niter < 200
        draugr_amg_free(h)
    end

    @testset "full resetup with allow_partial_resetup=0" begin
        h = draugr_amg_setup(Int32(n), nnz, pointer(rowptr), pointer(colval),
                             pointer(nzval), cfg, Int32(1), Int32(1))
        @test h > 0
        ret = draugr_amg_resetup(h, Int32(n), nnz, pointer(rowptr), pointer(colval),
                                 pointer(nzval), cfg, Int32(1), Int32(0), Int32(0))
        @test ret == 0
        hierarchy = Draugr._HIERARCHY_HANDLES[h]
        for lvl in hierarchy.levels
            @test lvl.R_map === nothing
        end
        draugr_amg_free(h)
    end

    @testset "full resetup then partial resetup" begin
        h = draugr_amg_setup(Int32(n), nnz, pointer(rowptr), pointer(colval),
                             pointer(nzval), cfg, Int32(1), Int32(0))
        @test h > 0
        # Full resetup with allow_partial_resetup=1 to enable future partial
        ret = draugr_amg_resetup(h, Int32(n), nnz, pointer(rowptr), pointer(colval),
                                 pointer(nzval), cfg, Int32(1), Int32(0), Int32(1))
        @test ret == 0
        hierarchy = Draugr._HIERARCHY_HANDLES[h]
        for lvl in hierarchy.levels
            @test lvl.R_map !== nothing
        end
        # Now partial resetup should work
        nzval2 = copy(nzval)
        nzval2 .*= 1.1
        ret = draugr_amg_resetup(h, Int32(n), nnz, pointer(rowptr), pointer(colval),
                                 pointer(nzval2), cfg, Int32(1), Int32(1), Int32(1))
        @test ret == 0

        b = ones(Float64, n)
        x = zeros(Float64, n)
        niter = draugr_amg_solve(h, Int32(n), pointer(x), pointer(b),
                                 cfg, 1e-10, Int32(200))
        @test niter > 0
        @test niter < 200
        draugr_amg_free(h)
    end

    @testset "zero-based indexing" begin
        rp0 = rowptr .- Int32(1)
        cv0 = colval .- Int32(1)
        h = draugr_amg_setup(Int32(n), nnz, pointer(rp0), pointer(cv0),
                             pointer(nzval), cfg, Int32(0), Int32(1))
        @test h > 0

        b = ones(Float64, n)
        x = zeros(Float64, n)
        niter = draugr_amg_solve(h, Int32(n), pointer(x), pointer(b),
                                 cfg, 1e-10, Int32(200))
        @test niter > 0
        @test niter < 200
        draugr_amg_free(h)
    end

    @testset "invalid handles" begin
        ret = draugr_amg_resetup(Int32(9999), Int32(n), nnz, pointer(rowptr),
                                 pointer(colval), pointer(nzval), cfg,
                                 Int32(1), Int32(1), Int32(1))
        @test ret == -1
        err = unsafe_string(draugr_amg_last_error())
        @test length(err) > 0

        b = ones(Float64, n)
        x = zeros(Float64, n)
        ret = draugr_amg_solve(Int32(9999), Int32(n), pointer(x), pointer(b),
                               cfg, 1e-10, Int32(50))
        @test ret == -1

        ret = draugr_amg_cycle(Int32(9999), Int32(n), pointer(x), pointer(b), cfg)
        @test ret == -1

        @test draugr_amg_free(Int32(9999)) == -1
    end

    @testset "@cfunction pointers" begin
        ptrs = draugr_amg_get_cfunctions()
        @test ptrs.config_from_json isa Ptr
        @test ptrs.last_error isa Ptr
        @test ptrs.setup isa Ptr
        @test ptrs.resetup isa Ptr
        @test ptrs.solve isa Ptr
        @test ptrs.cycle isa Ptr
        @test ptrs.free isa Ptr
        @test ptrs.config_free isa Ptr
        for ptr in ptrs
            @test ptr != C_NULL
        end
    end

    draugr_amg_config_free(cfg)
end
