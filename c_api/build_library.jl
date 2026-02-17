# build_library.jl — Compile Draugr into a shared library
#
# Usage:
#   julia --project=/path/to/Draugr.jl c_api/build_library.jl [output_dir]
#
# Output directory defaults to DraugrCompiled/ next to this script.

using Pkg
Pkg.instantiate()

# PackageCompiler must be available in the default environment or this project
try
    @eval using PackageCompiler
catch
    @info "Installing PackageCompiler..."
    Pkg.add("PackageCompiler")
    @eval using PackageCompiler
end

src_dir = dirname(dirname(@__FILE__))  # Draugr.jl root
out_dir = length(ARGS) >= 1 ? ARGS[1] : joinpath(src_dir, "DraugrCompiled")
precompile_file = joinpath(src_dir, "c_api", "generate_precompile.jl")

# Keep defaults fast for local iteration:
#   - incremental=true reuses the base sysimage and is usually much faster.
#   - force=false avoids needless full replacement when not necessary.
# Override with:
#   DRAUGR_FORCE_BUILD=1          (force rebuild)
#   DRAUGR_INCREMENTAL=1          (enable incremental mode)
#   DRAUGR_SKIP_PRECOMPILE=1      (skip warmup script)
force_build = get(ENV, "DRAUGR_FORCE_BUILD", "1") == "1"
incremental = get(ENV, "DRAUGR_INCREMENTAL", "1") == "1"
skip_precompile = get(ENV, "DRAUGR_SKIP_PRECOMPILE", "0") == "1"

@info "Building Draugr shared library" src_dir out_dir incremental force_build skip_precompile

create_library(
    src_dir, out_dir;
    lib_name = "libdraugr",
    precompile_execution_file = skip_precompile ? String[] : [precompile_file],
    header_files = [joinpath(src_dir, "c_api", "draugr_amg.h")],
    incremental = incremental,
    filter_stdlibs = true,
    force = force_build,
)

@info "Library built successfully" out_dir
println("\nFiles in $(joinpath(out_dir, "lib")):")
for f in readdir(joinpath(out_dir, "lib"))
    println("  ", f)
end
