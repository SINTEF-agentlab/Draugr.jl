# Draugr C API

This directory contains the C API and build scripts. It uses its own Project.toml so C-library builds do not pull in test or optional dependencies from the main project.

## Building the library

```bash
julia --project=c_api c_api/build_library.jl [output_dir]
```

`output_dir` defaults to `DraugrCompiled/` at repo root. The resulting tree includes the shared library, headers, and a minimal Julia runtime.