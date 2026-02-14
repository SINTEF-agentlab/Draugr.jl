module ParallelAMGMetalExt

using ParallelAMG
using Metal
using KernelAbstractions

"""
    amg_setup(A::CSRMatrix{Tv, Ti, <:MtlVector, <:MtlVector, <:MtlVector}, config; backend) -> AMGHierarchy

AMG setup for a `CSRMatrix` backed by Metal `MtlVector` arrays.
Automatically uses `MetalBackend()` as the default backend.

Since Metal does not have a native sparse CSR type, users should construct the
`CSRMatrix` directly from `MtlVector`s:

```julia
using Metal
rp = MtlVector(rowptr_cpu)
cv = MtlVector(colval_cpu)
nzv = MtlVector(nzval_cpu)
A = CSRMatrix(rp, cv, nzv, nrow, ncol)
hierarchy = amg_setup(A, config)
```
"""
function ParallelAMG.amg_setup(A::CSRMatrix{Tv, Ti, <:MtlVector, <:MtlVector, <:MtlVector},
                               config::AMGConfig=AMGConfig();
                               backend=MetalBackend()) where {Tv, Ti}
    return invoke(ParallelAMG.amg_setup, Tuple{CSRMatrix{Tv, Ti}, AMGConfig},
                  A, config; backend=backend)
end

"""
    amg_resetup!(hierarchy, A_new::CSRMatrix{Tv, Ti, <:MtlVector, <:MtlVector, <:MtlVector}, config)

AMG resetup for a `CSRMatrix` backed by Metal `MtlVector` arrays.
Automatically uses `MetalBackend()` as the default backend.
"""
function ParallelAMG.amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                                  A_new::CSRMatrix{Tv, Ti, <:MtlVector, <:MtlVector, <:MtlVector},
                                  config::AMGConfig=AMGConfig();
                                  backend=MetalBackend()) where {Tv, Ti}
    nlevels = length(hierarchy.levels)
    if nlevels == 0
        ParallelAMG._update_coarse_solver!(hierarchy, A_new; backend=backend)
        return hierarchy
    end
    # Copy nonzero values from new matrix into existing level 1
    level1 = hierarchy.levels[1]
    ParallelAMG._copy_nzvals!(level1.A, A_new; backend=backend)
    ParallelAMG.update_smoother!(level1.smoother, level1.A; backend=backend)
    # Update subsequent levels via Galerkin products
    for lvl in 1:(nlevels - 1)
        level = hierarchy.levels[lvl]
        next_level = hierarchy.levels[lvl + 1]
        ParallelAMG.galerkin_product!(next_level.A, level.A, level.P, level.R_map; backend=backend)
        ParallelAMG.update_smoother!(next_level.smoother, next_level.A; backend=backend)
    end
    # Recompute coarsest dense matrix and LU
    last_level = hierarchy.levels[nlevels]
    ParallelAMG._recompute_coarsest_dense!(hierarchy, last_level; backend=backend)
    copyto!(hierarchy.coarse_lu, hierarchy.coarse_A)
    LinearAlgebra.LAPACK.getrf!(hierarchy.coarse_lu, hierarchy.coarse_ipiv)
    hierarchy.coarse_factor = LinearAlgebra.LU(hierarchy.coarse_lu, hierarchy.coarse_ipiv, 0)
    return hierarchy
end

end # module
