module DraugrMetalExt

using Draugr
using Metal
using KernelAbstractions, LinearAlgebra

"""
    amg_setup(A::CSRMatrix{Tv, Ti, <:MtlVector, <:MtlVector, <:MtlVector}, config; backend, block_size) -> AMGHierarchy

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
function Draugr.amg_setup(A::CSRMatrix{Tv, Ti, <:MtlVector, <:MtlVector, <:MtlVector},
                               config::AMGConfig=AMGConfig();
                               backend=MetalBackend(),
                               block_size::Int=64) where {Tv, Ti}
    return invoke(Draugr.amg_setup, Tuple{CSRMatrix{Tv, Ti}, AMGConfig},
                  A, config; backend=backend, block_size=block_size)
end

"""
    amg_resetup!(hierarchy, A_new::CSRMatrix{Tv, Ti, <:MtlVector, <:MtlVector, <:MtlVector}, config)

AMG resetup for a `CSRMatrix` backed by Metal `MtlVector` arrays.
Uses backend and block_size from the hierarchy.
"""
function Draugr.amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                                  A_new::CSRMatrix{Tv, Ti, <:MtlVector, <:MtlVector, <:MtlVector},
                                  config::AMGConfig=AMGConfig()) where {Tv, Ti}
    backend = hierarchy.backend
    block_size = hierarchy.block_size
    A_csr = Draugr.csr_to_cpu(A_new)
    nlevels = length(hierarchy.levels)
    if nlevels == 0
        Draugr._update_coarse_solver!(hierarchy, A_csr; block_size=block_size)
        return hierarchy
    end
    # Copy nonzero values from new matrix into existing level 1
    level1 = hierarchy.levels[1]
    Draugr._copy_nzvals!(level1.A, A_csr; block_size=block_size)
    Draugr.update_smoother!(level1.smoother, level1.A; block_size=block_size)
    # Update subsequent levels via Galerkin products
    for lvl in 1:(nlevels - 1)
        level = hierarchy.levels[lvl]
        next_level = hierarchy.levels[lvl + 1]
        Draugr.galerkin_product!(next_level.A, level.A, level.P, level.R_map; block_size=block_size)
        Draugr.update_smoother!(next_level.smoother, next_level.A; block_size=block_size)
    end
    # Recompute coarsest dense matrix and LU
    last_level = hierarchy.levels[nlevels]
    Draugr._recompute_coarsest_dense!(hierarchy, last_level)
    hierarchy.coarse_factor = lu(hierarchy.coarse_A)
    return hierarchy
end

end # module
