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
    amg_resetup!(hierarchy, A_new::CSRMatrix{Tv, Ti, <:MtlVector, <:MtlVector, <:MtlVector}, config; partial=true, update_P=false)

AMG resetup for a `CSRMatrix` backed by Metal `MtlVector` arrays.
Converts to a CPU `CSRMatrix` and forwards to the main `CSRMatrix`-based resetup.
"""
function Draugr.amg_resetup!(hierarchy::AMGHierarchy{Tv, Ti},
                                  A_new::CSRMatrix{Tv, Ti, <:MtlVector, <:MtlVector, <:MtlVector},
                                  config::AMGConfig=AMGConfig();
                                  partial::Bool=true,
                                  update_P::Bool=false) where {Tv, Ti}
    A_csr = Draugr.csr_to_cpu(A_new)
    return Draugr.amg_resetup!(hierarchy, A_csr, config; partial=partial, update_P=update_P)
end

end # module
