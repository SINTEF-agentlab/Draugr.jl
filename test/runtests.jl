using Test
using Draugr
import Draugr: colvals
using SparseArrays
using LinearAlgebra
using Random
import Jutul
using Jutul.StaticCSR: StaticSparsityMatrixCSR, static_sparsity_sparse

include("helpers.jl")

@testset "Draugr" begin
    include("test_csr_matrix.jl")
    include("test_coarsening.jl")
    include("test_prolongation_galerkin.jl")
    include("test_smoothers.jl")
    include("test_amg.jl")
    include("test_jutul.jl")
    include("test_jlarrays.jl")
    include("test_extensions.jl")
    include("test_c_api.jl")
    include("test_real_matrices.jl")
end
