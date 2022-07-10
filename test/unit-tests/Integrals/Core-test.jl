using Test
using Quiqbox
using Quiqbox: Fγ
using QuadGK: quadgk

@testset "Core.jl tests" begin

tolerance1 = 1e-8
tolerance2 = 5e-8
perturbStep = rand(-1e-1:2e-3:1e-1)
fNumInt = (γ, u) -> quadgk(t -> t^(2γ)*exp(-u*t^2), 0, 1; rtol=tolerance1)[1]
range = -10:(0.2+perturbStep):2
for γ in 0:24
    @test all([isapprox(fNumInt(γ, 10.0^e), Fγ(γ, 10.0^e), atol=tolerance2) 
                for e in range])
end

nuc = ["H", "F"]
nucCoords = [[-0.8664394281409157, 0.0, 0.0], [0.8664394281409157, 0.0, 0.0]]
b1 = genBasisFunc(nucCoords[1], GaussFunc(2.0, 1.0), "D", (true,))
b2 = genBasisFunc(nucCoords[2], ("STO-3G", "F"))
bfm1, bfm2 = b1 .+ b2[1:2]
bs_bf_bfs_bfm = [b1, bfm1, b2..., bfm2]

@test try eeInteractions(bs_bf_bfs_bfm); true catch; end
@test try coreH(bs_bf_bfs_bfm, nuc, nucCoords); true catch; end

end