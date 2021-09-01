using Test
using Quiqbox

@testset "OneBody.jl tests" begin
    errT1 = 1e-10
    errT2 = 1e-10
    nucs = fill("H", 2)
    cens = [[-0.7, 0.0, 0.0], [ 0.7, 0.0, 0.0]]
    bf1 = genBasisFunc(cens[1], ("STO-3G", "H"))[]
    bf2 = genBasisFunc(cens[2], ("STO-3G", "H"))[]
    bs = [bf1, bf2]

    S = [1.0 0.6593182058508896; 0.6593182058508896 1.0]
    @test isapprox(overlap(bf1, bf1), ones(1,1), atol=errT1)
    @test isapprox(overlap(bf1, bf2), S[2]*ones(1,1), atol=errT1)
    @test isapprox(overlaps(bs), S)

    V = [-1.8804408905227634 -1.1948346220535715; -1.1948346220535715 -1.8804408905227632]
    @test isapprox(nucAttraction(bf1, bf2, nucs, cens), V[2]*ones(1,1), atol=errT1)
    @test isapprox(nucAttractions(bs, nucs, cens), V, atol=errT1)

    T = [0.7600318799755844 0.23645465829079276; 0.23645465829079276 0.7600318799755844]
    @test isapprox(elecKinetic(bf1, bf2), T[2]*ones(1,1,1), atol=errT1)
    @test isapprox(elecKinetics(bs), T)

    Hc = coreH(bs, nucs, cens)
    @test isapprox(Hc, T+V, atol=errT1)
    @test isapprox(coreHij(bf1, bf1, nucs, cens)[], Hc[1], atol=errT2)
    @test isapprox(coreHij(bf2, bf1, nucs, cens)[], 
                   coreHij(bf1, bf2, nucs, cens)[], atol=errT2)
    @test isapprox(coreHij(bf2, bf1, nucs, cens)[], Hc[3], atol=errT2)
    @test isapprox(coreHij(bf2, bf2, nucs, cens)[], Hc[4], atol=errT2)
end