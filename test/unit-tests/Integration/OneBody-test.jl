using Test
using Quiqbox

@testset "OneBody.jl tests" begin
    errT = 1e-10
    nucs = fill("H", 2)
    cens = [[-0.7, 0.0, 0.0], [ 0.7, 0.0, 0.0]]
    bf1 = genBasisFunc(cens[1], ("STO-3G", "H"))[]
    bf2 = genBasisFunc(cens[2], ("STO-3G", "H"))[]
    bs = [bf1, bf2]

    S = [1.0 0.6593182058508896; 0.6593182058508896 1.0]
    @test isapprox(overlap(bf1, bf1), ones(1,1,1), atol=errT)
    @test isapprox(overlap(bf1, bf2), S[2]*ones(1,1,1), atol=errT)
    @test isapprox(overlaps(bs), cat(S; dims = ndims(S) + 1))

    V = [-1.8804408905227634 -1.1948346220535715; -1.1948346220535715 -1.8804408905227632]
    @test isapprox(nucAttraction(bf1, bf2, nucs, cens), V[2]*ones(1,1,1), atol=errT)
    @test isapprox(nucAttractions(bs, nucs, cens), cat(V; dims = ndims(V) + 1), atol=errT)

    T = [0.7600318799755844 0.23645465829079276; 0.23645465829079276 0.7600318799755844]
    @test isapprox(elecKinetic(bf1, bf2), T[2]*ones(1,1,1), atol=errT)
    @test isapprox(elecKinetics(bs), cat(T; dims = ndims(T) + 1))

    @test isapprox(coreH(bs, nucs, cens), cat(T+V; dims = ndims(V) + 1), atol=errT)
end