using Test
using Quiqbox

@testset "OneBody.jl tests" begin

errT1 = 1e-10
errT2 = 1e-10
errT3 = 1e-10
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
@test isapprox(neAttraction(bf1, bf2, nucs, cens), V[2]*ones(1,1), atol=errT1)
@test isapprox(neAttractions(bs, nucs, cens), V, atol=errT1)

T = [0.7600318799755844 0.23645465829079276; 0.23645465829079276 0.7600318799755844]
@test isapprox(eKinetic(bf1, bf2), T[2]*ones(1,1,1), atol=errT1)
@test isapprox(eKinetics(bs), T)

Hc = coreH(bs, nucs, cens)
@test isapprox(Hc, T+V, atol=errT1)
@test isapprox(coreHij(bf1, bf1, nucs, cens)[], Hc[1], atol=errT2)
@test isapprox(coreHij(bf2, bf1, nucs, cens)[], 
                coreHij(bf1, bf2, nucs, cens)[], atol=errT2)
@test isapprox(coreHij(bf2, bf1, nucs, cens)[], Hc[3], atol=errT2)
@test isapprox(coreHij(bf2, bf2, nucs, cens)[], Hc[4], atol=errT2)

bfs = genBasisFunc([0,1,0], (3,2), "D")
S = [0.05500737317448707; 0.00788990951420727; 
    -0.0;                 0.06075573582055236; 
    -0.0;                 0.04948443651454198]
@test isapprox(overlap(bfs, bf1), S, atol=errT3)

T = [0.031092263857623516; 0.0244106493741589; 
        0.0;                  0.0488771655445106; 
        0.0;                  0.0140048092957122] |> transpose
@test isapprox(eKinetic(bf1, bfs), T, atol=errT3)

V = [
    -0.05113501486 0.0           0.0          -0.01876634304 0.0          -0.01616252693; 
    0.0          -0.01876634304 0.0           0.0           0.0           0.0; 
    0.0           0.0          -0.01616252693 0.0           0.0           0.0; 
    -0.01876634304 0.0           0.0          -0.05728524140 0.0          -0.01687630012; 
    0.0           0.0           0.0           0.0          -0.01687630012 0.0; 
    -0.01616252693 0.0           0.0          -0.01687630012 0.0          -0.04643023450
    ]
@test isapprox(neAttraction(bfs, bfs, nucs, cens), V, atol=errT3)

end