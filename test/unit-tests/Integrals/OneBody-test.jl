using Test
using Quiqbox

@testset "OneBody.jl tests" begin

errT1 = 1e-10
errT2 = 1e-10
errT3 = 1e-10
nucs = fill("H", 2)
cens = [[-0.7, 0.0, 0.0], [ 0.7, 0.0, 0.0]]
bf1 = genBasisFunc(cens[1], "STO-3G", "H")[]
bf2 = genBasisFunc(cens[2], "STO-3G", "H")[]
bs1 = [bf1, bf2]
gf1 = GaussFunc(1.0, 0.5)
bs2 = genBasisFunc.(cens, Ref(gf1), ["S", "P"])

S1 = [1.0 0.6593182058508896; 0.6593182058508896 1.0]
@test isapprox(overlap(bf1, bf1), 1.0, atol=errT1)
@test isapprox(overlap(bf1, bf2), S1[2], atol=errT1)
@test isapprox(overlaps(bs1), S1)
S2 = [ 0.49217531080382560 -0.1293031997077190 0.0 0.0; 
      -0.12930319970771903  0.1230438277009564 0.0 0.0; 
       0.0 0.0 0.1230438277009564 0.0; 
       0.0 0.0 0.0 0.1230438277009564]
@test isapprox(overlaps(bs2), S2)


T1 = [0.7600318799755844 0.23645465829079276; 0.23645465829079276 0.7600318799755844]
@test isapprox(eKinetic(bf1, bf2), T1[2], atol=errT1)
@test isapprox(eKinetic(bf2, bf1), T1[2], atol=errT1)
@test isapprox(eKinetics(bs1), T1)
T2 = [ 0.73826296620573840 -0.19654086355573297 0.0 0.0; 
      -0.19654086355573297  0.30760956925239100 0.0 0.0; 
      0.0 0.0 0.307609569252391 0.0; 
      0.0 0.0 0.0 0.307609569252391]
@test isapprox(eKinetics(bs2), T2)

V1 = [-1.8804408905227634 -1.1948346220535715; -1.1948346220535715 -1.8804408905227632]
@test isapprox(neAttraction(bf1, bf2, nucs, cens), V1[2], atol=errT1)
@test isapprox(neAttraction(bf2, bf1, nucs, cens), V1[2], atol=errT1)
@test isapprox(neAttractions(bs1, nucs, cens), V1, atol=errT1)
V2 = [-1.1351554253080056  0.3097686023841543 0.0 0.0; 
       0.3097686023841543 -0.2357553434205944 0.0 0.0; 
       0.0 0.0 -0.20768294743402105 0.0; 
       0.0 0.0 0.0 -0.20768294743402105]
@test isapprox(neAttractions(bs2, nucs, cens), V2, atol=errT1)

Hc = coreH(bs1, nucs, cens)
@test isapprox(Hc, T1+V1, atol=errT1)
@test isapprox(coreHij(bf1, bf1, nucs, cens)[], Hc[1], atol=errT2)
@test isapprox(coreHij(bf2, bf1, nucs, cens)[], 
                coreHij(bf1, bf2, nucs, cens)[], atol=errT2)
@test isapprox(coreHij(bf2, bf1, nucs, cens)[], Hc[3], atol=errT2)
@test isapprox(coreHij(bf2, bf2, nucs, cens)[], Hc[4], atol=errT2)

bfs = genBasisFunc([0.0, 1.0, 0.0], (3.0, 2.0), "D")
S3 = [0.05500737317448707; 0.00788990951420727; 
      0.0;                 0.06075573582055236; 
      0.0;                 0.04948443651454198]
@test isapprox(overlap.(bfs, bf1), S3, atol=errT3)

T3 = [0.031092263857623516; 0.0244106493741589; 
      0.0;                  0.0488771655445106; 
      0.0;                  0.0140048092957122]
@test isapprox(eKinetic.(bf1, bfs), T3, atol=errT3)

V3 = [
     -0.05113501486 0.0           0.0          -0.01876634304 0.0          -0.01616252693; 
      0.0          -0.01876634304 0.0           0.0           0.0           0.0; 
      0.0           0.0          -0.01616252693 0.0           0.0           0.0; 
     -0.01876634304 0.0           0.0          -0.05728524140 0.0          -0.01687630012; 
      0.0           0.0           0.0           0.0          -0.01687630012 0.0; 
     -0.01616252693 0.0           0.0          -0.01687630012 0.0          -0.04643023450
     ]
@test isapprox(neAttractions((bfs,), nucs, cens), V3, atol=errT3)

bfF32_1 = genBasisFunc(Float32[1.0, 1.0, 1.0], (1.2f0, 2.3f0), :p)
bfF32_2 = genBasisFunc(Float32[1.0, 3.0, 0.0], ([1.25f0, 5.12f0], [2.3f0, -1.1f0]))
bsF32 = [bfF32_1, bfF32_2]
H2 = ["H", "H"]
H2coords = [[0.4f0, 0.0f0, 0.0f0], [-0.4f0, 0.02f0, 0.0f0]]
@test coreH(bsF32, H2, H2coords) isa Matrix{Float32}

end