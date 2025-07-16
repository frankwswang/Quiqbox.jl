using Test
using Quiqbox
using Quiqbox: PrimGaussTypeOrbInfo, GaussProductInfo, GaussCoulombFieldCache, 
               computePGTOrbOneBodyRepulsion!, computePGTOrbTwoBodyRepulsion!
using LinearAlgebra: norm

@testset "Coulomb-Interaction-Based Features" begin

gtoData1 = PrimGaussTypeOrbInfo((0.1, 0.2, 0.3), 2.0, (1, 7, 2))
gtoData2 = PrimGaussTypeOrbInfo((0.1, 0.2, 0.3), 2.0, (1, 1, 0))
gtoData3 = PrimGaussTypeOrbInfo((0.1, 0.2, 0.3), 2.0, (0, 0, 0))
gtoData4 = PrimGaussTypeOrbInfo((0.3, 0.1, 0.5), 2.0, (0, 0, 0))
gtoData5 = PrimGaussTypeOrbInfo((0.3, 0.1, 0.5), 2.0, (2, 0, 0))
gtoData6 = PrimGaussTypeOrbInfo((0.1, 0.2, 0.3), 2.0, (2, 0, 0))

gtoProd11 = GaussProductInfo((gtoData1, gtoData1))
gtoProd12 = GaussProductInfo((gtoData1, gtoData2))
gtoProd21 = GaussProductInfo((gtoData2, gtoData1))
gtoProd22 = GaussProductInfo((gtoData2, gtoData2))
gtoProd33 = GaussProductInfo((gtoData3, gtoData3))
gtoProd34 = GaussProductInfo((gtoData3, gtoData4))
gtoProd35 = GaussProductInfo((gtoData3, gtoData5))
gtoProd36 = GaussProductInfo((gtoData3, gtoData6))
gtoProd44 = GaussProductInfo((gtoData4, gtoData4))
gtoProd63 = GaussProductInfo((gtoData6, gtoData3))

coord = (0., 0., 0.)

emptyCache = GaussCoulombFieldCache(Float64, Count(3))
sizedCache = GaussCoulombFieldCache(Float64, Count(3), Quiqbox.True())

@test computePGTOrbOneBodyRepulsion!(emptyCache, coord, gtoProd12) == 
      computePGTOrbOneBodyRepulsion!(sizedCache, coord, gtoProd12) ≈ 0.00021406291700540685
@test computePGTOrbOneBodyRepulsion!(emptyCache, coord, gtoProd11) == 
      computePGTOrbOneBodyRepulsion!(sizedCache, coord, gtoProd11) ≈ 0.00016035624620095473
@test computePGTOrbOneBodyRepulsion!(emptyCache, coord, gtoProd33) == 
      computePGTOrbOneBodyRepulsion!(sizedCache, coord, gtoProd33) ≈ 1.3209276479060006
@test computePGTOrbOneBodyRepulsion!(emptyCache, coord, gtoProd34) == 
      computePGTOrbOneBodyRepulsion!(sizedCache, coord, gtoProd34) ≈ 1.102953813735257
@test computePGTOrbOneBodyRepulsion!(emptyCache, coord, gtoProd35) == 
      computePGTOrbOneBodyRepulsion!(sizedCache, coord, gtoProd35) ≈ 0.1305787950084035
@test computePGTOrbOneBodyRepulsion!(emptyCache, coord, gtoProd63) == 
      computePGTOrbOneBodyRepulsion!(sizedCache, coord, gtoProd63) == 
      computePGTOrbOneBodyRepulsion!(emptyCache, coord, gtoProd36) == 
      computePGTOrbOneBodyRepulsion!(sizedCache, coord, gtoProd36) ≈ 0.11995218441914622


gtoData7 = PrimGaussTypeOrbInfo((0.9, 0.6, 0.1), 2.5, (1, 1, 0))
gtoData8 = PrimGaussTypeOrbInfo((0.6, 0.7, 0.8), 3.0, (3, 1, 2))

gtoProd14 = GaussProductInfo((gtoData1, gtoData4))
gtoProd41 = GaussProductInfo((gtoData4, gtoData1))
gtoProd78 = GaussProductInfo((gtoData7, gtoData8))
gtoProd87 = GaussProductInfo((gtoData8, gtoData7))

@test computePGTOrbTwoBodyRepulsion!(sizedCache, gtoProd11, gtoProd22) ==
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd11, gtoProd22) ≈ 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd22, gtoProd11) ≈ 
      1.7675350484831864e-6

@test computePGTOrbTwoBodyRepulsion!(sizedCache, gtoProd12, gtoProd12) == 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd12, gtoProd12) ≈ 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd12, gtoProd21) ≈ 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd21, gtoProd21) ≈ 
      6.267963629018787e-8

@test computePGTOrbTwoBodyRepulsion!(sizedCache, gtoProd33, gtoProd44) == 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd33, gtoProd44) ≈ 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd44, gtoProd33) ≈ 
      0.7291219052871128

@test computePGTOrbTwoBodyRepulsion!(sizedCache, gtoProd14, gtoProd78) == 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd14, gtoProd78) ≈ 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd41, gtoProd78) ≈ 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd41, gtoProd87) ≈ 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd14, gtoProd87) ≈ 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd78, gtoProd14) ≈ 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd87, gtoProd14) ≈ 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd87, gtoProd41) ≈ 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd78, gtoProd41) ≈ 
      -2.4175946692430508e-9


bf1 = genGaussTypeOrb((0.1, 0.2, 0.3), 2.0, (1, 7, 2))
bf2 = genGaussTypeOrb((0.1, 0.2, 0.3), 2.0, (1, 1, 0))
bf3 = genGaussTypeOrb((0.3, 0.1, 0.5), 2.0, (0, 0, 0))
bf4 = genGaussTypeOrb((0.9, 0.6, 0.1), 2.5, (1, 1, 0))
bf5 = genGaussTypeOrb((0.6, 0.7, 0.8), 3.0, (3, 1, 2))

nucs1 = [:He, :H]
coords1 = [(0., 0., 0.), (0., 0., 0.)]
V1 = neAttractions(nucs1, coords1, [bf1, bf2])
@test V1[2] == V1[3] == neAttraction(nucs1, coords1, bf1, bf2) == 
      -3computePGTOrbOneBodyRepulsion!(emptyCache, coord, gtoProd12)
@test V1[1] == neAttraction(nucs1, coords1, bf1, bf1) == 
      -3computePGTOrbOneBodyRepulsion!(emptyCache, coord, gtoProd11)
@test V1[4] == neAttraction(nucs1, coords1, bf2, bf2) == 
      -3computePGTOrbOneBodyRepulsion!(emptyCache, coord, gtoProd22)

ERI1 = eeInteractions([bf1, bf3, bf4, bf5])
@test ERI1[1, 2, 3, 4] == 
      eeInteraction(bf1, bf3, bf4, bf5) ≈ 
      eeInteraction(bf1, bf3, bf5, bf4) ≈ 
      eeInteraction(bf3, bf1, bf5, bf4) ≈ 
      eeInteraction(bf3, bf1, bf4, bf5) ≈ 
      eeInteraction(bf4, bf5, bf1, bf3) ≈ 
      eeInteraction(bf4, bf5, bf3, bf1) ≈ 
      eeInteraction(bf5, bf4, bf3, bf1) ≈ 
      eeInteraction(bf5, bf4, bf1, bf3) ≈ 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd14, gtoProd78)
@test ERI1[1] == eeInteraction(bf1, bf1, bf1, bf1) == 
      computePGTOrbTwoBodyRepulsion!(emptyCache, gtoProd11, gtoProd11)


consH = [0.1543289673, 0.5353281423, 0.4446345422]
bfH = genGaussTypeOrb((0., 0., 0.), [3.425250914, 0.6239137298, 0.1688554040], 
                      consH, innerRenormalize=true)
bfLi1 = genGaussTypeOrb((1.4, 0., 0.), [16.11957475, 2.93620066, 0.7946504870], 
                        [0.1543289673, 0.5353281423, 0.4446345422], innerRenormalize=true)
bfLi2 = genGaussTypeOrb((1.4, 0., 0.), [0.6362897469, 0.1478600533, 0.0480886784], 
                        [-0.09996722919, 0.3995128261, 0.7001154689], innerRenormalize=true)
bfLi3 = genGaussTypeOrb((1.4, 0., 0.), [0.6362897469, 0.1478600533, 0.0480886784], 
                        [0.1559162750, 0.6076837186, 0.3919573931], (1, 0, 0), 
                        innerRenormalize=true)
bfLi4 = genGaussTypeOrb((1.4, 0., 0.), [0.6362897469, 0.1478600533, 0.0480886784], 
                        [0.1559162750, 0.6076837186, 0.3919573931], (0, 1, 0), 
                        innerRenormalize=true)
bfLi5 = genGaussTypeOrb((1.4, 0., 0.), [0.6362897469, 0.1478600533, 0.0480886784], 
                        [0.1559162750, 0.6076837186, 0.3919573931], (0, 0, 1), 
                        innerRenormalize=true)

bsLiH = [bfH, bfLi1, bfLi2, bfLi3, bfLi4, bfLi5]
nucs2 = [:H, :Li]
coords2 = [(0., 0., 0.), (1.4, 0., 0.)]

V2 = neAttractions(nucs2, coords2, bsLiH)
@test all(vec(V2) .≈ [-3.1880952107948826, -1.9932525008344286, -1.4155676040098832, 
                       1.2630060649812684, 0.0, 0.0, -1.9932525008344286, -8.6953599854861, 
                      -1.1279531550756718, 0.05959228438098337, 0.0, 0.0, 
                      -1.4155676040098832, -1.1279531550756718, -1.5854895488805438, 
                       0.1297201051017447, 0.0, 0.0, 1.2630060649812684, 
                       0.05959228438098337, 0.1297201051017447, -1.637219301724801, 0.0, 
                       0.0, 0.0, 0.0, 0.0, 0.0, -1.546580055447794, 0.0, 0.0, 0.0, 0.0, 
                       0.0, 0.0, -1.546580055447794])
ERI2 = eeInteractions(bsLiH)
for n in 1:Quiqbox.symmetric4DArrEleNum(bsLiH|>length)
    i, j, k, l = Quiqbox.convertIndex1DtoTri4D(n)
    @test ERI2[i, j, k, l] == ERI2[i, j, l, k] == ERI2[j, i, l, k] == ERI2[j, i, k, l] == 
          ERI2[k, l, i, j] == ERI2[k, l, j, i] == ERI2[l, k, j, i] == ERI2[l, k, i, j]
end


gtfCore1 = Quiqbox.TypedReturn(x->exp(-2.0norm(x)^2), Float64)
gtf1 = Quiqbox.EncodedField(gtfCore1, Count(1))
gto1 = Quiqbox.PolyRadialFunc(gtf1, (1, 1, 0))
bf2t = Quiqbox.PrimitiveOrb((0.1, 0.2, 0.3), gto1)

V3 = neAttractions(nucs2, coords2, [bf2t, bfH])
@test all(isapprox.(neAttractions(nucs2, coords2, [bf2, bfH]), V3, atol=5e-8))
@test neAttraction(nucs2, coords2, bf2t, bfH) ≈ neAttraction(nucs2, coords2, bfH, bf2t)

bf1D1 = Quiqbox.PrimitiveOrb((0.3,), gtf1)
bf1D2 = genGaussTypeOrb((0.2,), [2.0, 1.5], [1.1, 2.2], (2,))
ERI3 = eeInteractions([bf1D1, bf1D2])

@test ERI3[1, 2, 1, 2] ≈ 
      eeInteraction(bf1D1, bf1D2, bf1D1, bf1D2) ≈ 
      eeInteraction(bf1D1, bf1D2, bf1D2, bf1D1) ≈ 
      eeInteraction(bf1D2, bf1D1, bf1D1, bf1D2) ≈ 
      eeInteraction(bf1D2, bf1D1, bf1D2, bf1D1)

@test ERI3[1, 1, 1, 2] ≈ 
      eeInteraction(bf1D1, bf1D1, bf1D1, bf1D2) ≈ 
      eeInteraction(bf1D1, bf1D1, bf1D2, bf1D1) ≈ 
      eeInteraction(bf1D1, bf1D2, bf1D1, bf1D1) ≈ 
      eeInteraction(bf1D2, bf1D1, bf1D1, bf1D1)

@test ERI3[2, 2, 2, 1] ≈
      eeInteraction(bf1D2, bf1D2, bf1D2, bf1D1) ≈ 
      eeInteraction(bf1D2, bf1D2, bf1D1, bf1D2) ≈ 
      eeInteraction(bf1D2, bf1D1, bf1D2, bf1D2) ≈ 
      eeInteraction(bf1D1, bf1D2, bf1D2, bf1D2)

end