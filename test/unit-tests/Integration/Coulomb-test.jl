using Test
using Quiqbox
using Quiqbox: PrimGaussTypeOrbInfo, GaussProductInfo, computePGTOrbOneBodyRepulsion, 
               computePGTOrbTwoBodyRepulsion

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

nucCoord = (0., 0., 0.)

@test computePGTOrbOneBodyRepulsion(nucCoord, gtoProd12) ≈ 0.00021406291700540685
@test computePGTOrbOneBodyRepulsion(nucCoord, gtoProd11) ≈ 0.00016035624620095473
@test computePGTOrbOneBodyRepulsion(nucCoord, gtoProd33) ≈ 1.3209276479060006
@test computePGTOrbOneBodyRepulsion(nucCoord, gtoProd34) ≈ 1.102953813735257
@test computePGTOrbOneBodyRepulsion(nucCoord, gtoProd35) ≈ 0.1305787950084035
@test computePGTOrbOneBodyRepulsion(nucCoord, gtoProd63) ==
      computePGTOrbOneBodyRepulsion(nucCoord, gtoProd36) ≈ 0.11995218441914622


gtoData7 = PrimGaussTypeOrbInfo((0.9, 0.6, 0.1), 2.5, (1, 1, 0))
gtoData8 = PrimGaussTypeOrbInfo((0.6, 0.7, 0.8), 3.0, (3, 1, 2))

gtoProd14 = GaussProductInfo((gtoData1, gtoData4))
gtoProd41 = GaussProductInfo((gtoData4, gtoData1))
gtoProd78 = GaussProductInfo((gtoData7, gtoData8))
gtoProd87 = GaussProductInfo((gtoData8, gtoData7))

@test computePGTOrbTwoBodyRepulsion(gtoProd11, gtoProd22) ≈ 
      computePGTOrbTwoBodyRepulsion(gtoProd22, gtoProd11) ≈ 
      1.7675350484831864e-6

@test computePGTOrbTwoBodyRepulsion(gtoProd12, gtoProd12) ≈ 
      computePGTOrbTwoBodyRepulsion(gtoProd12, gtoProd21) ≈ 
      computePGTOrbTwoBodyRepulsion(gtoProd21, gtoProd21) ≈ 
      6.267963629018787e-8

@test computePGTOrbTwoBodyRepulsion(gtoProd33, gtoProd44) ≈ 
      computePGTOrbTwoBodyRepulsion(gtoProd44, gtoProd33) ≈ 
      0.7291219052871128

@test computePGTOrbTwoBodyRepulsion(gtoProd14, gtoProd78) ≈ 
      computePGTOrbTwoBodyRepulsion(gtoProd41, gtoProd78) ≈ 
      computePGTOrbTwoBodyRepulsion(gtoProd41, gtoProd87) ≈ 
      computePGTOrbTwoBodyRepulsion(gtoProd14, gtoProd87) ≈ 
      computePGTOrbTwoBodyRepulsion(gtoProd78, gtoProd14) ≈ 
      computePGTOrbTwoBodyRepulsion(gtoProd87, gtoProd14) ≈ 
      computePGTOrbTwoBodyRepulsion(gtoProd87, gtoProd41) ≈ 
      computePGTOrbTwoBodyRepulsion(gtoProd78, gtoProd41) ≈ -2.4175946692430508e-9

end