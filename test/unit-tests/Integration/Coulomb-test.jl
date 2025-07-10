using Test
using Quiqbox
using Quiqbox: PrimGaussTypeOrbInfo, GaussProductInfo, computePGTOrbPointCoulombField

@testset "Coulomb-Interaction-Based Features" begin

gtoData1 = PrimGaussTypeOrbInfo((0.1, 0.2, 0.3), 2.0, (1, 7, 2))
gtoData2 = PrimGaussTypeOrbInfo((0.1, 0.2, 0.3), 2.0, (1, 1, 0))
gtoData3 = PrimGaussTypeOrbInfo((0.1, 0.2, 0.3), 2.0, (0, 0, 0))
gtoData4 = PrimGaussTypeOrbInfo((0.3, 0.1, 0.5), 2.0, (0, 0, 0))
gtoData5 = PrimGaussTypeOrbInfo((0.3, 0.1, 0.5), 2.0, (2, 0, 0))
gtoData6 = PrimGaussTypeOrbInfo((0.1, 0.2, 0.3), 2.0, (2, 0, 0))

gProdData1 = GaussProductInfo((gtoData1, gtoData2))
gProdData2 = GaussProductInfo((gtoData1, gtoData1))
gProdData3 = GaussProductInfo((gtoData3, gtoData3))
gProdData4 = GaussProductInfo((gtoData3, gtoData4))
gProdData5 = GaussProductInfo((gtoData3, gtoData5))
gProdData6 = GaussProductInfo((gtoData6, gtoData3))
gProdData7 = GaussProductInfo((gtoData3, gtoData6))

nucInfo1 = 1 => (0., 0., 0.)

@test computePGTOrbPointCoulombField(nucInfo1, gProdData1) ≈ -0.00021406291700540685
@test computePGTOrbPointCoulombField(nucInfo1, gProdData2) ≈ -0.00016035624620095473
@test computePGTOrbPointCoulombField(nucInfo1, gProdData3) ≈ -1.3209276479060006
@test computePGTOrbPointCoulombField(nucInfo1, gProdData4) ≈ -1.102953813735257
@test computePGTOrbPointCoulombField(nucInfo1, gProdData5) ≈ -0.1305787950084035
@test computePGTOrbPointCoulombField(nucInfo1, gProdData6) ==
      computePGTOrbPointCoulombField(nucInfo1, gProdData7) ≈ -0.11995218441914622

end