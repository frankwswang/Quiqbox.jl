using Test
using Quiqbox
using Quiqbox: inSymOfCore, outSymOfCore

@testset "Differentiation.jl" begin

nuc = ["H", "H"]
nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]

grid = GridBox(1, 3.0)
gf1 = GaussFunc(0.7, 1.0)
bs1 = genBasisFunc.(grid.point, Ref([gf1]))
pars1 = markParams!(bs1)[[1, 9, 25, 33]]
S1 = overlaps(bs1)
HFres1 = runHF(bs1, nuc, nucCoords, printInfo=false)
grad1 = gradOfHFenergy(pars1, bs1, S1, HFres1.C, nuc, nucCoords)
grad1_t = [1.2560795063144674, 1.2560795063144674, 4.050658426012163, 0]
t1 = 1e-14
t2 = 1e-10
@test isapprox(grad1[1], grad1[2], atol=t1)
@test isapprox.(grad1, grad1_t, atol=t2) |> all

HFres1_2 = runHF(bs1, nuc, nucCoords, HFconfig((HF=:UHF,)), printInfo=false)
grad1_2 = gradOfHFenergy(pars1, bs1, overlaps(bs1), HFres1.C, nuc, nucCoords)
@test isapprox(grad1_2[1], grad1_2[2], atol=t1)
@test isapprox.(grad1_2, grad1_t, atol=t2) |> all

bfSource = genBasisFunc(missing, "STO-2G", "H")[]
gfs = bfSource.gauss |> collect
cens = genSpatialPoint.(nucCoords)
bs2 = genBasisFunc.(cens, Ref(gfs), normalizeGTO=true)
pars2 = markParams!(bs2, true)
S2 = overlaps(bs2)
HFres2 = runHF(bs2, nuc, nucCoords, printInfo=false)
grad2 = gradOfHFenergy(pars2, bs2, S2, HFres2.C, nuc, nucCoords)
@test isapprox(grad2[1], -grad2[2], atol=t2)
@test isapprox(grad2[1], -0.14578887741248214, atol=t2)
@test all(grad2[3:6] .== 0)
grad2_tp = [-0.02766590712707717,  0.03295656668565583, 
             0.09464147744656481, -0.059960502688767015]
@test isapprox.(grad2[7:end], grad2_tp, atol=t2) |> all

bs3 = bs1[[1,5]] .* bs2 # basis set of BasisFuncMix
pars3 = markParams!(bs3, true)
S3 = overlaps(bs3)
HFres3 = runHF(bs3, nuc, nucCoords, printInfo=false)
grad3 = gradOfHFenergy(pars3, HFres3)
grad3_t = [-0.16065229189030664,  -0.24121983138882722, -0.1480105654113556, 
            0.004774655654032574, -0.08411039003567462, -0.3321735674713898, 
           -0.4154684797052023,   -0.0573980050282312,  -0.3058823962913687, 
           -0.41897421621649544,   0.6569617792509108,   0.10172360617210854, 
            1.2107782292475324,    0.13565576045251876,  1.6060314866310013, 
            0.05883838041270473,   0.7017475834540088,  -1.2886966023861315, 
            2.762948610664803,   -16.53654831551982]
@test isapprox.(grad3, grad3_t, atol=5000t2) |> all

end