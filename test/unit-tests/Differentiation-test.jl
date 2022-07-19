using Test
using Quiqbox
using Quiqbox: inSymOfCore, outSymOfCore

@testset "Differentiation.jl" begin

nuc = ["H", "H"]
nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]

grid = GridBox(1, 3.0)
gf1 = GaussFunc(0.7, 1.0)
bs1 = genBasisFunc.(grid.box, Ref([gf1]))
pars1 = markParams!(bs1)[[1, 9, 25, 33]]
S1 = overlaps(bs1)
HFres1 = runHF(bs1, nuc, nucCoords, printInfo=false)
grad1 = gradHFenergy(bs1, pars1, HFres1.C, S1, nuc, nucCoords)
grad1_t = [1.2560795063144674, 1.2560795063144674, 4.050658426012163, 0]          
t1 = 1e-14
t2 = 1e-10
@test isapprox(grad1[1], grad1[2], atol=t1)
@test isapprox.(grad1, grad1_t, atol=t2) |> all

HFres1_2 = runHF(bs1, nuc, nucCoords, HFconfig((HF=:UHF,)), printInfo=false)
grad1_2 = gradHFenergy(bs1, pars1, HFres1.C, overlaps(bs1), nuc, nucCoords)
@test isapprox(grad1_2[1], grad1_2[2], atol=t1)
@test isapprox.(grad1_2, grad1_t, atol=t2) |> all

bfSource = genBasisFunc(missing, "STO-2G", "H")[]
gfs = bfSource.gauss |> collect
cens = genSpatialPoint.(nucCoords)
bs2 = genBasisFunc.(cens, Ref(gfs), normalizeGTO=true)
pars2 = markParams!(bs2, true)
S2 = overlaps(bs2)
HFres2 = runHF(bs2, nuc, nucCoords, printInfo=false)
grad2 = gradHFenergy(bs2, pars2, HFres2.C, S2, nuc, nucCoords)
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
grad3 = gradHFenergy(bs3, pars3, HFres3.C, S3, nuc, nucCoords)
grad3_t = [-0.160652283480453,    -0.24121981492839778, -0.14801057840914905, 
            0.004774654062581023, -0.08411038580851136, -0.3321735435333388, 
           -0.41546850231745325,  -0.05739800844313432, -0.30588238575435295, 
           -0.41897418842244705,   0.6569617941550787,   0.10172360874247582, 
            1.2107781224373306,    0.1356557505200704,   1.6060316024819183, 
            0.05883838363746288,   0.7017476405008756,  -1.2886966014088115, 
            2.7629486397083984,  -16.536549207706194]
@test isapprox.(grad3, grad3_t, atol=t2) |> all

end       