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
grad3_t = [-0.16065229026420086,  -0.24121982820608456, -0.14801056792457273, 
            0.004774655346313956, -0.08411038921832216, -0.33217356284279526, 
           -0.4154684840774442,   -0.05739800568852736, -0.3058823942539609, 
           -0.41897421084231595,   0.6569617821327393,   0.10172360666910765, 
            1.2107782085950103,    0.13565575853200829,  1.6060315090316117, 
            0.05883838103623499,   0.7017475944844322,  -1.28869660219716, 
            2.762948616280592,   -16.536548488030494]
@show max( abs.(grad3 - grad3_t)... )
@show grad3
@test isapprox.(grad3, grad3_t, atol=1e4t2) |> all

end