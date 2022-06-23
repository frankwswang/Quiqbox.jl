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

grad1_t = [1.256079506314511, 1.256079506314511, 4.050658426012203, 0]
t1 = 1e-14
t2 = 1e-10
@test isapprox(grad1[1], grad1[2], atol=t1)
@test isapprox.(grad1, grad1_t, atol=t2) |> all

HFres1_2 = runHF(bs1, nuc, nucCoords, HFconfig((HF=:UHF,)), printInfo=false)
grad1_2 = gradHFenergy(bs1, pars1, HFres1.C, overlaps(bs1), nuc, nucCoords)
@test isapprox(grad1_2[1], grad1_2[2], atol=t1)
@test isapprox.(grad1_2, grad1_t, atol=t2) |> all

bfSource = genBasisFunc(missing, ("STO-2G", "H"))[]
gfs = bfSource.gauss |> collect
cens = genSpatialPoint.(nucCoords)
bs2 = genBasisFunc.(cens, Ref(gfs), normalizeGTO=true)
pars2 = markParams!(bs2, true)
S2 = overlaps(bs2)
HFres2 = runHF(bs2, nuc, nucCoords, printInfo=false)
grad2 = gradHFenergy(bs2, pars2, HFres2.C, S2, nuc, nucCoords)

@test isapprox(grad2[1], -grad2[2], atol=t2)
@test isapprox(grad2[1], -0.14578887741248492, atol=t2)
@test all(grad2[3:6] .== 0)
grad2_tp = [-0.027665907127074563, 0.03295656668564936, 
             0.0946414774465647, -0.059960502688767015]
@test isapprox.(grad2[7:end], grad2_tp, atol=t2) |> all

bs3 = bs1[1:2] .* bs2 # basis set of BasisFuncMix
pars3 = markParams!(bs3, true)
S3 = overlaps(bs3)
HFres3 = runHF(bs3, nuc, nucCoords, printInfo=false)
grad3 = gradHFenergy(bs3, pars3, HFres3.C, S3, nuc, nucCoords)
grad3_t = [-0.1606522976568731, -0.24121984267560467, -0.148010556498878, 
            0.004774656745277748, -0.0841103929342053, -0.3321735838855099, 
           -0.41546846420017175, -0.05739800268666162, -0.3058824035165118, 
           -0.41897423527464034, 0.6569617690312483, 0.10172360440963082, 
            1.2107783024863934, 0.13565576726311543, 1.6060314071929958, 
            0.058838378201514954, 0.7017475443374457, -1.2886966030562688, 
            2.762948590749896, -16.536547703755087]
@test isapprox.(grad3, grad3_t, atol=t2) |> all

end