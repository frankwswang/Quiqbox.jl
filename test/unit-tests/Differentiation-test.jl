using Test
using Quiqbox
using Quiqbox: inSymOfCore, outSymOfCore

@testset "Differentiation.jl" begin

nuc = ["H", "H"]
nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]

grid = GridBox(1, 3.0)
gf1 = GaussFunc(0.7,1)
bs1 = genBasisFunc.(grid.box, Ref([gf1]))
pars1 = markParams!(bs1)[[1, 9, 25, 33]]
S1 = overlaps(bs1)
HFres1 = runHF(bs1, nuc, nucCoords, printInfo=false)
grad1 = gradHFenergy(bs1, pars1, HFres1.C, S1, nuc, nucCoords)

grad1_t = [1.2560795063145, 1.2560795063145, 4.050658426012191, 0]
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
cens = makeCenter.(nucCoords)
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
grad3_t = [-0.16065551469684997,  -0.24122613937626658,  -0.14800558438801004, 
            0.004775265521558752, -0.08411200995800303,  -0.3321827410644079, 
           -0.41545981416982775,  -0.057396696361203633, -0.3058864341798564, 
           -0.4189848674642474,    0.6569560675015951,    0.10172262112922723, 
            1.2108191613204624,    0.13565956677394558,   1.6059870900856348, 
            0.05883714461140787,   0.7017257214438946,   -1.288696976191361, 
            2.762937480409832,   -16.536206409405054]
@test isapprox.(grad3, grad3_t, atol=t2) |> all

end