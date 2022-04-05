using Test
using Quiqbox
using Quiqbox: inSymOfCore, outSymOfCore

@testset "Differentiation.jl" begin

pb1 = ParamBox(1, :a)
@test inSymOfCore(pb1) == :a == outSymOfCore(pb1)
@test dataOf(pb1)[] == pb1[] == inValOf(pb1)
@test pb1[] == pb1() == 1 == outValOf(pb1)
@test pb1.map == Quiqbox.itself == mapOf(pb1)
@test pb1.canDiff[] == false == isDiffParam(pb1)
toggleDiff!(pb1)
@test pb1.canDiff[] == true
disableDiff!(pb1)
@test pb1.canDiff[] == false
enableDiff!(pb1)
@test pb1.canDiff[] == true
pb2 = inVarCopy(pb1)
@test dataOf(pb1) === dataOf(pb2) === pb1.data
pb2_2 = outValCopy(pb1)
@test dataOf(pb1) == dataOf(pb2_2)
@test dataOf(pb1) !== dataOf(pb2_2)

pb3 = ParamBox(1, :b, abs)
@test inSymOfCore(pb3) == :x_b
@test outSymOfCore(pb3) == :b
@test mapOf(pb3) == abs

pb4 = ParamBox(1.2, :c, x->x^2, :x)
@test inSymOfCore(pb4) == :x
@test outSymOfCore(pb4) == :c
@test startswith(nameof(pb4.map) |> string, "f_c")


pb5 = outValCopy(pb4)
@test pb5() == pb5[] == pb4()
@test pb5.map == Quiqbox.itself


nuc = ["H", "H"]
nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]

grid = GridBox(1, 3.0)
gf1 = GaussFunc(0.7,1)
bs1 = genBasisFunc.(grid.box, Ref([gf1]))
pars1 = uniqueParams!(bs1, filterMapping=true)[[1, 3, 4]]
S1 = overlaps(bs1)
HFres1 = runHF(bs1, nuc, nucCoords, printInfo=false)
grad1 = gradHFenergy(bs1, pars1, HFres1.C, S1, nuc, nucCoords)

grad1_t = [1.2560794975855811, 1.2560794975855811, 4.050658417242262]
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
pars2 = uniqueParams!(bs2, filterMapping=true)
S2 = overlaps(bs2)
HFres2 = runHF(bs2, nuc, nucCoords, printInfo=false)
grad2 = gradHFenergy(bs2, pars2, HFres2.C, S2, nuc, nucCoords)

@test isapprox(grad2[1], -grad2[2], atol=t2)
@test isapprox(grad2[1], -0.1457888774124827, atol=t2)
@test all(grad2[3:6] .== 0)
grad2_tp = [-0.027665907127075284, 0.032956566685641786, 
             0.09464147744656182, -0.05996050268876785]
@test isapprox.(grad2[7:end], grad2_tp, atol=t2) |> all

bs3 = bs1[1:2] .* bs2
pars3 = uniqueParams!(bs3, filterMapping=true)
S3 = overlaps(bs3)
HFres3 = runHF(bs3, nuc, nucCoords, printInfo=false)
grad3 = gradHFenergy(bs3, pars3, HFres3.C, S3, nuc, nucCoords)
grad3_t = [-0.16065844917972594,   -0.2412318831261559,  -0.14800104892200816, 
            0.0047758208191306765, -0.08411348495227357, -0.3321910941342715, 
           -0.41545192368309514,   -0.05739550474615052, -0.305890110677716, 
           -0.418994565936164,      0.6569508663895771,   0.10172172415606714, 
            1.2108564323167958,     0.13566303261927992,  1.6059466647203495, 
            0.05883601934114816,    0.7017058141955855,  -1.28869731530369, 
            2.7629273455100396,   -16.535895083933973]
@test isapprox.(grad3, grad3_t, atol=t2) |> all

end