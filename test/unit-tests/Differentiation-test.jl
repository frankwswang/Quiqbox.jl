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

grad1_t = [1.2560794980711092, 1.2560794980711092, 4.0506584177304115, 0]
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
@test isapprox(grad2[1], -0.1457888774124827, atol=t2)
@test all(grad2[3:6] .== 0)
grad2_tp = [-0.027665907127074563, 0.03295656668564936, 
             0.0946414774465647, -0.059960502688767015]
@test isapprox.(grad2[7:end], grad2_tp, atol=t2) |> all

bs3 = bs1[1:2] .* bs2 # basis set of BasisFuncMix
pars3 = markParams!(bs3, true)
S3 = overlaps(bs3)
HFres3 = runHF(bs3, nuc, nucCoords, printInfo=false)
grad3 = gradHFenergy(bs3, pars3, HFres3.C, S3, nuc, nucCoords)
grad3_t = [-0.16065844915491534, -0.24123188307759297, -0.14800104896035532, 
            0.004775820814435731, -0.08411348493980272, -0.33219109406364683, 
           -0.41545192374980927, -0.057395504756225624, -0.3058901106466324, 
           -0.41899456585416445, 0.6569508664335536, 0.1017217241636512, 
            1.2108564320016715, 0.1356630325899765, 1.6059466650621448, 
            0.05883601935066231, 0.7017058143639032, -1.2886973153008285, 
            2.762927345595733, -16.53589508656623]
@test isapprox.(grad3, grad3_t, atol=t2) |> all

end