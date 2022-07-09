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
grad3_t = [-0.16065576967254214,  -0.24122663844403264, -0.1480051903060922, 
            0.004775313771418176, -0.08411213811964297, -0.3321834668515459, 
           -0.4154591285771253,   -0.05739659282337302, -0.3058867536343978, 
           -0.4189857101548055,    0.6569556155931322,   0.1017225431937995, 
            1.2108223997457819,    0.1356598679175225,   1.6059835775728044, 
            0.05883704683832264,   0.7017239917546765,  -1.288697005703953, 
            2.762936599808623,   -16.536179358779652]
@show grad3
          [-0.16062864424199147, -0.2411735493547454, -0.14804711194047943, 
            0.004770180375113158, -0.0840985035587043, -0.33210626165404944, 
            -0.41553205683211697, -0.05740760658222216, -0.3058527627387774, 
            -0.4188960659822836, 0.6570036772359015, 0.101730832059271, 
            1.210477916219335, 0.13562783299744025, 1.6063572261633066, 
            0.05884744723321427, 0.7019079596644313, -1.2886938155694363, 
            2.763030266020559, -16.53905682081113]
@test isapprox.(grad3, grad3_t, atol=t2) |> all

end