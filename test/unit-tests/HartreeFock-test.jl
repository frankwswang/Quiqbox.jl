using Test
using Quiqbox
using Quiqbox: splitSpins, HFfinalVars
using Suppressor: @suppress_out, @capture_out

include("../../test/test-functions/Shared.jl")

@testset "HartreeFock.jl" begin

# function splitSpins
@test splitSpins(Val(:RHF), 5) == (2,)
@test splitSpins(Val(:RHF), (2,)) == (2,)
try splitSpins(Val(:RHF), (1,3)) == (2,) catch err; @test (err isa ErrorException) end
@test splitSpins(Val(:UHF), 5) == (2,3)
@test splitSpins(Val(:UHF), (2,)) == (2,2)
@test splitSpins(Val(:UHF), (3,2)) == (3,2)


errorThreshold1 = 5e-8
errorThreshold2 = 1e-12
errorThreshold3 = 1e-5

nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0], [0.0, 0.0, 0.0]]
nuc = ["H", "H", "O"]
bs = genBasisFunc.(nucCoords, "STO-3G", nuc) |> flatten
gtb = GTBasis(bs)
S = overlaps(bs)
X = S^(-0.5)
Hcore = coreH(bs, nuc, nucCoords)
HeeI = eeInteractions(bs)
Ne = getCharge(nuc)
scfMethods = (:ADIIS, :DIIS, :EDIIS, :DD)
thresholds = (1e-4, 1e-8, 1e-10, 1e-15)
solvers = Dict(1=>[:solver=>:LCM], 2=>[:solver=>:LCM], 
               3=>[:solver=>:LCM], 4=>[:solver=>:LCM])

local res1, res1_2, res1_3, res1_4, res1_5, res2, res2_2, res2_3
SCFc1 = SCFconfig(scfMethods, thresholds)
SCFc2 = SCFconfig(scfMethods, thresholds, solvers)
HFc1 = HFconfig((C0=Val(:Hcore), SCF=SCFc1))
HFc2 = HFconfig((SCF=SCFc1,))
HFc3 = HFconfig(HF=:UHF, C0=:GWH, SCF=SCFc2)
HFc3_2 = HFconfig(HF=:UHF, C0=Quiqbox.getCfromGWH(Val(:UHF), S, Hcore, X), SCF=SCFc2)
HFc4 = HFconfig((HF=Val(:UHF), SCF=SCFc2))
HFc5 = HFconfig((C0=:Hcore, SCF=SCFconfig(threshold=1e-15)))
HFc6 = HFconfig((C0=:Hcore, SCF=SCFconfig()))
HFc6_2 = HFconfig((C0=Quiqbox.getCfromHcore(Val(:RHF), X, Hcore), SCF=SCFconfig()))
HFc7 = HFconfig((C0=(zeros(1,1),), SCF=SCFconfig()))
try
    runHF(bs, nuc, nucCoords, HFc7)
catch err
    @test (err isa DimensionMismatch)
end

try
    runHF(bs, nuc, nucCoords, splitSpins(Val(:UHF), Ne))
catch err
    @test (err isa ErrorException)
end

@suppress_out begin
    res1   = runHF(bs, nuc, nucCoords, HFc1)
    res1_2interm = runHFcore(bs, nuc, nucCoords, HFc2)
    res1_2 = HFfinalVars(gtb, nuc, nucCoords, X, res1_2interm...)
    res1_3interm = runHFcore(bs, nuc, nucCoords, Ne, HFc5)
    res1_3 = HFfinalVars(gtb, nuc, nucCoords, X, res1_3interm...)
    res1_4 = runHF(bs, nuc, nucCoords, splitSpins(Val(:RHF), Ne), HFc6)
    res1_5 = runHF(bs, nuc, nucCoords, Ne, HFc6_2)
    res2   = runHF(bs, nuc, nucCoords, HFc3)
    res2_2 = runHF(bs, nuc, nucCoords, HFc4)
    res2_3 = runHF(bs, nuc, nucCoords, HFc3_2, splitSpins(Val(:UHF), Ne))
end

@test isapprox(res2.Ehf, res2_2.Ehf, atol=100errorThreshold1)
@test isapprox(res1.Ehf, res1_2.Ehf, atol=errorThreshold1)
@test isapprox(res1.Ehf, res1_3.Ehf, atol=errorThreshold1)
@test isapprox(res1_3.Ehf, res1_4.Ehf, atol=2e-12)
@test hasEqual(res1_4, res1_5)
@test hasEqual(res2, res2_3)

@test begin
    tVars1 = deepcopy(res1.temp[1])
    Quiqbox.popHFtempVars!((tVars1,))
    push!(tVars1.Cs, res1.temp[1].Cs[end])
    push!(tVars1.Fs, res1.temp[1].Fs[end])
    push!(tVars1.Ds, res1.temp[1].Ds[end])
    push!(tVars1.Es, res1.temp[1].Es[end])
    push!(tVars1.shared.Dtots, res1.temp[1].shared.Dtots[end])
    push!(tVars1.shared.Etots, res1.temp[1].shared.Etots[end])
    hasEqual(tVars1, res1.temp[1])
end

@test res1.Ehf == Quiqbox.getEhf(Hcore, HeeI, res1.C, (Ne÷2,))
res1Ehf1 = Quiqbox.getEhf((changeHbasis(Hcore, res1.C[begin]),), 
                          (changeHbasis(HeeI, res1.C[begin]),), (Ne÷2,))
@test isapprox(res1.Ehf, res1Ehf1, atol=errorThreshold2)
gtb2 = [sum(c.*gtb.basis) for c in eachcol(res1.C[begin])] |> GTBasis
res1Ehf2 = Quiqbox.getEhf(gtb2, nuc, nucCoords, Ne)
@test isapprox(res1Ehf1, res1Ehf2, atol=errorThreshold2)
@test isapprox(res1.Ehf, -93.7878386328627, atol=errorThreshold1)

# Note: the orbital coefficients of 4th and 5th columns are so close that based on the 
# numerical error of each machine the position of them might switch.

@test isapprox(res1.C[1][1:5, [1,2,3,6,7]], 
[ 0.010895919  0.088981101  0.121607884  0.0  0.0  1.914545199  3.615733319; 
  0.010895919  0.088981101 -0.121607884  0.0  0.0  1.914545199 -3.615733319; 
 -0.994229867 -0.26343918   0.0          0.0  0.0  0.049696489  0.0; 
 -0.04159303   0.872249144  0.0          0.0  0.0 -3.396657443  0.0; 
  0.0          0.0          1.094020465  0.0  0.0  0.0          2.778672546; 
  0.0          0.0          0.0          1.0  0.0  0.0          0.0; 
  0.0          0.0          0.0          0.0  1.0  0.0          0.0][1:5, [1,2,3,6,7]], 
atol=errorThreshold1)

@test  isapprox(vcat(res1.C[1][6:7,:][:], res1.C[1][1:5, 4:5][:]) |> sort, 
                vcat(fill(0,22), fill(1,2)), atol=errorThreshold1)
@test  isapprox(res1.C[1][6:7, 4:5][:] |> sort, [0,0,1,1], atol=errorThreshold1)

@test isapprox(res1.F[1], 
    [-2.255358688 -1.960982029  -4.484369214 -2.511689786  0.483603806  0.0  0.0; 
     -1.960982029 -2.255358688  -4.484369214 -2.511689786 -0.483603806  0.0  0.0; 
     -4.484369214 -4.484369214 -20.920383179 -5.363456843  0.0          0.0  0.0; 
     -2.511689786 -2.511689786  -5.363456843 -2.896377602  0.0          0.0  0.0; 
      0.483603806 -0.483603806   0.0          0.0         -1.280927053  0.0  0.0; 
      0.0          0.0           0.0          0.0          0.0 -0.661307596  0.0; 
      0.0          0.0           0.0          0.0          0.0  0.0 -0.661307596], 
atol=errorThreshold1)
@test isapprox(res1.Eo[1], 
    [-20.930384473, -1.616675719, -1.284466204, -0.661307596, 
      -0.661307596,  1.060815281,  1.847804072], 
atol=errorThreshold1)
@test res1.occu[1] == ("↿⇂", "↿⇂", "↿⇂", "↿⇂", "↿⇂", "0", "0")

D1 = res1.D[1]
@test isapprox(D1*S*D1, D1, atol=errorThreshold1)


@test begin
    tVars2 = deepcopy(res2.temp)
    Quiqbox.popHFtempVars!(tVars2)
    push!(tVars2[1].Cs, res2.temp[1].Cs[end])
    push!(tVars2[1].Fs, res2.temp[1].Fs[end])
    push!(tVars2[1].Ds, res2.temp[1].Ds[end])
    push!(tVars2[1].Es, res2.temp[1].Es[end])
    push!(tVars2[2].Cs, res2.temp[2].Cs[end])
    push!(tVars2[2].Fs, res2.temp[2].Fs[end])
    push!(tVars2[2].Ds, res2.temp[2].Ds[end])
    push!(tVars2[2].Es, res2.temp[2].Es[end])
    push!(tVars2[2].shared.Dtots, res2.temp[2].shared.Dtots[end])
    push!(tVars2[2].shared.Etots, res2.temp[2].shared.Etots[end])
    hasEqual(tVars2, res2.temp)
end

@test res2.Ehf == Quiqbox.getEhf(Hcore, HeeI, res2.C, (Ne÷2, Ne-Ne÷2))
HcoreUHF2 = changeHbasis.(Ref(Hcore), res2.C)
a, b, c = changeHbasis(HeeI, res2.C...)
res2Ehf1 = Quiqbox.getEhf(HcoreUHF2, (a, b), c, Quiqbox.splitSpins(Val(:UHF), Ne))
@test isapprox(res2.Ehf, res2Ehf1, atol=errorThreshold2)
@test isapprox(res2.Ehf, -93.78783863286264, atol=errorThreshold1)

res2_C1t = [ 0.01089592   0.088980957  0.121608177  0.0  0.0  1.914545206  3.615733309; 
             0.01089592   0.088980957 -0.121608177  0.0  0.0  1.914545206 -3.615733309; 
            -0.994229867 -0.263439184  0.0          0.0  0.0  0.049696469  0.0; 
            -0.041593031  0.8722494    0.0          0.0  0.0 -3.396657377  0.0; 
             0.0          0.0          1.09402069   0.0  0.0  0.0          2.778672457; 
             0.0          0.0          0.0          1.0  0.0  0.0          0.0; 
             0.0          0.0          0.0          0.0  1.0  0.0          0.0]

res2_C2t = [ 0.010895919  0.088981246  0.121607591  0.0  0.0  1.914545192  3.615733329; 
             0.010895919  0.088981246 -0.121607591  0.0  0.0  1.914545192 -3.615733329; 
            -0.994229867 -0.263439176  0.0          0.0  0.0  0.049696508  0.0; 
            -0.041593029  0.872248887  0.0          0.0  0.0 -3.396657509  0.0; 
             0.0          0.0          1.09402024   0.0  0.0  0.0          2.778672635; 
             0.0          0.0          0.0          1.0  0.0  0.0          0.0; 
             0.0          0.0          0.0          0.0  1.0  0.0          0.0]

ids = [1,2,3,6,7]
@test isapprox(res2.C[1][1:5, ids], res2_C1t[1:5, ids], atol=errorThreshold3)
@test isapprox(res2.C[2][1:5, ids], res2_C2t[1:5, ids], atol=errorThreshold3)

@test  isapprox(vcat(res2.C[1][6:7,:][:], res2.C[1][1:5, 4:5][:]) |> sort, 
                vcat(fill(0,22), fill(1,2)), atol=errorThreshold1)
@test  isapprox(res2.C[1][6:7, 4:5][:] |> sort, [0,0,1,1], atol=errorThreshold1)

@test  isapprox(vcat(res2.C[2][6:7,:][:], res2.C[2][1:5, 4:5][:]) |> sort, 
                vcat(fill(0,22), fill(1,2)), atol=errorThreshold1)
@test  isapprox(res2.C[2][6:7, 4:5][:] |> sort, [0,0,1,1], atol=errorThreshold1)

res2_F1t = [-2.255358683 -1.960982031 -4.484369221  -2.511689801  0.483603803  0.0  0.0; 
            -1.960982031 -2.255358683 -4.484369221  -2.511689801 -0.483603803  0.0  0.0; 
            -4.484369221 -4.484369221 -20.920383216 -5.363456851  0.0          0.0  0.0; 
            -2.511689801 -2.511689801 -5.363456851  -2.896377637  0.0          0.0  0.0; 
             0.483603803 -0.483603803  0.0           0.0         -1.280927066  0.0  0.0; 
             0.0          0.0          0.0           0.0          0.0 -0.661307619  0.0; 
             0.0          0.0          0.0           0.0          0.0  0.0 -0.661307619]

res2_F2t = [-2.255358705 -1.960982032  -4.484369213 -2.511689786  0.483603812  0.0  0.0; 
            -1.960982032 -2.255358705  -4.484369213 -2.511689786 -0.483603812  0.0  0.0; 
            -4.484369213 -4.484369213 -20.920383196 -5.363456842  0.0          0.0  0.0; 
            -2.511689786 -2.511689786  -5.363456842 -2.896377589  0.0          0.0  0.0; 
             0.483603812 -0.483603812   0.0          0.0         -1.280927041  0.0  0.0; 
             0.0          0.0           0.0          0.0          0.0 -0.661307591  0.0; 
             0.0          0.0           0.0          0.0          0.0  0.0 -0.661307591]

@test isapprox(res2.F[1], res2_F1t, atol=errorThreshold3)
@test isapprox(res2.F[2], res2_F2t, atol=errorThreshold3)

res2_Eo1t = [-20.93038451, -1.616675748, -1.28446622,  -0.66130762, 
              -0.66130762,  1.060815274,  1.847804062]

res2_Eo2t = [-20.93038449, -1.616675711, -1.284466186, -0.661307591, 
              -0.661307591, 1.060815276,  1.847804083]

compr2Arrays3((res2_Eo1=res2.Eo[1], res2_Eo1t=res2_Eo1t), 10errorThreshold1)
compr2Arrays3((res2_Eo2=res2.Eo[2], res2_Eo2t=res2_Eo2t), 10errorThreshold1)

@test all( res2.occu .== ( ("↿", "↿", "↿", "↿", "↿", "0", "0"), 
                           ("⇂", "⇂", "⇂", "⇂", "⇂", "0", "0") ) )
D2s = res2.D
for D in D2s
    @test isapprox(D*S*D, D, atol=errorThreshold1)
end

# Potential energy curve tests.
rhfs = [ 7.275712508,  0.721327344, -0.450914129, -0.860294199, -1.029212153, -1.098483134, 
        -1.12116316,  -1.12068526,  -1.108423332, -1.090377472, -1.06979209,  -1.048358444, 
        -1.026929524, -1.005949751, -0.985675141, -0.966266831, -0.947825531, -0.93040561, 
        -0.914025059, -0.898675515, -0.884331776, -0.870959502, -0.858520505, -0.846975806, 
        -0.836287041, -0.826416815, -0.81732844,  -0.808985378, -0.801350617, -0.794386143, 
        -0.788052639, -0.782309455, -0.777114838, -0.772426373, -0.768201548, -0.764398372, 
        -0.760975967, -0.757895111, -0.755118673, -0.752611963, -0.750342966, -0.748282481, 
        -0.746404172, -0.74468454,  -0.743102829, -0.741640881, -0.740282953, -0.73901551, 
        -0.737826999, -0.736707621, -0.735649109, -0.734644515, -0.733688011, -0.732774716, 
        -0.73190053,  -0.731062005, -0.730256222, -0.729480696, -0.728733293, -0.728012163, 
        -0.727315685, -0.726642428, -0.725991112, -0.725360583, -0.72474979,  -0.724157774, 
        -0.723583648, -0.723026588, -0.72248583,  -0.721960657, -0.721450398, -0.720954422, 
        -0.720472134, -0.720002974, -0.719546412, -0.719101945, -0.718669099, -0.718247424, 
        -0.717836493, -0.7174359,   -0.717045259, -0.716664205, -0.716292389, -0.715929478, 
        -0.715575157, -0.715229124, -0.714891092, -0.714560787, -0.714237946, -0.713922319, 
        -0.713613668, -0.713311763, -0.713016385, -0.712727326, -0.712444385, -0.712167369, 
        -0.711896094, -0.711630384, -0.711370069, -0.711114987]

uhfs = [ 7.275712508,  0.721327344, -0.450914129, -0.860294199, -1.029212153, -1.098483134, 
        -1.12116316,  -1.12068526,  -1.108423332, -1.09037743,  -1.069791366, -1.048704062, 
        -1.032357188, -1.020787228, -1.01259153,  -1.006780872, -1.002658864, -0.999733913, 
        -0.997657971, -0.996184131, -0.99513707,  -0.994392271, -0.993861306, -0.993481374, 
        -0.99320788,  -0.993009217, -0.992863103, -0.992753979, -0.992671126, -0.992607258, 
        -0.992557465, -0.992518423, -0.992487828, -0.992464006, -0.992445669, -0.992431764, 
        -0.992421407, -0.992413842, -0.99240843,  -0.99240464,  -0.992402043, -0.992400301, 
        -0.992399158, -0.992398424, -0.992397962, -0.992397678, -0.992397506, -0.992397405, 
        -0.992397346, -0.992397312, -0.992397294, -0.992397283, -0.992397278, -0.992397275, 
        -0.992397274, -0.992397273, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272]

nuc2 = ["H", "H"]
rng = 0.1:0.2:19.9
Et1 = Float64[]
Et2 = Float64[]
n = 0
for i in rng
    n += 1
    nucCoords2 = [[0, 0.0, 0.0], [i, 0.0, 0.0]]

    bs = genBasisFunc.(nucCoords2, "3-21G") |> flatten
    local res1, res2
    info1 = @capture_out begin
        @show i
        res1 = runHF(bs, nuc2, nucCoords2, printInfo=true)
        @show length(res1.temp[begin].Es) res1.Ehf
    end
    info2 = @capture_out begin
        @show i
        res2 = runHF(bs, nuc2, nucCoords2, HFconfig((HF=:UHF,)), printInfo=true)
        @show length(res2.temp[begin].Es) res2.Ehf
    end
    push!(Et1, res1.Ehf+res1.Enn)
    push!(Et2, res2.Ehf+res1.Enn)
    !isapprox(Et1[n], rhfs[n], atol=errorThreshold1) && println(info1)
    !isapprox(Et2[n], uhfs[n], atol=errorThreshold1) && println(info2)
end

compr2Arrays2((Et1=Et1, rhfs=rhfs), 95, errorThreshold1, 0.6)
compr2Arrays2((Et2=Et2, uhfs=uhfs), 16, errorThreshold1, 5e-5, <)

end