using Test
using Quiqbox
using Suppressor: @suppress_out

@testset "HartreeFock.jl" begin

errorThreshold1 = 1e-8
errorThreshold2 = 5e-5
errorThreshold3 = 5e-4

nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0], [0.0, 0.0, 0.0]]
nuc = ["H", "H", "O"]
bs = genBasisFunc.(nucCoords, ("STO-3G", "STO-3G", ("STO-3G", "O"))) |> flatten
S = overlaps(bs)
Hcore = coreH(bs, nuc, nucCoords)
HeeI = eeInteractions(bs)
Ne = getCharge(nuc)
scfMethods = (:ADIIS, :DIIS, :EDIIS, :DD)
thresholds = (1e-4, 1e-8, 1e-10, 1e-12)
solvers = Dict(1=>[:solver=>:LCM], 2=>[:solver=>:LCM], 
               3=>[:solver=>:LCM], 4=>[:solver=>:LCM])

local res1, res2, res1_2, res2_2
SCFc1 = SCFconfig(scfMethods, thresholds)
SCFc2 = SCFconfig(scfMethods, thresholds, solvers)
HFc1 = HFconfig((C0=:Hcore, SCF=SCFc1))
HFc2 = HFconfig((SCF=SCFc1,))
HFc3 = HFconfig((HF=:UHF, C0=:GWH, SCF=SCFc2))
HFc4 = HFconfig((HF=:UHF, SCF=SCFc2))

@suppress_out begin
    res1   = runHF(bs, nuc, nucCoords, HFc1)
    res1_2 = runHF(bs, nuc, nucCoords, HFc2)
    res2   = runHF(bs, nuc, nucCoords, HFc3)
    res2_2 = runHF(bs, nuc, nucCoords, HFc4)
end

@test isapprox(res2.Ehf, res2_2.Ehf, atol=errorThreshold2)
@test isapprox(res1.Ehf, res1_2.Ehf, atol=errorThreshold1)

@test begin
    tVars1 = deepcopy(res1.temp[1])
    Quiqbox.popHFtempVars!(tVars1)
    push!(tVars1.Cs, res1.temp[1].Cs[end])
    push!(tVars1.Fs, res1.temp[1].Fs[end])
    push!(tVars1.Ds, res1.temp[1].Ds[end])
    push!(tVars1.Es, res1.temp[1].Es[end])
    push!(tVars1.shared.Dtots, res1.temp[1].shared.Dtots[end])
    push!(tVars1.shared.Etots, res1.temp[1].shared.Etots[end])
    hasEqual(tVars1, res1.temp[1])
end

@test isapprox(res1.Ehf, Quiqbox.getEᵀ(Hcore, HeeI, res1.C, (Ne,)), atol=errorThreshold1)

@test isapprox(res1.Ehf, -93.7878386290283, atol=errorThreshold1)

# Note: the molecular energies of 4th and 5th columns are so close that based on the 
# numerical error of each machine the position of them might switch.

@test isapprox(res1.C[1][1:5, [1,2,3,6,7]], 
[ 0.010896028  0.08898146   0.121616979  0.0  0.0  1.914545182  3.615733013; 
  0.010896028  0.08898146  -0.121616979  0.0  0.0  1.914545182 -3.615733013; 
 -0.994229824 -0.263439324  0.0          0.0  0.0  0.049696601 -0.0; 
 -0.041593381  0.872248503 -0.0          0.0  0.0 -3.396657603  0.0; 
  0.0         -0.0          1.094027455  0.0  0.0  0.0          2.778669794; 
  0.0          0.0          0.0          1.0  0.0  0.0         -0.0; 
  0.0          0.0          0.0          0.0  1.0  0.0         -0.0][1:5, [1,2,3,6,7]], 
atol=errorThreshold2)

@test  isapprox(vcat(res1.C[1][6:7,:][:], res1.C[1][1:5, 4:5][:]) |> sort, 
                vcat(fill(0,22), fill(1,2)), atol=errorThreshold1)
@test  isapprox(res1.C[1][6:7, 4:5][:] |> sort, [0,0,1,1], atol=errorThreshold1)

@test isapprox(res1.F[1], 
    [-2.255344255 -1.960972809 -4.484361945  -2.511676807  0.483598435  0.0  0.0; 
     -1.960972809 -2.255344255 -4.484361945  -2.511676807 -0.483598435  0.0  0.0; 
     -4.484361945 -4.484361945 -20.920337881 -5.363448416  0.0          0.0  0.0; 
     -2.511676807 -2.511676807 -5.363448416  -2.896362719 -0.0          0.0  0.0; 
      0.483598435 -0.483598435  0.0          -0.0         -1.280915978  0.0  0.0; 
      0.0          0.0          0.0           0.0          0.0         -0.661290126  0.0; 
      0.0          0.0          0.0           0.0          0.0          0.0 -0.661290126], 
atol=errorThreshold2)

@test isapprox(res1.Emo[1], 
    [-20.930339307, -1.616661402, -1.284455652, -0.661290126, 
      -0.661290126,  1.060822819,  1.847810076], 
atol=errorThreshold1)

@test res1.occu[1] == [2, 2, 2, 2, 2, 0, 0]

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

@test isapprox(res2.Ehf, Quiqbox.getEᵀ(Hcore, HeeI, res2.C, (Ne÷2, Ne-Ne÷2)), 
                atol=errorThreshold3)

@test isapprox(res2.Ehf, -93.78783862153227, atol=errorThreshold2)

@test isapprox.((res2.C[1][1:5, [1,2,3,6,7]], res2.C[2][1:5, [1,2,3,6,7]]), 
([ 0.010895987  0.088966933  0.121651324  0.0  0.0  1.914545857  3.615731858; 
   0.010895987  0.088966933 -0.121651324  0.0  0.0  1.914545857 -3.615731858; 
  -0.994229826 -0.263439695  0.0          0.0  0.0  0.04969458  -0.0; 
  -0.041593302  0.872274277 -0.0          0.0  0.0 -3.396650986  0.0; 
   0.0         -0.0          1.094053848  0.0  0.0  0.0          2.778659402; 
   0.0         -0.0          0.0          1.0  0.0  0.0         -0.0; 
   0.0         -0.0          0.0          0.0  1.0  0.0         -0.0][1:5, [1,2,3,6,7]], 
 [ 0.010895938  0.088995556  0.121571667  0.0  0.0  1.914544527  3.615734537; 
   0.010895938  0.088995556 -0.121571667  0.0  0.0  1.914544527 -3.615734537; 
  -0.994229874 -0.263438779  0.0          0.0  0.0  0.049698486 -0.0; 
  -0.041593037  0.872223499 -0.0          0.0  0.0 -3.396664028  0.0; 
   0.0         -0.0          1.093992632  0.0  0.0  0.0          2.778683504; 
  -0.0          0.0          0.0          1.0  0.0  0.0         -0.0; 
  -0.0          0.0          0.0          0.0  1.0  0.0         -0.0][1:5, [1,2,3,6,7]]), 
atol=errorThreshold3*2) |> all

@test  isapprox(vcat(res2.C[1][6:7,:][:], res2.C[1][1:5, 4:5][:]) |> sort, 
                vcat(fill(0,22), fill(1,2)), atol=errorThreshold1)
@test  isapprox(res2.C[1][6:7, 4:5][:] |> sort, [0,0,1,1], atol=errorThreshold1)

@test  isapprox(vcat(res2.C[2][6:7,:][:], res2.C[2][1:5, 4:5][:]) |> sort, 
                vcat(fill(0,22), fill(1,2)), atol=errorThreshold1)
@test  isapprox(res2.C[2][6:7, 4:5][:] |> sort, [0,0,1,1], atol=errorThreshold1)

@test isapprox.(res2.F, 
([-2.255348475 -1.960978669 -4.484368474  -2.511687446  0.483599893  0.0  0.0; 
  -1.960978669 -2.255348475 -4.484368474  -2.511687446 -0.483599893  0.0  0.0; 
  -4.484368474 -4.484368474 -20.920369474 -5.363455961 -0.0          0.0  0.0; 
  -2.511687446 -2.511687446 -5.363455961  -2.896380604  0.0          0.0  0.0; 
   0.483599893 -0.483599893 -0.0           0.0         -1.280930357  0.0  0.0; 
   0.0          0.0          0.0           0.0          0.0         -0.661306034  0.0; 
   0.0          0.0          0.0           0.0          0.0          0.0 -0.661306034],
 [-2.255357452 -1.960978074 -4.484364189  -2.511681833  0.483603458  0.0  0.0; 
  -1.960978074 -2.255357452 -4.484364189  -2.511681833 -0.483603458  0.0  0.0; 
  -4.484364189 -4.484364189 -20.920360963 -5.363451041  0.0          0.0  0.0; 
  -2.511681833 -2.511681833 -5.363451041  -2.8963628   -0.0          0.0  0.0; 
   0.483603458 -0.483603458  0.0          -0.0         -1.28091496   0.0  0.0; 
   0.0          0.0          0.0           0.0          0.0         -0.661295305  0.0; 
   0.0          0.0          0.0           0.0          0.0          0.0 -0.661295305]), 
atol=errorThreshold3) |> all

@test isapprox.(res2.Emo, 
([-20.930370891, -1.616676585, -1.284472038, -0.661306034, 
   -0.661306034, 1.060819291, 1.847800507],
 [-20.930362238, -1.616663502, -1.284451995, -0.661295305, 
   -0.661295305, 1.060817247, 1.8478124]), 
atol=errorThreshold3) |> all

@test ( res2.occu .== ([1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0]) ) |> prod

D2s = res2.D
for D in D2s
    @test isapprox(D*S*D, D, atol=errorThreshold1)
end

# Potential energy curve tests.
nuc2 = ["H", "H"]
range = 0.1:0.2:20.1
Erhf = Float64[]
Euhf = Float64[]
Enuc = Float64[]

for i in range
    nucCoords2 = [[0, 0.0, 0.0], [i, 0.0, 0.0]]

    bs = genBasisFunc.(nucCoords2, "3-21G") |> flatten
    res = runHF(bs, nuc2, nucCoords2, printInfo=false)
    push!(Erhf, res.Ehf)
    push!(Euhf, runHF(bs, nuc2, nucCoords2, HFconfig((HF=:UHF,)), printInfo=false).Ehf)
    push!(Enuc, res.Enn)
end

rhfs = [ 7.275712508,  0.721327344, -0.450914129, -0.860294199, -1.029212153, -1.098483134, 
        -1.12116316,  -1.12068526,  -1.108423332, -1.090377472, -1.06979209,  -1.048358444, 
        -1.026929524, -1.005949751, -0.985675141, -0.966266828, -0.947825529, -0.93040561, 
        -0.914025059, -0.898675515, -0.884331776, -0.870959502, -0.858520505, -0.846975806, 
        -0.836287041, -0.826416815, -0.81732844,  -0.808985378, -0.801350617, -0.794386143, 
        -0.788052639, -0.782309455, -0.777114838, -0.772426373, -0.768201548, -0.764398372, 
        -0.760975967, -0.757895111, -0.755118673, -0.752611963, -0.750342966, -0.748282481, 
        -0.746404172, -0.74468454,  -0.743102829, -0.741640881, -0.740282953, -0.73901551, 
        -0.737826999, -0.736707621, -0.735649109, -0.734644515, -0.733688011, -0.732774716, 
        -0.73190053,  -0.731062005, -0.730256222, -0.729480696, -0.728733293, -0.728012163, 
        -0.727315685, -0.726642428, -0.725991112, -0.725360583, -0.72474979,  -0.724157774, 
        -0.723583648, -0.723026588, -0.72248583,  -0.721960657, -0.721450398, -0.720954422, 
        -0.720472134, -0.720002974, -0.719546412, -0.719101945, -0.718669076, -0.718247424, 
        -0.717836489, -0.717435899, -0.717045259, -0.716664205, -0.716292388, -0.715929478, 
        -0.715575157, -0.715229124, -0.714891092, -0.714560784, -0.714237946, -0.713922319, 
        -0.713613662, -0.71331176,  -0.713016384, -0.712727325, -0.712444379, -0.712165488, 
        -0.711876331, -0.707452753, -0.711352741, -0.710986418, -0.430694887]

uhfs = [ 7.275712839,  0.721327346, -0.450914119, -0.86029418,  -1.029212126, -1.098483095, 
        -1.121163097, -1.120685083, -1.108422503, -1.090367463, -1.069643099, -1.048656193, 
        -1.032357137, -1.020787226, -1.012591523, -1.006780867, -1.002658836, -0.999733889, 
        -0.997657959, -0.996184122, -0.995137062, -0.994392267, -0.993861304, -0.993481373, 
        -0.993207879, -0.993009217, -0.992863102, -0.992753979, -0.992671126, -0.99260725, 
        -0.992557458, -0.992518419, -0.992487826, -0.992464005, -0.992445668, -0.992431763, 
        -0.992421406, -0.992413842, -0.99240843,  -0.99240464,  -0.992402043, -0.992400301, 
        -0.992399158, -0.992398424, -0.992397962, -0.992397678, -0.992397506, -0.992397405, 
        -0.992397346, -0.992397312, -0.992397294, -0.992397283, -0.992397278, -0.992397275, 
        -0.992397274, -0.992397273, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272, 
        -0.992397272, -0.992397272, -0.992397272, -0.992397272, -0.992397272]

Et1 = Erhf + Enuc
Et2 = Euhf + Enuc
bools1 = isapprox.(Et1, rhfs, atol=2*errorThreshold3)
bools2 = isapprox.(Et2, uhfs, atol=errorThreshold3)
ids1 = findall(isequal(false), bools1)
ids2 = findall(isequal(false), bools2)
length(ids1) > 0 && (@show Et1[ids1] rhfs[ids1])
length(ids2) > 0 && (@show Et2[ids2] uhfs[ids2])

@test all(length(ids1) == 0)
@test all(length(ids2) == 0)

end