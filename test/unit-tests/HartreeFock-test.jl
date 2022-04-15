using Test
using Quiqbox
using Suppressor: @suppress_out

@testset "HartreeFock.jl" begin

errorThreshold1 = 1e-8
errorThreshold2 = 1e-4

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

@test isapprox(res2.Ehf, res2_2.Ehf, atol=1e-5)
@test isapprox(res1.Ehf, res1_2.Ehf, atol=1e-6)

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

@test isapprox(res1.Ehf, -93.7878386326277, atol=errorThreshold1)

# Note: the molecular energies of 4th and 5th columns are so close that based on the 
# numerical error of each machine the position of them might switch.

@test isapprox(res1.C[1][1:5, [1,2,3,6,7]], 
[ 0.010895948  0.088981718  0.121610580  0.0  0.0  1.914545170  3.615733228; 
    0.010895948  0.088981718 -0.121610580  0.0  0.0  1.914545170 -3.615733228; 
    -0.994229856 -0.263439202  0.0          0.0  0.0  0.049696590 -0.0; 
    -0.041593119  0.872248048  0.0          0.0  0.0 -3.396657723 -0.0; 
    0.0         -0.0          1.094022537  0.0  0.0  0.0          2.778671730; 
    0.0          0.0          0.0          1.0  0.0  0.0          0.0; 
    0.0          0.0          0.0          0.0  1.0  0.0          0.0][1:5, [1,2,3,6,7]], 
atol=errorThreshold2)

@test  isapprox(vcat(res1.C[1][6:7,:][:], res1.C[1][1:5, 4:5][:]) |> sort, 
                vcat(fill(0,22), fill(1,2)), atol=errorThreshold1)
@test  isapprox(res1.C[1][6:7, 4:5][:] |> sort, [0,0,1,1], atol=errorThreshold1)

@test isapprox(res1.F[1], 
[-2.255355742 -1.960980213  -4.484367700 -2.511687004  0.483602713  0.0  0.0; 
    -1.960980213 -2.255355742  -4.484367700 -2.511687004 -0.483602713  0.0  0.0; 
    -4.484367700 -4.484367700 -20.920373317 -5.363455107  0.0          0.0  0.0; 
    -2.511687004 -2.511687004  -5.363455107 -2.896374204 -0.0          0.0  0.0; 
    0.483602713 -0.483602713   0.0         -0.0         -1.280925059  0.0  0.0; 
    0.0          0.0           0.0          0.0          0.0 -0.661303892  0.0; 
    0.0          0.0           0.0          0.0          0.0  0.0 -0.661303892], 
atol=errorThreshold2)

@test isapprox(res1.Emo[1], 
[-20.930371651088734, -1.6166714662844843, -1.2844637689430092, -0.6613027570108698, 
    -0.661302757010868, 1.0608175741517432, 1.8478054667733597], 
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
                atol=errorThreshold1)

@test isapprox(res2.Ehf, -93.7878386328625, atol=errorThreshold1)

@test isapprox.((res2.C[1][1:5, [1,2,3,6,7]], res2.C[2][1:5, [1,2,3,6,7]]), 
([ 0.01089592  0.08898109  0.12160791  0.0  0.0  1.91454520  3.61573332; 
    0.01089592  0.08898109 -0.12160791  0.0  0.0  1.91454520 -3.61573332; 
    -0.99422987 -0.26343918  0.0         0.0  0.0  0.04969649  0.0; 
    -0.04159303  0.87224917  0.0         0.0  0.0 -3.39665744 -0.0; 
    0.0         0.0         1.09402049  0.0  0.0  0.0         2.77867254; 
    0.0         0.0         0.0         1.0  0.0  0.0         0.0; 
    0.0         0.0         0.0         0.0  1.0  0.0         0.0][1:5, [1,2,3,6,7]], 
    [ 0.01089592  0.08898112  0.12160785  0.0  0.0  1.91454520  3.61573332; 
    0.01089592  0.08898112 -0.12160785  0.0  0.0  1.91454520 -3.61573332; 
    -0.99422987 -0.26343918 -0.0         0.0  0.0  0.04969649  0.0; 
    -0.04159303  0.87224912 -0.0         0.0  0.0 -3.39665745  0.0; 
    0.0        -0.0         1.09402044  0.0  0.0  0.0         2.77867255; 
    0.0         0.0         0.0         1.0  0.0  0.0         0.0; 
    0.0         0.0         0.0         0.0  1.0  0.0         0.0][1:5, [1,2,3,6,7]]), 
atol=errorThreshold2) |> all

@test  isapprox(vcat(res2.C[1][6:7,:][:], res2.C[1][1:5, 4:5][:]) |> sort, 
                vcat(fill(0,22), fill(1,2)), atol=errorThreshold1)
@test  isapprox(res2.C[1][6:7, 4:5][:] |> sort, [0,0,1,1], atol=errorThreshold1)

@test  isapprox(vcat(res2.C[2][6:7,:][:], res2.C[2][1:5, 4:5][:]) |> sort, 
                vcat(fill(0,22), fill(1,2)), atol=errorThreshold1)
@test  isapprox(res2.C[2][6:7, 4:5][:] |> sort, [0,0,1,1], atol=errorThreshold1)

@test isapprox.(res2.F, 
([-2.25535869 -1.96098203  -4.48436922 -2.51168980  0.48360380  0.0  0.0; 
    -1.96098203 -2.25535869  -4.48436922 -2.51168980 -0.48360380  0.0  0.0; 
    -4.48436922 -4.48436922 -20.92038321 -5.36345685  0.0         0.0  0.0; 
    -2.51168980 -2.51168980  -5.36345685 -2.89637762 -0.0         0.0  0.0; 
    0.48360380 -0.48360380   0.0        -0.0        -1.28092706  0.0  0.0; 
    0.0         0.0          0.0         0.0         0.0 -0.66130761  0.0; 
    0.0         0.0          0.0         0.0         0.0  0.0 -0.66130761],
    [-2.25535870 -1.96098203  -4.48436922 -2.51168979  0.48360381  0.0  0.0; 
    -1.96098203 -2.25535870  -4.48436922 -2.51168979 -0.48360381  0.0  0.0; 
    -4.48436922 -4.48436922 -20.92038320 -5.36345684 -0.0         0.0  0.0; 
    -2.51168979 -2.51168979  -5.36345684 -2.89637760  0.0         0.0  0.0; 
    0.48360381 -0.48360381  -0.0         0.0        -1.28092705  0.0  0.0; 
    0.0         0.0          0.0         0.0         0.0 -0.66130760  0.0; 
    0.0         0.0          0.0         0.0         0.0  0.0 -0.66130760]), 
atol=errorThreshold2) |> all

@test isapprox.(res2.Emo, 
([-20.930384503, -1.616675737, -1.284466209, -0.661307611, 
    -0.661307611,  1.060815275,  1.847804069],
    [-20.930384496, -1.616675722, -1.284466197, -0.661307600, 
    -0.661307600,  1.060815276,  1.847804076]), 
atol=errorThreshold2) |> all

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

    push!(Erhf, runHF(bs, nuc2, nucCoords2, printInfo=false).Ehf)
    push!(Euhf, runHF(bs, nuc2, nucCoords2, HFconfig((HF=:UHF,)), printInfo=false).Ehf)
    push!(Enuc, nnRepulsions(nuc2, nucCoords2))
end

rhfs = [ 7.275712610373,  0.721327425358, -0.450914043001, -0.860294145337, 
        -1.029211895595, -1.098482919344, -1.1211630526,   -1.120685233186, 
        -1.108423327321, -1.090377441376, -1.069791890024, -1.04835836562, 
        -1.026929475848, -1.005949728492, -0.98567513333,  -0.966266829712, 
        -0.947825530697, -0.930405604974, -0.91402504744,  -0.89867549644, 
        -0.884331750143, -0.870959470736, -0.858520470378, -0.846975769316, 
        -0.836287004213, -0.826416778652, -0.817328404285, -0.808985343967, 
        -0.801350584529, -0.794386112574, -0.788052610965, -0.782309428586, 
        -0.777114812945, -0.772426349011, -0.768201525603, -0.764398350215, 
        -0.760975946893, -0.757895091056, -0.75511865398,  -0.75261194475, 
        -0.750342948136, -0.74828246371,  -0.746404155536, -0.744684524077, 
        -0.743102813151, -0.741640865194, -0.740282938015, -0.739015495636, 
        -0.737826984748, -0.736707606945, -0.735649095172, -0.734644501045, 
        -0.733687997847, -0.732774702301, -0.731900516696, -0.731061991693, 
        -0.730256209147, -0.729480683615, -0.728733280739, -0.728012150488, 
        -0.727315673183, -0.726642416292, -0.725991100124, -0.725360570773, 
        -0.724749778867, -0.724157762914, -0.723583636232, -0.723026576666, 
        -0.722485818435, -0.721960645612, -0.721450386841, -0.720954411001, 
        -0.720472123582, -0.720002963618, -0.719546401045, -0.719101934388, 
        -0.718669088723, -0.718247413846, -0.717836482627, -0.71743588951, 
        -0.717045249141, -0.716664195111, -0.716292377603, -0.715929468288, 
        -0.715575147391, -0.71522911471,  -0.714891082788, -0.714560777313, 
        -0.71423793637,  -0.713922309745, -0.713613659952, -0.713311753559, 
        -0.713016377709, -0.712727128398, -0.71244425118,  -0.712167361249, 
        -0.711895952512, -0.711630166362, -0.711369938473, -0.711100587417, 
        -0.710864978544]

uhfs = [ 7.275712621796,  0.72132739244,  -0.450914071514, -0.860294155274, 
        -1.029212110646, -1.0984830933,   -1.121163118023, -1.120685227611, 
        -1.10842330357,  -1.090377445981, -1.069792065781, -1.04870444924, 
        -1.032357156434, -1.020787192945, -1.012591492492, -1.006780830167, 
        -1.002658815209, -0.999733901015, -0.997657832375, -0.996183603534, 
        -0.995136916495, -0.994391730235, -0.993860901748, -0.993481147602, 
        -0.99320776654,  -0.993009159531, -0.992863073239, -0.992753963779, 
        -0.992671118191, -0.992607253863, -0.992557462317, -0.992518421363, 
        -0.992487827135, -0.992464006117, -0.992445668325, -0.992431763592, 
        -0.99242140648,  -0.992413841744, -0.992408429933, -0.992404640149, 
        -0.992402043047, -0.992400301455, -0.992399158486, -0.992398424239, 
        -0.992397962409, -0.992397677907, -0.992397506203, -0.992397404647, 
        -0.992397345764, -0.992397312286, -0.992397293617, -0.992397283402, 
        -0.992397277917, -0.992397275026, -0.99239727353,  -0.99239727277, 
        -0.99239727239,  -0.992397272204, -0.992397272115, -0.992397272072, 
        -0.992397272053, -0.992397272044, -0.992397272039, -0.992397272038, 
        -0.992397272037, -0.992397272037, -0.992397272036, -0.992397272036, 
        -0.992397272036, -0.992397272036, -0.992397272036, -0.992397272036, 
        -0.992397272036, -0.992397272036, -0.992397272036, -0.992397272036, 
        -0.992397272036, -0.992397272036, -0.992397272036, -0.992397272036, 
        -0.992397272036, -0.992397272036, -0.992397272036, -0.992397272036, 
        -0.992397272036, -0.992397272036, -0.992397272036, -0.992397272036, 
        -0.992397272036, -0.992397272036, -0.992397272036, -0.992397272036, 
        -0.992397272036, -0.992397272036, -0.992397272036, -0.992397272036, 
        -0.992397272036, -0.992397272036, -0.992397272036, -0.992397272036, 
        -0.992397272036]

@test isapprox.(Erhf .+ Enuc, rhfs, atol=errorThreshold2) |> all
@test isapprox.(Euhf .+ Enuc, uhfs, atol=errorThreshold2) |> all

end