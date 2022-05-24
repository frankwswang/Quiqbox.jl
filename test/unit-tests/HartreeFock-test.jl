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

@test isapprox(res1.Ehf, -93.78783863249271, atol=errorThreshold1)

# Note: the molecular energies of 4th and 5th columns are so close that based on the 
# numerical error of each machine the position of them might switch.

@test isapprox(res1.C[1][1:5, [1,2,3,6,7]], 
[0.01089595 0.08898205 0.12161138 0.0 0.0 1.91454515 3.6157332; 
 0.01089595 0.08898205 -0.12161138 0.0 0.0 1.91454515 -3.6157332; 
 -0.99422985 -0.2634392 -0.0 0.0 0.0 0.04969664 0.0; 
 -0.04159314 0.87224746 -0.0 0.0 0.0 -3.39665787 0.0; 
 0.0 -0.0 1.09402315 0.0 0.0 0.0 2.77867149; 
 -0.0 0.0 0.0 1.0 0.0 0.0 -0.0; 
 -0.0 0.0 0.0 0.0 1.0 0.0 -0.0][1:5, [1,2,3,6,7]], 
atol=errorThreshold2)

@test  isapprox(vcat(res1.C[1][6:7,:][:], res1.C[1][1:5, 4:5][:]) |> sort, 
                vcat(fill(0,22), fill(1,2)), atol=errorThreshold1)
@test  isapprox(res1.C[1][6:7, 4:5][:] |> sort, [0,0,1,1], atol=errorThreshold1)

@test isapprox(res1.F[1], 
[-2.25535526 -1.96097994 -4.48436744 -2.51168649 0.48360253 0.0 0.0; 
 -1.96097994 -2.25535526 -4.48436744 -2.51168649 -0.48360253 0.0 0.0; 
 -4.48436744 -4.48436744 -20.92037142 -5.36345481 -0.0 0.0 0.0; 
 -2.51168649 -2.51168649 -5.36345481 -2.89637349 -0.0 0.0 0.0; 
 0.48360253 -0.48360253 -0.0 -0.0 -1.28092484 0.0 0.0; 
 0.0 0.0 0.0 0.0 0.0 -0.66130322 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 -0.66130322], 
atol=errorThreshold2)

@test isapprox(res1.Emo[1], 
[-20.930372760177335, -1.6166717611251709, -1.2844641966496144, -0.6613032221010628, 
  -0.6613032221010611, 1.0608174350839286, 1.8478052261514148], 
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
([0.01089599 0.08896693 0.12165132 0.0 0.0 1.91454586 3.61573186; 
  0.01089599 0.08896693 -0.12165132 0.0 0.0 1.91454586 -3.61573186; 
  -0.99422983 -0.2634397 0.0 0.0 0.0 0.04969458 -0.0; 
  -0.0415933 0.87227428 -0.0 0.0 0.0 -3.39665099 0.0; 
  0.0 -0.0 1.09405385 0.0 0.0 0.0 2.7786594; 
  0.0 -0.0 0.0 1.0 0.0 0.0 -0.0; 
  0.0 -0.0 0.0 0.0 1.0 0.0 -0.0][1:5, [1,2,3,6,7]], 
 [0.01089594 0.08899556 0.12157167 0.0 0.0 1.91454453 3.61573454; 
  0.01089594 0.08899556 -0.12157167 0.0 0.0 1.91454453 -3.61573454; 
  -0.99422987 -0.26343878 0.0 0.0 0.0 0.04969849 -0.0; 
  -0.04159304 0.8722235 -0.0 0.0 0.0 -3.39666403 0.0; 
  0.0 -0.0 1.09399263 0.0 0.0 0.0 2.7786835; 
  -0.0 0.0 0.0 1.0 0.0 0.0 -0.0; 
  -0.0 0.0 0.0 0.0 1.0 0.0 -0.0][1:5, [1,2,3,6,7]]), 
atol=errorThreshold3*2) |> all

@test  isapprox(vcat(res2.C[1][6:7,:][:], res2.C[1][1:5, 4:5][:]) |> sort, 
                vcat(fill(0,22), fill(1,2)), atol=errorThreshold1)
@test  isapprox(res2.C[1][6:7, 4:5][:] |> sort, [0,0,1,1], atol=errorThreshold1)

@test  isapprox(vcat(res2.C[2][6:7,:][:], res2.C[2][1:5, 4:5][:]) |> sort, 
                vcat(fill(0,22), fill(1,2)), atol=errorThreshold1)
@test  isapprox(res2.C[2][6:7, 4:5][:] |> sort, [0,0,1,1], atol=errorThreshold1)

@test isapprox.(res2.F, 
([-2.25534848 -1.96097867 -4.48436847 -2.51168745 0.48359989 0.0 0.0; 
  -1.96097867 -2.25534848 -4.48436847 -2.51168745 -0.48359989 0.0 0.0; 
  -4.48436847 -4.48436847 -20.92036947 -5.36345596 -0.0 0.0 0.0; 
  -2.51168745 -2.51168745 -5.36345596 -2.8963806 0.0 0.0 0.0; 
  0.48359989 -0.48359989 -0.0 0.0 -1.28093036 0.0 0.0; 
  0.0 0.0 0.0 0.0 0.0 -0.66130603 0.0; 
  0.0 0.0 0.0 0.0 0.0 0.0 -0.66130603],
 [-2.25535745 -1.96097807 -4.48436419 -2.51168183 0.48360346 0.0 0.0; 
  -1.96097807 -2.25535745 -4.48436419 -2.51168183 -0.48360346 0.0 0.0; 
  -4.48436419 -4.48436419 -20.92036096 -5.36345104 0.0 0.0 0.0; 
  -2.51168183 -2.51168183 -5.36345104 -2.8963628 -0.0 0.0 0.0; 
  0.48360346 -0.48360346 0.0 -0.0 -1.28091496 0.0 0.0; 
  0.0 0.0 0.0 0.0 0.0 -0.6612953 0.0; 
  0.0 0.0 0.0 0.0 0.0 0.0 -0.6612953]), 
atol=errorThreshold3) |> all

@test isapprox.(res2.Emo, 
([-20.93037089, -1.61667659, -1.28447204, -0.66130603, -0.66130603, 1.06081929, 1.84780051],
 [-20.93036224, -1.6166635, -1.28445199, -0.6612953, -0.6612953, 1.06081725, 1.8478124]), 
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

rhfs = [7.2757127668545785, 0.7213274973807815, -0.45091403741562086, -0.8602941081782769, 
       -1.029212042962497, -1.0984830692109457, -1.1211630938050485, -1.1206852133802436, 
       -1.1084232875772644, -1.090377431346928, -1.0697920067388855, -1.0483581554376222, 
       -1.0269292965561245, -1.0059496482904273, -0.9856751085213245, -0.9662668272003632, 
       -0.9478255291021089, -0.9304055920594522, -0.9140250175338589, -0.8986754483787756, 
       -0.8843316855905916, -0.8709593928951231, -0.8585203830415647, -0.84697567621216, 
       -0.8362869086029778, -0.8264166831406037, -0.8173283107718037, -0.8089852537041184, 
       -0.80135049822817, -0.7943860305307234, -0.7880525331800337, -0.7823093548711516, 
       -0.7771147429995311, -0.7724262824850949, -0.7682014621347472, -0.7643982894580374, 
       -0.7609758885279007, -0.7578950347980836, -0.7551185995812673, -0.7526118919951361, 
       -0.7503428968435247, -0.7482824137255113, -0.7464041067274371, -0.7446844763345989, 
       -0.7431027663814544, -0.7416408193171333, -0.7402828929633551, -0.7390154513501331, 
       -0.7378269411781682, -0.7367075640466078, -0.7356490529059183, -0.7346444593763232, 
       -0.7336879567449889, -0.7327746617380139, -0.7319004766472027, -0.7310619521338488, 
       -0.7302561700566191, -0.7294806449739997, -0.7287332425283924, -0.7280121126903584, 
       -0.7273156357831136, -0.7266423792743919, -0.7259910634741232, -0.7253605344772064, 
       -0.7247497429133235, -0.7241577272899429, -0.7235836009263843, -0.7230265416678303, 
       -0.7224857837341566, -0.721960611198566, -0.7214503527061584, -0.7209543771351253, 
       -0.7204720899769308, -0.7200029302656689, -0.7195463679372501, -0.7191019015188943, 
       -0.7186690558847754, -0.7182473814302621, -0.7178364503825718, -0.7174355559543605, 
       -0.717045177409274, -0.7166641954891616, -0.7162923767323952, -0.7159294686836044, 
       -0.7155750969889383, -0.7152291123960711, -0.7148910800968856, -0.7145607776283089, 
       -0.7142379383095195, -0.7139223116085559, -0.7136136583848317, -0.7133117533316834, 
       -0.7130162970210063, -0.7127269870567569, -0.7124443816747767, -0.7121666551371263, 
       -0.7118960074309009, -0.7116299731258694, -0.7113696254804788, -0.7111149619025527, 
       -0.710864697537912]

uhfs = [7.275713284672634, 0.7213287429469424, -0.4509137256541762, -0.8602642905808884, 
        -1.0292118396174126, -1.0984826682014646, -1.1211627994943825, -1.120684870316314, 
        -1.1084230302151359, -1.0903771963586357, -1.0697918159158242, -1.0487040764653512, 
        -1.0323568330364417, -1.020786874742396, -1.0125911711771394, -1.0067804527198425, 
        -1.002658552644098, -0.9997335279516318, -0.9976574646767506, -0.9961827657365068, 
        -0.995135758707874, -0.9943913235328596, -0.9938606846775989, -0.9934809999819486, 
        -0.9932075642496354, -0.9930090531215714, -0.9928630168322488, -0.99275393365695, 
        -0.9926711019925506, -0.9926072450947927, -0.9925574575413081, -0.9925184187489117, 
        -0.992487825698046, -0.9924640053239382, -0.9924456678871436, -0.9924317633502091, 
        -0.9924214063466105, -0.9924138416706483, -0.992408429893042, -0.9924046401265596, 
        -0.9924020430350436, -0.9924003014481895, -0.9923991584821694, -0.9923984242373424, 
        -0.992397962407588, -0.992397677906542, -0.9923975062026943, -0.9923974046469842, 
        -0.9923973457642107, -0.9923973122862855, -0.992397293616628, -0.9923972834016768, 
        -0.9923972779168063, -0.9923972750259389, -0.9923972735299914, -0.9923972727698045, 
        -0.9923972723903837, -0.9923972722043475, -0.9923972721147248, -0.9923972720722959, 
        -0.992397272052554, -0.9923972720435261, -0.9923972720394666, -0.9923972720376729, 
        -0.9923972720368915, -0.9923972720365589, -0.9923972720364199, -0.9923972720363609, 
        -0.9923972720363384, -0.9923972720363285, -0.9923972720363253, -0.9923972720363229, 
        -0.9923972720363224, -0.9923972720363213, -0.992397272036322, -0.9923972720363218, 
        -0.9923972720363226, -0.9923972720363219, -0.9923972720363219, -0.9923972720363223, 
        -0.9923972720363218, -0.9923972720363222, -0.9923972720363219, -0.9923972720363217, 
        -0.9923972720363222, -0.992397272036322, -0.9923972720363218, -0.9923972720363217, 
        -0.9923972720363218, -0.992397272036321, -0.9923972720363216, -0.992397272036322, 
        -0.992397272036322, -0.9923972720363219, -0.992397272036322, -0.9923972720363214, 
        -0.9923972720363217, -0.9923972720363221, -0.9923972720363211, -0.9923972720363214, 
        -0.9923972720363219]

@test isapprox.(Erhf .+ Enuc, rhfs, atol=errorThreshold2) |> all
@test isapprox.(Euhf .+ Enuc, uhfs, atol=errorThreshold2) |> all

end