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

    local res1, res2
    @suppress_out begin
        res1 = runHF(bs, nuc, nucCoords; HFtype=:RHF, initialC=:Hcore,
                    scfConfig=SCFconfig([:ADIIS, :DIIS, :EDIIS, :SD], 
                                        [1e-4, 1e-8, 1e-10, 1e-12]))
        res2 = runHF(bs, nuc, nucCoords; HFtype=:UHF, 
                    scfConfig=SCFconfig([:ADIIS, :DIIS, :EDIIS, :SD], 
                                        [1e-4, 1e-8, 1e-10, 1e-12],
                                        Dict(1=>[:solver=>:LCM],
                                             2=>[:solver=>:LCM],
                                             3=>[:solver=>:LCM],
                                             4=>[:solver=>:LCM])))
    end


    @test begin
        tVars1 = deepcopy(res1.temp)
        Quiqbox.popHFtempVars!(tVars1)
        push!(tVars1.Cs, res1.temp.Cs[end])
        push!(tVars1.Fs, res1.temp.Fs[end])
        push!(tVars1.Ds, res1.temp.Ds[end])
        push!(tVars1.Es, res1.temp.Es[end])
        push!(tVars1.shared.Dtots, res1.temp.shared.Dtots[end])
        push!(tVars1.shared.Etots, res1.temp.shared.Etots[end])
        hasEqual(tVars1, res1.temp)
    end

    @test isapprox(res1.E0HF, Quiqbox.getEᵀ(Hcore, HeeI, res1.C, Ne), atol=errorThreshold1)

    @test isapprox(res1.E0HF, -93.7878386326277, atol=errorThreshold1)

    # Note: the molecular energies of 4th and 5th columns are so close that based on the 
    # numerical error of each machine the position of them might switch.

    @test isapprox(res1.C[1:5, [1,2,3,6,7]], 
    [ 0.010895948  0.088981718  0.121610580  0.0  0.0  1.914545170  3.615733228; 
      0.010895948  0.088981718 -0.121610580  0.0  0.0  1.914545170 -3.615733228; 
     -0.994229856 -0.263439202  0.0          0.0  0.0  0.049696590 -0.0; 
     -0.041593119  0.872248048  0.0          0.0  0.0 -3.396657723 -0.0; 
      0.0         -0.0          1.094022537  0.0  0.0  0.0          2.778671730; 
      0.0          0.0          0.0          1.0  0.0  0.0          0.0; 
      0.0          0.0          0.0          0.0  1.0  0.0          0.0][1:5, [1,2,3,6,7]], 
    atol=errorThreshold2)

    @test  isapprox(vcat(res1.C[6:7,:][:], res1.C[1:5, 4:5][:]) |> sort, 
                    vcat(fill(0,22), fill(1,2)), atol=errorThreshold1)
    @test  isapprox(res1.C[6:7, 4:5][:] |> sort, [0,0,1,1], atol=errorThreshold1)

    @test isapprox(res1.F, 
    [-2.255355742 -1.960980213  -4.484367700 -2.511687004  0.483602713  0.0  0.0; 
     -1.960980213 -2.255355742  -4.484367700 -2.511687004 -0.483602713  0.0  0.0; 
     -4.484367700 -4.484367700 -20.920373317 -5.363455107  0.0          0.0  0.0; 
     -2.511687004 -2.511687004  -5.363455107 -2.896374204 -0.0          0.0  0.0; 
      0.483602713 -0.483602713   0.0         -0.0         -1.280925059  0.0  0.0; 
      0.0          0.0           0.0          0.0          0.0 -0.661303892  0.0; 
      0.0          0.0           0.0          0.0          0.0  0.0 -0.661303892], 
    atol=errorThreshold2)

    @test isapprox(res1.Emo, 
    [-20.930374644, -1.616672450, -1.284464364, -0.661303892, 
      -0.661303892,  1.060817046,  1.847805126], 
    atol=errorThreshold1)

    @test res1.occu == [2, 2, 2, 2, 2, 0, 0]

    D1 = res1.D
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

    @test isapprox(res2.E0HF, Quiqbox.getEᵀ(Hcore, HeeI, res2.C, (Ne÷2, Ne-Ne÷2)), 
                   atol=errorThreshold1)

    @test isapprox(res2.E0HF, -93.7878386328625, atol=errorThreshold1)

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
    atol=errorThreshold2) |> prod

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
    atol=errorThreshold2) |> prod

    @test isapprox.(res2.Emo, 
    ([-20.930384503, -1.616675737, -1.284466209, -0.661307611, 
       -0.661307611,  1.060815275,  1.847804069],
     [-20.930384496, -1.616675722, -1.284466197, -0.661307600, 
       -0.661307600,  1.060815276,  1.847804076]), 
    atol=errorThreshold2) |> prod

    @test ( res2.occu .== ([1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0]) ) |> prod

    D2s = res2.D
    for D in D2s
        @test isapprox(D*S*D, D, atol=errorThreshold1)
    end

end