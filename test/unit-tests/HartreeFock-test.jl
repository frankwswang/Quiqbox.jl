using Test
using Quiqbox
using Suppressor: @suppress_out

@testset "HartreeFock.jl" begin

    errorThresholds = (1e-6, 1e-8)

    nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]
    mol = ["H", "H"]
    bs = genBasisFunc.(nucCoords, Ref(("STO-3G", "H"))) |> flatten

    local res1, res2
    @suppress_out begin
        res1 = runHF(bs, mol, nucCoords; HFtype=:RHF, initialC=:GWH,
                    scfConfig=Quiqbox.SCFconfig([:ADIIS, :DIIS, :EDIIS, :SD], 
                                                [1e-4, 1e-8, 1e-10, 1e-12]))
        res2 = runHF(bs, mol, nucCoords; HFtype=:RHF, 
                    scfConfig=Quiqbox.SCFconfig([:ADIIS, :DIIS, :EDIIS, :SD], 
                                                [1e-4, 1e-8, 1e-10, 1e-12],
                                                Dict(1=>[:solver=>:Convex],
                                                     2=>[:solver=>:Convex],
                                                     3=>[:solver=>:Convex],
                                                     4=>[:solver=>:Convex])))
    end

    for (res, errorThreshold) in zip((res1, res2), errorThresholds)
        @test isapprox(res.E0HF, -1.8310000393482668, atol=errorThreshold)
        @test isapprox(res.C, [-0.54893404 -1.21146407; -0.54893404 1.21146407], 
                       atol=errorThreshold)
        @test isapprox(res.F, [-0.36553735 -0.59388538; -0.59388538 -0.36553735], 
                       atol=errorThreshold)
        @test isapprox(res.Emo, [-0.57820298,  0.67026777], atol=errorThreshold)
        @test res.occu == [2,0]
        D = res.D
        S = overlaps(bs)[:,:,1]
        @test isapprox(D*S*D, D, atol=errorThreshold)
    end

end