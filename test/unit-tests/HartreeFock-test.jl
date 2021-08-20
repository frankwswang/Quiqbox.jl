push!(LOAD_PATH, "./Quiqbox")
using Quiqbox
using Test

@testset "HartreeFock.jl" begin

    errorThreshold = 1e-8

    nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]
    mol = ["H", "H"]
    bs = BasisFunc.(nucCoords, Ref(("STO-3G", "H"))) |> flatten
    res = runHF(bs, mol, nucCoords; HFtype=:RHF, printInfo=false)
    @test isapprox(res.E0HF, -1.8310000393482668, atol=errorThreshold)
    @test isapprox(res.C, [-0.54893404 -1.21146407; -0.54893404 1.21146407], atol=errorThreshold)
    @test isapprox(res.F, [-0.36553735 -0.59388538; -0.59388538 -0.36553735], atol=errorThreshold)
    @test isapprox(res.Emo, [-0.57820298,  0.67026777], atol=errorThreshold)
    @test res.occu == [2,0]
    D = res.D
    S = overlaps(bs)[:,:,1]
    @test isapprox(D*S*D, D, atol=errorThreshold)

end