using Test
using Quiqbox
using Suppressor: @suppress_out

@testset "HartreeFock.jl" begin

    errorThreshold = 1e-9

    nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0], [0.0, 0.0, 0.0]]
    mol = ["H", "H", "O"]
    bs = genBasisFunc.(nucCoords, ("STO-3G", "STO-3G", ("STO-3G", "O"))) |> flatten
    S = dropdims(overlaps(bs), dims=3)

    local res1, res2
    @suppress_out begin
        res1 = runHF(bs, mol, nucCoords; HFtype=:RHF, initialC=:Hcore,
                    scfConfig=Quiqbox.SCFconfig([:ADIIS, :DIIS, :EDIIS, :SD], 
                                                [1e-4, 1e-8, 1e-10, 1e-12]))
        res2 = runHF(bs, mol, nucCoords; HFtype=:UHF, 
                    scfConfig=Quiqbox.SCFconfig([:ADIIS, :DIIS, :EDIIS, :SD], 
                                                [1e-4, 1e-8, 1e-10, 1e-12],
                                                Dict(1=>[:solver=>:Direct],
                                                     2=>[:solver=>:Direct],
                                                     3=>[:solver=>:Direct],
                                                     4=>[:solver=>:Direct])))
    end

    @test isapprox(res1.E0HF, -93.7878386326277, atol=errorThreshold)

    @test isapprox(res1.C, 
    [0.0108959475  0.088981718  0.1216105801 0.0 0.0 -1.9145451703  3.6157332281; 
     0.0108959475  0.088981718 -0.1216105801 0.0 0.0 -1.9145451703 -3.6157332281; 
    -0.9942298564 -0.263439202  0.0          0.0 0.0 -0.0496965897 -0.0; 
    -0.0415931194  0.8722480485 0.0          0.0 0.0  3.3966577232 -0.0; 
     0.0          -0.0          1.0940225371 0.0 0.0  0.0           2.7786717301; 
     0.0           0.0          0.0          1.0 0.0  0.0           0.0; 
     0.0           0.0          0.0          0.0 1.0  0.0           0.0], 
    atol=errorThreshold)
    
    @test isapprox(res1.F, 
    [-2.2553557425 -1.9609802133  -4.4843676997 -2.5116870044  0.4836027134 0.0 0.0; 
     -1.9609802133 -2.2553557425  -4.4843676997 -2.5116870044 -0.4836027134 0.0 0.0; 
     -4.4843676997 -4.4843676997 -20.9203733167 -5.3634551066  0.0          0.0 0.0; 
     -2.5116870044 -2.5116870044  -5.3634551066 -2.8963742039 -0.0          0.0 0.0; 
      0.4836027134 -0.4836027134   0.0          -0.0          -1.2809250587 0.0 0.0; 
      0.0           0.0            0.0           0.0           0.0 -0.6613038925  0.0; 
      0.0           0.0            0.0           0.0           0.0  0.0 -0.6613038925], 
    atol=errorThreshold)
    
    @test isapprox(res1.Emo, 
    [-20.9303746442, -1.6166724499, -1.2844643644, -0.6613038925, 
      -0.6613038925,  1.0608170463,  1.8478051263], 
    atol=errorThreshold)
    
    @test res1.occu == [2, 2, 2, 2, 2, 0, 0]
    
    D1 = res1.D
    @test isapprox(D1*S*D1, D1, atol=errorThreshold)


    @test isapprox(res2.E0HF, -93.7878386328625, atol=errorThreshold)
    
    @test isapprox.(res2.C, 
    ([-0.0108959194 -0.088981086   0.1216079134 0.0 0.0 -1.9145451998  3.6157333178; 
      -0.0108959194 -0.088981086  -0.1216079135 0.0 0.0 -1.9145451998 -3.6157333178; 
       0.9942298673  0.2634391803  0.0          0.0 0.0 -0.0496964865  0.0; 
       0.04159303   -0.8722491705  0.0          0.0 0.0  3.3966574362 -0.0; 
       0.0          -0.0           1.0940204878 0.0 0.0 -0.0           2.7786725369; 
       0.0           0.0           0.0          1.0 0.0  0.0           0.0; 
       0.0           0.0           0.0          0.0 1.0  0.0           0.0], 
     [-0.0108959194  0.0889811163  0.1216078542 0.0 0.0 -1.9145451984 -3.6157333198; 
      -0.0108959194  0.0889811163 -0.1216078542 0.0 0.0 -1.9145451984  3.6157333198; 
       0.9942298674 -0.2634391795 -0.0          0.0 0.0 -0.0496964906  0.0; 
       0.0415930299  0.8722491168 -0.0          0.0 0.0  3.3966574499 -0.0; 
      -0.0          -0.0           1.0940204422 0.0 0.0  0.0          -2.7786725549; 
       0.0           0.0           0.0          1.0 0.0  0.0           0.0; 
       0.0           0.0           0.0          0.0 1.0  0.0           0.0]), 
    atol=errorThreshold) |> prod
    
    @test isapprox.(res2.F, 
    ([-2.2553586896 -1.9609820313  -4.4843692187 -2.5116897961  0.4836038059 0.0 0.0; 
      -1.9609820313 -2.2553586896  -4.4843692187 -2.5116897961 -0.4836038059 0.0 0.0; 
      -4.4843692187 -4.4843692187 -20.9203832094 -5.363456848   0.0          0.0 0.0; 
      -2.5116897961 -2.5116897961  -5.363456848  -2.8963776226 -0.0          0.0 0.0; 
       0.4836038059 -0.4836038059   0.0          -0.0          -1.2809270581 0.0 0.0; 
       0.0           0.0            0.0           0.0           0.0 -0.6613076106  0.0; 
       0.0           0.0            0.0           0.0           0.0  0.0 -0.6613076106],
     [-2.2553586979 -1.9609820317  -4.4843692157 -2.5116897904  0.4836038095 0.0 0.0; 
      -1.9609820317 -2.2553586979  -4.4843692157 -2.5116897904 -0.4836038095 0.0 0.0; 
      -4.4843692157 -4.4843692157 -20.920383202  -5.3634568448 -0.0          0.0 0.0; 
      -2.5116897904 -2.5116897904  -5.3634568448 -2.896377604   0.0          0.0 0.0; 
       0.4836038095 -0.4836038095  -0.0           0.0          -1.2809270492 0.0 0.0; 
       0.0           0.0            0.0           0.0           0.0 -0.6613075996  0.0; 
       0.0           0.0            0.0           0.0           0.0  0.0 -0.6613075996]), 
    atol=errorThreshold) |> prod
    
    @test isapprox.(res2.Emo, 
    ([-20.9303845033, -1.6166757369, -1.2844662094, -0.6613076106, 
       -0.6613076106,  1.0608152747,  1.8478040688],
     [-20.9303844958, -1.6166757223, -1.2844661972, -0.6613075996, 
       -0.6613075996,  1.0608152755,  1.8478040763]), 
    atol=errorThreshold) |> prod
    
    @test ( res2.occu .== ([1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0]) ) |> prod
    
    D2s = res2.D
    for D in D2s
        @test isapprox(D*S*D, D, atol=errorThreshold)
    end

end