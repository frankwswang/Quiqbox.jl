using Quiqbox: numEps, hasApprox
using WignerSymbols
using Test


function testCG(j1, m1, j2, m2, j3)

    t = 2numEps(Float64)

    #= (m1-1,  m2)   (m1  ,  m2) =#
    #=     cg1___________cg3     =#
    #=       |\          |       =#
    #=       |  \        |       =#
    #=       |    \      |       =#
    #=       |      \    |       =#
    #=       |        \  |       =#
    #=       |__________\|       =#
    #=     cg4           cg2     =#
    #= (m1-1,m2-1)   (m1  ,m2-1) =#

    cg1_1 = CGcoeff(j1, m1-1, j2, m2,   j3)
    cg2_1 = CGcoeff(j1, m1,   j2, m2-1, j3)
    cg3_1 = genCGcoeff(+, cg1_1, cg2_1)
    cg3_2 = genCGcoeff(+, cg2_1, cg1_1)
    cg4_1 = genCGcoeff(-, cg1_1, cg2_1)
    cg4_2 = genCGcoeff(-, cg2_1, cg1_1)

    @test cg3_1 == cg3_2
    @test cg3_1.m1 == m1
    @test cg3_1.m2 == m2
    @test cg4_1 == cg4_2
    @test cg4_1.m1 == m1-1
    @test cg4_1.m2 == m2-1
    @test typeof(cg3_1) == typeof(cg4_1) == CGcoeff{Float64, Int(2j1), Int(2j2), Int(2j3)}
    @test hasApprox(cg3_1.coeff, getCGcoeff(cg3_1), getCGcoeff(j1, m1, j2, m2, j3), 
          clebschgordan(Float64, j1, m1, j2, m2, j3), atol=t)
    @test hasApprox(cg4_1.coeff, getCGcoeff(cg4_1), getCGcoeff(j1, m1-1, j2, m2-1, j3), 
          clebschgordan(Float64, j1, m1-1, j2, m2-1, j3), atol=t)

    cg1_2 = genCGcoeff(-, cg3_1, cg2_1)
    cg1_3 = genCGcoeff(-, cg2_1, cg3_1)
    cg2_2 = genCGcoeff(-, cg3_1, cg1_1)
    cg2_3 = genCGcoeff(-, cg1_1, cg3_1)
    cg1_4 = genCGcoeff(+, cg4_1, cg2_1)
    cg1_5 = genCGcoeff(+, cg2_1, cg4_1)
    cg2_4 = genCGcoeff(+, cg4_1, cg1_1)
    cg2_5 = genCGcoeff(+, cg1_1, cg4_1)

    @test cg1_1.m1 == cg1_2.m1 == cg1_3.m1 == cg1_4.m1 == cg1_5.m1
    @test cg1_1.m2 == cg1_2.m2 == cg1_3.m2 == cg1_4.m2 == cg1_5.m2
    @test hasApprox(cg1_1.coeff, cg1_2.coeff, cg1_3.coeff, cg1_4.coeff, cg1_5.coeff, atol=t)

    @test cg2_1.m1 == cg2_2.m1 == cg2_3.m1 == cg2_4.m1 == cg2_5.m1
    @test cg2_1.m2 == cg2_2.m2 == cg2_3.m2 == cg2_4.m2 == cg2_5.m2
    @test hasApprox(cg2_1.coeff, cg2_2.coeff, cg2_3.coeff, cg2_4.coeff, cg2_5.coeff, atol=t)
end

j1m1j2m2j3_1 = (2, 2, 2, -1, 3)
testCG(j1m1j2m2j3_1...)

j1m1j2m2j3_2 = (1.5, 0.5, 2, -1, 3.5)
testCG(j1m1j2m2j3_2...)
