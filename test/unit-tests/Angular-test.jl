using Test
using Quiqbox
using Quiqbox: WeakComp, CartSHarmonics, CGcoeff, genCGcoeff, getCGcoeff

@testset "Angular.jl" begin

@test try
    WeakComp(())
catch
    true
end

ang = (3,2,5)
point = (0.3, 0.5, -0.6)
CartSHarmonics(ang)(point) == prod(point .^ WeakComp(ang).tuple)

testCG = function (j1::Real, m1::Real, j2::Real, m2::Real, j3::Real; 
                   refVals::NTuple{2, Real}=(0, 0))

    t = 2Quiqbox.numEps(Float64)

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

    @test isapprox(cg3_1.coeff, first(refVals), atol=t)
    @test isapprox(cg3_1.coeff, getCGcoeff(cg3_1), atol=t)
    @test isapprox(cg3_1.coeff, getCGcoeff(j1, m1, j2, m2, j3), atol=t)

    @test isapprox(cg4_1.coeff, last(refVals), atol=t)
    @test isapprox(cg4_1.coeff, getCGcoeff(cg4_1), atol=t)
    @test isapprox(cg4_1.coeff, getCGcoeff(j1, m1-1, j2, m2-1, j3), atol=t)

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
    @test isapprox(cg1_1.coeff, cg1_2.coeff, atol=t)
    @test isapprox(cg1_1.coeff, cg1_3.coeff, atol=t)
    @test isapprox(cg1_1.coeff, cg1_4.coeff, atol=t)
    @test isapprox(cg1_1.coeff, cg1_5.coeff, atol=t)

    @test cg2_1.m1 == cg2_2.m1 == cg2_3.m1 == cg2_4.m1 == cg2_5.m1
    @test cg2_1.m2 == cg2_2.m2 == cg2_3.m2 == cg2_4.m2 == cg2_5.m2
    @test isapprox(cg2_1.coeff, cg2_2.coeff, atol=t)
    @test isapprox(cg2_1.coeff, cg2_3.coeff, atol=t)
    @test isapprox(cg2_1.coeff, cg2_4.coeff, atol=t)
    @test isapprox(cg2_1.coeff, cg2_5.coeff, atol=t)
end

j1m1j2m2j3_1 = (2, 2, 2, -1, 3)
refVals1 = (0.5477225575051661, 0.5477225575051661)
testCG(j1m1j2m2j3_1..., refVals=refVals1)

j1m1j2m2j3_2 = (1.5, 0.5, 2, -1, 3.5)
refVals2 = (0.5855400437691199, 0.6546536707079772)
testCG(j1m1j2m2j3_2..., refVals=refVals2)

end