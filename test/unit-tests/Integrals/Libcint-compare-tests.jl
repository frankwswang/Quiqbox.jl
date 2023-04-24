using Test
using Quiqbox
using Quiqbox: BasisFuncTexts, ElementNames, BasisSetNames

include("../../../test/test-functions/Shared.jl")
include("../../../test/test-functions/Libcint/Libcint.jl")

@testset "Quiqbox and Libcint Gaussian integral tests" begin

atms = map(1:8) do i
    list = findall(x->x!==nothing, BasisFuncTexts[i])
    ElementNames[list[rand(1:end)]]
end
nuc = ["H", "H"]
nucCoords = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]
center = [0, rand(0:0.2:1), 0]
errT1 = 5e-13

oFilter(f) = bs -> f(filter(i->lOf(i)<2, bs))

fs1 = (oFilter(overlaps), 
       oFilter(eKinetics), 
       oFilter(x->neAttractions(x, nuc, nucCoords)), 
       oFilter(eeInteractions))
fs2 = (oFilter(overlapsLibcint), 
       oFilter(elecKineticsLibcint), 
       oFilter(x->nucAttractionsLibcint(x, nuc, nucCoords)), 
       oFilter(eeInteractionsLibcint))
fns = (:overlaps, :eKinetics, :neAttractions, :eeInteractions)

for (f1, f2, fn) in zip(fs1, fs2, fns), pair in zip(BasisSetNames, atms)
    bs = genBasisFunc(center, pair...)
    if (orbitalNumOf.(bs) |> sum) < 26
        bs2 = hcat(decompose.(bs)...) |> vec
        # Test result consistency of libcint functions for BasisFuncs
        res = compr2Arrays1(f2(bs), f2(bs2), 0.002errT1)
        !res && println("The above errors are from Libcint functions for: ", (fn, pair...))
        @test res
    end
    bl = compr2Arrays1(f1(bs), f2(bs), errT1)
    !bl && println("Failed Case for errors between Quiqbox and Libcint: ", (fn, pair...))
    @test bl
end


# GTO coefficient pair (α, d) reusing function tests
bfSource = genBasisFunc([-0.7, 0.0, 0.0], "cc-pVTZ")[2]
errT2 = 5e-15

gf1 = GaussFunc(2.2, 0.3)
bf1 = genBasisFunc([-0.7, 0.0, 0.0], (bfSource.gauss..., gf1), normalizeGTO=true)
gf2 = GaussFunc(2.5, -0.25)
bf2 = genBasisFunc([-0.7, 0.0, 0.0], (bfSource.gauss..., gf2), normalizeGTO=true)
bf3 = genBasisFunc([-0.7, 0.0, 0.0], (gf1, bfSource.gauss[1:end-1]..., gf2), 
                   normalizeGTO=true)
gf3 = GaussFunc(2.5, -0.15)
bf4 = genBasisFunc([-0.7, 0.0, 0.0], (bfSource.gauss[2:end]..., gf3), normalizeGTO=true)

## getIntX1X1X2X2!
res1_1 = eeInteraction(bf1, bf1, bf2, bf2)
res1_2 = eeInteractionLibcint(bf1, bf1, bf2, bf2)[]
@test isapprox(res1_1, res1_2, atol=errT2)

## getIntX1X2X1X2!
res2_1 = eeInteraction(bf1, bf2, bf1, bf2)
res2_2 = eeInteractionLibcint(bf1, bf2, bf1, bf2)[]
@test isapprox(res2_1, res2_2, atol=errT2)

## getIntX1X2X2X1!
res3_1 = eeInteraction(bf1, bf2, bf2, bf1)
res3_2 = eeInteractionLibcint(bf1, bf2, bf2, bf1)[]
@test isapprox(res3_1, res3_2, atol=errT2)

## getIntX1X1X2X3!
res4_1 = eeInteraction(bf1, bf1, bf2, bf3)
res4_2 = eeInteractionLibcint(bf1, bf1, bf2, bf3)[]
@test isapprox(res4_1, res4_2, atol=errT2)

## getIntX1X2X3X3!
res5_1 = eeInteraction(bf1, bf2, bf3, bf3)
res5_2 = eeInteractionLibcint(bf1, bf2, bf3, bf3)[]
@test isapprox(res5_1, res5_2, atol=errT2)

## getIntX1X2X3X1!
res6_1 = eeInteraction(bf1, bf2, bf3, bf1)
res6_2 = eeInteractionLibcint(bf1, bf2, bf3, bf1)[]
@test isapprox(res6_1, res6_2, atol=errT2)

## getIntX1X2X2X3!
res7_1 = eeInteraction(bf1, bf2, bf2, bf3)
res7_2 = eeInteractionLibcint(bf1, bf2, bf2, bf3)[]
@test isapprox(res7_1, res7_2, atol=errT2)

## getIntX1X2X3X4!
res8_1 = eeInteraction(bf1, bf2, bf3, bf4)
res8_2 = eeInteractionLibcint(bf1, bf2, bf3, bf4)[]
@test isapprox(res8_1, res8_2, atol=errT2)

end