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
errT = 1e-12

oFilter(f) = bs-> f(filter(i->lOf(i)<2, bs))

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
    if (orbitalNumOf.(bs) |> sum) < 12
        bs2 = hcat(decompose.(bs)...) |> vec
        # Test result consistency of libcint functions for BasisFuncs
        res = compr2Arrays1(f2(bs), f2(bs2), 0.001errT)
        !res && println("The above errors are from Libcint functions for: ", (fn, pair...))
        @test res
    end
    bl = compr2Arrays1(f1(bs), f2(bs), errT)
    !bl && println("Failed Case for errors between Quiqbox and Libcint: ", (fn, pair...))
    @test bl
end

end