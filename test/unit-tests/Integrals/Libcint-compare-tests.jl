using Test
using Quiqbox
using Quiqbox: BasisFuncTexts, ElementNames, BasisSetNames

include("../../../test/test-functions/Shared.jl")
include("../../../test/test-functions/Libcint/Libcint.jl")

@testset "Integral engine Quiqbox VS Libcint tests" begin

atms = map(1:8) do i
    list = findall(x->x!==nothing, BasisFuncTexts[i])
    ElementNames[list[rand(1:end)]]
end
nuc = ["H", "H"]
nucCoords = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]
center = [0, rand(0:0.2:1), 0]
errT = 1e-12

oFilter(f) = bs-> f(filter(i->typeof(i).parameters[1]<2, bs))

fs1 = (oFilter(overlaps), 
        oFilter(elecKinetics), 
        oFilter(x->nucAttractions(x, nuc, nucCoords)), 
        oFilter(eeInteractions))
fs2 = (oFilter(overlapsLibcint), 
        oFilter(elecKineticsLibcint), 
        oFilter(x->nucAttractionsLibcint(x, nuc, nucCoords)), 
        oFilter(eeInteractionsLibcint))
fns = (:overlaps, :elecKinetics, :nucAttractions, :eeInteractions)

for (f1, f2, fn) in zip(fs1, fs2, fns), pair in zip(BasisSetNames, atms)
    bs = genBasisFunc(center, pair)
    bl = compr2Arrays(f1(bs), f2(bs), errT)
    !bl && println("Failed Case:", (fn, pair...))
    @test bl
end

end