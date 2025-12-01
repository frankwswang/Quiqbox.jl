using Test
using Quiqbox
using Quiqbox: MultiOrbitalData, PrimOrbPointer, CompOrbPointer, OneToIndex, MemoryPair, 
               markObj, buildOrbCoreWeight!, absSqrtInv

@testset "Framework.jl" begin

pgto1 = genGaussTypeOrb((0.0, 0.0, 1.0), 2.0, (1, 1, 0), renormalize=true )
pgto2 = genGaussTypeOrb((0.0, 1.0, 0.0), 3.0, (0, 0, 0), renormalize=true )
pgto3 = genGaussTypeOrb((1.0, 0.0, 1.0), 4.0, (1, 1, 2), renormalize=true )
pgto4 = genGaussTypeOrb((1.0, 0.0, 1.0), 4.0, (1, 1, 2), renormalize=false)

 cons = [2.0, -1.0, 3.0]
cgto1 = CompositeOrb([pgto1, pgto2, pgto3], cons, renormalize=false)
cgto2 = CompositeOrb([pgto1, pgto2, pgto3], cons, renormalize=true )
cgto3 = CompositeOrb([pgto2, pgto1, pgto4], cons, renormalize=false)
cgto4 = CompositeOrb([pgto2, pgto1, pgto4], cons, renormalize=true )
cgto5 = CompositeOrb([pgto4], [1.0], renormalize=true)

bs = Quiqbox.genMemory([pgto1, pgto4, cgto1, cgto2, cgto3, cgto4, cgto5])
data = MultiOrbitalData(bs)

@test length(data.config) == 3
ptr1 = Quiqbox.PrimOrbPointer(Count(3), Float64, OneToIndex(1), true)
ptr2 = Quiqbox.PrimOrbPointer(Count(3), Float64, OneToIndex(3), true)
ptr3 = Quiqbox.PrimOrbPointer(Count(3), Float64, OneToIndex(2), true)
ptr4 = Quiqbox.PrimOrbPointer(Count(3), Float64, OneToIndex(2))
ptr5 = Quiqbox.CompOrbPointer(MemoryPair([ptr1, ptr2, ptr3], cons))
ptr6 = Quiqbox.CompOrbPointer(ptr5.inner, true)
ptr7 = Quiqbox.CompOrbPointer(MemoryPair([ptr2, ptr1, ptr4], cons))
ptr8 = Quiqbox.CompOrbPointer(ptr7.inner, true)
ptr9 = Quiqbox.CompOrbPointer(MemoryPair([ptr4], [1.0]), true)
@test data.format[1] == ptr1
@test data.format[2] == ptr4
@test markObj(data.format[3]) == markObj(ptr5)
@test markObj(data.format[4]) == markObj(ptr6)
@test markObj(data.format[5]) == markObj(ptr7)
@test markObj(data.format[6]) == markObj(ptr8)
@test markObj(data.format[7]) == markObj(ptr9)

for bl in (Quiqbox.True(), Quiqbox.False())

    normInfo = Quiqbox.initializeOrbIntegral(Quiqbox.OneBodyIntegral{3, Float64}(), 
                                             Quiqbox.genOverlapSampler(), data, bl)
    @test absSqrtInv(overlap(pgto4, pgto4)) == buildOrbCoreWeight!(normInfo, ptr9)[] == 
                                               buildOrbCoreWeight!(normInfo, ptr3)[]

    ws1 = buildOrbCoreWeight!.(Ref(normInfo), [ptr1, ptr2, ptr3]) .* cons
    @test cons' * overlaps(cgto1.basis) * cons ≈ overlap(cgto1, cgto1) == 
           ws1' * overlaps(PrimitiveOrb.(cgto1.basis, renormalize=false)) * ws1
    @test ws1 == buildOrbCoreWeight!(normInfo, ptr5)
    @test ws1[end] == (absSqrtInv∘overlap)(pgto4, pgto4) * cons[end]

    ws2 = buildOrbCoreWeight!(normInfo, ptr6)
    @test overlap(cgto2, cgto2) ≈ 1 ≈ 
          ws2' * overlaps(PrimitiveOrb.(cgto2.basis, renormalize=false)) * ws2

    ws3 = buildOrbCoreWeight!(normInfo, ptr7)
    @test cons' * overlaps(cgto3.basis) * cons ≈ overlap(cgto3, cgto3) == 
           ws3' * overlaps(PrimitiveOrb.(cgto3.basis, renormalize=false)) * ws3

    ws4 = buildOrbCoreWeight!(normInfo, ptr8)
    @test overlap(cgto4, cgto4) ≈ 1 ≈ 
          ws4' * overlaps(PrimitiveOrb.(cgto4.basis, renormalize=false)) * ws4
end

end