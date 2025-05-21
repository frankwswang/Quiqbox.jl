using Test
using Quiqbox
using Quiqbox: PackedMemory, indexedPerturb, OneToIndex

@testset "Collection.jl" begin

vecMem1 = VectorMemory(rand(3))
@test zero(vecMem1) == zeros(3)

shpMem1 = ShapedMemory(rand(3, 3))
@test zero(shpMem1) == zeros(3, 3)

pckMem1 = PackedMemory(rand(3, 3))
@test zero(shpMem1) == zeros(3, 3)
@test PackedMemory(shpMem1|>zero) == zeros(3, 3)

tpl1 = (1.1, 3, 4)
indexedPerturb(+, tpl1, OneToIndex(2)=>0.3) == (1.1, 3.3, 4)
arr1 = [1.1, 3, 4]
indexedPerturb(÷, arr1, OneToIndex(3)=>2) == [1.1, 3, 2]

end