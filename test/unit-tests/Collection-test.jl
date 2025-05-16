using Test
using Quiqbox
using Quiqbox: PackedMemory

@testset "Collection.jl" begin

vecMem1 = VectorMemory(rand(3))
@test zero(vecMem1) == zeros(3)

shpMem1 = ShapedMemory(rand(3, 3))
@test zero(shpMem1) == zeros(3, 3)

pckMem1 = PackedMemory(rand(3, 3))
@test zero(shpMem1) == zeros(3, 3)
@test PackedMemory(shpMem1|>zero) == zeros(3, 3)

end