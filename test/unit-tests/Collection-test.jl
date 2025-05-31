using Test
using Quiqbox
using Quiqbox: PackedMemory, indexedPerturb, OneToIndex, setIndex, rightCircShift, 
               MemoryLinker

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

tpl2 = (3, 2, 1.1)
@test setIndex(tpl2, 1, 3) == (3, 2, 1)
tpl3 = setIndex(tpl2, -0.1, 3, +)
@test (3, 2, 1.0) === tpl3 !== (3, 2, 1)

tpl4 = Tuple(1:4)
@test rightCircShift(tpl4) == (4, 1, 2, 3)
@test (rightCircShift∘rightCircShift)(tpl4) == (3, 4, 1, 2)

mat1 = rand(3,4)
ids1 = Memory{Int}([9, 6, 12, 2, 11, 10, 3, 5, 1, 4])
ids1_o = map(OneToIndex, ids1)
mat1_ml = MemoryLinker(mat1, ids1_o)
mat1_sub = mat1[ids1]
@test mat1_ml == mat1_sub
mat1_ml .+= 1
@test mat1_ml == (mat1_sub .+ 1)
ids1_o[1] = OneToIndex(1)
@test mat1_ml[1] == first(mat1) != mat1[begin+ids1[1]-1]

end