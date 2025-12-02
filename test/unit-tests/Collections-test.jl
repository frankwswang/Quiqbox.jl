using Test
using Quiqbox
using Quiqbox: PackedMemory, indexedPerturb, OneToIndex, setIndex, MemoryLinker, 
               getNestedLevel, decoupledCopy, extractMemory, genMemory

@testset "Collections.jl" begin

vecMem1 = LinearMemory(rand(3))
@test eltype(vecMem1) == Float64
@test ndims(vecMem1) == 1
@test firstindex(vecMem1) == firstindex(vecMem1, 1) == 1
@test lastindex(vecMem1) == lastindex(vecMem1, 1) == length(vecMem1)
@test size(vecMem1) == (length(vecMem1),) == size(vecMem1|>LinearIndices) == (3,)
@test axes(vecMem1) == (eachindex(vecMem1),) == (Base.OneTo(3),)
@test iterate(vecMem1) == (first(vecMem1), 2)
for (ele, m, n) in zip(vecMem1, LinearIndices(vecMem1), eachindex(vecMem1))
    @test m==n && vecMem1[m] == ele
end
@test zero(vecMem1) == zeros(3)
vecMem1c = copy(vecMem1)
@test vecMem1c == vecMem1 && typeof(vecMem1c) == typeof(vecMem1)
vecMem1c[1] += 1
@test vecMem1c != vecMem1
@test typeof(vecMem1c|>decoupledCopy) == typeof(vecMem1c)


shpMem1 = ShapedMemory(rand(2, 3, 1))
@test eltype(shpMem1) == Float64
@test ndims(shpMem1) == 3
foreach(((), 1, 2, 3)) do i
    @test firstindex(shpMem1, i...) == 1
end
foreach(((), 1, 2, 3), (length(shpMem1), 2, 3, 1)) do i, j
    @test lastindex(shpMem1, i...) == j
end
@test size(shpMem1) == size(shpMem1|>LinearIndices) == (2, 3, 1)
@test axes(shpMem1) == (Base.OneTo(2), Base.OneTo(3), Base.OneTo(1))
@test eachindex(shpMem1) == Base.OneTo(6)
@test iterate(shpMem1) == (first(shpMem1), 2)
for (ele, m, n) in zip(shpMem1, LinearIndices(shpMem1), eachindex(shpMem1))
    @test m==n && shpMem1[m] == ele
end
@test zero(shpMem1) == zeros(2, 3, 1)
shpMem2 = ShapedMemory([1 2])
shpMem3 = similar(shpMem2, (2,))
shpMem4 = similar(shpMem2, Float64, (2,))
@test size(shpMem3) == size(shpMem4) == (2,)
@test shpMem3 isa ShapedMemory{Int, 1}
@test shpMem4 isa ShapedMemory{Float64, 1}
shpMem3 .= 1
shpMem4 .= [-1.0, 2.0]
shpMem5 = shpMem3 + shpMem4
shpMem6 = shpMem3 - shpMem4
@test shpMem5 == [0.0, 3.0]
@test shpMem6 == [2.0,-1.0]
@test typeof(shpMem5) == typeof(shpMem6) == ShapedMemory{Float64, 1}
shpMem1c = copy(shpMem1)
@test shpMem1c == shpMem1 && typeof(shpMem1c) == typeof(shpMem1)
shpMem1c[1] += 1
@test shpMem1c != shpMem1
@test typeof(shpMem1c|>decoupledCopy) == typeof(shpMem1c)
shpMem7 = ShapedMemory(Float64, 1.0)
@test eltype(shpMem7) == Float64
@test length(shpMem7) == 1
@test size(shpMem7) == ()
shpMem7[] = 2
@test shpMem7[] === 2.0

pckMem1 = PackedMemory(shpMem1)
@test getNestedLevel(pckMem1|>typeof).level == 1
@test pckMem1 == shpMem1
@test pckMem1 !== shpMem1
pckMem1_2 = PackedMemory{Float64}(undef, size(pckMem1))
pckMem1_2 .= pckMem1
@test pckMem1_2 == pckMem1
@test typeof(pckMem1_2) == typeof(pckMem1)
@test getNestedLevel(pckMem1_2|>typeof).level == 1
try PackedMemory{Vector{Int}}(undef, (1,)) catch err; err isa AssertionError end
pckMem2 = PackedMemory(pckMem1)
@test getNestedLevel(pckMem2|>typeof).level == 1
@test pckMem2 == pckMem1
@test pckMem2 !== pckMem1
mat1 = rand(3, 2)
pckMem3 = PackedMemory(reshape( [pckMem1, mat1], (1, 2) ))
@test copy(pckMem3) isa Array
@test getNestedLevel(pckMem3|>typeof).level == 2
@test pckMem3[begin] === pckMem1  #> Only preserve `PackedMemory`
@test pckMem3[end] == mat1
@test pckMem3[end] !== mat1
pckMem4 = PackedMemory([mat1, mat1])
@test all(pckMem4 .== Ref(mat1))
@test !any(pckMem4 .=== Ref(mat1))
@test try PackedMemory([mat1, pckMem3]) catch; true end
pckMem1 isa Quiqbox.DirectMemory{Float64, ndims(shpMem1)}
pckMem1c = copy(pckMem1)
@test pckMem1c == pckMem1 && typeof(pckMem1c) == typeof(pckMem1)
pckMem1c[1] += 1
@test pckMem1c != pckMem1
@test typeof(pckMem1c|>decoupledCopy) == typeof(pckMem1c)

@test eltype(pckMem1) == eltype(pckMem2) == Float64
@test typeof(pckMem1) <: eltype(pckMem3)
@test pckMem4 isa PackedMemory{Float64, PackedMemory{Float64, Float64, 2}, 1}
@test ndims(pckMem3) == 2
foreach(((), 1, 2)) do i
    @test firstindex(pckMem3, i...) == 1
end
foreach(((), 1, 2), (length(pckMem3), 1, 2)) do i, j
    @test lastindex(pckMem3, i...) == j
end
@test size(pckMem3) == size(pckMem3|>LinearIndices) == (1, 2)
@test axes(pckMem3) == (Base.OneTo(1), Base.OneTo(2))
@test eachindex(pckMem3) == Base.OneTo(2)
@test iterate(pckMem3) == (first(pckMem3), 2)
for (ele, m, n) in zip(pckMem3, LinearIndices(pckMem3), eachindex(pckMem3))
    @test m==n && pckMem3[m] == ele
end
@test zero(pckMem1) == zero(shpMem1)
@test zero(pckMem3) == PackedMemory(pckMem3.value .|> zero)


tpl1 = (1.1, 3, 4)
indexedPerturb(+, tpl1, OneToIndex(2)=>0.3) == (1.1, 3.3, 4)
arr1 = [1.1, 3, 4]
indexedPerturb(÷, arr1, OneToIndex(3)=>2) == [1.1, 3, 2]


tpl2 = (3, 2, 1.1)
@test setIndex(tpl2, 1, 3) == (3, 2, 1)
tpl3 = setIndex(tpl2, -0.1, 3, +)
@test (3, 2, 1.0) === tpl3 !== (3, 2, 1)


mat2 = rand(3,4)
ids1 = Memory{Int}([9, 6, 12, 2, 11, 10, 3, 5, 1, 4])
ids1_o = map(OneToIndex, ids1)
mat2_ml = MemoryLinker(mat2, ids1_o)
mat2_sub = mat2[ids1]
@test mat2_ml == mat2_sub
mat2_ml .+= 1
@test mat2_ml == (mat2_sub .+ 1)
ids1_o[1] = OneToIndex(1)
@test mat2_ml[1] == first(mat2) != mat2[begin+ids1[1]-1]

@test eltype(mat2_ml) == Float64
@test ndims(mat2_ml) == 1
@test firstindex(mat2_ml) == firstindex(mat2_ml, 1) == 1
@test lastindex(mat2_ml) == lastindex(mat2_ml, 1) == length(mat2_ml)
@test size(mat2_ml) == (length(mat2_ml),) == size(mat2_ml|>LinearIndices) == (length(ids1),)
@test axes(mat2_ml) == (eachindex(mat2_ml),) == (Base.OneTo(10),)
@test iterate(mat2_ml) == (first(mat2_ml), 2)
for (ele, m, n) in zip(mat2_ml, LinearIndices(mat2_ml), eachindex(mat2_ml))
    @test m==n && mat2_ml[m] == ele
end

v1 = [1]
v2 = [v1 for _ in 1:3]
v2[1][] = 2
v2_2 = decoupledCopy(v2)
v2_2[2][] = 3
@test v2_2 == [[2], [3], [2]]
@test !any(v2_2 .=== Ref(v1))

mem1 = Memory{Real}([1, 2, 3.0])
@test extractMemory(mem1) === mem1
@test genMemory(mem1) !== mem1
@test all(genMemory(mem1) .== [1., 2., 3.])
@test typeof(genMemory(1)) == Memory{Int}
@test genMemory(1)[] === 1
mem2 = genMemory(true, 5)
@test mem2 == fill(true, 5)
@test genMemory(mem2) !== mem2
@test extractMemory(mem2) === mem2
@test extractMemory(mem2) === mem2
@test all(genMemory(itself, Float64, 10) .=== collect(1:1.0:10))
@test genMemory(x->x^2, Int, 10) == [x^2 for x in 1:10]

end