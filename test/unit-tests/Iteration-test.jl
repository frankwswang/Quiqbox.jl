using Test
using Quiqbox: rightCircShift, OneToIndex, OneToRange, shiftLinearIndex, shiftAxialIndex

@testset "Iteration.jl" begin

tpl1 = Tuple(1:4)
@test rightCircShift(tpl1) == (4, 1, 2, 3)
@test (rightCircShift∘rightCircShift)(tpl1) == (3, 4, 1, 2)

r1 = OneToRange(0)
@test length(r1) == 0
@test eltype(r1) == OneToIndex
i = OneToIndex()
for _ in r1
    i += 1
end
@test i == OneToIndex(1)
r2 = OneToRange(OneToIndex(3))
v = OneToIndex[]
for i in r2
    i isa OneToIndex && push!(v, i)
end
@test v == OneToIndex.(1:3)
@test length(r2) == 3

@test shiftLinearIndex(rand(3), OneToIndex(2)) == 2
@test shiftLinearIndex(tuple(1,2), OneToIndex(2)) == 2
@test shiftLinearIndex(rand(3), 1:2) == 1:2
@test shiftLinearIndex(tuple(1,2,4), 2:3) == 2:3


@test shiftAxialIndex(rand(3, 3), OneToIndex(2)) == (2, 2) == 
      ntuple(i->shiftAxialIndex(rand(3, 3), OneToIndex(2), i) , Val(2))
@test shiftAxialIndex(rand(3, 3), ( OneToIndex(), OneToIndex(2) )) == (1, 2)
@test shiftAxialIndex(rand(3), OneToIndex(2), 1) == 2
@test try shiftAxialIndex(fill(rand()), (OneToIndex(2),)) catch; true end

|end