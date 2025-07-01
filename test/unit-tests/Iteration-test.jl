using Test
using Quiqbox: rightCircShift, OneToIndex, OneToRange

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

end