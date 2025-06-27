using Test
using Quiqbox
using Quiqbox: markObj

@testset "Layout.jl" begin

m1 = rand(3, 3)
m1m = Quiqbox.ShapedMemory(m1)

@test !(m1m.value === m1)
@test !(m1m.value == m1)
@test m1m == m1

m1_mrk = markObj(m1)
m1m_mrk = markObj(m1m)
@test m1_mrk isa Quiqbox.ValueMarker
@test m1m_mrk isa Quiqbox.ValueMarker
@test m1_mrk == m1m_mrk

v1 = rand(5)
v1_1 = Quiqbox.VectorMemory(v1)
v1_2 = Quiqbox.ShapedMemory(v1)

@test v1 == v1_1
@test v1 == v1_2
v1_2_mrk = markObj(v1_2)
@test v1_2_mrk isa Quiqbox.ValueMarker
@test markObj(v1) == markObj(v1_1) == v1_2_mrk

ks1 = (:a, :b, :c)
vs1 = (1, 2.0, 3.0)
d1_1 = Dict(ks1 .=> vs1)
d1_2 = Dict(ks1 .=> collect(vs1))
@test markObj(d1_1) == markObj(d1_2)

ks2 = (:d,)
vs2 = (4.0,)
d2_1 = Quiqbox.buildDict(ks2 .=> vs2)
d2_2 = Dict(ks2 .=> vs2)
@test d2_1 == d2_2
@test markObj(d2_1) == markObj(d2_2)

d2_3 = Quiqbox.buildDict(ks2 .=> Int.(vs2))
d2_4 = Dict(ks2 .=> Int.(vs2))
@test d2_1 == d2_1
@test d2_1 == d2_3 == d2_4
@test markObj(d2_1) == markObj(d2_3) == markObj(d2_4)

d3_1 = Quiqbox.TypedEmptyDict{Symbol, Float64}()
d3_2 = Quiqbox.Dict{Symbol, Float64}()
@test d3_1 == d3_1
@test d3_1 == d3_2
@test markObj(d3_1) == markObj(d3_2)

end