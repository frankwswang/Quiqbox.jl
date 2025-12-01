using Test
using Quiqbox
using Quiqbox: OneToIndex, ChainedAccess, markObj, MemoryPair, AtomicUnit, AtomicGrid, 
               Identifier, EncodedDict, MemorySplitter

@testset "Query.jl" begin

@test OneToIndex() == OneToIndex(2) - 1
idxBox = [i for i in OneToIndex(2)]
@test idxBox == [OneToIndex(2)]
@test eltype(idxBox) == OneToIndex
@test OneToIndex(OneToIndex(), Count(2)) .+ [1, -1] == OneToIndex.([4, 2])
@test OneToIndex() < OneToIndex(2)
@test OneToIndex(2) > OneToIndex()
@test OneToIndex() <= OneToIndex()
@test OneToIndex() >= OneToIndex()
@test OneToIndex() == OneToIndex() === OneToIndex(1)

indexer = ChainedAccess(( OneToIndex(2), OneToIndex(1), OneToIndex(4) ))
source = [[1], [[2, 3, 4, 5], 6]]
@test Quiqbox.getEntry(source, indexer) == source[2][1][4]

m1 = rand(3, 3)
m1m = Quiqbox.ShapedMemory(m1)
m1_mrk = markObj(m1)
m1m_mrk = markObj(m1m)
@test m1_mrk isa Quiqbox.ValueMarker
@test m1m_mrk isa Quiqbox.ValueMarker
@test m1_mrk == m1m_mrk

v1 = rand(5)
v1_1 = Quiqbox.LinearMemory(v1)
v1_2 = Quiqbox.ShapedMemory(v1)
v1_2_mrk = markObj(v1_2)
@test v1_2_mrk isa Quiqbox.ValueMarker
@test markObj(v1) == markObj(v1_1) == v1_2_mrk

ks1 = (:a, :b, :c)
vs1 = (1, 2.0, 3.0)
d1_1 = Dict(ks1 .=> vs1)
d1_2 = Dict(ks1 .=> collect(vs1))
@test markObj(d1_1) == markObj(d1_2)

d3_1 = Quiqbox.EmptyDict{Symbol, Float64}()
d3_2 = Quiqbox.Dict{Symbol, Float64}()
@test d3_1 == d3_1
@test d3_1 == d3_2
@test markObj(d3_1) == markObj(d3_2)

a = rand(100)
b = rand(101)
c = rand(100)
try MemoryPair(a, b) catch; true end
mp1 = MemoryPair(a, c)
bl = true
for ((i, j), pair) in zip(zip(a, c), mp1)
    bl *= ((i => j) === pair)
end
@test bl
mp2 = MemoryPair(collect(1:3), collect(2:2:6))
@test mp2[2] == mp2[OneToIndex(2)] == (2 => 4)

u1 = AtomicUnit(1)
u2 = AtomicUnit(1)
@test u1 == u2 && u1 !== u2
@test Identifier(u1) != Identifier(u2)
@test markObj(u1) == markObj(u2)

g1 = AtomicGrid( Quiqbox.genMemory([1])   )
g2 = AtomicGrid( Quiqbox.genMemory([1.0]) )
@test g1 == g2 && g1 !== g2
@test Identifier(g1) != Identifier(g2)
@test markObj(g1) == markObj(g2)

function edEncoder(a::Int)
    iseven(a) && a > 0
end
ed1 = EncodedDict{Bool, Float64, Int}(edEncoder)
pair = Quiqbox.encodeGet(ed1, 1, nothing, true)
@test haskey(ed1, pair.first) == !(pair.second === nothing)
@test pair == (edEncoder(1) => nothing)
@test length(ed1) == 0
@test let v=collect(ed1); v == [] && eltype(v) == Pair{Bool, Float64} end
@test get(ed1, pair.first, nothing) === nothing
get!(ed1, pair.first, 1.1)
@test ed1[false] == 1.1
@test length(ed1) == 1
@test collect(ed1) == [false=>1.1]
setindex!(ed1, 1.0, false)
setindex!(ed1, 2.0, true)
@test collect(ed1) == [false=>1.0, true=>2.0]
@test all(ele==Pair(Bool(i-1), i) for (ele, i) in zip( ed1, 1:length(ed1) ))

v2 = Memory{Bool}([true, true, false, true, false])
ms1 = MemorySplitter(MemoryPair(rand(Int, 5), rand(Float64, 5)), v2)
ms1.sector.left .= 1:5
ms1.sector.right .= 0.0:1.5:6.0
v2Ref = Real[1, 2, 3.0, 4, 6.0]
@test all(i === j for (i, j) in zip(v2Ref, ms1))
@test all(i==0 for i in sum(ms1 .- v2Ref))
@test length(ms1) == 5
@test [ms1[i] for i in eachindex(ms1)] == v2Ref
@test eltype(ms1) == Union{Int, Float64}
@test firstindex(ms1) == 1
@test lastindex(ms1) == 5
ms1[end] = 5
@test all(ms1 .== 1:5)

end