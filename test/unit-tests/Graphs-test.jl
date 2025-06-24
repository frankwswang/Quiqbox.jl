using Test
using Quiqbox
using Quiqbox: GridVertex, genVertexCaller

@testset "Graphs.jl" begin

v1 = genTensorVar([1, 2], :a, true)
f = ParamGraphCaller(v1)

@test f.source == Quiqbox.initializeFixedSpanSet()
@test typeof(f.source) == typeof(Quiqbox.initializeFixedSpanSet())

@test try f((unit=[1], grid=[[2]])) catch; true end
@test try f((unit=[1], grid=nothing)) catch; true end
@test try f((unit=nothing, grid=[[2]])) catch; true end

@test f(initializeSpanParamSet(nothing)) == v1()
@test f(( unit=nothing, grid=Quiqbox.genBottomMemory() )) == v1()

gv1 = GridVertex(Quiqbox.AtomicGrid([1, 2] |> ShapedMemory), true, :a)
gv2 = GridVertex(Quiqbox.AtomicGrid([1, 2] |> ShapedMemory), true, :a)
gv3 = GridVertex(Quiqbox.AtomicGrid([1, 2] |> ShapedMemory), true, :b)

@test gv1 != gv2
@test Quiqbox.markObj(gv1) == Quiqbox.markObj(gv2)
@test gv1 != gv3
@test Quiqbox.markObj(gv1) != Quiqbox.markObj(gv3)
gv1f = genVertexCaller(gv1)
gv2f = genVertexCaller(gv2)
gv3f = genVertexCaller(gv3)
@test Quiqbox.markObj(gv1f) == Quiqbox.markObj(gv2f) == Quiqbox.markObj(gv3f)

end