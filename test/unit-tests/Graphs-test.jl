using Test
using Quiqbox

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

end