using Test
using Quiqbox
using Quiqbox: inSymOfCore, outSymOfCore

@testset "Parameters.jl" begin

pb1 = ParamBox(1, :a)
@test inSymOfCore(pb1) == :a == outSymOfCore(pb1)
@test dataOf(pb1)[] == pb1[] == inValOf(pb1)
@test pb1[] == pb1() == 1 == outValOf(pb1)
@test pb1.map == Quiqbox.itself == mapOf(pb1)
@test pb1.canDiff[] == false == isDiffParam(pb1)
toggleDiff!(pb1)
@test pb1.canDiff[] == true
disableDiff!(pb1)
@test pb1.canDiff[] == false
enableDiff!(pb1)
@test pb1.canDiff[] == true
pb2 = inVarCopy(pb1)
@test dataOf(pb1) === dataOf(pb2) === pb1.data
pb2_2 = outValCopy(pb1)
@test dataOf(pb1) == dataOf(pb2_2)
@test dataOf(pb1) !== dataOf(pb2_2)

pb3 = ParamBox(1, :b, abs)
@test inSymOfCore(pb3) == :x_b
@test outSymOfCore(pb3) == :b
@test mapOf(pb3) == abs

pb4 = ParamBox(1.2, :c, x->x^2, :x)
@test inSymOfCore(pb4) == :x
@test outSymOfCore(pb4) == :c
@test startswith(nameof(pb4.map) |> string, "f_c")


pb5 = outValCopy(pb4)
@test pb5() == pb5[] == pb4()
@test pb5.map == Quiqbox.itself

pb6 = ParamBox(1.1, :p1, abs, :x)
pb7 = changeMapping(pb6, x->x^1.5)
@test pb7() == 1.1^1.5
@test pb7.data === pb7.data
@test pb7.dataName == :x_p1
@test typeof(pb7).parameters[2] == :p1

end