using Test
using Quiqbox

@testset "Differentiation.jl" begin

pb1 = ParamBox(1, :a)
@test inSymOf(pb1) == :a == outSymOf(pb1)
@test dataOf(pb1)[] == pb1[] == inValOf(pb1)
@test pb1[] == pb1() == 1 == outValOf(pb1)
@test pb1.map == Quiqbox.itself == mapOf(pb1)
@test pb1.canDiff[] == true == isDiffParam(pb1)
toggleDiff!(pb1)
@test pb1.canDiff[] == false
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
@test inSymOf(pb3) == :x_b
@test outSymOf(pb3) == :b
@test mapOf(pb3) == abs

pb4 = ParamBox(1.2, :c, x->x^2, :x)
@test inSymOf(pb4) == :x
@test outSymOf(pb4) == :c
@test startswith(nameof(pb4.map) |> string, "f_c")


pb5 = outValCopy(pb4)
@test pb5() == pb5[] == pb4()
@test pb5.map == Quiqbox.itself
end