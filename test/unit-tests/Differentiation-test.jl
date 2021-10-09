using Test
using Quiqbox

@testset "Differentiation.jl" begin

pb1 = ParamBox(1, :a)
@test typeof(pb1).parameters[1] == :a
@test pb1[] == pb1() == 1
@test pb1.map[] == Quiqbox.itself
@test pb1.canDiff[] == true
toggleDiff!(pb1)
@test pb1.canDiff[] == false

end