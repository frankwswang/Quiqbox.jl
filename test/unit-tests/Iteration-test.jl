using Test
using Quiqbox: rightCircShift

@testset "Iteration.jl" begin

tpl1 = Tuple(1:4)
@test rightCircShift(tpl1) == (4, 1, 2, 3)
@test (rightCircShift∘rightCircShift)(tpl1) == (3, 4, 1, 2)

end