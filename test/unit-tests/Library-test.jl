using Test
using Quiqbox: checkBSList

@testset "Library.jl tests" begin

@test try checkBSList(); true catch; false end

end