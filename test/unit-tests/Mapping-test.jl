using Test
using Quiqbox
using Quiqbox: getOutputType, PairCoupler

@testset "Mapping.jl" begin

returnF64 = TypedReturn(identity, Float64)

@test getOutputType(returnF64) == Float64

stableAdd1 = Quiqbox.StableBinary(+, Float64)

stableAdd2 = Quiqbox.StableBinary(stableAdd1, Float64)

@test stableAdd1 === stableAdd2

addF64 = Quiqbox.PairCoupler(+, returnF64, returnF64)

@test getOutputType(addF64) == Float64

end