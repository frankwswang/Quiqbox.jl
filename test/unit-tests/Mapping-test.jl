using Test
using Quiqbox
using Quiqbox: getOutputType, TypedBinary, StableBinary, PairCoupler, Storage, 
               BinaryReduce, Contract

@testset "Mapping.jl" begin

# function trySimplify
sf1 = trySimplify(TypedReturn(abs, Float64))
sf2 = trySimplify(abs)
@test sf1 === sf2 === abs

# function TypedReturn
returnF64 = TypedReturn(identity, Float64)
@test getOutputType(returnF64) == Float64

# function TypedBinary
tb1 = TypedBinary(TypedReturn((x, y)->x+y, ComplexF64), Int, Float64)
@test try tb1(1, 2) catch e; e isa MethodError end
@test tb1(1, 2.0) === 3.0 + 0.0im
stableAdd1 = StableBinary(+, Float64)
stableAdd2 = StableBinary(stableAdd1, Float64)
@test stableAdd1 === stableAdd2

# function PairCoupler
addF64 = PairCoupler(+, returnF64, returnF64)
@test getOutputType(addF64) == Float64

# function Storage
mat = rand(3, 3)
sf = Storage(mat)
@test sf() === sf(addF64) === mat
@test getOutputType(sf) == Matrix{Float64}

# function BinaryReduce
br1 = BinaryReduce(tb1, *)
br2 = Contract(Real, Real, Real)
v1 = [1, 3, 4, 2, 5]
v2 = [-2, -1.1, 4.0, 0.0, -2.3]
@test br1(v1, v2) === ComplexF64(prod(v1 .+ v2))
@test br2(v1, v2) === sum(v1 .* v2)
@test getOutputType(br1) == ComplexF64
@test getOutputType(br2) == Real

end