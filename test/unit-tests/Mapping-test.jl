using Test
using Quiqbox
using Quiqbox: trySimplify, getOutputType, TypedBinary, StableBinary, PairCoupler, Storage, 
               BinaryReduce, Contract, TupleSplitHeader

@testset "Mapping.jl" begin

# `trySimplify`
sf1 = trySimplify(TypedReturn(abs, Float64))
sf2 = trySimplify(abs)
@test sf1 === sf2 === abs

# `TypedReturn`
returnF64 = TypedReturn(identity, Float64)
@test getOutputType(returnF64) == Float64

# `TypedBinary`
tb1 = TypedBinary(TypedReturn((x, y)->x+y, ComplexF64), Int, Float64)
@test try tb1(1, 2) catch e; e isa MethodError end
@test tb1(1, 2.0) === 3.0 + 0.0im
stableAdd1 = StableBinary(+, Float64)
stableAdd2 = StableBinary(stableAdd1, Float64)
@test stableAdd1 === stableAdd2

# `PairCoupler`
addF64 = PairCoupler(+, returnF64, returnF64)
@test getOutputType(addF64) == Float64

# `Storage`
mat = rand(3, 3)
sf = Storage(mat, :mat)
@test sf() === sf(addF64) === mat
@test getOutputType(sf) == Matrix{Float64}
@test Quiqbox.markObj(Quiqbox.Storage([1])) == 
      Quiqbox.markObj(Quiqbox.Storage([1]|>Quiqbox.genMemory))

# `BinaryReduce`
br1 = BinaryReduce(tb1, *)
br2 = Contract(Real, Real, Real)
v1 = [1, 3, 4, 2, 5]
v2 = [-2, -1.1, 4.0, 0.0, -2.3]
@test br1(v1, v2) === ComplexF64(prod(v1 .+ v2))
@test br2(v1, v2) === sum(v1 .* v2)
@test getOutputType(br1) == ComplexF64
@test getOutputType(br2) == Real

# `TupleSplitHeader`
tupleAdd = TupleSplitHeader{2}(+)
@test tupleAdd((2, 3)) == 5

end