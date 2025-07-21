using Test
using Quiqbox
using Quiqbox: MemoryPair, OccupationState

@testset "Particles.jl" begin

nucs = Memory{Symbol}([:H, :Li, :H])
nucCoords = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [-0.1, 0.1, 0.2]]
res1 = NuclearCluster(nucs, nucCoords)
ids = [3, 1, 2]
res2 = NuclearCluster(MemoryPair(nucs[ids], Tuple.(nucCoords[ids]) ))
@test all(res1.layout .== res2.layout)

os1 = OccupationState([1,2,3])
os2 = OccupationState((1,2,3))
c1, c2, c3 = os1
@test os1.layout == os2.layout == [c1, c2, c3]

end