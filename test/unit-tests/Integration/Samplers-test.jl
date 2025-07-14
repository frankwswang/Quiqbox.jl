using Test
using Quiqbox: genCoulombPointFieldSampler

@testset "Samplers.jl" begin
    point = (1.0, 2.0, 3.0)
    charges = (2.0, -1.0)
    op1 = genCoulombPointFieldSampler(charges, point)
    getNorm = (coord::NTuple{3, Float64}) -> mapreduce(x->x*x, +, coord) |> sqrt
    f1 = op1(getNorm, getNorm)
    coord = (1.1, -0.4, 2.0)
    @test f1(coord) == getNorm(coord)^2 / getNorm(coord .- point) * prod(charges)
end