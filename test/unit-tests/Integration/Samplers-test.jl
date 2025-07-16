using Test
using Quiqbox
using Quiqbox: genCoulombMultiPointSampler, genCoulombInteractionSampler

@testset "Samplers.jl" begin
    points = [(1.0, 2.0, 3.0), (-1.0, 2.0, 2.0)]
    charges = [2.0, 1.0]
    op1 = genCoulombMultiPointSampler(charges, points)
    getNormCore = (coord::NTuple{3, Float64}) -> mapreduce(x->x*x, +, coord) |> sqrt
    getNorm = Quiqbox.TypedCarteFunc(getNormCore, Float64, Count(3))
    f1 = op1(getNorm, getNorm)
    coord = (1.1, -0.4, 2.0)
    f1_val = mapreduce(+, charges, points) do charge, point
        -getNorm(coord)^2 / getNorm(coord .- point) * charge
    end
    @test f1(coord) ≈ f1_val
    op2 = genCoulombInteractionSampler(Float64, Count(3))
    f2 = op2(getNorm, getNorm)
    @test f2(first(points), coord) ≈ 
          getNorm(first(points)) * (inv∘getNorm)(first(points) .- coord) * getNorm(coord)
end