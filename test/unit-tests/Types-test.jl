using Test
using Quiqbox

@testset "Types.jl" begin

@test Quiqbox.PrimGTO <: Quiqbox.PrimitiveOrb

@test Quiqbox.EvalPrimOrb <: Quiqbox.ScaledOrbital

@test Quiqbox.PrimGTOcore <: Quiqbox.PrimitiveOrbCore

@test Quiqbox.EvalPrimGTO <: Quiqbox.EvalPrimOrb

@test Quiqbox.CompGTO <: Quiqbox.CompositeOrb

@test Quiqbox.WeightedPF <: Quiqbox.PairCombine

@test Quiqbox.EvalCompOrb <: Quiqbox.ScaledOrbital

@test Quiqbox.EvalCompGTO <: Quiqbox.EvalCompOrb

@test Quiqbox.FPrimGTO <: Quiqbox.ComponentOrb

@test Quiqbox.FCompGTO <: Quiqbox.ComponentOrb

@test Quiqbox.FixedSpanIndexSet <: Quiqbox.AbstractSpanIndexSet

@test Quiqbox.TypedSpanParamSet <: Quiqbox.AbstractSpanParamSet

end