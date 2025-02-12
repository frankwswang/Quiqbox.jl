using Test
using Quiqbox

@testset "Types.jl" begin

@test Quiqbox.FlatParamVec <: Quiqbox.MiscParamVec

@test !(Quiqbox.MiscParamSet <: Quiqbox.MiscParamVec)

@test !(Quiqbox.FlatParamSet <: Quiqbox.MiscParamVec)

@test !(Quiqbox.FlatParamSet <: Quiqbox.FlatParamVec)

@test Quiqbox.TypedFlatParamSet <: Quiqbox.FlatParamVec

@test Quiqbox.TypedFlatParamSet <: Quiqbox.FlatParamSet

@test Quiqbox.PrimitiveParamSet <: Quiqbox.TypedFlatParamSet

@test Quiqbox.TypedMiscParamSet <: Quiqbox.MiscParamVec

@test Quiqbox.TypedMiscParamSet <: Quiqbox.MiscParamSet

@test Quiqbox.FlatParamSet <: Quiqbox.AbstractFlatParamSet

@test Quiqbox.FlatParamSetMixedVec <: Quiqbox.AbstractMiscParamSet

@test Quiqbox.MiscParamSet <: Quiqbox.AbstractMiscParamSet

@test Quiqbox.PrimGTO <: Quiqbox.PrimitiveOrb

@test Quiqbox.EvalPrimOrb <: Quiqbox.ScaledOrbital

@test Quiqbox.PrimGTOcore <: Quiqbox.PrimitiveOrbCore

@test Quiqbox.EvalPrimGTO <: Quiqbox.EvalPrimOrb

@test Quiqbox.CompGTO <: Quiqbox.CompositeOrb

@test Quiqbox.WeightedPF <: Quiqbox.PairCombine

@test Quiqbox.EvalCompOrb <: Quiqbox.ScaledOrbital

@test Quiqbox.EvalCompGTO <: Quiqbox.EvalCompOrb

@test Quiqbox.FPrimGTO <: Quiqbox.FrameworkOrb

@test Quiqbox.FCompGTO <: Quiqbox.FrameworkOrb

@test eltype( Quiqbox.OrbCoreDataSeq{Int, 2, Quiqbox.PrimitiveOrbCore{Int, 2}, 
              Vector{Quiqbox.ShapedMemory{Int, 1}}} ) <: Quiqbox.OrbCoreData

@test Quiqbox.FilteredFlatParamSet <: Quiqbox.Filtered

@test Quiqbox.TypedParamInput <: Quiqbox.Filtered

end