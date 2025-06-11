using Test
using Quiqbox

@testset "Types.jl" begin

@test Quiqbox.NestParam <: Quiqbox.ParamBox

@test Quiqbox.ReducibleParam <: Quiqbox.ParamBox

@test Quiqbox.ScreenParam <: Quiqbox.ReduceParam

@test Quiqbox.NestFixedParIn{Float64} <: Quiqbox.CoreFixedParIn{Float64}

@test Quiqbox.PrimGTO <: Quiqbox.PrimitiveOrb

@test Quiqbox.WrappedField <: Quiqbox.EncodedField

@test Quiqbox.EncodedFieldFunc <: Quiqbox.FieldParamFunc

@test Quiqbox.RadialField <: Quiqbox.EncodedField

@test Quiqbox.RadialFieldFunc <: Quiqbox.FieldParamFunc

@test Quiqbox.NullaryField <: Quiqbox.CurriedField

@test Quiqbox.GaussFunc <: Quiqbox.CurriedField

@test Quiqbox.GaussFieldFunc <: Quiqbox.FieldParamFunc

@test Quiqbox.AxialProduct{<:Quiqbox.RealOrComplex, 1} <: Quiqbox.ProductField

@test Quiqbox.CartAngMomFunc <: Quiqbox.NullaryField

@test Quiqbox.PolyRadialFunc <: Quiqbox.CoupledField

@test Quiqbox.PolyGaussFunc <: Quiqbox.PolyRadialFunc

@test Quiqbox.PolyRadialFieldFunc <: Quiqbox.FieldParamFunc

@test Quiqbox.PolyGaussFieldFunc <: Quiqbox.PolyRadialFieldFunc

@test Quiqbox.OptSpanValueSet <: Quiqbox.OptionalSpanSet

@test Quiqbox.TypedVoidSet <: Quiqbox.OptSpanValueSet

@test Quiqbox.OptSpanParamSet <: Quiqbox.OptionalSpanSet

@test Quiqbox.TypedSpanParamSet <: Quiqbox.OptSpanParamSet

@test Quiqbox.FixedSpanParamSet <: Quiqbox.TypedSpanParamSet

@test Quiqbox.PGTOrbData <: Quiqbox.PrimOrbData

@test Quiqbox.ParamMapper <: Quiqbox.ChainMapper

end