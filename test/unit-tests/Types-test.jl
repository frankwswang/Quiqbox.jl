using Test
using Quiqbox

@testset "Types.jl" begin

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

@test Quiqbox.FixedSpanIndexSet <: Quiqbox.AbstractSpanIndexSet

@test Quiqbox.TypedSpanParamSet <: Quiqbox.AbstractSpanParamSet

@test Quiqbox.PrimGTOData <: Quiqbox.PrimOrbData

@test Quiqbox.ParamMapper <: Quiqbox.ChainMapper

end