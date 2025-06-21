using Test
using Quiqbox
using Quiqbox: DirectMemory, PackedMemory

@testset "Types.jl" begin

@test Quiqbox.UnitParam <: Quiqbox.SpanParam
@test Quiqbox.GridParam <: Quiqbox.SpanParam

@test Quiqbox.SpanParam <: Quiqbox.ParamBox

@test   Quiqbox.NestParam <: Quiqbox.ParamBox
@test !(Quiqbox.SpanParam <: Quiqbox.NestParam)

@test   Quiqbox.GridParam{Int} <: Quiqbox.NestParam{Int, Int}
@test !(Quiqbox.GridParam{Int} <: Quiqbox.NestParam{Int, <:PackedMemory{Int}})

for E in (Int, DirectMemory{Int, 1}, PackedMemory{Int, DirectMemory{Int, 1}, 1})
    @test   Quiqbox.Span{E} <: Quiqbox.Pack{E}
    @test !(Quiqbox.Pack{E} <: Quiqbox.Span{E})
    @test   Quiqbox.ParamBox{Int, E} <: Quiqbox.ReducibleParam{Int, E}
    @test !(Quiqbox.ReducibleParam{Int, E} <: Quiqbox.ParamBox{Int, E})
    @test   Quiqbox.NestFixedParIn{Int, E} <: Quiqbox.CoreFixedParIn{Int, E}
    @test !(Quiqbox.CoreFixedParIn{Int, E} <: Quiqbox.NestFixedParIn{Int, E})
end

@test   Quiqbox.ScreenParam <: Quiqbox.ReduceParam
@test !(Quiqbox.ReduceParam <: Quiqbox.ScreenParam)

@test   Quiqbox.NestFixedParIn <: Quiqbox.CoreFixedParIn
@test !(Quiqbox.CoreFixedParIn <: Quiqbox.NestFixedParIn)

@test Quiqbox.PrimGTO <: Quiqbox.PrimitiveOrb

@test Quiqbox.WrappedField <: Quiqbox.EncodedField

@test Quiqbox.EncodedFieldFunc <: Quiqbox.FieldParamFunc

@test Quiqbox.RadialField <: Quiqbox.EncodedField

@test Quiqbox.RadialFieldFunc <: Quiqbox.FieldParamFunc

@test Quiqbox.NullaryField <: Quiqbox.ModularField

@test Quiqbox.GaussFunc <: Quiqbox.ModularField

@test Quiqbox.GaussFieldFunc <: Quiqbox.FieldParamFunc

@test Quiqbox.AxialProduct{<:Quiqbox.RealOrComplex, 1} <: Quiqbox.ProductField

@test Quiqbox.CartAngMomentum <: Quiqbox.NullaryField

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

@test Quiqbox.NamedParamMapper <: Quiqbox.ParamMapper

end