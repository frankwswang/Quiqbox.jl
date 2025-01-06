abstract type TraitAction{I} <: FieldlessFunction end

abstract type AnyInterface <: Any end

abstract type AnyTrait <: Any end

abstract type Trait{I<:AnyInterface} <: AnyTrait end

struct SelectTrait{I<:AnyInterface} <: TraitAction{I} end

function returnUndefinedTraitError(::Type{TR}, ::Type{T}) where {TR<:AnyInterface, T}
    throw(ArgumentError("The trait of `$T` for `$TR` has yet to been defined."))
end


abstract type PropertyStyle <: AnyInterface end

abstract type Differentiability <: PropertyStyle end

struct CanDiff <: Differentiability end

struct NonDiff <: Differentiability end

(::SelectTrait{Differentiability})(::T) where {T} = 
returnUndefinedTraitError(Differentiability, T)


abstract type FunctionStyle <: AnyInterface end

abstract type ParameterizationStyle <: FunctionStyle end

struct TypedParamFunc{T} <: ParameterizationStyle end

struct GenericFunction <: ParameterizationStyle end

(::SelectTrait{ParameterizationStyle})(::T) where {T} = 
returnUndefinedTraitError(ParameterizationStyle, T)


abstract type InputStyle <: FunctionStyle end

struct AnyInput <: InputStyle end

struct TupleInput{N} <: InputStyle end

struct ScalarInput <: InputStyle end

struct VectorInput <: InputStyle end

(::SelectTrait{InputStyle})(::F) where {F<:Function} = 
SelectTrait{InputStyle}()(F)

(::SelectTrait{InputStyle})(::Type{F}) where {F<:Function} = 
returnUndefinedTraitError(InputStyle, F)

abstract type IntegralStyle <: AnyInterface end

abstract type MultiBodyIntegral <: IntegralStyle end

struct OneBodyIntegral <: MultiBodyIntegral end

struct TwoBodyIntegral <: MultiBodyIntegral end