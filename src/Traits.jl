

abstract type TraitAction{I} <: TaggedOperator end

abstract type AnyInterface <: Any end

abstract type AnyTrait <: Any end

abstract type Trait{I<:AnyInterface} <: AnyTrait end

struct SelectTrait{I<:AnyInterface} <: TraitAction{I} end

abstract type StructureStyle <: AnyInterface end

struct ParamBoxAccess <: StructureStyle end

struct ContainParamBox <: Trait{ParamBoxAccess} end

struct WithoutParamBox <: Trait{ParamBoxAccess} end

function returnUndefinedTraitError(::Type{TR}, ::Type{T}) where {TR<:AnyTrait, T}
    throw(ArgumentError("The trait of `$T` for `$TR` has yet to been defined."))
end

(::SelectTrait{ParamBoxAccess})(::T) where {T} = 
returnUndefinedTraitError(ParamBoxAccess, T)


abstract type PropertyStyle <: AnyInterface end

abstract type Differentiability <: PropertyStyle end

struct CanDiff <: Differentiability end

struct NonDiff <: Differentiability end

(::SelectTrait{Differentiability})(::T) where {T} = 
returnUndefinedTraitError(Differentiability, T)


abstract type FunctionStyle <: AnyInterface end

abstract type ParameterStyle <: FunctionStyle end

struct HasParamBox <: ParameterStyle end

struct HasParamVal <: ParameterStyle end

struct NoParameter <: ParameterStyle end

(::SelectTrait{ParameterStyle})(::T) where {T} = 
returnUndefinedTraitError(ParameterStyle, T)


abstract type InputStyle <: FunctionStyle end

struct AnyInput <: InputStyle end

struct MagnitudeInput <: InputStyle end

struct TupleInput{N} <: InputStyle end

struct VectorInput <: InputStyle end

(::SelectTrait{InputStyle})(::F) where {F<:Function} = 
returnUndefinedTraitError(InputStyle, F)