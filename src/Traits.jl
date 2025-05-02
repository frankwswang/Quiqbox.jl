abstract type TraitAction{I} <: FieldlessFunction end

abstract type AnyInterface <: Any end

abstract type AnyTrait <: Any end

abstract type Trait{I<:AnyInterface} <: AnyTrait end

struct SelectTrait{I<:AnyInterface} <: TraitAction{I} end

function returnUndefinedTraitError(::Type{TR}, ::Type{T}) where {TR<:AnyInterface, T}
    throw(ArgumentError("The trait of `$T` for `$TR` has yet to been defined."))
end


abstract type FunctionStyle <: AnyInterface end

abstract type Differentiability <: FunctionStyle end

struct CanDiff <: Differentiability end

struct NonDiff <: Differentiability end

(::SelectTrait{Differentiability})(::T) where {T} = 
returnUndefinedTraitError(Differentiability, T)


abstract type InputStyle <: FunctionStyle end

struct AnyInput <: InputStyle end

struct TupleInput{T, N} <: InputStyle end

(::SelectTrait{InputStyle})(::F) where {F<:Function} = AnyInput()

struct CoordInput{N} <: InputStyle end

formatInput(::AnyInput, x::Any) = itself(x)

function formatInput(::TupleInput{T, 1}, x::Any) where {T}
    (x isa T) || throw(ArgumentError("`x` must be a `$T`."))
    (x,)
end

function formatInput(::TupleInput{T, 1}, x::Tuple) where {T}
    (x isa Tuple{T}) || throw(ArgumentError("`x` must be a `Tuple{$T}`."))
    x
end

function formatInput(::TupleInput{T, N}, x::Tuple) where {T, N}
    (x isa NTuple{N, T}) || throw(ArgumentError("`x` must be a `NTuple{$N, $T}`."))
    x
end

function formatInput(::CoordInput{N}, x::NTuple{N, Number}) where {N}
    itself(x)
end

function formatInput(::CoordInput{N}, x::AbstractVector{<:Number}) where {N}
    ntuple(i->x[begin+i-1], N)
end

formatInput(f::Function, x) = formatInput(SelectTrait{InputStyle}()(f), x)


abstract type IntegralStyle <: AnyInterface end

abstract type MultiBodyIntegral{D} <: IntegralStyle end

struct OneBodyIntegral{D} <: MultiBodyIntegral{D} end

struct TwoBodyIntegral{D} <: MultiBodyIntegral{D} end