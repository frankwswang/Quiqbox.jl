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

struct EuclideanInput{N} <: InputStyle end

formatInput(::AnyInput, x::Any) = itself(x)

formatInput(::EuclideanInput{1}, x::Real) = (x,)

formatInput(::EuclideanInput{N}, x::NTuple{N, Real}) where {N} = itself(x)

function formatInput(::EuclideanInput{N}, x::AbstractVector{<:Real}) where {N}
    ntuple(i->x[begin+i-1], N)
end

formatInput(f::Function, x) = formatInput(SelectTrait{InputStyle}()(f), x)


abstract type IntegralStyle <: AnyInterface end

abstract type MultiBodyIntegral{D} <: IntegralStyle end

struct OneBodyIntegral{D} <: MultiBodyIntegral{D} end

struct TwoBodyIntegral{D} <: MultiBodyIntegral{D} end


strictTypeJoin(TL::Type, TR::Type) = typejoin(TL, TR)

strictTypeJoin(::Type{T}, ::Type{Complex{T}}) where {T<:Real} = RealOrComplex{T}