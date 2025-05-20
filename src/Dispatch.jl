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

struct CartesianInput{N} <: InputStyle end

formatInput(::AnyInput, x::Any) = itself(x)

formatInput(::CartesianInput{1}, x::Real) = (x,)

formatInput(::CartesianInput{N}, x::NTuple{N, Real}) where {N} = itself(x)

function formatInput(::CartesianInput{N}, x::AbstractVector{<:Real}) where {N}
    ntuple(i->x[begin+i-1], N)
end

formatInput(f::Function, x) = formatInput(SelectTrait{InputStyle}()(f), x)


abstract type IntegralStyle <: AnyInterface end

abstract type MultiBodyIntegral{D} <: IntegralStyle end

struct OneBodyIntegral{D} <: MultiBodyIntegral{D} end

struct TwoBodyIntegral{D} <: MultiBodyIntegral{D} end


strictTypeJoin(TL::Type, TR::Type) = typejoin(TL, TR)

strictTypeJoin(::Type{TL}, ::Type{TR}) where {TL<:RealOrComplex, TR<:RealOrComplex} = 
promote_type(TL, TR)