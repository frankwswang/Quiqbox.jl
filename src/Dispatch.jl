export Count

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

formatInput(::AnyInput, input::Any) = itself(input)

formatInput(::CartesianInput{1}, input::Real) = (input,)

formatInput(::CartesianInput{N}, input::NTuple{N, Real}) where {N} = itself(input)

function formatInput(::CartesianInput{N}, input::AbstractVector{<:Real}) where {N}
    ntuple(Val(N)) do i
        input[shiftLinearIndex(input, i)]
    end
end

formatInput(f::Function, x) = formatInput(SelectTrait{InputStyle}()(f), x)


abstract type IntegralStyle <: AnyInterface end

struct MultiBodyIntegral{D, C, N} <: IntegralStyle end
const OneBodyIntegral{D, C} = MultiBodyIntegral{D, C, 1}
const TwoBodyIntegral{D, C} = MultiBodyIntegral{D, C, 2}


strictTypeJoin(TL::Type, TR::Type) = typejoin(TL, TR)

strictTypeJoin(::Type{TL}, ::Type{TR}) where {TL<:RealOrComplex, TR<:RealOrComplex} = 
promote_type(TL, TR)


"""

    getOutputType(target::Function) -> Type

Infer the output type of `target`.
"""
getOutputType(::F) where {F<:Function} = getOutputType(F)

getOutputType(::Type{<:Function}) = Any

getOutputType(::Type{<:Base.ComposedFunction{F}}) where {F<:Function} = getOutputType(F)

getOutputType(::Type{Base.Splat{F}}) where {F<:Function} = getOutputType(F)

getOutputType(::Type{Union{}}) = 
throw(AssertionError("`Union{}` is not a valid input argument."))


Base.broadcastable(o::ValueType) = Ref(o)

struct True  <: ValueType end
struct False <: ValueType end
const Boolean = Union{True, False}

negate(::True) = False()
negate(::False) = True()


getTypeValue(::Type{True }) = true
getTypeValue(::Type{False}) = false

getTypeValue(::V) where {V<:ValueType} = getTypeValue(V)


function getMethodNum(f::Function)
    (length ∘ methods)(f)
end


struct Count{N} <: StructuredType

    function Count{N}() where {N}
        checkPositivity(N::Int, true)
        new{N}()
    end
end

Count(N::Integer) = Count{Int(N)}()

const Nil = Count{0}
const One = Count{1}