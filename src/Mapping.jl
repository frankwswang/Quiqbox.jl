export ReturnTyped, PairCoupler

struct ReturnTyped{T, F<:Function} <: TypedEvaluator{T}
    f::F

    function ReturnTyped(f::F, ::Type{T}) where {F<:Function, T}
        new{T, F}(f)
    end
end

ReturnTyped(::Type{T}) where {T} = ReturnTyped(itself, T)

ReturnTyped(f::ReturnTyped{TO}, ::Type{TN}) where {TO, TN} = ReturnTyped(f.f, T)

(f::ReturnTyped{T, F})(arg...; kws...) where {T, F} = convert(T, f.f(arg...; kws...))

const Return{T} = ReturnTyped{T, ItsType}


struct StableBinary{T, F<:Function} <: TypedEvaluator{T}
    f::F

    function StableBinary(f::F, ::Type{T}) where {F<:Function, T}
        new{T, F}(f)
    end
end

(f::StableBinary{T, F})(argL::T, argR::T) where {T, F} = convert(T, f.f(argL, argR))

const StableAdd{T} = StableBinary{T, typeof(+)}
const StableMul{T} = StableBinary{T, typeof(*)}
const ElementalSub{T} = StableBinary{T, typeof(.-)}

StableAdd(::Type{T}) where {T} = StableBinary(+, T)
StableMul(::Type{T}) where {T} = StableBinary(*, T)
ElementalSub(::Type{T}) where {T} = StableBinary(.-, T)


struct Left end

struct Right end

const Lateral = Union{Left, Right}


struct LateralPartial{F<:Function, A<:NonEmptyTuple{Any}, L<:Lateral} <: Modifier
    f::F
    arg::A
    side::L

    function LateralPartial(f::F, args::NonEmptyTuple{Any}, side::L) where 
                           {F<:Function, L<:Lateral}
        new{F, typeof(args), L}(f, args, side)
    end
end

const LPartial{F<:Function, A<:NonEmptyTuple{Any}} = LateralPartial{F, A, Left }
const RPartial{F<:Function, A<:NonEmptyTuple{Any}} = LateralPartial{F, A, Right}

(f::LPartial)(arg...; kws...) = f.f(f.arg..., arg...; kws...)
(f::RPartial)(arg...; kws...) = f.f(arg..., f.arg...; kws...)

LPartial(f::Function, args::NonEmptyTuple{Any}) = LateralPartial(f, args, Left() )
RPartial(f::Function, args::NonEmptyTuple{Any}) = LateralPartial(f, args, Right())


const absSqrtInv = inv ∘ sqrt ∘ abs


function typedMap(op::F, obj::AbstractArray, ::Type{T}=Union{}) where {F<:Function, T}
    if isempty(obj)
        similar(obj, T)
    else
        map(op, obj)
    end
end

function typedMap(op::F, obj::Union{Tuple, NamedTuple}, ::Type=Union{}) where {F<:Function}
    map(op, obj)
end


struct InputLimiter{N, F<:Function} <: Modifier
    f::F

    function InputLimiter(f::F, ::Val{N}) where {F<:Function, N}
        checkPositivity(N::Int)
        new{numberOfInput, F}(f)
    end
end

InputLimiter(f::InputLimiter, ::Val{N}) where {N} = InputLimiter(f.f, Val(N))

(f::InputLimiter{N})(arg::Vararg{Any, N}) where {N} = f.f(arg...)


struct ChainMapper{F<:FunctionChainUnion{Function}} <: Mapper
    chain::F

    function ChainMapper(chain::F) where {F<:AbstractMemory{<:Function}}
        checkEmptiness(chain, :chain)
        new{F}(chain)
    end

    function ChainMapper(chain::F) where {F<:GeneralTupleUnion{ NonEmptyTuple{Function} }}
        new{F}(chain)
    end
end

ChainMapper(chain::AbstractArray{<:Function}) = ChainMapper(chain|>ShapedMemory)

function getField(obj, f::ChainMapper, finalizer::F=itself) where {F<:Function}
    typedMap(f.chain) do mapper
        mapper(obj) |> finalizer
    end
end


struct PairCoupler{J<:Function, FL<:Function, FR<:Function} <: CompositeFunction
    joint::J
    left::FL
    right::FR
end

function PairCoupler(joint::F) where {F<:Function}
    function buildPairCombine(left::Function, right::Function)
        PairCoupler(joint, left, right)
    end
end

(f::PairCoupler)(arg::Vararg) = f.joint(f.left(arg...), f.right(arg...))


struct ParamFreeFunc{F<:Function} <: CompositeFunction
    core::F

    function ParamFreeFunc(f::F) where {F<:Function}
        if !isParamBoxFree(f)
            throw(AssertionError("`f` should not contain any `$ParamBox`."))
        end
        new{F}(f)
    end
end

ParamFreeFunc(f::ParamFreeFunc) = itself(f)

(f::ParamFreeFunc)(args...) = f.core(args...)


struct Lucent end
struct Opaque end

struct Deref{F<:Function} <: Modifier
    f::F
end

(f::Deref)(arg::AbstractArray{<:Any, 0}) = f.f(arg[])
(f::Deref)(arg::Tuple{Any}) = f.f(arg|>first)
(f::Deref)(arg::NamedTuple{<:Any, <:Tuple{Any}}) = f.f(arg|>first)


struct TupleHeader{N, F<:Function} <: Modifier
    f::F

    function TupleHeader(f::F, ::Val{N}) where {F<:Function, N}
        checkPositivity(N::Int, true)
        new{N, F}(f)
    end
end

TupleHeader(f::TupleHeader, ::Val{N}) where {N} = TupleHeader(f.f, Val(N))

TupleHeader(::Val{N}) where {N} = TupleHeader(itself, Val(N))

(f::TupleHeader{N})(head, body...) where {N} = 
f.f(formatInput(TupleInput{Any, N}(), head), body...)

const TypedTupleFunc{T, D, F<:Function} = ReturnTyped{T, TupleHeader{D, F}}

TypedTupleFunc(f::Function, ::Type{T}, ::Val{D}) where {T, D} = 
ReturnTyped(TupleHeader(f, Val(D)), T)

TypedTupleFunc(f::ReturnTyped, ::Type{T}, ::Val{D}) where {T, D} = 
ReturnTyped(TupleHeader(f.f, Val(D)), T)


struct SelectHeader{N, K, F<:Function} <: Modifier
    f::F

    function SelectHeader{N, K}(f::F) where {N, K, F<:Function}
        checkPositivity(K, true)
        N < K && throw(AssertionError("N must be no less than K=$K."))
        new{N::Int, K::Int, F}(f)
    end
end

(f::SelectHeader{N, K})(arg::Vararg{Any, N}) where {N, K} = f.f(arg[begin+K-1])

(f::SelectHeader{N, 0})(::Vararg{Any, N}) where {N} = f.f()



"""

    getOpacity(f::Function) -> Union{Lucent, Opaque}

If a return value is a `Lucent`, that means `f` either does not contain any `ParamBox`, or 
has a specialized `unpackFunc` method that separates its embedded `ParamBox`.
"""
function getOpacity(f::Function)
    isParamBoxFree(f) ? Lucent() : Opaque()
end

getOpacity(::ParamFreeFunc) = Lucent()


struct GetRange <: Mapper
    start::Int
    final::Int
    step::Int

    function GetRange(start::Int, final::Int, step::Int=1)
        checkPositivity(start)
        checkPositivity(final)
        checkPositivity(step|>abs)
        new(start, final, step)
    end
end

function (f::GetRange)(obj)
    map(obj, f.start:f.step:f.final) do i
        getField(obj, OneToIndex(i))
    end
end


struct FloatingMonomial{T<:Real, D} <: CompositeFunction
    center::NTuple{D, T}
    degree::WeakComp{D}

    FloatingMonomial(center::NonEmptyTuple{T}, degree::WeakComp{D}) where {T, D} = 
    new{T, D}(center, degree)
end

FloatingMonomial(center::NonEmptyTuple{T, D}, degree::NonEmptyTuple{Int, D}) where {T, D} = 
FloatingMonomial(center, WeakComp(degree))

function (f::FloatingMonomial{T, D})(coord::Union{NTuple{D, T}, AbstractVector{T}}
                                     ) where {T<:Real, D}
    mapreduce(StableMul(T), enumerate(f.center), f.degree.tuple) do (i, cen), pwr
        (coord[begin+i-1] - cen)^pwr
    end
end