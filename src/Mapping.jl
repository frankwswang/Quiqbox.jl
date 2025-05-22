export TypedReturn, PairCoupler

getOutputType(::F) where {F<:Function} = getOutputType(F)

getOutputType(::Type{<:Function}) = Any


"""

    trySimplify(f::Function) -> Function

Returns a potentially simplified function that returns the same results (characterized by 
`$isequal`) as the original `f`, given any valid input arguments `args...` that do 
not throw errors: 

    isequal(trySimplify(f)(args...), f(args...)) == true
"""
trySimplify(f::Function) = f


struct ApplyConvert{T, F<:Function} <: TypedEvaluator{T}
    f::F

    function ApplyConvert(f::F, ::Type{T}) where {F<:Function, T}
        new{T, F}(f)
    end
end

ApplyConvert(::Type{T}) where {T} = ApplyConvert(itself, T)

ApplyConvert(f::ApplyConvert, ::Type{T}) where {T} = ApplyConvert(f.f, T)

(f::ApplyConvert{T, F})(arg::Vararg) where {T, F<:Function} = convert(T, f.f(arg...))

getOutputType(::Type{<:ApplyConvert{T}}) where {T} = T

trySimplify(f::ApplyConvert) = trySimplify(f.f)


struct TypedReturn{T, F<:Function} <: TypedEvaluator{T}
    f::F

    function TypedReturn(f::F, ::Type{T}) where {F<:Function, T}
        checkBottomType(T)
        new{T, F}(f)
    end
end

TypedReturn(f::TypedReturn{TO}, ::Type{TN}) where {TO, TN} = TypedReturn(f.f, TN)

function (f::TypedReturn{T, F})(arg::Vararg) where {T, F<:Function}
    caller = getLazyConverter(f.f, T)
    caller(arg...)
end

@generated function getLazyConverter(f::Function, ::Type{T}) where {T}
    if getOutputType(f) <: T
        return :(f)
    else
        return :(ApplyConvert(f, $T))
    end
end

getOutputType(::Type{<:TypedReturn{T}}) where {T} = T

trySimplify(f::TypedReturn) = trySimplify(f.f)


struct TypedBinary{T, F<:Function, TL, TR} <: TypedEvaluator{T}
    f::F

    function TypedBinary(f::TypedReturn{T, F}, ::Type{TL}, ::Type{TR}) where 
                        {F<:Function, T, TL, TR}
        checkBottomType(TL)
        checkBottomType(TR)
        new{T, F, TL, TR}(f.f)
    end
end

function (f::TypedBinary{T, F, TL, TR})(argL::TL, argR::TR) where {T, F<:Function, TL, TR}
    op = getLazyConverter(f.f, T)
    op(argL, argR)
end

const StableBinary{T, F<:Function} = TypedBinary{T, F, T, T}

StableBinary(f::Function, ::Type{T}) where {T} = TypedBinary(TypedReturn(f, T), T, T)

StableBinary(f::TypedBinary, ::Type{T}) where {T} = StableBinary(f.f, T)

getOutputType(::Type{<:TypedBinary{T}}) where {T} = T

trySimplify(f::TypedBinary) = trySimplify(f.f)

const StableAdd{T} = StableBinary{T, typeof(+)}
StableAdd(::Type{T}) where {T} = StableBinary(+, T)

const StableMul{T} = StableBinary{T, typeof(*)}
StableMul(::Type{T}) where {T} = StableBinary(*, T)


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

(f::LPartial{F, A})(arg...; kws...) where {F<:Function, A<:NonEmptyTuple{Any}} = 
f.f(f.arg..., arg...; kws...)
(f::RPartial{F, A})(arg...; kws...) where {F<:Function, A<:NonEmptyTuple{Any}} = 
f.f(arg..., f.arg...; kws...)

LPartial(f::Function, args::NonEmptyTuple{Any}) = LateralPartial(f, args, Left() )
RPartial(f::Function, args::NonEmptyTuple{Any}) = LateralPartial(f, args, Right())

getOutputType(::Type{<:LateralPartial{F}}) where {F<:Function} = getOutputType(F)


const absSqrtInv = inv ∘ sqrt ∘ abs


function modestTypingMap(op::F, obj::AbstractArray) where {F<:Function}
    if isempty(obj)
        similar(obj, Union{})
    else
        map(op, obj)
    end
end

modestTypingMap(op::F, obj::Union{Tuple, NamedTuple}) where {F<:Function} = map(op, obj)


struct InputLimiter{N, F<:Function} <: Modifier
    f::F

    function InputLimiter(f::F, ::Val{N}) where {F<:Function, N}
        checkPositivity(N::Int)
        new{numberOfInput, F}(f)
    end
end

InputLimiter(f::InputLimiter, ::Val{N}) where {N} = InputLimiter(f.f, Val(N))

(f::InputLimiter{N, F})(arg::Vararg{Any, N}) where {N, F<:Function} = f.f(arg...)

getOutputType(::Type{<:InputLimiter{<:Any, F}}) where {F<:Function} = getOutputType(F)


struct ChainMapper{F<:FunctionChainUnion{Function}} <: Mapper
    chain::F

    function ChainMapper(chain::F) where {F<:CustomMemory{<:Function}}
        checkEmptiness(chain, :chain)
        new{F}(chain)
    end

    function ChainMapper(chain::F) where {F<:GeneralTupleUnion{ NonEmptyTuple{Function} }}
        new{F}(chain)
    end
end

ChainMapper(chain::AbstractArray{<:Function}) = ChainMapper(chain|>ShapedMemory)

function getField(obj, f::ChainMapper, finalizer::F=itself) where {F<:Function}
    modestTypingMap(f.chain) do mapper
        mapper(obj) |> finalizer
    end
end

function getOutputType(::Type{ChainMapper{F}}) where {F<:NonEmptyTuple{Function}}
    Tuple{getOutputType.(fieldtypes(F))...}
end

function getOutputType(::Type{ChainMapper{ NamedTuple{S, F} }}) where 
                      {S, F<:NonEmptyTuple{Function}}
    NamedTuple{S, Tuple{getOutputType.(fieldtypes(F))...}}
end

function getOutputType(::Type{<:ChainMapper{ <:CustomMemory{F, N} }}) where {F<:Function, N}
    type = getOutputType(F)
    genParametricType(CustomMemory, (;F=type, N))
end


inferBinaryOpOutputType(::ArithmeticOperator, ::Type{T}, 
                        ::Type{T}) where {T<:RealOrComplex} = 
T

inferBinaryOpOutputType(::ArithmeticOperator, ::Type{T}, 
                        ::Type{Complex{T}}) where {T<:Real} = 
Complex{T}

inferBinaryOpOutputType(::ArithmeticOperator, ::Type{Complex{T}}, 
                        ::Type{T}) where {T<:Real} = 
Complex{T}

inferBinaryOpOutputType(::typeof(^), ::Type{T}, ::Integer) where {T<:Real} = 
T

inferBinaryOpOutputType(::Function, ::Type, ::Type) = Any

inferBinaryOpOutputType(::TypedReturn{T}, ::Type, ::Type) where {T} = T


function genTypedCoupler(coupler::F, encoderL::FL, encoderR::FR) where {F, FL, FR}
    argTypeL = getOutputType(encoderL)
    argTypeR = getOutputType(encoderR)
    inferredType = inferBinaryOpOutputType(coupler, argTypeL, argTypeR)
    returnType = typeintersect(getOutputType(F), inferredType)
    TypedBinary(TypedReturn(coupler, returnType), argTypeL, argTypeR)
end

genTypedCoupler(coupler::TypedBinary, ::Function, ::Function) = itself(coupler)


struct PairCoupler{T, J<:TypedBinary{T}, FL<:Function, FR<:Function} <: TypedEvaluator{T}
    coupler::J
    left::FL
    right::FR

    function PairCoupler(coupler::F, left::FL, right::FR) where 
                        {F<:Function, FL<:Function, FR<:Function}
        formattedCoupler = genTypedCoupler(coupler, left, right)
        T = getOutputType(formattedCoupler)
        J = typeof(formattedCoupler)
        new{T, J, FL, FR}(formattedCoupler, left, right)
    end
end

function (f::PairCoupler{T, J})(arg::Vararg) where {T, J<:TypedBinary{T}}
    f.coupler(f.left(arg...), f.right(arg...))
end

getOutputType(::Type{<:PairCoupler{T}}) where {T} = T


struct CartesianHeader{N, F<:Function} <: Modifier
    f::F

    function CartesianHeader(f::F, ::Val{N}) where {F<:Function, N}
        checkPositivity(N::Int, true)
        new{N, F}(f)
    end
end

CartesianHeader(f::CartesianHeader, ::Val{N}) where {N} = CartesianHeader(f.f, Val(N))

CartesianHeader(::Val{N}) where {N} = CartesianHeader(itself, Val(N))

(f::CartesianHeader{N, F})(head::T, body::Vararg) where {N, F<:Function, T} = 
f.f(formatInput(CartesianInput{N}(), head), body...)

getOutputType(::Type{<:CartesianHeader{<:Any, F}}) where {F<:Function} = getOutputType(F)

const TypedCarteFunc{T, D, F<:Function} = TypedReturn{T, CartesianHeader{D, F}}

TypedCarteFunc(f::Function, ::Type{T}, ::Val{D}) where {T, D} = 
TypedReturn(CartesianHeader(f, Val(D)), T)

TypedCarteFunc(f::TypedReturn, ::Type{T}, ::Val{D}) where {T, D} = 
TypedReturn(CartesianHeader(f.f, Val(D)), T)


struct SelectHeader{N, K, F<:Function} <: Modifier
    f::F

    function SelectHeader{N, K}(f::F) where {N, K, F<:Function}
        checkPositivity(K, true)
        N < K && throw(AssertionError("N must be no less than K=$K."))
        new{N::Int, K::Int, F}(f)
    end
end

(f::SelectHeader{N, K, F})(arg::Vararg{Any, N}) where {N, K, F<:Function} = 
f.f(arg[begin+K-1])

(f::SelectHeader{N, 0, F})(::Vararg{Any, N}) where {N, F<:Function} = f.f()

getOutputType(::Type{<:SelectHeader{<:Any, <:Any, F}}) where {F<:Function} = 
getOutputType(F)


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


struct FloatingMonomial{T<:Real, D} <: TypedEvaluator{T}
    center::NTuple{D, T}
    degree::WeakComp{D}

    FloatingMonomial(center::NonEmptyTuple{T}, degree::WeakComp{D}) where {T, D} = 
    new{T, D}(center, degree)
end

FloatingMonomial(center::NonEmptyTuple{T, D}, degree::NonEmptyTuple{Int, D}) where {T, D} = 
FloatingMonomial(center, WeakComp(degree))

function evalFloatingMonomial(f::FloatingMonomial{T, D}, 
                              coord::NTuple{D, Real}) where {T<:Real, D}
    res = mapreduce(*, coord, f.center, f.degree.tuple) do c1, c2, pwr
        (c1 - c2)^pwr
    end

    convert(T, res)
end

(::SelectTrait{InputStyle})(::FloatingMonomial{<:Real, D}) where {D} = CartesianInput{D}()

(f::FloatingMonomial)(coord) = evalFloatingMonomial(f, formatInput(f, coord))

getOutputType(::Type{<:FloatingMonomial{T}}) where {T<:Real} = T


struct Storage{T} <: TypedEvaluator{T}
    value::T
end

(f::Storage{T})(::Vararg) where {T} = f.value

getOutputType(::Type{Storage{T}}) where {T} = T


struct BinaryReduce{T, J<:TypedBinary{T}, F<:StableBinary{T}} <: TypedEvaluator{T}
    coupler::J
    reducer::F
end

function BinaryReduce(coupler::J, reducer::Function) where {T, J<:TypedBinary{T}}
    BinaryReduce(coupler, StableBinary(reducer, T))
end

function (f::BinaryReduce{T, J, F})(left::LinearSequence, right::LinearSequence) where 
                                   {T, J<:TypedBinary{T}, F<:StableBinary{T}}
    mapreduce(f.reducer, left, right) do l, r
        f.coupler(l, r)
    end
end

const Contract{T<:Number, EL<:Number, ER<:Number, J<:TypedBinary{T, typeof(*), EL, ER}} = 
      BinaryReduce{T, J, StableBinary{T, typeof(+)}}

Contract(::Type{T}, ::Type{EL}, ::Type{ER}) where {T<:Number, EL<:Number, ER<:Number} = 
BinaryReduce(TypedBinary(TypedReturn(*, T), EL, ER), StableBinary(+, T))

getOutputType(::Type{<:BinaryReduce{T}}) where {T} = T