struct ReturnTyped{T, F<:Function} <: TypedEvaluator{T}
    f::F

    function ReturnTyped(f::F, ::Type{T}) where {F<:Function, T}
        new{T, F}(f)
    end
end

ReturnTyped(::Type{T}) where {T} = ReturnTyped(itself, T)

ReturnTyped(f::ReturnTyped{T}, ::Type{T}) where {T} = itself(f)

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

struct StableContract{T} <: TypedEvaluator{T} end

function (::StableContract{T})(a, b) where {T}
    mapreduce(StableMul(T), StableAdd(T), a, b)
end


struct ParamSelectFunc{F<:Function, T<:Tuple{ Vararg{Encoder} }} <: ParamFuncBuilder{F}
    apply::F
    select::T
end

ParamSelectFunc(f::Function, select::Encoder) = ParamSelectFunc(f, (select,))

ParamSelectFunc(f::Function) = ParamSelectFunc(f, ())

const EmptySelectFunc{F<:Function} = ParamSelectFunc{F, Tuple{}}

ParamSelectFunc(f::EmptySelectFunc) = itself(f)

ParamSelectFunc(f::ParamSelectFunc, select::Tuple{Vararg{Encoder}}) = 
ParamSelectFunc(f.apply, (f.select..., select...))

function (f::ParamSelectFunc)(input, param)
    f.apply(input, getField.(Ref(param), f.select)...)
end

(f::EmptySelectFunc)(input, _) = f.apply(input)

const GetParamFunc{F<:Function, T<:Encoder} = ParamSelectFunc{F, Tuple{T}}


struct OnlyHead{F<:Function} <: FunctionModifier
    f::F
end

(f::OnlyHead)(arg1, ::Vararg) = f.f(arg1)


struct OnlyBody{F<:Function} <: FunctionModifier
    f::F
end

(f::OnlyBody)(_, args...) = f.f(args...)


struct PairCombine{J<:Function, FL<:Function, FR<:Function} <: FunctionCombiner
    joint::J
    left::FL
    right::FR
end

PairCombine(joint::F) where {F<:Function} = 
(left::Function, right::Function) -> PairCombine(joint, left, right)

(f::PairCombine)(arg, args...) = f.joint( f.left(arg, args...), f.right(arg, args...) )


struct ChainReduce{J<:Function, C<:LinearMemory{<:Function}} <: FunctionCombiner
    joint::J
    chain::C

    function ChainReduce(joint::J, chain::C) where {J, C<:VectorMemory{<:Function}}
        checkEmptiness(chain, :chain)
        new{J, C}(joint, chain)
    end

    function ChainReduce(joint::J, chain::C) where {J, C<:AbstractVector{<:Function}}
        checkEmptiness(chain, :chain)
        new{J, C}(joint, getMemory(chain))
    end
end

ChainReduce(joint::Function, chain::NonEmptyTuple{Function}) = 
ChainReduce(joint, VectorMemory(chain))

ChainReduce(joint::F) where {F<:Function} = Base.Fix1(ChainReduce, joint)

const CountedChainReduce{J, P, N} = ChainReduce{J, VectorMemory{P, N}}

(f::CountedChainReduce)(arg, args...) = 
mapreduce(o->o(arg, args...), f.joint, f.chain.value)

(f::ChainReduce)(arg, args...) = mapreduce(o->o(arg, args...), f.joint, f.chain)


struct InsertInward{C<:Function, F<:Function} <: FunctionCombiner
    apply::C
    dress::F
end

(f::InsertInward)(arg, args...) = f.apply(f.dress(arg, args...), args...)


struct Storage{T} <: CompositeFunction
    val::T
end

(f::Storage)(::Vararg) = f.val

struct Unit{T} <: CompositeFunction end

Unit(::Type{T}) where {T} = Unit{T}()

(f::Unit{T})(::Vararg) where {T} = one(T)


struct Power{F<:Function, N} <: CompositeFunction
    f::F

    Power(f::F, ::Val{N}) where {F<:Function, N} = new{F, N}(f)
end

Power(f::Function, n::Int) = Power(f, Val(n))

(f::Power{<:Function, N})(arg::Vararg) where {N} = f.f(arg...)^(N::Int)


struct ShiftByArg{T, D} <: FieldlessFunction end

function (::ShiftByArg{T, D})(input::Union{NTuple{D, T}, AbstractVector{T}}, 
                              args::Vararg{T, D}) where {T, D}
    input .- args
end


struct HermitianContract{T, F1<:ReturnTyped{T}, F2<:ReturnTyped{T}} <: FunctionCombiner
    diagonal::Memory{F1}
    uppertri::Memory{F2}

    function HermitianContract(dFuncs::Memory{F1}, uFuncs::Memory{F2}) where 
                              {T, F1<:ReturnTyped{T}, F2<:ReturnTyped{T}}
        checkLength(uFuncs, :uFuncs, triMatEleNum(length(dFuncs)-1))
        new{T, F1, F2}(dFuncs, uFuncs)
    end
end

function (f::HermitianContract{T})(input, weights::AbstractVector{T}) where {T}
    res = zero(T)
    len = length(f.diagonal)

    for i in 1:len
        c = weights[begin+i-1]
        res += f.diagonal[begin+i-1](input) * c' * c
    end

    for j in 1:length(f.uppertri)
        m, n = convertIndex1DtoTri2D(j)
        c1 = weights[begin+m-1]
        c2 = weights[begin+n]
        val = f.uppertri[begin+j-1](input) * c1' * c2
        res += val + val'
    end

    res
end


struct Left end

struct Right end

const Lateral = Union{Left, Right}


struct LateralPartial{F<:Function, A<:NonEmptyTuple{Any}, L<:Lateral} <: FunctionModifier
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


struct KeywordPartial{F, A<:NonEmptyTuple{Pair{Symbol, <:Any}}} <: FunctionModifier
    f::F
    arg::A
    replaceable::Bool

    function KeywordPartial(f::F, pairs::NonEmptyTuple{Pair{Symbol, <:Any}}, 
                            replaceable::Bool=true) where {F<:Function}
        new{F, typeof(pairs)}(f, pairs, replaceable)
    end
end

function (f::KeywordPartial)(arg...; kws...)
    if f.replaceable
        f.f(arg...; f.arg..., kws...)
    else
        f.f(arg...; kws..., f.arg...)
    end
end


const AbsSqrtInv = inv∘sqrt∘abs


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


# struct Bifurcator{F<:Function, FL<:Function, FR<:Function} <: FunctionCombiner
#     finalizer::F
#     left::FL
#     right::FR
# end

# Bifurcator(finalizer::F) where {F<:Function} = 
# (left::Function, right::Function) -> Bifurcator(finalizer, left, right)

# (f::Bifurcator)(arg, args...) = f.finalizer(f.left(arg, args...), f.right(arg, args...))

# struct EntryEncoder{F<:Function} <: CompositeFunction
#     core::F
#     link::ChainedAccess

#     EntryEncoder(core::F, link::ChainedAccess) where {F<:Function} = new{F}(core, link)
# end

# (f::EntryEncoder)(args...) = f.core(args...)


struct InputLimiter{N, F<:Function} <: FunctionModifier
    f::F

    function InputLimiter(f::F, ::Val{N}) where {F<:Function, N}
        checkPositivity(N::Int)
        new{numberOfInput, F}(f)
    end
end

InputLimiter(f::InputLimiter, ::Val{N}) where {N} = InputLimiter(f.f, Val(N))

(f::InputLimiter{N})(arg::Vararg{Any, N}) where {N} = f.f(arg...)


struct ChainMapper{F<:FunctionChainUnion} <: Mapper
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
    map(f.chain) do mapper
        mapper(obj)
    end |> finalizer
end


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

struct Deref{F<:Function} <: FunctionModifier
    f::F
end

(f::Deref)(arg::AbstractArray{<:Any, 0}) = f.f(arg[])
(f::Deref)(arg::Tuple{Any}) = f.f(arg|>first)
(f::Deref)(arg::NamedTuple{<:Any, <:Tuple{Any}}) = f.f(arg|>first)


struct TupleHeader{N, F<:Function} <: FunctionModifier
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


struct SelectHeader{N, K, F<:Function} <: FunctionModifier
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
    (Base.issingletontype(f|>typeof) || isParamBoxFree(f)) ? Lucent() : Opaque()
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