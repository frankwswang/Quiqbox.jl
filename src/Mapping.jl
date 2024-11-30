struct VectorMemory{T, L} <: AbstractMemory{T, 1}
    value::Memory{T}
    shape::Val{L}

    function VectorMemory(value::Memory{T}, ::Val{L}) where {T, L}
        checkLength(value, :value, L)
        new{T, L}(value, Val(L))
    end
end

VectorMemory(input::AbstractArray) = VectorMemory(getMemory(input), Val(length(input)))

VectorMemory(input::NonEmptyTuple{Any, L}) where {L} = 
VectorMemory(getMemory(input), Val(L+1))

VectorMemory(input::VectorMemory)  = itself(input)

import Base: size, getindex, setindex!, iterate, length
size(::VectorMemory{<:Any, L}) where {L} = (L,)

getindex(arr::VectorMemory, i::Int) = getindex(arr.value, i)

setindex!(arr::VectorMemory, val, i::Int) = setindex!(arr.value, val, i)

iterate(arr::VectorMemory) = iterate(arr.value)
iterate(arr::VectorMemory, state) = iterate(arr.value, state)

length(::VectorMemory{<:Any, L}) where {L} = L

const LinearMemory{T} = Union{Memory{T}, VectorMemory{T}}


struct ReturnTyped{T, F<:Function} <: TypedEvaluator{T, F}
    f::F

    function ReturnTyped(f::F, ::Type{T}) where {F<:Function, T}
        new{T, F}(f)
    end
end

ReturnTyped(::Type{T}) where {T} = ReturnTyped(itself, T)

(f::ReturnTyped{T, F})(arg...) where {T, F} = convert(T, f.f(arg...))

const Return{T} = ReturnTyped{T, ItsType}


struct StableBinary{T, F<:Function} <: TypedEvaluator{T, F}
    f::F

    function StableBinary(f::F, ::Type{T}) where {F<:Function, T}
        new{T, F}(f)
    end
end

(f::StableBinary{T, F})(argL::T, argR::T) where {T, F} = convert(T, f.f(argL, argR))

const StableAdd{T} = StableBinary{T, typeof(+)}
const StableMul{T} = StableBinary{T, typeof(*)}

StableBinary(f::Function) = Base.Fix1(StableBinary, f)


struct PointerFunc{F<:Function, T<:Tuple{Vararg{FlatParamSetIdxPtr}}} <: ParamOperator
    apply::F
    pointer::T
    sourceID::UInt
end

(f::PointerFunc)(input, param) = f.apply(input, getField.(Ref(param), f.pointer)...)

(f::PointerFunc{<:Function, Tuple{}})(input) = f.apply(input)

(f::PointerFunc{<:Function, Tuple{}})(input, _) = f(input)

const PointOneFunc{F, T} = PointerFunc{F, Tuple{T}}


struct OnlyHead{F<:Function} <: Evaluator{F}
    f::F
end

(f::OnlyHead)(arg1, ::Vararg) = f.f(arg1)


struct OnlyBody{F<:Function} <: Evaluator{F}
    f::F
end

(f::OnlyBody)(_, args...) = f.f(args...)


struct PairCombine{J<:Function, FL<:Function, FR<:Function} <: ChainedOperator{J}
    joint::J
    left::FL
    right::FR
end

PairCombine(joint::F) where {F<:Function} = 
(left::Function, right::Function) -> PairCombine(joint, left, right)

(f::PairCombine)(arg, args...) = f.joint( f.left(arg, args...), f.right(arg, args...) )


struct ChainReduce{J<:Function, C<:LinearMemory{<:Function}} <: ChainedOperator{J}
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


struct InsertOnward{C<:Function, F<:Function} <: ParamOperator
    apply::C
    dress::F
end

(f::InsertOnward)(arg, args...) = f.dress(f.apply(arg, args...), args...)


struct InsertInward{C<:Function, F<:Function} <: ParamOperator
    apply::C
    dress::F
end

(f::InsertInward)(arg, args...) = f.apply(f.dress(arg, args...), args...)


struct Storage{T} <: CompositeFunction
    val::T
end

(f::Storage)(::Vararg) = f.val


struct ShiftByArg{T<:Real, D} <: FieldlessFunction end

(::ShiftByArg{T, D})(input::NTuple{D, Real}, args::Vararg{T, D}) where {T, D} = 
(input .- args)


function evalFunc(func::F, input::T) where {F<:Function, T}
    fCore, pSet = unpackFunc(func)
    evalFunc(fCore, pSet, input)
end

function evalFunc(fCore::F, pSet::AbstractParamSet, input::T) where {F<:Function, T}
    fCore(input, evalParamSet(pSet))
end

function evalFunc(fCore::F, pVals::AbtVecOfAbtArr, input::T) where {F<:Function, T}
    fCore(input, pVals)
end


unpackFunc(f::Function) = unpackFunc!(f, initializeParamSet(FlatParamSet))

unpackFunc(f::AbstractAmplitude{T}) where {T} = 
unpackFunc!(f, initializeParamSet(FlatParamSet, T))

unpackFunc!(f::Function, paramSet::AbstractVector) = 
unpackFunc!(SelectTrait{ParameterStyle}()(f), f, paramSet)

unpackFunc!(::DefinedParamFunc, f::Function, paramSet::AbstractVector) = 
unpackParamFunc!(f, paramSet)

const FieldPointerDict{T} = Dict{<:ChainPointer, <:ChainPointer{T}}

abstract type FieldParamPointer{R} <: Any end

struct MixedFieldParamPointer{R<:FieldPointerDict} <: FieldParamPointer{R}
    core::R
end

function anchorFieldPointerDictCore(d::FieldPointerDict, anchor::ChainPointer)
    map( collect(d) ) do pair
        ChainPointer(anchor, pair.first) => pair.second
    end
end

function anchorFieldPointerDict(d::FieldPointerDict, anchor::ChainPointer)
    (Dict∘anchorFieldPointerDictCore)(d, anchor)
end

function unpackFunc!(::GeneralParamFunc, f::Function, paramSet::AbstractVector)
    params, anchors = getFieldParams(f)
    if isempty(params)
        PointerFunc(f, (), pSetId), paramSet, MixedFieldParamPointer(T, pSetId)
    else
        ids = locateParam!(paramSet, params)
        paramFieldPointer = MixedFieldParamPointer(Dict(anchors .=> ids), pSetId)
        fCopy = deepcopy(f)
        paramDoubles = deepcopy(params)
        foreach(p->setScreenLevel!(p.source, 1), paramDoubles)
        evalCore = function (x, pVals::Vararg)
            for (p, v) in zip(paramDoubles, pVals)
                setVal!(p, v)
            end
            fCopy(x)
        end
        PointerFunc(evalCore, ids, pSetId), paramSet, paramFieldPointer
    end
end