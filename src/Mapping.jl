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


struct ParamFilterFunc{F<:Function, T<:CompositePointer} <: ParamFuncBuilder{F}
    apply::F
    scope::T
end

(f::ParamFilterFunc)(input, param) = f.apply(input, getField(param, f.scope))


struct ParamSelectFunc{F<:Function, T<:Tuple{Vararg{ChainIndexer}}} <: ParamFuncBuilder{F}
    apply::F
    select::T
end

ParamSelectFunc(f::Function) = ParamSelectFunc(f, ())

function (f::ParamSelectFunc)(input, param)
    f.apply(input, getField(param, f.select)...)
end

(f::ParamSelectFunc{<:Function, Tuple{}})(input, _) = f.apply(input)

const GetParamFunc{F<:Function, T<:ChainIndexer} = ParamSelectFunc{F, Tuple{T}}


struct OnlyHead{F<:Function} <: FunctionModifier
    f::F
end

(f::OnlyHead)(arg1, ::Vararg) = f.f(arg1)


struct OnlyBody{F<:Function} <: FunctionModifier
    f::F
end

(f::OnlyBody)(_, args...) = f.f(args...)


struct PairCombine{J<:Function, FL<:Function, FR<:Function} <: JoinedOperator{J}
    joint::J
    left::FL
    right::FR
end

PairCombine(joint::F) where {F<:Function} = 
(left::Function, right::Function) -> PairCombine(joint, left, right)

(f::PairCombine)(arg, args...) = f.joint( f.left(arg, args...), f.right(arg, args...) )


struct ChainReduce{J<:Function, C<:LinearMemory{<:Function}} <: JoinedOperator{J}
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


# struct InsertOnward{C<:Function, F<:Function}
#     apply::C
#     dress::F
# end

# (f::InsertOnward)(arg, args...) = f.dress(f.apply(arg, args...), args...)


struct InsertInward{C<:Function, F<:Function} <: FunctionComposer
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

function evalFunc(fCore::F, pSet::Union{DirectParamSource, TypedParamInput}, 
                  input::T) where {F<:Function, T}
    fCore(input, evalParamSet(pSet))
end

function evalFunc(fCore::F, pVals::AbtVecOfAbtArr, input::T) where {F<:Function, T}
    fCore(input, pVals)
end

#! Possibly adding memoization in the future to generate/use the same param set to avoid 
#! bloating `Quiqbox.IdentifierCache` and prevent repeated computation.
unpackFunc(f::Function) = unpackFunc!(f, initializeParamSet(f))

unpackFunc!(f::Function, paramSet::AbstractVector) = 
unpackFunc!(SelectTrait{ParameterizationStyle}()(f), f, paramSet)

unpackFunc!(::TypedParamFunc, f::Function, paramSet::AbstractVector) = 
unpackParamFunc!(f, paramSet)

unpackFunc!(::GenericFunction, f::Function, paramSet::AbstractVector) = 
unpackTypedFunc!(f, paramSet)

const FieldPtrPair{T} = Pair{<:ChainPointer, <:FlatParamSetIdxPtr{T}}
const FieldPtrPairs{T} = AbstractVector{<:FieldPtrPair{T}}
const FieldPtrDict{T} = AbstractDict{<:ChainPointer, <:FlatParamSetIdxPtr{T}}
const EmptyFieldPtrDict{T} = TypedEmptyDict{Union{}, FlatPSetInnerPtr{T}}
const FiniteFieldPtrDict{T, N} = FiniteDict{N, <:ChainPointer, <:FlatParamSetIdxPtr{T}}

const FieldValDict{T} = AbstractDict{<:FlatParamSetIdxPtr{T}, <:Union{T, AbstractArray{T}}}
const ParamValOrDict{T} = Union{AbtVecOfAbtArr{T}, FieldValDict{T}}

abstract type FieldParamPointer{R} <: Any end

struct MixedFieldParamPointer{T, R<:FieldPtrDict{T}} <: FieldParamPointer{R}
    core::R
    tag::Identifier
end

function MixedFieldParamPointer(paramPairs::FieldPtrPairs{T}, tag::Identifier) where {T}
    coreDict = buildDict(paramPairs, EmptyFieldPtrDict{T})
    MixedFieldParamPointer(coreDict, tag)
end


# `f` should only take one input.
unpackTypedFunc!(f::Function, paramSet::AbstractVector, 
                 paramSetId::Identifier=Identifier(paramSet)) = 
unpackTypedFunc!(ReturnTyped(f, Any), paramSet, paramSetId)

function unpackTypedFuncCore!(f::ReturnTyped{T}, paramSet::AbstractVector) where {T}
    params, anchors = getFieldParams(f)
    if isempty(params)
        ParamSelectFunc(f), paramSet, FieldPtrPair{T}[]
    else
        ids = locateParam!(paramSet, params)
        fDummy = deepcopy(f.f)
        paramDoubles = getFieldParams(fDummy)
        foreach(p->setScreenLevel!(p.source, 1), paramDoubles)
        evalCore = function (x, pVals::Vararg)
            for (p, v) in zip(paramDoubles, pVals)
                setVal!(p, v)
            end
            fDummy(x)
        end
        paramPairs = getMemory(ChainPointer(:apply, anchors) .=> ids)
        ParamSelectFunc(ReturnTyped(evalCore, T), ids), paramSet, paramPairs
    end
end

function unpackTypedFunc!(f::ReturnTyped{T}, paramSet::AbstractVector, 
                          paramSetId::Identifier=Identifier(paramSet)) where {T}
    fCore, _, paramPairs = unpackTypedFuncCore!(f, paramSet)
    ptrDict = buildDict(paramPairs, EmptyFieldPtrDict{T})
    paramPtr = MixedFieldParamPointer(ptrDict, paramSetId)
    fCore, paramSet, paramPtr
end


struct Identity <: DirectOperator end

(::Identity)(f::Function) = itself(f)


struct LeftPartial{F<:Function, A<:NonEmptyTuple{Any}} <: FunctionModifier
    f::F
    header::A

    function LeftPartial(f::F, arg, args...) where {F<:Function}
        allArgs = (arg, args...)
        new{F, typeof(allArgs)}(f, allArgs)
    end
end

(f::LeftPartial)(arg...) = f.f(f.header..., arg...)


const AbsSqrtInv = inv∘sqrt∘abs