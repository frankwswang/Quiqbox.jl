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


struct ReturnTyped{T, F<:Function} <: TaggedFunction
    f::F

    function ReturnTyped(f::F, ::Type{T}) where {F, T}
        new{T, F}(f)
    end
end

ReturnTyped(::Type{T}) where {T} = ReturnTyped(itself, T)

(f::ReturnTyped{T, F})(arg) where {T, F} = convert(T, f.f(arg))

const Return{T} = ReturnTyped{T, ItsType}




struct PointerFunc{F<:Function, T<:NonEmptyTuple{Union{Int, Memory{Int}}}} <: ParamOperator
    apply::F
    pointer::T
    sourceID::UInt
end

(f::PointerFunc)(input, param) = f.apply(input, getindex.(Ref(param), f.pointer)...)

const PointOneFunc{F} = PointerFunc{F, Tuple{Int}}

# struct PointerFunc{F<:Function, N, T<:Tuple{Vararg{Any, N}}} <: ParamOperator
#     apply::F
#     pointer::NTuple{N, Int}
#     type::Type{T}
#     sourceID::UInt

#     function PointerFunc(apply::F, pointer::NonEmptyTuple{Int, N}, ::Type{T}, 
#                          sourceID::UInt) where {F, N, T<:NonEmptyTuple{Any, N}}
#         new{F, N+1, T}(apply, pointer, T, sourceID)
#     end
# end

# function evalFunc(f::PointerFunc, input, param)
#     # getIds = map((x, y)->ReturnTyped(y)∘GetIndex(x), f.pointer, fieldtypes(f.type))
#     args = map(f.pointer, fieldtypes(f.type)) do x, y
#         (ReturnTyped(y)∘getindex)(param, x)
#     end
#     f.apply(input, args...)
# end

# const PointOneFunc{F, T} = PointerFunc{F, 1, Tuple{T}}


struct OnlyInput{F<:Function} <: Evaluator{F}
    f::F
end

(f::OnlyInput)(input, _) = f.f(input)


struct OnlyParam{F<:Function} <: Evaluator{F}
    f::F
end

(f::OnlyParam)(_, param) = f.f(param)


struct ParamFunc{F<:Function} <: Evaluator{F}
    f::F
end

(f::ParamFunc)(input, param) = f.f(input, param)

struct PairCombine{J<:Function, FL<:ParamOperator, FR<:ParamOperator} <: ChainedOperator{J}
    joint::J
    left::FL
    right::FR
end

const AddPair{FL, FR} = PairCombine{typeof(+), FL, FR}
const MulPair{FL, FR} = PairCombine{typeof(*), FL, FR}

PairCombine(joint::F) where {F<:Function} = 
(left::ParamOperator, right::ParamOperator) -> PairCombine(joint, left, right)

(f::PairCombine)(input, param) = f.joint( f.left(input, param), f.right(input, param) )

#! Test!!
struct ChainReduce{J<:Function, C<:LinearMemory{<:ParamOperator}} <: ChainedOperator{J}
    joint::J
    chain::C

    function ChainReduce(joint::J, chain::C) where {J, C<:VectorMemory{<:ParamOperator}}
        checkEmptiness(chain, :chain)
        new{J, C}(joint, chain)
    end

    function ChainReduce(joint::J, chain::C) where {J, C<:AbstractVector{<:ParamOperator}}
        checkEmptiness(chain, :chain)
        new{J, C}(joint, getMemory(chain))
    end
end

ChainReduce(joint::Function, chain::NonEmptyTuple{ParamOperator}) = 
ChainReduce(joint, VectorMemory(chain))

ChainReduce(joint::F) where {F<:Function} = Base.Fix1(ChainReduce, joint)

const CountedChainReduce{J, P, N} = ChainReduce{J, VectorMemory{P, N}}

(f::CountedChainReduce)(input, param) = 
mapreduce(o->o(input, param), f.joint, f.chain.value)

(f::ChainReduce)(input, param) = mapreduce(o->o(input, param), f.joint, f.chain)

const AddChain{C} = ChainReduce{typeof(+), C}
const MulChain{C} = ChainReduce{typeof(*), C}


struct InsertOnward{C<:ParamOperator, F<:ParamOperator} <: ParamOperator
    apply::C
    dress::F
end

(f::InsertOnward)(input, param) = f.dress(f.apply(input, param), param)


struct InsertInward{C<:ParamOperator, F<:ParamOperator} <: ParamOperator
    apply::C
    dress::F
end

(f::InsertInward)(input, param) = f.apply(f.dress(input, param), param)


function evalFunc(func::F, input) where {F<:Function}
    fCore, pars = unpackFunc(func)
    fCore(input, map(obtain, pars))
end


unpackFunc(f::Function) = unpackFunc!(f, ParamBox[])

unpackFunc!(f::Function, paramSet::PBoxAbtArray) = 
unpackFunc!(SelectTrait{ParameterStyle}()(f), f, paramSet)

unpackFunc!(::IsParamFunc, f::Function, paramSet::PBoxAbtArray) = 
unpackParamFunc!(f, paramSet)

function unpackFunc!(::NotParamFunc, f::Function, paramSet::PBoxAbtArray)
    params = getParams(f)
    ids = locateParam!(paramSet, params)
    fCopy = deepcopy(f)
    paramDoubles = deepcopy(params)
    foreach(p->setScreenLevel!(p.source, 1), paramDoubles)
    evalCore = function (x, pVals::Vararg)
        for (p, v) in zip(paramDoubles, pVals)
            setVal!(p, v)
        end
        fCopy(x)
    end
    PointerFunc(evalCore, ids, objectid(paramSet)), paramSet
end