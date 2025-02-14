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

ReturnTyped(f::ReturnTyped{T}, ::Type{T}) where {T} = itself(f)

(f::ReturnTyped{T, F})(arg...; kws...) where {T, F} = convert(T, f.f(arg...; kws...))

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

StableAdd(::Type{T}) where {T} = StableBinary(+, T)
StableMul(::Type{T}) where {T} = StableBinary(*, T)


struct Retrieve{P<:CompositePointer} <: FunctionComposer
    rule::P
end

const GetFlavor{T} = Retrieve{IndexPointer{Flavor{T}, 1}}

function (f::Retrieve)(input)
    getField(input, f.rule)
end

const Select{P<:EntryPointer} = Retrieve{P}

const Filter{P<:EffectivePtrStack} = Retrieve{P}

struct EncodeApply{N, E<:NTuple{N, Function}, F<:Function} <: FunctionCombiner
    encode::E
    apply::F
end

EncodeApply(encode::Function, apply::Function) = EncodeApply((encode,), apply)

(f::EncodeApply)(args...) = f.apply(map(f->f(args...), f.encode)...)

(f::EncodeApply{0})(args...) = f.apply(args...)


struct ParamFilterFunc{F<:Function, 
                       T<:NonEmptyTuple{EffectivePtrStack}} <: ParamFuncBuilder{F}
    apply::F
    scope::T
end

ParamFilterFunc(apply::F, scope::P) where {F<:Function, P<:EffectivePtrStack} = 
ParamFilterFunc(apply, (scope,))

function (f::ParamFilterFunc)(input, param)
    f.apply(input, getField.(Ref(param), f.scope)...)
end


struct ParamSelectFunc{F<:Function, T<:Tuple{Vararg{ChainIndexer}}} <: ParamFuncBuilder{F}
    apply::F
    select::T
end

ParamSelectFunc(f::Function) = ParamSelectFunc(f, ())

const EmptySelectFunc{F<:Function} = ParamSelectFunc{F, Tuple{}}

ParamSelectFunc(f::EmptySelectFunc) = itself(f)

ParamSelectFunc(f::ParamSelectFunc, select::Tuple{Vararg{ChainIndexer}}) = 
ParamSelectFunc(f.apply, (f.select..., select...))

function (f::ParamSelectFunc)(input, param)
    f.apply(input, getField(param, f.select)...)
end

(f::EmptySelectFunc)(input, _) = f.apply(input)

const GetParamFunc{F<:Function, T<:ChainIndexer} = ParamSelectFunc{F, Tuple{T}}


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


# struct InsertOnward{C<:Function, F<:Function}
#     apply::C
#     dress::F
# end

# (f::InsertOnward)(arg, args...) = f.dress(f.apply(arg, args...), args...)


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


# struct Plus{T} <: CompositeFunction
#     val::T
# end

# (f::Plus{T})(arg::T) where {T} = arg + f.val


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

function (f::HermitianContract{T})(params::FilteredVecOfArr{T}, 
                                   vInput::AbstractVector{T}) where {T}
    res = zero(T)
    len = length(f.diagonal)

    for i in 1:len
        c = vInput[begin+i-1]
        res += f.diagonal[begin+i-1](params) * c' * c
    end

    for j in 1:length(f.uppertri)
        m, n = convertIndex1DtoTri2D(j)
        c1 = vInput[begin+m-1]
        c2 = vInput[begin+n]
        val = f.uppertri[begin+j-1](params) * c1' * c2
        res += val + val'
    end

    res
end


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