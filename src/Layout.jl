abstract type TensorPointer{T, N} <: Any end

struct TensorType{T, N} <: StructuredType
    shape::NTuple{N, Int}

    TensorType(::T) where {T} = new{T, 0}(())
    TensorType(::Type{T}=Any, shape::NTuple{N, Int}=()) where {T, N} = new{T, N}(shape)
end

TensorType(t::TensorType) = itself(t)
TensorType(a::AbstractArray{T, N}) where {T, N} = TensorType(T, size(a))
TensorType(::ElementalParam{T}) where {T} = TensorType(T)
TensorType(p::FlattenedParam{T}) where {T} = TensorType(T, outputSizeOf(p))
TensorType(p::JaggedParam{T, N}) where {T, N} = 
TensorType(AbstractArray{T, N}, outputSizeOf(p))

(::TensorType{T, 0})() where {T} = T

(::TensorType{T, N})() where {T, N} = AbstractArray{T, N}


struct FirstIndex <: StructuredType end

const GeneralIndex = Union{Int, Symbol, FirstIndex}


struct ChainPointer{T, N, C<:Tuple{Vararg{GeneralIndex}}} <: TensorPointer{T, N}
    chain::C
    type::TensorType{T, N}

    ChainPointer(chain::C, type::TensorType{T, N}=TensorType()) where 
                {T, N, C<:Tuple{Vararg{GeneralIndex}}} = 
    new{T, N, C}(chain, type)
end

const IndexPointer{T, N} = ChainPointer{T, N, Tuple{Int}}

const FlatPSetInnerPtr{T} = ChainPointer{T, 0, Tuple{FirstIndex, Int}}

const FlatParamSetIdxPtr{T} = Union{IndexPointer{T}, FlatPSetInnerPtr{T}}

ChainPointer(sourceType::TensorType=TensorType()) = ChainPointer((), sourceType)

ChainPointer(entry::GeneralIndex, type::TensorType=TensorType()) = 
ChainPointer((entry,), type)

ChainPointer(prev::GeneralIndex, here::ChainPointer) = 
ChainPointer(ChainPointer(prev), here)

ChainPointer(prev::ChainPointer, here::ChainPointer) = 
ChainPointer((prev.chain..., here.chain...), here.type)


getField(ptr) = Base.Fix2(getField, ptr)

const GetField{T, N, C} = Base.Fix2{typeof(getField), ChainPointer{T, N, C}}

const GetIndex{T, N} = GetField{T, N, Tuple{Int}}

getField(obj, ::FirstIndex) = first(obj)

getField(obj, entry::Symbol) = getfield(obj, entry)

getField(obj, entry::Int) = getindex(obj, entry)

getField(obj, ptr::ChainPointer) = foldl(getField, ptr.chain, init=obj)

getField(obj::AbstractDict, ptr::ChainPointer) = getindex(obj, ptr)


function evalField(obj, ptr::ChainPointer)
    field = getField(obj, ptr)
    field isa ParamBox && (field = obtain(field))
    convert(ptr.type(), field)
end

evalField(obj, entry::GeneralIndex) = getField(obj, entry)

evalField(ptr) = Base.Fix2(evalField, ptr)

const EvalField{T, N, C} = Base.Fix2{typeof(evalField), ChainPointer{T, N, C}}