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
end

const IndexPointer{T, N} = ChainPointer{T, N, Tuple{Int}}

ChainPointer(sourceType::TensorType=TensorType(Any)) = ChainPointer((), sourceType)

ChainPointer(entry::GeneralIndex, source::Any=Any) = 
ChainPointer((entry,), TensorType(source))

ChainPointer(prev::GeneralIndex, here::ChainPointer) = 
ChainPointer(ChainPointer(prev), here)

ChainPointer(prev::ChainPointer, here::ChainPointer) = 
ChainPointer((prev.chain..., here.chain...), here.type)


getField(obj, ::FirstIndex) = first(obj)

getField(obj, entry::Symbol) = getfield(obj, entry)

getField(obj, entry::Int) = getindex(obj, entry)

getField(obj, ptr::ChainPointer) = foldl(getField, ptr.chain, init=obj)


evalField(obj, ptr::ChainPointer) = convert(ptr.type(), getField(obj, ptr))

evalField(obj, entry::GeneralIndex) = getField(obj, entry)

evalField(ptr) = Base.Fix2(evalField, ptr)

const EvalField{T, N, C} = Base.Fix2{typeof(evalField), ChainPointer{T, N, C}}