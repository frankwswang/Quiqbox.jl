struct TensorType{T, N} <: StructuredType
    type::Type{T}
    size::NTuple{N, Int}
end

TensorType(::Type{T}) where {T} = TensorType(T, ())

TensorType(arr::AbstractArray{T}) where {T} = TensorType(T, size(arr))

TensorType(::T) where {T} = TensorType(T)

abstract type TensorPointer{T} end

abstract type SymbolPointer end

struct Data0D end
struct Data1D end
struct Data2D end
struct DataXD end

const DataDim = Union{Data0D, Data1D, Data2D, DataXD}

getDataDim(::Type{<:ElementalParam}) = Data0D()
getDataDim(::Type{<:FlattenedParam}) = Data1D()
getDataDim(::Type{<:JaggedParam}) = Data2D()
getDataDim(::Type{<:Any}) = DataXD()
getDataDim(::T) where {T<:JaggedParam} = getDataDim(T)


struct IndexPointer{T<:DataDim} <: CompositeFunction
    idx::Int
    dim::T
end

const GetIndex = IndexPointer{DataXD}

IndexPointer(idx::Int, ele) = IndexPointer(idx, getDataDim(ele))
IndexPointer(idx::Int) = IndexPointer(idx, DataXD())

const IntOrSym = Union{Int, Symbol}

struct FieldPointer{T<:TensorType, C<:IntOrSym} <: TensorPointer{T}
    entry::C
    type::T
end

FieldPointer(entry::IntOrSym, objOrType=Any) = FieldPointer(entry, TensorType(objOrType))

struct ChainPointer{T<:TensorType, C<:Tuple{Missing, Vararg{IntOrSym}}} <: TensorPointer{T}
    chain::C
    type::T
end

ChainPointer(objOrType::Any=Any) = 
ChainPointer((missing,), TensorType(objOrType))

ChainPointer(entry::IntOrSym, type::TensorType=TensorType(Any)) = 
ChainPointer((missing, entry), type)

ChainPointer(prev::FieldPointer, here::FieldPointer) = 
ChainPointer((missing, prev.entry, here.entry), here.type)

ChainPointer(prev::FieldPointer, here::ChainPointer) = 
ChainPointer((missing, prev.entry, Base.tail(here.chain)...), here.type)

ChainPointer(prev::ChainPointer, here::FieldPointer) = 
ChainPointer((prev.chain..., here.entry), here.type)

ChainPointer(prev::ChainPointer, here::ChainPointer) = 
ChainPointer((prev.chain..., Base.tail(here.chain)...), here.type)

ChainPointer(prev::IntOrSym, here::Union{FieldPointer, ChainPointer}) = 
ChainPointer(ChainPointer(prev), here)

getField(obj, ::Missing) = itself(obj)

getField(obj, entry::Symbol) = getfield(obj, entry)

getField(obj, entry::Int) = getindex(obj, entry)

getField(obj, ptr::FieldPointer) = getField(obj, ptr.entry)

getField(obj, ptr::ChainPointer) = foldl(getField, ptr.chain, init=obj)