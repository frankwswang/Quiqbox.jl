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


struct TensorType{T, N} <: StructuredType
    type::Type{T}
    size::NTuple{N, Int}
end

TensorType(::Type{T}) where {T} = TensorType(T, ())

TensorType(arr::AbstractArray{T}) where {T} = TensorType(T, size(arr))

TensorType(::T) where {T} = TensorType(T)


struct FieldSymbol{S<:MissSymInt, T<:TensorType} <: SymbolPointer
    entry::S
    type::T

    FieldSymbol(entry::S, type::T=TensorType(Any)) where {S<:MissSymInt, T} = 
    new{S, T}(entry, type)
end

FieldSymbol(::Type{T}=Any) where {T} = FieldSymbol(missing, TensorType(T))

FieldSymbol(obj::FieldSymbol) = itself(obj)


struct FieldLinker{S<:MissSymInt, T} <: SymbolPointer
    prev::Union{FieldLinker, FieldSymbol}
    here::FieldSymbol{S, T}
end

FieldLinker(here::Union{FieldSymbol, MissSymInt}) = 
FieldLinker(FieldSymbol(), FieldSymbol(here))

FieldLinker(prev::Union{FieldLinker, FieldSymbol}, here::FieldLinker) = 
FieldLinker(FieldLinker(prev, here.prev), here.here)

FieldLinker(prev::FieldLinker{Missing, Any}, here::FieldSymbol) = 
FieldLinker(prev.prev, here)

FieldLinker(entry) = FieldLinker(FieldSymbol(entry), FieldSymbol())

FieldLinker(entry::Union{Symbol, Int}, here::Union{FieldSymbol, MissSymInt}) = 
FieldLinker(FieldSymbol(entry), here)

# Only delete `FieldSymbol{Missing, Any}` at `.here`, as `FieldSymbol{Missing, Any}` at 
# `.prev` is kept as the null symbol.


getField(obj, ::Missing) = itself(obj)

getField(obj, entry::Symbol) = getfield(obj, entry)

getField(obj, entry::Int) = getindex(obj, entry)

getField(obj, entry::FieldSymbol{<:Any, T}) where {T} = getField(obj, entry.entry)

getField(obj, ptr::FieldLinker) = getField(getField(obj, ptr.prev), ptr.here)