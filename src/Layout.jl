abstract type NestedTensorPointer{T, L} <: Any end

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

const IndexTuple = NonEmptyTuple{Union{Int, FirstIndex}}


struct ChainPointer{T, N, C<:Tuple{Vararg{GeneralIndex}}} <: NestedTensorPointer{T, 0}
    chain::C
    type::TensorType{T, N}

    ChainPointer(chain::C, type::TensorType{T, N}=TensorType()) where 
                {T, N, C<:Tuple{Vararg{GeneralIndex}}} = 
    new{T, N, C}(chain, type)
end

const IndexPointer{T, N} = ChainPointer{T, N, Tuple{Int}}

const ChainIndexer{T, N, C<:IndexTuple} = ChainPointer{T, N, C}

ChainPointer(sourceType::TensorType=TensorType()) = ChainPointer((), sourceType)

ChainPointer(entry::GeneralIndex, type::TensorType=TensorType()) = 
ChainPointer((entry,), type)

ChainPointer(prev::GeneralIndex, here::ChainPointer) = 
ChainPointer(ChainPointer(prev), here)

ChainPointer(prev::ChainPointer, here::ChainPointer) = 
ChainPointer((prev.chain..., here.chain...), here.type)

struct AllPassPointer{T, N} <: NestedTensorPointer{T, N} end

const OneLayerAllPass{T} = AllPassPointer{T, 1}


function nestedLevelOf(obj::AbstractArray)
    level = 0
    if eltype(itself.(obj)) <: AbstractArray
        level += 1 + (nestedLevelOf∘first)(obj)
    end
    level
end


getField(ptr) = Base.Fix2(getField, ptr)

const GetField{T, N, C} = Base.Fix2{typeof(getField), ChainPointer{T, N, C}}

const GetIndex{T, N} = GetField{T, N, Tuple{Int}}

getField(obj, ::AllPassPointer) = itself(obj)

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


abstract type FiniteDict{N, K, T} <: AbstractDict{K, T} end


struct SingleEntryDict{K, T} <: FiniteDict{1, K, T}
    key::K
    value::T
end


struct TypedEmptyDict{K, T} <: FiniteDict{0, K, T} end

TypedEmptyDict() = TypedEmptyDict{Union{}, Union{}}()


buildDict(p::Pair{K, T}) where {K, T} = SingleEntryDict(p.first, p.second)

function buildDict(ps::NonEmpTplOrAbtArr{<:Pair}, 
                   emptyBuiler::Type{<:FiniteDict{0}}=TypedEmptyDict)
    isempty(ps) ? emptyBuiler() : Dict(ps)
end

buildDict(::Tuple{}, emptyBuiler::Type{<:FiniteDict{0}}=TypedEmptyDict) = emptyBuiler()

buildDict(emptyBuiler::Type{<:FiniteDict{0}}=TypedEmptyDict) = buildDict((), emptyBuiler)


import Base: isempty, length, collect, keys, values, getindex, iterate

isempty(::SingleEntryDict) = false
isempty(::TypedEmptyDict) = true

length(::FiniteDict{N}) where {N} = N

collect(d::SingleEntryDict) = [d.key => d.value]
collect(::TypedEmptyDict{K, T}) where {K, T} = Pair{K, T}[]

keys(d::SingleEntryDict) = Set( (d.key,) )
keys(::TypedEmptyDict{K}) where {K} = Set{K}()

values(d::SingleEntryDict) = Set( (d.value,) )
values(::TypedEmptyDict{<:Any, T}) where {T} = Set{T}()

function getindex(d::SingleEntryDict{K}, key::K) where {K}
    if key == d.key
        d.value
    else
        throw(KeyError(key))
    end
end

function getindex(::T, key) where {T<:TypedEmptyDict}
    throw(AssertionError("This dictionary (`::$T`) is meant to be empty."))
end

function iterate(d::SingleEntryDict, state::Int)
    if state <= 1
        (d.key  => d.value, 2)
    else
        nothing
    end
end

iterate(::TypedEmptyDict, state::Int) = nothing

iterate(d::FiniteDict) = iterate(d, 1)


struct BlackBox <: QueryBox{Any}
    value::Any
end


function canDirectlyStore(::T) where {T}
    isbitstype(T) || isprimitivetype(T) || issingletontype(T)
end

canDirectlyStore(::Union{String, Type, Symbol}) = true

const DefaultIdentifierCacheSizeLimit = 500

const IdentifierCache = LRU{RefVal{Any}, UInt}(maxsize=DefaultIdentifierCacheSizeLimit)

function backupIdentifier(ref::RefVal{Any})
    val = ref[]
    id = objectid(val)
    LRUCache.setindex!(IdentifierCache, id, ref)
    id, WeakRef(val)
end

function emptyIdentifierCache()
    LRUCache.empty!(IdentifierCache)
    nothing
end

function resizeIdentifierCache(size::Int)
    checkPositivity(size)
    LRUCache.resize!(IdentifierCache, maxsize=size)
    nothing
end

struct Identifier <: IdentityMarker{Any}
    code::UInt
    link::Union{WeakRef, BlackBox}

    function Identifier(obj::Any)
        id, link = if canDirectlyStore(obj)
            objectid(obj), BlackBox(obj)
        else
            backupIdentifier( Ref{Any}(obj) )
        end
        new(id, link)
    end
end


import Base: ==, hash

function ==(id1::Identifier, id2::Identifier)
    code1 = id1.code
    code2 = id2.code
    if code1 == code2
        source1 = id1.link.value
        source2 = id2.link.value
        if code1 != objectid(source1)
            false
        else
            source1 === source2
        end
    else
        false
    end
end

function hash(id::Identifier, hashCode::UInt)
    hashCode = hash(id.code, hashCode)
    hash(objectid(id.link.value), hashCode)
end