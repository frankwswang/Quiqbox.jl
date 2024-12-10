using LRUCache

struct Flavor{T} <: StructuredType end

struct Volume{T} <: StructuredType
    axis::NonEmptyTuple{Int}

    Volume{T}(axis::NonEmptyTuple{Int}) where {T} = new{T}(axis)
end

const TensorType{T} = Union{Flavor{T}, Volume{T}}

TensorType(::Type{T}=Any) where {T} = Flavor{T}()

TensorType(::Type{T}, axis::NonEmptyTuple{Int}) where {T} = Volume{T}(axis)

TensorType(t::TensorType) = itself(t)

TensorType(::AbstractArray{T, 0}) where {T} = Flavor{T}()

TensorType(a::AbstractArray{T}) where {T} = Volume{T}(size(a))

TensorType(::ElementalParam{T}) where {T} = Flavor{T}()

TensorType(p::FlattenedParam{T}) where {T} = Volume{T}(outputSizeOf(p))

TensorType(p::JaggedParam{T, N}) where {T, N} = 
Volume{AbstractArray{T, N}}(outputSizeOf(p))


struct FirstIndex <: StructuredType end

const GeneralFieldName = Union{Int, Symbol, FirstIndex, Nothing}


struct ChainPointer{A<:TensorType, L, C<:NTuple{L, GeneralFieldName}} <: NestedPointer{L, L}
    chain::C
    type::A

    ChainPointer(chain::C, type::A=TensorType()) where 
                {A<:TensorType, L, C<:NTuple{L, GeneralFieldName}} = 
    new{A, L, C}(chain, type)
end

const AllPassPtr{A<:TensorType} = ChainPointer{A, 0, Tuple{}}

const TypedPointer{T, A<:TensorType{T}} = ChainPointer{A}

const IndexPointer{A<:TensorType} = ChainPointer{A, 1, Tuple{Int}}

const PointPointer{T, L, C<:NTuple{L, GeneralFieldName}} = ChainPointer{Flavor{T}, L, C}

const ArrayPointer{T, L, C<:NTuple{L, GeneralFieldName}} = ChainPointer{Volume{T}, L, C}

const ChainIndexer{A<:TensorType, L, C<:NTuple{L, Union{Int, FirstIndex, Nothing}}} = 
      ChainPointer{A, L, C}

ChainPointer(sourceType::TensorType=TensorType()) = ChainPointer((), sourceType)

ChainPointer(entry::GeneralFieldName, type::TensorType=TensorType()) = 
ChainPointer((entry,), type)


linkPointer(prev::ChainPointer, here::ChainPointer) = 
ChainPointer((prev.chain..., here.chain...), here.type)

linkPointer(prev::Union{GeneralFieldName, NonEmptyTuple{GeneralFieldName}}, 
            here::ChainPointer) = 
linkPointer(ChainPointer(prev), here)


function nestedLevelOf(obj::AbstractArray)
    level = 0
    if eltype(itself.(obj)) <: AbstractArray
        level += 1 + (nestedLevelOf∘first)(obj)
    end
    level
end


getField(ptr) = Base.Fix2(getField, ptr)

const GetField{A<:TensorType, L, C<:NTuple{L, GeneralFieldName}} = 
      Base.Fix2{typeof(getField), ChainPointer{A, L, C}}

const GetScalarIdx{T} = GetField{Flavor{T}, 1, Tuple{Int}}

getField(obj, ::AllPassPtr) = itself(obj)

getField(obj, ::FirstIndex) = first(obj)

getField(obj, entry::Symbol) = getfield(obj, entry)

getField(obj, entry::Int) = getindex(obj, entry)

getField(obj, ::Nothing) = getindex(obj)

getField(obj, ptr::ChainPointer) = foldl(getField, ptr.chain, init=obj)

getField(obj::AbstractDict, ptr::ChainPointer) = getindex(obj, ptr)


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
    if isempty(ps)
        emptyBuiler()
    elseif length(ps) == 1
        buildDict(ps|>first)
    else
        Dict(ps)
    end
end

buildDict(::Tuple{}, emptyBuiler::Type{<:FiniteDict{0}}=TypedEmptyDict) = emptyBuiler()

buildDict(emptyBuiler::Type{<:FiniteDict{0}}=TypedEmptyDict) = buildDict((), emptyBuiler)


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

==(bb1::BlackBox, bb2::BlackBox) = (bb1.value === bb2.value)

hash(bb::BlackBox, hashCode::UInt) = hash(objectid(bb.value), hashCode)


function canDirectlyStore(::T) where {T}
    isbitstype(T) || isprimitivetype(T) || issingletontype(T)
end

canDirectlyStore(::Union{String, Type, Symbol}) = true

const DefaultIdentifierCacheSizeLimit = 500

const IdentifierCache = LRU{BlackBox, RefVal{Any}}(maxsize=DefaultIdentifierCacheSizeLimit)

function backupIdentifier(ref::BlackBox)
    LRUCache.get!(IdentifierCache, ref, Ref{Any}(ref.value))
    WeakRef(IdentifierCache[ref][])
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

function deleteIdentifierCacheFor(obj::Any)
    key = BlackBox(obj)
    if haskey(IdentifierCache, key)
        LRUCache.delete!(IdentifierCache, key)
    end
    nothing
end

struct Identifier <: IdentityMarker{Any}
    code::UInt
    link::Union{WeakRef, BlackBox}

    function Identifier(obj::Any)
        link = if canDirectlyStore(obj)
            BlackBox(obj)
        else
            backupIdentifier(BlackBox(obj))
        end
        new(objectid(obj), link)
    end
end

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