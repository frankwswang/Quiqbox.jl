using LRUCache
using Base: issingletontype

struct OneToIndex <: StructuredInfo
    idx::Int

    function OneToIndex(idx::Int)
        checkPositivity(idx)
        new(idx)
    end

    OneToIndex() = new(1)
end

OneToIndex(idx::OneToIndex) = itself(idx)

struct UnitIndex <: StructuredInfo
    idx::OneToIndex
end

UnitIndex(idx::Int) = UnitIndex(idx|>OneToIndex)


struct GridIndex <: StructuredInfo
    idx::OneToIndex
end

GridIndex(idx::Int) = GridIndex(idx|>OneToIndex)


const SpanIndex = Union{UnitIndex, GridIndex}
const GeneralIndex = Union{Int, SpanIndex, OneToIndex, Base.CartesianIndex}
const GeneralField = Union{GeneralIndex, Symbol, Nothing}

struct ChainedAccess{L, C<:NTuple{L, GeneralField}} <: Getter
    chain::C

    ChainedAccess(chain::C) where {L, C<:NTuple{L, GeneralField}} = new{L, C}(chain)
end

const Pass = ChainedAccess{0, Tuple{}}
const GetIndex{T<:GeneralIndex} = ChainedAccess{1, Tuple{T}}
const GetOneToIndex = GetIndex{OneToIndex}

GetIndex{T}(idx::Union{Int, OneToIndex}) where {T<:GeneralIndex} = ChainedAccess(idx|>T)

const GetGridEntry = ChainedAccess{2, Tuple{GridIndex, OneToIndex}}

ChainedAccess() = ChainedAccess(())

ChainedAccess(entry::GeneralField) = ChainedAccess((entry,))

ChainedAccess(prev::ChainedAccess, here::ChainedAccess) = 
ChainedAccess((prev.chain..., here.chain...))

ChainedAccess(prev::GeneralField, here::ChainedAccess) = 
ChainedAccess((prev, here.chain...))

ChainedAccess(prev::ChainedAccess, here::GeneralField) = 
ChainedAccess((prev.chain..., here))

getField(obj, ::Nothing) = getindex(obj)

getField(obj, ::Pass) = itself(obj)

getField(obj, entry::Symbol) = getfield(obj, entry)

getField(obj, entry::Int) = getindex(obj, entry)

getField(obj, i::Base.CartesianIndex) = getindex(obj, i)

getField(obj, i::OneToIndex) = getindex(obj, firstindex(obj)::Int + i.idx - 1)

getField(obj, i::UnitIndex) = getField(obj.unit, i.idx)

getField(obj, i::GridIndex) = getField(obj.grid, i.idx)

function getField(obj, acc::ChainedAccess{L, C}) where {L, C<:NTuple{L, GeneralField}}
    for i in acc.chain
        obj = getField(obj, i)
    end
    obj
end

(f::Encoder)(obj) = getField(obj, f)


abstract type FiniteDict{N, K, T} <: EqualityDict{K, T} end


struct SingleEntryDict{K, T} <: FiniteDict{1, K, T}
    key::K
    value::T
end


struct TypedEmptyDict{K, T} <: FiniteDict{0, K, T} end

TypedEmptyDict() = TypedEmptyDict{Union{}, Union{}}()


buildDict(p::Pair{K, T}) where {K, T} = SingleEntryDict(p.first, p.second)

function buildDict(ps::Union{Tuple{Vararg{Pair}}, AbstractVector{<:Pair}}, 
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

collect(d::SingleEntryDict{K, T}) where {K, T} = Memory{Pair{K, T}}([d.key => d.value])
collect(::TypedEmptyDict{K, T}) where {K, T} = Memory{Pair{K, T}}(undef, 0)

function get(d::SingleEntryDict{K}, key::K, default::Any) where {K}
    ifelse(key == d.key, d.value, default)
end

keys(d::SingleEntryDict) = Set((d.key,))
keys(::TypedEmptyDict{K}) where {K} = Set{K}()

values(d::SingleEntryDict) = Set((d.value,))
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
        (d.key => d.value, 2)
    else
        nothing
    end
end

iterate(::TypedEmptyDict, state::Int) = nothing

iterate(d::FiniteDict) = iterate(d, 1)


struct EgalBox{T} <: QueryBox{T}
    value::T
end

const BlackBox = EgalBox{Any}

==(bb1::EgalBox, bb2::EgalBox) = (bb1.value === bb2.value)

hash(bb::EgalBox, hashCode::UInt) = hash(objectid(bb.value), hashCode)


struct TypeBox{T} <: QueryBox{Type{T}}
    value::Type{T}
end

==(::TypeBox{T1}, ::EgalBox{T2}) where {T1, T2} = (T1 <: T2) && (T2 <: T1)

hash(::TypeBox{T}, hashCode::UInt) where {T} = hash(objectid(T), hashCode)


function canDirectlyStoreInstanceOf(::Type{T}) where {T}
    isbitstype(T) || isprimitivetype(T) || issingletontype(T)
end

canDirectlyStoreInstanceOf(::Type{Symbol}) = true

canDirectlyStoreInstanceOf(::Type{String}) = true

canDirectlyStoreInstanceOf(::Type{<:IdentityMarker}) = true

canDirectlyStore(::T) where {T} = canDirectlyStoreInstanceOf(T)

canDirectlyStore(::Type) = true

const DefaultIdentifierCacheSizeLimit = 500

const IdentifierCache = LRU{BlackBox, RefVal{Any}}(maxsize=DefaultIdentifierCacheSizeLimit)

function backupIdentifier(ref::BlackBox)
    LRUCache.get!(IdentifierCache, ref) do
        Ref{Any}(ref.value)
    end
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


function leftFoldHash(initHash::UInt, objs::Union{AbstractArray, Tuple}; 
                      marker::F=itself) where {F<:Function}
    init = hash(sizeOf(objs), initHash)
    hashFunc = (x::UInt, y) -> leftFoldHash(x, y; marker)
    foldl(hashFunc, objs; init)
end

leftFoldHash(initHash::UInt, obj::Any; marker::F=itself) where {F<:Function} = 
hash(marker(obj), initHash)


struct ValueMarker{T} <: IdentityMarker{T}
    code::UInt
    data::T

    ValueMarker(input::T) where {T} = new{T}(hash(input), input)
end

function ==(marker1::ValueMarker, marker2::ValueMarker)
    if marker1.code == marker2.code
        marker1.data == marker2.data
    else
        false
    end
end

const IdMarkerPair{M<:IdentityMarker} = Pair{Symbol, M}

const ValMkrPair{T} = IdMarkerPair{ValueMarker{T}}

struct FieldMarker{S, N} <: IdentityMarker{S}
    code::UInt
    data::NTuple{N, IdMarkerPair}

    function FieldMarker(input::T) where {T}
        fieldSyms = fieldnames(T)
        issingletontype(T) && (return ValueMarker(input))
        markers = map(fieldSyms) do sym
            getfield(input, sym) |> markObj
        end
        inputName = nameof(T)
        data = map(=>, fieldSyms, markers)
        new{inputName, length(fieldSyms)}(leftFoldHash(hash(inputName), markers), data)
    end
end

function ==(marker1::FieldMarker{S}, marker2::FieldMarker{S}) where {S}
    if marker1.code == marker2.code
        marker1.data == marker2.data
    else
        false
    end
end

struct BlockMarker <: IdentityMarker{Union{AbstractArray, Tuple}}
    code::UInt
    data::Union{AbstractArray, Tuple}

    function BlockMarker(input::Union{AbstractArray, Tuple})
        containerHash = hash(input isa Tuple ? Tuple : AbstractArray)
        code = mapfoldl(leftFoldHash, input, init=containerHash) do ele
            markObj(ele)
        end
        new(code, input)
    end
end

function ==(marker1::BlockMarker, marker2::BlockMarker)
    if marker1.code == marker2.code
        data1 = marker1.data
        data2 = marker2.data
        if data1 === data2
            true
        else
            isSame = true
            for (i, j) in zip(data1, data2)
                isSame = ( i===j || markObj(i) == markObj(j) )
                isSame || break
            end
            isSame
        end
    else
        false
    end
end


function isPrimVarCollection(arg::AbstractArray{T}) where {T}
    ET = isconcretetype(T) ? T : eltype( map(itself, arg) )
    canDirectlyStoreInstanceOf(ET)
end

function isPrimVarCollection(arg::Tuple)
    all(canDirectlyStore(i) for i in arg)
end


function markObj(input::Type)
    ValueMarker(input)
end

function markObj(input::Union{AbstractArray, Tuple})
    isPrimVarCollection(input) ? ValueMarker(input) : BlockMarker(input)
end

function markObj(input::AbstractEqualityDict)
    pairs = collect(input)
    ks = map(Base.Fix2(getfield, :first), pairs)
    vs = map(pairs) do pair
        markObj(pair.second)
    end
    newDict = (Dict∘map)(=>, ks, vs)
    ValueMarker(newDict)
end

function markObj(input::AbstractDict)
    ValueMarker(input)
end

function markObj(input::T) where {T}
    if canDirectlyStore(input)
        ValueMarker(input)
    elseif isstructtype(T) && !issingletontype(T)
        FieldMarker(input)
    else
        Identifier(input)
    end
end

markObj(marker::IdentityMarker) = itself(marker)


function lazyMarkObj!(cache::AbstractDict{EgalBox{T}, <:IdentityMarker}, input) where {T}
    get!(cache, EgalBox{T}(input)) do
        markObj(input)
    end
end


==(pm1::IdentityMarker, pm2::IdentityMarker) = false

function hash(id::IdentityMarker, hashCode::UInt)
    hash(id.code, hashCode)
end

function compareObj(obj1::T1, obj2::T2) where {T1, T2}
    obj1 === obj2 || markObj(obj1) == markObj(obj2)
end


struct NullCache{T} <: CustomCache{T} end