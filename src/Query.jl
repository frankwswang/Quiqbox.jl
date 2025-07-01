using LRUCache
using Base: issingletontype

struct OneToIndex <: CustomAccessor
    idx::Int

    function OneToIndex(idx::Int, ::V=True()) where {V<:Boolean} #! Remove second argument
        getTypeValue(V) && checkPositivity(idx)
        new(idx)
    end

    function OneToIndex(idx::OneToIndex, ::Count{N}) where {N}
        new(idx.idx + N)
    end

    OneToIndex() = new(1)
end

OneToIndex(idx::OneToIndex) = itself(idx)

+(idx::OneToIndex, i::Integer) = OneToIndex(idx.idx + i)
+(i::Integer, idx::OneToIndex) = OneToIndex(idx.idx + i)
-(idx::OneToIndex, i::Integer) = OneToIndex(idx.idx - i)

Int(idx::OneToIndex) = getfield(idx, :idx)


struct PointEntry <: CustomAccessor end #> For getindex(obj) implementation

struct UnitSector <: CustomAccessor end
struct GridSector <: CustomAccessor end
const SpanSector = Union{UnitSector, GridSector}

const LinearAccessor = Union{PointEntry, Int, OneToIndex}
const SimpleAccessor = Union{LinearAccessor, Base.CartesianIndex, SpanSector, Symbol}
const AbstractAccessor = Union{SimpleAccessor, CustomAccessor}

struct ChainedAccess{C<:Tuple{ Vararg{SimpleAccessor} }} <: CustomAccessor
    chain::C
end

const DirectAccess{T<:SimpleAccessor} = ChainedAccess{Tuple{T}}
const AllPassAccess = ChainedAccess{Tuple{}}

ChainedAccess() = ChainedAccess(())

ChainedAccess(entry::SimpleAccessor) = ChainedAccess((entry,))

ChainedAccess(prev::ChainedAccess, here::ChainedAccess) = 
ChainedAccess((prev.chain..., here.chain...))

ChainedAccess(prev::SimpleAccessor, here::ChainedAccess) = 
ChainedAccess((prev, here.chain...))

ChainedAccess(prev::ChainedAccess, here::SimpleAccessor) = 
ChainedAccess((prev.chain..., here))

Base.broadcastable(c::ChainedAccess) = Ref(c)


getEntry(obj, ::PointEntry) = getindex(obj)

getEntry(obj, entry::Int) = getindex(obj, entry)

getEntry(obj::GeneralCollection, i::OneToIndex) = getindex(obj, shiftLinearIndex(obj, i))

getEntry(obj, i::Base.CartesianIndex) = getindex(obj, i)

getEntry(obj, ::UnitSector) = obj.unit

getEntry(obj, ::GridSector) = obj.grid

getEntry(obj, entry::Symbol) = getproperty(obj, entry)

getEntry(obj, ::AllPassAccess) = itself(obj)

function getEntry(obj, acc::DirectAccess)
    getEntry(obj, first(acc.chain))
end

function getEntry(obj, acc::ChainedAccess)
    head, body = acc.chain
    temp = getEntry(obj, head)
    getEntry(temp, ChainedAccess(body))
end


struct TypedEmptyDict{K, V} <: EqualityDict{K, V} end

TypedEmptyDict() = TypedEmptyDict{Union{}, Union{}}()

length(::TypedEmptyDict) = 0

collect(::TypedEmptyDict{K, V}) where {K, V} = Memory{Pair{K, V}}(undef, 0)

haskey(::TypedEmptyDict{K}, ::K) where {K} = false

get(::TypedEmptyDict{K}, ::K, default::Any) where {K} = itself(default)

get!(::TypedEmptyDict{K, V}, ::K, default::V) where {K, V} = itself(default)

setindex!(d::TypedEmptyDict{K, V}, ::V, ::K) where {K, V} = itself(d)

function getindex(::T, key) where {T<:TypedEmptyDict}
    throw(AssertionError("This dictionary (`::$T`) is meant to be empty."))
end

iterate(::TypedEmptyDict, ::Int) = nothing
iterate(d::TypedEmptyDict) = iterate(d, 1)


struct IndexDict{K, T} <: EqualityDict{K, T}
    indexer::Dict{K, OneToIndex}
    storage::Vector{Pair{K, T}}

    function IndexDict{K, T}() where {K, T}
        (K <: OneToIndex) && throw(ArgumentError("K::Type{$K} cannot be `OneToIndex`."))
        new{K, T}(Dict{K, OneToIndex}(), Pair{K, T}[])
    end
end

length(d::IndexDict) = length(d.storage)

collect(d::IndexDict) = copy(d.storage)

function haskey(d::IndexDict{K}, key::K) where {K}
    haskey(d.indexer, key)
end

function get(d::IndexDict{K, V}, key::K, default::V) where {K, V}
    res = get(d.indexer, key, nothing)
    res === nothing ? default : getEntry(d.storage, res).second
end

function get!(d::IndexDict{K, V}, key::K, default::V) where {K, V}
    index = get(d.indexer, key, nothing)
    if index === nothing
        d.indexer[key] = OneToIndex(length(d.storage) + 1)
        push!(d.storage, key=>default)
        default
    else
        getEntry(d.storage, index).second
    end
end

function getindex(d::IndexDict{K}, accessor) where {K}
    index = if accessor isa OneToIndex
        accessor
    else
        keyType = typeof(accessor)
        (keyType <: K) || throw(AssertionError("accessor::$keyType is not a valid key."))
        getindex(d.indexer, accessor)
    end
    getEntry(d.storage, index).second
end

function setindex!(d::IndexDict{K, V}, value::V, accessor) where {K, V}
    pairs = d.storage

    if accessor isa OneToIndex
        localIndex = shiftLinearIndex(values, accessor)
        key, _ = getindex(pairs, localIndex)
        setindex!(pairs, key=>value, localIndex)
    else
        index = if haskey(d.indexer, accessor)
            localIndex = shiftLinearIndex(pairs, getindex(d.indexer, accessor))
            setindex!(pairs, accessor=>value, localIndex)
        else
            push!(pairs, accessor=>value)
            (OneToIndex∘length)(pairs)
        end
        setindex!(d.indexer, index, accessor)
    end

    d
end

iterate(d::IndexDict, state) = iterate(d.storage, state)
iterate(d::IndexDict) = iterate(d.storage)

indexKey(d::IndexDict, index::OneToIndex) = getindex(d.storage, index).first

keyIndex(d::IndexDict{K}, key::K) where {K} = getindex(d.indexer, key)


struct EgalBox{T} <: QueryBox{T}
    value::T
end

const BlackBox = EgalBox{Any}

==(bb1::EgalBox, bb2::EgalBox) = (bb1.value === bb2.value)

function hash(bb::EgalBox, hashCode::UInt)
    hash(objectid(bb.value), hash(typeof(bb), hashCode))
end


struct TypeBox{T} <: QueryBox{Type{T}}
    value::Type{T}

    TypeBox(::Type{T}) where {T} = new{T::Type}()
end

TypeBox(::Type{Union{}}) = 
throw(AssertionError("`TypeBox` cannot be instantiated with `Union{}`."))

==(::TypeBox{T1}, ::TypeBox{T2}) where {T1, T2} = (T1 <: T2) && (T2 <: T1)

function hash(::TypeBox{T}, hashCode::UInt) where {T}
    hash(hash(T), hash(TypeBox, hashCode))
end


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

    function Identifier(obj::Any=nothing)
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


struct BottomArrayMarker{N} <: IdentityMarker{AbstractArray{Union{}, N}}
    code::UInt
    data::NTuple{N, Int}

    function BottomArrayMarker(input::AbstractArray{Union{}, N}) where {N}
        shape = size(input)
        new{N}(hash(shape), shape)
    end
end

function ==(marker1::BottomArrayMarker, marker2::BottomArrayMarker)
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
            markObj(getfield(input, sym))
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
    if input isa AbstractArray && eltype(input) == Union{}
        BottomArrayMarker(input)
    else
        isPrimVarCollection(input) ? ValueMarker(input) : BlockMarker(input)
    end
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


mutable struct AtomicUnit{T} <: QueryBox{T}
    @atomic value::T

    function AtomicUnit(input::T) where {T}
        new{T}(input)
    end
end

struct AtomicGrid{E<:AbstractMemory} <: QueryBox{E}
    value::E
end


const AtomicLocker{T} = Union{AtomicUnit{T}, AtomicGrid{T}}

==(ab1::AtomicLocker, ab2::AtomicLocker) = (ab1.value == ab2.value)

function hash(ab::AtomicLocker, hashCode::UInt)
    hash(objectid(ab.value), hash(typeof(ab), hashCode))
end

getindex(l::AtomicLocker) = l.value


atomicEval(obj) = itself(obj)

atomicEval(box::AtomicLocker) = box[]


function safelySetVal!(box::AtomicUnit, val)
    @atomic box.value = val
end

function safelySetVal!(box::AtomicGrid, val)
    safelySetVal!(box.value, val)
end

function safelySetVal!(box::AbstractArray, val)
    lk = ReentrantLock()
    lock(lk) do
        box .= val
    end
    box
end

setindex!(al::AtomicLocker, val) = safelySetVal!(al, val)


struct MemoryPair{L, R} <: QueryBox{Pair{L, R}}
    left::Memory{L}
    right::Memory{R}

    function MemoryPair(left::AbstractVector{L}, right::AbstractVector{R}) where {L, R}
        if length(left) != length(right)
            throw(AssertionError("`left` and `right` should have the same length."))
        end

        lData = extractMemory(left)
        rData = extractMemory(right)
        new{eltype(lData), eltype(rData)}(lData, rData)
    end
end

function iterate(mp::MemoryPair)
    res = iterate(zip(mp.left, mp.right))
    if res === nothing
        nothing
    else
        (l, r), state = res
        (l=>r), state
    end
end

function iterate(mp::MemoryPair, state)
    res = iterate(zip(mp.left, mp.right), state)
    if res === nothing
        nothing
    else
        (l, r), state = res
        (l=>r), state
    end
end

length(mp::MemoryPair) = length(mp.left)

eltype(::MemoryPair{L, R}) where {L, R} = Pair{L, R}