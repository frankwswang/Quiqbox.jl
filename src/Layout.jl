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


struct ChainPointer{A<:TensorType, L, C<:NTuple{L, GeneralFieldName}} <: EntryPointer
    chain::C
    type::A

    ChainPointer(chain::C, type::A=TensorType()) where 
                {A<:TensorType, L, C<:NTuple{L, GeneralFieldName}} = 
    new{A, L, C}(chain, type)
end

const AllPassPointer{A<:TensorType} = ChainPointer{A, 0, Tuple{}}

const TypedPointer{T, A<:TensorType{T}} = ChainPointer{A}

const IndexPointer{A<:TensorType} = ChainPointer{A, 1, Tuple{Int}}

const PointPointer{T, L, C<:NTuple{L, GeneralFieldName}} = ChainPointer{Flavor{T}, L, C}

const ArrayPointer{T, L, C<:NTuple{L, GeneralFieldName}} = ChainPointer{Volume{T}, L, C}

const ChainIndexer{A<:TensorType, L, C<:NTuple{L, Union{Int, FirstIndex, Nothing}}} = 
      ChainPointer{A, L, C}

ChainPointer(sourceType::TensorType=TensorType()) = ChainPointer((), sourceType)

ChainPointer(entry::GeneralFieldName, type::TensorType=TensorType()) = 
ChainPointer((entry,), type)

ChainPointer(prev::ChainPointer, here::ChainPointer) = 
ChainPointer((prev.chain..., here.chain...), here.type)

ChainPointer(prev::GeneralFieldName, here::ChainPointer) = 
ChainPointer((prev, here.chain...), here.type)

struct ChainFilter{L, U, C<:Tuple{Vararg{PointerStack{L, U}}}} <: PointerStack{L, U}
    chain::C

    ChainFilter(::Tuple{}=()) = new{0, 0, Tuple{}}( () )

    ChainFilter(chain::NonEmptyTuple{PointerStack{L, U}}) where {L, U} = 
    new{L, U, typeof(chain)}(chain)
end

const AllPassFilter = ChainFilter{0, 0, Tuple{}}

const SingleFilter{L, U, C<:PointerStack{L, U}} = ChainFilter{L, U, Tuple{C}}

ChainFilter(prev::ChainFilter, here::ChainFilter) = 
ChainFilter((prev.chain..., here.chain...))

ChainFilter(prev::PointerStack, here::ChainFilter) = 
ChainFilter((prev, here.chain...))

ChainFilter(prev::ChainFilter, here::PointerStack) = 
ChainFilter((prev.chain..., here))

ChainFilter(prev::PointerStack, here::PointerStack) = ChainFilter((prev, here))

ChainFilter(obj::ChainFilter) = itself(obj)
ChainFilter(obj::PointerStack) = ChainFilter((obj,))


struct AwaitFilter{P<:PointerStack} <: StaticPointer
    ptr::P
end

AwaitFilter(ptr::AwaitFilter) = itself(ptr)


struct FilteredObject{T, P<:ChainFilter} <: QueryBox{T}
    obj::T
    ptr::P
end

FilteredObject(obj::FilteredObject, ptr::ChainFilter) = 
FilteredObject(obj.obj, ChainFilter(obj.ptr, ptr))

FilteredObject(obj, ptr::PointerStack) = FilteredObject(obj, ChainFilter(ptr))

FilteredObject(obj, ptr::AwaitFilter) = FilteredObject(obj, ptr.ptr)


getFieldCore(obj, ::AllPassPointer) = itself(obj)

getFieldCore(obj, ::FirstIndex) = first(obj)

getFieldCore(obj, entry::Symbol) = getfield(obj, entry)

getFieldCore(obj, entry::Int) = getindex(obj, entry)

getFieldCore(obj, ::Nothing) = getindex(obj)

getFieldCore(obj, ptr::ChainPointer) = foldl(getFieldCore, ptr.chain, init=obj)

getField(obj, ptr::GeneralFieldName) = getFieldCore(obj, ptr)

getField(obj, ptr::EntryPointer) = getFieldCore(obj, ptr)

function getField(obj::FilteredObject, ptr::EntryPointer)
    getFieldCore(obj.obj, getField(obj.ptr, ptr))
end

function getField(scope::ChainFilter, ptr::EntryPointer)
    for i in reverse(scope.chain)
        ptr = getField(i, ptr)
    end
    ptr
end


function filterObjectCore(obj, ptr::ChainFilter)
    evalField(itself, obj, ptr)
end

function filterObjectCore(obj, ptr::PointerStack)
    map(x->getField(obj, x), ptr)
end

getField(obj, ptr::PointerStack) = filterObjectCore(obj, ptr)

function getField(scope::PointerStack, ptr::PointerStack)
    ChainFilter(scope, ptr)
end

function getField(obj::FilteredObject, ptr::PointerStack)
    FilteredObject(obj.obj, ChainFilter(obj.ptr, ptr))
end

function getField(obj, ptr::AwaitFilter)
    FilteredObject(obj, ptr)
end

function getField(obj, ptrs::NonEmpTplOrAbtArr{<:ActivePointer})
    map(x->getField(obj, x), ptrs)
end

getField(obj::Any) = itself(obj)

getField(obj::FilteredObject) = getField(obj.obj, obj.ptr)


function evalField(f::F, obj, ptr::EntryPointer) where {F<:Function}
    getField(obj, ptr) |> f
end

function evalField(f::F, obj, ptr::PointerStack) where {F<:Function}
    evalFieldCore(f, obj, ChainFilter(ptr))
end

function evalField(f::F, obj::FilteredObject, ptr::PointerStack) where {F<:Function}
    res = getField(obj, ptr)
    evalFieldCore(f, res.obj, res.ptr)
end

function evalFieldCore(f::F, obj, ptr::ChainFilter) where {F<:Function}
    body..., tip = ptr.chain
    scope = ChainFilter(body)
    map(tip) do idx
        evalFieldCore(f, obj, scope, idx)
    end
end

function evalFieldCore(f::F, obj, ::AllPassFilter) where {F<:Function}
    f(obj)
end

function evalFieldCore(f::F, obj, scope::PointerStack, 
                       ptr::ActivePointer) where {F<:Function}
    evalField(f, obj, getField(scope, ptr))
end

function evalFieldCore(f::F, obj, scope::PointerStack, 
                       ptrs::NonEmpTplOrAbtArr{<:ActivePointer}) where {F<:Function}
    map(ptrs) do ptr
        evalField(f, obj, getField(scope, ptr))
    end
end


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