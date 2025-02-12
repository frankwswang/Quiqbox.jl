using LRUCache


getMemory(arr::Memory) = itself(arr)

function getMemory(arr::AbstractArray{T}) where {T}
    eleT = if isconcretetype(T) || isempty(arr)
        T
    else
        mapreduce(typejoin, arr, init=Union{}) do ele
            typeof(ele)
        end
    end
    Memory{eleT}(vec(arr))
end

function getMemory(obj::NonEmptyTuple{Any})
    mem = Memory{eltype(obj)}(undef, length(obj))
    mem .= obj
    mem
end

getMemory(obj::Any) = getMemory((obj,))


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

TensorType(p::ParamBox{T, N}) where {T, N} = 
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

const IndexPointer{A<:TensorType, N} = ChainPointer{A, N, NTuple{N, Int}}

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


struct AwaitFilter{P<:PointerStack} <: StaticPointer{P}
    ptr::P
end

AwaitFilter(ptr::AwaitFilter) = itself(ptr)

struct FilteredObject{T, P<:PointerStack} <: ViewedObject{T, P}
    obj::T
    ptr::P
end

FilteredObject(obj::FilteredObject, ptr::PointerStack) = 
FilteredObject(obj.obj, getField(obj.ptr, ptr))

FilteredObject(obj, ptr::AwaitFilter) = FilteredObject(obj, ptr.ptr)

const Filtered{T} = Union{T, FilteredObject{T}}
const FilteredVecOfArr{T} = Filtered{<:AbtVecOfAbtArr{T}}


getFieldCore(obj, entry::Int) = getindex(obj, entry)

getFieldCore(obj, entry::Symbol) = getfield(obj, entry)

getFieldCore(obj, ::FirstIndex) = first(obj)

getFieldCore(obj, ::Nothing) = getindex(obj)

getFieldCore(obj, ::AllPassPointer) = itself(obj)

getFieldCore(obj, ptr::ChainPointer) = foldl(getField, ptr.chain, init=obj)

const GeneralEntryPointer = Union{EntryPointer, GeneralFieldName}

getField(obj, ptr::GeneralEntryPointer) = getFieldCore(obj, ptr)

getField(ptr::ChainPointer, idx::GeneralFieldName) = ChainPointer(ptr, ChainPointer(idx))

function getField(obj::FilteredObject, ptr::GeneralEntryPointer)
    getField(obj.obj, getField(obj.ptr, ptr))
end

const InstantPtrCollection = Union{PointerStack, NonEmpTplOrAbtArr{ActivePointer}}

getField(obj, ptr::InstantPtrCollection) = evalField(itself, obj, ptr)

function getField(obj::FilteredObject, ptr::PointerStack)
    FilteredObject(obj, ptr)
end

function getField(obj, ptr::AwaitFilter)
    FilteredObject(obj, ptr)
end

function getField(prev::PointerStack, here::AwaitFilter)
    AwaitFilter(getField(prev, here.ptr))
end

function getField(prev::AwaitFilter, here::PointerStack)
    AwaitFilter(getField(prev.ptr, here))
end

getField(obj::Any) = itself(obj)

getField(obj::FilteredObject) = getField(obj.obj, obj.ptr)


function evalField(f::F, obj, ptr::GeneralEntryPointer) where {F<:Function}
    getField(obj, ptr) |> f
end

function evalField(f::F, obj, ptr::InstantPtrCollection) where {F<:Function}
    map(x->evalField(f, obj, x), ptr)
end


function mapLayout(op::F, collection::Any) where {F<:Function}
    map(op, collection)
end

function mapLayout(op::F, collection::FilteredObject) where {F<:Function}
    evalField(op, collection.obj, collection.ptr)
end


abstract type FiniteDict{N, K, T} <: EqualityDict{K, T} end


struct SingleEntryDict{K, T} <: FiniteDict{1, K, T}
    key::K
    value::T
end


struct TypedEmptyDict{K, T} <: FiniteDict{0, K, T} end

TypedEmptyDict() = TypedEmptyDict{Union{}, Union{}}()


buildDict(p::Pair{K, T}) where {K, T} = SingleEntryDict(p.first, p.second)

function buildDict(ps::NonEmpTplOrAbtArr{Pair}, 
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


struct BlackBox <: QueryBox{Any}
    value::Any
end

==(bb1::BlackBox, bb2::BlackBox) = (bb1.value === bb2.value)

hash(bb::BlackBox, hashCode::UInt) = hash(objectid(bb.value), hashCode)


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


struct ElementWiseMatcher{F<:Function, T<:AbstractArray} <: IdentityMarker{T}
    code::UInt
    marker::F
    data::T

    function ElementWiseMatcher(input::T, marker::F=itself) where 
                               {T<:AbstractArray, F<:Function}
        new{F, T}(leftFoldHash(hash(nothing), input; marker), marker, input)
    end
end

function elementWiseMatch(obj1::Any, obj2::Any; marker1::M1=itself, marker2::M2=itself, 
                          compareFunction::F=(===)) where {M1<:Function, M2<:Function, 
                                                           F<:Function}
    compareFunction(marker1(obj1), marker2(obj2))
end

function elementWiseMatch(obj1::AbstractArray{<:Any, N}, obj2::AbstractArray{<:Any, N}; 
                          marker1::M1=itself, marker2::M2=itself, 
                          compareFunction::F=(===)) where {N, M1<:Function, M2<:Function, 
                                                           F<:Function}
    if size(obj1) == size(obj1)
        all( elementWiseMatch(marker1(i), marker2(j); 
                              compareFunction) for (i, j) in zip(obj1, obj2) )
    else
        false
    end
end

function elementWiseMatch(::AbstractArray, ::AbstractArray; marker1::M1=itself, 
                          marker2::M2=itself, compareFunction::F=(===)) where 
                         {M1<:Function, M2<:Function, F<:Function}
    false
end

function ==(matcher1::ElementWiseMatcher, matcher2::ElementWiseMatcher)
    if matcher1.code == matcher2.code
        elementWiseMatch(matcher1.data, matcher2.data, compareFunction=isequal, 
                         marker1=matcher1.marker, marker2=matcher2.marker)
    else
        false
    end
end


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
        propertySyms = propertynames(input)
        if issingletontype(T) || isempty(propertySyms)
            return ValueMarker(input)
        end
        markers = map(propertySyms) do sym
            getproperty(input, sym) |> markObj
        end
        inputName = nameof(T)
        data = map(=>, propertySyms, markers)
        new{inputName, length(propertySyms)}(leftFoldHash(hash(inputName), markers), data)
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
    if isstructtype(T) && !issingletontype(T)
        FieldMarker(input)
    elseif canDirectlyStore(input)
        ValueMarker(input)
    else
        Identifier(input)
    end
end

markObj(marker::IdentityMarker) = itself(marker)


function lazyMarkObj!(cache::AbstractDict{BlackBox, <:IdentityMarker}, input)
    get!(cache, BlackBox(input)) do
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