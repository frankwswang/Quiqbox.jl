export VectorMemory, ShapedMemory

struct NestedLevel{T}
    level::Int

    function NestedLevel(::Type{T}, level::Int=0) where {T}
        checkPositivity(level, true)
        new{T}(level)
    end
end

getNestedLevelCore(::Type{T}, level::Int) where {T} = (T, level)

function getNestedLevelCore(::Type{T}, level::Int) where {T<:AbstractArray}
    getNestedLevelCore(eltype(T), level+1)
end

getNestedLevel(::Type{T}) where {T} = NestedLevel(getNestedLevelCore(T, 0)...)


getCoreType(::NestedLevel{T}) where {T} = T

getCoreType(::Type{T}) where {T} = T

getCoreType(::Type{E}) where {T, E<:AbstractArray{T}} = getCoreType(T)


function getPackType(coreType::Type{T}, level::Int) where {T}
    res = coreType
    for _ in 1:level
        res = ifelse(isconcretetype(res), AbstractArray{res}, AbstractArray{<:res})
    end
    res
end

getPackType(l::NestedLevel{T}) where {T} = getPackType(T, l.level)

function getPackType(::Type{<:AbstractArray{T, N}}) where {T, N}
    innerT = getPackType(T)
    if isconcretetype(innerT)
        AbstractArray{innerT, N}
    else
        AbstractArray{<:innerT, N}
    end
end

getPackType(::Type{T}) where {T} = T


function checkReshapingAxis(shape::Tuple{Vararg{Int}})
    if any(i < 0 for i in shape)
        throw(AssertionError("All axis sizes should be non-negative."))
    end
    nothing
end


function checkReshapingAxis(arr::AbstractArray, shape::Tuple{Vararg{Int}})
    checkReshapingAxis(shape)
    len  = length(arr)
    if prod(shape) != len
        throw(AssertionError("The product of reshaping axes should be equal to the "*
                             "target array's length."))
    end
    len
end

struct TruncateReshape{N}
    axis::NTuple{N, Int}
    mark::NTuple{N, Symbol}
    truncate::TernaryNumber # 0: off, 1: keep leading entires, 2: keep trailing entires

    function TruncateReshape(axis::NonEmptyTuple{Int, N}, 
                             mark::NonEmptyTuple{Symbol, N}=ntuple( _->:e, Val(N+1) ); 
                             truncate::Union{Bool, TernaryNumber}=false) where {N}
        checkReshapingAxis(axis)
        new{N+1}(axis, mark, TernaryNumber(truncate|>Int))
    end

    function TruncateReshape(refArr::AbstractArray{T, N}, 
                             mark::NTuple{N, Symbol}=ntuple( _->:e, Val(N) ); 
                             truncate::Union{Bool, TernaryNumber}=false) where {T, N}
        N==0 && throw(AssertionError("The dimension of `refArr` should be at least one."))
        new{N}(size(refArr), mark, TernaryNumber(truncate|>Int))
    end

    TruncateReshape(f::TruncateReshape{N}; 
                    truncate::Union{Bool, TernaryNumber}=f.truncate) where {N} = 
    new{N}(f.axis, f.mark, TernaryNumber(truncate|>Int))
end

function (f::TruncateReshape{N})(arr::AbstractArray) where {N}
    extent = prod(f.axis)
    truncate = Int(f.truncate)
    v = if truncate == 0
        arr
    elseif truncate == 1
        arr[begin:begin+extent-1]
    else
        arr[end-extent+1:end]
    end
    reshape(v, f.axis)
end


struct MemoryLinker{T, A<:AbstractArray{T}} <: AbstractArray{T, 1}
    value::A
    scope::Memory{OneToIndex}

    function MemoryLinker(value::A, scope::AbstractArray{OneToIndex}) where 
                         {T, A<:AbstractArray{T}}
        new{T, A}(value, extractMemory(scope))
    end
end
#> Iteration interface
iterate(arr::MemoryLinker) = iterate(arr, firstindex(arr.scope))
function iterate(arr::MemoryLinker, state)
    res = iterate(arr.scope, state)
    if res === nothing
        nothing
    else
        indexer, state = res
        getEntry(arr.value, indexer), state
    end
end
#> Abstract-array (and indexing) interface
size(arr::MemoryLinker) = size(arr.scope)
getindex(arr::MemoryLinker, i::Int) = getEntry(arr.value, getindex(arr.scope, i))
function setindex!(arr::MemoryLinker, val, i::Int)
    idxInner = firstindex(arr.value) + getindex(arr.scope, i).idx - 1
    setindex!(arr.value, val, idxInner)
end
IndexStyle(::MemoryLinker) = IndexLinear()


struct VectorMemory{T, L} <: CustomMemory{T, 1}
    value::Memory{T}
    shape::Val{L}

    function VectorMemory(value::Memory{T}, ::Val{L}) where {T, L}
        checkLength(value, :value, L)
        new{T, L::Int}(value, Val(L))
    end

    function VectorMemory{T}(::UndefInitializer, ::Val{L}) where {T, L}
        new{T, L::Int}(Memory{T}(undef, L), Val(L))
    end

    function VectorMemory(arr::VectorMemory{T, L}) where {T, L}
        new{T, L::Int}(arr.value, arr.shape)
    end
end

VectorMemory(input::GeneralCollection) = 
VectorMemory(extractMemory(input), Val(input|>length))
#> Iteration interface
iterate(arr::VectorMemory) = iterate(arr.value)
iterate(arr::VectorMemory, state) = iterate(arr.value, state)
#> Abstract-array (and indexing) interface
size(::VectorMemory{<:Any, L}) where {L} = (L,)
getindex(arr::VectorMemory, i::Int) = getindex(arr.value, i)
setindex!(arr::VectorMemory, val, i::Int) = setindex!(arr.value, val, i)
IndexStyle(::VectorMemory) = IndexLinear()
#> Optional interface (better performance than the default implementation)
zero(arr::VectorMemory{T, L}) where {T, L} = VectorMemory(zero(arr.value), Val(L))
#>> Necessary for `copy` to return the same container type
function similar(arr::VectorMemory, ::Type{T}=eltype(arr), 
                 (len,)::Tuple{Int}=size(arr)) where {T}
    VectorMemory(similar(arr.value, T, len), Val(len))
end

const LinearMemory{T} = Union{Memory{T}, VectorMemory{T}}


struct ShapedMemory{T, N} <: CustomMemory{T, N}
    value::Memory{T}
    shape::NTuple{N, Int}

    function ShapedMemory(value::Memory{T}, shape::Tuple{Vararg{Int}}) where {T}
        checkReshapingAxis(value, shape)
        new{T, length(shape)}(copy(value), shape)
    end

    function ShapedMemory{T}(::UndefInitializer, shape::NTuple{N, Int}) where {T, N}
        checkReshapingAxis(shape)
        new{T, N}(Memory{T}(undef, prod(shape)), shape)
    end

    function ShapedMemory(arr::ShapedMemory{T, N}) where {T, N}
        new{T, N}(arr.value, arr.shape)
    end
end

ShapedMemory(value::AbstractArray{T}, shape::Tuple{Vararg{Int}}=size(value)) where {T} = 
ShapedMemory(extractMemory(value), shape)

ShapedMemory(::Type{T}, value::AbstractArray{T}) where {T} = ShapedMemory(value)

ShapedMemory(::Type{T}, value::T) where {T} = ShapedMemory( fill(value) )
#> Iteration interface
iterate(arr::ShapedMemory) = iterate(arr.value)
iterate(arr::ShapedMemory, state) = iterate(arr.value, state)
#> Abstract-array (and indexing) interface
size(arr::ShapedMemory) = arr.shape
getindex(arr::ShapedMemory, i::Int) = getindex(arr.value, i)
setindex!(arr::ShapedMemory, val, i::Int) = setindex!(arr.value, val, i)
IndexStyle(::ShapedMemory) = IndexLinear()
#> Optional interface
#>> Necessary for `copy` to return the same container type
function similar(arr::ShapedMemory, ::Type{T}=eltype(arr), 
                 shape::Tuple{Vararg{Int}}=size(arr)) where {T}
    ShapedMemory(similar(arr.value, T, prod(shape)), shape)
end


function binaryApply(op::F, arr1::ShapedMemory{T1}, arr2::ShapedMemory{T2}) where 
                    {F<:Function, T1, T2}
    if size(arr1) != size(arr2)
        throw(DimensionMismatch("`arr1` has size $(arr1.shape); "*
                                "`arr2` has size $(arr2.shape)."))
    end
    val = Memory{promote_type(T1, T2)}(op(arr1.value, arr2.value))
    ShapedMemory(val, arr1.shape)
end

+(arr1::ShapedMemory, arr2::ShapedMemory) = binaryApply(+, arr1, arr2)
-(arr1::ShapedMemory, arr2::ShapedMemory) = binaryApply(-, arr1, arr2)


abstract type AbstractPackedMemory{T, E, N} <: CustomMemory{E, N} end

const AbstractPack{T} = Union{T, AbstractPackedMemory{T}}

struct PackedMemory{T, E<:AbstractPack{T}, N} <: AbstractPackedMemory{T, E, N}
    value::ShapedMemory{E, N}
    level::NestedLevel{T}

    function PackedMemory(arr::AbstractArray{E, N}) where {E, N}
        formattedValue = map(arr) do ele
            extractPackedMemory(ele)
        end |> ShapedMemory
        nucType, eleType, shellLevel = checkPackedMemoryElement(formattedValue)
        new{nucType, eleType, N}(formattedValue, shellLevel)
    end

    function PackedMemory{T}(::UndefInitializer, shape::NTuple{N, Int}) where {T, N}
        intersection = typeintersect(T, AbstractArray)
        if !(intersection <: Union{}) && intersection <: AbstractArray
            throw(AssertionError("`T=$T` should not contain any (non-`Union{}`) subtypes "*
                                 "of `AbstractArray.`"))
        end
        value = ShapedMemory{T}(undef, shape)
        new{T, T, N}(value, getNestedLevel(ShapedMemory{T, N}))
    end
end

function checkPackedMemoryElement(::T) where {T<:ShapedMemory}
    shellLevel = getNestedLevel(T)
    nucType = getCoreType(shellLevel)
    eleType = eltype(T)
    isAllowedEle = if shellLevel.level == 1
        TypeBox(nucType) == TypeBox(eleType)
    else
        eleType <: PackedMemory{nucType}
    end
    isAllowedEle || throw(AssertionError("Illegal type-parameter combination (T, E): "*
                                         "($nucType, $eleType)."))
    nucType, eleType, shellLevel
end
#> Only preserve `PackedMemory` and non-`AbstractArray` objects
extractPackedMemory(obj::Any) = itself(obj)

extractPackedMemory(arr::PackedMemory) = itself(arr)

extractPackedMemory(arr::AbstractArray) = PackedMemory(arr)

const DirectMemory{T, N} = PackedMemory{T, T, N}
const NestedMemory{T, E<:PackedMemory{T}, N} = PackedMemory{T, E, N}
#> Iteration interface
iterate(arr::PackedMemory) = iterate(arr.value)
iterate(arr::PackedMemory, state) = iterate(arr.value, state)
#> Abstract-array (and indexing) interface
size(arr::PackedMemory) = size(arr.value)
getindex(arr::PackedMemory, i::Int) = getindex(arr.value, i)
setindex!(arr::PackedMemory, val, i::Int) = setindex!(arr.value, val, i)
IndexStyle(::PackedMemory) = IndexLinear()
#> Additional interface
ShapedMemory(arr::PackedMemory) = arr.value
#>> Necessary for `copy` to return the same container type
function similar(arr::DirectMemory{T1}, ::Type{T2}=eltype(arr), 
                 shape::Tuple{Vararg{Int}}=size(arr)) where {T1, T2<:T1}
    PackedMemory{T2}(undef, shape)
end


genPackMemoryType(::Type{T}) where {T} = T

genPackMemoryType(::Type{T}) where {T<:PackedMemory} = T

function genPackMemoryType(::Type{<:AbstractArray{T, N}}) where {T, N}
    coreT = getCoreType(T)
    innerT = genPackMemoryType(T)
    PackedMemory{coreT, innerT, N}
end


recursiveCompareSize(::Any, ::Any) = true

recursiveCompareSize(::AbstractArray, ::Any) = false

recursiveCompareSize(::Any, ::AbstractArray) = false

function recursiveCompareSize(arr1::AbstractArray, arr2::AbstractArray)
    if size(arr1) == size(arr2)
        for (ele1, ele2) in zip(arr1, arr2)
            recursiveCompareSize(ele1, ele2) || (return false)
        end
        true
    else
        false
    end
end

recursiveCompareSize(arr1::AbstractArray{<:Number}, arr2::AbstractArray{<:Number}) = 
size(arr1) == size(arr2)


const BottomMemory = Memory{Union{}}

genBottomMemory() = BottomMemory(undef, 0)


getMinimalEleType(obj::Any) = typeof(obj)

function getMinimalEleType(collection::AbstractArray{T}) where {T}
    if Base.isconcretetype(T)
        T
    elseif isempty(collection)
        Union{}
    else
        mapreduce(typeof, strictTypeJoin, collection)
    end
end

getMinimalEleType(collection::Tuple) = 
mapreduce(typeof, strictTypeJoin, collection, init=Union{})

#! Change `.tuple` to another name
struct WeakComp{N} # Weak composition of an integer
    tuple::NTuple{N, Int}
    total::Int

    function WeakComp(t::NonEmptyTuple{Int, M}) where {M}
        if any(i < 0 for i in t)
            throw(DomainError(t, "The element(s) of `t` should all be non-negative."))
        end
        new{M+1}(t, sum(t))
    end
end


"""

    extractMemory(obj::$GeneralCollection) -> Memory

Extract the `Memory` inside `obj::$AbstractMemory` (without copying if possible). If `obj` 
is not a `$AbstractMemory` a `AbstractArray`, `Tuple`, or `NamedTuple`, its entires will be 
filled into a newly constructed `Memory`.
"""
extractMemory(arr::Memory) = itself(arr)

extractMemory(arr::PackedMemory) = arr.value.value

extractMemory(arr::CustomMemory) = arr.value::Memory

function extractMemory(arr::AbstractArray{T}) where {T}
    eleT = if isconcretetype(T) || isempty(arr)
        T
    else
        mapreduce(typeof, strictTypeJoin, arr, init=Union{})
    end
    Memory{eleT}(vec(arr))
end

function extractMemory(obj::Tuple)
    mem = Memory{eltype(obj)}(undef, length(obj))
    mem .= obj
    mem
end

extractMemory(obj::NamedTuple) = extractMemory(obj|>values)


"""

    genMemory(obj) -> Memory

Generate a `Memory` filled with the entires from `obj::AbstractArray`. If `obj` is not an 
`AbstractArray`, it will be encapsulated in a one-element `Memory`.
"""
genMemory(arr::AbstractArray) = Memory{getMinimalEleType(arr)}(arr)

function genMemory(obj::T) where {T}
    mem = Memory{T}(undef, 1)
    mem[] = obj
    mem
end


"""

    decoupledCopy(obj::AbstractArray) -> AbstractArray

Recursively copy `obj` and decouple the correlations among its elements.
"""
function decoupledCopy(arr::AbstractArray)
    map(decoupledCopy, arr)
end

function decoupledCopy(obj::T)::T where {T}
    if canDirectlyStoreInstanceOf(T)
        obj
    else
        deepcopy(obj)
    end
end



function indexedPerturb(op::F, source::Tuple, idxVal::Pair{OneToIndex, T}) where 
                       {F<:Function, T}
    oneToIdx, val = idxVal
    ntuple(source|>length) do i
        i == oneToIdx.idx ? op(source[begin+i-1], val) : source[begin+i-1]
    end
end

function indexedPerturb(op::F, source::AbstractArray, idxVal::Pair{OneToIndex, T}) where 
                       {F<:Function, T}
    res = copy(source)

    oneToIdx, val = idxVal
    idx = oneToIdx.idx
    firstIdx = firstindex(source)
    if firstIdx <= idx <= lastindex(source)
        localIdx = firstIdx + idx - 1
        res[localIdx] = op(source[localIdx], val)
    end

    res
end


tightenCollection(::Nothing) = genBottomMemory()

tightenCollection(arr::AbstractVector) = isempty(arr) ? genBottomMemory() : genMemory(arr)


function setIndex(tpl::NTuple{N, Any}, val::T, idx::Int, 
                  modifier::F=SelectHeader{2, 2}(itself)) where {N, T, F<:Function}
    f = let offset=(firstindex(tpl) - 1), valNew=modifier(tpl[idx], val)
        function (j::Int)
            jIdx = j + offset
            isequal(idx, jIdx) ? valNew : tpl[jIdx]
        end
    end

    ntuple(f, Val(N))
end