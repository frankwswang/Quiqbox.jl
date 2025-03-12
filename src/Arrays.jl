export VectorMemory, ShapedMemory

struct NestedLevel{T}
    level::Int

    function NestedLevel(::Type{T}, level::Int=0) where {T}
        checkPositivity(level, true)
        new{T}(level)
    end
end

getNestedLevelCore(::Type{T}, level::Int) where {T} = (T, level)

function getNestedLevelCore(::Type{<:AbstractArray{T}}, level::Int) where {T}
    getNestedLevelCore(T, level+1)
end

getNestedLevel(::Type{T}) where {T} = NestedLevel(getNestedLevelCore(T, 0)...)

getCoreType(::NestedLevel{T}) where {T} = T

function getSpanType(l::NestedLevel)
    res = getCoreType(l)
    for _ in l.level
        res = AbstractArray{<:res}
    end
    res
end

function getSpanType(::Type{<:AbstractArray{T, N}}) where {T, N}
    innerT = getSpanType(T)
    if isconcretetype(innerT)
        AbstractArray{innerT, N}
    else
        AbstractArray{<:innerT, N}
    end
end

getSpanType(::Type{T}) where {T} = T


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


struct VectorMemory{T, L} <: AbstractMemory{T, 1}
    value::Memory{T}
    shape::Val{L}

    function VectorMemory(value::Memory{T}, ::Val{L}) where {T, L}
        checkLength(value, :value, L)
        new{T, L}(value, Val(L))
    end

    function VectorMemory{T}(::UndefInitializer, ::Val{L}) where {T, L}
        new{T, L}(Memory{T}(undef, L), Val(L))
    end

    function VectorMemory(arr::VectorMemory{T, L}) where {T, L}
        new{T, L}(arr.value, arr.shape)
    end
end

VectorMemory(input::AbstractArray) = VectorMemory(getMemory(input), Val(length(input)))

VectorMemory(input::NonEmptyTuple{Any, L}) where {L} = 
VectorMemory(getMemory(input), Val(L+1))

size(::VectorMemory{<:Any, L}) where {L} = (L,)

getindex(arr::VectorMemory, i::Int) = getindex(arr.value, i)

setindex!(arr::VectorMemory, val, i::Int) = setindex!(arr.value, val, i)

iterate(arr::VectorMemory) = iterate(arr.value)
iterate(arr::VectorMemory, state) = iterate(arr.value, state)

length(::VectorMemory{<:Any, L}) where {L} = L

const LinearMemory{T} = Union{Memory{T}, VectorMemory{T}}


struct ShapedMemory{T, N} <: AbstractMemory{T, N}
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
ShapedMemory(getMemory(value), shape)

ShapedMemory(::Type{T}, value::AbstractArray{T}) where {T} = ShapedMemory(value)

ShapedMemory(::Type{T}, value::T) where {T} = ShapedMemory( fill(value) )

getMemory(arr::ShapedMemory) = arr.value


size(arr::ShapedMemory) = arr.shape

firstindex(arr::ShapedMemory) = firstindex(arr.value)

lastindex(arr::ShapedMemory) = lastindex(arr.value)

getindex(arr::ShapedMemory, i::Int) = getindex(arr.value, i)
getindex(arr::ShapedMemory{<:Any, N}, i::Vararg{Int, N}) where {N} = 
getindex(reshape(arr.value, arr.shape), i...)

setindex!(arr::ShapedMemory, val, i::Int) = setindex!(arr.value, val, i)
setindex!(arr::ShapedMemory{<:Any, N}, val, i::Vararg{Int, N}) where {N} = 
setindex!(reshape(arr.value, arr.shape), val, i...)

iterate(arr::ShapedMemory) = iterate(arr.value)
iterate(arr::ShapedMemory, state) = iterate(arr.value, state)

length(arr::ShapedMemory) = length(arr.value)

axes(arr::ShapedMemory)	= map(Base.OneTo, size(arr))

function similar(arr::ShapedMemory, ::Type{T}=eltype(arr), 
                 shape::Tuple{Vararg{Int}}=size(arr)) where {T}
    ShapedMemory(similar(arr.value, T, prod(shape)), shape)
end

similar(arr::ShapedMemory{T}, shape::Tuple{Vararg{Int}}) where {T} = 
similar(arr, T, shape)


function binaryApply(op::F, arr1::ShapedMemory{T1}, arr2::ShapedMemory{T2}) where 
                 {F<:Function, T1, T2}
    if arr1.shape != arr2.shape
        throw(DimensionMismatch("`arr1` has size $(arr1.shape); "*
                                "`arr2` has size $(arr2.shape)."))
    end
    val = Memory{promote_type(T1, T2)}(op(arr1.value, arr2.value))
    ShapedMemory(val, arr1.shape)
end

+(arr1::ShapedMemory, arr2::ShapedMemory) = binaryApply(+, arr1, arr2)
-(arr1::ShapedMemory, arr2::ShapedMemory) = binaryApply(-, arr1, arr2)

viewElements(obj::ShapedMemory) = reshape(obj.value, obj.shape)
viewElements(obj::AbstractArray) = itself(obj)


abstract type AbstractNestedMemory{T, E, N} <: AbstractMemory{E, N} end

struct NestedMemory{T, E<:Union{ShapedMemory{T}, AbstractNestedMemory{T}}, 
                    N} <: AbstractNestedMemory{T, E, N}
    value::ShapedMemory{E, N}

    function NestedMemory(arr::AbstractArray{<:AbstractArray{T}, N}) where {T, N}
        l = getNestedLevel(arr|>typeof)
        value = map(arr) do ele
            getNestedMemory(ele)
        end |> ShapedMemory
        new{getCoreType(l), eltype(value), N}(value)
    end
end

getNestedMemory(obj::Any) = itself(obj)

getNestedMemory(arr::NestedMemory) = itself(arr)

getNestedMemory(arr::AbstractArray{T}) where {T} = ShapedMemory(arr)

getNestedMemory(arr::AbstractArray{<:AbstractArray{T}}) where {T} = NestedMemory(arr)


size(arr::NestedMemory) = size(arr.value)

firstindex(arr::NestedMemory) = firstindex(arr.value)

lastindex(arr::NestedMemory) = lastindex(arr.value)

getindex(arr::NestedMemory, i::Vararg{Int}) = getindex(arr.value, i...)

setindex!(arr::NestedMemory, val, i::Vararg{Int}) = setindex!(arr.value, val, i...)

iterate(arr::NestedMemory) = iterate(arr.value)
iterate(arr::NestedMemory, state) = iterate(arr.value, state)

length(arr::NestedMemory) = length(arr.value)

axes(arr::NestedMemory)	= map(Base.OneTo, size(arr))


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