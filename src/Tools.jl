export hasEqual, hasIdentical, flatten, markUnique, getUnique!

using Statistics: std, mean
using Symbolics
using LinearAlgebra: eigvals, svdvals, eigen


# Function for submodule loading and integrity checking.
function tryIncluding(subModuleName::String; subModulePath=(@__DIR__)[:]*"/SubModule")
    try
        include(subModulePath*"/"*subModuleName*".jl")
        return true
    catch err
        warning = """
        Submodule `$(subModuleName)` failed loading and won't be useable:

            $(err)

        `///magenta///However, this DOES NOT affect the functionality of the main module.`
        """
        printStyledInfo(warning, title="WARNING:\n", titleColor=:light_yellow)
        return false
    end
end


"""

    @compareLength inputArg1 inputArg2 argNames::String... -> length(inputArg1)

A macro that checks whether the lengths of 2 `Arrays`/`Tuples` are equal. It returns the 
lengths of the compared objects are the same; it throws a detailed ERROR message when they 
are not equal.

You can specify the name for the compared variables in arguments for better ERROR 
information.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> Quiqbox.@compareLength [1,2] [3,4]
2

julia> let a = [1,2], b = [3]
           Quiqbox.@compareLength a b
       end
ERROR: The lengths of a and b are NOT equal.
       a::Vector{Int64}   length: 2
       b::Vector{Int64}   length: 1

julia> Quiqbox.@compareLength [1,2] [3] "a"
ERROR: The lengths of a ([1, 2]) and [3] are NOT equal.
       a ([1, 2])::Vector{Int64}   length: 2
       [3]::Vector{Int64}   length: 1

julia> Quiqbox.@compareLength [1,2] [3] "a" "b"
ERROR: The lengths of a ([1, 2]) and b ([3]) are NOT equal.
       a ([1, 2])::Vector{Int64}   length: 2
       b ([3])::Vector{Int64}   length: 1
```
"""
macro compareLength(inputArg1, inputArg2, argNames::String...)
    # In a macro you must escape all the user inputs once and exactly once.
    ns0 = [string(inputArg1), string(inputArg2)]
    ns = ns0
    quote
        local arg1 = $(esc(inputArg1))
        local arg2 = $(esc(inputArg2))
        type = Union{AbstractArray, Tuple}
        (!(arg1 isa type && arg2 isa type)) && error("The compared objects have to be ", 
                                                     "Arrays or Tuples!\n")
        if length(arg1) != length(arg2)
            for i = 1:length($argNames)
                # Replace the default type control ERROR message.
                !($argNames[i] isa String) && error("The object's name has to be a "*
                                                    "`String`!\n")
                $ns[i] = $argNames[i]*" ($($ns0[i]))"
            end
            error("""The lengths of $($ns[1]) and $($ns[2]) are NOT equal.
                           $($ns0[1])::$(typeof(arg1))   length: $(length(arg1))
                           $($ns0[2])::$(typeof(arg2))   length: $(length(arg2))
                    """)
        end
        length(arg1)
    end
end


"""

    hasBoolRelation(boolOp::Function, obj1, obj2; ignoreFunction::Bool=false, 
                    ignoreContainer::Bool=false, decomposeNumberCollection::Bool=false) -> 
    Bool

Recursively apply the specified boolean operator to all the fields within 2 objects 
(normally 2 `struct`s in the same type). It returns `true` only if all comparisons 
performed return `true`. Note that the boolean operator should have method(s) defined for 
all the possible elements inside the compared objects.

If `ignoreFunction = true`, the function will ignore comparisons between Function-type 
fields.

If `ignoreContainer = true`, the function will ignore the difference of the container(s) as 
long as the boolean operator returns true for the field(s)/entry(s) from two objects 
respectively.

If `decomposeNumberCollection = true`, then `Tuple{Vararg{Number}}` and `Array{<:Number}` 
will be treated as decomposable containers.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
begin
    struct S
        a::Int
        b::Float64
    end

    a = S(1, 1.0)
    b = S(2, 0.5)
    c = S(2, 1.5)

    Quiqbox.hasBoolRelation(>, a, b) |> println
    Quiqbox.hasBoolRelation(>, b, a) |> println
    Quiqbox.hasBoolRelation(>, c, a) |> println
end

# output
false
false
true
```

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
begin
    struct S
        a::Int
        b::Float64
    end

    struct S2
        a::Int
        b::Float64
    end

    Quiqbox.hasBoolRelation(==, S(1,2), S2(1,2), ignoreContainer=true)
end

# output
true
```
"""
function hasBoolRelation(boolOp::F, obj1, obj2;
                         ignoreFunction::Bool=false, 
                         ignoreContainer::Bool=false,
                         decomposeNumberCollection::Bool=false) where {F<:Function}
    res = true
    t1 = typeof(obj1)
    t2 = typeof(obj2)
    if (t1 <: Function) && (t2 <: Function)
        ignoreFunction ? (return true) : (return boolOp(obj1, obj2))
    elseif (t1 <: Number) && (t2 <: Number)
        return boolOp(obj1, obj2)
    elseif t1 != t2 && !ignoreContainer
        return false
    elseif obj1 isa Union{Array, Tuple}
        if !decomposeNumberCollection && 
           (eltype(obj1) <: Number) && (eltype(obj2) <: Number)
            return boolOp(obj1, obj2)
        end
        length(obj1) != length(obj2) && (return false)
        !ignoreContainer && obj1 isa Matrix && (size(obj1) != size(obj2)) && (return false)
        for (i,j) in zip(obj1, obj2)
            res *= hasBoolRelation(boolOp, i, j; ignoreFunction, ignoreContainer, 
                                   decomposeNumberCollection)
            !res && (return false)
        end
    elseif obj1 isa Type || obj2 isa Type
        return boolOp(obj1, obj2)
    else
        fs1 = fieldnames(t1)
        fs2 = fieldnames(t2)
        if fs1 == fs2
            if length(fs1) == 0
                res = boolOp(obj1, obj2)
            else
                for i in fs1
                    isdefined(obj1, i) == (fieldDefined = isdefined(obj2, i)) && 
                    (fieldDefined ? nothing : continue)
                    res *= hasBoolRelation(boolOp, getproperty(obj1, i), 
                                           getproperty(obj2, i); ignoreFunction, 
                                           ignoreContainer, decomposeNumberCollection)
                    !res && (return false)
                end
            end
        else
            return false
        end
    end
    res
end
## Refer overload for `ParamBox` to Overload.jl.

"""

    hasBoolRelation(boolOp::F, obj1, obj2, obj3...; 
                    ignoreFunction::Bool=false, 
                    ignoreContainer::Bool=false,
                    decomposeNumberCollection::Bool=false) where {F<:Function} -> 
    Bool

Method for more than 2 objects. E.g.: `hasBoolRelation(>, a, b, c)` is equivalent to 
`hasBoolRelation(>, a, b) && hasBoolRelation(>, b, c)`.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
begin
    struct S
        a::Int
        b::Float64
    end

    a = S(1, 1.0)
    b = S(2, 0.5)
    c = S(2, 1.5)
    d = S(3, 2.0)

    Quiqbox.hasBoolRelation(>=, c, b, a) |> println
    Quiqbox.hasBoolRelation(>=, d, c, b) |> println
end

# output
false
true
```
"""
function hasBoolRelation(boolOp::F, obj1, obj2, obj3...; 
                         ignoreFunction::Bool=false, 
                         ignoreContainer::Bool=false,
                         decomposeNumberCollection::Bool=false) where {F<:Function}
    res = hasBoolRelation(boolOp, obj1, obj2; ignoreFunction, ignoreContainer)
    tmp = obj2
    if res
        for i in obj3[1:end]
            res *= hasBoolRelation(boolOp, tmp, i; ignoreFunction, ignoreContainer,
                                   decomposeNumberCollection)
            !res && break
            tmp = i
        end
    end
    res
end


"""

    hasEqual(obj1, obj2, obj3...; 
             ignoreFunction::Bool=false, 
             ignoreContainer::Bool=false,
             decomposeNumberCollection::Bool=false) -> 
    Bool

Compare if two objects are the equal.

If `ignoreFunction = true`, the function will ignore comparisons between Function-type 
fields.

If `ignoreContainer = true`, the function will ignore the difference of the container(s) 
and only compare the field(s)/entry(s) from two objects respectively.

If `decomposeNumberCollection = true`, then `Tuple{Vararg{Number}}` and `Array{<:Number}` 
will be treated as decomposable containers.

This is an instantiation of `Quiqbox.hasBoolRelation`.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
begin
    struct S
        a::Int
        b::Float64
    end
    a = S(1, 1.0)
    b = S(1, 1.0)
    c = S(1, 1.0)
    d = S(1, 1.1)

    hasEqual(a, b, c) |> println
    hasEqual(a, b, c, d) |> println
end

# output
true
false
```
"""
hasEqual(obj1, obj2, obj3...; ignoreFunction::Bool=false, ignoreContainer::Bool=false, 
        decomposeNumberCollection::Bool=false) = 
hasBoolRelation(==, obj1, obj2, obj3...; ignoreFunction, ignoreContainer, 
                decomposeNumberCollection)


"""

    hasIdentical(obj1, obj2, obj3...; 
                 ignoreFunction::Bool=false, 
                 ignoreContainer::Bool=false,
                 decomposeNumberCollection::Bool=false) -> 
    Bool

Compare if two objects are the Identical. An instantiation of `hasBoolRelation`.

If `ignoreFunction = true`, the function will ignore comparisons between Function-type 
fields.

If `ignoreContainer = true`, the function will ignore the difference of the container(s) 
and only compare the field(s)/entry(s) from two objects respectively.

If `decomposeNumberCollection = true`, then `Tuple{Vararg{Number}}` and `Array{<:Number}` 
will be treated as decomposable containers.

This is an instantiation of `Quiqbox.hasBoolRelation`.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
begin
    struct S
        a::Int
        b::Array{Float64, 1}
    end

    a = S(1, [1.0, 1.1])
    b = a
    c = b
    d = S(1, [1.0, 1.1])

    hasIdentical(a, b, c) |> println
    hasIdentical(a, b, c, d) |> println
end

# output
true
false
```
"""
hasIdentical(obj1, obj2, obj3...; ignoreFunction::Bool=false, ignoreContainer::Bool=false,
             decomposeNumberCollection::Bool=false) = 
hasBoolRelation(===, obj1, obj2, obj3...; ignoreFunction, ignoreContainer,
                decomposeNumberCollection)


"""

    printStyledInfo(str::String; title::String="", titleColor::Symbol=:light_blue) -> 
    Nothing

Print info with colorful title and automatically highlighted code blocks enclosed by ` `.

If you want to highlight other contents in different colors, you can also put them inside 
` ` and start it with "///theColorSymbolName///". The available color names follows the 
values of `color` keyword argument in function `printstyled`.

NOTE: There can only be one color in one ` ` quote.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> Quiqbox.printStyledInfo("This `///magenta///word` is in color magenta.")
This word is in color magenta.
```
"""
function printStyledInfo(str::String; 
                         title::String="", titleColor::Symbol=:light_blue)
    codeColor = :cyan
    codeQuotes = findall("`", str)
    l = codeQuotes |> length
    printstyled(title, bold=true, color=titleColor)
    if (l > 0) && (l |> iseven)
        print(str[1:codeQuotes[1][]-1])
        for i = 1:2:l-1
            subStr1 = str[codeQuotes[i][]+1:end]
            blockStr = subStr1[1 : findfirst("`", subStr1)[]-1]
            blockColor = codeColor
            startLoc = findfirst("///", blockStr)
            if startLoc !== nothing
                endLoc = findnext("///", blockStr, 2)
                if endLoc !== nothing
                    colorKey = Symbol(blockStr[startLoc[3]+1 : endLoc[1]-1])
                    blockColor = colorKey
                    blockStr = blockStr[1 : startLoc[1]-1]*blockStr[endLoc[3]+1 : end]
                end
            end
            printstyled(blockStr, color=blockColor)
            subStr2 = str[codeQuotes[i+1][]+1:end]
            loc = findfirst("`", subStr2)
            unBlockStr = subStr2[1 : ((loc===nothing) ? (subStr2|>length) : (loc[]-1))]
            print(unBlockStr)
        end
    else
        print(str)
    end
    println()
end


"""
    flatten(a::Tuple) -> Tuple

    flatten(a::Array) -> Array

Flatten `a::Union{Array, Tuple}` that contains `Array`s and/or `Tuple`s. Only operate on 
the outermost layer.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> flatten((:one, 2, [3, 4.0], ([5], "six"), "7"))
(:one, 2, 3.0, 4.0, [5], "six", "7")

julia> flatten([:one, 2, [3, 4.0], ([5], "six"), "7"])
7-element Vector{Any}:
  :one
 2
 3.0
 4.0
  [5]
  "six"
  "7"
```
"""
function flatten(c::Array)
    c2 = map( x->(x isa Union{Array, Tuple} ? x : (x,)), c )
    [(c2...)...]
end

function flatten(c::Tuple)
    c2 = map( x->(x isa Union{Array, Tuple} ? x : (x,)), c )
    ((c2...)...,)
end


"""

    arrayAlloc(arrayLength::Int, 
               anExampleOrType::Union{T, Type{T}}) where {T<:Real} -> 
    Ptr{T}

Allocate the memory for an array of specified length and element type, then return the 
pointer `Ptr` to it.
"""
function arrayAlloc(arrayLength::Int, elementType::Type{T}) where {T<:Real}
    memoryLen = arrayLength*sizeof(elementType) |> Cint
    ccall(:malloc, Ptr{T}, (Cint,), memoryLen)
end

arrayAlloc(arrayLength::Int, NumberExample::T) where {T<:Real} = 
arrayAlloc(arrayLength, typeof(NumberExample))


"""

    ArrayPointer{T, N} <: Any

Stores a pointer to the actual address of an array.

≡≡≡ Field(s) ≡≡≡

`ptr::Ptr{T}`: Pointer pointing to the memory address of the first element of the array.

`arr::Array{T, N}`: The mutable array linked to the pointer. As long as the pointer 
                    (memory) is not freed, the array is safely preserved.

≡≡≡ Initialization Method(s) ≡≡≡

    ArrayPointer(arr::Array{<:Real, N}, 
                 showReminder::Bool=true) where {N} -> ArrayPointer{T, N}

Create a `ArrayPointer` that contains a `Ptr` pointing to the actual memory address of the 
(1st element of the) `Array`.

To avoid memory leaking, the user should use `free(x.ptr)` after the usage of 
`x::ArrayPointer` to free the occupied memory.

If `showReminder=true`, the constructor will pop up a message to remind the user of 
such operation.
"""
struct ArrayPointer{T, N} <: Any
    ptr::Ptr{T}
    arr::Array{T, N}

    function ArrayPointer(arr::Array{<:Real, N}, showReminder::Bool=true) where {N}
        len = length(arr)
        elt =  eltype(arr)
        ptr = arrayAlloc(len, elt)
        unsafe_copyto!(ptr, pointer(arr |> copy), len)
        arr2 = unsafe_wrap(Array, ptr, size(arr))
        showReminder && printStyledInfo("""
            Generating a C-array pointer-like object x`::ArrayPointer{$(elt)}`...
            Remember to use free(x.ptr) afterwards to prevent potential memory leaking.
            """)
        new{elt, N}(ptr, arr2)
    end
end


"""

    markUnique(arr::AbstractArray, args...; 
               compareFunction::Function = hasEqual, kws...) -> 
    markingList:: Array{Int, 1}, uniqueList::Array

Return a `markingList` using `Int` number to mark each different elements from (and inside) 
the input argument(s) and a `uniqueList` to contain all the unique elements when 
`compareFunction` is set to `hasEqual` (in default).

`args` and `kws` are positional arguments and keywords arguments respectively as 
parameters of the specified `compareFunction`.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
markUnique([1, [1, 2],"s", [1, 2]])

# output
([1, 2, 3, 2], Any[1, [1, 2], "s"])
```

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
begin
    struct S
        a::Int
        b::Float64
    end

    a = S(1, 2.0)
    b = S(1, 2.0)
    c = S(1, 2.1)
    d = a

    markUnique([a,b,c,d])
end

# output
([1, 1, 2, 1], S[S(1, 2.0), S(1, 2.1)])
```
"""
function markUnique(arr::AbstractArray{T}, args...; 
                    compareFunction::F=hasEqual, kws...) where {T<:Any, F<:Function}
    @assert length(arr) >= 1 "The length of input array should be not less than 1."
    f = (b...)->compareFunction((b..., args...)...; kws...)
    res = Int[1]
    cmprList = T[arr[1]]
    for i = 2:length(arr)
        local j
        isNew = true
        for outer j = 1:length(cmprList)
            if f(cmprList[j], arr[i])
                isNew = false
                break
            end
            j += 1
        end
        push!(res, j)
        isNew && push!(cmprList, arr[i])
    end
    res, cmprList
end

"""

    getUnique!(arr::Array, args...; compareFunction::F = hasEqual, kws...) where 
              {F<:Function} -> 
    arr::Array

Similar to [`markUnique`](@ref) but instead, just directly return the input `Array` with 
repeated entries deleted.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> arr = [1, [1, 2],"s", [1, 2]]
4-element Vector{Any}:
 1
  [1, 2]
  "s"
  [1, 2]

julia> getUnique!(arr);

julia> arr
3-element Vector{Any}:
 1
  [1, 2]
  "s"
```
"""
function getUnique!(arr::AbstractArray{T}, args...; 
                    compareFunction::F = hasEqual, kws...) where {T<:Any, F<:Function}
    @assert length(arr) > 1 "The length of input array should be larger than 1."
    f = (b...)->compareFunction((b..., args...)...; kws...)
    cmprList = T[arr[1]]
    delList = Bool[false]
    for i = 2:length(arr)
        isNew = true
        for j = 1:length(cmprList)
            if f(cmprList[j], arr[i])
                isNew = false
                break
            end
        end
        isNew && push!(cmprList, arr[i])
        push!(delList, !isNew)
    end
    deleteat!(arr, delList)
end


"""
A dummy function that only returns its argument.
"""
itself(x) = x


"""
Similar as `replace` but for Symbols.
"""
function symbolReplace(sym::Symbol, pair::Pair{String, String}; count::Int=typemax(Int))
    replace(sym |> string, pair; count) |> Symbol
end


function renameFunc(fName::String, f::F) where {F<:Function}
    @eval ($(Symbol(fName)))(a...; b...) = $f(a...; b...)
end


"""
Recursively find the final value using the value of each iteration as the key for the 
next search.
"""
function recursivelyGet(dict::Dict, startKey::Any)
    res = nothing
    val = get(dict, startKey, missing)
    while !(val isa Missing)
        res = val
        val = get(dict, val, missing)
    end
    res
end


function isOscillateConverged(sequence::Vector{<:Real}, 
                              threshold1::Real, threshold2::Real=threshold1; 
                              leastCycles::Int=1, nPartition::Int=5, returnStd::Bool=false)
    @assert leastCycles>0 && nPartition>1
    len = length(sequence)
    len < leastCycles && (return false)
    slice = len ÷ nPartition
    lastPortion = sequence[max(end-slice, 1) : end]
    remained = sort(lastPortion)[end÷2+1 : end]
    b = std(remained) < threshold1 && abs(sequence[end] - mean(remained)) < threshold2
    returnStd ? (b, std(lastPortion)) : b
end


function splitTerm(term::Symbolics.Num)
    r1 = Symbolics.@rule +(~(~xs)) => [i for i in ~(~xs)]
    r2 = Symbolics.@rule *(~(~xs)) => [[i for i in ~(~xs)] |> prod]
    for r in [r1, r2]
        term = Symbolics.simplify(term, rewriter = r)
    end
    # Converting Symbolics.Arr to Base.Array
    if term isa Symbolics.Arr
        terms = term |> collect
    else
        terms = [term]
    end
    terms
end


function groupedSort(v::Vector, sortFunction::F=itself) where {F<:Function}
    sortedArr = sort(v, by=x->sortFunction(x))
    state1 = 1
    groups = typeof(v)[]
    next = iterate(sortedArr)
    while next !== nothing
        item, state = next
        next = iterate(sortedArr, state)
        if next === nothing || sortFunction(next[1]) != sortFunction(item)
            push!(groups, sortedArr[state1:state-1])
            state1 = state
        end
    end
    groups
end


function mapPermute(arr, permFunction)
    ks = [[true, x, i] for (x, i) in zip(arr, eachindex(arr))]
    arrNew = permFunction(arr)
    idx = Int[]
    for ele in arrNew
        i = findfirst(x -> x[1] == true && hasIdentical(x[2], ele), ks)
        push!(idx, i)
        ks[i][1] = false
    end
    idx
end


function getFunc(fSym::Symbol, failedResult=missing)
    try
        getfield(Quiqbox, fSym)
    catch
        try
            getfield(Main, fSym)
        catch
            try
                fSym |> string |> Meta.parse |> eval
            catch
                (_) -> failedResult
            end
        end
    end
end


struct Pf{C, F} <: ParameterizedFunction
    f::Function
end

Pf(c::Float64, f::Function) = Pf{c, nameOf(f)}(f)
Pf(c::Float64, f::Pf{C, F}) where {C, F} = Pf{c*C, F}(f.f)
Pf(c::Float64, ::Val{T}) where {T} = Pf{c, T}(getFunc(T, NaN))
Pf(c::Float64, ::Val{Pf{C, F}}) where {C, F} = Pf{c*C, F}(getFunc(F, NaN))


(f::Pf{C})(x::Real) where {C} = C * f.f(x)
(::Type{Pf{C, F}})(x::Real) where {C, F} = C * getFunc(F, NaN)(x)

Pf(c::Float64, ::Pf{C, :itself}) where {C} = Pf{c*C, :itself}(itself)
Pf(c::Float64, ::Val{:itself}) = Pf{c, :itself}(itself)
Pf(c::Float64, ::Val{Pf{C, :itself}}) where {C} = Pf{c*C, :itself}(itself)

(f::Pf{C, :itself})(x::Real) where {C} = C * x
(::Type{Pf{C, :itself}})(x::Real) where {C} = C * x


nameOf(f::ParameterizedFunction) = typeof(f)

nameOf(f) = nameof(f)