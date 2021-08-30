export hasEqual, hasIdentical, hasBoolRelation, markUnique, getUnique!, flatten, splitTerm

using Statistics: std, mean
using Symbolics
using LinearAlgebra: eigvals, svdvals, eigen


# Function for submudole loading and integrity checking.
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
    
A macro that checks whether the lengths of 2 `Arrays`/`Tuples` are equal. It throws 
a detailed ERROR message when the lengths are not equal. 

You can specify the name for the compared variables in arguments for better ERROR 
information.

≡≡≡ Example(s) ≡≡≡

```jldoctest
julia> @compareLength [1,2] [3,4]
2

julia> let a = [1,2], b = [3]
           @compareLength a b
       end
ERROR: The lengths of a and b are NOT equal.
       a::Vector{Int64}   length: 2
       b::Vector{Int64}   length: 1

julia> @compareLength [1,2] [3] "a"
ERROR: The lengths of a ([1, 2]) and [3] are NOT equal.
       a ([1, 2])::Vector{Int64}   length: 2
       [3]::Vector{Int64}   length: 1

julia> @compareLength [1,2] [3] "a" "b"
ERROR: The lengths of a ([1, 2]) and b ([3]) are NOT equal.
       a ([1, 2])::Vector{Int64}   length: 2
       b ([3])::Vector{Int64}   length: 1
```
"""
macro compareLength(inputArg1, inputArg2, argNames...)
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
                # Replace the defualt type control ERROR message.
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
    
    hasBoolRelation(boolOp::Function, obj1, obj2; ignoreFunction=false, 
                    ignoreContainerType=false) -> Bool

Recursively apply the specified boolean operator to all the fields within 2 objects 
(normally 2 `struct`s in the same type). It returns `true` only if all comparisons 
performed return `true`.

If `ignoreFunction = true`, the function will ignore comparisons between Function-type 
fields.

If `ignoreContainerType = true`, the funtion will ignore the type difference of the 
(outermost) container as long as the boolean operator returns true for inside fields. 


≡≡≡ Example(s) ≡≡≡

```
julia> begin
           struct S
               a::Int
               b::Float64
           end

           a = S(1, 1.0)
           b = S(2, 0.5)
           c = S(2, 1.5)
           
           @show hasBoolRelation(>, a, b)
           @show hasBoolRelation(>, b, a)
           @show hasBoolRelation(>, c, a)
       end
hasBoolRelation(>, a, b) = false
hasBoolRelation(>, b, a) = false
hasBoolRelation(>, c, a) = true
true

julia> begin
           struct S 
               a::Int
               b::Float64
           end 
            
           struct S2 
               a::Int
               b::Float64
           end
              
           hasBoolRelation(==, S(1,2), S2(1,2), ignoreContainerType=true)
       end
true
```
"""
function hasBoolRelation(boolOp::F, obj1, obj2;
                         ignoreFunction=false, 
                         ignoreContainerType=false) where {F<:Function}
    res = true
    t1 = typeof(obj1)
    t2 = typeof(obj2)
    if (t1 <: Function) && (t2 <: Function)
        ignoreFunction ? (return true) : (return boolOp(obj1, obj2))
    elseif (t1 <: Number) && (t2 <: Number)
        return boolOp(obj1, obj2)
    elseif t1 != t2 && !ignoreContainerType
        return false
    elseif obj1 isa Union{Array, Tuple}
        length(obj1) != length(obj2) && (return false)
        if ([i isa Number for i in obj1] |> prod) && ([i isa Number for i in obj2] |> prod)
            return boolOp.(obj1, obj2) |> prod
        end
        if obj1 isa Matrix
            size(obj1) != size(obj2) && (return false)
            obj1 = vec(obj1) # Still linked to the original container.
            obj2 = vec(obj2)
        end
        for (i,j) in zip(obj1, obj2)
            res *= hasBoolRelation(boolOp, i, j; ignoreFunction)
            !res && (return false)
        end
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
                                           getproperty(obj2, i); ignoreFunction)
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

    hasBoolRelation(boolOp::Function, obj1, obj2, obj3...; 
                    ignoreFunction=false, ignoreContainerType=false) -> Bool

hasBoolRelation for more than 2 objects. E.g.: `hasBoolRelation(>, a, b, c)` is equivalent 
to `hasBoolRelation(>, a, b) && hasBoolRelation(>, b, c)`.

≡≡≡ Example(s) ≡≡≡

```
julia> begin
           struct S
               a::Int
               b::Float64
           end

           a = S(1, 1.0)
           b = S(2, 0.5)
           c = S(2, 1.5)
           d = S(3, 2.0)
              
           @show hasBooleanRelation(>=, c, b, a)
           @show hasBooleanRelation(>=, d, c, b)
       end
hasBooleanRelation(>=, c, b, a) = false
hasBooleanRelation(>=, d, c, b) = true
true
```
"""
function hasBoolRelation(boolOp::F, obj1, obj2, obj3...; 
                         ignoreFunction=false, 
                         ignoreContainerType=false) where {F<:Function}
    res = hasBoolRelation(boolOp, obj1, obj2; ignoreFunction, ignoreContainerType)
    tmp = obj2
    if res
        for i in obj3[1:end]
            res *= hasBoolRelation(boolOp, tmp, i; ignoreFunction, ignoreContainerType)
            !res && break
            tmp = i
        end
    end
    res
end


"""

   hasEqual(obj1, obj2, obj3...; ignoreFunction=false, ignoreContainerType=false) -> Bool

Compare if two objects are the euqal. 

If `ignoreFunction = true` then the function will pop up a warning message when a field is 
a function.

If `ignoreContainerType = true` then the funtion will ignore the type difference of the 
(outermost) container as long as the inside fields are euqal. 

This function is an instantiation of `hasBoolRelation`.

≡≡≡ Example(s) ≡≡≡

```
julia> begin
           struct S
               a::Int
               b::Float64
           end
           a = S(1, 1.0)
           b = S(1, 1.0)
           c = S(1, 1.0)
           d = S(1, 1.1)

           @show hasEqual(a, b, c)
           @show hasEqual(a, b, c, d)
       end
hasEqual(a, b, c) = true
hasEqual(a, b, c, d) = false
false
```
"""
hasEqual(obj1, obj2, obj3...; ignoreFunction=false, ignoreContainerType=false) = 
hasBoolRelation(==, obj1, obj2, obj3...; ignoreFunction, ignoreContainerType)


"""

    hasIdentical(obj1, obj2, obj3...; 
                 ignoreFunction=false, ignoreContainerType=false) -> Bool

Compare if two objects are the Identical. An instantiation of `hasBoolRelation`.

If `ignoreFunction = true` then the function will pop up a warning message when a field is 
a function.

If `ignoreContainerType = true` then the funtion will ignore the type difference of the 
(outermost) container as long as the inside fields are identical.

This function is an instantiation of `hasBoolRelation`.

≡≡≡ Example(s) ≡≡≡

```
julia> begin
           struct S
               a::Int
               b::Array{Float64, 1}
           end
            
           a = S(1, [1.0, 1.1])
           b = a
           c = b
           d = S(1, [1.0, 1.1])

           @show hasIdentical(a, b, c)
           @show hasIdentical(a, b, c, d)
       end
hasIdentical(a, b, c) = true
hasIdentical(a, b, c, d) = false
false
```
"""
hasIdentical(obj1, obj2, obj3...; ignoreFunction=false, ignoreContainerType=false) = 
hasBoolRelation(===, obj1, obj2, obj3...; ignoreFunction, ignoreContainerType)


"""

    printStyledInfo(str::String; 
                    title::String="INFO:\\n", titleColor::Symbol=:light_blue) -> nothing

Print info with colorful title and automatically highlighted code blocks enclosed by ` `. 

If you want to highlight other contents in different colors, you can also put them inside 
` ` and start it with "///theColorSymbolName///". The available color names follows the 
values of `color` keyword arguement in function `printstyled`. 

NOTE: There can only be one color in one ` ` quote.

≡≡≡ Example(s) ≡≡≡

```
julia> Quiqbox.printStyledInfo("This `///magenta///word` is in color magenta.")
INFO:
```
`This` ``word`` `is in color magenta.`

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

```
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
    c2 = map( x->(x isa Union{Array, Tuple} ? x : [x]), c )
    [(c2...)...]
end

function flatten(c::Tuple)
    c2 = map( x->(x isa Union{Array, Tuple} ? x : [x]), c )
    ((c2...)...,)
end


"""

    arrayAlloc(arrayLength::Int, 
               anExampleOrType::Union{T, Type{T}}) where {T<:Real} -> Ptr{T}

Allocate the memory for an array of specified length and element type, then return the 
pointer to it.
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

    ArrayPointer(arr::Array{<:Real, N}; 
                 showReminder::Bool=true) where {N} -> ArrayPointer{T, N}

Create a `ArrayPointer` that contains a `Ptr` pointing to the actual memory address of the 
(1st element of the) `Array`.

To avoid memory leaking, the user should use `free(x.ptr)` after the usage of 
`x::ArrayPointer` to free the occupied memory.

If `showReminder=true`, the constuctor will pop up a message to remind the user of 
such operation.
"""
struct ArrayPointer{T, N} <: Any
    ptr::Ptr{T}
    arr::Array{T, N}

    function ArrayPointer(arr::Array{<:Real, N}; showReminder::Bool=true) where {N}
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

    markUnique(arr::AbstractArray, args...; compareFunction::Function = hasEqual, kws...)

Return a `markingList` using `Int` number to mark each different elements from 
(and inside) the input argument(s) and a `uniqueList` to contain all the unique 
elements when `compareFunction` is set to `hasEqual` (in default).

`args` and `kws` are positional arguments and keywords argguments respectively as 
parameters of the specified `compareFunction`.

≡≡≡ Example(s) ≡≡≡

```
julia> markUnique([1, [1, 2],"s", [1, 2]])
([1, 2, 3, 2], Any[1, [1, 2], "s"])

julia> begin 
           struct S
               a::Int
               b::Float64
           end
           
           a = S(1, 2.0)
           b = S(1, 2.0)
           c = S(1, 2.1)
           d = a
           
           markUnique(a,b,c,d)
       end
([1, 1, 2, 1], Any[S(1, 2.0), S(1, 2.1)])
```
"""
function markUnique(arr::AbstractArray, args...; 
                    compareFunction::F=hasEqual, kws...) where {F<:Function}
    @assert length(arr) >= 1 "The length of input array should be not less than 1."
    f = (b...)->compareFunction((b..., args...)...; kws...)
    res = Int[1]
    cmprList = eltype(arr)[arr[1]]
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

function getUnique!(arr::Array, args...; 
                    compareFunction::F = hasEqual, kws...) where {F<:Function}
    @assert length(arr) > 1 "The length of input array should be larger than 1."
    f = (b...)->compareFunction((b..., args...)...; kws...)
    cmprList = eltype(arr)[arr[1]]
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
A function that only returns its argument.
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
function recursivelyGet(dict::Dict, startKey)
    res = nothing
    val = get(dict, startKey, false)
    while val != false
        res = val
        val = get(dict, val, false)
    end
    res
end


function isOscillateConverged(sequence::Vector{<:Real}, threshold1::Real, threshold2::Real=threshold1; 
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


function splitTerm(term::Num)
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