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
        b::Vector{Float64}
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
function flatten(c::Array{T}) where {T}
    c2 = map( x->(x isa Union{Array, Tuple} ? x : (x,)), c )
    [(c2...)...]
end

function flatten(c::Tuple)
    c2 = map( x->(x isa Union{Array, Tuple} ? x : (x,)), c )
    ((c2...)...,)
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
itself(x::T) where {T} = x::T


"""
Similar as `replace` but for Symbols.
"""
function replaceSymbol(sym::Symbol, pair::Pair{String, String}; count::Int=typemax(Int))
    replace(sym |> string, pair; count) |> Symbol
end


function renameFunc(fName::Symbol, f::F) where {F<:Function}
    @eval ($(fName))(a...; b...) = $f(a...; b...)
end

renameFunc(fName::String, f) = renameFunc(Symbol(fName), f)


"""
Recursively find the final value using the value of each iteration as the key for the 
next search.
"""
function recursivelyGet(dict::Dict{K, V}, startKey::K, default=Array{V}(undef, 1)[]) where 
                       {K, V}
    res = default
    val = get(dict, startKey, missing)
    while !(val isa Missing)
        res = val
        val = get(dict, val, missing)
    end
    res
end

recursivelyGet(dict::Dict{K, <:Real}, startKey::K) where {K} = 
recursivelyGet(dict, startKey, NaN)


function isOscillateConverged(sequence::Vector{<:Real}, 
                              threshold1::Real, threshold2::Real=threshold1; 
                              leastCycles::Int=1, nPartition::Int=5, 
                              convergeToMax::Bool=false)
    @assert leastCycles>0 && nPartition>1
    len = length(sequence)
    len < leastCycles && (return false)
    slice = len ÷ nPartition
    lastPortion = sequence[max(end-slice, 1) : end]
    remain = sort(lastPortion)[convergeToMax ? (end÷2+1 : end) : (1 : end÷2+1)]
    b = std(remain) < threshold1 && 
        abs(sequence[end] - (convergeToMax ? max(remain...) : min(remain...))) < threshold2
    b, std(lastPortion)
end


splitTerm(trm::Symbolics.Num) = 
splitTermCore(trm.val)::Vector{<:Union{Real, SymbolicUtils.Symbolic}}

@inline function rewriteCore(trm, r)
    res = r(trm)
    res === nothing ? trm : res
end

function splitTermCore(trm::SymbolicUtils.Add)
    r1 = SymbolicUtils.@rule +(~(~xs)) => [i for i in ~(~xs)]
    r1(trm) .|> rewriteTerm
end

rewriteTerm(trm::SymbolicUtils.Add) = splitTermCore(trm)

rewriteTerm(trm::SymbolicUtils.Pow) = itself(trm)

function rewriteTerm(trm::SymbolicUtils.Div)
    r = @rule (~x) / (~y) => (~x) * (~y)^(-1)
    rewriteCore(trm, r)
end

function rewriteTerm(trm::SymbolicUtils.Mul)
    r = SymbolicUtils.@rule *(~(~xs)) => sort([i for i in ~(~xs)], 
                              by=x->(x isa SymbolicUtils.Symbolic)) |> prod
    rewriteCore(trm, r) |> SymbolicUtils.simplify
end

function splitTermCore(trm::SymbolicUtils.Mul)
    r1 = SymbolicUtils.@rule *(~(~xs)) => [i for i in ~(~xs)]
    r2 = SymbolicUtils.@rule +(~(~xs)) => [i for i in ~(~xs)]
    r3 = @acrule ~~vs * exp((~a)*((~x)^2+(~y)^2+(~z)^2)) * 
                        exp(-1*(~a)*((~x)^2+(~y)^2+(~z)^2)) => prod(~~vs)
    trms = rewriteCore(trm, r1)
    idx = findfirst(x-> x isa SymbolicUtils.Add, trms)
    if idx !== nothing
        sumTerm = popat!(trms, idx)
        var = SymbolicUtils.simplify(sort(trms, 
                                          by=x->(x isa SymbolicUtils.Symbolic)) |> prod)
        rewriteCore.((r2(sumTerm) .* var), Ref(r3)) .|> rewriteTerm |> flatten
    else
        [trms |> prod]
    end
end

splitTermCore(trm::SymbolicUtils.Div) = trm |> rewriteTerm |> splitTermCore

splitTermCore(trm) = [trm]


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


# Product Function
struct Pf{C, F} <: ParameterizedFunction{Pf, F}
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

getFunc(::Type{Pf{C, F}}, _=missing) where {C, F} = Pf{C, F}(getFunc(Val(F)))

getFunc(f::Function, _=missing) = itself(f)

getFunc(::Val{F}, failedResult=missing) where {F} = getFunc(F, failedResult)

getFunc(::Val{:itself}, _=missing) = itself


nameOf(f::ParameterizedFunction) = typeof(f)

nameOf(f) = nameof(f)


function arrayDiffCore!(vs::NTuple{N, Array{T}}) where {N, T}
    head = vs[argmin(length.(vs))]
    coms = T[]
    l = length(head)
    sizehint!(coms, l)
    i = 0
    while i < l
        i += 1
        ele = head[i]
        ids = zeros(Int, N)
        flag = false
        for (j, v) in enumerate(vs)
            k = findfirst(isequal(ele), v)
            k === nothing ? (flag=true; break) : (ids[j] = k)
        end
        flag && continue
        for (v, id) in zip(vs, ids)
            popat!(v, id)
        end
        push!(coms, ele)
        i -= 1
        l -= 1
    end
    (coms, vs...)
end

function arrayDiffCore!(v1::Array{T}, v2::Array{T}) where {T}
    a1, a2 = (length(v1) > length(v2)) ? (v2, v1) : (v1, v2)
    coms = T[]
    l = length(a1)
    sizehint!(coms, l)
    i = 0
    while i < l
        i += 1
        j = findfirst(isequal(a1[i]), a2)
        if j !== nothing
            popat!(a1, i)
            push!(coms, popat!(a2, j))
            i -= 1
            l -= 1
        end
    end
    coms, v1, v2
end

arrayDiff!(v1::Array{T}, v2::Array{T}) where {T} = arrayDiffCore!(v1, v2)

arrayDiff!(vs::Vararg{Array{T}, N}) where {T, N} = arrayDiffCore!(vs)

tupleDiff(ts::Vararg{NTuple{<:Any, T}, N}) where {T, N} = arrayDiff!((ts .|> collect)...)


struct FunctionType{F}
    f::Union{Symbol, Type{<:ParameterizedFunction}, Function}

    FunctionType{F}() where {F} = new{F}(F)
    FunctionType(f::F) where {F<:Function} = new{F}(f)
end

FunctionType(s::Symbol) = FunctionType{s}()

getFunc(ft::FunctionType{F}) where {F} = ft.f

function getFuncNum(f::Function, vNum::Symbolics.Num)::Symbolics.Num
    Symbolics.variable(f|>nameOf, T=Symbolics.FnType{Tuple{Any}, Real})(vNum)
end

function getFuncNum(::Pf{C, F}, vNum::Symbolics.Num) where {C, F}
    (C * Symbolics.variable(F, T=Symbolics.FnType{Tuple{Any}, Real})(vNum))::Symbolics.Num
end

function getFuncNum(::FunctionType{F}, vNum::Symbolics.Num) where {F}
    Symbolics.variable(F, T=Symbolics.FnType{Tuple{Any}, Real})(vNum)::Symbolics.Num
end

getFuncNum(::FunctionType{:itself}, vNum::Symbolics.Num) = vNum

getFuncNum(::typeof(itself), vNum::Symbolics.Num) = vNum

function genIndex(index::Int)
    @assert index >= 0
    genIndexCore(index)
end

genIndex(index::Nothing) = genIndexCore(index)

function genIndexCore(index)
    res = reshape(Union{Int, Nothing}[0], ()) |> collect
    res[] = index
    res
end

function genNamedTupleC(name::Symbol, defaultVars::AbstractArray)
    @inline function (t::T) where {T<:NamedTuple}
        container = getfield(Quiqbox, name)
        res = deepcopy(defaultVars)
        keys = fieldnames(container)
        d = Dict(keys .=> collect(1:length(defaultVars)))
        for (val, fd) in zip(t, fieldnames(T))
            res[d[fd]] = val
        end
        container(res...)
    end
end


convertNumber(num::Number, roundDigits::Int=-1, type::Type{<:Number}=Float64) = 
(roundDigits < 0  ?  num  :  round(num, digits=roundDigits)) |> type


fillNumber(num::Number) = fill(num)

fillNumber(num::Array{<:Any, 0}) = itself(num)


@inline genTupleCoords(coords::Vector{<:AbstractArray{<:Real}}) = 
        Tuple((Float64(i[1]), Float64(i[2]), Float64(i[3])) for i in coords)

@inline genTupleCoords(coords::Tuple{Vararg{NTuple{3,Float64}}}) = itself(coords)


@inline arrayToTuple(arr::Array) = Tuple(arr)

@inline arrayToTuple(tpl::Tuple) = itself(tpl)