export hasEqual, hasIdentical, hasApprox, flatten, markUnique, getUnique!, itself

using Statistics: std, mean
using LinearAlgebra: norm

"""

    getAtolDigits(::Type{T}) where {T<:Real} -> Int

Return the maximal number of digits kept after rounding of the input real number type `T`.
"""
function getAtolDigits(::Type{T}) where {T<:Real}
    val = log10(T|>numEps)
    res = max(0, -val) |> floor
    if res > typemax(Int)
        throw(DomainError(res, "This value is too large to be converted to Int."))
    else
        Int(res)
    end
end

function getAtolDigits(num::Real)
    if isnan(num)
        getAtolDigits(num|>typeof)
    else
        str = string(num)
        idx1 = findlast('e', str)
        if idx1 === nothing
            idx2 = findlast('.', str)
            length(str) - idx2
        elseif str[idx1+1] == '-'
            parse(Int, str[idx1+2:end])
        else
            0
        end
    end
end


"""

    getAtolVal(::Type{T}) where {T<:Real} -> Real

Return the absolute precision tolerance of the input real number type `T`.
"""
getAtolVal(::Type{T}) where {T<:Real} = ceil(numEps(T)*1.5, sigdigits=1)
getAtolVal(::Type{T}) where {T<:Integer} = one(T)


function roundToMultiOfStep(num::T, step::T) where {T<:Real}
    if isnan(step)
        num
    else
        invStep = inv(step)
        round(num * invStep, RoundNearest) / invStep
    end
end

roundToMultiOfStep(num::T, step::T) where {T<:Integer} = round(T, num/step) * step


nearestHalfOf(val::T) where {T<:Real} = 
roundToMultiOfStep(val/2, floor(numEps(T), sigdigits=1))

nearestHalfOf(val::Integer) = itself(val)


getNearestMid(num1::T, num2::T, atol::Real) where {T} = 
(isnan(atol) || num1==num2) ? num1 : roundToMultiOfStep((num1+num2)/2, atol)


function isApprox(x::T, y::T; atol=0) where {T}
    if isnan(atol)
        x==y
    else
        isapprox(x, y; atol)
    end
end


# Function for submodule loading and integrity checking.
function tryIncluding(subModuleName::String; subModulePath=(@__DIR__)[:]*"/SubModule")
    try
        include(subModulePath*"/"*subModuleName*".jl")
        return true
    catch err
        warning = """
        Submodule `$(subModuleName)` failed loading and won't be useable:

            $(err)

        `///magenta///However, this does not affect the functionality of the main module.`
        """
        printStyledInfo(warning, title="WARNING:\n", titleColor=:light_yellow)
        return false
    end
end


sizeOf(arr::AbstractArray) = size(arr)

sizeOf(tpl::Tuple) = (length(tpl),)


"""

    hasBoolRelation(boolOp::Function, obj1, obj2; ignoreFunction::Bool=false, 
                    ignoreContainer::Bool=false, decomposeNumberCollection::Bool=false) -> 
    Bool

Recursively apply the specified boolean operator `boolOp` to all the fields inside two 
objects (e.g., two `struct`s of the same type). It returns `true` if and only if all 
comparisons performed return `true`. Note that the boolean operator should have method(s) 
defined for all the possible fields inside the compared objects.

If `ignoreFunction = true`, comparisons between Function-type fields will be ignored.

If `ignoreContainer = true`, the difference of the container(s) will be ignored as long as 
the boolean operator returns true for the field(s)/entry(s) from two objects respectively.

If `decomposeNumberCollection = true`, `Tuple{Vararg{Number}}` and `Array{<:Number}` will 
be treated as decomposable containers.

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
function hasBoolRelation(boolOp::F, obj1::T1, obj2::T2; 
                         ignoreFunction::Bool=false, 
                         ignoreContainer::Bool=false, 
                         decomposeNumberCollection::Bool=false) where {T1, T2, F<:Function}
    res = true
    fs1 = fieldnames(T1)
    fs2 = fieldnames(T2)
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
    res
end

hasBoolRelation(boolOp::Function, obj1::F1, obj2::F2; 
                ignoreFunction::Bool=false, ignoreContainer::Bool=false, 
                decomposeNumberCollection::Bool=false) where 
               {StructuredFunction<:F1<:Function, StructuredFunction<:F2<:Function} = 
ifelse(ignoreFunction, true, boolOp(obj1, obj2))

hasBoolRelation(boolOp::Function, obj1::Number, obj2::Number; 
                ignoreFunction::Bool=false, ignoreContainer::Bool=false, 
                decomposeNumberCollection::Bool=false) = 
boolOp(obj1, obj2)

hasBoolRelation(boolOp::Function, obj1::Type{T1}, obj2::Type{T2}; 
                ignoreFunction::Bool=false, ignoreContainer::Bool=false, 
                decomposeNumberCollection::Bool=false) where {T1, T2} = 
boolOp(obj1, obj2)

function hasBoolRelation(boolOp::F, 
                         obj1::Union{AbstractArray, Tuple}, 
                         obj2::Union{AbstractArray, Tuple}; 
                         ignoreFunction::Bool=false, ignoreContainer::Bool=false, 
                         decomposeNumberCollection::Bool=false) where {F<:Function}
    if !decomposeNumberCollection && (eltype(obj1) <: Number) && (eltype(obj2) <: Number)
        return boolOp(obj1, obj2)
    end
    !ignoreContainer && 
    (typejoin(typeof(obj1), typeof(obj2))==Any || sizeOf(obj1)!=sizeOf(obj2)) && 
    (return false)
    length(obj1) != length(obj2) && (return false)
    res = true
    for (i,j) in zip(obj1, obj2)
        res *= hasBoolRelation(boolOp, i, j; ignoreFunction, ignoreContainer, 
                               decomposeNumberCollection)
        !res && (return false)
    end
    res
end
## Refer overload for `ParamBox` to Overload.jl.

"""

    hasBoolRelation(boolOp::Function, obj1, obj2, obj3...; 
                    ignoreFunction::Bool=false, 
                    ignoreContainer::Bool=false,
                    decomposeNumberCollection::Bool=false) -> 
    Bool

Method for more than 2 objects. If returns true if and only if `hasBoolRelation` returns 
true for every unique combination of two objects from the all the input objects under the 
transitive relation. E.g.: `hasBoolRelation(>, a, b, c)` is equivalent to 
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
function hasBoolRelation(boolOp::F, obj1, obj2, obj3, obj4...; 
                         ignoreFunction::Bool=false, 
                         ignoreContainer::Bool=false,
                         decomposeNumberCollection::Bool=false) where {F<:Function}
    res = hasBoolRelation(boolOp, obj1, obj2; ignoreFunction, ignoreContainer)
    tmp = obj2
    if res
        for i in (obj3, obj4...)
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

Compare if two containers (e.g. `struct`) are equal. An instantiation of 
[`hasBoolRelation`](@ref).
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

Compare if two containers (e.g. `struct`) are the Identical. An instantiation of 
[`hasBoolRelation`](@ref).

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

    hasApprox(obj1, obj2, obj3...; ignoreFunction::Bool=false, ignoreContainer::Bool=false,
              decomposeNumberCollection::Bool=false, atol::Real=1e-15) -> 
    Bool

Similar to [`hasEqual`](@ref), except it does not require the `Number`-typed fields 
(e.g. `Float64`) of the compared containers to have the exact same values, but rather the 
approximate values within an error threshold determined by `atol`, like in `isapprox`.
"""
hasApprox(obj1, obj2, obj3...; ignoreFunction::Bool=false, ignoreContainer::Bool=false,
          decomposeNumberCollection::Bool=false, atol::Real=1e-15) = 
hasBoolRelation((x, y)->hasApproxCore(x, y, atol), obj1, obj2, obj3...; ignoreFunction, 
                ignoreContainer, decomposeNumberCollection)

hasApproxCore(obj1::T1, obj2::T2, atol::Real=1e-15) where {T1<:Number, T2<:Number} = 
isapprox(obj1, obj2; atol)

function hasApproxCore(obj1::AbstractArray{<:Number}, obj2::AbstractArray{<:Number}, 
                       atol::Real=1e-15)
    if length(obj1) != length(obj2)
        false
    else
        isapprox.(obj1, obj2; atol) |> all
    end
end

hasApproxCore(obj1::NTuple{N, Number}, obj2::NTuple{N, Number}, 
              atol::Real=1e-15) where {N} = 
isapprox.(obj1, obj2; atol) |> all

hasApproxCore(obj1, obj2, _=1e-15) = (obj1 == obj2)


"""

    printStyledInfo(str::String; title::String="", titleColor::Symbol=:light_blue) -> 
    Nothing

Print info with colorful title and automatically highlighted code blocks enclosed by ` `.

If you want to highlight other contents in different colors, you can also start it with 
"///theColorSymbolName///" and then enclose it with ``. The available color names follows 
the values of `color` keyword argument in function `Base.printstyled`.

**NOTE:** There can only be one color in one ` ` quote.

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

    flatten(a::AbstractVector) -> AbstractVector

Flatten `a::Union{AbstractVector, Tuple}` that contains `AbstractArray`s and/or `Tuple`s. 
Only operate on the outermost container.

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
function flatten(c::AbstractVector{T}) where {T}
    c2 = map( x->(x isa Union{AbstractArray, Tuple} ? x : (x,)), c )
    [(c2...)...]
end

function flatten(c::Tuple)
    c2 = map( x->(x isa Union{AbstractArray, Tuple} ? x : (x,)), c )
    ((c2...)...,)
end

flatten(c::AbstractVector{<:Tuple}) = joinTuple(c...) |> collect

joinTuple(t1::Tuple, t2::Tuple, t3::Tuple...) = (t1..., joinTuple(t2, t3...)...)

joinTuple(t::Tuple) = itself(t)


"""

    markUnique(arr::AbstractArray{T}, args...; 
               compareFunction::Function=hasEqual, kws...) where {T} -> 
    Tuple{Vector{Int}, Vector{T}}

Return a `Vector{Int}` whose elements are indices to mark the elements inside `arr` such 
that same element will be marked with same index, and a `Vector{T}` containing all the 
unique elements. The definition of "unique" (or "same") is based on `compareFunction` 
which is set to [`hasEqual`](@ref) in default. `args` and `kws` are placeholders for the 
positional arguments and keyword arguments for `compareFunction` respectively.

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
    isempty(arr) && (return arr, T[])
    f = (a, b) -> compareFunction(a, b, args...; kws...)
    res = Int[1]
    cmprList = T[first(arr)]
    for ele in view(arr, (firstindex(arr)+1):lastindex(arr))
        j = firstindex(cmprList)
        isNew = true
        while j <= lastindex(cmprList)
            if f(cmprList[j], ele)
                isNew = false
                break
            end
            j += 1
        end
        push!(res, j)
        isNew && push!(cmprList, ele)
    end
    res, cmprList
end

"""

    getUnique!(arr::AbstractVector, args...; compareFunction::Function = hasEqual, 
               kws...) -> 
    AbstractVector

Similar to [`markUnique`](@ref) but instead, directly return the modified `arr` so that the 
repeated entries are deleted.

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
function getUnique!(arr::AbstractVector{T}, args...; 
                    compareFunction::F = hasEqual, kws...) where {T<:Any, F<:Function}
    isempty(arr) && (return arr)
    f = (a, b) -> compareFunction(a, b, args...; kws...)
    cmprList = T[arr[1]]
    delList = Bool[false]
    for ele in view(arr, (firstindex(arr)+1):lastindex(arr))
        isNew = true
        for candid in cmprList
            if f(candid, ele)
                isNew = false
                break
            end
        end
        isNew && push!(cmprList, ele)
        push!(delList, !isNew)
    end
    deleteat!(arr, delList)
end


"""

    itself(::T) -> T

A dummy function that returns its argument.
"""
@inline itself(x) = x

const iT = typeof(itself)

@inline themselves(xs::Vararg) = xs


"""

    replaceSymbol(sym::Symbol, pair::Pair{String, String}; count::Int=typemax(Int)) -> 
    Symbol

Similar as `Base.replace` but for Symbols.
"""
function replaceSymbol(sym::Symbol, pair::Pair{String, String}; count::Int=typemax(Int))
    replace(sym |> string, pair; count) |> Symbol
end


function isOscillateConvergedHead(seq, minLen, maxRemains)
    minLen > 1 || 
    throw(DomainError(minLen, "`minLen` should be larger than 1."))
    len = length(seq)
    len < minLen && (return [zero(seq[begin])])
    maxRemains > 1 || 
    throw(DomainError(maxRemains, "`maxRemains` should be larger than 1."))
    slice = min(maxRemains, len÷3+1)
    seq[(end-slice+1) : end]
end

function isOscillateConverged(seq::AbstractVector{<:Real}, 
                              tarThreshold::Real, 
                              target::Real=NaN, 
                              stdThreshold::Real=0.8tarThreshold; 
                              minLen::Int=5, maxRemains::Int=10, 
                              convergeToMax::Bool=false)
    tailEles = isOscillateConvergedHead(seq, minLen, maxRemains)
    length(tailEles) < 2 && (return (false, tailEles))
    valleys = sort!(tailEles)[ifelse(convergeToMax, (end÷2+1 : end), (1 : end÷2+1))]
    if isnan(target)
        target = convergeToMax ? valleys[argmax(valleys)] : valleys[argmin(valleys)]
    end
    b = std(valleys) <= stdThreshold && norm(target - seq[end]) <= tarThreshold
    # b || @show ("Scalar", std(valleys), norm(target - seq[end]))
    b, std(tailEles)
end

function isOscillateConverged(seq::AbstractVector{<:Array{<:Real}}, 
                              tarThreshold::Real, 
                              target::Real=0, 
                              stdThreshold::Real=0.75tarThreshold; 
                              minLen::Int=5, maxRemains::Int=10)
    isnan(target) && throw(AssertionError("`target` cannot be `NaN`."))
    tailEles = isOscillateConvergedHead(seq, minLen, maxRemains)
    length(tailEles) < 2 && (return (false, tailEles))
    lastPortionDiff = [norm(target .- val) for val in tailEles]
    lastDiff = lastPortionDiff[end]
    valleyIds = sortperm(lastPortionDiff)[begin : end÷2+1]
    b = lastDiff <= tarThreshold && all(i <= stdThreshold for i in std(tailEles[valleyIds]))
    # b || @show ("Vector", std(tailEles[valleyIds]), lastDiff)
    b, std(tailEles)
end


function groupedSort(v::AbstractVector{T}, sortFunction::F=itself) where {T, F<:Function}
    sortedArr = sort(v, by=sortFunction)
    state1 = 1
    groups = Vector{T}[]
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
        getproperty(Quiqbox, fSym)
    catch
        try
            getproperty(Main, fSym)
        catch
            try
                fSym |> string |> Meta.parse |> eval
            catch
                (_) -> failedResult
            end
        end
    end
end

getFunc(f::Function, _=missing) = itself(f)


nameOf(f::StructuredFunction) = typeof(f)

nameOf(f) = nameof(f)


function arrayDiffCore!(vs::NTuple{N, Array{T}}) where {N, T}
    head = vs[argmin(length.(vs))]
    coms = T[]
    l = length(head)
    i = 0
    ids = Array{Int}(undef, N)
    while i < l
        i += 1
        ele = head[i]
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

function arrayDiffCore!((v1, v2)::NTuple{2, Array{T}}) where {T}
    coms = T[]
    l = length(v1)
    i = 0
    while i < l
        i += 1
        j = findfirst(isequal(v1[i]), v2)
        if j !== nothing
            popat!(v1, i)
            push!(coms, popat!(v2, j))
            i -= 1
            l -= 1
        end
    end
    coms, v1, v2
end

tupleDiff(ts::Vararg{NTuple{<:Any, T}}) where {T} = arrayDiffCore!(ts .|> collect)


function genIndex(index::Int)
    index < 0 && throw(DomainError(index, "`index` should be non-negative."))
    genIndexCore(index)
end

genIndex(::Nothing) = genIndexCore(nothing)
genIndex() = genIndex(nothing)

function genIndexCore(index)
    res = reshape(Union{Int, Nothing}[0], ()) |> collect
    res[] = index
    res
end

function genNamedTupleC(name::Symbol, defaultVars::AbstractVector)
    @inline function (t::T) where {T<:NamedTuple}
        container = getproperty(Quiqbox, name)
        res = deepcopy(defaultVars)
        keys = fieldnames(container)
        d = Dict(keys .=> collect(1:length(defaultVars)))
        for (val, fd) in zip(t, fieldnames(T))
            res[d[fd]] = val
        end
        container(res...)
    end
end


fillObj(@nospecialize(obj::Any)) = fill(obj)

fillObj(@nospecialize(obj::Array{<:Any, 0})) = itself(obj)


tupleObj(@nospecialize(obj::Any)) = (obj,)

tupleObj(@nospecialize(obj::Tuple{Any})) = itself(obj)


@inline unzipObj(obj::Tuple{T}) where {T} = obj[begin]

unzipObj(@nospecialize(obj::Union{Tuple, Array})) = 
ifelse(length(obj)==1, obj[begin], obj)

unzipObj(@nospecialize(obj::Any)) = itself(obj)


arrayToTuple(@nospecialize(arr::AbstractArray)) = Tuple(arr)

arrayToTuple(@nospecialize(tpl::Tuple)) = itself(tpl)


genTupleCoords(::Type{T1}, 
               coords::Tuple{AbstractVector{<:T2}, Vararg{AbstractVector{<:T2}}}) where 
              {T1, T2} = 
map(x->Tuple(x.|>T1), coords)

genTupleCoords(::Type{T1}, coords::AbstractVector{<:AbstractVector{<:T2}}) where {T1, T2} = 
genTupleCoords(T1, coords|>Tuple)

genTupleCoords(::Type{T}, coords::Tuple{NTuple{D, T}, Vararg{NTuple{D, T}}}) where {D, T} = 
itself(coords)

genTupleCoords(::Type{T}, coords::AbstractVector{NTuple{D, T}}) where {D, T} = 
Tuple(coords)


uniCallFunc(f::F, argsOrder::NTuple{N, Int}, args...) where {F<:Function, N} = 
f(getindex.(Ref(args), argsOrder)...)


function mergeMultiObjs(::Type{T}, merge2Ofunc::F, o1::T, o2::T, o3::T, o4::T...; 
                        kws...) where {T, F<:Function}
    arr1 = T[o1, o2, o3, o4...]
    arr2 = T[]
    while length(arr1) >= 1
        i = 1
        while i < length(arr1)
            temp = merge2Ofunc(arr1[i], arr1[i+1]; kws...)
            if eltype(temp) <: T && length(temp) == 1
                arr1[i] = temp[]
                popat!(arr1, i+1)
            else
                reverse!(arr1, i, i+1)
                i += 1
            end
        end
        push!(arr2, popat!(arr1, i))
    end
    arr2
end


isNaN(::Any) = false
isNaN(n::Number) = isnan(n)


getBool(bl::Bool) = itself(bl)
getBool(::Val{BL}) where {BL} = BL::Bool


function skipIndices(arr::AbstractArray{Int}, ints::AbstractVector{Int})
    all(i > 0 for i in arr) || 
    throw(DomainError(arr, "Every element of `arr` should be positive."))
    if isempty(ints)
        arr
    else
        all(i > 0 for i in ints) || 
        throw(DomainError(ints, "Every element of `ints` should be positive."))
        maxIdx = maximum(arr)
        maxIdxN = maxIdx + length(ints)
        ints = filter!(x->x<=maxIdxN, sort(ints))
        idsN = deleteat!(collect(1:maxIdxN), ints)
        map(arr) do x
            idsN[x]
        end
    end
end


lazyCollect(@nospecialize t::Tuple) = collect(t)

lazyCollect(@nospecialize a::AbstractArray) = itself(a)

lazyCollect(@nospecialize o::Any) = collect(o)


asymSign(a::Real) = ifelse(a<0, -1, 1)


mutable struct FuncArgConfig{F<:Function} <: ConfigBox{Any, FuncArgConfig, F}
    posArgs::Vector{<:Any}
    keyArgs::Vector{<:Pair{Symbol}}
end

FuncArgConfig(::F, posArgs, keyArgs) where {F} = FuncArgConfig{F}(posArgs, keyArgs)
FuncArgConfig(::F, posArgs) where {F} = FuncArgConfig{F}(posArgs, Pair{Symbol}[])
FuncArgConfig(::F) where {F} = FuncArgConfig{F}(Any[], Pair{Symbol}[])

(facfg::FuncArgConfig{F})(f::F, args...; kws...) where {F} = 
f(args..., facfg.posArgs...; kws..., facfg.keyArgs...)


numEps(::Type{T}) where {T<:AbstractFloat} = eps(T)
numEps(::Type{T}) where {T<:Integer} = one(T)
numEps(::Type{Complex{T}}) where {T} = eps(T)


function genAdaptStepBl(infoLevel::Int, maxStep::Int)
    if infoLevel == 0
        (::Int)->false
    elseif infoLevel > 5
        (::Int)->true
    else
        function (i::Int)
            power = ceil(2 * maxStep^0.2)
            bound = 1 / 5^power
            i % floor(log(abs(infoLevel^power*bound*maxStep)+2, i) + 1) == 0
        end
    end
end

const FullStepLevel = 5


function shiftLastEle!(v, shiftVal)
    s = sum(v)
    shiftVal = abs(shiftVal)
    offset = if s < shiftVal
        signedShift = asymSign(s)*shiftVal
        s += signedShift
        v[end] += signedShift
        signedShift
    else
        zero(s)
    end
    s, offset
end


getValParm(::Val{T}) where {T} = T
getValParm(::Type{Val{T}}) where {T} = T


fct(a::Real) = factorial(a|>Int)


δ(a::Int, b::Int) = Int(a == b)


fastIsApprox(x::T1, y::T2=0.0) where {T1, T2} = abs(x - y) < 2(numEps∘promote_type)(T1, T2)


triMatEleNum(n::Int) = n * (n + 1) ÷ 2


function convert1DidxTo2D(n::Int, k::Int)
    bl = iseven(n)
    nRow = n + bl
    j, i = fldmod1(k, nRow)
    if j > i - bl
        i = n - i + 1
        j = n - j + 2 - bl
    else
        i -= bl
    end
    i, j
end


function convert1DidxTo4D(n::Int, m::Int)
    # Original solution
    # rangeShifter = 1e-12 # can't be too large or too small
    # nG = floor(Int, (sqrt(1+8m) + 1)/2 - rangeShifter)
    nGupper = 0.5*(sqrt(1+8m) + 1)
    nG = floor(nGupper)
    nG = Int(nG - (nG==nGupper))
    l, k = convert1DidxTo2D(n, nG)
    rsd = m - nG*(nG-1)÷2
    j, i = convert1DidxTo2D(n, rsd)
    i, j, k, l
end


mapMapReduce(tp::NTuple{N}, fs::NTuple{N, Function}, op::F=*) where {N, F} = 
reduce(op, map((x, y)->y(x), tp, fs))

mapMapReduce(tp::NTuple{N}, f::F1, op::F2=*) where {N, F1, F2} = reduce(op, map(f, tp))


rmsOf(arr::AbstractArray) = norm(arr) / (sqrt∘length)(arr)


function keepOnly!(a::AbstractArray, idx::Int)
    length(a) < idx && error()
    e = a[idx]
    deleteat!(a, (firstindex(a)+1):lastindex(a))
    a[] = e
end


function betterSum(x) # Improved Kahan–Babuška algorithm fro array summation
    xFirxt = first(x)
    s = xFirxt .- xFirxt
    c = xFirxt .- xFirxt

    for xi in x
        t = s .+ xi
        c += map(s, t, xi) do si, ti, xij
            if abs(si) >= abs(xij)
                (si - ti) + xij
            else
                (xij - ti) + si
            end
        end
        s = t
    end
    s .+ c
end

n1Power(x::Int) = ifelse(isodd(x), -1, 1)
n1Power(x::Int, ::Type{T}) where {T} = ifelse(isodd(x), T(-1), T(1))
n1Power(x::NTuple{N, Int}) where {N} = (n1Power∘sum)(x)
n1Power(x::NTuple{N, Int}, ::Type{T}) where {N, T} = n1Power(sum(x), T)