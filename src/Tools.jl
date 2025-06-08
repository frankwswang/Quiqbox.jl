export markUnique, getUnique!, itself

using Statistics: std, mean
using LinearAlgebra: norm

function isApprox(x::T, y::T; atol=0) where {T}
    if isnan(atol)
        x==y
    else
        isapprox(x, y; atol)
    end
end


sizeOf(arr::AbstractArray) = size(arr)

sizeOf(tpl::Tuple) = (length(tpl),)


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

    flatVectorizeCore(a::Union{AbstractArray, Tuple, NamedTuple}) -> AbstractVector

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> flatVectorize([:one, 2, ([3, [:four, (five=5, six=6)]], "7"), "8", (9.0, 10)])
10-element Vector{Any}:
   :one
  2
  3
   :four
  5
  6
   "7"
   "8"
  9.0
 10.0
```
"""
flatVectorizeCore(a::Any) = fill(a)
flatVectorizeCore(a::AbstractArray) = vec(a)
flatVectorizeCore(a::Union{Tuple, NamedTuple}) = collect(a)

const CollectionType = Union{Tuple, NamedTuple, AbstractArray}

function flatVectorize(a::CollectionType)
    if isempty(a)
        Union{}[]
    else
        mapreduce(vcat, a) do i
            if i isa CollectionType
                flatVectorize(i)
            else
                flatVectorizeCore(i)
            end
        end |> flatVectorizeCore
    end
end


"""

    markUnique(arr::AbstractArray{T}; compareFunction::Function=isequal) where {T} -> 
    Tuple{Vector{Int}, Vector{T}}

Return a `Vector{Int}` whose elements are indices to mark the elements inside `arr` such 
that same element will be marked with same index, and a `Vector{T}` containing all the 
unique elements. The definition of "unique" (or "same") is based on `compareFunction` 
which is set to `Base.isequal` in default.

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
function markUnique(arr::AbstractArray{T}; compareFunction::F=isequal) where 
                   {T, F<:Function}
    cmprList = T[]
    len = length(arr)
    if len == 0
        similar(arr, Int), cmprList
    elseif len == 1
        markList = similar(arr, Int)
        markList[] = firstindex(arr)
        markList, [first(arr)]
    else
        sizehint!(cmprList, len)
        markList = markUniqueCore!(compareFunction, cmprList, arr)
        markList, cmprList
    end
end

function markUnique(tpl::Tuple{Vararg{Any, N}}; compareFunction::F=isequal) where 
                   {N, F<:Function}
    cmprList = getMinimalEleType(tpl)[]
    if N == 0
        (), cmprList
    elseif N == 1
        (firstindex(tpl),), [first(tpl)]
    else
        sizehint!(cmprList, N)
        markList = markUniqueCore!(compareFunction, cmprList, tpl)
        markList, cmprList
    end
end

function markUnique((a, b)::NTuple{2, Any}; compareFunction::F=isequal) where {F<:Function}
    if compareFunction(a, b)
        (1, 1), [a]
    else
        (1, 2), [a, b]
    end
end

function markUniqueCore!(compareFunc::F, compressedSeq::AbstractVector, 
                         sequence::GeneralCollection) where {F<:Function}
    iFirst = firstindex(compressedSeq)
    map(sequence) do ele
        i = iFirst
        isNew = true
        while i <= lastindex(compressedSeq)
            if compareFunc(compressedSeq[i], ele)
                isNew = false
                break
            end
            i += 1
        end
        isNew && push!(compressedSeq, ele)
        i
    end
end


"""

    getUnique!(arr::AbstractVector, args...; compareFunction::Function=isequal, 
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
                    compareFunction::F=isequal, kws...) where {T<:Any, F<:Function}
    isempty(arr) && (return arr)
    cmprList = T[arr[1]]
    delList = Bool[false]
    for ele in view(arr, (firstindex(arr)+1):lastindex(arr))
        isNew = true
        for candid in cmprList
            if compareFunction(candid, ele)
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

const ItsType = typeof(itself)

const ItsTalike = Union{ItsType, typeof(identity)}

@inline themselves(xs::Vararg) = xs

const FOfThem = typeof(themselves)


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


nameOf(f::CompositeFunction) = typeof(f)

nameOf(f) = nameof(f)


function arrayDiffCore!(vs::Tuple{Array{T}, Array{T}, Vararg{Array{T}, N}}) where {N, T}
    head = vs[argmin(length.(vs))]
    coms = T[]
    l = length(head)
    i = 0
    ids = Array{Int}(undef, N+2)
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

tupleDiff(t::NonEmptyTuple{T}, ts::NonEmptyTuple{T}...) where {T} = 
arrayDiffCore!((t, ts...) .|> collect)


deref(a) = getindex(a)
deref(::Nothing) = nothing

# function genArray0D(::Type{T}, val) where {T}
#     res = reshape(T[0], ()) |> collect
#     res[] = val
#     res
# end

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

#! TODO: Find best merge algorithm
function mergeMultiObjs(::Type{T}, mergeTwoObjs::F, objs::AbstractArray{<:T}; 
                        kws...) where {T, F<:Function}
    arr1 = (collect∘vec)(objs)
    arr2 = T[]
    while length(arr1) >= 1
        i = 1
        while i < length(arr1)
            temp = mergeTwoObjs(arr1[i], arr1[i+1]; kws...)
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


mutable struct FuncArgConfig{F<:Function} <: ConfigBox
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


fastIsApprox(x::T1, y::T2=0.0) where {T1, T2} = abs(x - y) < 2(numEps∘promote_type)(T1, T2)


triMatEleNum(n::Int) = n * (n + 1) ÷ 2


# {for j in 1:N, i=1:j} <=> 
# {for n in 1:N(N+1)/2}
function convertIndex1DtoTri2D(n::Int)
    # Using (7 - 0.1) instead of 7 to avoid potential round-off errors.
    j = (floor(Int, (sqrt(8.0n - 6.9) - 1.0)) >> 1) + 1
    i = n - j * (j-1) ÷ 2
    i, j
end

# {for l in 1:N, k in 1:l, j in 1:l, i in 1:ifelse(l==j, k, j)} <=> 
# {for n in (M=N(N+1)÷2; M(M+1)÷2)}
function convertIndex1DtoTri4D(n::Int)
    p, q = convertIndex1DtoTri2D(n)
    i, j = convertIndex1DtoTri2D(p)
    k, l = convertIndex1DtoTri2D(q)
    (i, j, k, l)
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

#> Improved Kahan–Babuška algorithm for array summation
function betterSum(x::AbstractArray{<:Number})
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

# reduceTwoBy(compareFunc::F, o1, o2) where {F<:Function} = 
# compareFunc(o1, o2) ? [o1] : [o1, o2]


struct SplitArg{N, F<:Function} <: Function
    f::F

    SplitArg{N}(f::F) where {N, F} = new{N, F}(f)
end

(f::SplitArg{N, F})(arg::AbstractArray) where {N, F} = f.f(arg[begin:begin+(N-1)]...)

function getTypeParam(::Type{T}, idx::Int, default=Any) where {T}
    hasproperty(T, :parameters) ? T.parameters[idx] : default
end

function genParamTypeVector(::Type{PT1}, ::Type{PT2}, idx::Int=1) where {PT1, PT2}
    T = getTypeParam(PT1, idx)
    ifelse(T===Any, PT2[], PT2{numT}[])
end


function registerObjFrequency(objs::AbstractVector)
    ofDict = Dict(objs .=> 0)
    map(objs) do m
        times = if haskey(ofDict, m)
            ofDict[m] += 1
        else
            get!(ofDict, m, 1)
        end
        (m, times)
    end
end


function intersectMultisetsCore!(transformation::F, 
                                 s1::AbstractVector{T}, s2::AbstractVector) where 
                                {T, F<:Function}
    sMkr1 = map(transformation, s1)
    sMkr2 = map(transformation, s2)

    sMkr1Counted = registerObjFrequency(sMkr1)
    sMkr2Counted = registerObjFrequency(sMkr2)

    sMkrMutual = intersect(sMkr1Counted, sMkr2Counted)

    if !isempty(sMkrMutual)
        sMkr1cDict = Dict(sMkr1Counted .=> eachindex(sMkr1))
        sMkr2cDict = Dict(sMkr2Counted .=> eachindex(sMkr2))
        i1 = getindex.(Ref(sMkr1cDict), sMkrMutual)
        i2 = getindex.(Ref(sMkr2cDict), sMkrMutual)
        deleteat!(s2, sort(i2))
        splice!(s1, sort(i1))
    else
        similar(s1, 0)
    end
end

# If `x==y` is `true`, `transformation(x)==transformation(y)` should always return `true`.
function intersectMultisets!(s1::AbstractVector{T1}, s2::AbstractVector{T2}; 
                             transformation::F=itself) where {T1, T2, F}
    if s1 === s2
        res = copy(s1)
        empty!(s1)
        res
    else
        intersectMultisetsCore!(transformation, s1, s2)
    end
end


function lazyTupleMap(f::F, (obj1, obj2)::NTuple{2, Any}) where {F}
    res1 = f(obj1)
    res2 = obj2 === obj1 ? res1 : f(obj2)
    (res1, res2)
end

function lazyTupleMap(f::F, (obj1, obj2, obj3, obj4)::NTuple{4, Any}) where {F}
    res1 = f(obj1)
    res2 = obj2 === obj1 ? res1 : f(obj2)
    res3 = obj3 === obj1 ? res1 : (obj3 === obj2 ? res2 : f(obj3))
    res3 = if obj4 === obj1
        res1
    else
        obj4 === obj2 ? res2 : (obj4 === obj3 ? res3 : f(obj4))
    end
    (res1, res2, res3, res3)
end


extractRealNumberType(::Type{<:Number}) = Float64

extractRealNumberType(::Type{T}) where {T<:Real} = T

extractRealNumberType(::Type{Complex{T}}) where {T<:Real} = T


function genParametricType(base::UnionAll, typeVars::Tuple)
    type = base{typeVars...}
    for var in reverse(typeVars)
        type = if var isa TypeVar
            UnionAll(var, type)
        else
            type
        end
    end
    type
end

function genParametricType(base::UnionAll, typePars::NamedTuple)
    typeVars = map(keys(typePars), values(typePars)) do k, v
        if v isa Type && !isconcretetype(v)
            TypeVar(k, v)
        else
            v
        end
    end

    genParametricType(base, typeVars)
end