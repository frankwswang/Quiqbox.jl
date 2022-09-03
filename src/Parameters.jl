export ParamBox, inValOf, inSymOf, inSymOfCore, inSymValOf, outValOf, outSymOf, 
       outSymOfCore, outSymValOf, dataOf, mapOf, getVar, getVarDict, outValCopy, inVarCopy, 
       fullVarCopy, enableDiff!, disableDiff!, isDiffParam, toggleDiff!, isDepParam, 
       changeMapping

export FLevel

# Function Level
struct FLevel{L} <: MetaParam{FLevel} end

FLevel(::Type{itselfT}) = FLevel{0}
FLevel(::Type{typeof(Base.identity)}) = FLevel{0}
FLevel(::Type{<:Function}) = FLevel{1}
FLevel(::Type{StructFunction{F}}) where {F} = FLevel(F)
FLevel(::Type{<:ParameterizedFunction{T1, T2}}) where {T1, T2} = 
FLevel{getFLevel(T1)+getFLevel(T2)}
FLevel(::Type{DressedItself{L}}) where {L} = FLevel{L}
FLevel(::F) where {F<:Function} = FLevel(F)
const FI = FLevel(itself)

getFLevel(::Type{FLevel{L}}) where {L} = L
getFLevel(::Type{F}) where {F<:Function} = (getFLevel∘FLevel)(F)
getFLevel(::F) where {F<:Function} = getFLevel(F)
getFLevel(sym::Symbol) = (getFLevel∘getFunc)(sym)

"""

    ParamBox{T, V, FL<:FLevel} <: DifferentiableParameter{T, ParamBox}

Parameter container that can enable differentiation.

≡≡≡ Field(s) ≡≡≡

`data::Array{Array{T, 0}}`: The container of the data (i.e. the input variable) stored in 
the `ParamBox` that can be accessed by syntax `[]`. The value of the data stored in an 
arbitrary `ParamBox{T}` `pb` can be modified using the syntax `pb[]`.

`dataName::Symbol`: The name of the input variable.

`map::Function`: The mapping of the data within the same domain (`.map(::T)->T`). The 
result (i.e., the value of the output variable) can be accessed by syntax `()`.

`canDiff::Array{Bool, 0}`: Indicator of whether the output variable is "marked" as 
differentiable with respect to the input variable in the differentiation process. In other 
words, it determines whether the output variable represented by the `ParamBox` is treated 
as a dependent variable or an independent variable.

`index::Array{<:Union{Int, Nothing}, 0}`: Additional index assigned to the `ParamBox`.

≡≡≡ Initialization Method(s) ≡≡≡

    ParamBox(data, name=:undef, dataName=Symbol($(IVsymSuffix), name); 
             index=nothing, canDiff=false) -> 
    ParamBox{<:Any, dataName, $(FI)}

    ParamBox(data, name, mapFunction, dataName=Symbol($(IVsymSuffix), name); 
             index=nothing, canDiff=true) -> 
    ParamBox{<:Any, name}

=== Positional argument(s) ===

`data::Union{Array{T, 0}, T}`: The input variable (`Array{T, 0}`) or the value of it to be 
stored. If the latter is the first argument, then it will directly be assigned to `.data[]`.

`name::Symbol`: Specify the name of the output variable represented by the constructed 
`ParamBox`. It's equal to the type parameter `V` of the constructed `ParamBox`.

`mapFunction::Function`: The mapping (`mapFunction(::T)->T`) of the stored data (input 
variable), which will be assigned to the field `.map`. After constructing a `ParamBox`, e.g 
`pb = ParamBox(x, yName, f)`, `pb[]` returns the value of `x`, and `pb()` returns the value 
of `f(x)::T`. When `mapFunction` is not provided, `.map` is set to [`itself`](@ref) that 
maps the stored data to an identical output variable.

`dataName::Symbol`: The name of the stored data, i.e, the name of the input variable.

=== Keyword argument(s) ===

`index::Union{Int, Nothing}`: The index of the constructed `ParamBox`. It's should be left 
with its default value unless the user plans to utilize the index of a `ParamBox` for 
specific application other than differentiation.

`canDiff::Bool`: Determine whether the output variable is marked as "differentiable" with 
respect to the input variable.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> ParamBox(1.0)
ParamBox{Float64, :undef, $(FI)}(1.0)[∂][undef]

julia> ParamBox(1.0, :a)
ParamBox{Float64, :a, $(FI)}(1.0)[∂][a]

julia> ParamBox(1.0, :a, abs)
ParamBox{Float64, :a, $(FLevel(abs))}(1.0)[∂][x_a]
```

**NOTE 1:** The rightmost "`[∂][IV]`" in the printed info indicates the differentiability 
and the name (with an assigned index) of the independent variable held by the `ParamBox`. 
When the `ParamBox` is marked as a "differentiable parameter", "`[∂]`" is bold and green 
instead of just being grey, and `IV` is the name of the input variable.

**NOTE 2:** The output variable of a `ParamBox` is normally used to differentiate a 
parameter functional (e.g., the Hartree-Fock energy). However, the derivative with respect 
to the corresponding input variable can also be computed to when the `ParamBox` is marked 
as differentiable.
"""
struct ParamBox{T, V, FL<:FLevel} <: DifferentiableParameter{T, ParamBox}
    data::Array{Array{T, 0}, 0}
    dataName::Symbol
    map::Function
    canDiff::Array{Bool, 0}
    index::Array{<:Union{Int, Nothing}, 0}

    function ParamBox{T, V}(f::F, data::Array{T, 0}, index, canDiff, dataName) where 
                           {T, V, F<:Function}
        @assert Base.return_types(f, (T,))[1] == T
        new{T, V, FLevel(F)}(fill(data), dataName, f, canDiff, index)
    end

    ParamBox{T, V}(data::Array{T, 0}, index, canDiff, dataName) where {T, V} = 
    new{T, V, FI}(fill(data), dataName, itself, canDiff, index)
end

function ParamBox(::Val{V}, mapFunction::F, data::Array{T, 0}, 
                  index=genIndex(nothing), canDiff=fill(true), 
                  dataName=Symbol(IVsymSuffix, V)) where {V, F<:Function, T}
    if getFLevel(F) != 0
        fSym = mapFunction |> nameOf
        fStr = fSym |> string
        if startswith(fStr, '#')
            idx = parse(Int, fStr[2:end])
            fSym = "f_" * string(V) * numToSubs(idx) |> Symbol
            mapFunction = renameFunc(fSym, mapFunction)
        end
    end
    ParamBox{T, V}(mapFunction, data, index, canDiff, dataName)
end

ParamBox(::Val{V}, data::Array{T, 0}, index=genIndex(nothing), 
         canDiff=fill(false), dataName=Symbol(IVsymSuffix, V)) where {V, T} = 
ParamBox{T, V}(data, index, canDiff, dataName)

ParamBox(::Val{V}, ::itselfT, data::Array{T, 0}, index=genIndex(nothing), 
         canDiff=fill(false), dataName=Symbol(IVsymSuffix, V)) where {V, T} = 
ParamBox{T, V}(data, index, canDiff, dataName)

ParamBox(::Val{V}, pb::ParamBox{T, <:Any, FI}) where {V, T} = 
ParamBox{T, V}(pb.data[], pb.index, pb.canDiff, pb.dataName)

ParamBox(::Val{V}, pb::ParamBox{T}; canDiff::Array{Bool, 0}=pb.canDiff) where {T, V} = 
ParamBox{T, V}(pb.map, pb.data[], pb.index, canDiff, pb.dataName)

ParamBox(data::T, name::Symbol=:undef, dataName::Symbol=Symbol(IVsymSuffix, name); 
         index::Union{Int, Nothing}=nothing, canDiff::Bool=false) where {T} = 
ParamBox(Val(name), fillObj(data), genIndex(index), fill(canDiff), dataName)

ParamBox(data::T, name::Symbol, mapFunction::F, dataName::Symbol=Symbol(IVsymSuffix, name); 
         index::Union{Int, Nothing}=nothing, canDiff::Bool=true) where {T, F<:Function} = 
ParamBox(Val(name), mapFunction, fillObj(data), genIndex(index), fill(canDiff), dataName)


"""

    inValOf(pb::ParamBox{T}) where {T} -> T

Return the value of the input variable of `pb`. Equivalent to `pb[]`.
"""
@inline inValOf(pb::ParamBox) = pb.data[][]


"""

    inSymOf(pb::ParamBox) -> Symbol

Return the name (with the index if available) of the input variable of `pb`.
"""
inSymOf(pb::ParamBox) = ( string(inSymOfCore(pb)) * numToSubs(pb.index[]) ) |> Symbol


"""

    inSymValOf(pb::ParamBox{T}) where {T} -> ::Pair{Symbol, T}

Return a `Pair` of the name (with the index if available) and the value of the input 
variable of `pb`.
"""
@inline inSymValOf(pb::ParamBox{T}) where {T} = (inSymOf(pb) => pb.data[][])


"""

    outValOf(pb::ParamBox) -> Number

Return the value of the output variable of `pb`. Equivalent to `pb()`.
"""
@inline outValOf(pb::ParamBox) = callGenFunc(pb.map, pb.data[][])

@inline outValOf(pb::ParamBox{<:Any, <:Any, FI}) = inValOf(pb)

(pb::ParamBox)() = outValOf(pb)
# (pb::ParamBox)() = Base.invokelatest(pb.map, pb.data[][])::Float64


"""

    outSymOf(pb::ParamBox) -> Symbol

Return the name (with the index if available) of the output variable of `pb`.
"""
outSymOf(pb::ParamBox) = ( string(outSymOfCore(pb)) * numToSubs(pb.index[]) ) |> Symbol


"""

    outSymValOf(pb::ParamBox) -> ::Pair{Symbol, T}

Return a `Pair` of the name (with the index if available) and the value of the output 
variable of `pb`.
"""
@inline outSymValOf(pb::ParamBox{T}) where {T} = (outSymOf(pb) => outValOf(pb))


"""

    inSymOfCore(pb::ParamBox) -> Symbol

Return the `Symbol` of the input variable of `pb`.
"""
@inline inSymOfCore(pb::ParamBox) = pb.dataName


"""

    outSymOfCore(pb::ParamBox) -> Symbol

Return the `Symbol` of the output variable of `pb`.
"""
@inline outSymOfCore(::ParamBox{<:Any, V}) where {V} = V


getTypeParams(::ParamBox{T, V, FL}) where {T, V, FL} = (T, V, FL)


"""

    dataOf(pb::ParamBox{T}) where {T} -> Array{T, 0}

Return the 0-D `Array` containing data stored in `pb`.
"""
@inline dataOf(pb::ParamBox) = pb.data[]


"""

    mapOf(pb::ParamBox) -> Function

Return the mapping function of `pb`.
"""
@inline mapOf(pb::ParamBox) = pb.map


"""

    getVar(pb::ParamBox{T}, forDifferentiation::Bool=false) -> Symbol

Return the name of (the output variable of) `pb`. If `forDifferentiation` is set to `true`, 
the name of the independent variable held by `pb` is returned.
"""
function getVar(pb::ParamBox, forDifferentiation::Bool=false)
    if forDifferentiation
        getVarCore(pb)[end][1]
    else
        outSymOf(pb)
    end
end

function getVarCore(pb::ParamBox{T}) where {T}
    outVarVal = outSymOf(pb)=>outValOf(pb)
    isDiffParam(pb) ? Pair{Symbol, T}[outVarVal, inSymOf(pb)=>inValOf(pb)] : 
                      Pair{Symbol, T}[outVarVal]
end


"""

    getVarDict(obj::Union{ParamBox, Tuple{Vararg{ParamBox}}, AbstractArray{<:ParamBox}}) -> 
    Dict{Symbol}

Return a `Dict` that stores the independent variable(s) of the parameter container(s) and 
its(their) corresponding value(s).

**NOTE: Once `obj` is mutated, the generated `Dict` may no longer be up to date.**
"""
getVarDict(pbs::AbstractArray{<:ParamBox}) = vcat(getVarCore.(pbs)...) |> Dict
getVarDict(pbs::Tuple{Vararg{ParamBox}}) = getVarDict(pbs |> collect)
getVarDict(pbs::ParamBox) = getVarCore(pbs) |> Dict


"""

    outValCopy(pb::ParamBox{T, V}) where {T} -> ParamBox{T, V, $(FI)}

Return a new `ParamBox` of which the input variable is a **deep copy** of the output 
variable of `pb`.
"""
outValCopy(pb::ParamBox{<:Any, V}) where {V} = 
ParamBox(Val(V), fill(pb()), genIndex(nothing))


"""

    inVarCopy(pb::ParamBox) -> ParamBox{<:Number, <:Any, $(FI)}

Return a new `ParamBox` of which the input variable is a **shallow copy** of the input 
variable of `pb`.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> pb1 = ParamBox(-2.0, :a, abs)
ParamBox{Float64, :a, $(FLevel(abs))}(-2.0)[∂][x_a]

julia> pb2 = inVarCopy(pb1)
ParamBox{Float64, :a, $(FI)}(-2.0)[∂][a]

julia> pb1[] == pb2[] == -2.0
true

julia> pb1[] = 1.1
1.1

julia> pb2[]
1.1
```
"""
inVarCopy(pb::ParamBox{<:Any, V}) where {V} = 
ParamBox(Val(V), pb.data[], genIndex(nothing), fill(false), pb.dataName)


"""

    fullVarCopy(pb::T) where {T<:ParamBox} -> T

A shallow copy of the input `ParamBox`.
"""
fullVarCopy(pb::ParamBox{<:Any, V}) where {V} = ParamBox(Val(V), pb)


const NoDiffMark = superscriptSym['!']


"""

    enableDiff!(pb::ParamBox) -> ParamBox

Mark the input `pb` as "differentiable" and then return it.
"""
function enableDiff!(pb::ParamBox)
    pb.canDiff[] = true
    pb
end


"""

    disableDiff!(pb::ParamBox) -> ParamBox

Mark the `pb` as "non-differentiable" and then return it.
"""
function disableDiff!(pb::ParamBox)
    pb.canDiff[] = false
    pb
end


"""

    isDiffParam(pb::ParamBox) -> Bool

Return the Boolean value of if `pb` is differentiable.
"""
isDiffParam(pb::ParamBox) = pb.canDiff[]


"""

    toggleDiff!(pb::ParamBox) -> Bool

Toggle the differentiability of the input `pb` and then return it.
"""
toggleDiff!(pb::ParamBox) = begin pb.canDiff[] = !isDiffParam(pb) end


"""

    isDepParam(pb::ParamBox) -> Bool

Return the Boolean value of if `pb` is considered a dependent parameter that is a 
differentiable function with respect to the input variable it stores.
"""
isDepParam(pb::ParamBox{<:Any, <:Any, FI}) = false
isDepParam(pb::ParamBox{<:Any, <:Any, FL}) where {FL} = isDiffParam(pb)


isNotDiffNorInVar(pb::ParamBox{<:Any, <:Any, FI}) = false
isNotDiffNorInVar(pb::ParamBox{<:Any, <:Any, FL}) where {FL} = !isDiffParam(pb)


"""

    changeMapping(pb::ParamBox, mapFunction::Function, outputName::Symbol=V; 
                  canDiff::Bool=isDiffParam(pb)) -> 
    ParamBox{T, outputName}

Change the mapping function of `pb`. The name of the output variable of the returned 
`ParamBox` can be specified by `outputName`, and its differentiability is determined by 
`canDiff`.
"""
changeMapping(pb::ParamBox{T, V, FL}, mapFunction::F, outputName::Symbol=V; 
              canDiff::Bool=isDiffParam(pb)) where {T, V, FL, F<:Function} = 
ParamBox(Val(outputName), mapFunction, pb.data[], 
         genIndex( ifelse(canDiff, pb.index[], nothing) ), fill(canDiff), pb.dataName)


compareParamBoxCore1(pb1::ParamBox, pb2::ParamBox) = (pb1.data[] === pb2.data[])

compareParamBoxCore2(pb1::ParamBox{<:Any, V1}, pb2::ParamBox{<:Any, V2}) where {V1, V2} = 
V1==V2 && compareParamBoxCore1(pb1, pb2) && (typeof(pb1.map) === typeof(pb2.map))

compareParamBoxCore2(pb1::ParamBox{<:Any, V1, FI}, 
                     pb2::ParamBox{<:Any, V2, FI}) where {V1, V2} = 
V1==V2 && compareParamBoxCore1(pb1, pb2)

@inline function compareParamBox(pb1::ParamBox, pb2::ParamBox)
    ifelse(( (bl=isDiffParam(pb1)) == isDiffParam(pb2) ),
        ifelse( bl, 
            compareParamBoxCore1(pb1, pb2), 

            compareParamBoxCore2(pb1, pb2)
        ),

        false
    )
end


function addParamBox(pb1::ParamBox{T, V, FI}, pb2::ParamBox{T, V, FI}, 
                     roundAtol::Real=nearestHalfOf(getAtolVal(T))) where {T, V}
    if compareParamBox(pb1, pb2)
        mulParamBox(2, pb1)
    else
        ParamBox(Val(V), itself, fill(roundToMultiOfStep(pb1[] + pb2[], roundAtol)))
    end
end

function addParamBox(pb1::ParamBox{T, V, FL1}, pb2::ParamBox{T, V, FL2}, 
                     roundAtol::Real=nearestHalfOf(getAtolVal(T))) where {T, V, FL1, FL2}
    if isDiffParam(pb1) && compareParamBox(pb1, pb2)
        # bl = isDiffParam(pb1)
        # pbs = (pb1, pb2)
        # pb1, pb2 = bl ? pbs : outValCopy.(pbs)
        ParamBox(Val(V), combineParFunc(+, pb1.map, pb2.map), pb1.data[], 
                 genIndex(nothing), fill(pb1.canDiff[]), min(pb1.dataName, pb2.dataName))
    else
        ParamBox(Val(V), fill(roundToMultiOfStep(pb1() + pb2(), roundAtol)))
    end
end


function mulParamBox(c::T1, pb::ParamBox{T2, V}, 
                     roundAtol::Real=nearestHalfOf(getAtolVal(T2))) where {T1, T2, V}
    if isDiffParam(pb)
        mapFunc = Pf(convert(T2, roundToMultiOfStep(c, roundAtol)), pb.map)
        ParamBox(Val(V), mapFunc, pb.data[], genIndex(nothing), fill(true), pb.dataName)
    else
        ParamBox(Val(V), itself, 
                 fill(roundToMultiOfStep(convert(T2, pb()*c), roundAtol)))
    end
end

function mulParamBox(pb1::ParamBox{T, V, FI}, pb2::ParamBox{T, V, FI}, 
                     roundAtol::Real=nearestHalfOf(getAtolVal(T))) where {T, V}
    if isDiffParam(pb1) && compareParamBox(pb1, pb2)
        ParamBox(Val(V), Xf(2, itself), pb1.data[], genIndex(nothing), 
                 fill(pb1.canDiff[]), min(pb1.dataName, pb2.dataName))
    else
        ParamBox(Val(V), fill(roundToMultiOfStep(pb1[] * pb2[], roundAtol)))
    end
end

function mulParamBox(pb1::ParamBox{T, V, FL1}, pb2::ParamBox{T, V, FL2}, 
                     roundAtol::Real=nearestHalfOf(getAtolVal(T))) where {T, V, FL1, FL2}
    if isDiffParam(pb1) && compareParamBox(pb1, pb2)
        # bl = isDiffParam(pb1)
        # pbs = (pb1, pb2)
        # pb1, pb2 = bl ? pbs : outValCopy.(pbs)
        ParamBox(Val(V), combineParFunc(*, pb1.map, pb2.map), pb1.data[], 
                 genIndex(nothing), fill(pb1.canDiff[]), min(pb1.dataName, pb2.dataName))
    else
        ParamBox(Val(V), fill(roundToMultiOfStep(pb1() * pb2(), roundAtol)))
    end
end

function reduceParamBoxes(pb1::ParamBox{T, V, FL1}, pb2::ParamBox{T, V, FL2}, 
                          roundAtol::Real=nearestHalfOf(getAtolVal(T))) where 
                         {T, V, FL1, FL2}
    if pb1 === pb2 || hasIdentical(pb1, pb2)
        [pb1]
    elseif hasEqual(pb1, pb2)
        [deepcopy(pb1)]
    else
        outVal1 = pb1()
        outVal2 = pb2()
        if isApprox(outVal1, outVal2, atol=2roundAtol)
            [genExponent( getNearestMid(outVal1, outVal2, roundAtol) )]
        else
            [pb1, pb2]
        end
    end
end