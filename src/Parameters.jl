export ParamBox, inValOf, inSymOf, inSymOfCore, inSymValOf, outValOf, outSymOf, 
       outSymOfCore, outSymValOf, dataOf, mapOf, getVar, getVarDict, outValCopy, inVarCopy, 
       enableDiff!, disableDiff!, isDiffParam, toggleDiff!, isDepParam, changeMapping

export FLevel

# Function Level
struct FLevel{L} <: MetaParam{FLevel} end

FLevel(::Type{itselfT}) = FLevel{0}
FLevel(::Type{typeof(Base.identity)}) = FLevel{0}
FLevel(::Type{<:Function}) = FLevel{1}
FLevel(::Type{<:ParameterizedFunction{T1, T2}}) where {T1, T2} = 
      FLevel{getFLevel(T1) + getFLevel(T2)}
FLevel(::F) where {F<:Function} = FLevel(F)
const FI = FLevel(itself)

FLevel(sym::Symbol) = FLevel(getFunc(sym))
FLevel(::TypedFunction{F}) where {F} = FLevel(F)
FLevel(::Type{TypedFunction{F}}) where {F} = FLevel(F)

getFLevel(::Type{FLevel{L}}) where {L} = L
getFLevel(f::Function) = getFLevel(f |> FLevel)
getFLevel(::TypedFunction{F}) where {F} = getFLevel(F |> FLevel)
getFLevel(::Type{T}) where {T} = getFLevel(T |> FLevel)

"""

    ParamBox{T, V, FL<:FLevel} <: DifferentiableParameter{T, ParamBox}

Parameter container that can enable differentiation.

≡≡≡ Field(s) ≡≡≡

`data::Array{T, 0}`: The container of the data (i.e. the value of the input variable) 
stored in the `ParamBox` that can be accessed by syntax `[]`. The data value stored in an 
arbitrary `ParamBox{T}` `pb` can be modified using the syntax `pb[] = aNewVal`.

`dataName::Symbol`: The name of the input variable.

`map::Function`: The mapping of the data within the same domain (`.map(::T)->T`). The 
result (i.e., the value of the output variable) can be accessed by syntax `()`.

`canDiff::Array{Bool, 0}`: Indicator of whether the output variable is "marked" as 
differentiable with respect to the input variable in the differentiation process. In other 
words, it determines whether the output variable represented by the `ParamBox` is treated 
as a dependent variable or an independent variable.

`index::Array{<:Union{Int, Nothing}, 0}`: Additional index assigned to the `ParamBox`.

≡≡≡ Initialization Method(s) ≡≡≡

    ParamBox(data, dataName=:undef; index=nothing) -> ParamBox{<:Any, dataName, $(FI)}

    ParamBox(data, name, mapFunction, dataName=:undef; index=nothing, canDiff=true) ->
    ParamBox{<:Any, name}

=== Positional argument(s) ===

`data::Union{Array{T, 0}, T}`: The input value to be stored or the container of it. If the 
latter is the first argument, then it will directly be assigned to `.data`.

`name::Symbol`: Specify the name of the output variable represented by the constructed 
`ParamBox`. It's not required when `mapFunction` is not provided because then the output 
variable is considered the same as the input variable. It's equal to the type parameter `V` 
of the constructed `ParamBox`.

`mapFunction::Function`: The mapping of the stored data (`mapFunction(::T)->T`), which will 
be assigned to the field `.map`. After constructing a `ParamBox`, e.g 
`pb = ParamBox(x, yName, f)`, `pb[]` returns the value of `x`, and `pb()` returns the value 
of `f(x)::T`. When `mapFunction` is not provided, `.map` is set to [`itself`](@ref) that 
maps the stored data to itself.

`dataName::Symbol`: The name of the stored data, i.e, the name of the input variable.

=== Keyword argument(s) ===

`index::Union{Int, Nothing}`: The index of the constructed `ParamBox`. It's should be left 
with its default value unless the user plans to utilize the index of a `ParamBox` for 
specific application other than differentiation.

`canDiff::Bool`: Determine whether the output variable is marked as "differentiable".

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
    data::Array{T, 0}
    dataName::Symbol
    map::Function
    canDiff::Array{Bool, 0}
    index::Array{<:Union{Int, Nothing}, 0}

    function ParamBox{T, V}(f::F, data::Array{T, 0}, index, canDiff, 
                            dataName=:undef) where {T, V, F<:Function}
        @assert Base.return_types(f, (T,))[1] == T
        dName = ifelse(dataName == :undef, Symbol("x_" * string(V)), dataName)
        new{T, V, FLevel(F)}(data, dName, f, canDiff, index)
    end

    ParamBox{T, V}(data::Array{T, 0}, index) where {T, V} = 
    new{T, V, FI}(data, V, itself, fill(false), index)
end

function ParamBox(::Val{V}, mapFunction::F, data::Array{T, 0}, index, canDiff, 
                  dataName=:undef) where {V, F<:Function, T}
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

ParamBox(::Val{V}, data::Array{T, 0}, index) where {V, T} = ParamBox{T, V}(data, index)

ParamBox(::Val{V}, ::itselfT, data::Array{T, 0}, index, _...) where {V, T} = 
ParamBox{T, V}(data, index)

ParamBox(::Val{V}, pb::ParamBox{T, <:Any, FI}) where {V, T} = 
ParamBox{T, V}(pb.data, pb.index)

ParamBox(::Val{V}, pb::ParamBox{T}; canDiff::Array{Bool, 0}=pb.canDiff) where {T, V} = 
ParamBox{T, V}(pb.map, pb.data, pb.index, canDiff, pb.dataName)

ParamBox(data::T, dataName::Symbol=:undef; index::Union{Int, Nothing}=nothing) where {T} = 
ParamBox(Val(dataName), fillObj(data), genIndex(index))

ParamBox(data::T, name::Symbol, mapFunction::F, dataName::Symbol=:undef; 
         index::Union{Int, Nothing}=nothing, canDiff::Bool=true) where {T, F<:Function} = 
ParamBox(Val(name), mapFunction, fillObj(data), genIndex(index), fill(canDiff), dataName)


mapTypeOf(::ParamBox{<:Any, <:Any, FL}) where {FL} = FL


"""

    inValOf(pb::ParamBox{T}) where {T} -> T

Return the value of the input variable of `pb`. Equivalent to `pb[]`.
"""
@inline inValOf(pb::ParamBox) = pb.data[]


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
@inline inSymValOf(pb::ParamBox{T}) where {T} = (inSymOf(pb) => pb.data[])


"""

    outValOf(pb::ParamBox) -> Number

Return the value of the output variable of `pb`. Equivalent to `pb()`.
"""
@inline outValOf(pb::ParamBox) = callGenFunc(pb.map, pb.data[])

@inline outValOf(pb::ParamBox{<:Any, <:Any, FI}) = inValOf(pb)

(pb::ParamBox)() = outValOf(pb)
# (pb::ParamBox)() = Base.invokelatest(pb.map, pb.data[])::Float64


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
@inline inSymOfCore(pb::ParamBox{<:Any, <:Any, FI}) = outSymOfCore(pb)


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
@inline dataOf(pb::ParamBox) = pb.data


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

function getVarCore(pb::ParamBox{T, <:Any, FL}) where {T, FL}
    dvSym = outSymOf(pb)
    ivVal = inValOf(pb)
    ovVal = outValOf(pb)
    ifelse(isDiffParam(pb), Pair{Symbol, T}[dvSym=>ovVal, inSymOf(pb)=>ivVal], 
                            Pair{Symbol, T}[dvSym=>ovVal])
end

getVarCore(pb::ParamBox{T, <:Any, FI}) where {T} = 
Pair{Symbol, T}[inSymOf(pb) => inValOf(pb)]


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
ParamBox{Float64, :x_a, $(FI)}(-2.0)[∂][x_a]

julia> pb1[] == pb2[] == -2.0
true

julia> pb1[] = 1.1
1.1

julia> pb2[]
1.1
```
"""
inVarCopy(pb::ParamBox{T}) where {T} = 
ParamBox{T, inSymOfCore(pb)}(pb.data, genIndex(nothing))


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


"""

    changeMapping(pb::ParamBox, mapFunction::Function, outputName::Symbol=V; 
                  canDiff::Bool=true) -> 
    ParamBox{T, outputName}

Change the mapping function of `pb`. The name of the output variable of the returned 
`ParamBox` can be specified by `outputName`, and its differentiability is determined by 
`canDiff`.
"""
function changeMapping(pb::ParamBox{T, V, FL}, mapFunction::F, outputName::Symbol=V; 
                       canDiff::Bool=true) where {T, V, FL, F<:Function}
    dn = pb.dataName
    if (FL==FI && FLevel(F)!=FI)
        dnStr = string(dn)
        dn = Symbol(dnStr * "_" * dnStr)
    end
    ParamBox(Val(outputName), mapFunction, pb.data, 
             genIndex( ifelse(canDiff, pb.index[], nothing) ), fill(canDiff), dn)
end


compareParamBoxCore1(pb1::ParamBox, pb2::ParamBox) = (pb1.data === pb2.data)

compareParamBoxCore2(pb1::ParamBox, pb2::ParamBox) = 
compareParamBoxCore1(pb1, pb2) && (typeof(pb1.map) === typeof(pb2.map))

function compareParamBox(pb1::ParamBox, pb2::ParamBox)
    ifelse(( (bl=isDiffParam(pb1)) == isDiffParam(pb2) ),
        ifelse( bl, 
            compareParamBoxCore1(pb1, pb2), 

            compareParamBoxCore2(pb1, pb2)
        ),

        false
    )
end

compareParamBox(pb1::ParamBox{<:Any, <:Any, FI}, pb2::ParamBox{<:Any, <:Any, FI}) = 
compareParamBoxCore1(pb1, pb2)

compareParamBox(pb1::ParamBox{<:Any, <:Any, FI}, pb2::ParamBox) = 
ifelse(isDiffParam(pb2), compareParamBoxCore1(pb1, pb2), false)

compareParamBox(pb1::ParamBox, pb2::ParamBox{<:Any, <:Any, FI}) = 
compareParamBox(pb2, pb1)


function mulParamBoxCore(c::T1, con::ParamBox{T2, <:Any, FI}, 
                         roundAtol::Real=nearestHalfOf(getAtolVal(T))) where {T1, T2}
    conNew = fill(roundToMultiOfStep(convert(T2, con.data[]*c), roundAtol))
    mapFunction = itself
    dataName = :undef
    conNew, mapFunction, dataName
end

function mulParamBoxCore(c::T1, con::ParamBox{T2}, 
                         roundAtol::Real=nearestHalfOf(getAtolVal(T))) where {T1, T2}
    conNew = con.data
    mapFunction = Pf(convert(T2, roundToMultiOfStep(c, roundAtol)), con.map)
    conNew, mapFunction, con.dataName
end