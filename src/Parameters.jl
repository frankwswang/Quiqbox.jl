export ParamBox, inValOf, inSymOf, inSymOfCore, inSymValOf, outValOf, outSymOf, 
       outSymOfCore, outSymValOf, dataOf, mapOf, outValCopy, inVarCopy, enableDiff!, 
       disableDiff!, isDiffParam, toggleDiff!, changeMapping

using Symbolics: Num

# Function Level
struct FLevel{L1, L2} <: MetaParam{FLevel} end

# FLevel(::Any) = FLevel{0, 0}
FLevel(::Type{typeof(itself)}) = FLevel{1, 0}
FLevel(::Type{<:Function}) = FLevel{2, 0}
FLevel(::Type{<:ParameterizedFunction{<:Any, <:Any}}) = FLevel{3, 0}
FLevel(::Type{<:ParameterizedFunction{<:Any, itself}}) = FLevel{3, 1}
FLevel(::Type{<:ParameterizedFunction{<:Any, <:Function}}) = FLevel{3, 2}
FLevel(::Type{<:ParameterizedFunction{<:Any, <:ParameterizedFunction}}) = FLevel{3, 3}
FLevel(::F) where {F<:Function} = FLevel(F)

FLevel(sym::Symbol) = FLevel(getFunc(sym))
FLevel(::TypedFunction{F}) where {F} = FLevel(F)
FLevel(::Type{TypedFunction{F}}) where {F} = FLevel(F)

getFLevel(::Type{FLevel{L1, L2}}) where {L1, L2} = (L1, L2)
getFLevel(f::Function) = getFLevel(f |> FLevel)
getFLevel(::TypedFunction{F}) where {F} = getFLevel(F |> FLevel)
getFLevel(::Type{T}) where {T} = getFLevel(T |> FLevel)


"""

    ParamBox{T, V, FL} <: DifferentiableParameter{ParamBox, T}

Parameter container that can enable parameter differentiations.

≡≡≡ Field(s) ≡≡≡

`data::Array{T, 0}`: The data (parameter) stored in a 0-D `Array` that can be accessed by 
syntax `[]`.

`dataName::Symbol`: The name assigned to the stored data.

`map::Function`: The mathematical mapping of the data. The mapped result can be accessed by 
syntax `()`.

`canDiff::Array{Bool, 0}`: Indicator that whether this container (mapped variable) is 
marked as "differentiable".

`index::Array{<:Union{Int, Nothing}, 0}`: Additional index assigned to the parameter.

≡≡≡ Initialization Method(s) ≡≡≡

    ParamBox(data::Union{Array{T, 0}, T}, dataName::Symbol=:undef; 
             index::Union{Int, Nothing}=nothing) where {T<:Number} -> 
    ParamBox{T, dataName, $(FLevel(itself))}

    ParamBox(data::Union{Array{T, 0}, T}, name::Symbol, mapFunction::Function, 
             dataName::Symbol=:undef; index::Union{Int, Nothing}=nothing, 
             canDiff::Bool=true) where {T<:Number} ->
    ParamBox{T, name, $(FLevel)(mapFunction)}

`name` specifies the name of the (mapped) variable the `ParamBox` represents, which helps 
with symbolic representation and automatic differentiation.

`mapFunction`: The (mathematical) mapping of the data, which will be stored in the field 
`map`. It is for the case where the variable represented by the `ParamBox` is dependent on 
another independent variable of which the value is the stored data in the container. After 
initializing a `ParamBox`, e.g `pb1 = ParamBox(x, mapFunction=f)`, `pb[]` returns `x`, and 
`pb()` returns `f(x)::T`. `mapFunction` is set to `$(itself)` in default, which is a dummy 
function that maps the data to itself.

`canDiff` determines whether the mapped math variable is "marked" as differentiable (i.e., 
the mapping is a differentiable function) with respect to the stored data. In other 
words, it determines whether the mapped variable `ParamBox` generated during the automatic 
differentiation procedure is treated as a dependent variable or an independent variable 
regardless of the mapping relation.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> ParamBox(1.0)
ParamBox{Float64, :undef, $(FLevel(itself))}(1.0)[∂][undef]

julia> ParamBox(1.0, :a)
ParamBox{Float64, :a, $(FLevel(itself))}(1.0)[∂][a]

julia> ParamBox(1.0, :a, abs)
ParamBox{Float64, :a, $(FLevel(abs))}(1.0)[∂][x_a]
```

**NOTE 1:** The rightmost "`[∂][IV]`" in the printing info indicates the differentiability 
and the name of the represented independent variable `:IV`. When the `ParamBox` is marked 
as a "differentiable function", "`[∂]`" is in color green (otherwise it's in grey).

**NOTE 2:** It's always the (mapped) variable `V` generated by a `ParamBox{<:Any, V}` that 
is used to construct a basis, whereas the underlying independent variable is used to 
differentiate the basis (in other words, only when `mapFunction = $(itself)` or 
`canDiff = false` is the independent variable same as the mapped variable/represented 
parameter).
"""
struct ParamBox{T, V, FL<:FLevel} <: DifferentiableParameter{ParamBox, T}
    data::Array{T, 0}
    dataName::Symbol
    map::Function
    canDiff::Array{Bool, 0}
    index::Array{<:Union{Int, Nothing}, 0}

    function ParamBox{T, V}(f::F, data::Array{T, 0}, index, canDiff, 
                            dataName=:undef) where {T, V, F<:Function}
        @assert Base.return_types(f, (T,))[1] == T
        dName = (dataName == :undef) ? Symbol("x_" * string(V)) : dataName
        new{T, V, FLevel(F)}(data, dName, f, canDiff, index)
    end

    ParamBox{T, V}(data::Array{T, 0}, index) where {T, V} = 
    new{T, V, FLevel(itself)}(data, V, itself, fill(false), index)
end

function ParamBox(::Val{V}, mapFunction::F, data::Array{T, 0}, index, canDiff, 
                  dataName=:undef) where {V, F<:Function, T}
    L1, _ = getFLevel(F)
    Ftype = F
    if L1 == 2
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

ParamBox(::Val{V}, ::typeof(itself), data::Array{T, 0}, index, _...) where {V, T} = 
ParamBox{T, V}(data, index)

ParamBox(::Val{V}, pb::ParamBox{T, <:Any, FLevel(itself)}) where {V, T} = 
ParamBox{T, V}(pb.data, pb.index)

ParamBox(::Val{V}, pb::ParamBox) where {V} = 
ParamBox(Val(V), pb.map, pb.data, pb.index, pb.canDiff, pb.dataName)

ParamBox(data::T, dataName::Symbol=:undef; index::Union{Int, Nothing}=nothing) where {T} = 
ParamBox(Val(dataName), fillNumber(data), genIndex(index))

ParamBox(data::T, name::Symbol, mapFunction::F, dataName::Symbol=:undef; 
         index::Union{Int, Nothing}=nothing, canDiff::Bool=true) where {T, F<:Function} = 
ParamBox(Val(name), mapFunction, fillNumber(data), genIndex(index), fill(canDiff), dataName)


"""

    inValOf(pb::ParamBox) -> Number

Return the value of stored data (independent variable) of the input `ParamBox`. Equivalent 
to `pb[]`.
"""
@inline inValOf(pb::ParamBox{T}) where {T} = pb.data[]::T


"""

    inSymOf(pb::ParamBox) -> Symbolics.Num

Return the variable`::Symbolics.Num` of stored data (independent variable) of the input 
`ParamBox`.
"""
@inline function inSymOf(pb::ParamBox{T}) where {T}
    idx = pb.index[]
    hasIdx = idx isa Int
    ivSym = inSymOfCore(pb)
    hasIdx ? Symbolics.variable(ivSym, idx) : Symbolics.variable(ivSym)
end


"""

    inSymValOf(pb::ParamBox{T}) where {T} -> ::Pair{Symbolics.Num, T}

Return a `Pair` of the stored independent variable of the input `ParamBox` and its 
corresponding value.
"""
@inline inSymValOf(pb::ParamBox{T}) where {T} = 
        (inSymOf(pb) => pb.data[])::Pair{Symbolics.Num, T}


"""

    outValOf(pb::ParamBox) -> Number

Return the value of mapped data (dependent variable) of the input `ParamBox`. Equivalent to 
`pb()`.
"""
@inline outValOf(pb::ParamBox) = pb.map(pb.data[])

@inline outValOf(pb::ParamBox{T, <:Any, FLevel(itself)}) where {T} = inValOf(pb)

(pb::ParamBox)() = outValOf(pb)
# (pb::ParamBox)() = Base.invokelatest(pb.map, pb.data[])::Float64


"""

    outSymOf(pb::ParamBox) -> Symbolics.Num

Return the variable`::Symbolics.Num` of mapped data (dependent variable) of the input 
`ParamBox`.
"""
@inline outSymOf(pb::ParamBox{T, <:Any, FLevel(itself)}) where {T} = inSymOf(pb)

@inline function outSymOf(pb::ParamBox)
    idx = pb.index[]
    hasIdx = idx isa Int
    vSym = outSymOfCore(pb)
    hasIdx ? Symbolics.variable(vSym, idx) : Symbolics.variable(vSym)
end


"""

    outSymValOf(pb::ParamBox) -> ::Pair{Symbolics.Num, T}

Return a `Pair` of the dependent variable represented by the input `ParamBox` and the 
corresponding value (mapped value).
"""
@inline outSymValOf(pb::ParamBox{T}) where {T} = (inSymOf(pb) => outValOf(pb))


"""

    inSymOfCore(pb::ParamBox) -> Symbol

Return the `Symbol` of the stored data (independent variable) of the input `ParamBox`.
"""
@inline inSymOfCore(pb::ParamBox) = pb.dataName


"""

    outSymOfCore(pb::ParamBox) -> Symbol

Return the `Symbol` of the mapped data (dependent variable) of the input `ParamBox`.
"""
@inline outSymOfCore(::ParamBox{<:Any, V}) where {V} = V


"""

    dataOf(pb::ParamBox{T}) where {T} -> Array{T, 0}

Return the 0-D `Array` of the data stored in the input `ParamBox`.
"""
@inline dataOf(pb::ParamBox) = pb.data


"""

    mapOf(pb::ParamBox) -> Function

Return the mapping function of the input `ParamBox`.
"""
@inline mapOf(pb::ParamBox) = pb.map


"""

    outValCopy(pb::ParamBox{T, V}) where {T} -> ParamBox{T, V, $(FLevel(itself))}

Return a new `ParamBox` of which the stored data is a **deep copy** of the mapped data from 
the input `ParamBox`.
"""
outValCopy(pb::ParamBox{<:Any, V}) where {V} = 
ParamBox(Val(V), fill(pb()), genIndex(nothing))


"""

    inVarCopy(pb::ParamBox) -> ParamBox{<:Number, <:Any, $(FLevel(itself))}

Return a new `ParamBox` of which the stored data is a **shallow copy** of the stored data 
from the input `ParamBox`.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> pb1 = ParamBox(-2.0, :a, abs)
ParamBox{Float64, :a, $(FLevel(abs))}(-2.0)[∂][x_a]

julia> pb2 = inVarCopy(pb1)
ParamBox{Float64, :x_a, $(FLevel(itself))}(-2.0)[∂][x_a]

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

Mark the input `ParamBox` as "differentiable" and return the marked `ParamBox`.
"""
function enableDiff!(pb::ParamBox)
    pb.canDiff[] = true
    pb
end


"""

    disableDiff!(pb::ParamBox) -> ParamBox

Mark the input `ParamBox` as "non-differentiable" and return the marked `ParamBox`.
"""
function disableDiff!(pb::ParamBox)
    pb.canDiff[] = false
    pb
end


"""

    isDiffParam(pb::ParamBox) -> Bool

Return the Boolean value of if the input `ParamBox` is "differentiable".
"""
isDiffParam(pb::ParamBox) = pb.canDiff[]


"""

    toggleDiff!(pb::ParamBox) -> Bool

Toggle the differentiability (`pb.canDiff[]`) of the input `ParamBox` and return the 
altered result.
"""
toggleDiff!(pb::ParamBox) = begin pb.canDiff[] = !pb.canDiff[] end


"""

    changeMapping(pb::ParamBox{T, V}, mapFunction::F; 
                  index::Union{Int, Nothing}=nothing, canDiff::Bool=true) where 
                 {T, V, F<:Function} -> 
    ParamBox{T, V}

Return a `ParamBox` that contains the input `ParamBox`'s `data::Array{T, 0}` with the 
newly assigned mapping function.
"""
changeMapping(pb::ParamBox{T, V}, mapFunction::F; 
              index::Union{Int, Nothing}=nothing, canDiff::Bool=true) where 
             {T, V, F<:Function} = 
ParamBox(Val(V), mapFunction, pb.data, genIndex(index), fill(canDiff), pb.dataName)

"""

    changeMapping(pb::ParamBox{T}, name::Symbol, mapFunction::F; 
                  index::Union{Int, Nothing}=nothing, canDiff::Bool=true) where 
                 {T, F<:Function} -> 
    ParamBox{T, name}

"""
changeMapping(pb::ParamBox{T}, name::Symbol, mapFunction::F; 
              index::Union{Int, Nothing}=nothing, canDiff::Bool=true) where 
             {T, F<:Function} = 
ParamBox(Val(name), mapFunction, pb.data, genIndex(index), fill(canDiff), pb.dataName)