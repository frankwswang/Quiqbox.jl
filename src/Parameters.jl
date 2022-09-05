export ParamBox, inValOf, outValOf, inSymOf, outSymOf, indVarOf, dataOf, mapOf, outValCopy, 
       fullVarCopy, enableDiff!, disableDiff!, isDiffParam, toggleDiff!, changeMapping

export FLevel

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

`data::Array{Pair{Array{T, 0}, Symbol}, 0}`: The container of the input variable data 
in the form of a `Pair` of its value container and symbol) stored in the `ParamBox`. The 
value of the input variable can be accessed by syntax `[]`; to modify it, for example for a 
`pb::ParamBox{T}`, use the syntax `pb[] = newVal` where `newVal` is the new value that is 
or can be converted into type `T`.

`map::Function`: The mapping of the value of the input variable (i.e. the input value) 
within the same domain (`.map(::T)->T`). The result (i.e., the value of the output 
variable, or the "output value") can be accessed by syntax `()`.

`canDiff::Array{Bool, 0}`: Indicator of whether the output variable is "marked" as 
differentiable with respect to the input variable in the differentiation process. In other 
words, it determines whether the output variable represented by the `ParamBox` is treated 
as a dependent variable or an independent variable.

`index::Array{<:Union{Int, Nothing}, 0}`: Additional index assigned to the `ParamBox`.

≡≡≡ Initialization Method(s) ≡≡≡

    ParamBox(inVar::Union{T, Array{T, 0}}, outSym::Symbol=:undef, 
             inSym::Symbol=Symbol(IVsymSuffix, outSym); 
             index::Union{Int, Nothing}=nothing, canDiff::Bool=false) where {T} -> 
    ParamBox{T, outSym, $(FI)}

    ParamBox(inVar::Union{T, Array{T, 0}}, outSym::Symbol, mapFunction::Function, 
             inSym::Symbol=Symbol(IVsymSuffix, outSym); 
             index::Union{Int, Nothing}=nothing, canDiff::Bool=true) where {T} -> 
    ParamBox{T, outSym}

=== Positional argument(s) ===

`inVar::Union{T, Array{T, 0}}`: The value or the container of the input variable to be 
stored. If the latter is the type of `data`, then it will directly used to construct 
`.data[]` with without any copy.

`outSym::Symbol`: The symbol of the output variable represented by the constructed 
`ParamBox`. It's equal to the type parameter `V` of the constructed `ParamBox`.

`inSym::Symbol`: The symbol of the input variable held by the constructed `ParamBox`.

`mapFunction::Function`: The mapping (`mapFunction(::T)->T`) of the input variable, which 
will be assigned to the field `.map`. When `mapFunction` is not provided, `.map` is set to 
[`itself`](@ref) that maps the input variable to an identical-valued output variable.

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
and the name (the symbol with an assigned index if applied) of the independent variable 
tied to the `ParamBox`. When the `ParamBox` is marked as differentiable, "`[∂]`" is green 
and `IV` is the name of the stored input variable; when it's marked as non-differentiable, 
"`[∂]`" is grey and `IV` is the name of the output variable.

**NOTE 2:** The output variable of a `ParamBox` is normally used to differentiate a 
parameter functional (e.g., the Hartree-Fock energy). However, the derivative with respect 
to the stored input variable can also be computed to when the `ParamBox` is marked as 
differentiable.
"""
struct ParamBox{T, V, FL<:FLevel} <: DifferentiableParameter{T, ParamBox}
    data::Array{Pair{Array{T, 0}, Symbol}, 0}
    map::Function
    canDiff::Array{Bool, 0}
    index::Array{<:Union{Int, Nothing}, 0}

    function ParamBox{T, V}(f::F, data::Pair{Array{T, 0}, Symbol}, index, canDiff) where 
                           {T, V, F<:Function}
        @assert Base.return_types(f, (T,))[1] == T
        new{T, V, FLevel(F)}(fill(data), f, canDiff, index)
    end

    ParamBox{T, V}(data::Pair{Array{T, 0}, Symbol}, index, canDiff) where {T, V} = 
    new{T, V, FI}(fill(data), itself, canDiff, index)
end

function ParamBox(::Val{V}, mapFunction::F, data::Pair{Array{T, 0}, Symbol}, 
                  index=genIndex(nothing), canDiff=fill(true)) where {V, F<:Function, T}
    if getFLevel(F) != 0
        fSym = mapFunction |> nameOf
        fStr = fSym |> string
        if startswith(fStr, '#')
            idx = parse(Int, fStr[2:end])
            fSym = Symbol("f_", V,  numToSubs(idx))
            mapFunction = renameFunc(fSym, mapFunction)
        end
    end
    ParamBox{T, V}(mapFunction, data, index, canDiff)
end

ParamBox(::Val{V}, ::itselfT, data::Pair{Array{T, 0}, Symbol}, index=genIndex(nothing), 
         canDiff=fill(false)) where {V, T} = 
ParamBox{T, V}(data, index, canDiff)

ParamBox(::Val{V}, pb::ParamBox{T}, canDiff::Array{Bool, 0}=pb.canDiff) where {V, T} = 
ParamBox{T, V}(pb.map, pb.data[], pb.index, canDiff)

ParamBox(inVar::Union{T, Array{T, 0}}, outSym::Symbol=:undef, 
         inSym::Symbol=Symbol(IVsymSuffix, outSym); 
         index::Union{Int, Nothing}=nothing, canDiff::Bool=false) where {T} = 
ParamBox(Val(outSym), itself, 
         fillObj(inVar)=>inSym, genIndex(index), fill(canDiff))

ParamBox(inVar::Union{T, Array{T, 0}}, outSym::Symbol, mapFunction::Function, 
         inSym::Symbol=Symbol(IVsymSuffix, outSym); 
         index::Union{Int, Nothing}=nothing, canDiff::Bool=true) where {T} = 
ParamBox(Val(outSym), mapFunction, 
         fillObj(inVar)=>inSym, genIndex(index), fill(canDiff))


"""

    inValOf(pb::ParamBox{T}) where {T} -> T

Return the value of the input variable of `pb`. Equivalent to `pb[]`.
"""
@inline inValOf(pb::ParamBox) = pb.data[][begin][]


"""

    outValOf(pb::ParamBox) -> Number

Return the value of the output variable of `pb`. Equivalent to `pb()`.
"""
@inline outValOf(pb::ParamBox) = callGenFunc(pb.map, pb.data[][begin][])

@inline outValOf(pb::ParamBox{<:Any, <:Any, FI}) = inValOf(pb)

(pb::ParamBox)() = outValOf(pb)
# (pb::ParamBox)() = Base.invokelatest(pb.map, pb.data[][begin][])::Float64


"""

    inSymOf(pb::ParamBox) -> Symbol

Return the symbol of the input variable of `pb`.
"""
@inline inSymOf(pb::ParamBox) = pb.data[][end]


"""

    outSymOf(pb::ParamBox) -> Symbol

Return the symbol of the output variable of `pb`.
"""
@inline outSymOf(::ParamBox{<:Any, V}) where {V} = V


"""

    indVarOf(pb::ParamBox{T}) -> Pair{}

Return (the name and the value of) the independent variable tied to `pb`. Specifically, 
return the input variable stored in `pb` when `pb` is marked as differentiable; return the 
output variable of `pb` when `pb` is marked as non-differentiable. Thus, it is the variable 
`pb` represents to differentiate any (differentiable) function of [`ParamBox`](@ref)es.
"""
function indVarOf(pb::ParamBox)
    idx = numToSubs(pb.index[])
    if isDiffParam(pb)
        Symbol(inSymOf(pb), idx) => inValOf(pb)
    else
        Symbol(outSymOf(pb), idx) => outValOf(pb)
    end
end


getTypeParams(::ParamBox{T, V, FL}) where {T, V, FL} = (T, V, FL)


"""

    dataOf(pb::ParamBox{T}) where {T} -> Pair{Array{T, 0}, Symbol}

Return the `Pair` of the input variable and its symbol.
"""
@inline dataOf(pb::ParamBox) = pb.data[]


"""

    mapOf(pb::ParamBox) -> Function

Return the mapping function of `pb`.
"""
@inline mapOf(pb::ParamBox) = pb.map


"""

    outValCopy(pb::ParamBox{T, V}) where {T} -> ParamBox{T, V, $(FI)}

Return a new `ParamBox` of which the input variable's value is equal to the output 
variable's value of `pb`.
"""
outValCopy(pb::ParamBox{<:Any, V}) where {V} = 
ParamBox(Val(V), itself, fill(pb())=>Symbol(IVsymSuffix, V))


"""

    fullVarCopy(pb::T) where {T<:ParamBox} -> T

A shallow copy of the input `ParamBox`.
"""
fullVarCopy(pb::ParamBox{<:Any, V}) where {V} = ParamBox(Val(V), pb)


"""

    isDiffParam(pb::ParamBox) -> Bool

Return the Boolean value of whether `pb` is differentiable with respect to its input 
variable.
"""
isDiffParam(pb::ParamBox) = pb.canDiff[]


"""

    enableDiff!(pb::ParamBox) -> Bool

Mark `pb` as "differentiable" and then return `true`.
"""
function enableDiff!(pb::ParamBox)
    if pb.canDiff[]
        true
    else
        pb.index[] = nothing
        pb.canDiff[] = true
    end
end


"""

    disableDiff!(pb::ParamBox) -> Bool

Mark `pb` as "non-differentiable" and then return `false`.
"""
function disableDiff!(pb::ParamBox)
    if pb.canDiff[]
        pb.index[] = nothing
        pb.canDiff[] = false
    else
        false
    end
end


"""

    toggleDiff!(pb::ParamBox) -> Bool

Toggle the differentiability of the input `pb` and then return the updated boolean value.
"""
function toggleDiff!(pb::ParamBox)
    pb.index[] = nothing
    pb.canDiff[] = !pb.canDiff[]
end


isNotDiffNorInVar(pb::ParamBox{<:Any, <:Any, FI}) = false
isNotDiffNorInVar(pb::ParamBox{<:Any, <:Any, FL}) where {FL} = !isDiffParam(pb)


"""

    changeMapping(pb::ParamBox, mapFunction::Function=$(itself), outSym::Symbol=V; 
                  canDiff::Bool=isDiffParam(pb)) -> 
    ParamBox{T, outSym}

Change the mapping function of `pb`. The symbol of the output variable of the returned 
`ParamBox` can be specified by `outSym`, and its differentiability is determined by 
`canDiff`.
"""
changeMapping(pb::ParamBox{T, V, FL}, mapFunction::F=itself, outSym::Symbol=V; 
              canDiff::Bool=isDiffParam(pb)) where {T, V, FL, F<:Function} = 
ParamBox(Val(outSym), mapFunction, pb.data[], 
         genIndex( ifelse(canDiff==isDiffParam(pb)==true, pb.index[], nothing) ), 
         fill(canDiff))


compareParamBoxCore1(pb1::ParamBox, pb2::ParamBox) = 
(pb1.data[][begin] === pb2.data[][begin])

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


function addParamBox(pb1::ParamBox{T, V, FL1}, pb2::ParamBox{T, V, FL2}, 
                     roundAtol::Real=nearestHalfOf(getAtolVal(T))) where {T, V, FL1, FL2}
    if isDiffParam(pb1) && compareParamBox(pb1, pb2)
        ParamBox(Val(V), combineParFunc(+, pb1.map, pb2.map), 
                 pb1.data[][begin]=>min(pb1.data[][end], pb2.data[][end]), 
                 genIndex(nothing), fill(true))
    else
        ParamBox(Val(V), itself, 
                 fill(roundToMultiOfStep(pb1() + pb2(), roundAtol))=>Symbol(IVsymSuffix, V))
    end
end


function mulParamBox(c::Number, pb::ParamBox{T, V}, 
                     roundAtol::Real=nearestHalfOf(getAtolVal(T))) where {T, V}
    if isDiffParam(pb)
        mapFunc = Pf(convert(T, roundToMultiOfStep(c, roundAtol)), pb.map)
        ParamBox(Val(V), mapFunc, pb.data[], genIndex(nothing), fill(true))
    else
        ParamBox(Val(V), itself, 
                 fill(roundToMultiOfStep(T(pb()*c), roundAtol))=>Symbol(IVsymSuffix, V))
    end
end

function mulParamBox(pb1::ParamBox{T, V, FL1}, pb2::ParamBox{T, V, FL2}, 
                     roundAtol::Real=nearestHalfOf(getAtolVal(T))) where {T, V, FL1, FL2}
    if isDiffParam(pb1) && compareParamBox(pb1, pb2)
        ParamBox(Val(V), combineParFunc(*, pb1.map, pb2.map), 
                 pb1.data[][begin]=>min(pb1.data[][end], pb2.data[][end]), 
                 genIndex(nothing), fill(true))
    else
        ParamBox(Val(V), itself, 
                 fill(roundToMultiOfStep(pb1() * pb2(), roundAtol))=>Symbol(IVsymSuffix, V))
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