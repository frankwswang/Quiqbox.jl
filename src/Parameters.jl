export ParamBox, inValOf, outValOf, inSymOf, outSymOf, isInSymEqual, isOutSymEqual, 
       indParOf, indSymOf, dataOf, mapOf, outValCopy, fullVarCopy, enableDiff!, 
       disableDiff!, isDiffParam, toggleDiff!, changeMapping

using StructArrays

struct OneParam{T} <: VariableBox{T, OneParam}
    label::Symbol
    data::Array0D{T}
    index::MutableIndex
end

OneParam(label::Symbol, data::Union0D{T}, index::IntOrNone=nothing) where {T} = 
OneParam{T}(label, fillObj(data), genIndex(index))

OneParam(labelToData::Pair{Symbol, <:Union0D{T}}, index::IntOrNone=nothing) where {T} = 
OneParam{T}(labelToData[begin], fillObj(labelToData[end]), genIndex(index))

OneParam(op::OneParam) = itself(op)

const OPessential{T} = Union{OneParam{T}, Pair{Symbol, <:Union0D{T}}}

const OneParamStructVec{T} = 
      StructVector{OneParam{T}, NamedTuple{(:label, :data, :index), 
                   Tuple{Vector{Symbol}, Vector{Array0D{T}}, Vector{MutableIndex}}}, Int}


valOf(par::OneParam) = par.data[]
symOf(par::OneParam) = Symbol(par.label, numToSubs(par.index[]))
# indexOf(par::OneParam) = par.index[]

"""

    ParamBox{T, V, F<:Function} <: DifferentiableParameter{T, ParamBox}

Parameter container that can enable differentiation.

≡≡≡ Field(s) ≡≡≡

`data::Array{Pair{Array{T, 0}, Symbol}, 0}`: The container of the input variable data 
in the form of a `Pair` of its value container and symbol) stored in the `ParamBox`. The 
value of the input variable can be accessed by syntax `[]`; to modify it, for example for a 
`pb::ParamBox{T}`, use the syntax `pb[] = newVal` where `newVal` is the new value that is 
or can be converted into type `T`.

`map::Union{F, `[`DI`](@ref)`{F}}`: The mapping of the value of the input variable (i.e. 
the input value) within the same domain (`.map(::T)->T`). The result (i.e., the value of 
the output variable, or the "output value") can be accessed by syntax `()`.

`canDiff::Array{Bool, 0}`: Indicator of whether the output variable is "marked" as 
differentiable with respect to the input variable in the differentiation process. In other 
words, it determines whether the output variable represented by the `ParamBox` is treated 
as a dependent variable or an independent variable.

`index::Array{Union{Int, Nothing}, 0}`: Additional index assigned to the `ParamBox`.

≡≡≡ Initialization Method(s) ≡≡≡

    ParamBox(inVar::Union{T, Array{T, 0}}, outSym::Symbol=:undef, 
             inSym::Symbol=Symbol(IVsymSuffix, outSym); 
             index::Union{Int, Nothing}=nothing, canDiff::Bool=false) where {T} -> 
    ParamBox{T, outSym, $(iT)}

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

`inSym::Symbol`: The symbol of the input variable (without the index) held by the 
constructed `ParamBox`.

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
ParamBox{Float64, :undef, …}{0}[∂][undef]⟦=⟧[1.0]

julia> ParamBox(1.0, :a)
ParamBox{Float64, :a, …}{0}[∂][a]⟦=⟧[1.0]

julia> ParamBox(1.0, :a, abs)
ParamBox{Float64, :a, …}{1}[𝛛][x_a]⟦→⟧[1.0]
```

**NOTE 1:** The markers "`[∂][IV]`" in the printed info indicate the differentiability and 
the name (the symbol with an assigned index if applied) respectively of the independent 
variable tied to the `ParamBox`. When the `ParamBox` is marked as non-differentiable, 
"`[∂]`" is grey and `IV` corresponds to the name of the output variable; when the 
`ParamBox` is  marked as differentiable, "`[∂]`" becomes a green "`[𝛛]`", and `IV` 
corresponds to the name of the stored input variable.

**NOTE 2:** The output variable of a `ParamBox` is normally used to differentiate a 
parameter functional (e.g., the Hartree–Fock energy). However, the derivative with respect 
to the stored input variable can also be computed to when the `ParamBox` is marked as 
differentiable.
"""
struct MonoParamBox{T, V, F<:Function} <: ParamBox{T, V, F}
    data::Array0D{OneParam{T}}
    map::Union{F, DI{F}}
    index::MutableIndex
    canDiff::Array0D{Bool}

    function MonoParamBox{T, V}(mapFunction::F, data::OneParam{T}, 
                                index::Array0D{<:IntOrNone}, canDiff::Array0D{Bool}) where 
                               {T, V, F<:Function}
        checkFuncReturn(mapFunction, :mapFunction, T, T)
        new{T, V, dressOf(F)}(fill(data), mapFunction, index, canDiff)
    end

    MonoParamBox{T, V}(::IF, data::OneParam{T}, index, canDiff) where {T, V} = 
    new{T, V, iT}(fill(data), itself, index, canDiff)
end

const DefaultMPBoxMap = itself

struct PolyParamBox{T, V, F<:Function} <: ParamBox{T, V, F}
    data1::Array0D{OneParam{T}}
    data2::Array0D{OneParam{T}}
    data3::OneParamStructVec{T}
    map::Union{F, DI{F}}
    index::MutableIndex
    canDiff::Array0D{Bool}

    function PolyParamBox{T, V}(mapFunction::F, data1::OneParam{T}, data2::OneParam{T}, 
                                data3::AbstractVector{OneParam{T}}, 
                                index::Array0D{<:IntOrNone}, canDiff::Array0D{Bool}) where 
                               {T, V, F<:Function}
        checkFuncReturn(mapFunction, :mapFunction, Vector{T}, T)
        new{T, V, dressOf(F)}(fill(data1), fill(data2) StructArray(data3), 
                              f, index, canDiff)
    end
end

const DefaultPPBoxMap = sum

genParamBox(::Val{V}, mapFunction::F, data::OneParam{T}, 
            index::Union0D{<:IntOrNone}=genIndex(), 
            canDiff::Union0D{Bool}=getFLevel(F)>0) where {T, V, F<:Function} = 
MonoParamBox{T, V}(mapFunction, data, fillObj(index), fillObj(canDiff))

genParamBox(::Val{V}, mapFunction::F, data1::OneParam{T}, data2::OneParam{T}, 
            data3::AbstractVector{OneParam{T}}, 
            index::Union0D{<:IntOrNone}=genIndex(), 
            canDiff::Union0D{Bool}=false) where {T, V, F<:Function} = 
PolyParamBox{T, V}(mapFunction, data1, data2, data3, fillObj(index), fillObj(canDiff))

function genParamBox(::Val{V}, mapFunction::F, data::AbstractVector{OneParam{T}}, 
                     index::Union0D{<:IntOrNone}=genIndex(), 
                     canDiff::Union0D{Bool}=getFLevel(F)>0) where {T, V, F<:Function}
    isLenOne = checkCollectionMinLen(data, :data, 1)
    if isLenOne
        genParamBox(Val(V), mapFunction, data[], index, canDiff)
    else
        genParamBox(Val(V), mapFunction, data[begin], data[begin+1], data[begin+2:end], 
                    index, canDiff)
    end
end

genParamBox(::Val{V}, pb::MonoParamBox{T}, index::Union0D{<:IntOrNone}=genIndex(), 
            canDiff::Array0D{Bool}=copy(pb.canDiff)) where {V, T} = 
genParamBox(Val(V), pb.map, pb.data[], index, canDiff)

genParamBox(::Val{V}, pb::PolyParamBox{T}, index::Union0D{<:IntOrNone}=genIndex(), 
            canDiff::Array0D{Bool}=copy(pb.canDiff)) where {V, T} = 
genParamBox(Val(V), pb.map, pb.data1[], pb.data2[], pb.data3, index, canDiff)

genParamBox(data::OPessential{T}, outSym::Symbol=:undef, mapFunction::F=DefaultMPBoxMap; 
            index::IntOrNone=nothing, 
            canDiff::Bool=getFLevel(F)>0) where {T, F<:Function} = 
genParamBox(Val(outSym), mapFunction, OneParam(data), index, fill(canDiff))

function genParamBox(data::AbstractVector{<:Union{Union0D{T}, OPessential{T}}}, 
                     outSym::Symbol=:undef, 
                     mapFunction::F=ifelse(checkCollectionMinLen(data, :data, 1), 
                                           DefaultMPBoxMap, DefaultPPBoxMap); 
                     defaultDataLabel::Symbol=Symbol(IVsymSuffix, outSym), 
                     index::IntOrNone=nothing, 
                     canDiff::Bool=getFLevel(F)>0) where {T, F<:Function}
    data = map(data) do item
        item isa Union0D ? OneParam(defaultDataLabel, item) : OneParam(item)
    end
    genParamBox(Val(outSym), mapFunction, data, index, canDiff)
end

# function ParamBox(data::Union0D{T}, outSym::Symbol=:undef, mapFunction::F=itself; 
#                   defaultDataLabel::Symbol=Symbol(IVsymSuffix, outSym), 
#                   index::IntOrNone=nothing, canDiff::Bool=getFLevel(F)>0) where 
#                   {T, F<:Function}
#     data = map(data|>tupleObj) do item
#         item isa Union0D ? OneParam(defaultDataLabel, item) : item
#     end
#     ParamBox(Val(outSym), mapFunction, OneParam.(tupleObj(data)), 
#              genIndex(index), fill(canDiff))
# end

# function ParamBox(data::TorTupleLong{Union{Union0D{T}, OPessential{T}}}, 
#                   outSym::Symbol=:undef, mapFunction::Function=itself; 
#                   defaultDataLabel::Symbol=Symbol(IVsymSuffix, outSym), 
#                   index::Union{Int, Nothing}=nothing, canDiff::Bool=false)
#     data = map(data|>tupleObj) do item
#         item isa Union0D ? OneParam(defaultDataLabel, item) : item
#     end
#     ParamBox(Val(outSym), mapFunction, OneParam.(tupleObj(data)), 
#              genIndex(index), fill(canDiff))
# end

# ParamBox(inVar::Union0D{T}, outSym::Symbol=:undef, 
#          inSym::Symbol=(Symbol(IVsymSuffix, outSym),); 
#          index::Union{Int, Nothing}=nothing, canDiff::Bool=false) where {T} = 
# ParamBox(Val(outSym), itself, OneParam(inVar, inSym), genIndex(index), fill(canDiff))

# ParamBox(inVar::Union0D{T}, outSym::Symbol, mapFunction::Function, 
#          inSym::Symbol=Symbol(IVsymSuffix, outSym); 
#          index::Union{Int, Nothing}=nothing, canDiff::Bool=true) where {T} = 
# ParamBox(Val(outSym), mapFunction, OneParam(inVar, inSym), genIndex(index), fill(canDiff))

# ParamBox(inVar::UMOTuple{NPMO, T}, outSym::Symbol, mapFunction::Function, 
#          inSym::NMOTuple{NPMO, Symbol}=ntuple(_->Symbol(IVsymSuffix, outSym), Val(NPMO)); 
#          index::Union{Int, Nothing}=nothing, canDiff::Bool=true) where {NPMO, T} = 
# ParamBox(Val(outSym), mapFunction, 
#          OneParam.(inVar, inSym), genIndex(index), fill(canDiff))

const VPB{T} = Union{T, Array0D{T}, ParamBox{T}}


"""

    dataOf(pb::ParamBox{T}) where {T} -> Pair{Array{T, 0}, Symbol}

Return the `Pair` of the input variable and its symbol.
"""
@inline dataOf(pb::MonoParamBox) = pb.data
@inline dataOf(pb::PolyParamBox) = vcat(pb.data1, pb.data2, pb.data3)


"""

    inValOf(pb::ParamBox{T}) where {T} -> T

Return the value of the input variable of `pb`. Equivalent to `pb[]`.
"""
@inline inValOf(pb::ParamBox) = valOf.(dataOf(pb))


"""

    outValOf(pb::ParamBox) -> Number

Return the value of the output variable of `pb`. Equivalent to `pb()`.
"""
@inline outValOf(pb::ParamBox) = pb.map(inValOf(pb))

(pb::ParamBox)() = outValOf(pb)
# (pb::ParamBox)() = Base.invokelatest(pb.map, pb.data[][begin][])::Float64


"""

    inSymOf(pb::ParamBox) -> Symbol

Return the symbol of the input variable of `pb`.
"""
@inline inSymOf(pb::ParamBox) = symOf.(dataOf(pb))


"""

    outSymOf(pb::ParamBox) -> Symbol

Return the symbol of the output variable of `pb`.
"""
@inline outSymOf(::ParamBox{<:Any, V}) where {V} = V


# """

#     isInSymEqual(pb::ParamBox, sym::Symbol) -> Bool

# Return the Boolean value of whether the symbol of  `pb`'s input variable equals `sym`.
# """
# isInSymEqual(pb::ParamBox{<:Any, <:Any, <:Any, 1}, sym::TotTuple{Symbol}) = 
# inSymOf(pb) == unzipObj(sym)

# isInSymEqual(pb::ParamBox{<:Any, <:Any, <:Any, NP}, sym::NTuple{NP, Symbol}) where {NP} = 
# all(i==j for (i,j) in zip(inSymOfCore(pb), sym))

# isInSymEqual(pb::ParamBox, sym::Symbol, i::Int) = symOf(pb.data[][i]) == sym


# """

#     isOutSymEqual(::ParamBox, sym::Symbol) -> Bool

# Return the Boolean value of whether the symbol of  `pb`'s output variable equals `sym`.
# """
# isOutSymEqual(::ParamBox{<:Any, V}, sym::Symbol) where {V} = (V == sym)


"""

    indParOf(pb::ParamBox{<:Any, <:Any, <:Any, NP}) -> Tuple{Vararg{SemiMutableParameter{T}, NP}}

Return (the name and the value of) the independent variable tied to `pb`. Specifically, 
return the input variable stored in `pb` when `pb` is marked as differentiable; return the 
output variable of `pb` when `pb` is marked as non-differentiable. Thus, it is the variable 
`pb` represents to differentiate any (differentiable) function of [`ParamBox`](@ref)es.
"""
indParOf(pb::ParamBox) = (unzipObj∘ifelse)(isDiffParam(pb), dataOf(pb), pb)

"""

    indSymOf

"""
indSymOf(pb::ParamBox) = ifelse(isDiffParam(pb), outSymOf, inSymOf)(pb)


getTypeParams(::Type{<:ParamBox{T, V, F}}) where {T, V, F} = (T, V, F)

getTypeParams(::T) where {T<:ParamBox} = getTypeParams(T)


getFLevel(::Type{<:ParamBox{<:Any, <:Any, F}}) where {F} = getFLevel(F)

getFLevel(::T) where {T<:ParamBox} = getFLevel(T)


"""

    mapOf(pb::ParamBox) -> Function

Return the mapping function of `pb`.
"""
@inline mapOf(pb::ParamBox) = pb.map


"""

    outValCopy(pb::ParamBox{T, V}) where {T} -> ParamBox{T, V, $(iT)}

Return a new `ParamBox` of which the input variable's value is equal to the output 
variable's value of `pb`.
"""
outValCopy(pb::ParamBox{<:Any, V}) where {V} = 
genParamBox(Val(V), itself, OneParam(pb(), Symbol(IVsymSuffix, V)))

"""

    fullVarCopy(pb::T) where {T<:ParamBox} -> T

A shallow copy of the input `ParamBox`.
"""
fullVarCopy(pb::ParamBox{<:Any, V}) where {V} = 
genParamBox(Val(V), pb, pb.index[], pb.canDiff[])


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


"""

    changeMapping(pb::ParamBox{T, V}, mapFunction::Function, outSym::Symbol=V; 
                  canDiff::Union{Bool, Array{Bool, 0}}=isDiffParam(pb)) where {T, V} -> 
    ParamBox{T, outSym}

Change the mapping function of `pb`. The symbol of the output variable of the returned 
`ParamBox` can be specified by `outSym`, and its differentiability is determined by 
`canDiff`.
"""
changeMapping(pb::ParamBox{T, V}, mapFunction::Function, outSym::Symbol=V; 
              canDiff::Union0D{Bool}=isDiffParam(pb)) where {T, V} = 
genParamBox(Val(outSym), mapFunction, dataOf(pb), genIndex(), canDiff)


compareOneParam(op1::OneParam, op2::OneParam) = op1.data === op2.data

compareParamBoxCore1(pb1::MonoParamBox{T}, pb2::MonoParamBox{T}) where {T} = 
compareOneParam(dataOf(pb1)[], dataOf(pb2)[])

function compareParamBoxCore1(pb1::PloyParamBox{T}, pb2::PloyParamBox{T}) where {T}
    dataPb1 = dataOf(pb1)
    dataPb2 = dataOf(pb2)
    if length(dataPb1) == length(dataPb2)
        all( compareOneParam(op1, op2) for (op1, op2) in zip(dataPb1, dataPb2) )
    else
        false
    end
end

# compareParamBoxCore1(pb1::ParamBox{<:Any, <:Any, <:Any, NP}, 
#                      pb2::ParamBox{<:Any, <:Any, <:Any, NP}) where {NP} = 
# all( compareOneParam(op1, op2) for (op1, op2) in zip(pb1.data[], pb2.data))

compareParamBoxCore1(pb1::ParamBox, pb2::ParamBox) = false

function compareParamBoxCore2(pb1::ParamBox{T1, V1, F1}, 
                              pb2::ParamBox{T2, V2, F2}) where {T1, T2, V1, V2, F1, F2}
    bl = T1==T2 && V1==V2 && compareParamBoxCore1(pb1, pb2)
    if FLevel(F1) == FLevel(F2) == IL
        bl
    else
        bl * hasIdentical(pb1.map, pb2.map)
    end
end

@inline function compareParamBox(pb1::ParamBox, pb2::ParamBox)
    ifelse(( (bl=isDiffParam(pb1)) == isDiffParam(pb2) ),
        (bl ? compareParamBoxCore1(pb1, pb2) : compareParamBoxCore2(pb1, pb2)), false
    )
end


function addParamBox(pb1::MonoParamBox{T, V, FL1}, pb2::MonoParamBox{T, V, FL2}, 
                     roundAtol::Real=nearestHalfOf(getAtolVal(T))) where {T, V, FL1, FL2}
    if isDiffParam(pb1) && compareParamBox(pb1, pb2)
        genParamBox(Val(V), combinePF(+, pb1.map, pb2.map), 
                    pb1.data[][begin]=>min(pb1.data[][end], pb2.data[][end]), 
                    genIndex(), fill(true))
    else
        genParamBox(Val(V), itself, 
                 fill(roundToMultiOfStep(pb1() + pb2(), roundAtol))=>Symbol(IVsymSuffix, V))
    end
end


function mulParamBox(c::Number, pb::MonoParamBox{T, V}, 
                     roundAtol::Real=nearestHalfOf(getAtolVal(T))) where {T, V}
    if isDiffParam(pb)
        mapFunc = PF(pb.map, *, convert(T, roundToMultiOfStep(c, roundAtol)))
        ParamBox(Val(V), mapFunc, pb.data[], genIndex(), true)
    else
        ParamBox(Val(V), itself, 
                 fill(roundToMultiOfStep(T(pb()*c), roundAtol))=>Symbol(IVsymSuffix, V))
    end
end

function mulParamBox(pb1::MonoParamBox{T, V, FL1}, pb2::MonoParamBox{T, V, FL2}, 
                     roundAtol::Real=nearestHalfOf(getAtolVal(T))) where {T, V, FL1, FL2}
    if isDiffParam(pb1) && compareParamBox(pb1, pb2)
        ParamBox(Val(V), combinePF(*, pb1.map, pb2.map), 
                 pb1.data[][begin]=>min(pb1.data[][end], pb2.data[][end]), 
                 genIndex(), fill(true))
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
            res = outValCopy(pb1)
            res[] = getNearestMid(outVal1, outVal2, roundAtol)
            [res]
        else
            [pb1, pb2]
        end
    end
end