export genTensorVar, genMeshParam, genHeapParam, genCellParam, compareParamBox, uniqueParams, 
       dissectParam, setVal!, symOf, obtain, screenLevelOf, setScreenLevel!, inputOf, 
       sortParams!

const SymOrIndexedSym = Union{Symbol, IndexedSym}

struct Primitive end
struct Composite end
const StateType = Union{Primitive, Composite}

const Span{T} = Union{T, DirectMemory{T}}
const Pack{T} = Union{T, PackedMemory{T}}

abstract type ParamBox{T, E<:Pack{T}, S<:StateType} <: StateBox{E} end

const PrimitiveParam{T, E<:Pack{T}} = ParamBox{T, E, Primitive}
const CompositeParam{T, E<:Pack{T}} = ParamBox{T, E, Composite}

const UnitParam{T, S<:StateType} = ParamBox{T, T, S}
const SpanParam{T, E<:Span{T}, S<:StateType} = ParamBox{T, E, S}
const GridParam{T, N, S<:StateType} = SpanParam{T, DirectMemory{T, N}, S}
const NestParam{T, E<:Pack{T}, N, S<:StateType} = ParamBox{T, PackedMemory{T, E, N}, S}


const TensorVar{T, E<:Span{T}} = PrimitiveParam{T, E}
abstract type CellParam{T, E<:Pack{T}} <: CompositeParam{T, E} end
abstract type MeshParam{T, E<:Pack{T}, N} <: NestParam{T, E, N, Composite} end
abstract type HeapParam{T, E<:Pack{T}, N} <: NestParam{T, E, N, Composite} end

const TensorialParam{T, S<:StateType} = Union{UnitParam{T, S}, GridParam{T, <:Any, S}}
const AdaptableParam{T, E<:Pack{T}} = Union{CellParam{T, E}, MeshParam{T, E}}
const ReducibleParam{T, E<:Pack{T}, S<:StateType} = 
      Union{ParamBox{T, E, S}, NestParam{T, E, <:Any, S}}

const NestFixedParIn{T, E<:Pack{T}} = TriTupleUnion{ParamBox{T, E}}
const CoreFixedParIn{T, E<:Pack{T}} = TriTupleUnion{ReducibleParam{T, E}}

const ParamBoxAbtArr{P<:ParamBox, N} = AbstractArray{P, N}

const UnitOrVal{T} = Union{UnitParam{T}, T}
const UnitOrValVec{T} = AbstractVector{<:UnitOrVal{T}}

isOffsetEnabled(::ParamBox) = false

function checkScreenLevel(sl::Int, levels::NonEmptyTuple{Int})
    if !(sl in levels)
        throw(DomainError(sl, "This screen level ($(TernaryNumber(sl))) is not allowed."))
    end
    sl
end

checkScreenLevel(s::TernaryNumber, levels::NonEmptyTuple{Int}) = 
checkScreenLevel(Int(s), levels)


getScreenLevelOptions(::Type{<:PrimitiveParam}) = (1, 2)

screenLevelOf(p::PrimitiveParam) = (p.screen + 1)

function checkPrimParamElementalType(::Type{T}) where {T}
    if !canDirectlyStoreInstanceOf(T)
        throw(AssertionError("`T::Type{$T}` should have let $canDirectlyStoreInstanceOf "*
                             "return `true`."))
    end
    nothing
end

function checkPrimParamElementalType(::Type{T}) where 
                                    {T<:Union{Nothing, Missing, IdentityMarker}}
    throw(AssertionError("`T::Type{$T}` is not supported"))
end


mutable struct UnitVar{T} <: TensorVar{T, T}
    @atomic input::T
    const symbol::IndexedSym
    @atomic screen::Bool

    function UnitVar(input::T, symbol::SymOrIndexedSym, screen::Bool=false) where {T}
        checkPrimParamElementalType(T)
        new{T}(input, IndexedSym(symbol), screen)
    end
end

mutable struct GridVar{T, N} <: TensorVar{T, DirectMemory{T, N}}
    const input::DirectMemory{T, N}
    const symbol::IndexedSym
    @atomic screen::Bool

    function GridVar(input::AbstractArray{T, N}, symbol::SymOrIndexedSym, 
                     screen::Bool=false) where {T, N}
        N < 1 && throw(AssertionError("`N` must be larger than zero."))
        checkPrimParamElementalType(T)
        input = decoupledCopy(input)
        new{T, N}(input, IndexedSym(symbol), screen)
    end
end

genTensorVar(input::Any, symbol::SymOrIndexedSym, screen::Bool=false) = 
UnitVar(input, symbol, screen)

genTensorVar(input::AbstractArray, symbol::SymOrIndexedSym, screen::Bool=false) = 
GridVar(input, symbol, screen)

genTensorVar(input::AbtArray0D, symbol::SymOrIndexedSym, screen::Bool=false) = 
genTensorVar(first(input), symbol, screen)


getScreenLevelOptions(::Type{<:HeapParam}) = (0,)

screenLevelOf(p::HeapParam) = 0

struct ShapedParam{T, E<:Pack{T}, N, P<:ParamBox{T, E}} <: HeapParam{T, E, N}
    input::ShapedMemory{P, N}
    symbol::IndexedSym

    function ShapedParam(input::ShapedMemory{P, N}, symbol::SymOrIndexedSym) where 
                        {T, E, P<:ParamBox{T, E}, N}
        N < 1 && throw(AssertionError("`N` must be larger than zero."))
        checkEmptiness(input.value, :input)
        new{T, E, N, P}(copy(input), IndexedSym(symbol))
    end
end

genHeapParam(input::AbstractArray{<:ParamBox{T, E}}, 
             symbol::SymOrIndexedSym) where {T, E<:Pack{T}} = 
ShapedParam(ShapedMemory(input), symbol)

genHeapParam(input::AbstractArray{<:ParamBox, 0}, ::SymOrIndexedSym) = first(input)

genHeapParam(input::ShapedParam, symbol::IndexedSym=input.symbol) = 
ShapedParam(input.input, symbol)


abstract type TypedTensorFunc{T, N} <: CompositeFunction end

struct TypedReduce{T, F<:Function} <: TypedTensorFunc{T, 0}
    f::F
    type::Type{T}
end

TypedReduce(f::TypedReduce, ::Type{T}) where {T} = TypedReduce(f.f, T)

TypedReduce(::Type{T}) where {T} = TypedReduce(itself, T)

(f::TypedReduce)(arg, args...) = f.f(arg, args...)::getOutputType(f)

getOutputType(::Type{TypedReduce{T, F}}) where {T, F<:Function} = 
typeintersect(getOutputType(F), T)

getOutputType(::Type{<:TypedReduce{<:AbstractArray{T, N}, F}}) where {T, N, F<:Function} = 
typeintersect(getOutputType(F), getPackType(AbstractArray{T, N}))


struct TypedExpand{T, N, F<:Function} <: TypedTensorFunc{T, N}
    f::F
    type::Type{T}
    shape::TruncateReshape{N}

    function TypedExpand(f::Function, ::Type{T}, args::NonEmptyTuple{Any}, 
                         shape::MissingOr{TruncateReshape}=missing; 
                         truncate::Union{Bool, TernaryNumber}=
                         (ismissing(shape) ? false : shape.truncate)) where {T}
        shapeTemp = if ismissing(shape)
            if f isa TypedExpand
                f.shape
            else
                ReturnTyped(f, getPackType(AbstractArray{T}))(args...) |> size
            end
        else
            shape
        end
        shapeFinal = TruncateReshape(shapeTemp; truncate)

        fCore = (f isa TypedExpand) ? f.f : f

        ## Already checked by `TruncateReshape`
        # N = length(shapeFinal.axis)
        # N==0 && throw(AssertionError("The dimension of `f`'s returned value must be "*
        #                              "larger than zero."))
        # prod(shapeFinal.axis) == 0 && throw(AssertionError("The returned value of `f` "*
        #                                                    "must not be empty."))

        new{T, length(shapeFinal.axis), typeof(fCore)}(fCore, T, shapeFinal)
    end

    function TypedExpand(f::Function, args::NonEmptyTuple{Any})
        output = ReturnTyped(f, AbstractArray)(args...)
        shape = TruncateReshape(output|>size)
        type = eltype(output) |> genPackMemoryType
        fCore = (f isa TypedExpand) ? f.f : f

        new{type, length(shape.axis), typeof(fCore)}(fCore, type, shape)
    end

    function TypedExpand(f::TypedExpand{T, N, F}, 
                         shape::TruncateReshape{N}=f.shape) where {T, N, F<:Function}
        new{T, N, F}(f.f, T, shape)
    end
end

function TypedExpand(arr::AbstractArray{T, N}, 
                     shape::TruncateReshape{N}=TruncateReshape(arr)) where {T, N}
    TypedExpand(T, shape)
end

function (f::TypedExpand{T, N})(arg, args...) where {T, N}
    res = f.f(arg, args...)::getOutputType(f)
    f.shape(res)
end

function getOutputType(::Type{TypedExpand{T, N, F}}) where {T, N, F<:Function}
    type = getOutputType(F)
    if isconcretetype(type) && type <: AbstractArray{T, N}
        type
    else
        getPackType(AbstractArray{T, N})
    end
end


#= Additional Method =#
import Quiqbox: getNestedLevelCore
function getNestedLevelCore(::Type{<:ParamBox{<:Any, E}}, level::Int) where {E<:Pack}
    getNestedLevelCore(E, level)
end


function checkParamOffsetMethods(::Type{T}) where {T}
    (hasmethod(unitAdd, NTuple{2, T}) && hasmethod(unitSub, NTuple{2, T})) || 
    (hasmethod(+, NTuple{2, T}) && hasmethod(-, NTuple{2, T}))
end


function getScreenLevelOptionsCore(::Type{E}) where {E}
    Tuple(0:(checkParamOffsetMethods(E) * 2))
end

function getScreenLevelOptionsCore(::Type{E}) where {E<:AbstractArray}
    nl = getNestedLevel(E)
    if nl.level > 1
        (0,)
    else
        Tuple(0:(checkParamOffsetMethods(nl|>getCoreType) * 2))
    end
end


getScreenLevelOptions(::Type{<:CellParam{T, E}}) where {T, E<:Pack{T}} = 
getScreenLevelOptionsCore(E)

getScreenLevelOptions(::Type{<:MeshParam{T, E}}) where {T, E<:Pack{T}} = 
getScreenLevelOptionsCore(PackedMemory{T, E})

screenLevelOf(p::AdaptableParam) = Int(p.screen)


function unitAdd(a::NonEmptyTuple{Number, N}, b::NonEmptyTuple{Number, N}) where {N}
    map(a, b) do i, j
        typeof(i)(i + j)
    end
end

function unitSub(a::NonEmptyTuple{Number, N}, b::NonEmptyTuple{Number, N}) where {N}
    map(a, b) do i, j
        typeof(i)(i - j)
    end
end

unitOpMatch(::typeof(+)) = unitAdd

unitOpMatch(::typeof(-)) = unitSub

function unitOp(f::F, a::T1, b::T2) where {F<:Union{typeof(+), typeof(-)}, T1, T2}
    fSpecialize = unitOpMatch(f)
    if hasmethod(fSpecialize, Tuple{T1, T2})
        fSpecialize(a, b)
    else
        f(a, b) |> T1
    end
end

function unitOp(f::F, a::AbstractArray{<:Any, N}, b::AbstractArray{<:Any, N}) where 
               {F<:Union{typeof(+), typeof(-)}, N}
    res = similar(a)
    foreach(eachindex(res), a, b) do n, i, j
        res[n] = unitOp(f, i, j)
    end
    res
end


function decoupledCopy(arr::AbstractArray)
    map(decoupledCopy, arr) |> PackedMemory
end

function decoupledCopy(obj::T) where {T}
    if canDirectlyStoreInstanceOf(T)
        obj
    else
        deepcopy(obj)
    end
end


getTensorOutputTypeBound(::TypedReduce{E}) where {E} = E
getTensorOutputTypeBound(::TypedExpand{E}) where {E} = AbstractArray{E}

getTensorOutputShape(f::TypedExpand, ::Any) = f.shape.axis
getTensorOutputShape(::TypedReduce{E}, input) where {E} = ()
getTensorOutputShape(f::TypedReduce{E}, input) where {E<:AbstractArray} = 
getTensorOutputShape(f.f, input)
getTensorOutputShape(f::Function, input) = size(f(obtain.(input)...))

function formatOffset(lambda::TypedTensorFunc{<:Pack}, ::Missing, input)
    outType = getTensorOutputTypeBound(lambda)
    if getScreenLevelOptionsCore(outType) == (0,)
        ()
    else
        val = if outType <: Number
            zero(outType)
        elseif outType <: AbstractArray{<:Number}
            shape = getTensorOutputShape(lambda, input)
            PackedMemory(zeros(eltype(outType), shape))
        else
            buffer = lambda(obtain.(input)...)
            res = unitOp(-, buffer, buffer)
            res isa AbstractArray ? PackedMemory(res) : res
        end
        (val,)
    end
end

function formatOffset(lambda::TypedTensorFunc{<:Pack}, offset, input)
    outType = getTensorOutputTypeBound(lambda)
    if getScreenLevelOptionsCore(outType) == (0,)
        ()
    else
        val = if outType <: AbstractArray
            nl = getNestedLevel(outType)
            if nl != getNestedLevel(offset|>typeof)
                cT = getCoreType(nl)
                throw(AssertionError("The nested level and the core type of `offset` "*
                                     "should be `$(nl.level)` and `$cT`, respectively."))
            end

            if eltype(outType) <: Number
                size(offset) == getTensorOutputShape(lambda, input)
            else
                buffer = lambda(obtain.(input)...)
                recursiveCompareSize(buffer, offset)
            end || throw(DimensionMismatch("The shape of `offset` does not match that "*
                                            "of `lambda`'s returned value."))
            PackedMemory(offset)
        else
            convert(outType, offset)
        end
        (val,)
    end
end

# ([[x]], [x]) -> {[x]}, (x, x) -> {x}
getCellOutputLevels(::CoreFixedParIn{T, E}) where {T, E<:Pack{T}} = 
Set(getNestedLevel(E).level)
# ([x], [x]) -> {x, [x]}
function getCellOutputLevels(::NestFixedParIn{T, E}) where {T, E<:PackedMemory{T}}
    level = getNestedLevel(E).level
    Set((level, level-1))
end
#! Consider efficient construction for when `f` is a `ReturnTyped`.
function formatTensorFunc(f::Function, ::Type{TypedReduce}, 
                          input::CoreFixedParIn{T, E}) where {T, E<:Pack{T}}
    lambda = if f isa TypedReduce{<:Pack}
        f
    else
        type = f(obtain.(input)...) |> typeof |> genPackMemoryType
        TypedReduce(f, type)
    end
    targetLevels = getCellOutputLevels(input)
    actualLevel = getNestedLevel(lambda.type).level
    if !(actualLevel in targetLevels)
        throw("The nested level of `f`'s output is $actualLevel "*
              "which should have been within `$targetLevels`.")
    end
    lambda
end
# ([[x]], [x]) -> [[x]]; ([x], [x]) -> [[x]]; 
function formatTensorFunc(f::Function, ::Type{TypedExpand}, 
                          input::CoreFixedParIn{T, E}) where {T, E<:Pack{T}}
    lambda = f isa TypedExpand{<:Pack} ? f : TypedExpand(f, obtain.(input))
    targetEleLevel = getNestedLevel(E).level
    actualEleLevel = getNestedLevel(lambda.type).level
    if actualEleLevel != targetEleLevel
        throw("The nested level of `f`'s output is $actualEleLevel"*
              "which should have been `$(targetEleLevel+1)`.")
    end
    lambda
end


mutable struct ReduceParam{T, E<:Pack{T}, F<:Function, I<:CoreFixedParIn} <: CellParam{T, E}
    const lambda::TypedReduce{E, F}
    const input::I
    const symbol::IndexedSym
    @atomic screen::TernaryNumber
    @atomic offset::E

    function ReduceParam(lambda::TypedReduce{E, F}, input::I, 
                         symbol::SymOrIndexedSym, screen::TernaryNumber=TUS0, 
                         offset::Union{E, Missing}=missing) where 
                        {T, E<:Pack{T}, F, I<:CoreFixedParIn}
        sym = IndexedSym(symbol)
        offsetTuple = formatOffset(lambda, offset, input)
        if isempty(offsetTuple)
            new{T, E, F, I}(lambda, input, sym, screen)
        else
            new{T, E, F, I}(lambda, input, sym, screen, first(offsetTuple))
        end
    end
end

function genCellParam(func::Function, input::CoreFixedParIn, symbol::SymOrIndexedSym)
    lambda = formatTensorFunc(func, TypedReduce, input)
    ReduceParam(lambda, input, symbol, TUS0, missing)
end

function genCellParam(par::ReduceParam, symbol::SymOrIndexedSym=symOf(par))
    offset = isOffsetEnabled(par) ? par.offset : missing
    ReduceParam(par.lambda, par.input, symbol, par.screen, offset)
end

function genCellParam(input::UnitParam{T}, symbol::SymOrIndexedSym=symOf(input)) where {T}
    ReduceParam(TypedReduce(T), (input,), symbol)
end

function genCellParam(input::ParamBox{T, E}, symbol::SymOrIndexedSym=symOf(input)) where 
                     {T, E<:PackedMemory{T}}
    ReduceParam(TypedReduce(E), (input,), symbol)
end

genCellParam(var, varSym::SymOrIndexedSym, symbol::SymOrIndexedSym=varSym) = 
genCellParam(genTensorVar(var, varSym), symbol)


indexedSymOf(p::ParamBox) = p.symbol

symOf(p::ParamBox) = indexedSymOf(p).name

inputOf(p::ParamBox) = p.input


mutable struct ExpandParam{T, E<:Pack{T}, N, F<:Function, I<:CoreFixedParIn
                           } <: MeshParam{T, E, N}
    const lambda::TypedExpand{E, N, F}
    const input::I
    const symbol::IndexedSym
    @atomic screen::TernaryNumber
    @atomic offset::PackedMemory{T, E, N}

    function ExpandParam(lambda::TypedExpand{E, N, F}, input::I, 
                         symbol::SymOrIndexedSym, screen::TernaryNumber=TUS0, 
                         offset::Union{PackedMemory{T, E, N}, Missing}=missing
                         ) where {T, E<:Pack{T}, N, F, I<:CoreFixedParIn}
        sym = IndexedSym(symbol)
        offsetTuple = formatOffset(lambda, offset, input)
        if isempty(offsetTuple)
            new{T, E, N, F, I}(lambda, input, sym, screen)
        else
            new{T, E, N, F, I}(lambda, input, sym, screen, first(offsetTuple))
        end
    end
end

function genMeshParam(func::Function, input::CoreFixedParIn, symbol::SymOrIndexedSym)
    lambda = formatTensorFunc(func, TypedExpand, input)
    ExpandParam(lambda, input, IndexedSym(symbol), TUS0, missing)
end

function genMeshParam(par::ExpandParam, symbol::SymOrIndexedSym=symOf(par))
    offset = isOffsetEnabled(par) ? par.offset : missing
    ExpandParam(par.lambda, par.input, IndexedSym(symbol), par.screen, offset)
end

function isScreenLevelChangeable(::Type{T}) where {T<:ParamBox}
    minLevel, maxLevel = extrema( getScreenLevelOptions(T) )
    (maxLevel - minLevel) > 0
end

function isOffsetEnabled(pb::T) where {T<:AdaptableParam}
    isScreenLevelChangeable(T) && maximum( getScreenLevelOptions(T) ) > 0 && 
    isdefined(pb, :offset) # Only for safety
end


function indexParam(pb::ShapedParam, oneToIdx::Int, sym::MissingOr{Symbol}=missing)
    entry = pb.input[begin+oneToIdx-1]
    if ismissing(sym) || sym==symOf(entry)
        entry
    elseif entry isa MeshParam
        genMeshParam(entry, sym)
    else
        genCellParam(entry, sym)
    end
end

function indexParam(pb::AdaptableParam, oneToIdx::Int, sym::MissingOr{Symbol}=missing)
    ismissing(sym) && (sym = Symbol(symOf(pb), oneToIdx))
    genCellParam(GetOneToIndex(oneToIdx), (pb,), sym)
end

function indexParam(pb::UnitParam, oneToIdx::Int, sym::MissingOr{Symbol}=missing)
    if oneToIdx != 1
        throw(BoundsError(pb, oneToIdx))
    elseif ismissing(sym) || sym == symOf(res)
        pb
    else
        genCellParam(pb, sym)
    end
end


# level 0: λ(input) + offset
# level 1: itself(offset)
# level 2: offset
function setScreenLevel!(p::AdaptableParam, level::Int)
    checkScreenLevel(level, getScreenLevelOptions(p|>typeof))
    levelOld = screenLevelOf(p)
    if levelOld == level
    elseif levelOld == 0
        @atomic p.offset = obtain(p)
    elseif level == 0
        newVal = p.lambda((obtain(arg) for arg in p.input)...)
        @atomic p.offset = unitOp(-, p.offset, newVal)
    end
    @atomic p.screen = TernaryNumber(level)
    p
end

function setScreenLevel!(p::TensorVar, level::Int)
    checkScreenLevel(level, getScreenLevelOptions(p|>typeof))
    @atomic p.screen = Bool(level-1)
    p
end


isDependentParam(p::ParamBox) = (screenLevelOf(p)  < 1)
isPrimitiveInput(p::ParamBox) = (screenLevelOf(p) == 1)
isFrozenVariable(p::ParamBox) = (screenLevelOf(p) == 2)


getOutputSize(p::ShapedParam) = size(p.input)

getOutputSize(p::PrimitiveParam) = size(p.input)

function getOutputSize(p::AdaptableParam)
    if p.offset isa AbstractArray
        size(p.offset)
    else
        ()
    end
end


getOutputType(::P) where {P<:ParamBox} = getOutputType(P)

getOutputType(::Type{<:ParamBox}) = Any

getOutputType(::Type{<:ParamBox{T, E}}) where {T, E<:Pack{T}} = getPackType(E)

getOutputType(::Type{<:PrimitiveParam{T, E}}) where {T, E<:Pack{T}} = E

function getOutputType(::Type{<:HeapParam{T, E, N}}) where {T, E<:Pack{T}, N}
    innerType = getPackType(E)
    if isconcretetype(innerType)
        AbstractArray{innerType, N}
    else
        AbstractArray{<:innerType, N}
    end
end

const ParamEgalBox = EgalBox{ParamBox}
const UnitParamEgalBox = EgalBox{UnitParam}
const GridParamEgalBox = EgalBox{GridParam}
const NestParamEgalBox = EgalBox{NestParam}

function hasCycleCore!(::Set{ParamEgalBox}, ::Set{ParamEgalBox}, 
                       edge::Pair{<:PrimitiveParam, <:NothingOr{ParamBox}}, 
                       ::Bool, finalizer::F=itself) where {F<:Function}
    finalizer(edge)
    (false, edge.first)
end

function hasCycleCore!(localTrace::Set{ParamEgalBox}, history::Set{ParamEgalBox}, 
                       edge::Pair{<:CompositeParam, <:NothingOr{ParamBox}}, 
                       strictMode::Bool, finalizer::F=itself) where {F<:Function}
    here = edge.first

    if strictMode || screenLevelOf(here) == 0
        key = ParamEgalBox(here)
        if key in localTrace
            return (true, here)
        end

        if !(key in history)
            push!(localTrace, key)

            for p in here.input
                res = hasCycleCore!(localTrace, history, p=>here, strictMode, finalizer)
                if first(res)
                    return res
                end
            end

            pop!(localTrace, key)
            push!(history, key)
        end
    end

    finalizer(edge)
    (false, here)
end

function hasCycle(param::ParamBox; strictMode::Bool=true, finalizer::Function=itself, 
                  catcher::Array{ParamBox, 0}=Array{ParamBox, 0}( undef, () ))
    localTrace = Set{ParamEgalBox}()
    parHistory = Set{ParamEgalBox}()
    bl, lastP = hasCycleCore!(localTrace, parHistory, param=>nothing, strictMode, finalizer)
    catcher[] = lastP
    bl
end


function obtainCore!(cache::LRU{ParamEgalBox}, param::PrimitiveParam)
    input = param.input
    get!(cache, ParamEgalBox(param), decoupledCopy(input))::typeof(input)
end

function obtainCore!(cache::LRU{ParamEgalBox}, param::ShapedParam)
    map(param.input) do p
        obtainCore!(cache, p)
    end::getOutputType(param)
end

function obtainCore!(cache::LRU{ParamEgalBox}, param::AdaptableParam)
    key = ParamEgalBox(param)
    get!(cache, key) do
        if screenLevelOf(param) > 0
            decoupledCopy(param.offset)
        else
            inVal = (obtainCore!(cache, p) for p in param.input)
            body = param.lambda(inVal...)
            isOffsetEnabled(param) ? unitOp(+, body, param.offset) : body
        end
    end::getOutputType(param)
end

function checkParamCycle(param::ParamBox; strictMode=false, finalizer::Function=itself)
    catcher = Array{ParamBox, 0}(undef, ())
    if hasCycle(param; strictMode, finalizer, catcher)
        throw(AssertionError("`param`:\n    $param\n\n"*"has a reachable cycle at:\n    "*
                             "$(catcher[])"))
    end
end

function obtain(param::CompositeParam)
    checkParamCycle(param)
    if param isa AdaptableParam && screenLevelOf(param) > 0
        decoupledCopy(param.offset)
    else
        cache = LRU{ParamEgalBox, Any}(maxsize=100)
        obtainCore!(cache, param)
    end
end

obtain(param::PrimitiveParam) = decoupledCopy(param.input)

function obtain(params::ParamBoxAbtArr)
    if isempty(params)
        Union{}[]
    else
        cache = LRU{ParamEgalBox, Any}(maxsize=min( 500, 100length(params) ))
        map(params) do param
            checkParamCycle(param)
            obtainCore!(cache, param)
        end
    end
end

(pn::ParamBox)() = obtain(pn)


function setVal!(par::PrimitiveParam, val)
    if !isPrimitiveInput(par)
        throw(AssertionError("`isPrimitiveInput(par)` must return `true`."))
    end
    @atomic par.input = val
end

function setVal!(par::AdaptableParam, val)
    if !isPrimitiveInput(par)
        throw(AssertionError("`isPrimitiveInput(par)` must return `true`."))
    end
    @atomic par.offset = val
end


struct BoxCoreType{T, B<:Box} <: StructuredType end

struct ParamMarker{T} <: IdentityMarker{BoxCoreType{T, ParamBox}}
    code::UInt
    type::NestedLevel{T}
    data::IdentityMarker
    func::IdentityMarker
    meta::Tuple{ValMkrPair, ValMkrPair{Int}, ValMkrPair{Symbol}}

    function ParamMarker(param::P) where {P<:ParamBox}
        type = getNestedLevel(P)
        code = hash(type)

        switch = isOffsetEnabled(param)
        offset = :offset => markObj(switch ? param.offset : nothing)
        code = hash(offset.second.code, code)

        sl = screenLevelOf(param)
        screen = :screen => markObj(sl)
        code = hash(screen.second, code)

        sym = :symbol => (markObj∘symOf)(param)
        code = hash(sym.second, code)

        meta = (offset, screen, sym)

        func = markObj((P <: AdaptableParam && sl == 0) ? param.lambda : nothing)
        code = hash(func, code)

        data = if P <: PrimitiveParam
            Identifier(param.input)
        elseif switch && sl > 0
            Identifier(param)
        else
            checkParamCycle(param)
            markParamInput(param.input)
        end
        code = hash(data.code, code)

        new{getCoreType(type)}(code, type, data, func, meta)
    end
end

function markParamInput(input::NonEmptyTuple{ParamBox})
    markObj( ParamMarker.(input) )
end

function markParamInput(input::ParamBoxAbtArr)
    markObj(input)
end

#= Additional Method =#
markObj(input::ParamBox) = ParamMarker(input)

function ==(marker1::T, marker2::T) where {T<:ParamMarker}
    if marker1.code == marker2.code
        marker1.type == marker2.type && 
        marker1.data == marker2.data && 
        marker1.func == marker2.func && 
        marker1.meta == marker2.meta
    else
        false
    end
end


compareParamBox(p1::T, p2::T) where {T<:PrimitiveParam} = p1 === p2

function compareParamBox(p1::ParamBox{T, E, S}, p2::ParamBox{T, E, S}) where 
                        {T, E<:Pack{T}, S<:StateType}
    p1 === p2 || ParamMarker(p1) == ParamMarker(p2)
end

compareParamBox(::ParamBox, ::ParamBox) = false


function uniqueParamsCore!(source::ParamBoxAbtArr)
    unique!(ParamEgalBox, source)
    source
end

function uniqueParams(source::ParamBoxAbtArr)
    uniqueParamsCore!(source)
end

function uniqueParams(source)
    map(last, getFieldParams(source)) |> uniqueParamsCore!
end


function getFieldParams(source)
    paramPairs = Pair{ChainedAccess, ParamBox}[]
    getFieldParamsCore!(paramPairs, source, ChainedAccess())
    paramPairs
end

function isParamBoxFree(source)
    canDirectlyStore(source) || (getFieldParams(source) |> isempty)
end

function getFieldParamsCore!(paramPairs::AbstractVector{Pair{ChainedAccess, ParamBox}}, 
                             source::ParamBox, anchor::ChainedAccess)
    push!(paramPairs, anchor=>source)
    nothing
end

function getFieldParamsCore!(paramPairs::AbstractVector{Pair{ChainedAccess, ParamBox}}, 
                             source::T, anchor::ChainedAccess) where {T}
    searchParam = false
    if source isa Union{Tuple, AbstractArray}
        if isempty(source)
            return nothing
        else
            searchParam = true
            fields = eachindex(source)
        end
    elseif isstructtype(T) && !canDirectlyStore(source)
        searchParam = true
        fields = fieldnames(T)
    end
    if searchParam
        for fieldSym in fields
            field = getField(source, fieldSym)
            anchorNew = ChainedAccess(anchor, fieldSym)
            getFieldParamsCore!(paramPairs, field, anchorNew)
        end
    end
    nothing
end


struct ReduceShift{T, F<:Function} <: TypedTensorFunc{T, 0}
    apply::TypedReduce{T, F}
    shift::T
end

(f::ReduceShift)(args...) = unitOp(+, f.apply(args...), f.shift)

struct ExpandShift{T, N, F<:Function} <: TypedTensorFunc{T, N}
    apply::TypedExpand{T, N, F}
    shift::ShapedMemory{T, N}
end

(f::ExpandShift)(args...) = unitOp(+, f.apply(args...), f.shift)

extractTransformCore(::ReduceParam) = ReduceShift
extractTransformCore(::ExpandParam) = ExpandShift

function extractTransform(pb::AdaptableParam)
    fCore = if isOffsetEnabled(pb)
        extractTransformCore(pb)(pb.lambda, pb.offset)
    else
        pb.lambda
    end
    ReturnTyped(fCore, getOutputType(pb))
end

function extractTransform(pb::ShapedParam)
    ReturnTyped(itself, getOutputType(pb))
end

struct ParamBoxClassifier <: StatefulFunction{ParamBox}
    holder::Vector{Pair{ <:ParamBox, Array{Bool, 0} }}
    linker::Vector{Pair{ <:ParamBox, Array{Bool, 0} }}
    history::IdDict{ParamBox, Int}

    function ParamBoxClassifier()
        new(Pair{<:ParamBox, Array{Bool, 0}}[], Pair{<:ParamBox, Array{Bool, 0}}[], 
            IdDict{ParamBox, Int}())
    end
end

function (f::ParamBoxClassifier)(edge::Pair{<:ParamBox, <:NothingOr{ParamBox}})
    here, next = edge
    sector = ifelse(isDependentParam(here), f.linker, f.holder)
    hasDescendent = (next !== nothing)
    idx = get!(f.history, here) do
        push!(sector, here=>fill(false))
        lastindex(sector)
    end
    sector[idx].second[] = hasDescendent
    nothing
end


function dissectParamCore(pars::ParamBoxAbtArr)
    finalizer = ParamBoxClassifier()

    foreach(pars) do par
        checkParamCycle(par; finalizer)
    end

    source = initializeSpanParamSet()
    hidden = ParamBox[]
    output = ParamBox[]
    direct = ParamBox[]

    for (dest1, dest2, sector) in ( (source, direct, finalizer.holder), 
                                    (hidden, output, finalizer.linker) )
        for pair in sector
            param = pair.first
            dest = ifelse(pair.second[], dest1, dest2)
            if dest isa TypedSpanParamSet
                level = getNestedLevel(param|>typeof).level
                container = ifelse(level==0, first, last)(dest)
                push!(container, param)
            else
                push!(dest, param)
            end
        end
    end

    (source=source, hidden=hidden, output=output, direct=direct)
end

dissectParam(params::ParamBoxAbtArr) = (dissectParamCore∘unique)(ParamEgalBox, params)
dissectParam(source::Any) = (dissectParamCore∘uniqueParams)(source)
dissectParam(source::ParamBox) = dissectParamCore(source|>fill)


function getSourceParamSet(source; onlyVariable::Bool=true, includeSink::Bool=true)
    source, _, _, direct = dissectParam(source)

    if includeSink
        for par in direct
            push!(getfield(source, par isa UnitParam ? :unit : :grid), par)
        end
    end

    onlyVariable && foreach(sector->filter!(isPrimitiveInput, sector), source)

    source
end


function markParam!(param::ParamBox, 
                    indexDict::AbstractDict{Symbol, Int}=Dict{Symbol, Int}())
    sym = symOf(param)
    get!(indexDict, sym, 0)
    param.symbol.index = (indexDict[sym] += 1)
    nothing
end

function sortParams!(params::AbstractVector{<:ParamBox}; indexing::Bool=true)
    encoder = function (x::ParamBox)
        nl = getNestedLevel(x|>typeof)
        (screenLevelOf(x), nl.level, symbolFrom(x.symbol), objectid(x))
    end

    sort!(params, by=encoder)

    if indexing
        parIdxDict = Dict{Symbol, Int}()
        for par in params
            markParam!(par, parIdxDict)
        end
    end
    params
end


struct UnitParamEncoder{T}
    symbol::Symbol
    screen::TernaryNumber

    function UnitParamEncoder(::Type{T}, symbol::Symbol, screen::TernaryNumber) where {T}
        new{T}(symbol, screen)
    end
end

UnitParamEncoder(::Type{T}, symbol::Symbol, screen::Int=1) where {T} = 
UnitParamEncoder(T, symbol, TernaryNumber(screen))

(f::UnitParamEncoder)(input::UnitParam) = itself(input)

function (f::UnitParamEncoder{T})(input) where {T}
    p = genCellParam(T(input), f.symbol)
    setScreenLevel!(p, Int(f.screen))
end


const AbstractSpanSet{U<:AbstractVector, G<:AbstractVector} = @NamedTuple{unit::U, grid::G}

const AbstractSpanValueSet{U<:AbstractVector, G<:AbtVecOfAbtArr} = AbstractSpanSet{U, G}

const OptionalSpanValueSet{U<:NothingOr{AbstractVector}, G<:NothingOr{AbtVecOfAbtArr}} = 
      @NamedTuple{unit::U, grid::G}

const OptionalUnitValueSet{U<:NothingOr{AbstractVector}, G<:NothingOr{AbtBottomVector}} = 
      @NamedTuple{unit::U, grid::G}

const OptionalGridValueSet{U<:NothingOr{AbtBottomVector}, G<:NothingOr{AbtVecOfAbtArr}} = 
      @NamedTuple{unit::U, grid::G}

const OptionalVoidValueSet{U<:NothingOr{AbtBottomVector}, G<:NothingOr{AbtBottomVector}} = 
      @NamedTuple{unit::U, grid::G}

const AbstractSpanIndexSet{U<:AbstractVector{<:OneToIndex}, 
                           G<:AbstractVector{<:OneToIndex}} = 
      AbstractSpanSet{U, G}

const FixedSpanIndexSet{U<:OneToIndex, G<:OneToIndex} = 
      AbstractSpanIndexSet{Memory{U}, Memory{G}}

const AbstractSpanParamSet{U<:AbstractVector{<:UnitParam}, G<:AbstractVector{<:GridParam}} = 
      AbstractSpanSet{U, G}

const TypedSpanParamSet{T1<:UnitParam, T2<:GridParam} = 
      AbstractSpanParamSet{Vector{T1}, Vector{T2}}

const FixedSpanParamSet{T1<:UnitParam, T2<:GridParam} = 
      AbstractSpanParamSet{Memory{T1}, Memory{T2}}

genFixedVoidSpanSet() = (unit=genBottomMemory(), grid=genBottomMemory())

struct UnitInput end
struct GridInput end
struct SpanInput end
struct VoidInput end

getInputSymbol(::Type{UnitInput}) = :unit
getInputSymbol(::Type{GridInput}) = :grid
getInputSymbol(::Type{SpanInput}) = :span
getInputSymbol(::Type{VoidInput}) = :void

constrainSpanValueSet(::UnitInput, ::OptionalUnitValueSet) = getInputSymbol(UnitInput)
constrainSpanValueSet(::GridInput, ::OptionalGridValueSet) = getInputSymbol(GridInput)
constrainSpanValueSet(::SpanInput, ::OptionalSpanValueSet) = getInputSymbol(SpanInput)
constrainSpanValueSet(::VoidInput, ::OptionalVoidValueSet) = getInputSymbol(VoidInput)

function getParamInputType end


initializeSpanParamSet() = (unit=UnitParam[], grid=GridParam[])

initializeSpanParamSet(::Type{T}) where {T} = (unit=UnitParam{T}[], grid=GridParam{T}[])

initializeSpanParamSet(::Nothing) = genFixedVoidSpanSet()

initializeSpanParamSet(units::AbstractVector{<:UnitParam}, 
                       grids::AbstractVector{<:GridParam}) = (unit=units, grid=grids)

initializeSpanParamSet(unit::UnitParam) = (unit=genMemory(unit), grid=genBottomMemory())

initializeSpanParamSet(grid::GridParam) = (unit=genBottomMemory(), grid=genMemory(grid))


function locateParamCore!(params::AbstractVector, target::ParamBox)
    if isempty(params)
        push!(params, target)
        OneToIndex(1)
    else
        idx = findfirst(x->compareObj(x, target), params)
        if idx === nothing
            push!(params, target)
            idx = lastindex(params)
        end
        OneToIndex(idx - firstindex(params) + 1)
    end
end

function locateParam!(params::AbstractVector, target::ParamBox)
    (ChainedAccess∘locateParamCore!)(params, target)
end

function locateParam!(paramSet::AbstractSpanParamSet, target::UnitParam)
    GetIndex{UnitIndex}(locateParamCore!(paramSet.unit, target))
end

function locateParam!(paramSet::AbstractSpanParamSet, target::GridParam)
    GetIndex{GridIndex}(locateParamCore!(paramSet.grid, target))
end

const NamedParamTuple{S, N, P<:NTuple{N, ParamBox}} = NamedTuple{S, P}

const SpanParamNamedTuple{S, N, P<:NTuple{N, SpanParam}} = NamedParamTuple{S, N, P}

const ParamBoxSource = Union{ParamBoxAbtArr, Tuple{Vararg{ParamBox}}, NamedParamTuple, 
                             AbstractSpanParamSet}

function locateParam!(params::Union{AbstractSpanParamSet, AbstractVector}, 
                      subset::ParamBoxSource)
    typedMap(subset, GetIndex) do param
        locateParam!(params, param)
    end
end

function locateParam!(params::AbstractSpanParamSet, subset::AbstractSpanParamSet)
    unit = typedMap(x->locateParamCore!(params.unit, x), subset.unit, OneToIndex)
    grid = typedMap(x->locateParamCore!(params.grid, x), subset.grid, OneToIndex)
    SpanSetFilter(unit, grid)
end


const MultiSpanData{T} = Union{T, DirectMemory{<:T}, NestedMemory{<:T}}

struct MultiSpanDataCacheBox{T, G<:DirectMemory{<:T}, N<:NestedMemory{<:T}
                             } <: QueryBox{MultiSpanData{T}}
    unit::LRU{UnitParamEgalBox, T}
    grid::LRU{GridParamEgalBox, G}
    nest::LRU{NestParamEgalBox, N}

    function MultiSpanDataCacheBox(::Type{T}=Any; maxSize::Int=100) where {T}
        maxsize = maxSize
        G, N = if isconcretetype(T)
            DirectMemory{T},   NestedMemory{T}
        else
            DirectMemory{<:T}, NestedMemory{<:T}
        end
        new{T, G, N}(LRU{UnitParamEgalBox, T}(; maxsize), 
                     LRU{GridParamEgalBox, G}(; maxsize), 
                     LRU{NestParamEgalBox, N}(; maxsize))
    end
end


getSpanDataSectorKey(cache::MultiSpanDataCacheBox{T1}, param::UnitParam{T2}) where 
                    {T1, T2<:T1} = 
(cache.unit, UnitParamEgalBox(param))

getSpanDataSectorKey(cache::MultiSpanDataCacheBox{T1}, param::GridParam{T2}) where 
                    {T1, T2<:T1} = 
(cache.grid, GridParamEgalBox(param))

getSpanDataSectorKey(cache::MultiSpanDataCacheBox{T1}, param::NestParam{T2}) where 
                    {T1, T2<:T1} = 
(cache.nest, BestParamEgalBox(param))


formatSpanData(::Type{T}, val) where {T} = convert(T, val)

function formatSpanData(::Type{T}, val::AbstractArray) where {T}
    res = getPackedMemory(val)
    res::getPackType(T, getNestedLevel(res|>typeof).level)
end


function cacheParam!(cache::MultiSpanDataCacheBox{T}, param::ParamBox{<:T}) where {T}
    get!(getSpanDataSectorKey(cache, param)...) do
        formatSpanData(T, obtain(param))
    end::getOutputType(param)
end

function cacheParam!(cache::MultiSpanDataCacheBox, params::ParamBoxSource)
    typedMap(params) do param
        cacheParam!(cache, param)
    end
end


struct SpanSetFilter{U<:OneToIndex, G<:OneToIndex} <: Mapper
    scope::FixedSpanIndexSet{U, G}

    function SpanSetFilter(scope::AbstractSpanIndexSet)
        scope = map(scope) do sector
            isempty(sector) ? genBottomMemory() : Memory{OneToIndex}(sector)
        end
        new{(values∘map)(eltype, scope)...}(scope)
    end

    SpanSetFilter() = new{Union{}, Union{}}(genFixedVoidSpanSet())
end

SpanSetFilter(unit::AbstractVector{<:OneToIndex}, grid::AbstractVector{<:OneToIndex}) = 
SpanSetFilter((;unit, grid))

function SpanSetFilter(unitLen::Int, gridLen::Int)
    builder = x->OneToIndex(x)
    unit = typedMap(builder, Base.OneTo(unitLen))
    grid = typedMap(builder, Base.OneTo(gridLen))
    SpanSetFilter(unit, grid)
end

const VoidSetFilter = SpanSetFilter{Union{}, Union{}}

const UnitSetFilter = SpanSetFilter{OneToIndex, Union{}}

const GridSetFilter = SpanSetFilter{Union{}, OneToIndex}

function getSector(::Val{S}, target::AbstractSpanSet, oneToIds::Memory{T}, 
                   finalizer::F=itself) where {S, T<:OneToIndex, F<:Function}
    sector = getfield(target, S)
    iStart = firstindex(sector)
    if T <: Union{}
        genBottomMemory()
    elseif finalizer isa ItsType
        view(sector, map(x->(x.idx + iStart - 1), oneToIds))
    else
        map(oneToIds) do x
            getindex(sector, (x.idx + iStart - 1)) |> finalizer
        end
    end
end

#= Additional Method =#
function getField(obj::AbstractSpanSet, sFilter::SpanSetFilter, 
                  finalizer::F=itself) where {F<:Function}
    unit = getSector(Val(:unit), obj, sFilter.scope.unit, finalizer)
    grid = getSector(Val(:grid), obj, sFilter.scope.grid, finalizer)
    (; unit, grid)
end

getField(::AbstractSpanSet, ::VoidSetFilter) = genFixedVoidSpanSet()

function getField(sFilterPrev::SpanSetFilter, sFilterHere::SpanSetFilter)
    getField(sFilterPrev.scope, sFilterHere) |> SpanSetFilter
end

getField(::SpanSetFilter, ::VoidSetFilter) = SpanSetFilter()


const NamedFilter = Union{SpanSetFilter, ChainMapper}

struct TaggedSpanSetFilter{F<:NamedFilter} <: Mapper
    scope::F
    tag::Identifier
end

TaggedSpanSetFilter(scope::NamedFilter, paramSet::AbstractSpanParamSet) = 
TaggedSpanSetFilter(scope, Identifier(paramSet))

TaggedSpanSetFilter() = TaggedSpanSetFilter(SpanSetFilter(), Identifier(nothing))

#= Additional Method =#
getField(obj::AbstractSpanSet, tsFilter::TaggedSpanSetFilter, 
         finalizer::F=itself) where {F<:Function} = 
getField(obj, tsFilter.scope, finalizer)


abstract type AbstractParamFunc <: CompositeFunction end


const TypedParamFunc{T, F<:AbstractParamFunc} = ReturnTyped{T, F}

getParamInputType(::AbstractParamFunc) = SpanInput()


struct InputConverter{F<:Function} <: AbstractParamFunc
    core::ParamFreeFunc{F}
end

InputConverter(f::Function) = (InputConverter∘ParamFreeFunc)(f)

(f::InputConverter)(input, ::AbstractSpanValueSet) = f.core(input)

InputConverter(f::InputConverter) = itself(f)

getOutputType(::Type{InputConverter{F}}) where {F<:Function} = getOutputType(F)


struct ParamFormatter{F<:NamedFilter} <: AbstractParamFunc
    core::ParamFreeFunc{TaggedSpanSetFilter{F}}
end

ParamFormatter(f::TaggedSpanSetFilter) = (ParamFormatter∘ParamFreeFunc)(f)

(f::ParamFormatter)(::Any, params::AbstractSpanValueSet) = f.core(params)

ParamFormatter(f::ParamFormatter) = itself(f)

getOutputType(::Type{ParamFormatter{F}}) where {F<:NamedFilter} = getOutputType(F)


struct ParamBindFunc{F<:Function, C1<:UnitParam, C2<:GridParam} <: AbstractParamFunc
    core::F
    unit::Memory{C1}
    grid::Memory{C2}
end

function (f::ParamBindFunc{F})(input, valSet::AbstractSpanValueSet) where {F<:Function}
    for field in (:unit, :grid)
        foreach(getfield(f, field), getfield(valSet, field)) do p, v
            setVal!(p, v)
        end
    end

    f.core(input)
end

getOutputType(::Type{<:ParamBindFunc{F}}) where {F<:Function} = getOutputType(F)


const ParamFunctionChain = FunctionChainUnion{AbstractParamFunc}

struct ParamCombiner{B<:Function, E<:ParamFunctionChain} <: AbstractParamFunc
    binder::B
    encode::E

    function ParamCombiner(binder::B, encode::E) where {B<:Function, 
                           E<:GeneralTupleUnion{ NonEmptyTuple{AbstractParamFunc} }}
        new{B, E}(binder, encode)
    end

    function ParamCombiner(binder::B, encode::E) where 
                          {B<:Function, E<:CustomMemory{<:AbstractParamFunc}}
        checkEmptiness(encode, :encode)
        new{B, E}(binder, encode)
    end
end

ParamCombiner(binder::Function, encode::AbstractVector{<:AbstractParamFunc}) = 
ParamCombiner(binder, genMemory(encode))

function (f::ParamCombiner{B})(input, params::AbstractSpanValueSet) where {B<:Function}
    mapreduce(f.binder, f.encode) do encoder
        encoder(input, params)
    end
end

getOutputType(::Type{<:ParamCombiner{B}}) where {B<:Function} = getOutputType(B)


const ContextParamFunc{B<:Function, E<:Function, F<:NamedFilter} = 
      ParamCombiner{B, Tuple{ InputConverter{E}, ParamFormatter{F} }}

function ContextParamFunc(binder::Function, converter::Function, 
                          formatter::TaggedSpanSetFilter)
    ParamCombiner(binder, ( InputConverter(converter), ParamFormatter(formatter) ))
end

function ContextParamFunc(binder::Function, formatter::TaggedSpanSetFilter)
    ContextParamFunc(binder, itself, formatter)
end


const ParamFilterApply{B<:Function, E<:SpanSetFilter} = ContextParamFunc{B, ItsType, E}

# Specialized method due to the lack of compiler optimization
const ParamFreeApply{B<:Function} = ParamFilterApply{B, VoidSetFilter}

function (f::ParamFreeApply{B})(input, ::AbstractSpanValueSet) where {B<:Function}
    f.binder(input, genFixedVoidSpanSet())
end


struct ParamPipeline{E<:ParamFunctionChain} <: AbstractParamFunc
    encode::E

    function ParamPipeline(encode::E) where 
                          {E<:GeneralTupleUnion{ NonEmptyTuple{AbstractParamFunc} }}
        new{E}(encode)
    end

    function ParamPipeline(encode::E) where {E<:CustomMemory{<:AbstractParamFunc}}
        checkEmptiness(encode, :encode)
        new{E}(encode)
    end
end

ParamPipeline(binder::Function, encode::AbstractVector{<:AbstractParamFunc}) = 
ParamPipeline(binder, genMemory(encode))

function (f::ParamPipeline{E})(input, params::AbstractSpanValueSet) where 
                              {E<:ParamFunctionChain}
    for o in f.encode
        input = o(input, params)
    end
    input
end

function getOutputType(::Type{<:ParamPipeline{E}}) where 
                      {E<:GeneralTupleUnion{ NonEmptyTuple{AbstractParamFunc} }}
    getOutputType(E|>fieldtypes|>last)
end


# Specialized method due to the lack of compiler optimization
const InputPipeline{E<:FunctionChainUnion{InputConverter}} = ParamPipeline{E}

function (f::InputPipeline{E})(input, ::AbstractSpanValueSet) where 
                              {E<:FunctionChainUnion{InputConverter}}
    ∘(getfield.(f.encode, :core)...)(input)
end


#= Additional Method =#
getOpacity(::ParamBindFunc) = Opaque()

function getOpacity(f::F) where {F<:AbstractParamFunc}
    if !Base.issingletontype(F)
        fields = fieldnames(T)
        for fieldSym in fields
            field = getField(f, fieldSym)
            if getOpacity(field) isa Opaque
                return Opaque()
            end
        end
    end

    Lucent()
end


# f(input) => fCore(input, param)
function unpackFunc(f::Function)
    if isParamBoxFree(f)
        canDirectlyStore(f) || (f = deepcopy(f))
        InputConverter(f), initializeSpanParamSet(Union{})
    else
        f = deepcopy(f)
        source = getSourceParamSet(f)

        if !(isempty(source.unit) && isempty(source.grid))
            unitPars, gridPars = map(extractMemory, source)
            fCore = ParamBindFunc(f, unitPars, gridPars)
            paramSet = initializeSpanParamSet(unitPars, gridPars)
            fCore, paramSet
        end
    end
end

function unpackFunc!(f::Function, paramSet::AbstractSpanParamSet, 
                     paramSetId::Identifier=Identifier(paramSet))
    fCore, localParamSet = unpackFunc(f)
    if fCore isa InputConverter
        fCore
    else
        idxFilter = locateParam!(paramSet, localParamSet)
        tagFilter = TaggedSpanSetFilter(idxFilter, paramSetId)
        ContextParamFunc(fCore, tagFilter)
    end
end