export genTensorVar, genMeshParam, genHeapParam, genCellParam, compareParamBox, uniqueParams, 
       dissectParam, setVal!, symOf, obtain, screenLevelOf, setScreenLevel!, inputOf, 
       sortParams!, initializeSpanParamSet

const SymOrIndexedSym = Union{Symbol, IndexedSym}

struct Primitive end
struct Composite end
const StateType = Union{Primitive, Composite}

const Span{T} = Union{T, DirectMemory{T}}
const Pack{T} = Union{T, PackedMemory{T}} #> `Span{T} <: Pack{T}` for every `T`

abstract type ParamBox{T, E<:Pack{T}, S<:StateType} <: StateBox{E} end

const PrimitiveParam{T, E<:Span{T}} = ParamBox{T, E, Primitive}
const CompositeParam{T, E<:Pack{T}} = ParamBox{T, E, Composite}

const UnitParam{T, S<:StateType} = ParamBox{T, T, S}
const SpanParam{T, E<:Span{T}, S<:StateType} = ParamBox{T, E, S}
const GridParam{T, N, S<:StateType} = SpanParam{T, DirectMemory{T, N}, S}
const NestParam{T, E<:Pack{T}, N, S<:StateType} = ParamBox{T, PackedMemory{T, E, N}, S}

abstract type CellParam{T, E<:Pack{T}} <: CompositeParam{T, E} end
abstract type MeshParam{T, E<:Pack{T}, N} <: NestParam{T, E, N, Composite} end
abstract type HeapParam{T, E<:Pack{T}, N} <: NestParam{T, E, N, Composite} end

const TensorialParam{T, S<:StateType} = Union{UnitParam{T, S}, GridParam{T, <:Any, S}}
const AdaptableParam{T, E<:Pack{T}} = Union{CellParam{T, E}, MeshParam{T, E}}
const ReducibleParam{T, E<:Pack{T}, S<:StateType} = 
      Union{ParamBox{T, E, S}, NestParam{T, E, <:Any, S}}
#> Tuple of `ParamBox` with the same nested level
const NestFixedParIn{T, E<:Pack{T}} = TriTupleUnion{ParamBox{T, E}}
#> Tuple of `ParamBox` with nested level differences no more than 1
const CoreFixedParIn{T, E<:Pack{T}} = TriTupleUnion{ReducibleParam{T, E}}

const UnitOrVal{T} = Union{UnitParam{T}, T}
const UnitOrValVec{T} = AbstractVector{<:UnitOrVal{T}}

const ParamBoxAbtArr{P<:ParamBox, N} = AbstractArray{P, N}
const NamedParamTuple{S, N, P<:NTuple{N, ParamBox}} = NamedTuple{S, P}
const NamedSpanParamTuple{S, N, P<:NTuple{N, SpanParam}} = NamedParamTuple{S, N, P}
const DirectParamSource = Union{ParamBoxAbtArr, Tuple{Vararg{ParamBox}}, NamedParamTuple}

isOffsetEnabled(::ParamBox) = false

# function checkScreenLevel(p::ParamBox)
#     checkScreenLevel(screenLevelOf(p)::Int, getScreenLevelOptions(p|>typeof))
#     nothing
# end

checkScreenLevel(sl::Int, levels::NonEmptyTuple{Int}) = 
checkIntLevelMismatch(sl, levels, "screen")

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


mutable struct UnitVar{T} <: PrimitiveParam{T, T}
    @atomic input::T
    const symbol::IndexedSym
    @atomic screen::Bool

    function UnitVar(input::T, symbol::SymOrIndexedSym, screen::Bool=false) where {T}
        checkPrimParamElementalType(T)
        new{T}(input, IndexedSym(symbol), screen)
    end
end

mutable struct GridVar{T, N} <: PrimitiveParam{T, DirectMemory{T, N}}
    const input::DirectMemory{T, N}
    const symbol::IndexedSym
    @atomic screen::Bool

    function GridVar(input::AbstractArray{T, N}, symbol::SymOrIndexedSym, 
                     screen::Bool=false) where {T, N}
        N < 1 && throw(AssertionError("`N` must be larger than zero."))
        checkPrimParamElementalType(T)
        input = PackedMemory(input)
        new{T, N}(input, IndexedSym(symbol), screen)
    end
end

const TensorVar{T} = Union{UnitVar{T}, GridVar{T}}

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

function (f::TypedReduce{T})(arg, args...) where {T}
    getLazyConverter(f.f, getPackType(T))(arg, args...)
end

function getOutputType(::Type{TypedReduce{T, F}}) where {T, F<:Function}
    type = getOutputType(F)
    typeBound = getPackType(T)
    ifelse(type <: typeBound, type, typeBound)
end


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
                f(args...)::getPackType(AbstractArray{T}) |> size
            end
        else
            shape
        end
        shapeFinal = TruncateReshape(shapeTemp; truncate)

        fCore = (f isa TypedExpand) ? f.f : f

        new{T, length(shapeFinal.axis), typeof(fCore)}(fCore, T, shapeFinal)
    end

    function TypedExpand(f::Function, args::NonEmptyTuple{Any})
        output = f(args...)::AbstractArray
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
    rawRes = getLazyConverter(f.f, getPackType(AbstractArray{T, N}))(arg, args...)
    f.shape(rawRes)
end

function getOutputType(::Type{TypedExpand{T, N, F}}) where {T, N, F<:Function}
    type = getOutputType(F)
    typeBound = getPackType(AbstractArray{T, N})
    ifelse(type <: typeBound, type, typeBound)
end

#= Additional Method =#
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
# ParamBoxes with nested data cannot have screen level higher than 0
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
    idx = firstindex(b) - 1
    map(a) do ele
        idx += 1
        unitOp(f, ele, b[idx])
    end
end


getTensorOutputTypeBound(::TypedReduce{E}) where {E} = E
getTensorOutputTypeBound(::TypedExpand{E}) where {E} = AbstractArray{E}

getTensorOutputShape(f::TypedExpand,    ::CoreFixedParIn) = f.shape.axis
getTensorOutputShape( ::TypedReduce{E}, ::CoreFixedParIn) where {E} = ()
function getTensorOutputShape(f::TypedReduce{E}, input::CoreFixedParIn) where 
                             {E<:AbstractArray}
    fCore = f.f
    if fCore isa Union{TypedExpand, TypedReduce}
        getTensorOutputShape(fCore, input)
    else
        size(f(obtain.(input)...))
    end
end

function formatOffset(lambda::TypedTensorFunc{<:Pack}, ::Missing, input::CoreFixedParIn)
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

function formatOffset(lambda::TypedTensorFunc{<:Pack}, offset, input::CoreFixedParIn)
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


function formatTensorFunc(f::Function, ::Type{TypedReduce}, input::CoreFixedParIn)
    if f isa TypedReduce{<:Pack}
        f
    else
        type = getOutputType(f)
        if !isconcretetype(type)
            type = typeof(f(obtain.(input)...))
        end
        TypedReduce(f, genPackMemoryType(type))
    end
end

function formatTensorFunc(f::Function, ::Type{TypedExpand}, input::CoreFixedParIn)
    if f isa TypedExpand{<:Pack}
        f
    else
        TypedExpand(f, obtain.(input))
    end
end


# ([[x]], [x]) -> {[x]}; (x, x) -> {x}
function getCellOutputLevels(::CoreFixedParIn{T, E}) where {T, E<:Pack{T}}
    (getNestedLevel(E).level,)
end
# ([x], [x]) -> {x, [x]}
function getCellOutputLevels(::NestFixedParIn{T, E}) where {T, E<:PackedMemory{T}}
    level = getNestedLevel(E).level
    (level-1, level)
end

function checkReduceParamLevel(lambda::TypedReduce{<:Pack}, input::CoreFixedParIn)
    targetLevelRange = getCellOutputLevels(input)
    actualLevel = getNestedLevel(lambda.type).level
    checkIntLevelMismatch(actualLevel, targetLevelRange, "reduce-output nested")
end

# ([[x]], [x]) -> {[[x]]}; ([x], [x]) -> {[[x]]}
function checkExpandParamLevel(lambda::TypedExpand{<:Pack}, ::CoreFixedParIn{T, E}) where 
                              {T, E<:Pack{T}}
    targetEleLevel = getNestedLevel(E).level
    actualEleLevel = getNestedLevel(lambda.type).level
    checkIntLevelMismatch(actualEleLevel+1, (targetEleLevel+1,), "expand-output nested")
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
        checkReduceParamLevel(lambda, input)
        offsetTuple = formatOffset(lambda, offset, input)
        if isempty(offsetTuple)
            new{T, E, F, I}(lambda, input, sym, screen)
        else
            new{T, E, F, I}(lambda, input, sym, screen, first(offsetTuple))
        end
    end
end

const ScreenParam{T, E<:Pack{T}, P<:ParamBox{T, E}} = ReduceParam{T, E, ItsType, Tuple{P}}

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
        checkExpandParamLevel(lambda, input)
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

@generated function isScreenLevelChangeable(::Type{T}) where {T<:ParamBox}
    minLevel, maxLevel = extrema( getScreenLevelOptions(T) )
    res = (maxLevel - minLevel) > 0
    return :($res)
end

function isOffsetEnabled(pb::T) where {T<:AdaptableParam}
    isScreenLevelChangeable(T) && maximum( getScreenLevelOptions(T) ) > 0 && 
    isdefined(pb, :offset) # Only for safety
end


function indexParam(pb::ShapedParam, oneToIdx::Int, sym::MissingOr{Symbol}=missing)
    checkPositivity(oneToIdx)
    entry = pb.input[begin+oneToIdx-1]
    if ismissing(sym) || sym==symOf(entry)
        entry
    elseif entry isa MeshParam
        genMeshParam(entry, sym)
    else
        genCellParam(entry, sym)
    end
end

function indexParam(pb::SpanParam, oneToIdx::Int, sym::MissingOr{Symbol}=missing)
    checkPositivity(oneToIdx)
    if oneToIdx > (getOutputSize(pb) |> prod)
        throw(BoundsError(pb, oneToIdx))
    end
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


const ParamDataCache{T} = LRU{ParamEgalBox, T}

initializeParamDataCache(maxSize::Int=100, ::Type{T}=Any) where {T} = 
ParamDataCache{T}(maxsize=maxSize)


function cacheParamCore!(cache::ParamDataCache, param::ParamBox, 
                         evaluator::F=Base.Fix1(obtainCore!, cache)) where {F<:Function}
    get!(cache, ParamEgalBox(param)) do
        evaluator(param)
    end::getOutputType(param)
end

function cacheParam!(cache::ParamDataCache, param::ParamBox)
    cacheParamCore!(cache, param) |> decoupledCopy
end


function obtainCore!(cache::ParamDataCache, param::PrimitiveParam)
    input = param.input
    cacheParamCore!(cache, param, Storage(input))::typeof(input)
end

function obtainCore!(cache::ParamDataCache, param::ShapedParam)
    map(param.input) do p
        obtainCore!(cache, p)
    end::getOutputType(param)
end

function obtainCore!(cache::ParamDataCache, param::AdaptableParam)
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
        param.offset
    else
        cache = initializeParamDataCache()
        obtainCore!(cache, param)
    end |> decoupledCopy
end

obtain(param::PrimitiveParam) = decoupledCopy(param.input)

function obtain(params::ParamBoxAbtArr)
    if isempty(params)
        similar(params, Union{})
    else
        eleParamType = eltype(params)
        outputType = eleParamType <: PrimitiveParam ? getOutputType(eleParamType) : Any
        cache = initializeParamDataCache(min( 500, 100length(params) ), outputType)
        map(params) do param
            checkParamCycle(param)
            obtainCore!(cache, param) |> decoupledCopy
        end
    end
end

(pn::ParamBox)() = obtain(pn)


function setVal!(par::TensorVar, val)
    if !isPrimitiveInput(par)
        throw(AssertionError("`isPrimitiveInput(par)` must return `true`."))
    end
    if par isa UnitVar
        @atomic par.input = val
    else
        safelySetVal!(par.input, val)
    end
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


"""

    noStoredParam(source) -> Bool

Detect if `source` has no reachable `$ParamBox` by reflection-type functions, `getfield`and 
`getindex`. It returns `true` if `getFieldParams(source)` returns an empty collection. It 
is still possible for `noStoredParam` to return `true` if `source` is a generic 
function that indirectly references global variables being/storing `$ParamBox`.
"""
function noStoredParam(source::T) where {T}
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


"""

    ParamFreeFunc{F<:Function} <: CompositeFunction

A direct wrapper `struct` for `f::F` that does not have reachable `$ParamBox` through 
reflection functions.

≡≡≡ Field(s) ≡≡≡

`f::F`: Stored function. `$noStoredParam(f)` must return `true` when it is used to 
construct a instance of `ParamFreeFunc{F}`.

≡≡≡ Initialization Method(s) ≡≡≡

    ParamFreeFunc(f::F) where {F<:Function} -> ParamFreeFunc{T}
"""
struct ParamFreeFunc{F<:Function} <: CompositeFunction
    f::F

    function ParamFreeFunc(f::F) where {F<:Function}
        if !noStoredParam(f)
            throw(AssertionError("`f` should not contain any reachable `$ParamBox`."))
        end
        new{F}(f)
    end
end

ParamFreeFunc(f::ParamFreeFunc) = itself(f)

(f::ParamFreeFunc{F})(args...) where {F<:Function} = f.f(args...)

getOutputType(::Type{ParamFreeFunc{F}}) where {F<:Function} = getOutputType(F)

#= Additional Method =#
noStoredParam(::ParamFreeFunc) = true


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
    TypedReturn(fCore, getOutputType(pb))
end

function extractTransform(pb::ShapedParam)
    TypedReturn(itself, getOutputType(pb))
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

    for par in pars
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

    if onlyVariable
        for sector in source
            filter!(isPrimitiveInput, sector)
        end
    end

    source
end


function markParam!(param::ParamBox, 
                    indexDict::AbstractDict{Symbol, Int}=Dict{Symbol, Int}())
    sym = symOf(param)
    get!(indexDict, sym, 0)
    param.symbol.index = (indexDict[sym] += 1)
    nothing
end


function getParamOrderLabel(x::ParamBox)
    nl = getNestedLevel(x|>typeof)
    (screenLevelOf(x), nl.level, symbolFrom(x.symbol), objectid(x))
end

function sortParams!(params::AbstractVector{<:ParamBox}; 
                     indexing::Bool=true, by::F=getParamOrderLabel) where {F<:Function}
    sort!(params; by)

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


const OptionalSpanSet{U<:NothingOr{AbstractVector}, G<:NothingOr{AbstractVector}} = 
      @NamedTuple{unit::U, grid::G}

const OptSpanValueSet{U<:NothingOr{AbstractVector}, G<:NothingOr{AbtVecOfAbtArr}} = 
      OptionalSpanSet{U, G}

const OptSpanParamSet{U<:NothingOr{ AbstractVector{<:UnitParam} }, 
                      G<:NothingOr{ AbstractVector{<:GridParam} }} = 
      OptionalSpanSet{U, G}

const TypedVoidSet{U<:NothingOr{AbtBottomVector}, G<:NothingOr{AbtBottomVector}} = 
      OptionalSpanSet{U, G}

const TypedUnitSet{U<:AbstractVector} = OptionalSpanSet{U, <:NothingOr{AbtBottomVector}}

const TypedGridSet{G<:AbstractVector} = OptionalSpanSet{<:NothingOr{AbtBottomVector}, G}

const TypedSpanSet{U<:AbstractVector, G<:AbstractVector} = OptionalSpanSet{U, G}

const TypedUnitParamSet{U<:AbstractVector{<:UnitParam}} = TypedUnitSet{U}

const TypedGridParamSet{G<:AbstractVector{<:GridParam}} = TypedGridSet{G}

const TypedSpanParamSet{U<:AbstractVector{<:UnitParam}, G<:AbstractVector{<:GridParam}} = 
      TypedSpanSet{U, G}

const FixedSpanParamSet{UP<:UnitParam, GP<:GridParam} = 
      TypedSpanParamSet{Memory{UP}, Memory{GP}}

const ParamBoxSource = Union{DirectParamSource, OptSpanParamSet}


initializeFixedSpanSet(::Nothing) = (unit=nothing, grid=nothing)

initializeFixedSpanSet() = (unit=genBottomMemory(), grid=genBottomMemory())


#= Additional Method =#
function obtain(paramSet::OptSpanParamSet)
    map(paramSet) do sector
        (sector === nothing || isempty(sector)) ? nothing : obtain(sector)
    end
end

#= Additional Method =#
function cacheParam!(cache::ParamDataCache, params::ParamBoxSource)
    if params isa ParamBoxAbtArr && isempty(params)
        similar(params, Union{})
    else
        map(params) do param
            cacheParam!(cache, param)
        end
    end
end


struct UnitInput end
struct GridInput end
struct SpanInput end
struct VoidInput end
const MonoPackInput = Union{UnitInput, GridInput}
const OptSpanInput = Union{MonoPackInput, SpanInput, VoidInput}

restrictSpanSet(::UnitInput, s::TypedUnitSet) = (unit=s.unit,)
restrictSpanSet(::GridInput, s::TypedGridSet) = (grid=s.grid,)
restrictSpanSet(::SpanInput, s::TypedSpanSet) = itself(s)
restrictSpanSet(::VoidInput, s::TypedVoidSet) = initializeFixedSpanSet(nothing)

getInputSetType(::TypedUnitSet) = UnitInput
getInputSetType(::TypedGridSet) = GridInput
getInputSetType(::TypedSpanSet) = SpanInput
getInputSetType(::TypedVoidSet) = VoidInput

struct SpanSetCaller{T, A<:OptSpanInput, F<:Function} <: TypedEvaluator{T}
    core::TypedReturn{T, ParamFreeFunc{F}}

    function SpanSetCaller(f::TypedReturn{T}, ::A) where {T, A<:OptSpanInput}
        fInner = f.f
        new{T, A, typeof(fInner)}(TypedReturn(ParamFreeFunc(fInner), T))
    end

    function SpanSetCaller(f::SpanSetCaller{T}, ::A) where {T, A<:OptSpanInput}
        fCore = f.core
        new{T, A, typeof(fCore.f.f)}(fCore)
    end
end

getInputSetType(::SpanSetCaller{T, A}) where {T, A<:OptSpanInput} = A

function evaluateSpanInput(f::SpanSetCaller{T, UnitInput}, sector::AbstractVector) where {T}
    f.core((unit=sector,))
end

function evaluateSpanInput(f::SpanSetCaller{T, GridInput}, sector::AbstractVector) where {T}
    f.core((grid=sector,))
end

function evaluateSpanInput(f::SpanSetCaller, inputSet::OptSpanValueSet)
    formattedInput = restrictSpanSet(getInputSetType(f)(), inputSet)
    f.core(formattedInput)
end

(f::SpanSetCaller)(input::Union{AbstractVector, OptSpanValueSet}) = 
evaluateSpanInput(f, input)

(f::SpanSetCaller{T, VoidInput})() where {T} = 
evaluateSpanInput(f, initializeFixedSpanSet(nothing))

getOutputType(::Type{<:SpanSetCaller{T}}) where {T} = T


@generated function initializeSpanParamSet(::Type{T}=Any) where {T}
    upType = genParametricType(UnitParam, (;T))
    gpType = genParametricType(GridParam, (;T))
    return :( (unit=($upType)[], grid=($gpType)[]) )
end

initializeSpanParamSet(unit::UnitParam) = (unit=genMemory(unit), grid=genBottomMemory())

initializeSpanParamSet(grid::GridParam) = (unit=genBottomMemory(), grid=genMemory(grid))

initializeSpanParamSet(units::AbstractVector{<:UnitParam}, 
                       grids::AbstractVector{<:GridParam}) = (unit=units, grid=grids)


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

function locateParam!(paramSet::TypedUnitParamSet, target::UnitParam)
    GetIndex{UnitIndex}(locateParamCore!(paramSet.unit, target))
end

function locateParam!(paramSet::TypedGridParamSet, target::GridParam)
    GetIndex{GridIndex}(locateParamCore!(paramSet.grid, target))
end

function locateParam!(params::Union{OptSpanParamSet, AbstractVector}, 
                      subset::ParamBoxSource)
    if subset isa AbstractArray && isempty(subset)
        similar(subset, Union{})
    else
        map(subset) do param
            locateParam!(params, param)
        end
    end
end

function locateParam!(params::OptSpanParamSet, subset::OptSpanParamSet)
    units, grids = map(params, subset) do prev, here
        if here === nothing || isempty(here)
            genBottomMemory()
        else
            map(x->locateParamCore!(prev, x), here)
        end
    end
    SpanSetFilter(units, grids)
end


const SpanIndexSet{U<:AbstractVector{<:OneToIndex}, G<:AbstractVector{<:OneToIndex}} = 
      TypedSpanSet{U, G}

struct SpanSetFilter{U<:OneToIndex, G<:OneToIndex} <: Mapper
    scope::OptionalSpanSet{Memory{U}, Memory{G}}

    function SpanSetFilter(scope::SpanIndexSet)
        scope = map(scope) do sector
            isempty(sector) ? genBottomMemory() : Memory{OneToIndex}(sector)
        end
        new{(values∘map)(eltype, scope)...}(scope)
    end

    SpanSetFilter() = new{Union{}, Union{}}(initializeFixedSpanSet())
end

SpanSetFilter(unit::AbstractVector{<:OneToIndex}, grid::AbstractVector{<:OneToIndex}) = 
SpanSetFilter((;unit, grid))

function SpanSetFilter(unitLen::Int, gridLen::Int)
    unit = OneToIndex.(1:unitLen)
    grid = OneToIndex.(1:gridLen)
    SpanSetFilter(unit, grid)
end

const VoidSetFilter = SpanSetFilter{Union{}, Union{}}

const UnitSetFilter = SpanSetFilter{OneToIndex, Union{}}

const GridSetFilter = SpanSetFilter{Union{}, OneToIndex}

const FullSetFilter = SpanSetFilter{OneToIndex, OneToIndex}


getSector(sector::AbstractVector, oneToIds::Memory{OneToIndex}) = 
MemoryLinker(sector, oneToIds)

getSector(::NothingOr{AbstractVector}, ::Memory{Union{}}) = 
genBottomMemory()

function mapSector(sector::AbstractVector, oneToIds::Memory{OneToIndex}, 
                   finalizer::F) where {F<:Function}
    map(oneToIds) do x
        getField(sector, x) |> finalizer
    end
end

mapSector(::NothingOr{AbstractVector}, ::Memory{Union{}}, ::Function) = 
genBottomMemory()

#= Additional Method =#
function getField(target::OptionalSpanSet, sFilter::SpanSetFilter)
    getter = let coreFilter=sFilter.scope
        (sector, sym) -> getSector(sector, getfield(coreFilter, sym))
    end
    map(getter, target, (unit=:unit, grid=:grid))
end

function getField(target::OptionalSpanSet, sFilter::SpanSetFilter, 
                  finalizer::F) where {F<:Function}
    mapper = let coreFilter=sFilter.scope, f=finalizer
        (sector, sym) -> mapSector(sector, getfield(coreFilter, sym), f)
    end
    map(mapper, target, (unit=:unit, grid=:grid))
end

getField(::OptionalSpanSet, ::VoidSetFilter) = initializeFixedSpanSet()

function getField(sFilterPrev::SpanSetFilter, sFilterHere::SpanSetFilter)
    getField(sFilterPrev.scope, sFilterHere) |> SpanSetFilter
end

getField(::SpanSetFilter, ::VoidSetFilter) = SpanSetFilter()

struct TaggedSpanSetFilter{F<:SpanSetFilter} <: Mapper
    scope::F
    tag::Identifier
end

TaggedSpanSetFilter(scope::SpanSetFilter, paramSet::OptSpanParamSet) = 
TaggedSpanSetFilter(scope, Identifier(paramSet))

#= Additional Method =#
getField(obj::OptionalSpanSet, tsFilter::TaggedSpanSetFilter, 
         finalizer::F=itself) where {F<:Function} = 
getField(obj, tsFilter.scope, finalizer)

getOutputType(::Type{TaggedSpanSetFilter{F}}) where {F<:SpanSetFilter} = getOutputType(F)


struct InputConverter{F<:Function} <: AbstractParamFunc
    core::ParamFreeFunc{F}
end

InputConverter(f::F) where {F<:Function} = InputConverter(f|>ParamFreeFunc)

(f::InputConverter)(input, ::OptSpanValueSet) = f.core(input)

InputConverter(f::InputConverter) = itself(f)

getOutputType(::Type{InputConverter{F}}) where {F<:Function} = getOutputType(F)


struct ParamFormatter{F<:Function} <: AbstractParamFunc
    core::ParamFreeFunc{F}
end

ParamFormatter(f::F) where {F<:Function} = ParamFormatter(f|>ParamFreeFunc)

(f::ParamFormatter)(::Any, params::OptSpanValueSet) = f.core(params)

ParamFormatter(f::ParamFormatter) = itself(f)

getOutputType(::Type{ParamFormatter{F}}) where {F<:Function} = getOutputType(F)


struct ParamBindFunc{F<:Function, PU<:UnitParam, PG<:GridParam} <: AbstractParamFunc
    core::F
    unit::Memory{PU}
    grid::Memory{PG}

    function ParamBindFunc(core::F, units::AbstractVector{<:UnitParam}, 
                                    grids::AbstractVector{<:GridParam}) where {F<:Function}
        unitMem = genMemory(units)
        gridMem = genMemory(grids)

        new{F, eltype(unitMem), eltype(gridMem)}(core, unitMem, gridMem)
    end
end

const UnitParamBindFunc{F, PU} = ParamBindFunc{F, PU, Union{}}

const GridParamBindFunc{F, PG} = ParamBindFunc{F, Union{}, PG}

const VoidParamBindFunc{F} = ParamBindFunc{F, Union{}, Union{}}

function (f::UnitParamBindFunc)(input, params::OptSpanValueSet)
    for (p, v) in zip(f.unit, params.unit)
        setVal!(p, v)
    end

    f.core(input)
end

function (f::GridParamBindFunc)(input, params::OptSpanValueSet)
    for (p, v) in zip(f.grid, params.grid)
        setVal!(p, v)
    end

    f.core(input)
end

(f::VoidParamBindFunc)(input, ::OptSpanValueSet) = f.core(input)

function (f::ParamBindFunc)(input, params::OptSpanValueSet)
    for field in (:unit, :grid), (p, v) in zip(getfield(f, field), getfield(params, field))
        setVal!(p, v)
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

function (f::ParamCombiner{B, E})(input::T, params::OptSpanValueSet) where 
                                 {T, B<:Function, E<:ParamFunctionChain}
    fHead, fTail... = f.encode
    res = fHead(input, params)
    for fPart in fTail
        res = f.binder(res, fPart(input, params))
    end
    res
end

getOutputType(::Type{<:ParamCombiner{B}}) where {B<:Function} = getOutputType(B)


const ContextParamFunc{B<:Function, E<:Function, F<:Function} = 
      ParamCombiner{B, Tuple{ InputConverter{E}, ParamFormatter{F} }}

function ContextParamFunc(binder::Function, converter::Function, formatter::Function)
    ParamCombiner(binder, ( InputConverter(converter), ParamFormatter(formatter) ))
end

function ContextParamFunc(binder::Function, formatter::Function)
    ContextParamFunc(binder, itself, formatter)
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

ParamPipeline(encode::AbstractVector{<:AbstractParamFunc}) = 
ParamPipeline(genMemory(encode))

function (f::ParamPipeline{E})(input, params::OptSpanValueSet) where 
                              {E<:ParamFunctionChain}
    fHead, fTail... = f.encode
    res = fHead(input, params)
    for fPart in fTail
        res = fPart(res, params)
    end
    res
end

function getOutputType(::Type{<:ParamPipeline{E}}) where 
                      {E<:GeneralTupleUnion{ NonEmptyTuple{AbstractParamFunc} }}
    getOutputType(E|>fieldtypes|>last)
end


# f(input) => fCore(input, param)
function unpackFunc(f::Function)
    fLocal = deepcopy(f)
    if noStoredParam(fLocal)
        fCore = InputConverter(fLocal)
        paramSet = initializeFixedSpanSet()
    else
        source = getSourceParamSet(fLocal) #> `onlyVariable == true && includeSink == true`
        unitPars, gridPars = source
        fCore = ParamBindFunc(fLocal, unitPars, gridPars)
        paramSet = initializeSpanParamSet(unitPars, gridPars)
    end
    fCore, paramSet
end

function unpackFunc!(f::Function, paramSet::OptSpanParamSet, 
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


function genParamFinisher(param::ReduceParam, screen::TernaryNumber=param.screen; 
                          deepCopyLambda::Bool=false)
    offsetPreset = isOffsetEnabled(param) ? param.offset : missing
    let offset=offsetPreset, f=(deepCopyLambda ? deepcopy(param.lambda) : param.lambda)
        function finishReduceParam(input::CoreFixedParIn)
            ReduceParam(f, input, param.symbol, screen, offset)
        end
    end
end

function genParamFinisher(param::ExpandParam, screen::TernaryNumber=param.screen; 
                          deepCopyLambda::Bool=false)
    offsetPreset = isOffsetEnabled(param) ? param.offset : missing
    let offset=offsetPreset, f=(deepCopyLambda ? deepcopy(param.lambda) : param.lambda)
        function finishExpandParam(input::CoreFixedParIn)
            ExpandParam(f, input, param.symbol, screen, offset)
        end
    end
end


"""

    sever(param::ParamBox, screenSource::Bool=false) -> ParamBox

Eliminate the severable connection(s) in `param`. For `param::SpanParam`, it returns a 
`PrimitiveParam` with an output value same as `obtain(param)` when it is called; for any 
`param` being nested `ParamBox`, it recursively severs every upstream `ParamBox` inside 
`param` that is a `SpanParam`. When `screenSource` is set to `true`, every `PrimitiveParam` 
inside (or is) the returned the `ParamBox` will have `.screen` set to `true`.
"""
function sever(param::SpanParam, screenSource::Bool=false)
    val = obtain(param)
    genTensorVar(val, symOf(param), screenSource)
end

sever(param::ParamBox, screenSource::Bool=false) = severCore(param, screenSource)

function severCore(param::ShapedParam, screenSource::Bool)
    severedInput = map(param.input) do ele
        sever(ele, screenSource)
    end
    ShapedParam(severedInput, param.symbol)
end

function severCore(param::AdaptableParam, screenSource::Bool)
    severedInput = map(param.input) do ele
        sever(ele, screenSource)
    end
    finisher = genParamFinisher(param, deepCopyLambda=true)
    finisher(severedInput)
end