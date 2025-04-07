export genTensorVar, genMeshParam, genHeapParam, genCellParam, compareParamBox, uniqueParams, 
       dissectParam, setVal!, symOf, obtain, screenLevelOf, 
       setScreenLevel!, inputOf, setScreenLevel, sortParams!

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

# Screen level of primitive ParamBox should always be 1 since it can always be used as input 
# for composite ParamBox.
getScreenLevelOptions(::Type{<:PrimitiveParam}) = (1,)

screenLevelOf(p::PrimitiveParam) = 1

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

    function UnitVar(input::T, symbol::SymOrIndexedSym) where {T}
        checkPrimParamElementalType(T)
        new{T}(input, IndexedSym(symbol))
    end
end

struct GridVar{T, N} <: TensorVar{T, DirectMemory{T, N}}
    input::DirectMemory{T, N}
    symbol::IndexedSym

    function GridVar(input::AbstractArray{T, N}, symbol::SymOrIndexedSym) where {T, N}
        N < 1 && throw(AssertionError("`N` must be larger than zero."))
        checkPrimParamElementalType(T)
        input = decoupledCopy(input)
        new{T, N}(input, IndexedSym(symbol))
    end
end

genTensorVar(input::Any, symbol::SymOrIndexedSym) = UnitVar(input, symbol)

genTensorVar(input::AbstractArray, symbol::SymOrIndexedSym) = GridVar(input, symbol)

genTensorVar(input::AbtArray0D, symbol::SymOrIndexedSym) = 
genTensorVar(first(input), symbol)


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

(f::TypedReduce{T})(arg, args...) where {T} = f.f(arg, args...)::T

(f::TypedReduce{<:AbstractArray{T, N}})(arg, args...) where {T, N} = 
f.f(arg, args...)::getPackType(AbstractArray{T, N})


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
    res = f.f(arg, args...)::getPackType(AbstractArray{T, N})
    f.shape(res)
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
function setScreenLevel!(p::P, level::Int) where {P<:AdaptableParam}
    checkScreenLevel(level, getScreenLevelOptions(P))
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


setScreenLevel(p::CellParam, level::Int) = 
setScreenLevel!(genCellParam(p), level)

setScreenLevel(p::MeshParam, level::Int) = 
setScreenLevel!(genMeshParam(p), level)


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
        ShapedMemory{innerType, N}
    else
        ShapedMemory{<:innerType, N}
    end
end


function hasCycleCore!(::Set{BlackBox}, ::Set{BlackBox}, 
                       edge::Pair{<:PrimitiveParam, <:Union{Nothing, ParamBox}}, 
                       ::Bool, finalizer::F=itself) where {F<:Function}
    finalizer(edge)
    (false, edge.first)
end

function hasCycleCore!(localTrace::Set{BlackBox}, history::Set{BlackBox}, 
                       edge::Pair{<:CompositeParam, <:Union{Nothing, ParamBox}}, 
                       strictMode::Bool, finalizer::F=itself) where {F<:Function}
    here = edge.first

    if strictMode || screenLevelOf(here) == 0
        key = BlackBox(here)
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
    localTrace = Set{BlackBox}()
    parHistory = Set{BlackBox}()
    bl, lastP = hasCycleCore!(localTrace, parHistory, param=>nothing, strictMode, finalizer)
    catcher[] = lastP
    bl
end


function obtainCore!(cache::LRU{BlackBox, <:Any}, param::PrimitiveParam)
    input = param.input
    get!(cache, BlackBox(param), decoupledCopy(input))::typeof(input)
end

function obtainCore!(cache::LRU{BlackBox, <:Any}, param::ShapedParam)
    map(param.input) do p
        obtainCore!(cache, p)
    end::getOutputType(param)
end

function obtainCore!(cache::LRU{BlackBox, <:Any}, param::AdaptableParam)
    key = BlackBox(param)
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
    cache = LRU{BlackBox, Any}(maxsize=100)
    obtainCore!(cache, param)
end

obtain(param::PrimitiveParam) = decoupledCopy(param.input)

function obtain(params::ParamBoxAbtArr)
    if isempty(params)
        Union{}[]
    else
        cache = LRU{BlackBox, Any}(maxsize=min( 500, 100length(params) ))
        map(params) do param
            checkParamCycle(param)
            obtainCore!(cache, param)
        end
    end
end

(pn::ParamBox)() = obtain(pn)


function setVal!(par::PrimitiveParam, val)
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


function uniqueParams(source::ParamBoxAbtArr; onlyPrimitive::Bool=true)
    unique!(BlackBox, source)
    onlyPrimitive && filter!(isPrimitiveInput, source)
    source
end

function uniqueParams(source; onlyPrimitive::Bool=true)
    res = map(last, getFieldParams(source))
    uniqueParams(res; onlyPrimitive)
end

function getFieldParams(source)
    paramPairs = Pair{ChainedAccess, ParamBox}[]
    getFieldParamsCore!(paramPairs, source, ChainedAccess())
    paramPairs
end

function isParamBoxFree(source::T) where {T}
    Base.issingletontype(T) || (getFieldParams(source) |> isempty)
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
    elseif isstructtype(T) && !(Base.issingletontype(T))
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

uniqueParams(ps::ParamBoxAbtArr) = markUnique(ps, compareFunction=compareParamBox)[end]

struct ReduceShift{T, F} <: TypedTensorFunc{T, 0}
    apply::TypedReduce{T, F}
    shift::T
end

(f::ReduceShift)(args...) = unitOp(+, f.apply(args...), f.shift)

struct ExpandShift{T, N, F} <: TypedTensorFunc{T, N}
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

function (f::ParamBoxClassifier)(edge::Pair{<:ParamBox, <:Union{Nothing, ParamBox}})
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

dissectParam(params::ParamBoxAbtArr) = (dissectParamCore∘unique)(BlackBox, params)
dissectParam(source::Any) = (dissectParamCore∘uniqueParams)(source, onlyPrimitive=false)
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

const FixedSpanValueSet{T1, T2<:AbstractArray{T1}} = 
      AbstractSpanValueSet{Memory{T1}, Memory{T2}}

const AbstractSpanIndexSet{U<:AbstractVector{OneToIndex}, G<:AbstractVector{OneToIndex}} = 
      AbstractSpanSet{U, G}

const FixedSpanIndexSet = AbstractSpanIndexSet{Memory{OneToIndex}, Memory{OneToIndex}}

const AbstractSpanParamSet{U<:AbstractVector{<:UnitParam}, G<:AbstractVector{<:GridParam}} = 
      AbstractSpanSet{U, G}

const TypedSpanParamSet{T1<:UnitParam, T2<:GridParam} = 
      AbstractSpanParamSet{Vector{T1}, Vector{T2}}


initializeSpanParamSet() = (unit=UnitParam[], grid=GridParam[])

initializeSpanParamSet(::Type{T}) where {T} = (unit=UnitParam{T}[], grid=GridParam{T}[])

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

function locateParam!(paramSet::AbstractSpanParamSet, target::UnitParam)
    GetIndex{UnitIndex}(locateParamCore!(paramSet.unit, target))
end

function locateParam!(paramSet::AbstractSpanParamSet, target::GridParam)
    GetIndex{GridIndex}(locateParamCore!(paramSet.grid, target))
end

const ParamBoxSource = Union{ParamBoxAbtArr, Tuple{Vararg{ParamBox}}, AbstractSpanParamSet}

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


const MultiSpanData{T} = Union{T, DirectMemory{T}, NestedMemory{T}}

struct MultiSpanDataCacheBox{T} <: QueryBox{MultiSpanData{T}}
    unit::Dict{Identifier, T}
    grid::Dict{Identifier, DirectMemory{T}}
    nest::Dict{Identifier, NestedMemory{T}}
end

MultiSpanDataCacheBox(::Type{T}) where {T} = 
MultiSpanDataCacheBox(Dict{Identifier, T}(), 
                      Dict{Identifier, DirectMemory{T}}(), 
                      Dict{Identifier, NestedMemory{T}}())


getSpanDataSector(cache::MultiSpanDataCacheBox{T1}, 
                  ::UnitParam{T2}) where {T1, T2<:T1} = cache.unit

getSpanDataSector(cache::MultiSpanDataCacheBox{T1}, 
                  ::GridParam{T2}) where {T1, T2<:T1} = cache.grid

getSpanDataSector(cache::MultiSpanDataCacheBox{T1}, 
                  ::NestParam{T2}) where {T1, T2<:T1} = cache.nest


formatSpanData(::Type{T}, val) where {T} = T(val)

function formatSpanData(::Type{T}, val::AbstractArray) where {T}
    res = getPackedMemory(val)
    res::getPackType(T, getNestedLevel(res|>typeof).level)
end


function cacheParam!(cache::MultiSpanDataCacheBox{T}, param::ParamBox{<:T}) where {T}
    get!(getSpanDataSector(cache, param), Identifier(param)) do
        formatSpanData(T, obtain(param))
    end
end

# getParamCacheType(::Type{<:UnitParam{T}}) where {T} = T
# getParamCacheType(::Type{<:GridParam{T}}) where {T} = DirectMemory{T}
# getParamCacheType(::Type{<:NestParam{T}}) where {T} = NestedMemory{T}
# getParamCacheType(::Type{<:ParamBox{T} }) where {T} = Union{T, PackedMemory{T}}

# getParamCacheType(::Type{<:ParamBox }) = Any
# getParamCacheType(::Type{<:GridParam}) = DirectMemory
# getParamCacheType(::Type{<:NestParam}) = NestedMemory

function cacheParam!(cache::MultiSpanDataCacheBox, params::ParamBoxSource)
    # defaultEltype = params isa AbstractArray ? getParamCacheType(params|>eltype) : Union{}
    typedMap(params) do param
        cacheParam!(cache, param)
    end
end


# Methods for parameterized functions
function evalFunc(func::F, input) where {F<:Function}
    fCore, pSet, _ = unpackFunc(func)
    evalFunc(fCore, pSet, input)
end

function evalFunc(fCore::F, pSet, input) where {F<:Function}
    fCore(input, map(obtain, pSet))
end

#! Possibly adding memoization in the future to generate/use the same param set to avoid 
#! bloating `Quiqbox.IdentifierCache` and prevent repeated computation.
unpackFunc(f::F) where {F<:Function} = unpackFunc!(f, initializeSpanParamSet())

unpackFunc!(f::F, paramSet::AbstractSpanParamSet) where {F<:Function} = 
unpackFunc!(SelectTrait{ParameterizationStyle}()(f), f, paramSet)

unpackFunc!(::TypedParamFunc, f::Function, paramSet::AbstractSpanParamSet) = 
unpackParamFunc!(f, paramSet)

unpackFunc!(::GenericFunction, f::Function, paramSet::AbstractSpanParamSet) = 
unpackTypedFunc!(f, paramSet)


struct SpanSetFilter <: Filter
    scope::FixedSpanIndexSet
end

function SpanSetFilter()
    SpanSetFilter(( unit=Memory{OneToIndex}(undef, 0), grid=Memory{OneToIndex}(undef, 0) ))
end

function SpanSetFilter(unitLen::Int, gridLen::Int)
    unitIds = map(OneToIndex, Base.OneTo(unitLen))
    gridIds = map(OneToIndex, Base.OneTo(gridLen))
    SpanSetFilter(( unit=Memory{OneToIndex}(unitIds), grid=Memory{OneToIndex}(gridIds) ))
end

function SpanSetFilter(scope::AbstractSpanIndexSet)
    SpanSetFilter(map(getMemory, scope))
end

function SpanSetFilter(unit::AbstractVector{OneToIndex}, grid::AbstractVector{OneToIndex})
    SpanSetFilter((;unit, grid))
end

function getField(paramSet::AbstractSpanSet, sFilter::SpanSetFilter)
    firstIds = map(firstindex, paramSet)
    map(paramSet, firstIds, sFilter.scope) do sector, i, oneToIds
        view(sector, map(x->(x.idx + i - 1), oneToIds))
    end
end

function getField(sFilterPrev::SpanSetFilter, sFilterHere::SpanSetFilter)
    unit = map(sFilterHere.scope.unit) do idx
        getField(sFilterPrev.scope.unit, idx)
    end

    grid = map(sFilterHere.scope.grid) do idx
        getField(sFilterPrev.scope.grid, idx)
    end
    SpanSetFilter((;unit, grid))
end


# const ParamPointerDict = AbstractDict{<:ChainedAccess, <:GetIndex}
# const ParamPointerPairs = AbstractVector{<:Pair{<:ChainedAccess, <:GetIndex}}
# const FieldParamPairSet{P1<:ParamPointerPairs, P2<:ParamPointerPairs} = 
#       AbstractSpanSet{P1, P2}

# #! DirectFieldParamPointer
# struct MixedFieldParamPointer{R1<:ParamPointerDict, R2<:ParamPointerDict
#                               } <: FieldParamPointer
#     unit::R1
#     grid::R2
#     tag::Identifier
# end

# function MixedFieldParamPointer(paramPairs::FieldParamPairSet, tag::Identifier)
#     coreDict = map(buildDict, paramPairs)
#     MixedFieldParamPointer(coreDict.unit, coreDict.grid, tag)
# end


struct TaggedSpanSetFilter <: Mapper
    scope::SpanSetFilter
    tag::Identifier
end

#= Additional Method =#
getField(obj, tsFilter::TaggedSpanSetFilter) = getField(obj, tsFilter.scope)


abstract type AbstractParamFunc <: CompositeFunction end

struct ParamFreeFunc{F<:Function} <: AbstractParamFunc
    f::F

    function ParamFreeFunc(f::F) where {F<:Function}
        if !(Base.issingletontype(F) || isParamBoxFree(f))
            throw(AssertionError("`f` should not contain any `$ParamBox`."))
        end
        checkArgQuantity(f, 1)
        new{F}(f)
    end
end

ParamFreeFunc(f::ParamFreeFunc) = itself(f)

(f::ParamFreeFunc)(input, ::AbstractSpanValueSet) = f.f(input)


struct ParamBindFunc{F<:Function, C1<:UnitParam, C2<:GridParam} <: AbstractParamFunc
    core::F
    unit::Memory{C1}
    grid::Memory{C2}
end

function (f::ParamBindFunc)(input, valSet::AbstractSpanValueSet)
    for field in (:unit, :grid)
        foreach(getfield(f, field), getfield(valSet, field)) do p, v
            setVal!(p, v)
        end
    end

    f.core(input)
end


const ParamTupleEncoder{T<:AbstractParamFunc, D, F<:Function} = 
      ReturnTyped{T, TupleHeader{D, F}}


const ParamFuncSequence = Union{ NonEmptyTuple{AbstractParamFunc}, 
                                 LinearMemory{<:AbstractParamFunc} }

struct ParamCombiner{J<:Function, C<:ParamFuncSequence} <: AbstractParamFunc
    binder::ParamFreeFunc{J}
    encode::C

    function ParamCombiner(binder::J, encode::C) where 
                          {J<:Function, C<:NonEmptyTuple{ParamFuncSequence}}
        new{J, C}(ParamFreeFunc(binder), encode)
    end

    function ParamCombiner(binder::J, encode::C) where 
                          {J<:Function, C<:LinearMemory{<:ParamFuncSequence}}
        checkEmptiness(encode, :encode)
        new{J, C}(ParamFreeFunc(binder), encode)
    end
end

ParamCombiner(binder::Function, encode::AbstractVector{<:ParamFuncSequence}) = 
ParamCombiner(binder, getMemory(encode))

(f::ParamCombiner)(input, params::AbstractSpanValueSet) = 
mapreduce(o->o(input, params), f.binder, f.encode)

const ParamExtender{C<:LinearMemory{<:AbstractParamFunc}} = ParamCombiner{typeof(vcat), C}

(f::ParamExtender)(input, params::AbstractSpanValueSet) = 
map(o->o(input, params), f.encode)


struct ParamPipeline{C<:ParamFuncSequence} <: AbstractParamFunc
    encode::C

    function ParamPipeline(encode::C) where {C<:NonEmptyTuple{ParamFuncSequence}}
        new{C}(encode)
    end

    function ParamPipeline(encode::C) where {C<:LinearMemory{<:ParamFuncSequence}}
        checkEmptiness(encode, :encode)
        new{C}(encode)
    end
end

ParamPipeline(binder::Function, encode::AbstractVector{<:ParamFuncSequence}) = 
ParamPipeline(binder, getMemory(encode))

function (f::ParamPipeline)(input, params::AbstractSpanValueSet)
    for o in f.encode
        input = o(input, params)
    end
    input
end

#= Additional Method =#
getOpacity(::ParamFreeFunc) = Lucent()

getOpacity(::ParamBindFunc) = Opaque()


# f(input) => fCore(input, param)
function unpackFunc(f::Function)
    checkArgQuantity(f, 1)

    if !isLucent(f)
        f = deepcopy(f)
        source = getSourceParamSet(f)

        if !(isempty(source.unit) && isempty(source.grid))
            unitPars, gridPars = map(getMemory, source)
            fCore = ParamBindFunc(f, unitPars, gridPars)
            paramSet = initializeSpanParamSet(unitPars, gridPars)
            return (fCore, paramSet)
        end
    end

    ParamFreeFunc(f), initializeSpanParamSet(Union{})
end

function unpackFunc!(f::Function, paramSet::AbstractSpanParamSet, 
                     paramSetId::Identifier=Identifier(paramSet))
    fCore, localParamSet = unpackFunc(f)
    idxFilter = locateParam!(paramSet, localParamSet)
    scope = TaggedSpanSetFilter(idxFilter, paramSetId)
    EncodeParamApply(fCore, scope), paramSet
end