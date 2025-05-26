export transpileParam, ParamGraphCaller, evaluateGraph, functionalize

struct Directed end
struct Undirected end

const Orientation = Union{Directed, Undirected}

struct Static end
struct Active end

const Activity = Union{Static, Active}

abstract type AbstractGraph{O<:Orientation} end

abstract type ComputationGraph{T} <: AbstractGraph{Directed} end

abstract type TransformedGraph{O<:Orientation} <: AbstractGraph{Orientation} end

abstract type GraphVertex end

abstract type TypedVertex{T} <: GraphVertex end

abstract type AccessVertex{T} <: TypedVertex{T} end

abstract type ActionVertex{T} <: TypedVertex{T} end


getOutputType(::T) where {T<:TypedVertex} = getOutputType(T)


struct UnitVertex{T} <: AccessVertex{T}
    value::T
    active::Bool
    marker::Symbol
end

getOutputType(::Type{UnitVertex{T}}) where {T} = T

struct GridVertex{T, N} <: AccessVertex{ShapedMemory{T, N}}
    value::ShapedMemory{T, N}
    active::Bool
    marker::Symbol
end

getOutputType(::Type{GridVertex{T, N}}) where {T, N} = getPackType(ShapedMemory{T, N})

const TensorVertex{T} = Union{UnitVertex{T}, GridVertex{T}}

function genTensorVertex(val::T, active::Bool, marker::Symbol) where {T}
    UnitVertex(val, active, marker)
end

function genTensorVertex(val::AbstractArray{T}, active::Bool, marker::Symbol) where {T}
    GridVertex(ShapedMemory(val), active, marker)
end


isVertexActive(vertex::TensorVertex) = vertex.active

getVertexValue(vertex::TensorVertex) = vertex.value


# HalfEdge: Edge that does not hold complete information unless it's attached to a vertex.
abstract type HalfEdge{O<:Orientation} end

abstract type VertexReceptor <: HalfEdge{Directed} end

@enum VertexEffect::Int8 begin
    UnstableVertex = 0
    VertexAsAccess = 1
    VertexAsAction = 2
end

@enum VertexOutput::Int8 begin
    VertexToUnit = 0
    VertexToGrid = 1
    VertexToNest = 2
    VertexToWhat = 3
end

getVertexOutputLevel(::UnitVar) = VertexToUnit
getVertexOutputLevel(::GridVar) = VertexToGrid
function getVertexOutputLevel(::P) where {P<:CompositeParam}
    level = getNestedLevel(P).level
    if level == 0
        VertexToUnit
    elseif level == 1
        VertexToGrid
    else
        VertexToNest
    end
end


struct VertexTrait
    effect::VertexEffect
    output::VertexOutput

    VertexTrait() = new(UnstableVertex, VertexToWhat)
    VertexTrait(effect::VertexEffect, output::VertexOutput) = new(effect, output)
end


struct TupleReceptor{N} <: VertexReceptor
    trait::VectorMemory{VertexTrait, N}
    index::VectorMemory{OneToIndex, N}

    function TupleReceptor(nInput::Int, defaultTrait::VertexTrait=VertexTrait())
        checkPositivity(nInput)
        trait = VectorMemory(fill(defaultTrait, nInput))
        index = VectorMemory(fill(OneToIndex(), nInput))
        new{nInput}(trait, index)
    end
end

struct ArrayReceptor{N} <: VertexReceptor
    trait::ShapedMemory{VertexTrait, N}
    index::ShapedMemory{OneToIndex, N}

    function ArrayReceptor(sInput::NTuple{N, Int}, 
                           defaultTrait::VertexTrait=VertexTrait()) where {N}
        trait = ShapedMemory(fill(defaultTrait, sInput))
        index = ShapedMemory(fill(OneToIndex(), sInput))
        new{N}(trait, index)
    end
end

const CallVertexReceptor{N} = Union{TupleReceptor{N}, ArrayReceptor{N}}


struct CallVertex{T, I<:CallVertexReceptor, F<:Function} <: ActionVertex{T}
    apply::TypedReturn{T, F}
    marker::Symbol
    receptor::I # Could potentially have repeated input indices
end

getOutputType(::Type{<:CallVertex{T}}) where {T} = T

const ParamBoxVertexDict = IdDict{ParamBox, Pair{VertexTrait, OneToIndex}}


function updateVertexReceptor!(receptor::CallVertexReceptor, param::CompositeParam, 
                               dict::ParamBoxVertexDict)
    tIds = receptor.trait
    iIds = receptor.index
    for (i, j, p) in zip(eachindex(tIds), eachindex(iIds), param.input)
        trait, index = dict[p]
        tIds[i] = trait
        iIds[j] = index
    end
    nothing
end


function initializeReceptor(param::AdaptableParam, 
                            reference::NothingOr{ParamBoxVertexDict}=nothing)
    receptor = TupleReceptor(length(param.input))
    reference === nothing || updateVertexReceptor!(receptor, param, reference)
    receptor
end

function initializeReceptor(param::ShapedParam, 
                            reference::NothingOr{ParamBoxVertexDict}=nothing)
    receptor = ArrayReceptor(size(param.input))
    reference === nothing || updateVertexReceptor!(receptor, param, reference)
    receptor
end


function genParamVertex(param::CompositeParam, 
                        reference::NothingOr{ParamBoxVertexDict}=nothing)
    sl = screenLevelOf(param)
    sym = symbolFrom(param.symbol)
    if sl > 0
        genTensorVertex(param.offset, sl<2, sym)
    else
        apply = extractTransform(param)
        receptor = initializeReceptor(param, reference)
        CallVertex(apply, sym, receptor)
    end
end

function genParamVertex(param::PrimitiveParam)
    genTensorVertex(param.input, screenLevelOf(param)<2, symbolFrom(param.symbol))
end


function formVertexSectors(unitInput::AbstractVector{<:ParamBox}, 
                           gridInput::AbstractVector{<:ParamBox}, 
                           callNodes::AbstractVector{<:ParamBox})
    dict = ParamBoxVertexDict()
    effects = (VertexAsAccess, VertexAsAccess, VertexAsAction)
    counters = (fill(0), fill(0), fill(0))

    s = map(effects, counters, (unitInput, gridInput, callNodes)) do effect, counter, pars
        if isempty(pars)
            genBottomMemory()
        else
            map(pars) do par
                idx = (counter[] += 1)
                dict[par] = VertexTrait(effect, getVertexOutputLevel(par))=>OneToIndex(idx)
                if effect == VertexAsAction
                    genParamVertex(par, dict)
                else
                    genParamVertex(par)
                end
            end |> genMemory
        end
    end
    s, dict
end


struct UnitValueGraph{T} <: ComputationGraph{T}
    source::UnitVertex{T}

    function UnitValueGraph(vertex::UnitVertex{T}) where {T}
        new{T}(vertex)
    end
end

struct GridValueGraph{T, N} <: ComputationGraph{ShapedMemory{T, N}}
    source::GridVertex{T, N}

    function GridValueGraph(vertex::GridVertex{T, N}) where {T, N}
        new{T, N}(vertex)
    end
end

const SpanValueGraph{T} = Union{UnitValueGraph{T}, GridValueGraph{T}}

SpanValueGraph(vertex::UnitVertex) = UnitValueGraph(vertex)
SpanValueGraph(vertex::GridVertex) = GridValueGraph(vertex)


const TensorVertexSet{U<:AbstractVector{<:UnitVertex}, G<:AbstractVector{<:GridVertex}} = 
      TypedSpanSet{U, G}

struct SpanLayerGraph{T, V<:CallVertex{T}, U<:UnitVertex, G<:GridVertex, H<:CallVertex, 
                      } <: ComputationGraph{T}
    source::OptSpanValueSet{Memory{U}, Memory{G}}
    hidden::Memory{H}
    output::V

    function SpanLayerGraph(source::TensorVertexSet, hidden::AbstractVector{<:CallVertex}, 
                            output::V) where {T, V<:CallVertex{T}}
        unit, grid = map(tightenCollection, source)
        hidden = tightenCollection(hidden)
        new{T, V, eltype(unit), eltype(grid), eltype(hidden)}((;unit, grid), hidden, output)
    end
end

const UnitLayerGraph{T, V<:CallVertex{T}, U<:UnitVertex, H<:CallVertex} = 
      SpanLayerGraph{T, V, U, Union{}, H}
const GridLayerGraph{T, V<:CallVertex{T}, G<:GridVertex, H<:CallVertex} = 
      SpanLayerGraph{T, V, Union{}, G, H}


function genParamSortActivityCounter(::Type{T}) where {T<:ParamBox}
    counter = fill(0)
    f = function countAndLabel(param::T)
        screenLevelOf(param) == 2 && (counter[] += 1)
        getParamOrderLabel(param)
    end
    f, counter
end

function transpileParam(param::ParamBox, reindexInput!::Bool=false)
    sl = screenLevelOf(param)

    if sl < 0 || sl > 2
        throw(AssertionError("The screen level of `param`: $sl is not supported."))
    elseif sl == 0
        inputSetRaw, midPars, outPars, _ = dissectParam(param)

        inUPars, inGPars = inputSetRaw
        if isempty(inUPars) && isempty(inGPars)
            throw(AssertionError("`param` should have at least one input source."))
        end

        # Inactive (screen level: 2) params should be pushed to the end
        counters = map(inputSetRaw, (unit=UnitParam, grid=GridParam)) do params, type
            by, counter = genParamSortActivityCounter(type)
            sortParams!(params; indexing=reindexInput!, by)
            counter
        end

        inputSet = map(inputSetRaw, counters) do sector, counter
            nInactive = counter[]
            if nInactive == length(sector)
                genBottomMemory()
            else
                genMemory(@view sector[begin:end-nInactive])
            end
        end

        (unit, grid, hidden), mapper = formVertexSectors(inUPars, inGPars, midPars)
        graph = SpanLayerGraph((;unit, grid), hidden, genParamVertex(outPars[], mapper))
    else
        isActive = sl == 1

        inputSet = initializeSpanParamSet(isActive ? param : Union{})

        paramSym = symbolFrom(param.symbol)
        graph = genTensorVertex(obtain(param), isActive, paramSym) |> SpanValueGraph
    end

    graph, inputSet
end


function selectUpstreamVertex(graph::SpanLayerGraph, trait::VertexTrait, index::OneToIndex)
    effectType = trait.effect
    outputType = trait.output

    if effectType == VertexAsAccess
        if outputType == VertexToUnit
            vertex = getField(graph.source.unit, index)
        elseif outputType == VertexToGrid
            vertex = getField(graph.source.grid, index)
        else
            throw(AssertionError("`VertexToNest` is unsupported."))
        end
    elseif effectType == VertexAsAction
        vertex = getField(graph.hidden, index)
    else
        throw(AssertionError("`UnstableVertex` is not supported."))
    end

    vertex
end

function genVertexCaller(vertex::TensorVertex)
    value = getVertexValue(vertex)
    TypedReturn(SelectHeader{1, 0}(Storage(value, vertex.marker)), getOutputType(vertex))
end

function genVertexCaller(idx::OneToIndex, vertex::TensorVertex)
    if isVertexActive(vertex)
        fCore = GetIndex{ifelse(vertex isa UnitVertex, UnitIndex, GridIndex)}(idx)
        TypedReturn(fCore, getPackType(vertex|>getOutputType))
    else
        genVertexCaller(vertex)
    end
end

function genVertexCaller(graph::SpanLayerGraph, vertex::CallVertex{T}) where {T}
    receptor = vertex.receptor
    encoders = map(receptor.trait, receptor.index) do inputTrait, inputIndex
        prevVertex = selectUpstreamVertex(graph, inputTrait, inputIndex)
        genVertexCaller(ifelse(prevVertex isa TensorVertex, inputIndex, graph), prevVertex)
    end

    fCore = if receptor isa TupleReceptor
        ComposedApply(ChainMapper(encoders|>Tuple), splat(vertex.apply))
    else
        ComposedApply(ChainMapper(encoders), vertex.apply)
    end

    TypedReturn(fCore, T)
end


function functionalize(graph::SpanValueGraph)
    vertex = graph.source
    f = if isVertexActive(vertex)
        TypedReturn(ChainedAccess((nothing,)), getPackType(vertex|>getOutputType))
    else
        genVertexCaller(vertex)
    end
    inputStyle = ifelse(vertex isa UnitVertex, UnitInput, GridInput)
    SpanEvaluator(f, inputStyle())
end

function functionalize(graph::SpanLayerGraph)
    fCore = genVertexCaller(graph, graph.output)
    inputStyle = map(graph.source) do sector
        res = filter(x->isVertexActive(x), sector)
        isempty(res) ? nothing : res
    end |> getInputSetType
    SpanEvaluator(fCore, inputStyle())
end


evaluateGraph(g::SpanValueGraph) = getVertexValue(g.source)

function evaluateGraph(g::SpanLayerGraph)
    inVals = map(g.source) do sector
        eltype(sector) <: Union{} ? sector : map(getVertexValue, sector)
    end
    genVertexCaller(g, g.output)(inVals)
end


struct ParamGraphCaller{T, A, S<:FixedSpanParamSet, F<:Function} <: GraphEvaluator{T}
    source::S
    evaluate::SpanEvaluator{T, A, F}

    function ParamGraphCaller(param::ParamBox)
        graph, source = transpileParam(param)
        f = functionalize(graph)
        inputStyle = getInputSetType(f)
        new{getOutputType(f), inputStyle, typeof(source), typeof(f.core.f.f)}(source, f)
    end
end

function evalParamGraphCaller(f::F, input::OptSpanValueSet) where {F<:ParamGraphCaller}
    formattedInput = map(input, f.source) do inVals, params
        (inVals === nothing && params !== nothing) ? obtain(params) : inVals
    end
    f.evaluate(formattedInput)
end

const ParamGraphMonoCaller{T, A<:Union{MonoPackInput, HalfSpanInput}, S<:FixedSpanParamSet, 
                           F<:Function} = 
      ParamGraphCaller{T, A, S, F}

(f::ParamGraphMonoCaller)(sector::AbstractVector) = f.evaluate(sector)

(f::ParamGraphCaller)(input::OptSpanValueSet=initializeFixedSpanSet(nothing)) = 
evalParamGraphCaller(f, input)

getOutputType(::Type{<:ParamGraphCaller{T}}) where {T} = T


const FilterEvalParam{F<:SpanEvaluator, S<:SpanSetFilter} = ComposedApply{F, S}

const ParamMapper{S, F<:NamedTuple{ S, <:NonEmptyTuple{FilterEvalParam} }} = 
      ChainMapper{F}

function genParamMapper(params::NamedParamTuple; 
                        paramSet!Self::OptSpanParamSet=initializeSpanParamSet())
    checkEmptiness(params, :params)
    mapper = map(params) do param
        evaluator = ParamGraphCaller(param)
        inputFilter = locateParam!(paramSet!Self, evaluator.source)
        ComposedApply(inputFilter, evaluator.evaluate)
    end |> ChainMapper
    mapper, paramSet!Self
end


const EncodeParamApply{B<:Function, F<:ParamMapper} = ContextParamFunc{B, ItsType, F}