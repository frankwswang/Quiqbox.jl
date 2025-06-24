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
    value::AtomicUnit{T}
    active::Bool
    marker::Symbol
end

getOutputType(::Type{UnitVertex{T}}) where {T} = T

struct GridVertex{T, N} <: AccessVertex{ShapedMemory{T, N}}
    value::AtomicGrid{ShapedMemory{T, N}}
    active::Bool
    marker::Symbol
end

getOutputType(::Type{GridVertex{T, N}}) where {T, N} = ShapedMemory{T, N}

const TensorVertex{T} = Union{UnitVertex{T}, GridVertex{T}}


isVertexActive(vertex::TensorVertex) = vertex.active

getVertexValue(vertex::TensorVertex) = getindex(vertex.value)


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


struct TupleReceptor{L} <: VertexReceptor
    trait::VectorMemory{VertexTrait, L}
    index::VectorMemory{OneToIndex, L}

    function TupleReceptor(::Count{L}, defaultTrait::VertexTrait=VertexTrait()) where {L}
        checkPositivity(L) #> `L` should generally be small to avoid compiler overhead
        trait = VectorMemory(fill(defaultTrait, L))
        index = VectorMemory(fill(OneToIndex(), L))
        new{L}(trait, index)
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

const CallVertexReceptor = Union{TupleReceptor, ArrayReceptor}


struct CallVertex{T, R<:CallVertexReceptor, F<:Function} <: ActionVertex{T}
    apply::TypedReturn{T, F}
    marker::Symbol
    receptor::R #> Could potentially have repeated input indices
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
    receptor = TupleReceptor((Count∘length)(param.input))
    reference === nothing || updateVertexReceptor!(receptor, param, reference)
    receptor
end

function initializeReceptor(param::ShapedParam, 
                            reference::NothingOr{ParamBoxVertexDict}=nothing)
    receptor = ArrayReceptor(size(param.input))
    reference === nothing || updateVertexReceptor!(receptor, param, reference)
    receptor
end

#> `genParamVertex` will reference rather than copy the core data from the input `ParamBox`
genParamVertexCore(val::AtomicUnit, active::Bool, marker::Symbol) = 
UnitVertex(val, active, marker)

genParamVertexCore(val::AtomicGrid{<:DirectMemory}, active::Bool, marker::Symbol) = 
GridVertex(AtomicGrid(val.value.value), active, marker)

function genParamVertex(param::CompositeParam, 
                        reference::NothingOr{ParamBoxVertexDict}=nothing)
    sl = screenLevelOf(param)
    sym = symbolOf(param)
    if sl > 0
        genParamVertexCore(param.offset, sl<2, sym)
    else
        apply = extractTransform(param)
        receptor = initializeReceptor(param, reference)
        CallVertex(apply, sym, receptor)
    end
end

function genParamVertex(param::PrimitiveParam)
    genParamVertexCore(param.data, screenLevelOf(param)<2, symbolOf(param))
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
    source::TensorVertexSet{Memory{U}, Memory{G}}
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


function transpileParam(param::ParamBox, reindexInput!::Bool=false)
    sl = screenLevelOf(param)::Int

    if sl < 0 || sl > 2
        throw(AssertionError("The screen level of `param`: $sl is not supported."))
    elseif sl == 0
        inputSetRaw, midPars, outPars, _ = dissectParam(param)

        inUPars, inGPars = inputSetRaw
        if isempty(inUPars) && isempty(inGPars)
            throw(AssertionError("`param` should have at least one input source."))
        end

        #> Inactive (screen level: 2) params should be pushed to the end
        indices = map(inputSetRaw) do sector
            sortParams!(sector; indexing=reindexInput!)
            findfirst(x->screenLevelOf(x)==2, sector)
        end

        inputSet = map(inputSetRaw, indices) do sector, idx
            if idx === nothing
                genMemory(sector)
            elseif idx == firstindex(sector)
                genBottomMemory()
            else
                genMemory(@view sector[begin:idx-1])
            end
        end

        (unit, grid, hidden), mapper = formVertexSectors(inUPars, inGPars, midPars)
        graph = SpanLayerGraph((;unit, grid), hidden, genParamVertex(outPars[], mapper))
    else #> `sl in (1, 2)`
        isActive = sl == 1
        inputSet = isActive ? initializeSpanParamSet(param) : initializeFixedSpanSet()
        graph = genParamVertex(param) |> SpanValueGraph
    end

    graph, inputSet::FixedSpanParamSet #> `inputSet` only contains active `TensorVertex`
end


function selectUpstreamVertex(graph::SpanLayerGraph, trait::VertexTrait, index::OneToIndex)
    effectType = trait.effect
    outputType = trait.output

    if effectType == VertexAsAccess
        if outputType == VertexToUnit
            vertex = getEntry(graph.source.unit, index)
        elseif outputType == VertexToGrid
            vertex = getEntry(graph.source.grid, index)
        else
            throw(AssertionError("`VertexToNest` is unsupported."))
        end
    elseif effectType == VertexAsAction
        vertex = getEntry(graph.hidden, index)
    else
        throw(AssertionError("`UnstableVertex` is not supported."))
    end

    vertex
end

#> Vertex caller that effectively returns a pre-stored value no longer reference its source
function genVertexCaller(vertex::TensorVertex)
    value = getVertexValue(vertex)
    SelectHeader{1, 0}(Storage( copy(value), (nameof∘typeof)(vertex) ))
end

function genVertexCaller(idx::OneToIndex, vertex::TensorVertex)
    if isVertexActive(vertex)
        ifelse(vertex isa UnitVertex, GetUnitEntry, GetGridEntry)(idx)
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

    if receptor isa TupleReceptor
        nInput = length(receptor.trait)
        ComposedApply(ChainMapper(encoders|>Tuple), TupleSplitHeader{nInput}(vertex.apply))
    else
        ComposedApply(ChainMapper(encoders), vertex.apply)
    end
end


function functionalize(graph::SpanValueGraph)
    vertex = graph.source
    isUnit = vertex isa UnitVertex
    isActive = isVertexActive(vertex)

    inputStyle = ifelse(isActive, (isUnit ? UnitInput : GridInput), VoidInput)()

    fCore = if isActive
        ifelse(isUnit, GetUnitEntry, GetGridEntry)(OneToIndex())
    else
        genVertexCaller(vertex)
    end

    SpanSetCaller(TypedReturn(fCore, getOutputType(vertex)), inputStyle)
end

function functionalize(graph::UnitLayerGraph{T}) where {T}
    fCore = genVertexCaller(graph, graph.output)
    SpanSetCaller(TypedReturn(fCore, T), UnitInput())
end

function functionalize(graph::GridLayerGraph{T}) where {T}
    fCore = genVertexCaller(graph, graph.output)
    SpanSetCaller(TypedReturn(fCore, T), GridInput())
end

function functionalize(graph::SpanLayerGraph{T}) where {T}
    fCore = genVertexCaller(graph, graph.output)
    SpanSetCaller(TypedReturn(fCore, T), SpanInput())
end


evaluateGraph(g::SpanValueGraph) = getVertexValue(g.source)

function evaluateGraph(g::SpanLayerGraph{T}) where {T}
    inVals = map(g.source) do sector
        eltype(sector) <: Union{} ? sector : map(getVertexValue, sector)
    end
    convert(T, genVertexCaller(g, g.output)(inVals))
end


struct ParamGraphCaller{T, A<:OptSpanInput, S<:FixedSpanParamSet, F<:Function
                        } <: GraphEvaluator{T}
    source::S
    evaluate::SpanSetCaller{T, A, F}

    function ParamGraphCaller(param::ParamBox)
        graph, source = transpileParam(param)
        f = functionalize(graph)
        inputStyle = getInputSetType(f)
        new{getOutputType(f), inputStyle, typeof(source), typeof(f.core.f.f)}(source, f)
    end
end

function evalParamGraphCaller(f::F, input::OptSpanValueSet) where {F<:ParamGraphCaller}
    formattedInput = map(input, f.source) do inVals, inPars
        (inVals === nothing && !(eltype(inPars) <: Union{})) ? obtain(inPars) : inVals
    end
    f.evaluate(formattedInput)
end

const ParamGraphMonoCaller{T, A<:MonoPackInput, S<:FixedSpanParamSet, F<:Function} = 
      ParamGraphCaller{T, A, S, F}

function evalParamGraphCaller(f::ParamGraphMonoCaller, input::AbstractVector)
    f.evaluate(input)
end

(f::ParamGraphCaller)(input=initializeFixedSpanSet(nothing)) = 
evalParamGraphCaller(f, input)

getOutputType(::Type{<:ParamGraphCaller{T}}) where {T} = T


const FilterEvalParam{F<:SpanSetCaller, S<:SpanSetFilter} = ComposedApply{1, F, GetEntry{S}}

const ParamMapperCore = Union{FilterEvalParam, GetTypedUnit, GetTypedGrid}

const ParamMapper{F<:FunctionChainUnion{ParamMapperCore}} = ChainMapper{F}

const NamedParamMapper{S, F<:NamedTuple{ S, <:NonEmptyTuple{ParamMapperCore} }} = 
      ParamMapper{F}

function genParamMapper(params::DirectParamSource; 
                        paramSet!Self::OptSpanParamSet=initializeSpanParamSet())
    checkEmptiness(params, :params)
    mapper = map(params) do param
        if screenLevelOf(param) == 1
            paramIndexer = locateParam!(paramSet!Self, param)
            TypedReturn(GetEntry(paramIndexer), getOutputType(param))
        else
            evaluator = ParamGraphCaller(param)
            inputFilter = locateParam!(paramSet!Self, evaluator.source)
            ComposedApply(GetEntry(inputFilter), evaluator.evaluate)
        end
    end |> ChainMapper
    mapper, paramSet!Self
end