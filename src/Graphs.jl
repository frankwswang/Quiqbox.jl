export ParamGraph, evaluateGraph, compressParam

struct Directed end
struct Undirected end

const Orientation = Union{Directed, Undirected}

struct Static end
struct Dynamic end

const Mutability = Union{Static, Dynamic}

abstract type AbstractGraph{M<:Mutability, O<:Orientation} end

abstract type ComputationGraph{M<:Mutability} <: AbstractGraph{M, Directed} end

abstract type TransformedGraph{M<:Mutability} <: AbstractGraph{M, Directed} end

abstract type GraphVertex end

abstract type TypedVertex{T} <: GraphVertex end

abstract type AccessVertex{T} <: TypedVertex{T} end

abstract type ActionVertex{T} <: TypedVertex{T} end

struct UnitVertex{T} <: AccessVertex{T}
    value::T
    active::Bool
    marker::Symbol
end

struct GridVertex{T, N} <: AccessVertex{ShapedMemory{T, N}}
    value::ShapedMemory{T, N}
    active::Bool
    marker::Symbol
end

const TensorVertex{T} = Union{UnitVertex{T}, GridVertex{T}}

function genTensorVertex(val::T, active::Bool, marker::Symbol) where {T}
    UnitVertex(val, active, marker)
end

function genTensorVertex(val::AbstractArray{T}, active::Bool, marker::Symbol) where {T}
    GridVertex(ShapedMemory(val), active, marker)
end

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
    index::VectorMemory{Int, N}

    function TupleReceptor(nInput::Int, defaultTrait::VertexTrait=VertexTrait())
        checkPositivity(nInput)
        trait = VectorMemory(fill(defaultTrait, nInput))
        index = VectorMemory(fill(0, nInput))
        new{nInput}(trait, index)
    end
end

struct ArrayReceptor{N} <: VertexReceptor
    trait::ShapedMemory{VertexTrait, N}
    index::ShapedMemory{Int, N}

    function ArrayReceptor(sInput::NTuple{N, Int}, 
                           defaultTrait::VertexTrait=VertexTrait()) where {N}
        trait = ShapedMemory(fill(defaultTrait, sInput))
        index = ShapedMemory(fill(0, sInput))
        new{N}(trait, index)
    end
end

const CallVertexReceptor{N} = Union{TupleReceptor{N}, ArrayReceptor{N}}

struct CallVertex{T, I<:CallVertexReceptor, F<:Function} <: ActionVertex{T}
    apply::ReturnTyped{T, F}
    marker::Symbol
    receptor::I # Could potentially have repeated input indices
end

const ParamBoxVertexDict = IdDict{ParamBox, Pair{VertexTrait, Int}}

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
                            reference::Union{Nothing, ParamBoxVertexDict}=nothing)
    receptor = TupleReceptor(length(param.input))
    reference === nothing || updateVertexReceptor!(receptor, param, reference)
    receptor
end

function initializeReceptor(param::ShapedParam, 
                            reference::Union{Nothing, ParamBoxVertexDict}=nothing)
    receptor = ArrayReceptor(size(param.input))
    reference === nothing || updateVertexReceptor!(receptor, param, reference)
    receptor
end

function genParamVertex(param::CompositeParam, 
                        reference::Union{Nothing, ParamBoxVertexDict}=nothing)
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
        map(pars) do par
            idx = (counter[] += 1)
            dict[par] = VertexTrait(effect, getVertexOutputLevel(par))=>idx
            if effect == VertexAsAction
                genParamVertex(par, dict)
            else
                genParamVertex(par)
            end
        end
    end
    s, dict
end


struct ParamGraph{T, S1<:UnitVertex, S2<:GridVertex, H<:CallVertex, 
                  V<:CallVertex{T}} <: TransformedGraph{Static}
    origin::@NamedTuple{unit::Memory{UnitParam}, grid::Memory{GridParam}}
    source::@NamedTuple{unit::Memory{S1}, grid::Memory{S2}}
    hidden::Memory{H}
    output::V

    function ParamGraph(pb::ParamBox; indexing!Arg::Bool=false)
        (inUPars, inGPars), midPars, outPars, _ = classifyParams(pb)
        if isempty(inUPars) && isempty(inGPars)
            throw(AssertionError("`pb` should have at least one input source."))
        end

        sortParams!(inUPars, indexing=indexing!Arg)
        sortParams!(inGPars, indexing=indexing!Arg)
        origin = (unit=Memory{UnitParam}(inUPars), grid=Memory{GridParam}(inGPars))

        sectors, mapper = formVertexSectors(inUPars, inGPars, midPars)

        unitSource, gridSource, hidden = map(sectors) do sector
            isempty(sector) ? Memory{Union{}}(undef, 0) : getMemory(sector)
        end
        source = (unit=unitSource, grid=gridSource)

        outPar = outPars[]
        output = genParamVertex(outPar, mapper)

        new{getOutputType(outPar), eltype(unitSource), eltype(gridSource), eltype(hidden), 
            typeof(output)}(origin, source, hidden, output)
    end
end


isNodeActive(node::TensorVertex) = node.active
getNodeValue(node::TensorVertex) = node.value

getSourceInfo(vertex::TensorVertex) = (vertex.marker, vertex.active)=>vertex.value

const SourceNodeInfo{T} = Tuple{Symbol, Bool, T}

const ParamGraphCore{V1<:UnitVertex, V2<:GridVertex, V3<:CallVertex} = 
      @NamedTuple{unit::Memory{V1}, grid::Memory{V2}, call::Memory{V3}}

const GraphCore = Union{ParamGraphCore}

struct TranspiledGraph{T, B<:GraphCore, F<:Function} <: ComputationGraph{Static}
    attribute::B
    evaluator::ReturnTyped{T, F}
end

function TranspiledGraph(graph::ParamGraph)
    unitSource, gridSource = graph.source
    body = (unit=unitSource, grid=gridSource, call=graph.hidden)
    f = genCallVertexEvaluator(body, graph.output)
    TranspiledGraph(body, f)
end

const TranspiledParamGraph{T, B<:ParamGraphCore, F} = TranspiledGraph{T, B, F}

(f::TranspiledGraph)() = f.evaluator(f.attribute)

function prepareSourceInfo(nodes::AbstractVector{<:TensorVertex}, ::Nothing)
    getfield.(nodes, :value)
end

function prepareSourceInfo(nodes::AbstractVector{<:TensorVertex}, input::AbstractVector)
    i = firstindex(input)
    map(nodes) do node
        if node.active
            val = input[i]
            i += 1
            val
        else
            node.value
        end
    end
end

const UnitOrGridInput = Tuple{Union{Nothing, AbstractVector}, 
                              Union{Nothing, AbstractVector{<:AbstractArray} }}

function evaluateGraphCore(tpg::TranspiledParamGraph, (uInput, gInput)::UnitOrGridInput)
    unit, grid, call = tpg.attribute
    unitVals = prepareSourceInfo(unit, uInput)
    gridVals = prepareSourceInfo(grid, gInput)
    tpg.evaluator((unit=unitVals, grid=gridVals, call=call))
end

function evaluateGraph(graph::ParamGraph, inputPair::UnitOrGridInput)
    tpg = TranspiledGraph(graph)
    evaluateGraphCore(tpg, inputPair)
end

const NamedUGInput{T1<:AbstractVector, T2<:AbstractVector{<:AbstractArray}} = 
      @NamedTuple{unit::T1, grid::T2}

evaluateGraph(graph::ParamGraph, 
              inputPair::NamedUGInput=map(x->getNodeValue.(x), graph.source)) = 
evaluateGraph(graph, values(inputPair))

function selectVertexEvalGenerator(trait::VertexTrait)
    effectType = trait.effect
    outputType = trait.output
    if effectType == VertexAsAccess
        if outputType == VertexToUnit
            genUnitVertexEvaluator
        elseif outputType == VertexToGrid
            genGridVertexEvaluator
        else
            throw(AssertionError("`VertexToNest` is unsupported."))
        end
    elseif effectType == VertexAsAction
        genCallVertexEvaluator
    else
        throw(AssertionError("`UnstableVertex` is not supported."))
    end
end

function genUnitVertexEvaluator(::ParamGraphCore, idx::Int)
    # localIdx = firstindex(b.unit) + idx - 1
    # (Retrieve∘ChainedAccess)((:unit, localIdx))
    GetIndex{UnitIndex}(idx)
end

function genGridVertexEvaluator(::ParamGraphCore, idx::Int)
    # localIdx = firstindex(b.grid) + idx - 1
    # (Retrieve∘ChainedAccess)((:grid, localIdx))
    GetIndex{GridIndex}(idx)
end

function genCallVertexEvaluatorCore(::TupleReceptor, fs::AbstractArray{<:Function}, 
                                    apply::ReturnTyped{T}) where {T}
    ReturnTyped(ChainExpand(Tuple(fs), splat(apply.f)), T)
end

function genCallVertexEvaluatorCore(::ArrayReceptor, fs::AbstractArray{<:Function}, 
                                    apply::ReturnTyped{T}) where {T}
    ReturnTyped(ChainExpand(fs, apply.f), T)
end

function genCallVertexEvaluator(compactPG::ParamGraphCore, 
                                vertex::CallVertex{T}) where {T}
    receptor = vertex.receptor
    fs = map(receptor.trait, receptor.index) do t, i
        selectVertexEvalGenerator(t)(compactPG, i)
    end
    genCallVertexEvaluatorCore(receptor, fs, vertex.apply)
end

function genCallVertexEvaluator(compactPG::ParamGraphCore, i::Int)
    vertex = compactPG.call[begin+i-1]
    genCallVertexEvaluator(compactPG, vertex)
end

struct TensorInputEncoder{S1<:Symbol, S2<:Pair{ Symbol, <:NonEmptyTuple{Int} }}
    unit::Memory{S1}
    grid::Memory{S2}

    function TensorInputEncoder(unitSource::Memory{<:UnitVertex}, 
                                gridSource::Memory{<:GridVertex})
        unitInfo = if isempty(unitSource)
            Memory{Union{}}(undef, 0)
        else
            map(x->x.marker, filter(isNodeActive, unitSource))
        end
        gridInfo = if isempty(gridSource)
            Memory{Union{}}(undef, 0)
        else
            map(x->x.marker=>(size∘getNodeValue)(x), filter(isNodeActive, gridSource))
        end
        new{eltype(unitInfo), eltype(gridInfo)}(unitInfo, gridInfo)
    end
end

const UnitInputEncoder = TensorInputEncoder{Symbol, Union{}}
const GridInputEncoder{G<:Pair{ Symbol, <:NonEmptyTuple{Int} }} = 
                       TensorInputEncoder{Union{}, G}

function encodeUnitInput(unit::Memory{Symbol}, flattenedInput::AbstractVector)
    nUnit = length(unit)
    flattenedInput[begin:begin+nUnit-1]
end

function encodeGridInput(grid::Memory{<:Pair{ Symbol, <:NonEmptyTuple{Int} }}, 
                         flattenedInput::AbstractVector)
    idx = firstindex(flattenedInput)
    map(grid) do ele
        val = reshape(flattenedInput[idx], ele.second)
        idx += 1
        val
    end
end

(f::UnitInputEncoder)(flattenedInput::AbstractVector) = 
(encodeUnitInput(f.unit, flattenedInput), nothing)

(f::GridInputEncoder)(flattenedInput::AbstractVector) = 
(nothing, encodeGridInput(f.grid, flattenedInput))


function (f::TensorInputEncoder)(flattenedInput::AbstractVector)
    unitInput = encodeUnitInput(f.unit, flattenedInput)
    nUnit = length(f.unit)
    gridInput = encodeGridInput(f.unit, (@view flattenedInput[begin+nUnit:end]))
    unitInput, gridInput
end

const UnitAndGridInput{T, V<:AbstractArray} = Tuple{AbstractVector{T}, AbstractVector{V}}

function (f::TensorInputEncoder)((unitInput, gridInput)::UnitAndGridInput)
    unitInput = unitInput[begin:begin+length(f.unit)-1]
    gridInput = map(enumerate(f.grid)) do (i, grid)
        reshape(gridInput[begin+i-1], grid.second)
    end

    unitInput, gridInput
end

(f::TensorInputEncoder)() = (nothing, nothing)

(f::TensorInputEncoder)(args::NamedTuple) = f(args|>values)

abstract type GraphEvaluator{G} <: Evaluator{typeof(evaluateGraphCore)} end

const ComputableParamGraph{G<:TranspiledGraph} = 
      LPartial{typeof(evaluateGraphCore), Tuple{G}}

struct EvalParamGraph{T, G<:TranspiledGraph{T}, E<:TensorInputEncoder} <: GraphEvaluator{G}
    f::Base.ComposedFunction{ComputableParamGraph{G}, E}
end

(f::EvalParamGraph)() = f.f()
(f::EvalParamGraph)(flattenInput) = f.f(flattenInput)
(f::EvalParamGraph)(unitInput, gridInput) = f.f(unitInput, gridInput)


function compressParam(pb::ParamBox)
    graph = ParamGraph(pb)
    transpiledGraph = TranspiledGraph(graph)
    encoder = TensorInputEncoder(graph.source...)
    cgc = LPartial(evaluateGraphCore, (transpiledGraph,))
    EvalParamGraph(cgc∘encoder), graph.origin
end