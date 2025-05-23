export ValueParamGraph, LayerParamGraph, genParamGraph, transpileGraph, evaluateGraph, 
       compressParam

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
    apply::TypedReturn{T, F}
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


struct ValueParamGraph{T, V<:TensorVertex{T}} <: TransformedGraph{Directed}
    origin::SpanParam{T}
    source::V

    function ValueParamGraph(param::SpanParam{T}) where {T}
        sl = screenLevelOf(param)
        sl == 0 && throw(AssertionError("The screen level of `param` must larger than 0."))
        vertex = genTensorVertex(obtain(param), sl==1, symbolFrom(param.symbol))
        new{T, typeof(vertex)}(param, vertex)
    end
end

struct LayerParamGraph{T, S1<:UnitVertex, S2<:GridVertex, H<:CallVertex, 
                       V<:CallVertex{T}} <: TransformedGraph{Directed}
    origin::OptionalSpanSet{Memory{UnitParam}, Memory{GridParam}}
    source::OptionalSpanSet{Memory{S1},        Memory{S2}       }
    hidden::Memory{H}
    output::V

    function LayerParamGraph(param::ParamBox)
        (inUPars, inGPars), midPars, outPars, _ = dissectParam(param)
        if isempty(inUPars) && isempty(inGPars)
            throw(AssertionError("`param` should have at least one input source."))
        end

        sortParams!(inUPars, indexing=false)
        sortParams!(inGPars, indexing=false)
        origin = (unit=Memory{UnitParam}(inUPars), grid=Memory{GridParam}(inGPars))

        sectors, mapper = formVertexSectors(inUPars, inGPars, midPars)

        unitSource, gridSource, hidden = map(sectors) do sector
            isempty(sector) ? Memory{Union{}}(undef, 0) : genMemory(sector)
        end
        source = (unit=unitSource, grid=gridSource)

        outPar = outPars[]
        output = genParamVertex(outPar, mapper)

        new{getOutputType(outPar), eltype(unitSource), eltype(gridSource), eltype(hidden), 
            typeof(output)}(origin, source, hidden, output)
    end
end

function genParamGraph(param::ParamBox)
    sl = screenLevelOf(param)
    if sl == 0
        LayerParamGraph(param)
    else
        ValueParamGraph(param)
    end
end

const ParamGraph{T} = Union{ValueParamGraph{T}, LayerParamGraph{T}}


isNodeActive(node::TensorVertex) = node.active

getNodeValue(node::TensorVertex) = node.value


function evaluateGraph end

(f::ComputationGraph{T})() where {T} = evaluateGraph(f)::T
(f::ComputationGraph{T})(input::OptionalSpanValueSet) where {T} = evaluateGraph(f, input)::T


struct UnitValueGraph{T, M<:Activity} <: ComputationGraph{T}
    node::UnitVertex{T}

    function UnitValueGraph(vertex::UnitVertex{T}) where {T}
        new{T, ifelse(vertex.active, Active, Static)}(vertex)
    end
end

struct GridValueGraph{T, N, M<:Activity} <: ComputationGraph{ShapedMemory{T, N}}
    node::GridVertex{T, N}

    function GridValueGraph(vertex::GridVertex{T, N}) where {T, N}
        new{T, N, ifelse(vertex.active, Active, Static)}(vertex)
    end
end

const ValueGraph{T, M<:Activity} = Union{UnitValueGraph{T, M}, GridValueGraph{T, <:Any, M}}

ValueGraph(vertex::UnitVertex) = UnitValueGraph(vertex)
ValueGraph(vertex::GridVertex) = GridValueGraph(vertex)

const StaticValueGraph{T} = ValueGraph{T, Static}
const ActiveValueGraph{T} = ValueGraph{T, Active}


evaluateGraph(f::ValueGraph) = getNodeValue(f.node)

evaluateGraph(f::StaticValueGraph, ::OptionalVoidValueSet) = evaluateGraph(f)

getParamInputType(::UnitValueGraph) = UnitInput()
getParamInputType(::GridValueGraph) = GridInput()

function evaluateGraph(f::ActiveValueGraph, input::OptionalSpanValueSet)
    sector = constrainSpanValueSet(getParamInputType(f), input)
    content = getfield(input, sector)
    content === nothing ? evaluateGraph(f) : getindex(content)
end


const LayerGraphCore{VU<:UnitVertex, VG<:GridVertex, VC<:CallVertex} = 
      @NamedTuple{unit::Memory{VU}, grid::Memory{VG}, call::Memory{VC}}

const UnitLayerGraphCore{VU<:UnitVertex, VC<:CallVertex} = LayerGraphCore{VU, Union{}, VC}

const GridLayerGraphCore{VG<:GridVertex, VC<:CallVertex} = LayerGraphCore{Union{}, VG, VC}

const VoidLayerGraphCore{VC<:CallVertex} = LayerGraphCore{Union{}, Union{}, VC}

struct LayerGraph{T, B<:LayerGraphCore, F<:Function} <: ComputationGraph{T}
    attribute::B
    evaluator::TypedReturn{T, F}

    function LayerGraph(attribute::B, evaluator::TypedReturn{T, F}) where 
                       {T, B<:LayerGraphCore, F<:Function}
        attribute = map(attribute) do x
            ifelse(isempty(x), genBottomMemory(), x)
        end

        if isempty(attribute.unit) && isempty(attribute.grid)
            throw(AssertionError("`attribute.unit` and `attribute.grid` must not be "*
                                 "both empty."))
        end

        new{T, B, F}(attribute, evaluator)
    end
end

const UnitLayerGraph{T, B<:UnitLayerGraphCore, F<:Function} = LayerGraph{T, B, F}

const GridLayerGraph{T, B<:GridLayerGraphCore, F<:Function} = LayerGraph{T, B, F}

const VoidLayerGraph{T, B<:VoidLayerGraphCore, F<:Function} = LayerGraph{T, B, F}

getParamInputType(::LayerGraph) = SpanInput()

getParamInputType(::UnitLayerGraph) = UnitInput()

getParamInputType(::GridLayerGraph) = GridInput()

getParamInputType(::VoidLayerGraph) = 
throw(AssertionError("`$VoidLayerGraph` is not a valid construction of `$LayerGraph`."))


function transpileGraph(graph::ValueParamGraph)
    ValueGraph(graph.source)
end

function transpileGraph(graph::LayerParamGraph)
    unitSource, gridSource = graph.source
    body = (unit=unitSource, grid=gridSource, call=graph.hidden)
    f = genCallVertexEvaluator(body, graph.output)
    LayerGraph(body, f)
end


function evaluateGraph(f::LayerGraph)
    unit, grid, call = f.attribute
    unit, grid = map((unit, grid)) do field
        isempty(field) ? field : map(getNodeValue, field)
    end
    f.evaluator((;unit, grid, call))
end

function mixTensorInput(nodes::Memory{<:TensorVertex}, ::Nothing)
    if eltype(nodes) <: Union{}
        genBottomMemory()
    else
        map(getNodeValue, nodes)
    end
end

function mixTensorInput(nodes::Memory{<:TensorVertex}, input::AbstractVector)
    i = 1
    if eltype(nodes) <: Union{} || isempty(nodes)
        genBottomMemory()
    else
        map(nodes) do node
            if isNodeActive(node)
                value, i = iterate(input, i)
                value
            else
                getNodeValue(node)
            end
        end
    end
end

function evaluateGraph(tpg::LayerGraph, input::OptionalSpanValueSet)
    constrainSpanValueSet(getParamInputType(tpg), input)
    units, grids, call = tpg.attribute
    unit = mixTensorInput(units, input.unit)
    grid = mixTensorInput(grids, input.grid)
    tpg.evaluator((;unit, grid, call))
end


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

function genUnitVertexEvaluator(::LayerGraphCore, idx::Int)
    GetIndex{UnitIndex}(idx)
end

function genGridVertexEvaluator(::LayerGraphCore, idx::Int)
    GetIndex{GridIndex}(idx)
end

function genCallVertexEvaluatorCore(::TupleReceptor, fs::AbstractArray{<:Function}, 
                                    apply::TypedReturn{T}) where {T}
    TypedReturn(splat(apply.f) ∘ ChainMapper(fs|>Tuple), T)
end

function genCallVertexEvaluatorCore(::ArrayReceptor, fs::AbstractArray{<:Function}, 
                                    apply::TypedReturn{T}) where {T}
    TypedReturn(apply.f ∘ ChainMapper(fs), T)
end

function genCallVertexEvaluator(compactPG::LayerGraphCore, 
                                vertex::CallVertex{T}) where {T}
    receptor = vertex.receptor
    fs = map(receptor.trait, receptor.index) do t, i
        selectVertexEvalGenerator(t)(compactPG, i)
    end
    genCallVertexEvaluatorCore(receptor, fs, vertex.apply)
end

function genCallVertexEvaluator(compactPG::LayerGraphCore, i::Int)
    vertex = compactPG.call[begin+i-1]
    genCallVertexEvaluator(compactPG, vertex)
end

struct SpanInputFormatter{S1<:Symbol, S2<:Pair{ Symbol, <:NonEmptyTuple{Int} }} <: Mapper
    unit::Memory{S1}
    grid::Memory{S2}

    function SpanInputFormatter(unitSource::Memory{<:UnitVertex}, 
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

    function SpanInputFormatter(unit::UnitVertex)
        unitInfo = isNodeActive(unit) ? genMemory(unit.marker) : genBottomMemory()
        new{eltype(unitInfo), Union{}}(unitInfo, genBottomMemory())
    end

    function SpanInputFormatter(grid::GridVertex{T, N}) where {T, N}
        gridInfo = if isNodeActive(grid)
            genMemory(grid.marker => (size∘getNodeValue)(grid))
        else
            genBottomMemory()
        end
        new{Union{}, eltype(gridInfo)}(genBottomMemory(), gridInfo)
    end
end

function SpanInputFormatter(graph::LayerGraph)
    unit, grid, _ = graph.attribute
    SpanInputFormatter(unit, grid)
end

function SpanInputFormatter(graph::ValueGraph)
    SpanInputFormatter(graph.node)
end

const VoidInputFormatter = SpanInputFormatter{Union{}, Union{}}
const UnitInputFormatter = SpanInputFormatter{Symbol, Union{}}
const GridInputFormatter{G<:Pair{ Symbol, <:NonEmptyTuple{Int} }} = 
                         SpanInputFormatter{Union{}, G}

function formatUnitInput(unit::Memory{Symbol}, flattenedInput::AbstractVector)
    nUnit = length(unit)
    flattenedInput[begin:begin+nUnit-1]
end

function formatGridInput(grid::Memory{<:Pair{ Symbol, <:NonEmptyTuple{Int} }}, 
                         flattenedInput::AbstractVector)
    idx = firstindex(flattenedInput)
    map(grid) do ele
        val = reshape(flattenedInput[idx], ele.second)
        idx += prod(ele.second)
        val
    end
end

function getField(::AbstractVector, ::VoidInputFormatter)
    (unit=nothing, grid=nothing)
end

getField(flattenedInput::AbstractVector, encoder::UnitInputFormatter) = 
(unit=formatUnitInput(encoder.unit, flattenedInput), grid=nothing)

getField(flattenedInput::AbstractVector, encoder::GridInputFormatter) = 
(unit=nothing, grid=formatGridInput(encoder.grid, flattenedInput))

function getField(flattenedInput::AbstractVector, encoder::SpanInputFormatter)
    unitInput = formatUnitInput(encoder.unit, flattenedInput)
    nUnit = length(encoder.unit)
    gridInput = formatGridInput(encoder.unit, (@view flattenedInput[begin+nUnit:end]))
    (unit=unitInput, grid=gridInput)
end

getParamInputType(::UnitInputFormatter) = UnitInput()
getParamInputType(::GridInputFormatter) = GridInput()
getParamInputType(::SpanInputFormatter) = SpanInput()
getParamInputType(::VoidInputFormatter) = VoidInput()

function getField(spanInput::OptionalSpanValueSet, formatter::SpanInputFormatter)
    constrainSpanValueSet(getParamInputType(formatter), spanInput)
    spanInput
end

(f::SpanInputFormatter)() = (unit=nothing, grid=nothing)


struct ComputeGraph{T, G<:ComputationGraph{T}, F<:SpanInputFormatter} <: GraphEvaluator{G}
    formatter::F
    evaluator::LPartial{typeof(evaluateGraph), Tuple{G}}

    function ComputeGraph(graph::G, formatter::F) where 
                         {T, G<:ComputationGraph{T}, F<:SpanInputFormatter}
        new{T, G, F}(formatter, LPartial( evaluateGraph, (graph,) ))
    end
end

function (f::ComputeGraph)()
    f.formatter() |> f.evaluator
end

function (f::ComputeGraph)(input::Union{OptionalSpanValueSet, AbstractVector})
    f.formatter(input) |> f.evaluator
end

getOutputType(::Type{<:ComputeGraph{T}}) where {T} = T


function compressGraph(graph::ComputationGraph)
    encoder = SpanInputFormatter(graph)
    ComputeGraph(graph, encoder)
end

function compressGraph(graph::TransformedGraph)
    transpiledGraph = transpileGraph(graph)
    compressGraph(transpiledGraph)
end

function compressParam(param::ParamBox)
    paramGraph = genParamGraph(param)
    sl = screenLevelOf(param)
    inputSet = if sl == 2
        initializeSpanParamSet(nothing)
    elseif sl == 1
        initializeSpanParamSet(paramGraph.origin)
    elseif sl == 0
        paramGraph.origin
    else
        throw(AssertionError("The screen level of `param`: $sl is not supported."))
    end
    compressGraph(paramGraph), inputSet
end


const FilterComputeGraph{G<:ComputeGraph, S<:SpanSetFilter} = Base.ComposedFunction{G, S}

const ParamEncoderChain = NonEmptyTuple{Union{GetIndex{<:SpanIndex}, FilterComputeGraph}}

const ParamMapper{S, F<:NamedTuple{S, <:ParamEncoderChain}} = ChainMapper{F}

function genParamMapper(params::NamedParamTuple; 
                        paramSet!Self::AbstractSpanParamSet=initializeSpanParamSet())
    checkEmptiness(params, :params)
    mapper = map(params) do param
        if screenLevelOf(param) == 1
            locateParam!(paramSet!Self, param)
        else
            encoder, inputSet = compressParam(param)
            inputFilter = locateParam!(paramSet!Self, inputSet)
            encoder ∘ inputFilter
        end
    end |> ChainMapper
    mapper, paramSet!Self
end


const EncodeParamApply{B<:Function, F<:ParamMapper} = ContextParamFunc{B, ItsType, F}