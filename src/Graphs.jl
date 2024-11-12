export genGraphNode, evaluateNode, compressNode

using Base: Fix1

const NodeChildrenType{T} = TernaryNTupleUnion{GraphNode{T}}
const getParSym = symbolFrom∘indexedSymOf

function genConstFunc(::Type{T1}, val::T2) where {T1, T2}
    let res = val
        function (::T1)
            res
        end
    end
end


struct ValueNode{T, N, O} <: ContainerNode{T, N, O}
    value::ShapedMemory{ShapedMemory{T, N}, O}
    frozen::Bool
    marker::Symbol
    id::UInt

    function ValueNode(p::BaseParam{T, N}) where {T, N}
        sl = screenLevelOf(p)
        if sl == 0
            valRaw = obtainCore(p)
            frozen = false
        else
            valRaw = obtain(p)
            frozen = Bool(sl-1)
        end
        new{T, N, 0}( (ShapedMemory∘fill∘ShapedMemory)(T, valRaw), frozen, 
                      getParSym(p), objectid(p) )
    end

    function ValueNode(p::ParamLink{T, N, <:Any, O}) where {T, N, O}
        sl = screenLevelOf(p)
        if sl == 0
            valRaw = obtainCore(p)
            frozen = false
        else
            valRaw = obtain(p)
            frozen = Bool(sl-1)
        end
        new{T, N, O}( ShapedMemory( map(x->ShapedMemory(T, x), valRaw) ), frozen, 
                      getParSym(p), objectid(p) )
    end

    function ValueNode(p::PrimitiveParam{T, N}) where {T, N}
        frozen = (screenLevelOf(p) == 2)
        val = (ShapedMemory∘fill∘ShapedMemory)(T, obtain(p))
        new{T, N, 0}(val, frozen, getParSym(p), objectid(p))
    end
end

struct BatchNode{T, N, O, 
                 I<:ShapedMemory{<:DimSGNode{T, N}, O}} <: ReferenceNode{T, N, O, I}
    source::I
    marker::Symbol
    id::UInt

    function BatchNode(source::I, p::ParamGrid{T, N}) where 
                      {T, N, O, I<:AbstractArray{<:DimSGNode{T, N}, O}}
        new{T, N, O, I}(ShapedMemory(source), getParSym(p), objectid(p))
    end
end

struct BuildNode{T, N, O, F<:JaggedOperator{T, N, O}, S<:Union{ItsType, ValShifter{T}}, 
                 I<:NodeChildrenType{T}} <: OperationNode{T, N, O, I}
    operator::F
    shifter::S
    source::I
    marker::Symbol
    id::UInt

    function BuildNode(children::I, p::BaseParam{T, N}) where 
                      {T, N, I<:NodeChildrenType{T}}
        operator = p.lambda
        includeOffset = hasfield(typeof(p), :offset) && isOffsetEnabled(p)
        shifter = genValShifter(T, includeOffset ? p.offset : nothing)
        new{T, N, 0, typeof(operator), typeof(shifter), I}(operator, shifter, children, 
                                                           getParSym(p), objectid(p))
    end

    function BuildNode(children::I, p::ParamLink{T, N, <:Any, O}) where 
                      {T, N, O, I<:NodeChildrenType{T}}
        operator = p.lambda
        new{T, N, O, typeof(operator), ItsType, I}(operator, itself, children, 
                                              getParSym(p), objectid(p))
    end
end

const BuildNodeSourceNum{T, A} = 
      BuildNode{T, <:Any, <:Any, <:Any, <:Any, <:NTuple{A, GraphNode{T}}}

getBuildNodeSourceNum(::BuildNodeSourceNum{<:Any, A}) where {A} = A

struct IndexNode{T, N, I<:BuildNode{T, N}} <: ReferenceNode{T, N, 0, I}
    source::I
    index::Int
    marker::Symbol
    id::UInt

    function IndexNode(node::I, p::KnotParam{T, <:Any, <:Any, N, O}) where 
                      {T, N, O, I<:BuildNode{T, N, O}}
        new{T, N, I}(node, p.index, getParSym(p), objectid(p))
    end
end


genGraphNodeCore(p::PrimitiveParam) = ValueNode(p)
genGraphNodeCore(p::ParamFunctor) = ValueNode(p)

function genGraphNodeCore(source::NTuple{A, GraphNode{T}}, 
                          p::ParamFunctor{T, <:Any, <:ParamInput{T, A}}) where {T, A}
    BuildNode(source, p)
end

genGraphNodeCore(::Type{T}, p::CellParam{T}) where {T} = ValueNode(p)

genGraphNodeCore(val::Memory{<:DimSGNode{T, N}}, p::ParamGrid{T, N}) where {T, N} = 
BatchNode(ShapedMemory(val, p.input.shape), p)

genGraphNodeCore(val::Memory{<:GraphNode{T, N, O}}, 
                 p::KnotParam{T, <:Any, <:Any, N, O}) where {T, N, O} = 
IndexNode(getindex(val), p)

const ComputeNodeDict{T} = ParamPointerBox{ T, Dim0GNode{T}, DimSGNode{T}, 
                                             GraphNode{T}, typeof(genGraphNodeCore) }

genGraphNodeDict(::Type{T}, maxRecursion::Int=DefaultMaxParamPointerLevel) where {T} = 
ParamPointerBox(genGraphNodeCore, T, Dim0GNode{T}, DimSGNode{T}, GraphNode{T}, 
                 maxRecursion)

function genGraphNode(p::CompositeParam{T}; 
                      maxRecursion::Int=DefaultMaxParamPointerLevel) where {T}
    gNodeDict = genGraphNodeDict(T, maxRecursion)
    recursiveTransform!(genGraphNodeCore, gNodeDict, p)
end

genGraphNode(p::PrimitiveParam) = genGraphNodeCore(p)


evaluateNode(gn::ValueNode{T}) where {T} = directObtain.(gn.value)

evaluateNode(gn::BatchNode{T}) where {T} = map(evaluateNode, gn.source)

function evaluateNode(gn::BuildNode{T}) where {T}
    gn.operator( map(evaluateNode, gn.source)... ) |> gn.shifter
end

function evaluateNode(gn::IndexNode{T}) where {T}
    getindex(evaluateNode(gn.source), gn.index)
end

struct TemporaryStorage{T} <: QueryBox{Union{Tuple{T, Int}, Tuple{AbstractArray{T}, Int}}}
    d0::IdDict{UInt, Tuple{T, Int}}
    d1::IdDict{UInt, Tuple{AbstractArray{T}, Int}}

    TemporaryStorage(::Type{T}) where {T} = 
    new{T}(IdDict{UInt, Tuple{T, Int}}(), IdDict{UInt, Tuple{AbstractArray{T}, Int}}())
end

struct FixedSizeStorage{T, V<:AbstractArray{T}} <: QueryBox{Union{T, V}}
    d0::Memory{T}
    d1::Memory{V}
end

selectStorageSectorSym(::Dim0GNode) = :d0
selectStorageSectorSym(::DimSGNode) = :d1

selectInPSubset(::Val{0}, ps::AbstractVector) = first(ps)
selectInPSubset(::Val,    ps::AbstractVector) = itself(ps)

function genGetVal!(tempStorage::TemporaryStorage{T}, gn::ValueNode{T}) where {T}
    id = gn.id
    sectorSym = selectStorageSectorSym(gn)
    sector = getproperty(tempStorage, sectorSym)
    data = get(sector, id, nothing)
    if data === nothing
        idx = length(sector) + 1
        setindex!(sector, (evaluateNode(gn), idx), id)
    else
        idx = last(data)
    end
    function (storage::FixedSizeStorage{T}, ::AbtVecOfAbtArr{T})
        getindex(getproperty(storage, sectorSym), idx)
    end
end

function genGetVal!(::Type{T}, ::Val{N}, idx::Int) where {T, N}
    function (::FixedSizeStorage{T}, input::AbtVecOfAbtArr{T})
        getindex(selectInPSubset(Val(N), input), idx)
    end
end

function compressNodeCore!(tStorage::TemporaryStorage{T}, 
                           paramSet::DimParamSet{T}, gn::ValueNode{T, N, 0}) where {T, N}
    varIdx = if gn.frozen
        nothing
    else
        findfirst(par->objectid(par)==gn.id, selectInPSubset(Val(N), paramSet))
    end
    varIdx===nothing ? genGetVal!(tStorage, gn) : genGetVal!(T, Val(N), varIdx)
end

function compressNodeCore!(tStorage::TemporaryStorage{T}, 
                           paramSet::DimParamSet{T}, node::BatchNode{T}) where {T}
    fs=map(x->compressNodeCore!(tStorage, paramSet, x), node.source)
    let gs=fs, iterRanges=Iterators.product(axes(fs)...)
        function (storage::FixedSizeStorage{T}, input::AbtVecOfAbtArr{T})
            map( enumerate(iterRanges) ) do (i, _)
                gs[i](storage, input)
            end
        end
    end
end

function compressNodeCore!(tStorage::TemporaryStorage{T}, 
                           paramSet::DimParamSet{T}, node::BuildNode{T}) where {T}
    sourceNum = getBuildNodeSourceNum(node)
    compressBuildNodeCore!(Val(sourceNum), tStorage, paramSet, node)
end

function compressBuildNodeCore!(::Val{1}, tStorage::TemporaryStorage{T}, 
                                paramSet::DimParamSet{T}, node::BuildNode{T}) where {T}
    f = compressNodeCore!(tStorage, paramSet, node.source[1])
    let g=f, op=node.operator, sh=node.shifter
        function (storage::FixedSizeStorage{T}, input::AbtVecOfAbtArr{T})
            op( g(storage, input) ) |> sh
        end
    end
end

function compressBuildNodeCore!(::Val{2}, tStorage::TemporaryStorage{T}, 
                                paramSet::DimParamSet{T}, node::BuildNode{T}) where {T}
    fL, fR = compressNodeCore!.(Ref(tStorage), Ref(paramSet), node.source)
    let gL=fL, gR=fR, op=node.operator, sh=node.shifter
        function (storage::FixedSizeStorage{T}, input::AbtVecOfAbtArr{T})
            op( gL(storage, input), gR(storage, input) ) |> sh
        end
    end
end

function compressBuildNodeCore!(::Val{3}, tStorage::TemporaryStorage{T}, 
                                paramSet::DimParamSet{T}, node::BuildNode{T}) where {T}
    fL, fC, fR = compressNodeCore!.(Ref(tStorage), Ref(paramSet), node.source)
    let gL=fL, gC=fC, gR=fR, op=node.operator, sh=node.shifter
        function (storage::FixedSizeStorage{T}, input::AbtVecOfAbtArr{T})
            op( gL(storage, input), gC(storage, input), gR(storage, input) ) |> sh
        end
    end
end

function compressNodeCore!(tStorage::TemporaryStorage{T}, 
                           paramSet::DimParamSet{T}, node::IndexNode{T}) where {T}
    f = compressNodeCore!(tStorage, paramSet, node.source)
    let g=f, idx=node.index
        function (storage::FixedSizeStorage{T}, input::AbtVecOfAbtArr{T})
            getindex(g(storage, input), idx)
        end
    end
end


struct EvalGraphNode{T, F<:Function, S<:FixedSizeStorage{T}, 
                     V<:AbtVecOfAbtArr{T}} <: Evaluator{GraphNode{T}}
    f::F
    storage::S
    param::V
end

(gnf::EvalGraphNode)() = gnf(gnf.storage, gnf.param)

(gnf::EvalGraphNode{T})(ps::AbtVecOfAbtArr{T}) where {T} = gnf(gnf.storage, ps)

(gnf::EvalGraphNode)(s::FixedSizeStorage{T}, ps::AbtVecOfAbtArr{T}) where {T} = gnf.f(s, ps)

# function (gnf::EvalGraphNode{T})(ps::DimParamSet{T}; checkParamSet::Bool=true) where {T}
#     if checkParamSet
#         bl, errorMsg = isGraphParamSet(paramSet)
#         bl || throw( AssertionError(errorMsg) )
#     end
#     gnf( obtain.(ps) )
# end


function compressNodeINTERNAL(node::GraphNode{T}, paramSet::DimParamSet{T}) where {T}
    tStorage = TemporaryStorage(T)
    f = compressNodeCore!(tStorage, paramSet, node)
    sectors = map( getproperty.(Ref(tStorage), fieldnames(TemporaryStorage)) ) do data
        pairs = values(data)
        vals = first.(pairs)
        eleTypeUB = pairs |> eltype |> fieldtypes |> first
        eleType = typeintersect(eltype(vals), eleTypeUB)
        sector = Memory{eleType}(undef, length(data))
        for (idx, val) in zip(last.(pairs), vals)
            sector[idx] = val
        end
        sector
    end
    EvalGraphNode(f, FixedSizeStorage(sectors...), obtain.(paramSet))
end

function compressNode(node::GraphNode{T}, paramSet::DimParamSet{T}) where {T}
    bl, errorMsg = isGraphParamSet(paramSet)
    bl || throw( AssertionError(errorMsg) )
    compressNodeINTERNAL(node, paramSet)
end


const ParamSetErrorMessage1 = "Input argument does not meet the type requirement of being "*
                              "a parameter set: $(DimParamSet)"

const ParamSetErrorMessage2 = "All `$ElementalParam` must be inside a non-zero dimensional"*
                              " `AbstractArray` as the first element of `paramSet`."

const ParamSetErrorMessage3 = "The screen level of every input parameter (inspected by "*
                              "`$screenLevelOf`) must all be 1."

isGraphParamSet(::AbstractVector) = (false, ParamSetErrorMessage1)

function isGraphParamSet(paramSet::DimParamSet{T}) where {T}
    isInSet = true
    errorMsg = ""
    for i in eachindex(paramSet)
        p = paramSet[i]

        bl1 = i == firstindex(paramSet)
        bl2 = p isa AbstractArray{<:ElementalParam{T}} && ndims(p) > 0
        bl3 = (p isa InnerSpanParam && !(p isa ElementalParam))
        if !( ( !bl1 && bl3 ) || ( bl1 && (bl2 || bl3) ) )
            isInSet = false
            errorMsg = ParamSetErrorMessage2
            break
        end

        bl2 || (p = [p])

        for pp in p
            sl = screenLevelOf(pp)
            if sl != 1
                isInSet = false
                errorMsg = ParamSetErrorMessage3
                break
            end
        end
        isInSet || break
    end
    isInSet, errorMsg
end