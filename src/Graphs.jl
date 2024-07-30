export genGraphNode, evaluateNode, compressNode

using Base: Fix2

const NodeChildrenType{T} = TernaryNTupleUnion{GraphNode{T}}
const getParSym = symbolFrom∘indexedSymOf

function genConstFunc(::Type{T1}, val::T2) where {T1, T2}
    let res = val
        function (::T1)
            res
        end
    end
end


struct ValueNode{T, N} <: ContainerNode{T, N, 0}
    value::ShapedMemory{T, N}
    frozen::Bool
    marker::Symbol
    id::UInt

    function ValueNode(p::ParamToken{T, N}) where {T, N}
        sl = screenLevelOf(p)
        if sl == 0
            val = ShapedMemory(T, directObtain(p.memory))
            frozen = false
        else
            val = ShapedMemory(T, obtain(p))
            frozen = Bool(sl-1)
        end
        new{T, N}(val, frozen, getParSym(p), objectid(p))
    end

    function ValueNode(p::PrimitiveParam{T, N}) where {T, N}
        frozen = (screenLevelOf(p) == 2)
        val = ShapedMemory(T, obtain(p))
        new{T, N}(val, frozen, getParSym(p), objectid(p))
    end
end

struct BatchNode{T, N, O, I<:ShapedMemory{<:DimSGNode{T, N}, O}} <: ReferenceNode{T, N, O, I}
    source::I
    marker::Symbol
    id::UInt

    function BatchNode(source::I, p::ParamGrid{T, N}) where 
                      {T, N, O, I<:AbstractArray{<:DimSGNode{T, N}, O}}
        new{T, N, O, I}(ShapedMemory(source), getParSym(p), objectid(p))
    end
end

struct BuildNode{T, N, O, F<:ParamOperator{T, <:Any, N, O}, S<:Union{iT, ValShifter{T}}, 
                 I<:NodeChildrenType{T}} <: OperationNode{T, N, O, I}
    operator::F
    shifter::S
    source::I
    marker::Symbol
    id::UInt

    function BuildNode(children::I, p::ParamToken{T, N}) where 
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
        new{T, N, O, typeof(operator), iT, I}(operator, itself, children, 
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

    function IndexNode(node::I, p::NodeParam{T, <:Any, <:Any, N, O}) where 
                      {T, N, O, I<:BuildNode{T, N, O}}
        new{T, N, I}(node, p.index, getParSym(p), objectid(p))
    end
end

const ComputeNodeDict{T} = ParamPointerDict{T, Dim0GNode{T}, DimSGNode{T}, GraphNode{T}}

genGraphNodeDict(::Type{T}) where {T} = 
ParamPointerDict(T, Dim0GNode{T}, DimSGNode{T}, GraphNode{T})::ComputeNodeDict{T}

genGraphNodeCore(p::ParamFunctor) = ValueNode(p)
genGraphNodeCore(p::PrimitiveParam) = ValueNode(p)

function genGraphNodeCore(source::NTuple{A, GraphNode{T}}, 
                            p::ParamToken{T, <:Any, <:ParamInput{T, A}}) where {T, A}
    BuildNode(source, p)
end

genGraphNodeCore(::Type{T}, p::CellParam{T}) where {T} = ValueNode(p)

genGraphNodeCore(val::Memory{<:DimSGNode{T, N}}, p::ParamGrid{T, N}) where {T, N} = 
BatchNode(ShapedMemory(val, p.input.shape), p)

genGraphNodeCore(val::Memory{<:GraphNode{T, N, O}}, 
                 p::NodeParam{T, <:Any, <:Any, N, O}) where {T, N, O} = 
IndexNode(getindex(val), p)

function genGraphNode(p::CompositeParam{T}) where {T}
    gNodeDict = genGraphNodeDict(T)
    recursiveTransform!(genGraphNodeCore, gNodeDict, p)
end

genGraphNode(p::PrimitiveParam) = genGraphNodeCore(p)

evaluateNode(gn::ValueNode{T}) where {T} = directObtain(gn.value)

evaluateNode(gn::BatchNode{T}) where {T} = map(evaluateNode, gn.source)

function evaluateNode(gn::BuildNode{T}) where {T}
    gn.operator( map(evaluateNode, gn.source)... ) |> gn.shifter
end

function evaluateNode(gn::IndexNode{T}) where {T}
    getindex(evaluateNode(gn.source), gn.index)
end

struct TemporaryStorage{T} <: ValueStorage{T}
    d0::IdDict{UInt, Tuple{T, Int}}
    d1::IdDict{UInt, Tuple{AbstractArray{T}, Int}}

    TemporaryStorage(::Type{T}) where {T} = 
    new{T}(IdDict{UInt, Tuple{T, Int}}(), IdDict{UInt, Tuple{AbstractArray{T}, Int}}())
end

struct FixedSizeStorage{T, V<:AbstractArray{T}} <: ValueStorage{T}
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
        setindex!(sector, (directObtain(gn.value), idx), id)
    else
        idx = last(data)
    end
    function (storage::FixedSizeStorage{T}, ::AbtVecOfAbtArray{T})
        getindex(getproperty(storage, sectorSym), idx)
    end
end

function genGetVal!(::Type{T}, ::Val{N}, idx::Int) where {T, N}
    function (::FixedSizeStorage{T}, input::AbtVecOfAbtArray{T})
        getindex(selectInPSubset(Val(N), input), idx)
    end
end

function compressNodeCore!(tStorage::TemporaryStorage{T}, 
                           inParSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                           gn::ValueNode{T, N}) where {T, N}
    varIdx = if gn.frozen
        nothing
    else
        findfirst(par->objectid(par)==gn.id, selectInPSubset(Val(N), inParSet))
    end
    varIdx===nothing ? genGetVal!(tStorage, gn) : genGetVal!(T, Val(N), varIdx)
end

function compressNodeCore!(tStorage::TemporaryStorage{T}, 
                           inParSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                           gn::BatchNode{T}) where {T}
    let fs=map(x->compressNodeCore!(tStorage, inParSet, x), gn.source)
        function (storage::FixedSizeStorage{T}, input::AbtVecOfAbtArray{T})
            map(f->f(storage, input), fs)
        end
    end
end

function compressNodeCore!(tStorage::TemporaryStorage{T}, 
                           inParSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                           gn::BuildNode{T}) where {T}
    sourceNum = getBuildNodeSourceNum(gn)
    compressBuildNodeCore!(Val(sourceNum), tStorage, inParSet, gn)
end

function compressBuildNodeCore!(::Val{1}, tStorage::TemporaryStorage{T}, 
                                inParSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                                node::BuildNode{T}) where {T}
    f = compressNodeCore!(tStorage, inParSet, node.source[1])
    let apply=f, operator=node.operator, shifter=node.shifter
        function (storage::FixedSizeStorage{T}, input::AbtVecOfAbtArray{T})
            operator( apply(storage, input) ) |> shifter
        end
    end
end

function compressBuildNodeCore!(::Val{2}, tStorage::TemporaryStorage{T}, 
                                inParSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                                node::BuildNode{T}) where {T}
    fL, fR = compressNodeCore!.(Ref(tStorage), Ref(inParSet), node.source)
    let applyL=fL, applyR=fR, operator=node.operator, shifter=node.shifter
        function (storage::FixedSizeStorage{T}, input::AbtVecOfAbtArray{T})
            operator( applyL(storage, input), applyR(storage, input) ) |> shifter
        end
    end
end

function compressBuildNodeCore!(::Val{3}, tStorage::TemporaryStorage{T}, 
                                inParSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                                node::BuildNode{T}) where {T}
    fL, fC, fR = compressNodeCore!.(Ref(tStorage), Ref(inParSet), node.source)
    let applyL=fL, applyC=fC, applyR=fR, operator=node.operator, shifter=node.shifter
        function (storage::FixedSizeStorage{T}, input::AbtVecOfAbtArray{T})
            operator( applyL(storage, input), applyC(storage, input), 
                      applyR(storage, input) ) |> shifter
        end
    end
end

function compressNodeCore!(tStorage::TemporaryStorage{T}, 
                           inParSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                           node::IndexNode{T}) where {T}
    let apply=compressNodeCore!(tStorage, inParSet, node.source), idx=node.index
        function (storage::FixedSizeStorage{T}, input::AbtVecOfAbtArray{T})
            getindex(apply(storage, input), idx)
        end
    end
end

function compressNodeINTERNAL(node::GraphNode{T}, 
                              inParSet::AbstractVector{<:PrimDParSetEltype{T}}) where {T}
    tStorage = TemporaryStorage(T)
    f = compressNodeCore!(tStorage, inParSet, node)
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
    f, FixedSizeStorage(sectors...)
end

const ParamSetErrorMessage1 = "`$ElementalParam` must all be enclosed in a single array "*
                              "as the first element of `inputParamSet`."

const ParamSetErrorMessage2 = "The screen level of the input parameter (inspected by "*
                              "`$screenLevelOf`) for a dependent parameter must be 1."

function compressNode(node::GraphNode{T}, 
                      inParSet::AbstractVector{<:PrimDParSetEltype{T}}) where {T}
    for i in eachindex(inParSet)
        p = inParSet[i]

        bl1 = i == firstindex(inParSet)
        bl2 = p isa AbstractArray{<:ElementalParam{T}}
        bl3 = (p isa DoubleDimParam && !(p isa ElementalParam))
        if !( ( !bl1 && bl3 ) || ( bl1 && (bl2 || bl3) ) )
            throw(AssertionError(ParamSetErrorMessage1))
        end

        bl2 || (p = [p])

        for pp in p
            sl = screenLevelOf(pp)
            if sl != 1
                throw(DomainError((pp, sl), ParamSetErrorMessage2))
            end
        end
    end

    compressNodeINTERNAL(node, inParSet)
end