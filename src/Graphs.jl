export genGraphNode, evaluateNode

using Base: Fix2

const NodeChildrenType{T} = TernaryNTupleUnion{GraphNode{T}}
const getParSym = symbolFrom∘indexedSymOf

getChildNum(::Type{<:NonEmptyTuple{GraphNode{T}, N}}) where {T, N} = N+1


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
end

struct FixedSizeStorage{T, V<:AbstractArray{T}} <: ValueStorage{T}
    d0::Memory{T}
    d1::Memory{V}
end

selectStorageSectorSym(::Dim0GNode) = :d0
selectStorageSectorSym(::DimSGNode) = :d1

selectInPSubset(::Val{0}, ps::AbstractVector{<:PrimDParSetEltype{T}}) where {T} = first(ps)
selectInPSubset(::Val,    ps::AbstractVector{<:PrimDParSetEltype{T}}) where {T} = itself(ps)

function genGetVal(tempStorage::TemporaryStorage{T}, gn::ValueNode{T}) where {T}
    id = gn.id
    sectorSym = selectStorageSectorSym(gn)
    sector = getproperty(tempStorage, sectorSym)
    data = get(sector, id, nothing)
    if data === nothing
        idx = length(sector) + 1
        setindex!(sector, (gn.value, idx), id)
    else
        idx = last(data)
    end
    function (storage::FixedSizeStorage{T}, ::AbtVecOfAbtArray{T}) where {T}
        getindex(getproperty(storage, sectorSym), idx)
    end
end

function genGetVal(::Type{T}, ::Val{N}, idx::Int) where {T, N}
    function (::FixedSizeStorage{T}, input::AbtVecOfAbtArray{T})
        getindex(selectInPSubset(Val(N), input), idx)
    end
end

function compressNodeCore(tempStorage::TemporaryStorage{T}, 
                          inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                          gn::ValueNode{T, N}) where {T, N}
    varIdx = if gn.frozen
        nothing
    else
        findfirst(par->objectid(par)==gn.id, selectInPSubset(Val(N), inPSet))
    end
    varIdx===nothing ? genGetVal(tempStorage, gn) : genGetVal(T, Val(N), varIdx)
end

# function compressGraphNode(gn::GridNode{T}) where {T}
#     let fs=map(compressGraphNode, gn.node)
#         (x::AbtVecOfAbtArray{T}) -> collect( map(x->f(x), fs) )
#     end
# end

# function compressGraphNode(gn::ViewNode{T, N, O}) where {T}
#     (::AbtVecOfAbtArray{T}, x::BiAbtArray{T}) -> x[gn.index]
# end

# function compressGraphNode(gn::ViewNode{T, 0, O}) where {T}
#     (::AbtVecOfAbtArray{T}, x::AbstractArray{T}) -> x[gn.index]
# end

# function compressGraphNode(gn::OperatorNode{T, N, I}) where {T, N, I}
#     ChildNum = getChildNum(I)
#     compressGraphNodeCore(Val(ChildNum), gn)
# end

# function compressGraphNode(gn::ReferenceNode{T}) where {T}
#     genConstFunc(AbtVecOfAbtArray{T}, deepcopy( itself.(gn.val) ))
# end

# function compressGraphNodeCore(::Val{1}, node::OperatorNode{T}) where {T}
#     f1 = compressGraphNode(node.child[1])
#     f2 = hasfield(typeof(node), :shift) ? node.shift : itself
#     let apply1=f1, apply2=node.apply, apply3=f2
#         function (x::AbtVecOfAbtArray{T})
#             x |> apply1 |> apply2 |> apply3
#         end
#     end
# end

# function compressGraphNodeCore(::Val{2}, node::OperatorNode{T}) where {T}
#     fL, fR = compressGraphNode.(node.child)
#     f2 = hasfield(typeof(node), :shift) ? node.shift : itself
#     let apply1L=fL, apply1R=fR, apply2=node.apply, apply3=f2
#         function (x::AbtVecOfAbtArray{T})
#             apply2( apply1L(x), apply1R(x) ) |> apply3
#         end
#     end
# end

# function compressGraphNodeCore(::Val{3}, node::OperatorNode{T}) where {T}
#     fL, fC, fR = compressGraphNode.(node.child)
#     f2 = hasfield(typeof(node), :shift) ? node.shift : itself
#     let apply1L=fL, apply1C=fC, apply1R=fR, apply2=node.apply, apply3=f2
#         function (x::AbtVecOfAbtArray{T})
#             apply2( apply1L(x), apply1C(x), apply1R(x) ) |> apply3
#         end
#     end
# end

#####################################################################################


# selectInPSubset(::Val{0}, ps::AbstractVector{<:PrimDParSetEltype{T}}) where {T} = first(ps)
# selectInPSubset(::Val,    ps::AbstractVector{<:PrimDParSetEltype{T}}) where {T} = itself(ps)

# function genOperatorNode(childNodes::NTuple{A, GraphNode{T}}, 
#                          par::CellParam{T, <:Any, <:ParamInput{T, A}}) where {T, A}
#     offset = (isOffsetEnabled(par) ? par.offset : nothing)
#     ReductionNode(par.lambda, childNodes, getParSym(par), offset)
# end

# genOperatorNode(childNodes::NTuple{A, GraphNode{T}}, 
#                 par::GridParam{T, <:Any, <:ParamInput{T, A}}) where {T, A} = 
# MorphismNode(par.lambda, childNodes, getParSym(par))

# genOperatorNode(childNodes::NTuple{A, GraphNode{T}}, 
#                 par::ParamMesh{T, <:Any, <:ParamInput{T, A}}) where {T, A} = 
# LinkageNode(par.lambda, childNodes, getParSym(par))

# genOperatorNode(sourceNode::LinkageNode{T, F, <:NTuple{A, GraphNode{T}}, N, O}, 
#                 par::NodeParam{T, F, <:NTuple{A, DoubleDimParam{T}}, N, O}) where 
#                {T, A, F, N, O} = 
# ViewNode(sourceNode, par.index)

# genOperatorNode(childNodes::AbstractArray{GraphNode{T, N, 0}}, 
#                 ::ParamGrid{T, N}) where {T, N} = 
# GridNode(childNodes|>ShapedMemory)

# function genOperatorNode(inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
#                          par::DoubleDimParam{T, N}) where {T, N}
#     sl = checkScreenLevel(screenLevelOf(par), (1, 2))
#     sym = getParSym(par)
#     val = obtain(par)
#     idx = if sl == 2
#         nothing
#     else
#         findfirst(Fix2(compareParamContainer, par), selectInPSubset(Val(N), inPSet))
#     end
#     if idx === nothing
#         ValueNode(Val(N), val, sym, Bool(sl-1))
#     else
#         IndexNode(Val(N), val, sym, idx)
#     end
# end

# function genComputeGraphCore2(idDict::IdDict{ParamToken{T}, NodeMarker{<:GraphNode{T}}}, 
#                               inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
#                               par::DoubleDimParam{T, N}) where {T, N}
#     sl = checkScreenLevel(screenLevelOf(par), getScreenLevelOptions(par))

#     if sl == 0
#         # Depth-first search by recursive calling
#         marker = get!(idDict, par, NodeMarker(ReferenceNode(par), GraphNode{T, N}))
#         if !marker.visited
#             marker.visited = true
#             childNodes = map(par.input) do arg
#                 genComputeGraphCore2(idDict, inPSet, arg)
#             end
#             res = genOperatorNode(par, childNodes)
#             marker.value = res
#         else
#             marker.value
#         end
#     else
#         genComputeGraphCore1(inPSet, par)
#     end
# end

# function genComputeGraphCore2(idDict, inPSet, 
#                               pars::AbstractArray{<:DoubleDimParam{T}}) where {T}
#     map(pars) do par
#         genComputeGraphCore2(idDict, inPSet, par)
#     end
# end

# function genComputeGraphINTERNAL(inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
#                                  par::ParamToken{T}) where {T}
#     idDict = IdDict{ParamToken{T}, NodeMarker{<:GraphNode{T}}}()
#     genComputeGraphCore2(idDict, inPSet, par)
# end

# function genComputeGraphINTERNAL(inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
#                                  par::PrimitiveParam{T}) where {T}
#     genComputeGraphCore1(inPSet, par)
# end


# const ParamSetErrorMessage1 = "`$ElementalParam` must all be enclosed in a single array "*
#                               "as the first element of `inputParamSet`."

# const ParamSetErrorMessage2 = "The screen level of the input parameter (inspected by "*
#                               "`$screenLevelOf`) for a dependent parameter must be 1."

# function genComputeGraph(inputParamSet::AbstractVector{<:PrimDParSetEltype{T}}, 
#                          par::DoubleDimParam{T}) where {T}
#     for i in eachindex(inputParamSet)
#         p = inputParamSet[i]

#         bl1 = i == firstindex(inputParamSet)
#         bl2 = p isa AbstractArray{<:ElementalParam{T}}
#         bl3 = (p isa DoubleDimParam && !(p isa ElementalParam))
#         if !( ( !bl1 && bl3 ) || ( bl1 && (bl2 || bl3) ) )
#             throw(AssertionError(ParamSetErrorMessage1))
#         end

#         bl2 || (p = [p])

#         for pp in p
#             sl = screenLevelOf(pp)
#             if sl != 1
#                 throw(DomainError((pp, sl), ParamSetErrorMessage2))
#             end
#         end
#     end

#     genComputeGraphINTERNAL(inputParamSet, par)
# end




# # function compressGraphNodeCore(::Val{N}, node::OperatorNode{T, N, I}) where {N, T, N, I}
# #     fs = compressGraphNode.(node.child)
# #     rT = Tuple{(ifelse(n==0, T, AbstractArray{T, n}) for n in getChildDim(I))...}
# #     let apply1s=fs, apply2=node.apply, b=node.shift, argT=rT
# #         function (x::AbtVecOfAbtArray{T})
# #             splat(apply2)( map(f->f(x), apply1s)::argT ) |> b
# #         end
# #     end
# # end