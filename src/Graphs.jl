export genComputeGraph, evalGraphNode, compressGraphNode

struct FixedNode{T} <: StorageNode{T, 0}
    val::T
    marker::Symbol
end

struct IndexNode{T, O} <: StorageNode{T, O}
    var::Array{T, O}
    marker::Symbol
    idx::Int
end

const ChildNodeType{T, N} = Union{AbstractArray{<:GraphNode{T, 0}, N}, GraphNode{T, N}}

const NodeChildrenType{T} = TernaryTupleUnion{ChildNodeType{T}}


getChildNum(::Type{<:NonEmptyTuple{ChildNodeType{T}, N}}) where {T, N} = N+1

# getChildDimCore(::Type{<:AbstractArray{<:Any, N}}) where {N} = N

# getChildDimCore(::Type{<:StorageNode{<:Any, N}}) where {N} = N

# getChildDim(::Type{CT}) where {T, CT<:NonEmptyTuple{ChildNodeType{T}}} = 
# getChildDimCore.(fieldtypes(CT))

struct ReductionNode{T, I<:NodeChildrenType{T}, F} <: OperatorNode{T, 0, I, F}
    apply::TypedReduction{T, F}
    child::I
    marker::Symbol
    bias::T
end

struct ReferenceNode{T, O, I, F, G<:OperatorNode{T, O, I, F}} <: EffectNode{T, O, I}
    ref::G
    history::Array{T, O}
end


function compressGraphNode(gn::FixedNode{T}) where {T}
    (::AbtVecOfAbtArray{T})->itself.(gn.val)
end

function compressGraphNode(gn::IndexNode{T, 0}) where {T}
    (x::AbtVecOfAbtArray{T})->x[begin][gn.idx]
end

function compressGraphNode(gn::IndexNode{T}) where {T}
    (x::AbtVecOfAbtArray{T})->x[gn.idx]
end

function compressGraphNode(gn::OperatorNode{T, O, I}) where {T, O, I}
    ChildNum = getChildNum(I)
    compressGraphNodeCore(Val(ChildNum), gn)
end

function compressGraphNode(gn::ReferenceNode{T}) where {T}
    (::AbtVecOfAbtArray{T})->itself.(gn.val)
end

function makeGraphFuncComp(graphChild::AbstractArray{<:GraphNode{T, 0}}) where {T}
    fArr = map(compressGraphNode, graphChild)
    let fs=Tuple(fArr), dim=size(fArr)
        function (x::AbtVecOfAbtArray{T})
            fVals = collect(map(f->f(x), fs))
            reshape(fVals, dim)
        end
    end
end

makeGraphFuncComp(graphChild::GraphNode) = compressGraphNode(graphChild)

# Further Improved AD performance.
makeGraphFuncComp(graphChild::AbstractArray{<:GraphNode, 0}) = 
compressGraphNode(graphChild[])

# This has to be manually coded up, otherwise Enzyme (v0.12.22) will break down.
function compressGraphNodeCore(::Val{1}, node::OperatorNode{T}) where {T}
    f = makeGraphFuncComp(node.child[1])
    let apply1=f, apply2=node.apply, b=node.bias
        function (x::AbtVecOfAbtArray{T})
            # (apply2∘apply1)(x) + b # This causes Enzyme.jl (v0.12.22) to fail.
            apply2( apply1(x) ) + b
        end
    end
end

function compressGraphNodeCore(::Val{2}, node::OperatorNode{T}) where {T}
    fL, fR = makeGraphFuncComp.(node.child)
    let apply1L=fL, apply1R=fR, apply2=node.apply, b=node.bias
        function (x::AbtVecOfAbtArray{T})
            apply2( apply1L(x), apply1R(x) ) + b
        end
    end
end

function compressGraphNodeCore(::Val{3}, node::OperatorNode{T}) where {T}
    fL, fC, fR = makeGraphFuncComp.(node.child)
    let apply1L=fL, apply1C=fC, apply1R=fR, apply2=node.apply, b=node.bias
        function (x::AbtVecOfAbtArray{T})
            apply2( apply1L(x), apply1C(x), apply1R(x) ) + b
        end
    end
end

# # Failed generalization attempt due to Enzyme (v0.12.22).
# function compressGraphNodeCore(::Val{N}, node::OperatorNode{T, O, I}) where {N, T, O, I}
#     fs = makeGraphFuncComp.(node.child)
#     rT = Tuple{(ifelse(n==0, T, AbstractArray{T, n}) for n in getChildDim(I))...}
#     let apply1s=fs, apply2=node.apply, b=node.bias, argT=rT
#         function (x::AbtVecOfAbtArray{T})
#             splat(apply2)( map(f->f(x), apply1s)::argT ) + b
#         end
#     end
# end


evalGraphNode(gn::FixedNode{T}, ::AbtVecOfAbtArray{T}) where {T} = gn.val

evalGraphNode(gn::IndexNode{T, 0}, data::AbtVecOfAbtArray{T}) where {T} = data[begin][gn.idx]

evalGraphNode(gn::IndexNode{T}, data::AbtVecOfAbtArray{T}) where {T} = data[gn.idx]

function evalGraphNode(gn::ReductionNode{T}, data::AbtVecOfAbtArray{T}) where {T}
    gn.apply( broadcast(evalGraphNodeCore, gn.child, Ref(data))... ) + gn.bias
end

function evalGraphNodeCore(parInput::GraphNode{T}, data::AbtVecOfAbtArray{T}) where {T}
    evalGraphNode(parInput, data)
end

function evalGraphNodeCore(parInput::AbstractArray{<:GraphNode{T, 0}}, 
                           data::AbtVecOfAbtArray{T}) where {T}
    broadcast(evalGraphNode, parInput, Ref(data))
end

# Necessary for certain AD libraries to work properly/efficiently as of Julia 1.10.4
function evalGraphNodeCore(parInput::AbtArray0D{<:GraphNode{T, 0}}, 
                           data::AbtVecOfAbtArray{T}) where {T}
    evalGraphNode(parInput[], data)
end


const symbolFromPar = symbolFrom∘idxSymOf

fillElementalVal(::Type{T}, ele::T) where {T} = fill(ele)
fillElementalVal(::Type{T}, obj::AbstractArray{T}) where {T} = itself(obj)

selectInPSubset(::Val{0}, ps::AbstractVector{<:PrimDParSetEltype{T}}) where {T} = first(ps)
selectInPSubset(::Val,    ps::AbstractVector{<:PrimDParSetEltype{T}}) where {T} = itself(ps)

function genComputeGraphCore1(inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                              par::DimensionalParam{T, N}) where {T, N}
    idx = findfirst(Fix2(compareParamContainer, par), selectInPSubset(Val(N), inPSet))
    val = obtain(par)
    sym = symbolFromPar(par)
    idx === nothing ? FixedNode(val, sym) : IndexNode(fillElementalVal(T, val), sym, idx)
end

function genComputeGraphCore2(idDict::IdDict{ParamBox{T}, OperatorNode{T}}, 
                              inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                              par::DimensionalParam{T}) where {T}
    sl = screenLevelOf(par)

    if sl == 0
        refNode = get(idDict, par, nothing)
        if refNode === nothing
            childNodes = map(par.input) do arg
                genComputeGraphCore2(idDict, inPSet, arg)
            end
            node = ReductionNode(par.lambda, childNodes, symbolFromPar(par), par.offset)
            setindex!(idDict, node, par)
            node
        else
            ReferenceNode(refNode, fillElementalVal(T, par.memory))
        end
    elseif sl == 1
        genComputeGraphCore1(inPSet, par)
    elseif sl == 2
        FixedNode(obtain(par), symbolFromPar(par))
    else
        throwScreenLevelError(sl)
    end
end

function genComputeGraphCore2(idDict, inPSet, 
                              pars::AbstractArray{<:DimensionalParam{T}}) where {T}
    map(pars) do par
        genComputeGraphCore2(idDict, inPSet, par)
    end
end

function genComputeGraph(inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                         par::ParamBox{T}) where {T}
    idDict = IdDict{ParamBox{T}, OperatorNode{T}}()
    genComputeGraphCore2(idDict, inPSet, par)
end

genComputeGraph(inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                par::PrimitiveParam{T}) where {T} = 
genComputeGraphCore1(inPSet, par)