mutable struct ValueNode{T} <: TreeNode{T, iT, 0}
    val::T
    marker::Symbol
end

mutable struct IndexNode{T} <: TreeNode{T, iT, 0}
    val::T
    idx::Int
    marker::Symbol
end

struct LayerNode{T, F, N, I<:AbstractArray{<:TreeNode{T}, N}} <: TreeNode{T, F, N}
    connect::StableReduce{T, F}
    child::I
    marker::Symbol
end

computeTreeGraph(::AbstractVector{T}, tn::ValueNode{T}) where {T} = tn.val

computeTreeGraph(data::AbstractVector{T}, tn::IndexNode{T}) where {T} = data[tn.idx]

function computeTreeGraph(data::AbstractVector{T}, rn::LayerNode{T}) where {T}
    broadcast(computeTreeGraph, Ref(data), rn.child) |> rn.connect
end

# This is necessary for certain AD library to work properly/efficiently as of Julia 1.10
function computeTreeGraph(data::AbstractVector{T}, rn::LayerNode{T, <:Any, 0}) where {T}
    computeTreeGraph(data, rn.child[]) |> rn.connect
end

const symbolFromPar = symbolFrom∘idxSymOf

function genComputeGraphCore(parSet, par::SingleParam{T}) where {T}
    idx = findfirst(Fix2(compareParamContainer, par), parSet)
    if idx === nothing
        ValueNode(valOf(par), symbolFromPar(par))
    else
        IndexNode(valOf(par), idx, symbolFromPar(par))
    end
end

genComputeGraph(parSet, par::SinglePrimP{T}) where {T} = 
genComputeGraphCore(parSet, par)

function genComputeGraph(parSet, par::ParamNode{T}) where {T}
    sl = screenLevelOf(par)

    if sl == 0
        childNodes = map(dataOf(par)) do i
            genComputeGraph(parSet, i)
        end
        LayerNode(par.lambda, childNodes, symbolFromPar(par))
    elseif sl == 1
        genComputeGraphCore(parSet, par)
    elseif sl == 2
        ValueNode(valOf(par), symbolFromPar(par))
    else
        throwScreenLevelError(sl)
    end
end