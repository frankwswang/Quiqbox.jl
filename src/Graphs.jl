export genComputeGraph, computeTreeGraph, compressGraphNode

struct ValueNode{T} <: TreeNode{T, iT, 0}
    val::T
    marker::Symbol
end

struct IndexNode{T} <: TreeNode{T, iT, 0}
    val::T
    idx::Int
    marker::Symbol
end

struct LayerNode{T, F, N, I<:AbstractArray{<:TreeNode{T}, N}} <: TreeNode{T, F, N}
    connect::StableReduce{T, F}
    child::I
    marker::Symbol
    bias::T
end


function compressGraphNode(tn::ValueNode{T}) where {T}
    (::AbstractVector{T})->tn.val
end

function compressGraphNode(tn::IndexNode{T}) where {T}
    (x::AbstractVector{T})->x[tn.idx]
end

function compressGraphNode(tn::LayerNode{T}) where {T}
    fArr = map(compressGraphNode, tn.child)
    dim = size(fArr)
    fs = Tuple(fArr)
    let fs=fs, dim=dim, connect=tn.connect, b=tn.bias
        function (x::AbstractVector{T})
            fVals = collect(map(f->f(x), fs))
            connect( reshape(fVals, dim) ) + b
        end
    end
end


computeTreeGraph(::AbstractVector{T}, tn::ValueNode{T}) where {T} = tn.val

computeTreeGraph(data::AbstractVector{T}, tn::IndexNode{T}) where {T} = data[tn.idx]

function computeTreeGraph(data::AbstractVector{T}, ln::LayerNode{T}) where {T}
    ln.connect( broadcast(computeTreeGraph, Ref(data), ln.child) ) + ln.bias
end

# This is necessary for certain AD library to work properly/efficiently as of Julia 1.10
function computeTreeGraph(data::AbstractVector{T}, ln::LayerNode{T, <:Any, 0}) where {T}
    ln.connect( computeTreeGraph(data, ln.child[]) ) + ln.bias
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
        LayerNode(par.lambda, childNodes, symbolFromPar(par), par.offset[])
    elseif sl == 1
        genComputeGraphCore(parSet, par)
    elseif sl == 2
        ValueNode(valOf(par), symbolFromPar(par))
    else
        throwScreenLevelError(sl)
    end
end