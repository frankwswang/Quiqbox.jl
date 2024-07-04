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
end


# function compressGraphNodeV1(tn::Quiqbox.ValueNode{T}) where {T}
#     StableReduce(Vector{T}, (::AbstractVector{T})->tn.val)
# end

# function compressGraphNodeV1(tn::Quiqbox.IndexNode{T}) where {T}
#     StableReduce(Vector{T}, (x::AbstractVector{T})->x[tn.idx])
# end

# function compressGraphNodeV1(tn::LayerNode{T}) where {T}
#     let fs=map(compressGraphNodeV1, tn.child), connect=tn.connect
#         # function (x::AbstractVector{T})
#         #     connect.f(T[f(x) for f in fs])
#         # end
#         StableReduce(Vector{T}, (x::AbstractVector{T})->connect([f(x) for f in fs]))
#     end
# end

# function compressGraphNodeV2(tn::Quiqbox.ValueNode{T}) where {T}
#     (::AbstractVector{T})->tn.val
# end

# function compressGraphNodeV2(tn::Quiqbox.IndexNode{T}) where {T}
#     (x::AbstractVector{T})->x[tn.idx]
# end

# function compressGraphNodeV2(tn::LayerNode{T}) where {T}
#     let fs=map(compressGraphNodeV2, tn.child), connect=tn.connect
#         function (x::AbstractVector{T})
#             connect(T[f(x) for f in fs])
#         end
#     end
# end

# Best version
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
    let fs=fs, dim=dim, connect=tn.connect
        function (x::AbstractVector{T})
            fVals = collect(map(fs) do f; f(x) end)
            reshape(fVals, dim) |> connect
        end
    end
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