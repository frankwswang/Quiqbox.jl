struct InnerNode{T} <: TreeNode{T, iT, 0}
    idx::IntOrNone
    val::T
end

struct OuterNode{T, F, N, I<:AbstractArray{<:TreeNode{T}, N}} <: TreeNode{T, F, N}
    connect::StableReduce{T, F}
    child::I
end


function computeTreeGraph(data::AbstractVector{T}, tn::InnerNode{T}) where {T}
   tn.idx === nothing ? tn.val : data[tn.idx]
end

function computeTreeGraph(data::AbstractVector{T}, rn::OuterNode{T}) where {T}
   strictVecBinaryOp(computeTreeGraph, Ref(data), rn.child) |> rn.connect
end

function genComputeGraphCore(parSet, par::SingleParam{T}) where {T}
    InnerNode(findfirst(Fix2(compareParamContainer, par), parSet), valOf(par))
end

genComputeGraph(parSet, par::SinglePrimP{T}) where {T} = 
genComputeGraphCore(parSet, par)

function genComputeGraph(parSet, pn::ParamNode{T}) where {T}
    sl = screenLevelOf(pn)

    if sl == 0
        OuterNode(pn.lambda, strictVecBinaryOp(genComputeGraph, Ref(parSet), pn|>dataOf))
    elseif sl == 1
        genComputeGraphCore(parSet, pn)
    elseif sl == 2
        InnerNode(nothing, valOf(pn))
    else
        throwScreenLevelError(sl)
    end
end