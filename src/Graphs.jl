export genComputeGraph, evalGraphNode, compressGraphNode

struct FixedNode{T} <: GraphNode{T, iT, 0}
    val::T
    marker::Symbol
end

struct IndexNode{T} <: GraphNode{T, iT, 0}
    var::RefVal{T}
    marker::Symbol
    idx::Int
end

struct LayerNode{T, F, N, I<:AbstractArray{<:GraphNode{T}, N}} <: GraphNode{T, F, N}
    connect::StableReduce{T, F}
    child::I
    marker::Symbol
    bias::T
end


function compressGraphNode(gn::FixedNode{T}) where {T}
    (::AbstractVector{T})->gn.val
end

function compressGraphNode(gn::IndexNode{T}) where {T}
    (x::AbstractVector{T})->x[gn.idx]
end

function compressGraphNode(gn::LayerNode{T}) where {T}
    fArr = map(compressGraphNode, gn.child)
    let fs=Tuple(fArr), dim=size(fArr), connect=gn.connect, b=gn.bias
        function (x::AbstractVector{T})
            fVals = collect(map(f->f(x), fs))
            connect( reshape(fVals, dim) ) + b
        end
    end
end


evalGraphNode(gn::FixedNode{T}, ::AbstractVector{T}) where {T} = gn.val

evalGraphNode(gn::IndexNode{T}, data::AbstractVector{T}) where {T} = data[gn.idx]

function evalGraphNode(gn::LayerNode{T}, data::AbstractVector{T}) where {T}
    gn.connect( broadcast(evalGraphNode, gn.child, Ref(data)) ) + gn.bias
end

# Necessary for certain AD library to work properly/efficiently as of Julia 1.10.4
function evalGraphNode(gn::LayerNode{T, <:Any, 0}, data::AbstractVector{T}) where {T}
    gn.connect( evalGraphNode(gn.child[], data) ) + gn.bias
end

const symbolFromPar = symbolFrom∘idxSymOf

function genComputeGraphCore(parSet, par::SingleParam{T}) where {T}
    idx = findfirst(Fix2(compareParamContainer, par), parSet)
    if idx === nothing
        FixedNode(obtain(par), symbolFromPar(par))
    else
        IndexNode(Ref(obtain(par)), symbolFromPar(par), idx)
    end
end

genComputeGraph(parSet, par::SinglePrimP{T}) where {T} = 
genComputeGraphCore(parSet, par)

function genComputeGraph(parSet, par::ParamNode{T}) where {T}
    sl = screenLevelOf(par)

    if sl == 0
        childNodes = map(par.input) do i
            genComputeGraph(parSet, i)
        end
        LayerNode(par.lambda, childNodes, symbolFromPar(par), par.offset[])
    elseif sl == 1
        genComputeGraphCore(parSet, par)
    elseif sl == 2
        FixedNode(obtain(par), symbolFromPar(par))
    else
        throwScreenLevelError(sl)
    end
end