export genComputeGraph, evalGraphNode, compressGraphNode

struct FixedNode{T} <: StorageNode{T, 0}
    val::T
    marker::Symbol
end

struct IndexNode{T, N} <: StorageNode{T, N}
    var::Array{T, N}
    marker::Symbol
    idx::Int
end

const ChildNodeType{T, N} = Union{AbstractArray{<:GraphNode{T, 0}, N}, GraphNode{T, N}}

const NodeChildrenType{T} = Union{NonEmptyTuple{ChildNodeType{T}, 0}, NonEmptyTuple{ChildNodeType{T}, 1}}

getChildNum(::Type{<:NonEmptyTuple{ChildNodeType{T}, 0}}) where {T} = 1
getChildNum(::Type{<:NonEmptyTuple{ChildNodeType{T}, 1}}) where {T} = 2

struct ReductionNode{T, I<:NodeChildrenType{T}, F} <: OperatorNode{T, 0, I, F}
    apply::TypedReduction{T, F}
    child::I
    marker::Symbol
    bias::T
end

struct ReferenceNode{T, N, I, F, G<:OperatorNode{T, N, I, F}} <: OperatorNode{T, N, I, F}
    ref::G
    history::Array{T, N}
end


function compressGraphNode(gn::FixedNode{T}) where {T}
    (::AbstractVector{<:AbstractArray{T}})->gn.val
end

function compressGraphNode(gn::IndexNode{T, 0}) where {T}
    (x::AbstractVector{<:AbstractArray{T}})->x[begin][gn.idx]
end

function compressGraphNode(gn::IndexNode{T}) where {T}
    (x::AbstractVector{<:AbstractArray{T}})->x[gn.idx]
end

function compressGraphNode(gn::OperatorNode{T, N, I}) where {T, N, I}
    ChildNum = getChildNum(I)
    compressGraphNodeCore(Val(ChildNum), gn)
end

function compressGraphNodeCore(::Val{1}, gn::OperatorNode{T}) where {T}
    fArr = map(compressGraphNode, gn.child[1])
    let fs=Tuple(fArr), dim=size(fArr), apply=gn.apply, b=gn.bias
        function (x::AbstractVector{<:AbstractArray{T}})
            fVals = collect(map(f->f(x), fs))
            apply( reshape(fVals, dim) ) + b
        end
    end
end

function compressGraphNodeCore(::Val{2}, gn::OperatorNode{T}) where {T}
    fArr1, fArr2 = map.(compressGraphNode, gn.child)
    let fs1=Tuple(fArr1), dim1=size(fArr1), apply=gn.apply, b=gn.bias, 
        fs2=Tuple(fArr2), dim2=size(fArr2)
        function (x::AbstractVector{<:AbstractArray{T}})
            fVals1 = collect(map(f->f(x), fs1))
            fVals2 = collect(map(f->f(x), fs2))
            apply( reshape(fVals1, dim1), reshape(fVals2, dim2) ) + b
        end
    end
end


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

# Necessary for certain AD library to work properly/efficiently as of Julia 1.10.4
function evalGraphNodeCore(parInput::AbtArray0D{<:GraphNode{T, 0}}, 
                           data::AbtVecOfAbtArray{T}) where {T}
    evalGraphNode(parInput[], data)
end


const symbolFromPar = symbolFrom∘idxSymOf

fillElementalVal(::Type{T}, ele::T) where {T} = fill(ele)
fillElementalVal(::Type{T}, obj::AbstractArray{T}) where {T} = itself(obj)

selectInPSubset(::Val{0}, ps::AbstractVector{<:PrimDimParVecEle{T}}) where {T} = first(ps)
selectInPSubset(::Val,    ps::AbstractVector{<:PrimDimParVecEle{T}}) where {T} = itself(ps)

function genComputeGraphCore1(inPSet::AbstractVector{<:PrimDimParVecEle{T}}, 
                              par::DimensionalParam{T, N}) where {T, N}
    idx = findfirst(Fix2(compareParamContainer, par), selectInPSubset(Val(N), inPSet))
    if idx === nothing
        FixedNode(obtain(par), symbolFromPar(par))
    else
        IndexNode(fillElementalVal(T, obtain(par)), symbolFromPar(par), idx)
    end
end

function genComputeGraphCore2(idDict::IdDict{ParamBox{T}, OperatorNode{T}}, 
                              inPSet::AbstractVector{<:PrimDimParVecEle{T}}, 
                              par::DimensionalParam{T}) where {T}
    sl = screenLevelOf(par)

    if sl == 0
        refNode = get(idDict, par, nothing)
        if refNode === nothing
            childNodes = map(par.input) do arg
                map(arg) do i
                    genComputeGraphCore2(idDict, inPSet, i)
                end
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

genComputeGraph(inPSet::AbstractVector{<:PrimDimParVecEle{T}}, par::PrimitiveParam{T}) where {T} = 
genComputeGraphCore1(inPSet, par)

function genComputeGraph(inPSet::AbstractVector{<:PrimDimParVecEle{T}}, par::ParamBox{T}) where {T}
    idDict = IdDict{ParamBox{T}, OperatorNode{T}}()
    genComputeGraphCore2(idDict, inPSet, par)
end