export genComputeGraph, evalGraphNode, compressGraphNode

const ChildNodeType{T, N} = Union{AbstractArray{<:GraphNode{T, 0}, N}, GraphNode{T, N}}
const NodeChildrenType{T} = TernaryTupleUnion{ChildNodeType{T}}


getChildNum(::Type{<:NonEmptyTuple{ChildNodeType{T}, N}}) where {T, N} = N+1

fillElementalVal(::Val{0}, obj::Any) = fill(obj)
fillElementalVal(::Val,    obj::AbstractArray) = itself(obj)

unpackAA0D(::Type{T}, obj::AbtArray0D{<:T}) where {T} = obj[]
unpackAA0D(::Type{T}, obj::AbstractArray{<:T}) where {T} = itself(obj)


struct FixedNode{T, N} <: StorageNode{T, N}
    val::Array{T, N}
    marker::Symbol
end

struct IndexNode{T, N} <: StorageNode{T, N}
    val::Array{T, N}
    marker::Symbol
    idx::Int
end

struct ReferenceNode{T, N} <: StorageNode{T, N}
    id::UInt
    val::Array{T, N}

    ReferenceNode(pb::ParamBox{T, N}) where {T, N} = 
    new{T, N}(objectid(pb), fillElementalVal(Val(N), pb.memory))
end

struct EmptyNode{T, N} <: GraphNode{T, N} end

EmptyNode(::DimensionalParam{T, N}) where {T, N} = EmptyNode{T, N}()

struct ReductionNode{T, I<:NodeChildrenType{T}, F} <: OperatorNode{T, 0, I, F}
    apply::TypedReduction{T, F}
    child::I
    marker::Symbol
    bias::T
end

struct MorphismNode{T, I<:NodeChildrenType{T}, F, N} <: OperatorNode{T, N, I, F}
    apply::StableMorphism{T, F, N}
    child::I
    marker::Symbol
end

genOperatorNode(par::NodeParam{T, <:Any, <:NTuple{A, ParamBoxSingleArg{T}}}, 
                childNodes::NTuple{A, ChildNodeType{T}}) where {A, T} = 
ReductionNode(par.lambda, childNodes, symbolFromPar(par), par.offset)

genOperatorNode(par::ArrayParam{T, <:Any, <:NTuple{A, ParamBoxSingleArg{T}}}, 
                childNodes::NTuple{A, ChildNodeType{T}}) where {A, T} = 
MorphismNode(par.lambda, childNodes, symbolFromPar(par))


function compressGraphNode(gn::FixedNode{T}) where {T}
    (::AbtVecOfAbtArray{T})->unpackAA0D(T, gn.val)
end

function compressGraphNode(gn::IndexNode{T, 0}) where {T}
    (x::AbtVecOfAbtArray{T})->x[begin][gn.idx]
end

function compressGraphNode(gn::IndexNode{T}) where {T}
    (x::AbtVecOfAbtArray{T})->x[gn.idx]
end

function compressGraphNode(::EmptyNode{T, N}) where {T, N}
    val = reshape( fill( T() ), ntuple(_->1, Val(N)) )
    (::AbtVecOfAbtArray{T})->unpackAA0D(T, val)
end

function compressGraphNode(gn::OperatorNode{T, N, I}) where {T, N, I}
    ChildNum = getChildNum(I)
    compressGraphNodeCore(Val(ChildNum), gn)
end

function compressGraphNode(gn::ReferenceNode{T}) where {T}
    (::AbtVecOfAbtArray{T})->unpackAA0D(T, gn.val)
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
    f1 = makeGraphFuncComp(node.child[1])
    f2 = hasfield(typeof(node), :bias) ? Fix2(+, node.bias) : itself
    let apply1=f1, apply2=node.apply, apply3=f2
        function (x::AbtVecOfAbtArray{T})
            x |> apply1 |> apply2 |> apply3
        end
    end
end

function compressGraphNodeCore(::Val{2}, node::OperatorNode{T}) where {T}
    fL, fR = makeGraphFuncComp.(node.child)
    f2 = hasfield(typeof(node), :bias) ? Fix2(+, node.bias) : itself
    let apply1L=fL, apply1R=fR, apply2=node.apply, apply3=f2
        function (x::AbtVecOfAbtArray{T})
            apply2( apply1L(x), apply1R(x) ) |> apply3
        end
    end
end

function compressGraphNodeCore(::Val{3}, node::OperatorNode{T}) where {T}
    fL, fC, fR = makeGraphFuncComp.(node.child)
    f2 = hasfield(typeof(node), :bias) ? Fix2(+, node.bias) : itself
    let apply1L=fL, apply1C=fC, apply1R=fR, apply2=node.apply, apply3=f2
        function (x::AbtVecOfAbtArray{T})
            apply2( apply1L(x), apply1C(x), apply1R(x) ) |> apply3
        end
    end
end

# # Failed generalization attempt due to Enzyme (v0.12.22).
# function compressGraphNodeCore(::Val{N}, node::OperatorNode{T, N, I}) where {N, T, N, I}
#     fs = makeGraphFuncComp.(node.child)
#     rT = Tuple{(ifelse(n==0, T, AbstractArray{T, n}) for n in getChildDim(I))...}
#     let apply1s=fs, apply2=node.apply, b=node.bias, argT=rT
#         function (x::AbtVecOfAbtArray{T})
#             splat(apply2)( map(f->f(x), apply1s)::argT ) + b
#         end
#     end
# end


function evalGraphNode(gn::FixedNode{T}, ::AbtVecOfAbtArray{T}) where {T}
    unpackAA0D(T, gn.val)
end

function evalGraphNode(gn::IndexNode{T, 0}, data::AbtVecOfAbtArray{T}) where {T}
    data[begin][gn.idx]
end

function evalGraphNode(gn::IndexNode{T}, data::AbtVecOfAbtArray{T}) where {T}
    data[gn.idx]
end

function evalGraphNode(gn::ReferenceNode{T}, ::AbtVecOfAbtArray{T}) where {T}
    unpackAA0D(T, gn.val)
end

function evalGraphNode(gn::ReductionNode{T}, data::AbtVecOfAbtArray{T}) where {T}
    gn.apply( broadcast(evalGraphNodeCore, gn.child, Ref(data))... ) + gn.bias
end

function evalGraphNode(gn::MorphismNode{T}, data::AbtVecOfAbtArray{T}) where {T}
    gn.apply( broadcast(evalGraphNodeCore, gn.child, Ref(data))... )
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

selectInPSubset(::Val{0}, ps::AbstractVector{<:PrimDParSetEltype{T}}) where {T} = first(ps)
selectInPSubset(::Val,    ps::AbstractVector{<:PrimDParSetEltype{T}}) where {T} = itself(ps)

function genComputeGraphCore1(inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                              par::DimensionalParam{T, N}) where {T, N}
    idx = findfirst(Fix2(compareParamContainer, par), selectInPSubset(Val(N), inPSet))
    val = fillElementalVal(Val(N), obtain(par))
    sym = symbolFromPar(par)
    idx === nothing ? FixedNode(val, sym) : IndexNode(val, sym, idx)
end

function genComputeGraphCore2(idDict::IdDict{ParamBox{T}, NodeMarker{<:GraphNode{T}}}, 
                              inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                              par::DimensionalParam{T, N}) where {T, N}
    sl = checkScreenLevel(screenLevelOf(par), getScreenLevelRange(par))

    if sl == 0
        # Depth-first search by recursive calling
        marker = get!(idDict, par, NodeMarker(ReferenceNode(par), GraphNode{T, N}))
        if !marker.visited
            marker.visited = true
            childNodes = map(par.input) do arg
                genComputeGraphCore2(idDict, inPSet, arg)
            end
            res = genOperatorNode(par, childNodes)
            marker.value = res
        else
            marker.value
        end
    elseif sl == 1
        genComputeGraphCore1(inPSet, par)
    else
        FixedNode(fillElementalVal(Val(N), par|>obtain), symbolFromPar(par))
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
    idDict = IdDict{ParamBox{T}, NodeMarker{<:GraphNode{T}}}()
    genComputeGraphCore2(idDict, inPSet, par)
end

genComputeGraph(inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                par::PrimitiveParam{T}) where {T} = 
genComputeGraphCore1(inPSet, par)