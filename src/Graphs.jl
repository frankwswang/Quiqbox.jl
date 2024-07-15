export genComputeGraph, evalGraphNode, compressGraphNode

const ChildNodeType{T, N} = Union{AbstractArray{<:GraphNode{T, 0}, N}, GraphNode{T, N}}
const NodeChildrenType{T} = TernaryTupleUnion{ChildNodeType{T}}


getChildNum(::Type{<:NonEmptyTuple{ChildNodeType{T}, N}}) where {T, N} = N+1

packElementalVal(::Val{0}, obj::Any) = fill(obj)
packElementalVal(::Val,    obj::AbstractArray) = copy(obj)
packElementalVal(::Val{0}, obj::AbstractArray{<:Any, 0}) = copy(obj)

unpackAA0D(::Type{T}, obj::AbtArray0D{<:T}) where {T} = obj[]
unpackAA0D(::Type{T}, obj::AbstractArray{<:T}) where {T} = itself(obj)

function genConstFunc(::Type{T1}, val::T2) where {T1, T2}
    let res = val
        function (::T1)
            res
        end
    end
end


struct ValueNode{T, N, V<:AbstractArray{T, N}} <: StorageNode{T, N}
    val::V
    marker::Symbol

    function ValueNode(::Val{N}, obj, marker::Symbol) where {N}
        val = packElementalVal(Val(N), obj)
        new{eltype(val), ndims(val), typeof(val)}(val, marker)
    end
end

struct IndexNode{T, N, V<:AbstractArray{T, N}} <: StorageNode{T, N}
    val::V
    marker::Symbol
    index::Int

    function IndexNode(::Val{N}, obj, marker::Symbol, index::Int) where {N}
        val = packElementalVal(Val(N), obj)
        new{eltype(val), ndims(val), typeof(val)}(val, marker, index)
    end
end

struct ReferenceNode{T, N, V<:AbstractArray{T, N}} <: StorageNode{T, N}
    val::V
    id::UInt

    function ReferenceNode(pb::ParamBox{T, N}) where {T, N}
        val = packElementalVal(Val(N), pb.memory)
        new{T, N, typeof(val)}(val, objectid(pb))
    end
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


function compressGraphNode(gn::ValueNode{T}) where {T}
    genConstFunc(AbtVecOfAbtArray{T}, deepcopy( itself.(gn.val) ))
end

# # This might make Enzyme (v0.12.22-23) crash or output wrong gradient and pollute data
# function compressGraphNode(gn::ValueNode{T}) where {T}
#     let val = deepcopy(gn.val)
#         (::AbtVecOfAbtArray{T})->itself.(val)
#     end
# end

function compressGraphNode(gn::IndexNode{T, 0}) where {T}
    (x::AbtVecOfAbtArray{T})->x[begin][gn.index]
end

function compressGraphNode(gn::IndexNode{T}) where {T}
    (x::AbtVecOfAbtArray{T})->x[gn.index]
end

function compressGraphNode(::EmptyNode{T, N}) where {T, N}
    data = reshape( fill( T() ), ntuple(_->1, Val(N)) )
    genConstFunc(AbtVecOfAbtArray{T}, deepcopy( itself.(data) ))
end

function compressGraphNode(gn::OperatorNode{T, N, I}) where {T, N, I}
    ChildNum = getChildNum(I)
    compressGraphNodeCore(Val(ChildNum), gn)
end

function compressGraphNode(gn::ReferenceNode{T}) where {T}
    genConstFunc(AbtVecOfAbtArray{T}, deepcopy( itself.(gn.val) ))
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
    f2 = hasfield(typeof(node), :bias) ? Fix2(+, deepcopy(node.bias)) : itself
    let apply1=f1, apply2=node.apply, apply3=f2
        function (x::AbtVecOfAbtArray{T})
            x |> apply1 |> apply2 |> apply3
        end
    end
end

function compressGraphNodeCore(::Val{2}, node::OperatorNode{T}) where {T}
    fL, fR = makeGraphFuncComp.(node.child)
    f2 = hasfield(typeof(node), :bias) ? Fix2(+, deepcopy(node.bias)) : itself
    let apply1L=fL, apply1R=fR, apply2=node.apply, apply3=f2
        function (x::AbtVecOfAbtArray{T})
            apply2( apply1L(x), apply1R(x) ) |> apply3
        end
    end
end

function compressGraphNodeCore(::Val{3}, node::OperatorNode{T}) where {T}
    fL, fC, fR = makeGraphFuncComp.(node.child)
    f2 = hasfield(typeof(node), :bias) ? Fix2(+, deepcopy(node.bias)) : itself
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


function evalGraphNode(gn::ValueNode{T}, ::AbtVecOfAbtArray{T}) where {T}
    itself.(gn.val)
end

function evalGraphNode(gn::IndexNode{T, 0}, data::AbtVecOfAbtArray{T}) where {T}
    data[begin][gn.index]
end

function evalGraphNode(gn::IndexNode{T}, data::AbtVecOfAbtArray{T}) where {T}
    data[gn.index]
end

function evalGraphNode(gn::ReferenceNode{T}, ::AbtVecOfAbtArray{T}) where {T}
    itself.(gn.val)
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


const symbolFromPar = symbolFrom∘indexedSymOf

selectInPSubset(::Val{0}, ps::AbstractVector{<:PrimDParSetEltype{T}}) where {T} = first(ps)
selectInPSubset(::Val,    ps::AbstractVector{<:PrimDParSetEltype{T}}) where {T} = itself(ps)

function genComputeGraphCore1(inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                              par::DimensionalParam{T, N}) where {T, N}
    idx = findfirst(Fix2(compareParamContainer, par), selectInPSubset(Val(N), inPSet))
    val = obtain(par)
    sym = symbolFromPar(par)
    idx === nothing ? ValueNode(Val(N), val, sym) : IndexNode(Val(N), val, sym, idx)
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
        ValueNode(Val(N), obtain(par), symbolFromPar(par))
    end
end

function genComputeGraphCore2(idDict, inPSet, 
                              pars::AbstractArray{<:DimensionalParam{T}}) where {T}
    map(pars) do par
        genComputeGraphCore2(idDict, inPSet, par)
    end
end

function genComputeGraphINTERNAL(inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                                 par::ParamBox{T}) where {T}
    idDict = IdDict{ParamBox{T}, NodeMarker{<:GraphNode{T}}}()
    genComputeGraphCore2(idDict, inPSet, par)
end

function genComputeGraphINTERNAL(inPSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                                 par::PrimitiveParam{T}) where {T}
    genComputeGraphCore1(inPSet, par)
end


const ParamSetErrorMessage1 = "`$ElementalParam` must all be enclosed in a single array "*
                              "as the first element of `inputParamSet`."

const ParamSetErrorMessage2 = "The screen level of the input parameter (inspected by "*
                              "`$screenLevelOf`) for a dependent parameter must be 1."

function genComputeGraph(inputParamSet::AbstractVector{<:PrimDParSetEltype{T}}, 
                         par::DimensionalParam{T}) where {T}
    for i in eachindex(inputParamSet)
        p = inputParamSet[i]

        bl1 = i == firstindex(inputParamSet)
        bl2 = p isa AbstractArray{<:ElementalParam{T}}
        bl3 = (p isa DimensionalParam && !(p isa ElementalParam))
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

    genComputeGraphINTERNAL(inputParamSet, par)
end