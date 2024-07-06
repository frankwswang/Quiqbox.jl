export RealVar, ParamNode, NodeTuple, setScreenLevel!, setScreenLevel, symOf, inputOf, 
       dataOf, outValOf, screenLevelOf, markParams!, topoSort, getParams, getSingleParams

using Base: Fix2, Threads.Atomic

struct StableReduce{T, F<:Function} <: Function
    f::F

    function StableReduce(::Type{AT}, f::F) where {T, AT<:Union{T, AbstractArray{T}}, F}
        Base.return_types(f, (AT,))[1] == T || 
        throw(AssertionError("The lambda function `f`: `$f` should return `$T`."))
        new{T, F}(f)
    end
end

StableReduce(::Type{<:Union{T, AbstractArray{T}}}, srf::StableReduce{T}) where {T} = srf
StableReduce(::Type{AT}, srf::StableReduce{T1}) where 
            {T1, T2, AT<:Union{T2, AbstractArray{T2}}} = 
StableReduce(srf.f, AT)

StableReduce(::Type{T}) where {T} = StableReduce(T, itself)

(sf::StableReduce{T, F})(arg::T) where {F, T} = T(sf.f(arg))
(sf::StableReduce{T, F})(arg::Array0D{T}) where {F, T} = sf(arg[])
(sf::StableReduce{T, F})(arg::AbstractArray{T}) where {F, T} = T(sf.f(arg))

const SymOrIdxSym = Union{Symbol, IndexedSym}

struct RealVar{T<:Real, V<:AbtArray0D{T}} <: SinglePrimP{T}
    data::V
    symbol::IndexedSym

    RealVar(data::V, symbol::SymOrIdxSym; copyVal::Bool=true) where {T, V<:AbtArray0D{T}} = 
    new{T, V}((copyVal ? deepcopy(data) : data), IndexedSym(symbol))
end

RealVar(data::Real, symbol::Symbol; copyVal::Bool=true) = 
RealVar(fill(data), symbol; copyVal)

mutable struct ParamNode{T, F<:Function, I<:SParamAbtArray{T}} <: ParamBox{T, I}
    const lambda::StableReduce{T, F}
    const data::I
    const symbol::IndexedSym
    const offset::Atomic{T}
    @atomic memory::T
    @atomic screen::TernaryNumber

    function ParamNode(lambda::StableReduce{T, F}, data::I, 
                       symbol::SymOrIdxSym, 
                       offset::Atomic{T}, 
                       memory::T=zero(T), 
                       screen::TernaryNumber=TUS0) where {T, I, F}
        isempty(data) && throw(AssertionError("`data` should not be empty."))
        new{T, F, I}(lambda, data, IndexedSym(symbol), Atomic{T}(offset[]), memory, screen)
    end

    function ParamNode(::StableReduce{T, <:iTalike}, data::I, 
                       symbol::SymOrIdxSym, 
                       offset::Atomic{T}, 
                       memory::T=outValOf(data[]), 
                       screen::TernaryNumber=TUS0) where 
                      {T, P<:SinglePrimP{T}, I<:AbtArray0D{P}}
        new{T, iT, I}(StableReduce(T, itself), data, IndexedSym(symbol), 
                      Atomic{T}(offset[]), memory, screen)
    end
end

ParamNode(::StableReduce{T, <:iTalike}, ::Any, ::SymOrIdxSym, ::Atomic{T}, 
          ::T=zero(T), ::TernaryNumber=TUS0) where {T} = 
throw(AssertionError("Second argument should be an `Array0D{<:SinglePrimP{$T}}`."))

function ParamNode(lambda::Function, data::SParamAbtArray{T}, symbol::Symbol; 
                   init::T=zero(T)) where {T}
    VT = (map(outValOf, data) |> typeof)
    ParamNode(StableReduce(VT, lambda), data, symbol, Atomic{T}(zero(T)), init)
end

ParamNode(lambda::Function, vars::AbstractArray{T}, varsSym::AbtArrayOfOr{Symbol}, 
          symbol::Symbol; init::T=zero(T)) where {T<:Real} = 
ParamNode(lambda, RealVar.(vars, varsSym), symbol; init)

ParamNode(lambda::Function, data::SingleParam{T}, symbol::Symbol; 
          init::T=zero(T)) where {T} = 
ParamNode(StableReduce(T, lambda), fill(data), symbol, Atomic{T}(zero(T)), init)

ParamNode(lambda::Function, var::T, varSym::Symbol, symbol::Symbol; 
          init::T=zero(T)) where {T<:Real} = 
ParamNode(lambda, RealVar(var, varSym), symbol; init)

ParamNode(data::RealVar{T}, symbol::Symbol=symOf(data)) where {T} = 
ParamNode(StableReduce(T, itself), fill(data), symbol, Atomic{T}(zero(T)))

ParamNode(var::Real, varSym::Symbol, symbol::Symbol=varSym) = 
ParamNode(RealVar(var, varSym), symbol)

ParamNode(var::ParamNode{T}, symbol::Symbol=symOf(var); init::T=var.memory) where {T} = 
ParamNode(var.lambda, var.data, symbol, var.offset, init, var.screen)

const TensorInParamNode{T, F, PB, N} = ParamNode{T, F, <:AbstractArray{PB, N}}
const ScalarInParamNode{T, F, PB} = TensorInParamNode{T, F, PB, 0}


screenLevelOf(pn::ParamNode) = Int(pn.screen)

screenLevelOf(::RealVar) = 1

function setScreenLevelCore!(pn::ParamNode, level::Int)
    @atomic pn.screen = TernaryNumber(level)
end

function setScreenLevel!(pn::ParamNode, level::Int)
    levelOld = screenLevelOf(pn)
    if levelOld == level
    elseif levelOld == 0
        Threads.atomic_xchg!(pn.offset, outValOf(pn))
    elseif level == 0
        Threads.atomic_sub!(pn.offset, pn.lambda(map(outValOf, dataOf(pn))))
    end
    setScreenLevelCore!(pn, level)
    pn
end

setScreenLevel(pn::ParamNode, level::Int) = setScreenLevel!(ParamNode(pn), level)


function memorize!(pn::ParamNode{T}, newMem::T=ValOf(pn)) where {T}
    oldMem = pn.memory
    @atomic pn.memory = newMem
    oldMem
end


struct NodeTuple{T<:Real, N, PT<:NTuple{N, ParamNode{T}}} <: ParamStack{T, PT, N}
    data::PT
    symbol::IndexedSym

    NodeTuple(data::PT, symbol::SymOrIdxSym) where {T, N, PT<:NTuple{N, ParamNode{T}}} = 
    new{T, N, PT}(data, IndexedSym(symbol))
end

NodeTuple(nt::NodeTuple, symbol::Symbol) = NodeTuple(nt.data, symbol)


idxSymOf(pc::ParamContainer) = pc.symbol

symOf(pc::ParamContainer) = idxSymOf(pc).name

dataOf(pc::ParamContainer) = pc.data

mapOf(pn::ParamNode) = pn.lambda


outValOf(sv::SinglePrimP) = dataOf(sv)[]

function outValOf(pn::ParamNode{T}, fallbackVal::T=pn.memory) where {T}
    idSet = Set{UInt}(pn|>objectid)
    outValOfCore(Val(false), pn, idSet, fallbackVal)
end

function outValOfCore(::Val{BL}, pn::ParamNode{T}, 
                       idSet::Set{UInt64}, fallbackVal::T) where {BL, T}
    res = pn.offset[]
    sl = screenLevelOf(pn)
    if sl == 0
        id = objectid(pn)
        if BL && id in idSet
            res = fallbackVal
        else
            push!(idSet, id)
            res += map(dataOf(pn)) do child
                outValOfCore(Val(true), child, idSet, fallbackVal)
            end |> pn.lambda
        end
    elseif sl == -1
        res = fallbackVal
    end
    res
end

outValOfCore(::Val, par::PrimitiveParam{T},::Set{UInt}, ::T) where {T} = outValOf(par)

outValOf(pc::ParamStack) = outValOf.(dataOf(pc))

(pn::ParamContainer)() = outValOf(pn)

function inputOf(pn::ParamNode{T}) where {T}
    screen = screenLevelOf(pn)
    screen == 0 ? itself.(pn.data) : pn.offset
end

inputOf(pp::PrimitiveParam) = pp.data

isLikePrimitiveParam(::PrimitiveParam) = true
isLikePrimitiveParam(pn::ParamNode) = (screenLevelOf(pn) == 1)

function safelySetVal!(box::Atomic{T}, val::T) where {T}
    Threads.atomic_xchg!(box, val)
end

function safelySetVal!(box::AbtArray0D{T}, val::T) where {T}
    lk = ReentrantLock()
    lock(lk) do
        box[] = val
    end
end

function setInputVal!(par::ParamContainer{T}, var::T) where {T}
    isLikePrimitiveParam(par) || 
    throw(AssertionError("Input `par` does not behave like a primitive parameter."))
    safelySetVal!(inputOf(par), var)
    var
end

import Base: iterate, size, length, eltype, broadcastable
length(::FixedSizeParam{<:Any, N}) where {N} = N
eltype(np::FixedSizeParam) = eltype(np.data)

iterate(::FixedSizeParam{<:Any, 1}, args...) = iterate(1, args...)
size(::FixedSizeParam{<:Any, 1}, args...) = size(1, args...)
broadcastable(np::FixedSizeParam{<:Any, 1}) = Ref(np)

iterate(np::FixedSizeParam, args...) = iterate(np.data, args...)
size(np::FixedSizeParam, args...) = size(np.data, args...)
broadcastable(np::FixedSizeParam) = Base.broadcastable(np.data)


struct Marker <: AbstractMarker{UInt}
    typeID::UInt
    valueID::UInt

    Marker(data::T) where {T} = new(objectid(T), objectid(data))
end

struct ContainerMarker{N, M<:NTuple{N, AbstractMarker{UInt}}} <: AbstractMarker{UInt}
    typeID::UInt
    dataMarker::M
    funcID::UInt
    metaID::UInt
end

const NothingID = objectid(nothing)

function ContainerMarker(sv::T) where {T<:RealVar}
    m = Marker(sv.data)
    ContainerMarker{1, Tuple{Marker}}(objectid(T), (m,), NothingID, NothingID)
end

function ContainerMarker(pn::T) where {T<:ParamNode}
    m = Marker(pn.data)
    ContainerMarker{1, Tuple{Marker}}(
        objectid(T), (m,), objectid(pn.lambda.f), 
        objectid((pn.offset[], screenLevelOf(pn)))
    )
end

function ContainerMarker(nt::T) where {N, T<:NodeTuple{<:Any, N}}
    ms = ContainerMarker.(nt.data)
    ContainerMarker{N, typeof(ms)}(objectid(T), ms, NothingID, NothingID)
end


compareParamContainer(::ParamContainer, ::ParamContainer) = false

compareParamContainer(pc1::T, pc2::T) where {T<:ParamContainer} = 
pc1 === pc2 || ContainerMarker(pc1) == ContainerMarker(pc2)


operateBy(::typeof(+), pn1::ParamNode) = itself(pn1)
operateBy(::typeof(-), pn1::ParamNode{T}) where {T} = operateBy(*, T(-1), pn1)

repeatedlyApply(::typeof(+), pn::ParamNode{T}, times::Int) where {T} = 
operateBy(*, pn, T(times))

repeatedlyApply(::typeof(-), pn::ParamNode{T}, times::Int) where {T} = 
operateBy(-, operateBy(*, pn, T(times)))

repeatedlyApply(::typeof(*), pn::ParamNode{T}, times::Int) where {T} = 
operateBy(^, pn, T(times))

operateBy(op::F, pn1::ParamNode, num::Real) where {F<:Function} = 
ParamNode(OFC(itself, op, num), pn1, pn1.symbol)

operateBy(op::F, num::Real, pn1::ParamNode) where {F<:Function} = 
ParamNode(OCF(itself, op, num), pn1, pn1.symbol)

operateBy(op::CommutativeBinaryNumOps, num::Real, pn1::ParamNode) = 
operateBy(op, pn1::ParamNode, num::Real)

operateBy(op::F, pn::ParamNode{T}) where {F<:Function, T} = 
ParamNode(op∘pn.lambda.f, pn.data, symbol, pn.offset, pn.memory, pn.screen)

operateByCore(op::F, pn1::ParamNode{T}, pn2::ParamNode{T}) where {F<:Function, T} = 
ParamNode(SplitArg{2}(op), [pn1, pn2], Symbol(pn1.symbol, pn2.symbol))

operateByCore(op::F, pn1::ParamNode{T}, pn2::ParamNode{T}, 
              pns::Vararg{ParamNode{T}, N}) where {F<:Function, T, N} = 
ParamNode(SplitArg{N}(op), [pn1, pn2, pns...], Symbol(pn1.symbol, :_to_, pns[end].symbol))

function operateBy(op::F, pn1::ParamNode{T}, pn2::ParamNode{T}) where 
                  {F<:CommutativeBinaryNumOps, T}
    if symFromIndexSym(pn1.symbol) > symFromIndexSym(pn2.symbol)
        pn2, pn1 = pn1, pn2
    end
    if compareParamContainer(pn1, pn2)
        repeatedlyApply(op, pn1, 2)
    else
        operateByCore(op, pn1, pn2)
    end
end

operateBy(op::F, pn1::ParamNode{T}, pns::Vararg{ParamNode{T}, N}) where {F<:Function, T, N} = 
operateByCore(op, pn1, pns...)

operateBy(f::F, pns::AbstractArray{<:ParamNode{T}}) where {F<:Function, T} = 
ParamNode(f, pns, Symbol(pns[begin].symbol.name, :_to_, pns[end].symbol.name))

operateBy(f::F, ::Val{N}) where {F<:Function, N} = 
((args::Vararg{ParamNode{T}, N}) where {T}) -> operateBy(f, args...)

operateBy(f::F) where {F<:Function} = 
((args::AbstractArray{<:ParamNode{T}}) where {T}) -> operateBy(f, args)


addParamNode(pn1::ParamNode{T}, pn2::ParamNode{T}) where {T} = operateBy(+, pn1, pn2)

mulParamNode(pn1::ParamNode{T}, pn2::ParamNode{T}) where {T} = operateBy(*, pn1, pn2)
mulParamNode(pn::ParamNode{T}, coeff::T) where {T} = operateBy(*, pn, coeff)
mulParamNode(coeff::T, pn::ParamNode{T}) where {T} = mulParamNode(pn, coeff)


function sortParamContainers(::Type{C}, f::F, field::Symbol, roundAtol::T) where 
                            {T, C<:ParamFunction{T}, F}
    let roundAtol=roundAtol, f=f, field=field
        function (container::C)
            ele = getproperty(container, field)
            ( roundToMultiOfStep(f(container), nearestHalfOf(roundAtol)), 
              symFromIndexSym(ele.symbol), ContainerMarker(ele) )
        end
    end
end


function getParamFields(pf::T) where {T<:ParamFunction}
    fields = fieldnames(pf)
    ids = findall(x->(x isa ParamFunctions), fields)
    getproperty.(pf, fields[ids])
end

getParamFields(pf::ParamFunctions) = itself(pf)

const GetParTypes = (SingleParam, ParamContainer)

function getParCore(::Val{N}, p::ParamContainer{T}, ::Missing) where {N, T}
    GetParTypes[N]{T}[p]
end

function getParCore(::Val{N}, p::ParamContainer{T}, sym::Symbol) where {N, T}
    res = GetParTypes[N]{T}[]
    inSymbol(sym, symOf(ps)) && push!(res, p)
    res
end

function getParsCore(::Val{1}, p::ParamContainer, sym::SymOrMiss=missing)
    p isa SingleParam ? getParCore(Val(1), p, sym) : getParsCore(Val(1), p.data, sym)
end

function getParsCore(::Val{2}, p::ParamContainer, sym::SymOrMiss=missing)
    getParCore(Val(2), p, sym)
end

function getParsCore(::Val{N}, v::NonEmptyTupleOrAbtArray{T}, sym::SymOrMiss=missing) where 
                    {N, T<:Union{ParamContainer, ParamObject}}
    if isempty(v)
        genParamTypeVector(T, GetParTypes[N])
    else
        reduce(vcat, getParsCore.(Val(N), v, sym))
    end
end

function getParsCore(::Val{N}, f::ParamFunction{T}, sym::SymOrMiss=missing) where {N, T}
    mapreduce(vcat, getParamFields(f), init=GetParTypes[N]{T}[]) do field
        getParsCore(Val(N), field, sym)
    end
end


getSingleParams(data, sym::SymOrMiss=missing) = getParsCore(Val(1), data, sym)

getParams(data, sym::SymOrMiss=missing) = getParsCore(Val(2), data, sym)

throwScreenLevelError(sl) = 
throw(DomainError(sl, "This value is not supported as the screen level of a `ParamNode`."))

uniqueParams(ps::AbstractArray{<:ParamContainer}) = 
markUnique(ps, compareFunction=compareParamContainer)[end]


function markParams!(pars::AbstractVector{<:SingleParam})
    nodes, marks1, marks2 = topoSort(pars)
    leafNodes = nodes[.!marks1 .*   marks2]
    rootNodes = nodes[  marks1 .* .!marks2]
    selfNodes = nodes[.!marks1 .* .!marks2]
    indexDict = IdDict{Symbol, Int}()
    for i in leafNodes
        sym = i.symbol.name
        get!(indexDict, sym, 0)
        i.symbol.index = (indexDict[sym] += 1)
    end
    (leafNodes, rootNodes, selfNodes)
end

markParams!(b::Union{AbstractVector{T}, T}) where {T<:ParamObject} = 
markParams!(getSingleParams(b))


function topoSortCore!(orderedNodes::Vector{<:SingleParam{T}}, 
                       haveBranches::Vector{Bool}, connectRoots::Vector{Bool}, 
                       node::SingleParam{T}, recursive::Bool=false) where {T}
    sl = screenLevelOf(node)
    if sl in (0, 1)
        idx = findfirst(Fix2(compareParamContainer, node), orderedNodes)
        if idx === nothing
            hasBranch = if sl == 0
                for child in node.data
                    topoSortCore!(orderedNodes, haveBranches, connectRoots, child, true)
                end
                true
            else
                false
            end
            push!(orderedNodes, node)
            push!(haveBranches, hasBranch)
            push!(connectRoots, recursive)
        else
            connectRoots[idx] = recursive
        end
    else
        sl == 2 || throwScreenLevelError(sl)
    end
    nothing
end

function topoSortINTERNAL(nodes::AbstractVector{<:SingleParam{T}}) where {T}
    orderedNodes = SingleParam{T}[]
    haveBranches = Bool[]
    connectRoots = Bool[]
    for node in nodes
        topoSortCore!(orderedNodes, haveBranches, connectRoots, node)
    end
    orderedNodes, haveBranches, connectRoots
end

function topoSort(nodes::AbstractVector{<:SingleParam{T}}) where {T}
    uniqueParams(nodes) |> topoSortINTERNAL
end

topoSort(node::ParamNode{T}) where {T} = topoSortINTERNAL([node])


# Sever the connection of a node to other nodes
sever(pv::RealVar) = RealVar(outValOf(pv), pv.symbol)

sever(pn::ParamNode) = ParamNode(outValOf(pn), symOf(pn))

sever(ps::T) where {T<:ParamStack} = T(sever.(ps.data), ps.symbol)

sever(obj::Any) = deepcopy(obj)

sever(obj::Union{Tuple, AbstractArray}) = sever.(obj)

function sever(pf::T) where {T<:ParamFunction}
    severedFields = map(fieldnames(pf)) do field
        (sever∘getproperty)(pf, field)
    end
    T(severedFields...)
end