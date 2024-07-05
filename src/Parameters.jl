export RealVar, ParamNode, NodeTuple, setScreenLevel!, setScreenLevel, symOf, inputOf, 
       dataOf, valOf, screenLevelOf, markParams!, topoSort, getParams, getSingleParams

using Base: Fix2

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


struct RealVar{T<:Real} <: SinglePrimP{T}
    data::RefVal{T}
    symbol::IndexedSym

    RealVar(data::RefVal{T}, symbol::Union{IndexedSym, Symbol}) where {T} = 
    new{T}(Ref(data[]), IndexedSym(symbol))
end

RealVar(data::Real, symbol::Symbol) = RealVar(Ref(data), symbol)


struct ParamNode{T, F<:Function, I<:SParamAbtArray{T}} <: ParamBox{T, I}
    lambda::StableReduce{T, F}
    data::I
    symbol::IndexedSym
    offset::RefVal{T}
    screen::RefVal{TernaryNumber}

    function ParamNode(lambda::StableReduce{T, F}, data::I, 
                       symbol::Union{IndexedSym, Symbol}, 
                       offset::RefVal{T}, 
                       screen::RefVal{TernaryNumber}=Ref(TUS0)) where {T, I, F}
        isempty(data) && throw(AssertionError("`data` should not be empty."))
        new{T, F, I}(lambda, data, 
                     IndexedSym(symbol), 
                     Ref(offset[]), 
                     Ref{TernaryNumber}(screen[]))
    end

    ParamNode(::StableReduce{T, <:iTalike}, data::Array0D{P}, 
              symbol::Union{IndexedSym, Symbol}, 
              offset::RefVal{T}, 
              screen::RefVal{TernaryNumber}=Ref(TUS0)) where {T, P<:SinglePrimP{T}} = 
    new{T, iT, Array0D{P}}(StableReduce(T, itself), data, 
                           IndexedSym(symbol), 
                           Ref(offset[]), 
                           Ref{TernaryNumber}(screen[]))
end

ParamNode(::StableReduce{T, <:iTalike}, ::Any, ::Union{IndexedSym, Symbol}, ::RefVal{T}, 
          ::RefVal{TernaryNumber}=Ref(TUS0)) where {T} = 
throw(AssertionError("Second argument should be an `Array0D{<:SinglePrimP{$T}}`."))

ParamNode(lambda::Function, data::SParamAbtArray{T}, symbol::Symbol) where {T} = 
ParamNode(StableReduce(Vector{T}, lambda), data, symbol, (Ref∘zero)(T), Ref(TUS0))

ParamNode(lambda::Function, vars::AbstractArray{<:Real}, varsSym::AbtArrayOfOr{Symbol}, 
          symbol::Symbol) = 
ParamNode(lambda, RealVar.(vars, varsSym), symbol)

ParamNode(lambda::Function, data::SingleParam{T}, symbol::Symbol) where {T} = 
ParamNode(StableReduce(T, lambda), fill(data), symbol, (Ref∘zero)(T), Ref(TUS0))

ParamNode(lambda::Function, var::Real, varSym::Symbol, symbol::Symbol) = 
ParamNode(lambda, RealVar(var, varSym), symbol)

ParamNode(data::RealVar{T}, symbol::Symbol=symOf(data)) where {T} = 
ParamNode(StableReduce(T, itself), fill(data), symbol, (Ref∘zero)(T), Ref(TUS0))

ParamNode(var::ParamNode, symbol::Symbol=symOf(var)) = 
ParamNode(var.lambda, var.data, symbol, var.offset, var.screen)

ParamNode(var::Real, varSym::Symbol, symbol::Symbol=varSym) = 
ParamNode(RealVar(var, varSym), symbol)

const TensorInputNode{T, F, PB, N} = ParamNode{T, F, <:AbstractArray{PB, N}}
const ScalarInputNode{T, F, PB} = TensorInputNode{T, F, PB, 0}

screenLevelOf(pn::ParamNode) = Int(pn.screen[])

screenLevelOf(::RealVar) = 1

function setScreenLevel!(pn::ParamNode, level::Int)
    levelOld = screenLevelOf(pn)
    if levelOld == level
    elseif levelOld == 0
        pn.offset[] = valOf(pn)
    elseif level == 0
        pn.offset[] -= pn.lambda(pn|>dataOf.|>valOf)
    end
    pn.screen[] = TernaryNumber(level)
    pn
end

function setScreenLevel(pn::ParamNode, level::Int)
    pnNew = ParamNode(pn.lambda, pn.data, pn.symbol, pn.offset, pn.screen)
    setScreenLevel!(pnNew, level)
end


struct NodeTuple{T<:Real, N, PT<:NTuple{N, ParamNode{T}}} <: ParamStack{T, PT, N}
    data::PT
    symbol::IndexedSym

    NodeTuple(data::PT, symbol::Union{IndexedSym, Symbol}) where 
             {T, N, PT<:NTuple{N, ParamNode{T}}} = 
    new{T, N, PT}(data, IndexedSym(symbol))
end

NodeTuple(nt::NodeTuple, symbol::Symbol) = NodeTuple(nt.data, symbol)



idxSymOf(pc::ParamContainer) = pc.symbol

symOf(pc::ParamContainer) = idxSymOf(pc).name

dataOf(pc::ParamContainer) = pc.data

mapOf(pn::ParamNode) = pn.lambda

valOf(sv::SinglePrimP) = dataOf(sv)[]

function unsafeScreen!(pn::ParamNode{T}, screenLevel::Int, offset::T) where {T}
    pn.screen[] = TernaryNumber(screenLevel)
    pn.offset[] = offset
end

function valOf(pn::ParamNode{T}, fallbackValue::T=pn.offset[]) where {T}
    originalOffset = pn.offset[]
    res = originalOffset
    if screenLevelOf(pn) == 0
        unsafeScreen!(pn, 2, fallbackValue)
        res += pn.lambda(pn|>dataOf.|>valOf)
        unsafeScreen!(pn, 0, originalOffset)
    end
    res
end

valOf(pc::ParamContainer) = valOf.(dataOf(pc))

(pn::ParamContainer)() = valOf(pn)


inputOf(pc::ParamContainer) = dataOf(pc)

inputOfCore(pn::TensorInputNode) = pn.data
inputOfCore(pn::ScalarInputNode) = pn.data[]

function inputOf(pn::ParamNode{T}) where {T}
    screen = screenLevelOf(pn)
    if screen == 0
        inputOfCore(pn)
    elseif screen == 1
        pn.offset
    else # screen == 2
        valOf(pn)
    end
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
ParamNode(op∘pn.lambda.f, pn.data, symbol, pn.offset[], pn.screen[])

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
        i.symbol.index[] = (indexDict[sym] += 1)
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
sever(pv::RealVar) = RealVar(valOf(pv), pv.symbol)

sever(pn::ParamNode) = ParamNode(valOf(pn), symOf(pn))

sever(ps::T) where {T<:ParamStack} = T(sever.(ps.data), ps.symbol)

sever(obj::Any) = deepcopy(obj)

sever(obj::Union{Tuple, AbstractArray}) = sever.(obj)

function sever(pf::T) where {T<:ParamFunction}
    severedFields = map(fieldnames(pf)) do field
        (sever∘getproperty)(pf, field)
    end
    T(severedFields...)
end