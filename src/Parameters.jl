export NodeVar, ParamNode, NodeTuple, setScreenLevel!, setScreenLevel, symOf, 
       inputOf, obtain, setVal!, screenLevelOf, markParams!, topoSort, getParams, 
       getSingleParams

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

struct GridVar{T<:Real, N, V<:AbstractArray{T, N}}
    value::V
    symbol::IndexedSym
    axis::NTuple{N, Symbol}
    extent::NTuple{N, Int}

    function GridVar(value::V, symbol::SymOrIdxSym, axis::NTuple{N, Symbol}, 
                     extent::NTuple{N, Int}=size(value)) where 
                    {T, N, V<:AbstractArray{T, N}}
        new{T, N, V}(value, IndexedSym(symbol), axis, extent)
    end
end

mutable struct NodeVar{T} <: SinglePrimP{T}
    @atomic value::T
    const symbol::IndexedSym

    NodeVar(value::T, symbol::SymOrIdxSym) where {T} = new{T}(value, IndexedSym(symbol))
end


mutable struct ParamNode{T, F<:Function, I<:DParamAbtArray{T}} <: ParamBox{T, I, 0}
    const lambda::StableReduce{T, F}
    const input::I
    const symbol::IndexedSym
    @atomic offset::T
    @atomic memory::T
    @atomic screen::TernaryNumber

    function ParamNode(lambda::StableReduce{T, F}, input::I, 
                       symbol::SymOrIdxSym, 
                       offset::T, 
                       memory::T=zero(T), 
                       screen::TernaryNumber=TUS0) where {T, I, F}
        isempty(input) && throw(AssertionError("`input` should not be empty."))
        new{T, F, I}(lambda, input, IndexedSym(symbol), offset, memory, screen)
    end

    function ParamNode(::StableReduce{T, <:iTalike}, input::I, 
                       symbol::SymOrIdxSym, 
                       offset::T, 
                       memory::T=obtain(input[]), 
                       screen::TernaryNumber=TUS0) where 
                      {T, P<:SinglePrimP{T}, I<:AbtArray0D{P}}
        new{T, iT, I}(StableReduce(T, itself), input, IndexedSym(symbol), 
                      offset, memory, screen)
    end
end

ParamNode(::StableReduce{T, <:iTalike}, ::Any, ::SymOrIdxSym, ::T, 
          ::T=zero(T), ::TernaryNumber=TUS0) where {T} = 
throw(AssertionError("Second argument should be an `Array0D{<:SinglePrimP{$T}}`."))

function ParamNode(lambda::Function, input::SParamAbtArray{T}, symbol::Symbol; 
                   init::T=zero(T)) where {T}
    VT = (map(obtain, input) |> typeof)
    ParamNode(StableReduce(VT, lambda), input, symbol, zero(T), init)
end

ParamNode(lambda::Function, vars::AbstractArray{T}, varsSym::AbtArrayOfOr{Symbol}, 
          symbol::Symbol; init::T=zero(T)) where {T<:Real} = 
ParamNode(lambda, NodeVar.(vars, varsSym), symbol; init)

ParamNode(lambda::Function, input::SingleParam{T}, symbol::Symbol; 
          init::T=zero(T)) where {T} = 
ParamNode(StableReduce(T, lambda), fill(input), symbol, zero(T), init)

ParamNode(lambda::Function, var::T, varSym::Symbol, symbol::Symbol; 
          init::T=zero(T)) where {T<:Real} = 
ParamNode(lambda, NodeVar(var, varSym), symbol; init)

ParamNode(input::NodeVar{T}, symbol::Symbol=symOf(input)) where {T} = 
ParamNode(StableReduce(T, itself), fill(input), symbol, zero(T))

ParamNode(var::Real, varSym::Symbol, symbol::Symbol=varSym) = 
ParamNode(NodeVar(var, varSym), symbol)

ParamNode(var::ParamNode{T}, symbol::Symbol=symOf(var); init::T=var.memory) where {T} = 
ParamNode(var.lambda, var.input, symbol, var.offset, init, var.screen)

const TensorInParamNode{T, F, PB, N} = ParamNode{T, F, <:AbstractArray{PB, N}}
const ScalarInParamNode{T, F, PB} = TensorInParamNode{T, F, PB, 0}


screenLevelOf(pn::ParamNode) = Int(pn.screen)

screenLevelOf(::NodeVar) = 1

function setScreenLevelCore!(pn::ParamNode, level::Int)
    @atomic pn.screen = TernaryNumber(level)
end

function setScreenLevel!(pn::ParamNode, level::Int)
    levelOld = screenLevelOf(pn)
    if levelOld == level
    elseif levelOld == 0
        @atomic pn.offset = obtain(pn)
    elseif level == 0
        @atomic pn.offset -= pn.lambda(map(obtain, pn.input))
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


struct NodeTuple{T<:Real, N, PT<:NTuple{N, ParamNode{T}}} <: ParamBox{T, PT, N}
    input::PT
    symbol::IndexedSym

    NodeTuple(input::PT, symbol::SymOrIdxSym) where {T, N, PT<:NTuple{N, ParamNode{T}}} = 
    new{T, N, PT}(input, IndexedSym(symbol))
end

NodeTuple(nt::NodeTuple, symbol::Symbol) = NodeTuple(nt.input, symbol)


idxSymOf(pc::ParamContainer) = pc.symbol

symOf(pc::ParamContainer) = idxSymOf(pc).name

inputOf(pb::ParamBox) = pb.input


obtain(sv::PrimitiveParam) = sv.value

function obtain(pn::ParamNode{T}, fallbackVal::T=pn.memory) where {T}
    idSet = Set{UInt}(pn|>objectid)
    obtainCore(Val(false), pn, idSet, fallbackVal)
end

function obtainCore(::Val{BL}, pn::ParamNode{T}, 
                       idSet::Set{UInt64}, fallbackVal::T) where {BL, T}
    res = pn.offset
    sl = screenLevelOf(pn)
    if sl == 0
        id = objectid(pn)
        if BL && id in idSet
            res = fallbackVal
        else
            push!(idSet, id)
            res += map(pn.input) do child
                obtainCore(Val(true), child, idSet, fallbackVal)
            end |> pn.lambda
        end
    elseif sl == -1
        res = fallbackVal
    end
    res
end

obtainCore(::Val, par::PrimitiveParam{T},::Set{UInt}, ::T) where {T} = obtain(par)

# To be deprecated
obtain(nt::NodeTuple) = obtain.(nt.input)

(pn::ParamContainer)() = obtain(pn)

function setVal!(par::PrimitiveParam{T, 0}, val::T) where {T}
    @atomic par.value = val
end

function setVal!(par::PrimitiveParam{T, N}, val::AbstractArray{T, N}) where {T, N}
    safelySetVal!(par.value, val)
end

function setVal!(par::ParamNode{T}, val::T) where {T}
    isPrimitiveParam(par) || 
    throw(AssertionError("Input `par` does not behave like a primitive parameter."))
    @atomic par.offset = val
end

isPrimitiveParam(pn::ParamBox) = (screenLevelOf(pn) == 1)

# import Base: iterate, size, length, eltype, broadcastable
# length(::FixedSizeParam{<:Any, N}) where {N} = N
# eltype(np::FixedSizeParam) = eltype(np.input)

# iterate(::FixedSizeParam{<:Any, 1}, args...) = iterate(1, args...)
# size(::FixedSizeParam{<:Any, 1}, args...) = size(1, args...)
# broadcastable(np::FixedSizeParam{<:Any, 1}) = Ref(np)

# iterate(np::FixedSizeParam, args...) = iterate(np.input, args...)
# size(np::FixedSizeParam, args...) = size(np.input, args...)
# broadcastable(np::FixedSizeParam) = Base.broadcastable(np.input)


struct Marker <: AbstractMarker{UInt}
    typeID::UInt
    valueID::UInt

    Marker(input::T) where {T} = new(objectid(T), objectid(input))
end

struct ContainerMarker{N, M<:NTuple{N, AbstractMarker{UInt}}} <: AbstractMarker{UInt}
    typeID::UInt
    dataMarker::M
    funcID::UInt
    metaID::UInt
end

const NothingID = objectid(nothing)

function ContainerMarker(sv::T) where {T<:NodeVar}
    m = Marker(sv.value)
    ContainerMarker{1, Tuple{Marker}}(objectid(T), (m,), NothingID, NothingID)
end

function ContainerMarker(pn::T) where {T<:ParamNode}
    m = Marker(pn.input)
    ContainerMarker{1, Tuple{Marker}}(
        objectid(T), (m,), objectid(pn.lambda.f), 
        objectid((pn.offset, screenLevelOf(pn)))
    )
end

function ContainerMarker(nt::T) where {N, T<:NodeTuple{<:Any, N}}
    ms = ContainerMarker.(nt.input)
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
ParamNode(op∘pn.lambda.f, pn.input, symbol, pn.offset, pn.memory, pn.screen)

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
    p isa SingleParam ? getParCore(Val(1), p, sym) : getParsCore(Val(1), p.input, sym)
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


getSingleParams(input, sym::SymOrMiss=missing) = getParsCore(Val(1), input, sym)

getParams(input, sym::SymOrMiss=missing) = getParsCore(Val(2), input, sym)

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
                for child in node.input
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
sever(pv::NodeVar) = NodeVar(obtain(pv), pv.symbol)

sever(pn::ParamNode) = ParamNode(obtain(pn), symOf(pn))

sever(ps::T) where {T<:ParamBox} = T(sever.(ps.input), ps.symbol)

sever(obj::Any) = deepcopy(obj)

sever(obj::Union{Tuple, AbstractArray}) = sever.(obj)

function sever(pf::T) where {T<:ParamFunction}
    severedFields = map(fieldnames(pf)) do field
        (sever∘getproperty)(pf, field)
    end
    T(severedFields...)
end