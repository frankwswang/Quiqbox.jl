export NodeVar, GridVar, NodeParam, ArrayParam, NodeTuple, setScreenLevel!, setScreenLevel, 
       symOf, inputOf, obtain, setVal!, screenLevelOf, markParams!, topoSort, getParams

using Base: Fix2, Threads.Atomic, issingletontype
using LRUCache

function checkReturnType(f::F, ::Type{T}, argTs::Tuple{Vararg{DataType}}) where {F, T}
    bl = false
    returnT = Any
    try
        returnT = Base.return_types(f, argTs)[]
        bl = returnT <: T
    finally
        bl || throw(AssertionError("`f`: `$f` cannot be a generated function and it "*
                                    "should only return one value of `$T.`"))
    end
    returnT
end

struct TypedReduction{T, F<:Function} <: TypedFunction{T, F}
    f::F

    function TypedReduction(f::F, aT::Type{T}, aTs::Type...) where {F, T}
        Ts = (aT, aTs...)
        checkReturnType(f, T, Ts)
        new{T, F}(f)
    end

    function TypedReduction(f::F, aT::Type{<:AbstractArray{T}}, aTs::Type...) where {F, T}
        Ts = (aT, aTs...)
        checkReturnType(f, T, Ts)
        new{T, F}(f)
    end
end

# TypedReduction(::Type{<:Union{T, AbstractArray{T}}}, srf::TypedReduction{T}) where {T} = srf
# TypedReduction(::Type{AT}, srf::TypedReduction{T1}) where 
#             {T1, T2, AT<:Union{T2, AbstractArray{T2}}} = 
# TypedReduction(srf.f, AT)

TypedReduction(::Type{T}) where {T} = TypedReduction(itself, T)

# Type annotation prevents Enzyme.jl (v0.12.22) from breaking sometimes.
function (sf::TypedReduction{T, F})(arg1::AbtArrayOr{T}) where {T, F}
    sf.f(arg1)::T
end

# Type annotation prevents Enzyme.jl (v0.12.22) from breaking sometimes.
function (sf::TypedReduction{T, F})(arg1::AbtArrayOr{T}, 
                                    arg2::AbtArrayOr{T}) where {T, F}
    sf.f(arg1, arg2)::T
end

function (sf::TypedReduction{T, F})(arg1::AbtArrayOr{T}, 
                                    arg2::AbtArrayOr{T}, 
                                    arg3::AbtArrayOr{T}) where {T, F}
    sf.f(arg1, arg2, arg3)::T
end


struct StableMorphism{T, F<:Function, N} <:TypedFunction{T, F}
    f::F

    function StableMorphism(f::F, aT::Type{T}, aTs::Type...) where {T, F}
        Ts = (aT, aTs...)
        rT = checkReturnType(f, AbstractArray{T}, Ts)
        new{T, F, ndims(rT)}(f)
    end

    function StableMorphism(f::F, aT::Type{<:AbstractArray{T}}, aTs::Type...) where {T, F}
        Ts = (aT, aTs...)
        rT = checkReturnType(f, AbstractArray{T}, Ts)
        new{T, F, ndims(rT)}(f)
    end

    StableMorphism(::Type{T}, ::Val{N}) where {T, N} = new{T, iT, N}(itself)
end

function (sf::StableMorphism{T, F, N})(arg1::AbtArrayOr{T}) where {T, F, N}
    sf.f(arg1)::AbstractArray{T, N}
end

function (sf::StableMorphism{T, F, N})(arg1::AbtArrayOr{T}, 
                                       arg2::AbtArrayOr{T}) where {T, F, N}
    sf.f(arg1, arg2)::AbstractArray{T, N}
end

function (sf::StableMorphism{T, F, N})(arg1::AbtArrayOr{T}, 
                                       arg2::AbtArrayOr{T}, 
                                       arg3::AbtArrayOr{T}) where {T, F, N}
    sf.f(arg1, arg2, arg3)::AbstractArray{T, N}
end

returnDimOf(::Type{<:StableMorphism{<:Any, <:Any, N}}) where {N} = N
returnDimOf(::T) where {T<:StableMorphism} = returnDimOf(T)

const SymOrIdxSym = Union{Symbol, IndexedSym}


function checkScreenLevel(sl::Int, (levelMin, levelMax)::NTuple{2, Int})
    levelRange = levelMax - levelMin
    levelRange < 0 && 
    throw(DomainError(levelRange, "`levelMax - levelMin` must be nonnegative."))
    if !(levelMin <= sl <= levelMax)
        throw(DomainError(sl, "This screen level ($(TernaryNumber(sl))) is not allowed."))
    end
    sl
end

checkScreenLevel(s::TernaryNumber, levelMinMax::NTuple{2, Int}) = 
checkScreenLevel(Int(s), levelMinMax)

function checkPrimParamType(::Type{T}) where {T}
    isPermitted = isprimitivetype(T) || issingletontype(T) || isbitstype(T)
    if !isPermitted
        throw(DomainError(T, "The (elemental) type of `input`, when used as an argument, "*
                             "should make at least one of these functions return `true`:\n"*
                             "`$isprimitivetype`, `$issingletontype`, `$isbitstype`."))
    end
    nothing
end

#! Should be remove. Cause recursive self-reference
checkPrimParamType(::Type{<:PrimitiveParam{T}}) where {T} = checkPrimParamType(T)

checkPrimParamType(::PT) where {PT<:PrimitiveParam} = checkPrimParamType(PT)

mutable struct NodeVar{T} <: PrimitiveParam{T, 0}
    @atomic input::T
    const symbol::IndexedSym
    @atomic screen::TernaryNumber

    function NodeVar(input::T, symbol::SymOrIdxSym, screen::TernaryNumber=TPS1) where {T}
        checkPrimParamType(T)
        checkScreenLevel(screen, getScreenLevelRange(PrimitiveParam))
        new{T}(input, IndexedSym(symbol), screen)
    end
end

NodeVar(::AbstractArray, ::SymOrIdxSym) = 
throw(ArgumentError("`NodeVar` does not support `AbstractArray`-type `input`."))

# genSeqAxis(::Val{N}, sym::Symbol=:e) where {N} = (Symbol(sym, i) for i in 1:N) |> Tuple

mutable struct GridVar{T, N, V<:AbstractArray{T, N}} <: PrimitiveParam{T, N}
    const input::V
    const symbol::IndexedSym
    @atomic screen::TernaryNumber

    function GridVar(input::V, symbol::SymOrIdxSym, screen::TernaryNumber=TPS1) where 
                    {T, N, V<:AbstractArray{T, N}}
        checkPrimParamType(T)
        checkScreenLevel(screen, getScreenLevelRange(PrimitiveParam))
        N < 1 && throw(DomainError(N, "The dimension of `input` must be larger than 0."))
        new{T, N, V}(copy(input), IndexedSym(symbol), screen)
    end
end

getParamBoxArgDim(::Type{<:ParamBoxSingleArg{<:Any, N}}) where {N} = N
getParamBoxArgDim(::T) where {T<:ParamBoxSingleArg} = getParamBoxArgDim(T)

function checkParamBoxInput(input::ParamBoxInputType; dimMin::Int=0, dimMax=64)
    hasVariable = false
    for x in input
        hasVariable = checkParamBoxInputCore(hasVariable, x, (dimMin, dimMax))
    end
    if !hasVariable
        throw(ArgumentError("`input` must contain as least one non-constant parameter."))
    end
    nothing
end

function checkParamBoxInputCore(hasVariable::Bool, arg::T, dimMinMax::NTuple{2, Int}) where 
                               {T<:ParamBoxSingleArg}
    nDim = getParamBoxArgDim(arg)
    if !(dimMinMax[begin] <= nDim <= dimMinMax[end])
        throw(DomainError(nDim, "The input `arg`'s dimension falls outside the "*
                                "permitted range: $dimMinMax."))
    end
    if T <: AbstractArray
        if isempty(arg)
            throw(ArgumentError("Every `AbstractArray` in `arg` must be non-empty."))
        elseif !hasVariable && any( (screenLevelOf(y) < 2) for y in arg )
            hasVariable = true
        end
    elseif !hasVariable && screenLevelOf(arg) < 2
        hasVariable = true
    end
    hasVariable::Bool
end

#! Need to reformulate following `checkArrayParamArg`
mutable struct NodeParam{T, F<:Function, I<:ParamBoxInputType{T}} <: ParamBox{T, 0, I}
    const lambda::TypedReduction{T, F}
    const input::I
    const symbol::IndexedSym
    @atomic offset::T
    @atomic memory::T
    @atomic screen::TernaryNumber

    function NodeParam(lambda::TypedReduction{T, F}, input::I, 
                       symbol::SymOrIdxSym, 
                       offset::T, 
                       memory::T=zero(T), 
                       screen::TernaryNumber=TUS0) where {T, I, F}
        checkScreenLevel(screen, getScreenLevelRange(ParamBox{T, 0}))
        checkParamBoxInput(input)
        new{T, F, I}(lambda, input, IndexedSym(symbol), offset, memory, screen)
    end

    function NodeParam(::TypedReduction{T, <:iTalike}, input::I, 
                       symbol::SymOrIdxSym, 
                       offset::T, 
                       memory::T=obtain(input|>first), 
                       screen::TernaryNumber=TUS0) where 
                      {T, I<:Tuple{ElementalParam{T}}}
        checkScreenLevel(screen, getScreenLevelRange(ParamBox{T, 0}))
        checkParamBoxInput(input, dimMax=0)
        new{T, iT, I}(TypedReduction(T), input, IndexedSym(symbol), offset, memory, screen)
    end
end

function NodeParam(::TypedReduction{T, <:iTalike}, ::I, ::SymOrIdxSym, ::T, 
                   ::T=zero(T), ::TernaryNumber=TUS0) where {T, I}
    throw(ArgumentError("`$I` is not supported as the second argument when `lambda` "*
                        "functions like an identity morphism."))
end

function NodeParam(lambda::Function, input::ParamBoxInputType{T}, 
                   symbol::Symbol; init::T=zero(T)) where {T}
    ATs = map(x->typeof(x|>obtain), input)
    NodeParam(TypedReduction(lambda, ATs...), input, symbol, zero(T), init)
end

NodeParam(lambda::Function, input::ParamBoxSingleArg{T}, symbol::Symbol; 
          init::T=zero(T)) where {T} = 
NodeParam(lambda, (input,), symbol; init)

NodeParam(lambda::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
          symbol::Symbol; init::T=zero(T)) where {T} = 
NodeParam(lambda, (input1, input2), symbol; init)

NodeParam(lambda::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
          input3::ParamBoxSingleArg{T}, symbol::Symbol; init::T=zero(T)) where {T} = 
NodeParam(lambda, (input1, input2, input3), symbol; init)

NodeParam(par::NodeParam{T}, symbol::Symbol=symOf(par); init::T=par.memory) where {T} = 
NodeParam(par.lambda, par.input, symbol, par.offset, init, par.screen)

NodeParam(var::PrimitiveParam{T, 0}, symbol::Symbol=symOf(var)) where {T} = 
NodeParam(TypedReduction(T), (var,), symbol, zero(T))

NodeParam(var, varSym::Symbol, symbol::Symbol=varSym) = 
NodeParam(NodeVar(var, varSym), symbol)

# const TensorInNodeParam{T, F, PB, N} = NodeParam{T, F, <:AbstractArray{PB, N}}
# const ScalarInNodeParam{T, F, PB} = TensorInNodeParam{T, F, PB, 0}


function prepareArrayInit(::Type{T}, ::Val{N}, init::T=zero(T)) where {N, T}
    reshape(fill(init), ntuple(_->1, Val(N)))
end

function prepareArrayInit(::Type{T}, ::Val{N}, init::AbstractArray{T, N}) where {N, T}
    copy(init)
end

function checkArrayParamArg(::StableMorphism{T, <:iTalike, N}, input::I, 
                            init::Union{Missing, AbstractArray{T, N}}) where {T, N, I}
    if !(I <: Tuple{ParamBoxSingleArg{T, N}})
        throw(ArgumentError("`$I` is not supported as the second argument when `lambda` "*
                            "functions like an identity morphism."))
    end
    checkParamBoxInput(input, dimMin=1)
    ismissing(init) ? (input |> first |> obtain) : deepcopy(init)
end

function checkArrayParamArg(::StableMorphism{T, F, N}, input::I, 
                            init::Union{Missing, AbstractArray{T, N}}) where {T, F, N, I}
    checkParamBoxInput(input)
    if ismissing(init)
        reshape(fill(input|>first|>obtain|>first|>deepcopy), ntuple(_->1, Val(N)))
    else
        deepcopy(init)
    end
end

mutable struct ArrayParam{T, F<:Function, I<:ParamBoxInputType{T}, 
                          N, M<:AbstractArray{T, N}} <: ParamBox{T, N, I}
    const lambda::StableMorphism{T, F, N}
    const input::I
    const symbol::IndexedSym
    @atomic memory::M

    function ArrayParam(lambda::StableMorphism{T, F, N}, input::I, 
                        symbol::SymOrIdxSym, 
                        memory::Union{Missing, AbstractArray{T, N}}=missing) where 
                       {T, F, N, I<:ParamBoxInputType{T}}
        memory = checkArrayParamArg(lambda, input, memory)
        new{T, F, I, N, typeof(memory)}(lambda, input, IndexedSym(symbol), memory)
    end
end

function ArrayParam(lambda::Function, input::ParamBoxInputType{T}, symbol::SymOrIdxSym; 
                    init::Union{Missing, AbstractArray{T}}=missing) where {T}
    ATs = map(x->typeof(x|>obtain), input)
    lambda = StableMorphism(lambda, ATs...)
    ArrayParam(lambda, input, symbol, init)
end

ArrayParam(lambda::Function, input::ParamBoxSingleArg{T}, symbol::SymOrIdxSym; 
           init::Union{Missing, AbstractArray{T}}=missing) where {T} = 
ArrayParam(lambda, (input,), symbol; init)

ArrayParam(lambda::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
           symbol::SymOrIdxSym; init::Union{Missing, AbstractArray{T}}=missing) where {T} = 
ArrayParam(lambda, (input1, input2), symbol; init)

ArrayParam(lambda::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
           input3::ParamBoxSingleArg{T}, symbol::SymOrIdxSym; 
           init::Union{Missing, AbstractArray{T}}=missing) where {T} = 
ArrayParam(lambda, (input1, input2, input3), symbol; init)

ArrayParam(par::ArrayParam{T}, symbol::SymOrIdxSym=symOf(par); 
           init::Union{Missing, AbstractArray{T}}=par.memory) where {T} = 
ArrayParam(par.lambda, par.input, symbol, init)

ArrayParam(input::PrimitiveParam{T, N}, symbol::SymOrIdxSym=symOf(input)) where {T, N} = 
ArrayParam(StableMorphism(T, Val(N)), (input,), symbol)

ArrayParam(input::AbstractArray{<:ElementalParam{T}, N}, symbol::SymOrIdxSym) where {T, N} = 
ArrayParam(StableMorphism(T, Val(N)), (input,), symbol)

ArrayParam(val::AbstractArray, valSym::SymOrIdxSym, symbol::SymOrIdxSym=valSym) = 
ArrayParam(GridVar(val, valSym), symbol)


#!! Should disallow NodeVar of NodeVar etc... , Instead, design ReferenceNode

# struct ChainParam

# end

# ChainParam -> NodeVar of NodeVar

# struct ReferenceParam

# end

getScreenLevelRange(::Type{<:ParamBox}) = (0, 0)

getScreenLevelRange(::Type{<:ParamBox{<:Any, 0}}) = (0, 2)

getScreenLevelRange(::Type{<:PrimitiveParam}) = (1, 2)

getScreenLevelRange(::Type{<:DimensionalParam}) = (0, 2)

getScreenLevelRange(::T) where {T<:DimensionalParam} = getScreenLevelRange(T)


screenLevelOf(::ParamBox) = 0

screenLevelOf(p::ParamBox{<:Any, 0}) = Int(p.screen)

screenLevelOf(p::PrimitiveParam) = Int(p.screen)


function setScreenLevelCore!(p::DimensionalParam, level::Int)
    @atomic p.screen = TernaryNumber(level)
end

function setScreenLevel!(p::NodeParam, level::Int)
    levelOld = screenLevelOf(p)
    if levelOld == level
    elseif levelOld == 0
        @atomic p.offset = obtain(p)
    elseif level == 0
        @atomic p.offset -= p.lambda((obtain(arg) for arg in p.input)...)
    end
    setScreenLevelCore!(p, level)
    p
end

setScreenLevel!(p::PrimitiveParam, level::Int) = setScreenLevelCore!(p, level)


setScreenLevel(pn::NodeParam, level::Int) = setScreenLevel!(NodeParam(pn), level)

setScreenLevel(p::NodeVar, level::Int) = NodeVar(p.input, p.symbol, TernaryNumber(level))

setScreenLevel(p::GridVar, level::Int) = GridVar(p.input, p.symbol, TernaryNumber(level))


function memorize!(pn::ParamBox{T}, newMem::T=ValOf(pn)) where {T}
    oldMem = pn.memory
    @atomic pn.memory = newMem
    oldMem
end


struct NodeTuple{T<:Real, N, PT} <: ParamBox{T, N, PT}
    input::PT
    symbol::IndexedSym

    NodeTuple(input::PT, symbol::SymOrIdxSym) where {T, N, PT<:NTuple{N, NodeParam{T}}} = 
    new{T, N, PT}(input, IndexedSym(symbol))
end

NodeTuple(nt::NodeTuple, symbol::Symbol) = NodeTuple(nt.input, symbol)


indexedSymOf(pc::DimensionalParam) = pc.symbol

symOf(pc::DimensionalParam) = indexedSymOf(pc).name

inputOf(pb::DimensionalParam) = pb.input


mutable struct NodeMarker{T}
    visited::Bool
    value::T
end

NodeMarker(init::T, ::Type{U}=T) where {T, U} = NodeMarker{U}(false, init)

obtain(p::PrimitiveParam) = obtainINTERNAL(p)

function obtain(p::Union{ParamBox{T}, AbstractArray{<:ElementalParam{T}}}) where {T}
    lock( ReentrantLock() ) do
        obtainINTERNAL(p)
    end
end

# Sugar syntax. E.g., for obtaining values of the first element in a parameter set.
obtainINTERNAL(pars::AbstractArray{<:ElementalParam{T}}) where {T} = obtainINTERNAL.(pars)

obtainINTERNAL(p::PrimitiveParam) = directObtain(p)

directObtain(p::PrimitiveParam{<:Any, 0}) = p.input

directObtain(p::PrimitiveParam) = copy(p.input)

function obtainINTERNAL(p::ParamBox{T}) where {T}
    nodeMarkerDict = IdDict{ParamBox{T}, NodeMarker}()
    searchObtain(nodeMarkerDict, p)
end

function searchObtain(nodeMarkerDict::IdDict{ParamBox{T}, NodeMarker}, 
                      input::AbstractArray{<:DimensionalParam{T}}) where {T}
    map(input) do child
        searchObtain(nodeMarkerDict, child)
    end
end

function obtainSearchCore(nodeMarkerDict::IdDict{ParamBox{T}, NodeMarker}, 
                          p::ParamBox{T}) where {T}
    f = hasfield(typeof(p), :offset) ? Fix2(+, p.offset) : itself
    # Depth-first search by recursive calling
    marker = get!(nodeMarkerDict, p, NodeMarker(p.memory))
    if !marker.visited
        marker.visited = true
        res = p.lambda((searchObtain(nodeMarkerDict, x) for x in p.input)...) |> f
        marker.value = res
    else
        marker.value
    end
end

function searchObtain(nodeMarkerDict::IdDict{ParamBox{T}, NodeMarker}, 
                      p::ArrayParam{T}) where {T}
    obtainSearchCore(nodeMarkerDict, p)
end

function searchObtain(nodeMarkerDict::IdDict{ParamBox{T}, NodeMarker}, 
                      p::NodeParam{T}) where {T}
    screenLevelOf(p) == 0 ? obtainSearchCore(nodeMarkerDict, p) : p.offset
end

function searchObtain(::IdDict{ParamBox{T}, NodeMarker}, p::PrimitiveParam{T}) where {T}
    obtainINTERNAL(p)
end

# To be deprecated
obtain(nt::NodeTuple) = obtain.(nt.input)

(pn::DimensionalParam)() = obtain(pn)

function setVal!(par::PrimitiveParam{T, 0}, val::T) where {T}
    @atomic par.input = val
end

function setVal!(par::PrimitiveParam{T, N}, val::AbstractArray{T, N}) where {T, N}
    if Int(par.screen) == 1
        safelySetVal!(par.input, val)
    else
        throw(ArgumentError("`par` is a constant parameter that should not be modified."))
    end
end

function setVal!(par::NodeParam{T}, val::T) where {T}
    isPrimitiveParam(par) || 
    throw(ArgumentError("`par` must behave like a primitive parameter."))
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


#!!!!! Marker of ParamBox is not defined

struct ParamMarker{M<:NonEmptyTuple{AbstractMarker}} <: AbstractMarker{M}
    typeID::UInt
    marker::M
    funcID::UInt
    metaID::UInt
end

struct ValueMarker <: AbstractMarker{UInt}
    valueID::UInt

    ValueMarker(input) = new(objectid(input))
end

struct CollectionMarker <: AbstractMarker{Union{AbstractArray, Tuple}}
    data::Union{AbstractArray, Tuple}
end

struct ObjectMarker{T} <: AbstractMarker{T}
    data::T
end

markObj(input::PrimitiveParam) = ValueMarker(input)

markObj(input::DimensionalParam) = ParamMarker(input)

function isPrimVarCollection(arg::AbstractArray{T}) where {T}
    ET = isconcretetype(T) ? T : eltype(map(itself, arg))
    isprimitivetype(ET)
end

function isPrimVarCollection(arg::Tuple)
    all(isprimitivetype(i) for i in arg)
end

function markObj(input::Union{AbstractArray, Tuple})
    isPrimVarCollection(input) ? ValueMarker(input) : CollectionMarker(input)
end

markObj(input) = ObjectMarker(input)

const NothingID = objectid(nothing)

function ParamMarker(pn::T) where {T<:NodeParam}
    ParamMarker(
        objectid(T), markObj.((pn.input..., pn.offset)), objectid(pn.lambda.f), 
        objectid(screenLevelOf(pn))
    )
end

function ParamMarker(pn::T) where {T<:ArrayParam}
    ParamMarker(objectid(T), markObj.(pn.input), objectid(pn.lambda.f), NothingID)
end

#! To be discarded
function ParamMarker(nt::T) where {N, T<:NodeTuple{<:Any, N}}
    ms = ParamMarker.(nt.input)
    ParamMarker(objectid(T), ms, NothingID, NothingID)
end

compareMarker(pm1::AbstractMarker, pm2::AbstractMarker) = false

compareMarker(pm1::T, pm2::T) where {T<:ParamMarker{<:Tuple{Vararg{ValueMarker}}}} = 
pm1 == pm2

function compareMarker(pm1::T, pm2::T) where {T<:CollectionMarker}
    if pm1.data === pm2.data
        true
    elseif length(pm1.data) == length(pm2.data)
        isSame = true
        for (i, j) in zip(pm1.data, pm2.data)
            isSame = ( pm1.data === pm2.data || compareMarker(markObj(i), markObj(j)) )
            isSame || break
        end
        isSame
    else
        false
    end
end

compareMarker(pm1::T, pm2::T) where {T<:ObjectMarker} = pm1.data == pm2.data

function compareMarker(pm1::T, pm2::T) where {T<:ParamMarker}
    isSame = (pm1.funcID == pm2.funcID || pm1.metaID == pm2.metaID)
    if isSame
        for (marker1, marker2) in zip(pm1.marker, pm2.marker)
            isSame = compareMarker(marker1, marker2)
            isSame || break
        end
    end
    isSame
end

compareParamContainer(::DimensionalParam, ::DimensionalParam) = false

compareParamContainer(::DimensionalParam, ::Any) = false

compareParamContainer(::Any, ::DimensionalParam) = false

compareParamContainer(pc1::T, pc2::T) where {T<:PrimitiveParam} = pc1 === pc2

compareParamContainer(pc1::PBoxTypeArgNumOutDim{T, N, A}, 
                      pc2::PBoxTypeArgNumOutDim{T, N, A}) where {T, N, A} = 
pc1 === pc2 || compareMarker(ParamMarker(pc1), ParamMarker(pc2))


operateBy(::typeof(+), pn1::NodeParam) = itself(pn1)
operateBy(::typeof(-), pn1::NodeParam{T}) where {T} = operateBy(*, T(-1), pn1)

repeatedlyApply(::typeof(+), pn::NodeParam{T}, times::Int) where {T} = 
operateBy(*, pn, T(times))

repeatedlyApply(::typeof(-), pn::NodeParam{T}, times::Int) where {T} = 
operateBy(-, operateBy(*, pn, T(times)))

repeatedlyApply(::typeof(*), pn::NodeParam{T}, times::Int) where {T} = 
operateBy(^, pn, T(times))

operateBy(op::F, pn1::NodeParam, num::Real) where {F<:Function} = 
NodeParam(OFC(itself, op, num), pn1, pn1.symbol)

operateBy(op::F, num::Real, pn1::NodeParam) where {F<:Function} = 
NodeParam(OCF(itself, op, num), pn1, pn1.symbol)

operateBy(op::CommutativeBinaryNumOps, num::Real, pn1::NodeParam) = 
operateBy(op, pn1::NodeParam, num::Real)

operateBy(op::F, pn::NodeParam{T}) where {F<:Function, T} = 
NodeParam(op∘pn.lambda.f, pn.input, symbol, pn.offset, pn.memory, pn.screen)

operateByCore(op::F, pn1::NodeParam{T}, pn2::NodeParam{T}) where {F<:Function, T} = 
NodeParam(SplitArg{2}(op), [pn1, pn2], Symbol(pn1.symbol, pn2.symbol))

operateByCore(op::F, pn1::NodeParam{T}, pn2::NodeParam{T}, 
              pns::Vararg{NodeParam{T}, N}) where {F<:Function, T, N} = 
NodeParam(SplitArg{N}(op), [pn1, pn2, pns...], Symbol(pn1.symbol, :_to_, pns[end].symbol))

function operateBy(op::F, pn1::NodeParam{T}, pn2::NodeParam{T}) where 
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

operateBy(op::F, pn1::NodeParam{T}, pns::Vararg{NodeParam{T}, N}) where {F<:Function, T, N} = 
operateByCore(op, pn1, pns...)

operateBy(f::F, pns::AbstractArray{<:NodeParam{T}}) where {F<:Function, T} = 
NodeParam(f, pns, Symbol(pns[begin].symbol.name, :_to_, pns[end].symbol.name))

operateBy(f::F, ::Val{N}) where {F<:Function, N} = 
((args::Vararg{NodeParam{T}, N}) where {T}) -> operateBy(f, args...)

operateBy(f::F) where {F<:Function} = 
((args::AbstractArray{<:NodeParam{T}}) where {T}) -> operateBy(f, args)


addNodeParam(pn1::NodeParam{T}, pn2::NodeParam{T}) where {T} = operateBy(+, pn1, pn2)

mulNodeParam(pn1::NodeParam{T}, pn2::NodeParam{T}) where {T} = operateBy(*, pn1, pn2)
mulNodeParam(pn::NodeParam{T}, coeff::T) where {T} = operateBy(*, pn, coeff)
mulNodeParam(coeff::T, pn::NodeParam{T}) where {T} = mulNodeParam(pn, coeff)


function sortParamContainers(::Type{C}, f::F, field::Symbol, roundAtol::T) where 
                            {T, C<:ParamFunction{T}, F}
    let roundAtol=roundAtol, f=f, field=field
        function (container::C)
            ele = getproperty(container, field)
            ( roundToMultiOfStep(f(container), nearestHalfOf(roundAtol)), 
              symFromIndexSym(ele.symbol), ParamMarker(ele) )
        end
    end
end


function getParamFields(pf::T) where {T<:ParamFunction}
    fields = fieldnames(pf)
    ids = findall(x->(x isa ParamFunctions), fields)
    getproperty.(pf, fields[ids])
end

getParamFields(pf::ParamFunctions) = itself(pf)

getParams(p::DimensionalParam) = [p]

getParams(p::DimensionalParam, ::Missing) = [p]

function getParams(p::DimensionalParam, sym::Symbol)
    res = eltype(p)[]
    inSymbol(sym, symOf(ps)) && push!(res, p)
    res
end

function getParams(v::NonEmptyTupleOrAbtArray{T}, sym::SymOrMiss=missing) where 
                  {T<:Union{ParamContainer, ParamObject}}
    isempty(v) ? ParamContainer{T}[] : reduce(vcat, getParams.(v, sym))
end

function getParams(f::ParamFunction{T}, sym::SymOrMiss=missing) where {T}
    res = map(getParamFields(f)) do field
        getParams(field, sym)
    end
    len = length.(res) |> sum
    len == 0 ? ParamContainer{T}[] : reduce(vcat, res)
end


uniqueParams(ps::AbstractArray{<:DimensionalParam}) = 
markUnique(ps, compareFunction=compareParamContainer)[end]


function markParamsCore!(indexDict::IdDict{Symbol, Int}, leafPars)
    for i in leafPars
        sym = i.symbol.name
        get!(indexDict, sym, 0)
        i.symbol.index = (indexDict[sym] += 1)
    end
end

function markParams!(pars::AbstractVector{<:DimensionalParam{T}}) where {T}
    nodes, marks1, marks2 = topoSort(pars)
    leafPars = nodes[.!marks1 .*   marks2]
    rootPars = nodes[  marks1 .* .!marks2]
    selfPars = nodes[.!marks1 .* .!marks2]
    parIdxDict = IdDict{Symbol, Int}()

    par0Dids = findall(x->(x isa ElementalParam{T}), leafPars)
    leafParsFormated = if isempty(par0Dids)
        markParamsCore!(parIdxDict, leafPars)
        convert(Vector{PrimDParSetEltype{T}}, leafPars)
    else
        leafP0Ds = ElementalParam{T}[splice!(leafPars, par0Dids)...]
        markParamsCore!(parIdxDict, leafP0Ds)
        markParamsCore!(parIdxDict, leafPars)
        PrimDParSetEltype{T}[leafP0Ds, leafPars...]
    end
    (leafParsFormated, rootPars, selfPars)
end

markParams!(b::AbtArrayOr{<:ParamObject}) = markParams!(getParams(b))


function flattenPBoxInput(input::ParamBoxInputType{T}) where {T}
    mapreduce(vcat, input, init=DimensionalParam{T}[]) do parArg
        parArg isa DimensionalParam ? DimensionalParam{T}[parArg] : vec(parArg)
    end
end

function topoSortCore!(hbNodesIdSet::Set{UInt}, 
                       orderedNodes::Vector{<:DimensionalParam{T}}, 
                       haveBranches::Vector{Bool}, connectRoots::Vector{Bool}, 
                       node::DimensionalParam{T}, recursive::Bool=false) where {T}
    sl = checkScreenLevel(screenLevelOf(node), getScreenLevelRange(DimensionalParam{T}))

    if sl in (0, 1)
        idx = findfirst(Fix2(compareParamContainer, node), orderedNodes)
        if idx === nothing
            hasBranch = ifelse(sl == 0, true, false)
            if hasBranch
                id = objectid(node)
                isRegisteredSubRoot = (id in hbNodesIdSet)
                if !isRegisteredSubRoot
                    push!(hbNodesIdSet, objectid(node))
                    for child in flattenPBoxInput(node.input)
                        topoSortCore!(hbNodesIdSet, orderedNodes, haveBranches, 
                                      connectRoots, child, true)
                    end
                end
            end
            push!(orderedNodes, node)
            push!(haveBranches, hasBranch)
            push!(connectRoots, recursive)
        else
            connectRoots[idx] = recursive
        end
    end
    nothing
end

function topoSortINTERNAL(nodes::AbstractVector{<:DimensionalParam{T}}) where {T}
    orderedNodes = DimensionalParam{T}[]
    haveBranches = Bool[]
    connectRoots = Bool[]
    hbNodesIdSet = Set{UInt}()
    for node in nodes
        topoSortCore!(hbNodesIdSet, orderedNodes, haveBranches, connectRoots, node)
    end
    orderedNodes, haveBranches, connectRoots
end

function topoSort(nodes::AbstractVector{<:DimensionalParam{T}}) where {T}
    uniqueParams(nodes) |> topoSortINTERNAL
end

topoSort(node::NodeParam{T}) where {T} = topoSortINTERNAL([node])


# Sever the connection of a node to other nodes
sever(pv::NodeVar) = NodeVar(obtain(pv), pv.symbol)

# function sever(ps::T) where {T<:ParamBox}

#     T(sever.(ps.input), ps.symbol)
# end

sever(obj::Any) = deepcopy(obj)

sever(obj::Union{Tuple, AbstractArray}) = sever.(obj)

function sever(pf::T) where {T<:ParamFunction}
    severedFields = map(fieldnames(pf)) do field
        (sever∘getproperty)(pf, field)
    end
    T(severedFields...)
end