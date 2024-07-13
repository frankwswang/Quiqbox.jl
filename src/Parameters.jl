export NodeVar, GridVar, NodeParam, ArrayParam, NodeTuple, setScreenLevel!, setScreenLevel, 
       symOf, inputOf, obtain, setVal!, screenLevelOf, markParams!, topoSort, getParams

using Base: Fix2, Threads.Atomic
using LRUCache

unpackAA0Dtype(arg::Type{<:AbtArray0D}) = eltype(arg)
unpackAA0Dtype(arg::Type) = itself(arg)

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
        Ts = (aT, aTs...) .|> unpackAA0Dtype
        checkReturnType(f, T, Ts)
        new{T, F}(f)
    end

    function TypedReduction(f::F, aT::Type{<:AbstractArray{T}}, aTs::Type...) where {F, T}
        Ts = (aT, aTs...) .|> unpackAA0Dtype
        checkReturnType(f, T, Ts)
        new{T, F}(f)
    end
end

# TypedReduction(::Type{<:Union{T, AbstractArray{T}}}, srf::TypedReduction{T}) where {T} = srf
# TypedReduction(::Type{AT}, srf::TypedReduction{T1}) where 
#             {T1, T2, AT<:Union{T2, AbstractArray{T2}}} = 
# TypedReduction(srf.f, AT)

unpackAA0D(@nospecialize(arg::AbtArray0D)) = getindex(arg)
unpackAA0D(@nospecialize(arg::Any)) = itself(arg)

TypedReduction(::Type{T}) where {T} = TypedReduction(itself, T)

# Type annotation prevents Enzyme.jl (v0.12.22) from breaking sometimes.
function (sf::TypedReduction{T, F})(arg1::AbtArrayOr{T}) where {T, F}
    sf.f(unpackAA0D(arg1))::T
end

# Type annotation prevents Enzyme.jl (v0.12.22) from breaking sometimes.
function (sf::TypedReduction{T, F})(arg1::AbtArrayOr{T}, 
                                    arg2::AbtArrayOr{T}) where {T, F}
    sf.f(unpackAA0D(arg1), unpackAA0D(arg2))::T
end

function (sf::TypedReduction{T, F})(arg1::AbtArrayOr{T}, 
                                    arg2::AbtArrayOr{T}, 
                                    arg3::AbtArrayOr{T}) where {T, F}
    sf.f(unpackAA0D(arg1), unpackAA0D(arg2), unpackAA0D(arg3))::T
end


struct StableMorphism{T, F<:Function, N} <:TypedFunction{T, F}
    f::F

    function StableMorphism(f::F, aT::Type{T}, aTs::Type...) where {T, F}
        Ts = (aT, aTs...) .|> unpackAA0Dtype
        rT = checkReturnType(f, AbstractArray{T}, Ts)
        new{T, F, ndims(rT)}(f)
    end

    function StableMorphism(f::F, aT::Type{<:AbstractArray{T}}, aTs::Type...) where {T, F}
        Ts = (aT, aTs...) .|> unpackAA0Dtype
        rT = checkReturnType(f, AbstractArray{T}, Ts)
        new{T, F, ndims(rT)}(f)
    end

    StableMorphism(::Type{T}, ::Val{N}) where {T, N} = new{T, iT, N}(itself)
end

function (sf::StableMorphism{T, F, N})(arg1::AbtArrayOr{T}) where {T, F, N}
    sf.f(unpackAA0D(arg1))::AbstractArray{T, N}
end

function (sf::StableMorphism{T, F, N})(arg1::AbtArrayOr{T}, 
                                       arg2::AbtArrayOr{T}) where {T, F, N}
    sf.f(unpackAA0D(arg1), unpackAA0D(arg2))::AbstractArray{T, N}
end

function (sf::StableMorphism{T, F, N})(arg1::AbtArrayOr{T}, 
                                       arg2::AbtArrayOr{T}, 
                                       arg3::AbtArrayOr{T}) where {T, F, N}
    sf.f(unpackAA0D(arg1), unpackAA0D(arg2), unpackAA0D(arg3))::AbstractArray{T, N}
end

returnDimOf(::Type{<:StableMorphism{<:Any, <:Any, N}}) where {N} = N
returnDimOf(::T) where {T<:StableMorphism} = returnDimOf(T)

const SymOrIdxSym = Union{Symbol, IndexedSym}

genDParamSelfRefErrorMessage(::Type{T}) where {T<:DimensionalParam} = 
"`$T`is forbidden to directly (self) reference another DimensionalParam."

function checkScreenLevel(sl::Int, levelMin::Int, levelMax::Int)
    levelRange = levelMax - levelMin
    levelRange < 0 && 
    throw(DomainError(levelRange, "`levelMax - levelMin` must be nonnegative."))
    if !(levelMin <= sl <= levelMax)
        throw(DomainError(sl, "This screen level ($(TernaryNumber(sl))) is not allowed."))
    end
    sl
end

checkScreenLevel(s::TernaryNumber, levelMin::Int, levelMax::Int) = 
checkScreenLevel(Int(s), levelMin, levelMax)


function checkGridVarInputType(input::AbstractArray{<:Any, N}) where {N}
    N < 1 && throw(DomainError(N, "The dimension of `input` must be larger than 0."))
    vals = flatten(input|>vec)
    isempty(vals) && throw(ArgumentError("`input` must not be empty, nor can it be a "*
                                         "collection of empty collections."))
    if any(i isa DimensionalParam for i in vals)
        throw(AssertionError(T |> genDParamSelfRefErrorMessage))
    end
    nothing
end

const DefaultGridVarInputTypes = Union{Number, Symbol, String, Bool}

checkGridVarInputTypes(::AbstractArray{<:DefaultGridVarInputTypes}) = nothing

mutable struct NodeVar{T} <: PrimitiveParam{T, 0}
    @atomic input::T
    const symbol::IndexedSym

    NodeVar(input::T, symbol::SymOrIdxSym) where {T} = new{T}(input, IndexedSym(symbol))
end

NodeVar(::DimensionalParam, ::SymOrIdxSym) = 
throw(AssertionError(NodeVar |> genDParamSelfRefErrorMessage))

NodeVar(::AbstractArray, ::SymOrIdxSym) = 
throw(ArgumentError("`NodeVar` does not support `AbstractArray`-type `input`."))

# genSeqAxis(::Val{N}, sym::Symbol=:e) where {N} = (Symbol(sym, i) for i in 1:N) |> Tuple

struct GridVar{T, N, V<:AbstractArray{T, N}} <: PrimitiveParam{T, N}
    input::V
    symbol::IndexedSym
    screen::TernaryNumber

    function GridVar(input::V, symbol::SymOrIdxSym, 
                     screen::TernaryNumber=TPS1) where {T, N, V<:AbstractArray{T, N}}
        checkScreenLevel(screen, 1, 2)
        checkGridVarInputType(input)
        new{T, N, V}(copy(input), IndexedSym(symbol), screen)
    end
end

GridVar(::AbstractArray{<:DimensionalParam}, ::SymOrIdxSym) = 
throw(AssertionError(GridVar |> genDParamSelfRefErrorMessage))

#! use flatten to check ParamContainer Referencing ParamContainer

# getLambdaArgNum(::Type{<:NETupleOfDimPar{<:Any, NMO}}) where {NMO} = NMO + 1
# getLambdaArgNum(::Type{<:ParamBox{<:Any, I}}) where {I} = getLambdaArgNum(I)
# getLambdaArgNum(::T) where {T} = getLambdaArgNum(T)

getParamBoxArgDim(::Type{<:ParamBoxSingleArg{<:Any, N}}) where {N} = N
getParamBoxArgDim(::T) where {T<:ParamBoxSingleArg} = getParamBoxArgDim(T)

function checkParamBoxInput(input::ParamBoxInputType; dimMin::Int=-0, dimMax=64)
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
    elseif T <: ElementalParam
        throw(ArgumentError("`arg::ElementalParam` must be inside an `AbstractArray`."))
    elseif !hasVariable && screenLevelOf(arg) < 2
        hasVariable = true
    end
    hasVariable::Bool
end

mutable struct NodeParam{T, F<:Function, I<:ParamBoxInputType{T}} <: ParamBox{T, I, 0}
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
        checkParamBoxInput(input)
        new{T, F, I}(lambda, input, IndexedSym(symbol), offset, memory, screen)
    end

    function NodeParam(::TypedReduction{T, <:iTalike}, input::I, 
                       symbol::SymOrIdxSym, 
                       offset::T, 
                       memory::T=obtain(first(input)[]), 
                       screen::TernaryNumber=TUS0) where 
                      {T, I<:Tuple{AbtArray0D{<:ElementalParam{T}}}}
        checkParamBoxInput(input, dimMax=0)
        new{T, iT, I}(TypedReduction(T), input, IndexedSym(symbol), offset, memory, screen)
    end
end

function NodeParam(::TypedReduction{T, <:iTalike}, ::I, ::SymOrIdxSym, ::T, 
                   ::T=zero(T), ::TernaryNumber=TUS0) where {T, I}
    throw(ArgumentError("`$I` is not supported as the second argument when `lambda` "*
                        "functions like an identity morphism."))
end

packElemParam(parArg::ElementalParam) = fill(parArg)
packElemParam(parArg::Any) = itself(parArg)

function NodeParam(lambda::Function, input::ParamBoxInputType{T}, 
                   symbol::Symbol; init::T=zero(T)) where {T}
    input = packElemParam.(input)
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
NodeParam(TypedReduction(T), (fill(var),), symbol, zero(T))

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

mutable struct ArrayParam{T, F<:Function, I<:ParamBoxInputType{T}, 
                          N, M<:AbstractArray{T, N}} <: ParamBox{T, I, N}
    const lambda::StableMorphism{T, F, N}
    const input::I
    const symbol::IndexedSym
    @atomic memory::M

    function ArrayParam(lambda::StableMorphism{T, F, N}, input::I, 
                        symbol::SymOrIdxSym, 
                        memory::M=prepareArrayInit(T, Val(N))) where {T, F, N, I, M}
        checkParamBoxInput(input)
        new{T, F, I, N, M}(lambda, input, IndexedSym(symbol), copy(memory))
    end

    function ArrayParam(::StableMorphism{T, <:iTalike, N}, input::I, 
                        symbol::SymOrIdxSym, 
                        memory::M=obtain(first(input))) where 
                       {T, N, I<:Tuple{ParamBoxSingleArg{T, N}}, M}
        checkParamBoxInput(input, dimMin=1)
        new{T, iT, I, N, M}(StableMorphism(T, Val(N)), input, IndexedSym(symbol), 
                            copy(memory))
    end
end

function ArrayParam(::StableMorphism{T, <:iTalike}, ::I, ::SymOrIdxSym, 
                   ::T=fill(zero(T))) where {T, I}
    throw(ArgumentError("`$I` is not supported as the second argument when `lambda` "*
                        "functions like an identity morphism."))
end

function ArrayParam(lambda::Function, input::ParamBoxInputType{T}, 
                    symbol::Symbol; init::AbtArrayOr{T}=zero(T)) where {T}
    input = packElemParam.(input)
    ATs = map(x->typeof(x|>obtain), input)
    lambda = StableMorphism(lambda, ATs...)
    ArrayParam(lambda, input, symbol, prepareArrayInit(T, Val(lambda|>returnDimOf), init))
end

ArrayParam(lambda::Function, input::ParamBoxSingleArg{T}, symbol::Symbol; 
           init::AbtArrayOr{T}=zero(T)) where {T} = 
ArrayParam(lambda, (input,), symbol; init)

ArrayParam(lambda::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
           symbol::Symbol; init::AbtArrayOr{T}=zero(T)) where {T} = 
ArrayParam(lambda, (input1, input2), symbol; init)

ArrayParam(lambda::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
           input3::ParamBoxSingleArg{T}, symbol::Symbol; 
           init::AbtArrayOr{T}=zero(T)) where {T} = 
ArrayParam(lambda, (input1, input2, input3), symbol; init)

ArrayParam(par::ArrayParam{T}, symbol::Symbol=symOf(par); 
           init::AbtArrayOr{T}=par.memory) where {T} = 
ArrayParam(par.lambda, par.input, symbol, init)

ArrayParam(input::PrimitiveParam{T, N}, symbol::Symbol=symOf(input)) where {T, N} = 
ArrayParam(StableMorphism(T, Val(N)), (input,), symbol)

ArrayParam(input::AbstractArray{<:ElementalParam{T}, N}, symbol::Symbol) where {T, N} = 
ArrayParam(StableMorphism(T, Val(N)), (input,), symbol)

ArrayParam(val::AbstractArray, valSym::Symbol, symbol::Symbol=valSym) = 
ArrayParam(Grid(val, valSym), symbol)





#!! Should disallow NodeVar of NodeVar etc... , Instead, design ReferenceNode

# struct ChainParam

# end

# ChainParam -> NodeVar of NodeVar

# struct ReferenceParam

# end


screenLevelOf(pn::ParamBox) = Int(pn.screen)

screenLevelOf(::PrimitiveParam{<:Any, 0}) = 1

screenLevelOf(pp::PrimitiveParam) = Int(pp.screen)


function setScreenLevelCore!(pn::ParamBox, level::Int)
    @atomic pn.screen = TernaryNumber(level)
end

function setScreenLevel!(pn::ParamBox, level::Int)
    levelOld = screenLevelOf(pn)
    if levelOld == level
    elseif levelOld == 0
        @atomic pn.offset = obtain(pn)
    elseif level == 0
        @atomic pn.offset -= pn.lambda((map(obtain, arg) for arg in pn.input)...)
    end
    setScreenLevelCore!(pn, level)
    pn
end


setScreenLevel(pn::NodeParam, level::Int) = setScreenLevel!(NodeParam(pn), level)

setScreenLevel(gv::GridVar, level::Int) = setScreenLevel!(GridVar(gv), level)


function memorize!(pn::ParamBox{T}, newMem::T=ValOf(pn)) where {T}
    oldMem = pn.memory
    @atomic pn.memory = newMem
    oldMem
end


struct NodeTuple{T<:Real, N, PT} <: ParamBox{T, PT, N}
    input::PT
    symbol::IndexedSym

    NodeTuple(input::PT, symbol::SymOrIdxSym) where {T, N, PT<:NTuple{N, NodeParam{T}}} = 
    new{T, N, PT}(input, IndexedSym(symbol))
end

NodeTuple(nt::NodeTuple, symbol::Symbol) = NodeTuple(nt.input, symbol)


idxSymOf(pc::DimensionalParam) = pc.symbol

symOf(pc::DimensionalParam) = idxSymOf(pc).name

inputOf(pb::DimensionalParam) = pb.input


obtain(nv::PrimitiveParam{<:Any, 0}) = nv.input

obtain(gv::PrimitiveParam) = copy(gv.input)

function obtain(pn::NodeParam{T}, fallbackVal::T=pn.memory) where {T}
    idSet = Set{UInt}()
    obtainCore(idSet, pn, fallbackVal)
end

function obtainCore(idSet::Set{UInt64}, input::AbstractArray{<:ElementalParam{T}}, 
                    fallbackVal::T) where {T}
    map(input) do child
        obtainCore(idSet, child, fallbackVal)
    end
end

function obtainCore(idSet::Set{UInt64}, pn::DimensionalParam{T}, fallbackVal::T) where {T}
    #!!!!
    T(0)
end

function obtainCore(idSet::Set{UInt64}, pn::NodeParam{T}, fallbackVal::T) where {T}
    res = pn.offset
    sl = screenLevelOf(pn)
    if sl == 0
        id = objectid(pn)
        if id in idSet
            res = fallbackVal
        else
            f = pn.lambda
            push!(idSet, id)
            res += f((obtainCore(idSet, x, fallbackVal) for x in pn.input)...)
        end
    elseif sl == -1
        res = fallbackVal
    end
    res
end

obtainCore(::Set{UInt}, par::PrimitiveParam{T}, ::T) where {T} = obtain(par)

obtain(pars::AbstractArray{<:ElementalParam{T}}) where {T} = obtain.(pars)

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


struct Marker <: AbstractMarker{UInt}
    typeID::UInt
    valueID::UInt

    Marker(input::T) where {T} = new(objectid(T), objectid(input))
end

#!!!!! Marker of ParamBox is not defined

struct ContainerMarker{N, M<:NTuple{N, AbstractMarker{UInt}}} <: AbstractMarker{UInt}
    typeID::UInt
    dataMarker::M
    funcID::UInt
    metaID::UInt
end

const NothingID = objectid(nothing)

function ContainerMarker(sv::T) where {T<:NodeVar}
    m = Marker(sv.input)
    ContainerMarker{1, Tuple{Marker}}(objectid(T), (m,), NothingID, NothingID)
end

function ContainerMarker(gv::T) where {T<:GridVar}
    m = Marker(gv.input)
    ContainerMarker{1, Tuple{Marker}}(objectid(T), (m,), NothingID, 
                                      objectid(screenLevelOf(gv)))
end

function ContainerMarker(pn::T) where {T<:NodeParam}
    ms = Marker.(pn.input)
    ContainerMarker{length(ms), typeof(ms)}(
        objectid(T), ms, objectid(pn.lambda.f), 
        objectid((pn.offset, screenLevelOf(pn)))
    )
end

function ContainerMarker(pn::T) where {T<:ArrayParam}
    ms = Marker.(pn.input)
    ContainerMarker{length(ms), typeof(ms)}(objectid(T), ms, objectid(pn.lambda.f))
end

function ContainerMarker(nt::T) where {N, T<:NodeTuple{<:Any, N}}
    ms = ContainerMarker.(nt.input)
    ContainerMarker{N, typeof(ms)}(objectid(T), ms, NothingID, NothingID)
end


compareParamContainer(::DimensionalParam, ::DimensionalParam) = false

compareParamContainer(pc1::T, pc2::T) where {T<:DimensionalParam} = 
pc1 === pc2 || ContainerMarker(pc1) == ContainerMarker(pc2)

compareParamContainer(::DimensionalParam, ::Any) = false

compareParamContainer(::Any, ::DimensionalParam) = false


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
    sl = checkScreenLevel(screenLevelOf(node), 0, 2)

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