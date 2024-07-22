export CellVar, GridVar, CellParam, GridParam, NodeTuple, setScreenLevel!, setScreenLevel, 
       symOf, inputOf, obtain, setVal!, screenLevelOf, markParams!, topoSort, getParams

using Base: Fix2, Threads.Atomic, issingletontype
using Test: @inferred
using LRUCache


struct ShapedMemory{T, N}
    value::Memory{T}
    shape::NTuple{N, Int}

    ShapedMemory(value::AbstractArray{T}, 
                 shape::Tuple{Vararg{Int}}=size(value)) where {T} = 
    new{T, length(shape)}(value, vec(shape))
end


const BasicBinaryOpTargets = Union{Number, Bool}

typedAdd(a::T, b::T) where {T<:BasicBinaryOpTargets} = a + b
typedAdd(a::NonEmptyTuple{T, N}, b::NonEmptyTuple{T, N}) where {T, N} = typedAdd.(a, b)

typedSub(a::T, b::T) where {T<:BasicBinaryOpTargets} = a - b
typedSub(a::NonEmptyTuple{T, N}, b::NonEmptyTuple{T, N}) where {T, N} = typedSub.(a, b)

function checkTypedOpMethods(::Type{T}) where {T}
    hasmethod(typedAdd, NTuple{2, T}) && hasmethod(typedSub, NTuple{2, T})
end

function checkTypedOpMethods(::Type{NonEmptyTuple{T, N}}) where {T, N}
    hasmethod(typedAdd, NTuple{2, T}) && hasmethod(typedSub, NTuple{2, T})
end

genValShifter(::Type, val::Nothing) = itself
genValShifter(::Type{T}, val::T) where {T} = Fix2(typedAdd, val)
genValShifter(::Type{Nothing}, ::Nothing) = 
throw(ArgumentError("`genValShifter` does not support generating shifter for `Nothing`."))
const ValShifter{T} = Fix2{typeof(typedAdd), T}


excludeAbtArray0D(::T) where {T} = excludeAbtArray0D(T)
excludeAbtArray0D(::Type) = nothing
excludeAbtArray0D(::Type{<:AbtArray0D}) = 
throw(ArgumentError("`AbstractArray{<:Any, 0}` is not allowed as an input argument."))

excludeAbtArray(::T) where {T} = excludeAbtArray(T)
excludeAbtArray(::Type) = nothing
excludeAbtArray(::Type{<:AbstractArray}) = 
throw(ArgumentError("`AbstractArray` is not allowed as an input argument."))

function checkAbtArrayInput(::Type{V}) where {T, V<:AbstractArray{T}}
    excludeAbtArray0D(V)
    excludeAbtArray(T)
    (T, ndims(V))
end
checkAbtArrayInput(::Type{T}) where {T} = (T, 0)
checkAbtArrayInput(::T) where {T} = checkAbtArrayInput(T)

assertAbtArrayOutput(::Type{T}, ::Val{N}) where {T, N} = AbstractArray{T, N}
assertAbtArrayOutput(::Type{T}, ::Val{0}) where {T} = T

function checkReturnType(f::F, ::Type{T}, args::NonEmptyTuple{Any}) where {F, T}
    @inferred T f(args...)
    f(args...)
end


struct TypedReduction{T, F<:Function} <: TypedFunction{T, F}
    f::F

    function TypedReduction(f::F, arg::T, args...) where {F, T}
        allArgs = (arg, args...)
        checkAbtArrayInput.(allArgs)
        checkReturnType(f, T, allArgs)
        eleT = T <: AbstractArray ? eltype(T) : T
        new{eleT, F}(f)
    end

    function TypedReduction(::Type{T}) where {T}
        excludeAbtArray(T)
        new{T, iT}(itself)
    end
end

TypedReduction(srf::TypedReduction, arg::T, args...) where {T} = 
TypedReduction(srf.f, arg, args...)

TypedReduction(srf::TypedReduction, arg::AbstractArray{T}, args...) where {T} = 
TypedReduction(srf.f, arg, args...)

function (sf::TypedReduction{T, F})(arg::AbtArrayOr{T}, args::AbtArrayOr{T}...) where {T, F}
    allArgs = (arg, args...)
    excludeAbtArray0D.(allArgs)
    sf.f(allArgs...)::T
end


struct StableMorphism{T, F<:Function, N} <:TypedFunction{T, F}
    f::F

    function StableMorphism(f::F, arg::T, args...) where {T, F}
        allArgs = (arg, args...)
        checkAbtArrayInput.(allArgs)
        val = checkReturnType(f, AbstractArray{T}, allArgs)
        eleT = T <: AbstractArray ? eltype(T) : T
        new{eleT, F, ndims(val)}(f)
    end

    function StableMorphism(::Type{V}) where {T, N, V<:AbstractArray{T, N}}
        checkAbtArrayInput(V)
        new{T, iT, N}(itself)
    end
end

StableMorphism(srf::StableMorphism, arg::T, args...) where {T} = 
StableMorphism(srf.f, arg, args...)

StableMorphism(srf::StableMorphism, arg::AbstractArray{T}, args...) where {T} = 
StableMorphism(srf.f, arg, args...)

function (sf::StableMorphism{T, F, N})(arg::AbtArrayOr{T}, args::AbtArrayOr{T}...) where 
                                      {T, F, N}
    allArgs = (arg, args...)
    checkAbtArrayInput.(allArgs)
    sf.f(allArgs...)::AbstractArray{T, N}
end

genSeqAxis(::Val{N}, sym::Symbol=:e) where {N} = (Symbol(sym, i) for i in 1:N) |> Tuple

bundle(arg::T, args::T...) where {T} = collect( (arg, args...) )
const FofBundle = typeof(bundle)

mutable struct MorphicLinkage{T, F<:Function, N}
    const f::F
    const extent::Int
    @atomic axis::NonEmptyTuple{Tuple{Symbol, Int}}
    #! Incorporate ShapedMemory


    function MorphicLinkage(f::F, ::Type{V}, arg, args...; 
                            axis::Union{NonEmptyTuple{Tuple{Symbol,Int}}, Missing}=missing, 
                            extent::Union{Int, Missing}=missing) where {F<:Function, V}
        T, N = checkAbtArrayInput(V)
        @assert extent > 1
        allArgs = (arg, args...)
        excludeAbtArray0D.(allArgs)
        val = checkReturnType(f, AbstractArray{<:T}, allArgs)
        ismissing(extent) && (extent = length(val))
        ismissing(axis) && (axis = genSeqAxis( Val(1) ))
        new{T, F, N}(f, extent, Memory{T}(undef, extent), axis)
    end

    function MorphicLinkage(::V, ::V, ::Vararg{V, A}) where {V, A}
        T, N = checkAbtArrayInput(V)
        extent = A + 2
        new{T, FofBundle, N}(bundle, extent, Memory{T}(undef, extent), genSeqAxis( Val(1) ))
    end
end

function (ml::MorphicLinkage{T, F, N})(arg::T, args::T...) where {T, N, F}
    allArgs = (arg, args...)
    excludeAbtArray0D.(allArgs)
    res = ml.f(allArgs...)::AbstractArray{<:assertAbtArrayOutput(T, Val(N))}
    iBegin = firstindex(res)
    reshape(res[iBegin : (iBegin+ml.extent-1)], getindex.(ml.axis, 2))
end


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

function checkPrimParamElementalType(::Type{T}) where {T}
    if !(isbitstype(T) || issingletontype(T))
        throw(DomainError(T, "The (elemental) type of `input`, when used as an argument, "*
                             "should make at least one of these functions return `true`:\n"*
                             "`$isbitstype`, `$issingletontype`."))
    end
    nothing
end

function checkPrimParamElementalType(::Type{<:Union{Nothing, Missing}})
    throw(DomainError(T, "The (elemental) type of `input` cannot be `$Missing` or "*
                         "`$Nothing`"))
end

const SymOrIndexedSym = Union{Symbol, IndexedSym}


genTernaryNumber(num::Int) = TernaryNumber(num)
genTernaryNumber(num::TernaryNumber) = itself(num)

mutable struct CellVar{T} <: PrimitiveParam{T, 0}
    @atomic input::T
    const symbol::IndexedSym
    const screen::TernaryNumber

    function CellVar(input::T, symbol::SymOrIndexedSym, 
                     screen::Union{TernaryNumber, Int}=TPS1) where {T}
        checkPrimParamElementalType(T)
        checkScreenLevel(screen, getScreenLevelRange(PrimitiveParam))
        new{T}(input, IndexedSym(symbol), genTernaryNumber(screen))
    end
end


struct GridVar{T, N, V<:AbstractArray{T, N}} <: PrimitiveParam{T, N}
    input::V
    symbol::IndexedSym
    screen::TernaryNumber

    function GridVar(input::V, symbol::SymOrIndexedSym, 
                     screen::Union{TernaryNumber, Int}=TPS1) where 
                    {T, N, V<:AbstractArray{T, N}}
        checkPrimParamElementalType(T)
        checkScreenLevel(screen, getScreenLevelRange(PrimitiveParam))
        N < 1 && throw(DomainError(N, "The dimension of `input` must be larger than 0."))
        new{T, N, V}(copy(input), IndexedSym(symbol), genTernaryNumber(screen))
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

function checkParamContainerArgType1(I::Type, R::Type)
    if !(I <: R)
        throw(ArgumentError("`$I` is not supported as argument `input` when `lambda` "*
                            "functions like an identity morphism."))
    end
    nothing
end

function checkCellParamArg(::TypedReduction{T, <:iTalike}, input::I, 
                           shifter::S, memory::Union{T, Missing}) where {T, I, S}
    checkParamContainerArgType1(I, Tuple{ElementalParam{T}})
    checkParamBoxInput(input, dimMax=0)
    memory = deepcopy(ismissing(memory) ? (input[1]|>obtain|>shifter) : memory)
    TypedReduction(T), iT, memory
end

function checkCellParamArg(f::TypedReduction{T, F}, input::I, shifter::S, 
                           memory::Union{T, Missing}) where {T, F, I, S}
    checkParamBoxInput(input)
    f, F, deepcopy(ismissing(memory) ? shifter( f(obtain.(input)...) ) : memory)
end

initializeOffset(::Type) = nothing
initializeOffset(::Type{T}) where {T<:Number} = zero(T)

mutable struct CellParam{T, F<:Function, I<:ParamBoxInputType{T}} <: ParamBox{T, 0, I}
    const lambda::TypedReduction{T, F}
    const input::I
    const symbol::IndexedSym
    @atomic memory::T
    @atomic screen::TernaryNumber
    @atomic offset::T

    function CellParam(lambda::TypedReduction{T, F}, input::I, 
                       symbol::SymOrIndexedSym, 
                       memory::Union{T, Missing}=missing, 
                       screen::Union{TernaryNumber, Int}=TUS0, 
                       offset::Union{T, Nothing}=initializeOffset(T)) where 
                      {T, F, I<:ParamBoxInputType{T}}
        slRange = getScreenLevelRange(ParamBox{T, 0})
        screen = genTernaryNumber(screen)
        sl = checkScreenLevel(screen, slRange)
        shifter = genValShifter(T, offset)
        lambda, funcType, memory = checkCellParamArg(lambda, input, shifter, memory)
        symbol = IndexedSym(symbol)
        if slRange == (0, 0)
            new{T, funcType, I}(lambda, input, symbol, memory, screen)
        else
            offset===nothing && ( offset = (sl > 0 ? memory : typedSub(memory, memory)) )
            new{T, funcType, I}(lambda, input, symbol, memory, screen, offset)
        end
    end
end

function CellParam(func::Function, input::ParamBoxInputType{T}, symbol::SymOrIndexedSym; 
                   init::Union{T, Missing}=missing) where {T}
    lambda = TypedReduction(func, obtain.(input)...)
    CellParam(lambda, input, symbol, init, TUS0, initializeOffset(T))
end

CellParam(func::Function, input::ParamBoxSingleArg{T}, symbol::SymOrIndexedSym; 
          init::Union{T, Missing}=missing) where {T} = 
CellParam(func, (input,), symbol; init)

CellParam(func::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
          symbol::SymOrIndexedSym; 
          init::Union{T, Missing}=missing) where {T} = 
CellParam(func, (input1, input2), symbol; init)

CellParam(func::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
          input3::ParamBoxSingleArg{T}, symbol::SymOrIndexedSym; 
          init::Union{T, Missing}=missing) where {T} = 
CellParam(func, (input1, input2, input3), symbol; init)

function CellParam(par::CellParam{T}, symbol::SymOrIndexedSym=symOf(par); 
                   init::Union{T, Missing}=par.memory) where {T}
    offset = isOffsetEnabled(par) ? par.offset : nothing
    CellParam(par.lambda, par.input, symbol, init, par.screen, offset)
end

CellParam(input::PrimitiveParam{T, 0}, 
          symbol::SymOrIndexedSym=symOf(input)) where {T} = 
CellParam(TypedReduction(T), (input,), symbol)

CellParam(var, varSym::SymOrIndexedSym, symbol::SymOrIndexedSym=varSym) = 
CellParam(CellVar(var, varSym), symbol)

# const TensorInCellParam{T, F, PB, N} = CellParam{T, F, <:AbstractArray{PB, N}}
# const ScalarInCellParam{T, F, PB} = TensorInCellParam{T, F, PB, 0}

function checkGridParamArg(::StableMorphism{T, <:iTalike, N}, input::I, memory::M) where 
                           {T, N, I, M<:AbstractArray{T, N}}
    checkParamContainerArgType1(I, Tuple{ParamBoxSingleArg{T, N}})
    MI = typeof(input[1])
    if !(M <: MI)
        throw(AssertionError("The type of memory should be the subtype of type of "*
                             "`input[1]` (`::$M1`)"))
    end
    checkParamBoxInput(input, dimMin=1)
    StableMorphism(AbstractArray{T, N}), iT, deepcopy(memory)
end

function checkGridParamArg(::StableMorphism{T, <:iTalike, N}, input::I, ::Missing) where 
                           {T, N, I}
    checkParamContainerArgType1(I, Tuple{ParamBoxSingleArg{T, N}})
    checkParamBoxInput(input, dimMin=1)
    StableMorphism(AbstractArray{T, N}), iT, deepcopy(input[1]|>obtain)
end

function throwGridParamDimErrorMessage()
    throw(ArgumentError("Returned array should have dimension `N` larger than 0. Use "*
                        "`$CellParam` for returning scalar-type output."))
end

function checkGridParamArg(f::StableMorphism{T, F, N}, input::I, memory::M) where 
                           {T, F, N, I, M<:AbstractArray{T, N}}
    N < 1 && throwGridParamDimErrorMessage()
    checkParamBoxInput(input)
    checkReturnType(f, M, obtain.(input))
    f, F, deepcopy(memory)
end

function checkGridParamArg(f::StableMorphism{T, F, N}, input::I, ::Missing) where 
                           {T, F, N, I}
    N < 1 && throwGridParamDimErrorMessage()
    checkParamBoxInput(input)
    f, F, deepcopy( f(obtain.(input)...) )
end

struct GridParam{T, F<:Function, I<:ParamBoxInputType{T}, 
                 N, M<:AbstractArray{T, N}} <: ParamBox{T, N, I}
    lambda::StableMorphism{T, F, N}
    input::I
    symbol::IndexedSym
    memory::M

    function GridParam(lambda::StableMorphism{T, F, N}, input::I, 
                       symbol::SymOrIndexedSym, 
                       memory::Union{AbstractArray{T, N}, Missing}=missing) where 
                      {T, F, N, I<:ParamBoxInputType{T}}
        lambda, funcType, memory = checkGridParamArg(lambda, input, memory)
        new{T, funcType, I, N, typeof(memory)}(lambda, input, IndexedSym(symbol), memory)
    end
end

function GridParam(func::Function, input::ParamBoxInputType{T}, symbol::SymOrIndexedSym; 
                    init::Union{AbstractArray{T}, Missing}=missing) where {T}
    lambda = StableMorphism(func, obtain.(input)...)
    GridParam(lambda, input, symbol, init)
end

GridParam(func::Function, input::ParamBoxSingleArg{T}, symbol::SymOrIndexedSym; 
           init::Union{AbstractArray{T}, Missing}=missing) where {T} = 
GridParam(func, (input,), symbol; init)

GridParam(func::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
           symbol::SymOrIndexedSym; 
           init::Union{AbstractArray{T}, Missing}=missing) where {T} = 
GridParam(func, (input1, input2), symbol; init)

GridParam(func::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
           input3::ParamBoxSingleArg{T}, symbol::SymOrIndexedSym; 
           init::Union{AbstractArray{T}, Missing}=missing) where {T} = 
GridParam(func, (input1, input2, input3), symbol; init)

GridParam(par::GridParam{T}, symbol::SymOrIndexedSym=symOf(par); 
           init::Union{AbstractArray{T}, Missing}=par.memory) where {T} = 
GridParam(par.lambda, par.input, symbol, init)

GridParam(input::PrimitiveParam{T, N}, 
           symbol::SymOrIndexedSym=symOf(input)) where {T, N} = 
GridParam(StableMorphism(AbstractArray{T, N}), (input,), symbol)

GridParam(input::AbstractArray{<:ElementalParam{T}, N}, 
           symbol::SymOrIndexedSym) where {T, N} = 
GridParam(StableMorphism(AbstractArray{T, N}), (input,), symbol)

GridParam(val::AbstractArray, valSym::SymOrIndexedSym, symbol::SymOrIndexedSym=valSym) = 
GridParam(GridVar(val, valSym), symbol)


#!! struct ChainParam

# end

# ChainParam -> CellVar of CellVar

# struct ReferenceParam

# end

getScreenLevelRange(::Type{<:ParamBox}) = (0, 0)

getScreenLevelRange(::Type{<:ParamBox{T, 0}}) where {T} = (0, checkTypedOpMethods(T) * 2)

getScreenLevelRange(::Type{<:PrimitiveParam}) = (1, 2)

getScreenLevelRange(::Type{<:DimensionalParam}) = (0, 2)

getScreenLevelRange(::T) where {T<:DimensionalParam} = getScreenLevelRange(T)

function isScreenLevelChangeable(::T) where {T<:DimensionalParam}
    minLevel, maxLevel = getScreenLevelRange(T)
    (maxLevel - minLevel) > 0
end

isOffsetEnabled(::DimensionalParam) = false

function isOffsetEnabled(pb::T) where {T<:CellParam}
    isScreenLevelChangeable(pb) && getScreenLevelRange(T)[end] > 0 && 
    isdefined(pb, :offset) # Only for safety
end

screenLevelOf(::ParamBox) = 0

screenLevelOf(p::ParamBox{<:Any, 0}) = Int(p.screen)

screenLevelOf(p::PrimitiveParam) = Int(p.screen)


function setScreenLevelCore!(p::DimensionalParam, level::Int)
    @atomic p.screen = TernaryNumber(level)
end

function setScreenLevel!(p::T, level::Int) where {T<:CellParam}
    checkScreenLevel(level, getScreenLevelRange(T))
    levelOld = screenLevelOf(p)
    if levelOld == level
    elseif levelOld == 0
        @atomic p.offset = obtain(p)
    elseif level == 0
        newVal = p.lambda((obtain(arg) for arg in p.input)...)
        @atomic p.offset = typedSub(p.offset, newVal)
    end
    setScreenLevelCore!(p, level)
    p
end

setScreenLevel(pn::CellParam, level::Int) = setScreenLevel!(CellParam(pn), level)

setScreenLevel(p::CellVar, level::Int) = CellVar(p.input, p.symbol, TernaryNumber(level))

setScreenLevel(p::GridVar, level::Int) = GridVar(p.input, p.symbol, TernaryNumber(level))


function memorize!(pn::ParamBox{T}, newMem::T=ValOf(pn)) where {T}
    oldMem = pn.memory
    @atomic pn.memory = newMem
    oldMem
end

struct ParamList{T, N, I<:DimensionalParam{T, N}} <: ParamHeap{T, N, I}
    input::Memory{I}
    symbol::IndexedSym

    function ParamList(input::AbstractArray{P}, symbol::SymOrIndexedSym) where 
                      {T, N, P<:DimensionalParam{T, N}}
        new{T, N, P}(Memory{P}(vec(input)), IndexedSym(symbol))
    end
end

ParamList(pl::ParamList, symbol::IndexedSym) = ParamList(pl.input, symbol)


# struct TwiceThriceAbtArray{T, N, V<:AbstractArray{T, N}, 
#                            R<:TwiceThriceNTuple{V}} <: TupleOfAbtArrays{T, N, V, R}
#     arr::R
# end

# eleTypeOf(::TupleOfAbtArrays{T}) where {T} = T
# arrTypeOf(::TupleOfAbtArrays{<:Any, <:Any, V}) where {V} = V

function checkParamContainerArgType2(::Type{I}, ::Type{T}, ::Val{N}) where  {I, T, N}
    R = NonEmptyTuple{ParamBoxSingleArg{T, N}}
    I <: R || throw(ArgumentError("`I` should be a subtype of `$R`."))
    nothing
end

function checkParamContainerArgType3(name::AbstractString, len::Int, extent::Int)
    if len != extent
        throw(DomainError(len, "The length of `$name` should match `ml::MorphicLinkage`'s "*
                               "specification: $extent."))
    end
    nothing
end

# getParOutputType(::DimensionalParam{T, 0}) where {T} = T
# getParOutputType(::DimensionalParam{T, N}) where {T, N} = AbstractArray{T, N}
# getParOutputType(::AbstractArray{<:ElementalParam{T}, N}) where {T, N} = AbstractArray{T, N}

function checkParamNestArg(ml::MorphicLinkage{T, FofBundle, N}, input::I, 
                           memory::Union{Memory{ShapedMemory{T, N}}, Missing}) where 
                          {T, N, I}
    checkParamContainerArgType2(I, T, Val(N))
    if ismissing(memory)
        memory = Memory{ShapedMemory{T, N}}(input.|>obtain|>collect.|>ShapedMemory)
    else
        checkParamContainerArgType3("memory", length(memory), ml.extent)
    end
    checkParamBoxInput(input)
    ml, FofBundle, deepcopy(memory)
end

function checkParamNestArg(ml::MorphicLinkage{T, F, N}, input::I, 
                           memory::Union{Memory{ShapedMemory{T, N}}, Missing}) where 
                          {T, F, N, I}
    if ismissing(memory)
        memory = Memory{ShapedMemory{T, N}}(ShapedMemory.(ml.f(obatin.(input)...)) |> vec)
    else
        checkParamContainerArgType3("memory", length(memory), ml.extent)
    end
    checkParamBoxInput(input)
    ml, F, deepcopy(memory)
end

struct LinkParam{T, F<:Function, I<:ParamBoxInputType{T}, N} <: ParamBox{T, N, I}
    source::Tuple{MorphicLinkage{T, F, N}, I}
    index::Int
    memory::ShapedMemory{T, N}
end

struct ParamNest{T, F<:Function, I<:ParamBoxInputType{T}, N} <: ParamStack{T, N, I}
    linker::MorphicLinkage{T, F, N}
    input::I
    symbol::IndexedSym
    output::Memory{LinkParam{T, F, I, N}}

    function ParamNest(linker::MorphicLinkage{T, F, N}, input::I, 
                       symbol::SymOrIndexedSym, 
                       memory::Union{Memory{ShapedMemory{T, N}}, Missing}=missing) where 
                      {T, N, F, I<:ParamBoxInputType{T}}
        linker, funcType, memory = checkParamNestArg(linker, input, memory)
        lPars = map( enumerate(memory) ) do (idx, val)
            LinkParam((linker, input), idx, ShapedMemory(val))
        end
        output = Memory{LinkParam{T, F, I, N, V}}(lPars)
        new{T, funcType, I, N}(linker, input, IndexedSym(symbol), output)
    end
end

ParamNest(linker::MorphicLinkage{T, F, N}, input::I, symbol::SymOrIndexedSym, 
          output::Memory{LinkParam{T, F, I, N}}) where {T, N, F, I<:ParamBoxInputType{T}} = 
ParamNest(linker, input, symbol, Memory{ShapedMemory{T, N}}(getproperty.(output, memory)))


function ParamNest(func::Function, input::ParamBoxInputType{T}, 
                   symbol::SymOrIndexedSym) where {T}
    inputVal = obtain.(input)
    out = func(inputVal...)
    out isa AbstractArray || throw(AssertionError("`func` should output an AbstractArray."))
    linker = MorphicLinkage(func, typeof(out), inputVal...)
    ParamNest(linker, input, symbol)
end

ParamNest(input::TwiceThriceNTuple{ParamBoxSingleArg{T, N}}, 
          symbol::SymOrIndexedSym) where {T, N} = 
ParamNest(MorphicLinkage(obtain.(input)...), input, symbol)


struct NodeTuple{T<:Real, N, PT} <: ParamBox{T, N, PT}
    input::PT
    symbol::IndexedSym

    NodeTuple(input::PT, symbol::SymOrIndexedSym) where 
             {T, PT<:NonEmptyTuple{CellParam{T}}} = 
    new{T, length(input), PT}(input, IndexedSym(symbol))
end

NodeTuple(nt::NodeTuple, symbol::Symbol) = NodeTuple(nt.input, symbol)


indexedSymOf(pc::DimensionalParam) = pc.symbol

symOf(pc::DimensionalParam) = indexedSymOf(pc).name

inputOf(pb::DimensionalParam) = pb.input


mutable struct NodeMarker{T} #!Type-unstable
    visited::Bool
    value::T

    NodeMarker(init::T, ::Type{U}=T) where {T, U} = new{U}(false, init)
end

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
    markerDict = IdDict{ParamBox{T}, NodeMarker{<:AbstractArray{T}}}()
    searchObtain(markerDict, p)
end

function searchObtainLoop(markerDict::IdDict{ParamBox{T}, NodeMarker{<:AbstractArray{T}}}, 
                          input::AbstractArray{<:DimensionalParam{T}}) where {T}
    map(input) do child
        searchObtain(markerDict, child)
    end
end

function searchObtainLoop(markerDict::IdDict{ParamBox{T}, NodeMarker{<:AbstractArray{T}}}, 
                          input::DimensionalParam{T}) where {T}
    searchObtain(markerDict, input)
end

function searchObtainCore(shiftVal::F, 
                          markerDict::IdDict{ParamBox{T}, NodeMarker{<:AbstractArray{T}}}, 
                          p::ParamBox{T, N}) where {T, F<:Union{iT, ValShifter{T}}, N}
    # Depth-first search by recursive calling
    valBox = packElementalVal(Val(N), p.memory)
    marker = get!(markerDict, p, NodeMarker(valBox))::NodeMarker{typeof(valBox)}
    if !marker.visited
        marker.visited = true
        res = p.lambda( (searchObtainLoop(markerDict, x) for x in p.input)... ) |> shiftVal
        marker.value = packElementalVal(T, res)
    else
        marker.value
    end
end

function searchObtain(markerDict::IdDict{ParamBox{T}, NodeMarker{<:AbstractArray{T}}}, 
                      p::ParamBox{T, N}) where {T, N}
    sl = checkScreenLevel(screenLevelOf(p), getScreenLevelRange(ParamBox{T, N}))
    if sl == 0
        shiftVal = genValShifter(T, (isOffsetEnabled(p) ? p.offset : nothing))
        obtainElementalVal(T, searchObtainCore(shiftVal, markerDict, p))
    else
        p.offset
    end
end

searchObtain(::IdDict{ParamBox{T}, NodeMarker{<:AbstractArray{T}}}, 
             p::PrimitiveParam{T}) where {T} = 
directObtain(p)

#! Certain type of CellParam (0,0) should not need to include offset.

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

function setVal!(par::CellParam{T}, val::T) where {T}
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

function ParamMarker(pn::T) where {T<:CellParam}
    offset = isOffsetEnabled(pn) ? pn.offset : nothing
    ParamMarker(
        objectid(T), markObj.((pn.input..., offset)), objectid(pn.lambda.f), 
        objectid(screenLevelOf(pn))
    )
end

function ParamMarker(pn::T) where {T<:GridParam}
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


# operateBy(op::F, pn1::CellParam, num::Real) where {F<:Function} = 
# CellParam(OFC(itself, op, num), pn1, pn1.symbol)

# operateBy(op::F, num::Real, pn1::CellParam) where {F<:Function} = 
# CellParam(OCF(itself, op, num), pn1, pn1.symbol)

# operateBy(op::CommutativeBinaryNumOps, num::Real, pn1::CellParam) = 
# operateBy(op, pn1::CellParam, num::Real)

# reformulate(::typeof(+), pn1::CellParam) = itself(pn1)
# reformulate(::typeof(-), pn1::CellParam{T}) where {T} = reformulate(*, T(-1), pn1)

# repeatedlyApply(::typeof(+), pn::CellParam{T}, times::Int) where {T} = 
# reformulate(*, pn, T(times))

# repeatedlyApply(::typeof(-), pn::CellParam{T}, times::Int) where {T} = 
# reformulate(-, operateBy(*, pn, T(times)))

# repeatedlyApply(::typeof(*), pn::CellParam{T}, times::Int) where {T} = 
# reformulate(^, pn, T(times))

# function reformulate(op::F, pn1::CellParam{T}, pn2::CellParam{T}) where 
#                     {F<:Union{typeof(+), typeof(*)}, T<:Real}
#     if symFromIndexSym(pn1.symbol) > symFromIndexSym(pn2.symbol)
#         pn2, pn1 = pn1, pn2
#     end
#     if compareParamContainer(pn1, pn2)
#         repeatedlyApply(op, pn1, 2)
#     else
#         operateByCore(op, pn1, pn2)
#     end
# end


# addCellParam(pn1::CellParam{T}, pn2::CellParam{T}) where {T<:Real} = operateBy(+, pn1, pn2)

# mulCellParam(pn1::CellParam{T}, pn2::CellParam{T}) where {T<:Real} = operateBy(*, pn1, pn2)
# mulCellParam(pn::CellParam{T}, coeff::T) where {T} = operateBy(*, pn, coeff)
# mulCellParam(coeff::T, pn::CellParam{T}) where {T} = mulCellParam(pn, coeff)


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

topoSort(node::CellParam{T}) where {T} = topoSortINTERNAL([node])


# Sever the connection of a node to other nodes
sever(pv::CellVar) = CellVar(obtain(pv), pv.symbol)

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