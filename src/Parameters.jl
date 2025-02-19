export TensorVar, CellParam, GridParam, ParamGrid, setScreenLevel!, 
       setScreenLevel, symOf, inputOf, obtain, setVal!, screenLevelOf, 
       markParams!, topoSort, getParams, uniqueParams, ShapedMemory, directObtain, 
       memorize!, evalParamSource

using Base: Fix2, Threads.Atomic, issingletontype
using Test: @inferred

function checkReshapingAxis(shape::Tuple{Vararg{Int}})
    if any(i < 0 for i in shape)
        throw(AssertionError("All axis sizes should be non-negative."))
    end
    nothing
end


function checkReshapingAxis(arr::AbstractArray, shape::Tuple{Vararg{Int}})
    checkReshapingAxis(shape)
    len  = length(arr)
    if prod(shape) != len
        throw(AssertionError("The product of reshaping axes should be equal to the "*
                             "target array's length."))
    end
    len
end

struct TruncateReshape{N}
    axis::NTuple{N, Int}
    mark::NTuple{N, Symbol}
    truncate::TernaryNumber # 0: off, 1: keep leading entires, 2: keep trailing entires

    function TruncateReshape(axis::NonEmptyTuple{Int, N}, 
                             mark::NonEmptyTuple{Symbol, N}=ntuple( _->:e, Val(N+1) ); 
                             truncate::Union{Bool, TernaryNumber}=false) where {N}
        checkReshapingAxis(axis)
        new{N+1}(axis, mark, TernaryNumber(truncate|>Int))
    end

    function TruncateReshape(refArr::AbstractArray{T, N}, 
                             mark::NTuple{N, Symbol}=ntuple( _->:e, Val(N) ); 
                             truncate::Union{Bool, TernaryNumber}=false) where {T, N}
        N==0 && throw(AssertionError("The dimension of `refArr` should be at least one."))
        new{N}(size(refArr), mark, TernaryNumber(truncate|>Int))
    end

    TruncateReshape(f::TruncateReshape{N}; 
                    truncate::Union{Bool, TernaryNumber}=f.truncate) where {N} = 
    new{N}(f.axis, f.mark, TernaryNumber(truncate|>Int))
end

function (f::TruncateReshape{N})(arr::AbstractArray) where {N}
    extent = prod(f.axis)
    truncate = Int(f.truncate)
    v = if truncate == 0
        arr
    elseif truncate == 1
        arr[begin:begin+extent-1]
    else
        arr[end-extent+1:end]
    end
    reshape(v, f.axis)
end


struct ShapedMemory{T, N} <: AbstractMemory{T, N}
    value::Memory{T}
    shape::NTuple{N, Int}

    function ShapedMemory(value::Memory{T}, shape::Tuple{Vararg{Int}}) where {T}
        checkReshapingAxis(value, shape)
        new{T, length(shape)}(copy(value), shape)
    end

    function ShapedMemory{T}(::UndefInitializer, shape::NTuple{N, Int}) where {T, N}
        checkReshapingAxis(shape)
        new{T, N}(Memory{T}(undef, prod(shape)), shape)
    end

    function ShapedMemory(arr::ShapedMemory{T, N}) where {T, N}
        new{T, N}(arr.value, arr.shape)
    end
end

ShapedMemory(value::AbstractArray{T}, shape::Tuple{Vararg{Int}}=size(value)) where {T} = 
ShapedMemory(getMemory(value), shape)

ShapedMemory(::Type{T}, value::AbstractArray{T}) where {T} = ShapedMemory(value)

ShapedMemory(::Type{T}, value::T) where {T} = ShapedMemory( fill(value) )

getMemory(arr::ShapedMemory) = arr.value


size(arr::ShapedMemory) = arr.shape

firstindex(arr::ShapedMemory) = firstindex(arr.value)

lastindex(arr::ShapedMemory) = lastindex(arr.value)

getindex(arr::ShapedMemory, i::Int) = getindex(arr.value, i)
getindex(arr::ShapedMemory{<:Any, N}, i::Vararg{Int, N}) where {N} = 
getindex(reshape(arr.value, arr.shape), i...)

setindex!(arr::ShapedMemory, val, i::Int) = setindex!(arr.value, val, i)
setindex!(arr::ShapedMemory{<:Any, N}, val, i::Vararg{Int, N}) where {N} = 
setindex!(reshape(arr.value, arr.shape), val, i...)

iterate(arr::ShapedMemory) = iterate(arr.value)
iterate(arr::ShapedMemory, state) = iterate(arr.value, state)

length(arr::ShapedMemory) = length(arr.value)

axes(arr::ShapedMemory)	= map(Base.OneTo, size(arr))

function similar(arr::ShapedMemory, ::Type{T}=eltype(arr), 
                 shape::Tuple{Vararg{Int}}=size(arr)) where {T}
    ShapedMemory(similar(arr.value, T, prod(shape)), shape)
end

similar(arr::ShapedMemory{T}, shape::Tuple{Vararg{Int}}) where {T} = 
similar(arr, T, shape)


function binaryApply(op::F, arr1::ShapedMemory{T1}, arr2::ShapedMemory{T2}) where 
                 {F<:Function, T1, T2}
    if arr1.shape != arr2.shape
        throw(DimensionMismatch("`arr1` has size $(arr1.shape); "*
                                "`arr2` has size $(arr2.shape)."))
    end
    val = Memory{promote_type(T1, T2)}(op(arr1.value, arr2.value))
    ShapedMemory(val, arr1.shape)
end

+(arr1::ShapedMemory, arr2::ShapedMemory) = binaryApply(+, arr1, arr2)
-(arr1::ShapedMemory, arr2::ShapedMemory) = binaryApply(-, arr1, arr2)

viewElements(obj::ShapedMemory) = reshape(obj.value, obj.shape)
viewElements(obj::AbstractArray) = itself(obj)

directObtain(obj::AbstractArray) = itself.( viewElements(obj) )
directObtain(obj::Any) = itself(obj)


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


exclude0DimData(::Type) = nothing
exclude0DimData(::Type{<:AbtArray0D}) = 
throw(ArgumentError("`$(AbstractArray{<:Any, 0})` is not allowed as the argument."))

excludeAbtArray(::Type) = nothing
excludeAbtArray(::Type{<:AbstractArray}) = 
throw(ArgumentError("`AbstractArray` is not allowed as an input argument."))


function checkReturnType(f::F, ::Type{T}, args::NonEmptyTuple{Any}) where {F, T}
    @inferred T f(args...)
    f(args...)
end


struct TypedReduction{T, F<:Function} <: DualSpanFunction{T, 0, 0}
    f::F

    function TypedReduction(f::ReturnTyped{T, F}, arg, args...) where {T, F<:Function}
        allArgs = (arg, args...)
        excludeAbtArray(T)
        checkReturnType(f.f, T, allArgs)
        new{T, F}(f.f)
    end

    function TypedReduction(::Type{T}) where {T}
        excludeAbtArray(T)
        new{T, ItsType}(itself)
    end
end

TypedReduction(trf::TypedReduction{T}, arg, args...) where {T} = 
TypedReduction(ReturnTyped(trf.f, T), arg, args...)

function (sf::TypedReduction{T, F})(arg::AbtArr210L{T}, args::AbtArr210L{T}...) where {T, F}
    sf.f(arg, args...)::T
end

(::TypedReduction{T, ItsType})(arg::T) where {T} = itself(arg)


struct StableMorphism{T, F<:Function, N} <: DualSpanFunction{T, N, 0}
    f::F
    axis::TruncateReshape{N}

    function StableMorphism(f::ReturnTyped{<:AbstractArray{T}, F}, arg, args...; 
                            axis::MissingOr{TruncateReshape}=missing, 
                            truncate::Union{Bool, TernaryNumber}=
                            (ismissing(axis) ? false : axis.truncate)) where 
                           {T, F<:Function}
        allArgs = (arg, args...)
        val = checkReturnType(f.f, AbstractArray{T}, allArgs)
        N = ndims(val)
        N==0 && throw(AssertionError("The dimension of `f`'s returned value must be "*
                                     "larger than zero."))
        axis = TruncateReshape(ifelse(ismissing(axis), val, axis); truncate)
        new{T, F, N}(f.f, axis)
    end

    function StableMorphism(arg::V) where {N, T, V<:AbstractArray{T, N}}
        N==0 && throw(AssertionError("`N` must be larger than zero."))
        new{T, ItsType, N}(itself, TruncateReshape(arg))
    end
end

StableMorphism(srf::StableMorphism{T, <:Function, N}, arg, args...; 
               axis::MissingOr{TruncateReshape{N}}=srf.axis) where {T, N} = 
               StableMorphism(ReturnTyped(srf.f, AbstractArray{T, N}), arg, args...; axis)

function (sf::StableMorphism{T, F, N})(arg::AbtArr210L{T}, args::AbtArr210L{T}...) where 
                                      {T, F, N}
    res = sf.f(arg, args...)::AbstractArray{T}
    sf.axis(res)
end

(::StableMorphism{T, ItsType, N})(arg::AbstractArray{T, N}) where {T, N} = itself(arg)


function checkScreenLevel(sl::Int, levels::NonEmptyTuple{Int})
    if !(sl in levels)
        throw(DomainError(sl, "This screen level ($(TernaryNumber(sl))) is not allowed."))
    end
    sl
end

checkScreenLevel(s::TernaryNumber, levels::NonEmptyTuple{Int}) = 
checkScreenLevel(Int(s), levels)

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

struct TensorVar{T, N} <: PrimitiveParam{T, N}
    input::ShapedMemory{T, N}
    symbol::IndexedSym
    screen::TernaryNumber

    function TensorVar(input::AbstractArray{T, N}, symbol::SymOrIndexedSym, 
                       screen::Union{TernaryNumber, Int}=TPS1) where {T, N}
        checkPrimParamElementalType(T)
        checkScreenLevel(screen, getScreenLevelOptions(PrimitiveParam))
        input = ShapedMemory(input|>deepcopy)
        new{T, N}(input, IndexedSym(symbol), genTernaryNumber(screen))
    end
end

TensorVar(input::T, symbol::SymOrIndexedSym, 
          screen::Union{TernaryNumber, Int}=TPS1) where {T} = 
TensorVar(fill(input), symbol, screen)


function checkParamInput(input::ParamInputType; 
                         innerDimMin::Int=0, innerDimMax::Int=64, 
                         outerDimMin::Int=0, outerDimMax::Int=64)
    hasVariable = false
    innerDimMinMax = (innerDimMin, innerDimMax)
    outerDimMinMax = (outerDimMin, outerDimMax)
    for x in input
        hasVariable = checkParamInputCore(hasVariable, x, (innerDimMinMax, outerDimMinMax))
    end
    if !hasVariable
        throw(ArgumentError("`input` must contain as least one non-constant parameter."))
    end
    nothing
end

function checkParamInputCore(hasVariable::Bool, par::P, 
                             doubleDimMinMax::NTuple{2, NTuple{2, Int}}) where 
                            {T, N, O, P<:ParamBox{T, N, O}}
    for (dim, str, minMax) in zip((N, O), ("inner", "outer"), doubleDimMinMax)
        if !(minMax[begin] <= dim <= minMax[end])
            throw(DomainError(dim, "The input `par`'s $str dimension falls outside the "*
                                   "permitted range: $minMax."))
        end
    end
    hasVariable || (screenLevelOf(par) < 2) && (hasVariable = true)
    hasVariable
end

function checkParamContainerArgType1(I::Type, R::Type)
    if !(I <: R)
        throw(ArgumentError("`$I` is not supported as argument `input` when `lambda` "*
                            "functions like an identity morphism."))
    end
    nothing
end

function checkCellParamArg(::TypedReduction{T, <:ItsTalike}, input::I, shifter::S, 
                           memory::Union{ShapedMemory{T, 0}, T, Missing}) where {T, I, S}
    checkParamContainerArgType1(I, Tuple{ElementalParam{T}})
    checkParamInput(input, innerDimMax=0, outerDimMax=0)
    if ismissing(memory)
        memory = ShapedMemory( fill(input[1]|>obtain|>shifter) )
    elseif memory isa T
        memory = ShapedMemory( fill(memory) )
    end
    TypedReduction(T), ItsType, deepcopy(memory)
end

function checkCellParamArg(f::TypedReduction{T, F}, input::I, shifter::S, 
                           memory::Union{ShapedMemory{T, 0}, T, Missing}) where {T, F, I, S}
    checkParamInput(input)
    if ismissing(memory)
        memory = ShapedMemory( fill(f(obtain.(input)...)|>shifter) )
    elseif memory isa T
        memory = ShapedMemory( fill(memory) )
    end
    f, F, deepcopy(memory)
end

initializeOffset(::Type) = nothing
initializeOffset(::Type{T}) where {T<:Number} = zero(T)

mutable struct CellParam{T, F<:Function, I<:ParamInputType{T}} <: BaseParam{T, 0, I}
    const lambda::TypedReduction{T, F}
    const input::I
    const symbol::IndexedSym
    @atomic memory::ShapedMemory{T, 0}
    @atomic screen::TernaryNumber
    @atomic offset::T

    function CellParam(lambda::TypedReduction{T, F}, input::I, 
                       symbol::SymOrIndexedSym, 
                       memory::Union{ShapedMemory{T, 0}, T, Missing}=missing, 
                       screen::Union{TernaryNumber, Int}=TUS0, 
                       offset::Union{T, Nothing}=initializeOffset(T)) where 
                      {T, F, I<:ParamInputType{T}}
        levels = getScreenLevelOptions(BaseParam{T, 0})
        screen = genTernaryNumber(screen)
        sl = checkScreenLevel(screen, levels)
        shifter = genValShifter(T, offset)
        lambda, funcType, memory = checkCellParamArg(lambda, input, shifter, memory)
        symbol = IndexedSym(symbol)
        if levels == (0,)
            new{T, funcType, I}(lambda, input, symbol, memory, screen)
        else
            if offset===nothing
                offset = if sl > 0
                    directObtain(memory)
                else
                    memVal = directObtain(memory)
                    typedSub(memVal, memVal)
                end
            end
            new{T, funcType, I}(lambda, input, symbol, memory, screen, offset)
        end
    end
end

function CellParam(func::Function, input::ParamInputType{T}, symbol::SymOrIndexedSym; 
                   init::Union{ShapedMemory{T, 0}, T, Missing}=missing) where {T}
    lambda = TypedReduction(ReturnTyped(func, T), obtain.(input)...)
    CellParam(lambda, input, symbol, init, TUS0, initializeOffset(T))
end

CellParam(func::Function, input::ParamBox{T}, symbol::SymOrIndexedSym; 
          init::Union{ShapedMemory{T, 0}, T, Missing}=missing) where {T} = 
CellParam(func, (input,), symbol; init)

CellParam(func::Function, input1::ParamBox{T}, input2::ParamBox{T}, 
          symbol::SymOrIndexedSym; 
          init::Union{ShapedMemory{T, 0}, T, Missing}=missing) where {T} = 
CellParam(func, (input1, input2), symbol; init)

CellParam(func::Function, input1::ParamBox{T}, input2::ParamBox{T}, 
          input3::ParamBox{T}, symbol::SymOrIndexedSym; 
          init::Union{ShapedMemory{T, 0}, T, Missing}=missing) where {T} = 
CellParam(func, (input1, input2, input3), symbol; init)

function CellParam(par::CellParam{T}, symbol::SymOrIndexedSym=symOf(par); 
                   init::Union{ShapedMemory{T, 0}, T, Missing}=par.memory) where {T}
    offset = isOffsetEnabled(par) ? par.offset : nothing
    CellParam(par.lambda, par.input, symbol, init, par.screen, offset)
end

CellParam(input::ElementalParam{T}, symbol::SymOrIndexedSym=symOf(input)) where {T} = 
CellParam(TypedReduction(T), (input,), symbol)

CellParam(var, varSym::SymOrIndexedSym, symbol::SymOrIndexedSym=varSym) = 
CellParam(TensorVar(var, varSym), symbol)

const ItselfCParam{T} = CellParam{T, ItsType, Tuple{TensorVar{T, 0}}}


function checkGridParamArg(::StableMorphism{T, <:ItsTalike, N}, input::I, memory::M) where 
                          {T, N, O, I<:ParamBox{T, <:Any, O}, M<:AbstractArray{T, N}}
    checkParamContainerArgType1(I, Tuple{FlattenedParam{T, N}})
    MI = typeof(input[1])
    if !(M <: MI)
        throw(AssertionError("The type of `memory` should be a subtype of `$M1`."))
    end
    checkParamInput(input, innerDimMax=(O==0)*N, outerDimMax=(O==N)*N)
    StableMorphism(memory), ItsType, deepcopy(memory|>ShapedMemory)
end

function checkGridParamArg(::StableMorphism{T, <:ItsTalike, N}, input::I, ::Missing) where 
                          {T, N, O, I<:ParamBox{T, <:Any, O}}
    checkParamContainerArgType1(I, Tuple{FlattenedParam{T, N}})
    checkParamInput(input, innerDimMax=(O==0)*N, outerDimMax=(O==N)*N)
    memory = input[1]|>obtain
    StableMorphism(memory), ItsType, deepcopy(memory|>ShapedMemory)
end

function throwGridParamDimErrorMessage()
    throw(ArgumentError("Returned array should have dimension `N` larger than 0. Use "*
                        "`$CellParam` for returning scalar-type output."))
end

function checkGridParamArg(f::StableMorphism{T, F, N}, input::I, memory::M) where 
                           {T, F, N, I, M<:AbstractArray{T, N}}
    N < 1 && throwGridParamDimErrorMessage()
    checkParamInput(input)
    checkReturnType(f, M, obtain.(input))
    f, F, deepcopy(memory|>ShapedMemory)
end

function checkGridParamArg(f::StableMorphism{T, F, N}, input::I, ::Missing) where 
                           {T, F, N, I}
    N < 1 && throwGridParamDimErrorMessage()
    checkParamInput(input)
    f, F, deepcopy( f(obtain.(input)...)|>ShapedMemory )
end

mutable struct GridParam{T, F<:Function, I<:ParamInputType{T}, N} <: BaseParam{T, N, I}
    const lambda::StableMorphism{T, F, N}
    const input::I
    const symbol::IndexedSym
    @atomic memory::ShapedMemory{T, N}

    function GridParam(lambda::StableMorphism{T, F, N}, input::I, 
                       symbol::SymOrIndexedSym, 
                       memory::Union{AbstractArray{T, N}, Missing}=missing) where 
                      {T, F, N, I<:ParamInputType{T}}
        lambda, funcType, memory = checkGridParamArg(lambda, input, memory)
        new{T, funcType, I, N}(lambda, input, IndexedSym(symbol), memory)
    end
end

function GridParam(func::Function, input::ParamInputType{T}, symbol::SymOrIndexedSym; 
                   init::Union{AbstractArray{T}, Missing}=missing) where {T}
    lambda = StableMorphism(ReturnTyped(func, AbstractArray{T}), obtain.(input)...)
    GridParam(lambda, input, symbol, init)
end

GridParam(func::Function, input::ParamBox{T}, symbol::SymOrIndexedSym; 
          init::Union{AbstractArray{T}, Missing}=missing) where {T} = 
GridParam(func, (input,), symbol; init)

GridParam(func::Function, input1::ParamBox{T}, input2::ParamBox{T}, 
          symbol::SymOrIndexedSym; 
          init::Union{AbstractArray{T}, Missing}=missing) where {T} = 
GridParam(func, (input1, input2), symbol; init)

GridParam(func::Function, input1::ParamBox{T}, input2::ParamBox{T}, 
          input3::ParamBox{T}, symbol::SymOrIndexedSym; 
          init::Union{AbstractArray{T}, Missing}=missing) where {T} = 
GridParam(func, (input1, input2, input3), symbol; init)

GridParam(par::GridParam{T}, symbol::SymOrIndexedSym=symOf(par); 
          init::Union{AbstractArray{T}, Missing}=par.memory) where {T} = 
GridParam(par.lambda, par.input, symbol, init)

GridParam(input::FlattenedParam{T, N}, symbol::SymOrIndexedSym=symOf(input)) where {T, N} = 
GridParam(StableMorphism(input|>obtain), (input,), symbol)

GridParam(val::AbstractArray, valSym::SymOrIndexedSym, symbol::SymOrIndexedSym=valSym) = 
GridParam(TensorVar(val, valSym), symbol)


struct ParamGrid{T, N, I<:FlattenedParam{T, N}, O} <: ParamNest{T, N, I, O}
    input::ShapedMemory{I, O}
    symbol::IndexedSym

    function ParamGrid(input::ShapedMemory{I, O}, symbol::SymOrIndexedSym) where 
                      {T, N, I<:FlattenedParam{T, N}, O}
        exclude0DimData(input|>typeof)
        checkEmptiness(input.value, :input)
        new{T, N, I, O}(input, IndexedSym(symbol))
    end
end

function ParamGrid(input::AbstractArray{I, O}, symbol::SymOrIndexedSym) where 
                  {T, N, I<:FlattenedParam{T, N}, O}
    exclude0DimData(input|>typeof)
    ParamGrid(ShapedMemory(input), symbol)
end

ParamGrid(pl::ParamGrid, symbol::IndexedSym=pl.symbol) = ParamGrid(pl.input, symbol)


function checkParamContainerArgType2(len::Int, extent::Int)
    if len != extent
        throw(DomainError(len, "The length of `memory` should match "*
                               "`ml::FixedShapeLink`'s specification: $extent."))
    end
    nothing
end


function indexParam(pb::ParamGrid{<:Any, N}, idx::Int, 
                    sym::MissingOr{Symbol}=missing) where {N}
    entry = pb.input[idx]
    if ismissing(sym) || sym==symOf(entry)
        entry
    elseif iszero(N)
        CellParam(entry, sym)
    else
        GridParam(entry, sym)
    end
end


function indexParam(pb::FlattenedParam{T}, idx::Int, 
                    sym::MissingOr{Symbol}=missing) where {T}
    ismissing(sym) && (sym = Symbol(:_, pb.symbol.name))
    CellParam((Retrieve∘ChainPointer)(idx, TensorType(T)), pb, sym)
end

function indexParam(pb::ElementalParam, idx::Int, sym::MissingOr{Symbol}=missing)
    if idx != 1
        throw(BoundsError(pb, idx))
    elseif ismissing(sym) || sym == symOf(res)
        pb
    else
        CellParam(pb, sym)
    end
end

function indexParam(pb::ParamBox{T, N}, idx::Int, 
                    sym::MissingOr{Symbol}=missing) where {T, N}
    ismissing(sym) && (sym = Symbol(:_, pb.symbol.name))
    type = TensorType(AbstractArray{T, N}, outputSizeOf(pb))
    GridParam((Retrieve∘ChainPointer)(idx, type), pb, sym)
end


genDefaultRefParSym(input::ParamBox) = IndexedSym(:_, input.symbol)


getScreenLevelOptions(::Type{<:ParamGrid}) = (0, 2)

getScreenLevelOptions(::Type{<:LinkParam}) = (0,)

getScreenLevelOptions(::Type{<:BaseParam{T, 0}}) where {T} = 
Tuple(0:(checkTypedOpMethods(T) * 2))

getScreenLevelOptions(::Type{<:ParamToken}) = (0,)

getScreenLevelOptions(::Type{<:ParamBatch}) = (0,)

getScreenLevelOptions(::Type{<:PrimitiveParam}) = (1, 2)

getScreenLevelOptions(::Type{<:ParamBox}) = (0, 1, 2)

getScreenLevelOptions(::T) where {T<:ParamBox} = getScreenLevelOptions(T)

function isScreenLevelChangeable(::Type{T}) where {T<:ParamBox}
    minLevel, maxLevel = extrema( getScreenLevelOptions(T) )
    (maxLevel - minLevel) > 0
end

isOffsetEnabled(::ParamBox) = false

function isOffsetEnabled(pb::T) where {T<:CellParam}
    isScreenLevelChangeable(T) && maximum( getScreenLevelOptions(T) ) > 0 && 
    isdefined(pb, :offset) # Only for safety
end

screenLevelOf(p::BaseParam{<:Any, 0}) = Int(p.screen)

screenLevelOf(::ParamToken) = 0

screenLevelOf(::ParamBatch) = 0

screenLevelOf(p::PrimitiveParam) = Int(p.screen)

screenLevelOf(p::ParamGrid) = ifelse(all(l==2 for l in screenLevelOf.(p.input.value)), 2, 0)


function setScreenLevelCore!(p::ParamBox, level::Int)
    @atomic p.screen = TernaryNumber(level)
end

function setScreenLevel!(p::T, level::Int) where {T<:CellParam}
    checkScreenLevel(level, getScreenLevelOptions(T))
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

setScreenLevel(p::CellParam, level::Int) = 
setScreenLevel!(CellParam(p), level)

setScreenLevel(p::TensorVar, level::Int) = 
TensorVar(p.input, p.symbol, TernaryNumber(level))


function swapParamMem!(p::ParamLink{T, N}, 
                       memNew::AbstractArray{ShapedMemory{T, N}}) where {T, N}
    checkLength(memNew, :memNew, length(p.memory), "`length(p.memory)`")
    res = lock( ReentrantLock() ) do
        map(enumerate(memNew)) do (i, innerMemNew)
            memOld = directObtain(p.memory[i])
            p.memory[i] = innerMemNew
            memOld
        end
    end
    reshape(res, last.(p.lambda.axis))
end

function swapParamMem!(p::ParamLink{T, N}, memNew::ShapedMemory{T, N}, 
                       index::Int) where {T, N}
    memOld = directObtain(p.memory[index])
    lock( ReentrantLock() ) do; p.memory[index] = memNew end
    memOld
end

function swapParamMem!(p::LinkParam{T, N}, memNew::ShapedMemory{T, N}) where {T, N}
    swapParamMem!(p.input, memNew, p.index)
end

function swapParamMem!(p::BaseParam{T, N}, memNew::ShapedMemory{T, N}) where {T, N}
    memOld = directObtain(p.memory)
    @atomic p.memory = memNew
    memOld
end

memorize!(p::ParamLink{T, N}, memNew::JaggedAbtArray{T, N}) where {T, N} = 
swapParamMem!(p, map(ShapedMemory, memNew))

memorize!(p::ParamLink{T, 0}, memNew::AbstractArray{T}) where {T} = 
swapParamMem!(p, map(ShapedMemory∘fill, memNew))

memorize!(p::LinkParam{T, N}, memNew::AbstractArray{T, N}) where {T, N} = 
swapParamMem!(p, ShapedMemory(memNew))

memorize!(p::LinkParam{T, 0}, memNew::T) where {T} = 
swapParamMem!(p, (ShapedMemory∘fill)(memNew))

memorize!(p::BaseParam{T, N}, memNew::AbstractArray{T, N}) where {T, N} = 
swapParamMem!(p, ShapedMemory(memNew))

memorize!(p::BaseParam{T, 0}, memNew::T) where {T} = 
swapParamMem!(p, (ShapedMemory∘fill)(memNew))

memorize!(p::ParamFunctor) = memorize!(p, obtain(p))


indexedSymOf(p::ParamBox) = p.symbol

symOf(p::ParamBox) = indexedSymOf(p).name

inputOf(p::ParamBox) = p.input

isDependentParam(p::ParamBox) = (screenLevelOf(p) < 1)

isPrimitiveParam(p::ParamBox) = (screenLevelOf(p) == 1)

#? Maybe a more general type signature?
outputSizeOf(p::PrimitiveParam) = size(p.input)

outputSizeOf(p::ParamFunctor) = size(p.memory)

outputSizeOf(p::ParamGrid) = size(p.input)



mutable struct NodeMarker{T} <: StorageMarker{T}
    visited::Bool
    data::T

    NodeMarker(init::T, ::Type{U}=T) where {T, U} = new{U}(false, init)
end

const ParamDict0D{T, V} = IdDict{ElementalParam{T}, NodeMarker{V}}
const ParamDictSD{T, V} = IdDict{FlattenedParam{T}, NodeMarker{V}}
const ParamDictDD{T, V} = IdDict{ParamBox{T}, NodeMarker{V}}

const ParamInputSource{T} = Memory{<:ParamBox{T}}
const ParamDictId{T} = IdDict{ParamBox{T}, NodeMarker{<:ParamInputSource{T}}}

const DefaultMaxParamPointerLevel = 10

struct ParamPointerBox{T, V0, V1, V2, F<:Function} <: QueryBox{NodeMarker}
    d0::ParamDict0D{T, V0}
    d1::ParamDictSD{T, V1}
    d2::ParamDictDD{T, V2}
    id::ParamDictId{T}
    generator::F
    maxRecursion::Int

    ParamPointerBox(f::F, ::Type{T}, ::Type{V0}, ::Type{V1}, ::Type{V2}, 
                     maxRecursion::Int=DefaultMaxParamPointerLevel) where 
                    {F, T, V0, V1, V2} = 
    new{T, V0, V1, V2, F}( ParamDict0D{T, V0}(), ParamDictSD{T, V1}(), 
                           ParamDictDD{T, V2}(), ParamDictId{T}(), f, maxRecursion )
end

selectParamPointer(d::ParamPointerBox{T}, ::ElementalParam{T}) where {T} = d.d0
selectParamPointer(d::ParamPointerBox{T}, ::FlattenedParam{T}) where {T} = d.d1
selectParamPointer(d::ParamPointerBox{T}, ::ParamBox{T}) where {T} = d.d2
selectParamPointer(d::ParamPointerBox{T}, ::ParamPointer{T}) where {T} = d.id

getParamDataTypeUB(::ParamPointerBox{T, V0, <:Any, <:Any}, 
                   ::ElementalParam{T}) where {T, V0} = V0
getParamDataTypeUB(::ParamPointerBox{T, <:Any, V1, <:Any}, 
                   ::FlattenedParam{T}) where {T, V1} = V1
getParamDataTypeUB(::ParamPointerBox{T, <:Any, <:Any, V2}, 
                   ::ParamBox{T}) where {T, V2} = V2

function checkGetDataRecNum(counter::Int, maxRec::Int)
    if counter > maxRec
        throw( ErrorException("The recursive calling times passed the limit: $maxRec.") )
    end
    nothing
end

function getDataCore1(counter::Int, d::ParamPointerBox{T}, p::ParamBox{T}, 
                      failFlag) where {T}
    counter += 1
    checkGetDataRecNum(counter, d.maxRecursion)
    res = get(selectParamPointer(d, p), p, failFlag)
    res===failFlag ? failFlag : res.data
end

function getDataCore1(counter::Int, d::ParamPointerBox{T}, p::ParamPointer{T}, 
                      failFlag) where {T}
    counter += 1
    checkGetDataRecNum(counter, d.maxRecursion)
    res = get(d.id, p, failFlag)
    res===failFlag ? failFlag : d.generator(getDataCore2(counter, d, res.data, failFlag), p)
end

function getDataCore2(counter::Int, d::ParamPointerBox{T, V0, V1, V2}, 
                      s::ParamInputSource{T}, failFlag) where {T, V0, V1, V2}
    container = Memory{Any}(undef, length(s))
    flag = true
    eleTset = Set{Type}()
    for i in eachindex(s)
        res = getDataCore1(counter, d, s[i], failFlag)
        if res === failFlag
            flag = false
            break
        else
            container[i] = res
            push!(eleTset, typeof(res))
        end
    end
    flag ? ( Memory{Union{eleTset...}}(container)::Memory{<:Union{V0, V1, V2}} ) : failFlag
end

function getData(d::ParamPointerBox{T}, p::ParamBox{T}, default; 
                 failFlag=nothing) where {T}
    res = getDataCore1(0, d, p, failFlag)
    res===failFlag ? default : res
end

function getParamMarker!(pDict::ParamPointerBox{T}, transformer::F, 
                         p::ParamBox{T}) where {T, F}
    mem = transformer(p)
    tUB = getParamDataTypeUB(pDict, p)
    markerType = NodeMarker{tUB}
    get!(selectParamPointer(pDict, p), p) do
        NodeMarker(mem, tUB)
    end::markerType
end

function getParamMarker!(pDict::ParamPointerBox{T}, transformer::F, 
                         p::ParamPointer{T}) where {T, F}
    input = p.input
    eleT = eltype(input)
    parId = Memory{eleT}(undef, length(input))
    for (i, par) in enumerate(input)
        recursiveTransform!(transformer, pDict, par)
        parId[i] = par
    end
    markerType = NodeMarker{Memory{eleT}}
    get!(selectParamPointer(pDict, p), p) do
        NodeMarker(Memory{eleT}(parId))
    end::markerType
end

function recursiveTransformCore1!(generator::F, marker::NodeMarker, 
                                  p::ParamBox) where {F<:Function}
    # Depth-first search by recursive calling
    if !marker.visited
        marker.visited = true
        marker.data = generator(p)
    else
        marker.data
    end
end

function recursiveTransformCore2!(generator::F, marker::NodeMarker, p::ParamPointer{T}, 
                                  pDict::ParamPointerBox{T}) where {F<:Function, T}
    val = map(marker.data) do par
        res = getData(pDict, par, nothing)
        res === nothing && throw(ErrorException("Could note locate the value for $par."))
        res
    end
    generator(val, p)
end

function recursiveTransform!(transformer::F, pDict::ParamPointerBox{T}, 
                             p::PrimitiveParam{T, N}) where {F, T, N}
    marker = getParamMarker!(pDict, transformer, p)
    recursiveTransformCore1!(transformer, marker, p)
    marker.data
end

function recursiveTransform!(transformer::F, pDict::ParamPointerBox{T}, 
                             p::ParamBox{T}) where {F, T}
    marker = getParamMarker!(pDict, transformer, p)

    recursiveTransformCore1!(marker, p) do par
        sl = checkScreenLevel(screenLevelOf(par), getScreenLevelOptions(par|>typeof))
        res = if sl == 0
            map(ele->recursiveTransform!(transformer, pDict, ele), par.input)
        else
            T
        end
        transformer(res, par)
    end

    marker.data
end

function recursiveTransform!(transformer::F, pDict::ParamPointerBox{T}, 
                             p::ParamPointer{T}) where {F, T}
    marker = getParamMarker!(pDict, transformer, p)
    recursiveTransformCore2!(transformer, marker, p, pDict)
end


obtainCore(p::PrimitiveParam) = directObtain(p.input)
obtainCore(p::BaseParam) = directObtain(p.memory)
obtainCore(p::ParamLink) = reshape(map(directObtain, p.memory), last.(p.lambda.axis))

function obtainCore(inputVal::NTuple{A, AbtArr210L{T}}, 
                    p::BaseParam{T, <:Any, <:ParamInput{T, A}}) where {T, A}
    p.lambda(inputVal...)
end

# function obtainCore(inputVal::NTuple{A, AbtArr210L{T}}, 
#                     p::ParamMesh{T, <:Any, <:ParamInput{T, A}}) where {T, A}
#     f = p.lambda
#     valRaw = f.f( inputVal... )
#     Memory{eltype(valRaw)}(valRaw[begin:(begin + f.extent - 1)])
# end

function obtainCore(inputVal::NTuple{A, AbtArr210L{T}}, 
                    p::CellParam{T, <:Any, <:ParamInput{T, A}}) where {T, A}
    screenLevelOf(p)==0 || throw(AssertionError("The screen level of `p` should be 0."))
    shiftVal = genValShifter(T, (isOffsetEnabled(p) ? p.offset : nothing))
    p.lambda(inputVal...) |> shiftVal
end

obtainCore(::Type{T}, p::CellParam{T}) where {T} = p.offset

function obtainCore(val::Memory{T}, p::ParamGrid{T, 0}) where {T}
    reshape(val, p.input.shape)
end

obtainCore(val::Memory{<:AbstractArray{T, N}}, p::ParamGrid{T, N}) where {T, N} = 
reshape(val, p.input.shape)

const ParamValDict{T} = ParamPointerBox{ T, T, AbstractArray{T}, JaggedAbtArray{T}, 
                                          typeof(obtainCore) }

genParamValDict(::Type{T}, maxRecursion::Int=DefaultMaxParamPointerLevel) where {T} = 
ParamPointerBox(obtainCore, T, T, AbstractArray{T}, JaggedAbtArray{T}, maxRecursion)

# function searchObtain(pDict::ParamValDict{T}, p::ParamMesh{T}) where {T}
#     res = recursiveTransform!(obtainCore, pDict, p)
#     reshape(res, last.(p.lambda.axis)) |> collect
# end

searchObtain(pDict::ParamValDict{T}, p::ParamBox{T}) where {T} = 
recursiveTransform!(obtainCore, pDict, p)

obtainINTERNALcore(p::ParamBox{T}, maxRecursion::Int) where {T} = 
searchObtain(genParamValDict(T, maxRecursion), p)

obtainINTERNALcore(ps::AbstractArray{<:ParamBox{T}}, maxRecursion::Int) where {T} = 
map(p->obtainINTERNAL(p, maxRecursion), ps)

obtainINTERNAL(p::PrimitiveParam, ::Int) = obtainCore(p)

function obtainINTERNAL(p::BaseParam{T, 0}, maxRecursion::Int) where {T}
    isDependentParam(p) ? obtainINTERNALcore(p, maxRecursion) : obtainCore(T, p)
end

function obtainINTERNAL(p::CompositeParam{T}, maxRecursion::Int) where {T}
    obtainINTERNALcore(p, maxRecursion)
end

function obtainINTERNAL(p::ParamNest{T}, maxRecursion::Int) where {T}
    if any(isDependentParam, p.input)
        obtainINTERNALcore(p, maxRecursion)
    else
        obtainINTERNALcore(p.input, maxRecursion)
    end
end

function obtainINTERNAL(ps::AbstractArray{<:PrimitiveParam{T}}, ::Int) where {T}
    obtainINTERNALcore(ps, 0)
end

function obtainINTERNAL(ps::AbstractArray{<:ParamBox{T}}, maxRecursion::Int) where {T}
    if any(isDependentParam, ps)
        pValDict = genParamValDict(T, maxRecursion)
        map(p->searchObtain(pValDict, p), ps)
    else
        obtainINTERNALcore(ps, maxRecursion)
    end
end

function obtainINTERNAL(p::AbtArrayOr{<:ParamBox}, maxRecursion::Int)
    map(i->obtainINTERNAL(i, maxRecursion), p)
end

obtain(p::ParamBox; maxRecursion::Int=DefaultMaxParamPointerLevel) = 
obtainINTERNAL(p, maxRecursion)

obtain(p::ParamBoxUnionArr{ParamBox{T}}; 
       maxRecursion::Int=DefaultMaxParamPointerLevel) where {T} = 
obtainINTERNAL(p, maxRecursion)

obtain(p::ParamBoxUnionArr; maxRecursion::Int=DefaultMaxParamPointerLevel) = 
obtainINTERNAL(itself.(p), maxRecursion)

################################

(pn::ParamBox)() = obtain(pn)

function setVal!(par::PrimitiveParam{T, N}, val::AbstractArray{T, N}) where {T, N}
    if Int(par.screen) == 1
        safelySetVal!(par.input.value, val)
    else
        throw(ArgumentError("`par` is a constant parameter that should not be modified."))
    end
end

setVal!(par::PrimitiveParam{T, 0}, val::T) where {T} = setVal!(par, fill(val))

function setVal!(par::CellParam{T}, val::T) where {T}
    isPrimitiveParam(par) || 
    throw(ArgumentError("`par` must behave like a primitive parameter."))
    @atomic par.offset = val
end

# import Base: iterate, size, length, eltype, broadcastable
# length(::FixedSizeParam{<:Any, N}) where {N} = N
# eltype(np::FixedSizeParam) = eltype(np.input)

# iterate(::FixedSizeParam{<:Any, 1}, args...) = iterate(1, args...)
# size(::FixedSizeParam{<:Any, 1}, args...) = size(1, args...)
# broadcastable(np::FixedSizeParam{<:Any, 1}) = Ref(np)

# iterate(np::FixedSizeParam, args...) = iterate(np.input, args...)
# size(np::FixedSizeParam, args...) = size(np.input, args...)
# broadcastable(np::FixedSizeParam) = Base.broadcastable(np.input)


# struct CachedJParam{T, N, O, P<:ParamBox{T, N, O}, 
#                     E<:Union{T, AbstractArray{T, N}}} <: QueryBox{P}
#     source::P
#     cache::ShapedMemory{E, O}

#     function CachedJParam(source::P) where {T, N, O, P<:ParamBox{T, N, O}}
#         output = obtain(source)
#         new{T, N, O, P, eltype(output)}(source, ShapedMemory(output))
#     end
# end

# CachedJParam(param::CachedJParam) = CachedJParam(param.param)

# function extract!(cp::CachedJParam{T}; returnCache::Bool=false, 
#                   updateCache::Bool=true) where {T}
#     if returnCache
#         res = cp.cache
#     else
#         res = obtain(cp.param)
#         updateCache && (box.cache .= res)
#     end
#     res
# end

# const EleCJParam{T} = CachedJParam{T, 0, 0}
# const ISpCJParam{T, N} = CachedJParam{T, N, 0}
# const OSpCJParam{T, O} = CachedJParam{T, 0, O}


struct ParamMarker{T, N, O} <: IdentityMarker{ParamBox{T, N, O}}
    code::UInt
    data::IdentityMarker
    func::IdentityMarker
    meta::Tuple{ValMkrPair{<:Union{T, Nothing}}, ValMkrPair{Int}, ValMkrPair{Symbol}}

    function ParamMarker(p::P) where {T, N, O, P<:ParamBox{T, N, O}}
        offset = :offset => markObj(isOffsetEnabled(p) ? p.offset : nothing)
        code = offset.second.code

        sl = screenLevelOf(p)
        screen = :screen => markObj(sl)
        code = hash(screen.second, code)

        sym = :symbol => markObj(p.symbol.name)
        code = hash(sym.second, code)

        meta = (offset, screen, sym)

        func = markObj((P <: ParamFunctor && sl == 0) ? p.lambda : nothing)
        code = hash(func, code)

        data = markParam(p)
        code = hash(data.code, code)

        new{T, N, O}(code, data, func, meta)
    end
end

function markParam(input::NonEmptyTuple{ParamBox}, refParam::ParamBox)
    markObj( markParam.(input, Ref(refParam)) )
end

function markParam(input::ParamBoxUnionArr, ::ParamBox)
    markObj(input)
end

function markParam(param::PrimitiveParam, ::ParamBox)
    Identifier(param.input)
end

function markParam(param::ParamFunctor, refParam::ParamBox)
    if screenLevelOf(param) > 0 || param === refParam
        Identifier(param.memory)
    else
        markParam(param.input, refParam)
    end
end

function markParam(param::ParamNest, refParam::ParamBox)
    markParam(param.input, refParam)
end

markParam(param::ParamBox) = markParam(param, param)

markObj(input::ParamBox) = ParamMarker(input)

function ==(marker1::T, marker2::T) where {T<:ParamMarker}
    if marker1.code == marker2.code
        marker1.data == marker2.data
    else
        false
    end
end


compareParamBox(p1::T, p2::T) where {T<:PrimitiveParam} = p1 === p2

function compareParamBox(p1::ParamBox{T, N, O}, 
                         p2::ParamBox{T, N, O}) where {T, N, O}
    p1 === p2 || ParamMarker(p1) == ParamMarker(p2)
end

compareParamBox(::ParamBox, ::ParamBox) = false

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
#     if compareParamBox(pn1, pn2)
#         repeatedlyApply(op, pn1, 2)
#     else
#         operateByCore(op, pn1, pn2)
#     end
# end


# addCellParam(pn1::CellParam{T}, pn2::CellParam{T}) where {T<:Real} = operateBy(+, pn1, pn2)

# mulCellParam(pn1::CellParam{T}, pn2::CellParam{T}) where {T<:Real} = operateBy(*, pn1, pn2)
# mulCellParam(pn::CellParam{T}, coeff::T) where {T} = operateBy(*, pn, coeff)
# mulCellParam(coeff::T, pn::CellParam{T}) where {T} = mulCellParam(pn, coeff)


# function sortParamContainers(::Type{C}, f::F, field::Symbol, roundAtol::T) where 
#                             {T, C<:ParamFunction{T}, F}
#     let roundAtol=roundAtol, f=f, field=field
#         function (container::C)
#             ele = getproperty(container, field)
#             ( roundToMultiOfStep(f(container), nearestHalfOf(roundAtol)), 
#               symFromIndexSym(ele.symbol), ParamMarker(ele) )
#         end
#     end
# end
function getParams(source)
    (first∘getFieldParams)(source)
end

function getFieldParams(source::T) where {T}
    paramPairs = Tuple{ParamBox, ChainPointer}[]
    getFieldParamsCore!(paramPairs, source, (ChainPointer∘TensorType)(T))
    first.(paramPairs), last.(paramPairs)
end

function getFieldParamsCore!(paramPairs::Vector{Tuple{ParamBox, ChainPointer}}, 
                             source::ParamBox, anchor::ChainPointer)
    push!(paramPairs, (source, anchor))
    nothing
end

function getFieldParamsCore!(paramPairs::Vector{Tuple{ParamBox, ChainPointer}}, 
                             source::T, anchor::ChainPointer) where {T}
    searchParam = false
    if source isa Union{Tuple, AbstractArray}
        if isempty(source)
            return nothing
        else
            searchParam = true
            content = eachindex(source)
        end
    elseif isstructtype(T) && !( Base.issingletontype(T) )
        searchParam = true
        content = fieldnames(T)
    end
    if searchParam
        for fieldSym in content
            field = getField(source, fieldSym)
            outputValConstraint = field isa ParamBox ? TensorType(field) : TensorType()
            anchorNew = ChainPointer(anchor, ChainPointer(fieldSym, outputValConstraint))
            getFieldParamsCore!(paramPairs, field, anchorNew)
        end
    end
    nothing
end

uniqueParams(ps::AbstractArray{<:ParamBox}) = 
markUnique(ps, compareFunction=compareParamBox)[end]


function markParamsCore!(indexDict::IdDict{Symbol, Int}, leafPars)
    for i in leafPars
        sym = i.symbol.name
        get!(indexDict, sym, 0)
        i.symbol.index = (indexDict[sym] += 1)
    end
end

function markParams!(pars::AbstractVector{<:ParamBox{T}}) where {T}
    nodes, marks1, marks2 = topoSort(pars)
    leafPars = nodes[.!marks1 .*   marks2]
    rootPars = nodes[  marks1 .* .!marks2]
    selfPars = nodes[.!marks1 .* .!marks2]
    parIdxDict = IdDict{Symbol, Int}()

    par0Dids = findall(x->(x isa ElementalParam{T}), leafPars)
    leafParsFormated = if isempty(par0Dids)
        markParamsCore!(parIdxDict, leafPars)
        convert(Vector{PrimParamEle{T}}, leafPars)
    else
        leafP0Ds = ElementalParam{T}[splice!(leafPars, par0Dids)...]
        markParamsCore!(parIdxDict, leafP0Ds)
        markParamsCore!(parIdxDict, leafPars)
        PrimParamEle{T}[leafP0Ds, leafPars...]
    end
    (leafParsFormated, rootPars, selfPars) # inputParam, outputParam, selfParam
end

markParams!(b::AbtArrayOr) = b |> getParams |> markParams!

#!! Change `hbNodesIdSet` to IDSet
function topoSortCore!(hbNodesIdSet::Set{UInt}, 
                       orderedNodes::Vector{<:ParamBox{T}}, 
                       haveBranches::Vector{Bool}, connectRoots::Vector{Bool}, 
                       node::ParamBox{T}, recursive::Bool=false) where {T}
    sl = checkScreenLevel(screenLevelOf(node), getScreenLevelOptions(node))

    if sl in (0, 1)
        idx = findfirst(Fix2(compareParamBox, node), orderedNodes)
        if idx === nothing
            hasBranch = ifelse(sl == 0, true, false)
            if hasBranch
                id = objectid(node)
                isRegisteredSubRoot = (id in hbNodesIdSet)
                if !isRegisteredSubRoot
                    push!(hbNodesIdSet, objectid(node))
                    for child in node.input
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

function topoSortINTERNAL(nodes::AbstractVector{<:ParamBox{T}}) where {T}
    orderedNodes = ParamBox{T}[]
    haveBranches = Bool[]
    connectRoots = Bool[]
    hbNodesIdSet = Set{UInt}()
    for node in nodes
        topoSortCore!(hbNodesIdSet, orderedNodes, haveBranches, connectRoots, node)
    end
    orderedNodes, haveBranches, connectRoots
end

function topoSort(nodes::AbstractVector{<:ParamBox{T}}) where {T}
    uniqueParams(nodes) |> topoSortINTERNAL
end

topoSort(node::CellParam{T}) where {T} = topoSortINTERNAL([node])


# # Sever the connection of a node to other nodes
# sever(pv::TensorVar) = TensorVar(obtain(pv), pv.symbol)

# sever(obj::Any) = deepcopy(obj)

# sever(obj::Union{Tuple, AbstractArray}) = sever.(obj)

# function sever(pf::T) where {T<:ParamFunction}
#     severedFields = map(field->(sever∘getproperty)(pf, field), fieldnames(pf))
#     T(severedFields...)
# end

# struct ParamBoxCacheBox{T, P<:ParamBox{T}} <: QueryBox{Union{P, ShapedMemory{T}}}
#     param::Dict{Symbol, P}
#     cache::Dict{Symbol, ShapedMemory{T}}

#     function ParamBoxCacheBox(par::AbstractArray{P}, sym::AbstractArray{Symbol}) where 
#                              {T, P<:ParamBox{T}}
#         if size(par) != size(sym)
#             throw(AssertionError("`par` and `sym` should have the same length"))
#         end
#         val = ShapedMemory.(T, obtain.(par))
#         new{T, P}(Dict(sym .=> par), Dict(sym .=> val))
#     end
# end

# function extract(box::ParamBoxCacheBox{T}, sym::Symbol; 
#                  returnCache::Bool=false, updateCache::Bool=true) where {T}
#     if returnCache
#         res = itself.(box.cache[sym])
#     else
#         res = obtain(box.param[sym])
#         updateCache && (box.cache[sym] = res)
#     end
#     res
# end

#####


function genCellEncoder(::Type{T}, sym::Symbol; defaultScreenLevel::Int=1) where {T}
    function (input::ParamOrValue{T})
        if input isa ElementalParam{T}
            input
        else
            p = CellParam(T(input), sym)
            setScreenLevel!(p, defaultScreenLevel)
            p
        end
    end
end


evalParamSource(s::ParamBox) = obtain(s)

evalParamSource(s::ParamBoxUnionArr{<:ParamBox, 1}) = obtain(s)

evalParamSource(s::GeneralParamInput{T, <:FlatParamVec{T}}) where {T} = mapLayout(obtain, s)

evalParamSource(s::GeneralParamInput) = mapLayout(evalParamSource, s)


# FlatParamVec: [[d0...], d1...]          # Used for  input-param set

# MiscParamVec: [[[d0...], d1...], d2...] # Used for output-param set

pushParam!(arr::AbstractArray, param::ParamBox) = push!(arr, param)


#? Possibly rename the field names in Flat/MiscParamSet?
## (Also applied for TemporaryStorage, FixedSizeStorage, ParamPointerBox)
#? Allow AbstractParamSet to be empty if all sections are empty?
struct FlatParamSet{T, P1<:ElementalParam{<:T}, 
                    P2<:FlattenedParam{<:T}} <: AbstractFlatParamSet{T, Vector{P1}, P2}
    d0::Vector{P1}
    d1::Vector{P2}

    FlatParamSet{T}(d0::Vector{P1}, d1::Vector{P2}) where 
                   {T, P1<:ElementalParam{<:T}, P2<:FlattenedParam{<:T}} = 
    new{T, P1, P2}(d0, d1)
end

const TypedFlatParamSet{T, P1<:ElementalParam{T}, P2<:FlattenedParam{T}} = 
      FlatParamSet{T, P1, P2}

const PrimitiveParamSet{T, P1<:ElementalParam{T}, P2<:InnerSpanParam{T}} = 
      TypedFlatParamSet{T, P1, P2}

size(fps::FlatParamSet) = size(fps.d1) .+ 1

firstindex(::FlatParamSet) = 1

lastindex(fps::FlatParamSet) = length(fps.d1) + 1

getFlatSetIndexCore(target, i::Int) = 
getindex(target.d1, i+firstindex(target.d1)-2)

function getFlatSetIndex(target, i::Int)
    if i == 1
        target.d0
    else
        getFlatSetIndexCore(target, i)
    end
end

function getFlatSetIndex(target, sector::Symbol, i::Int)
    getindex(getfield(target, sector), i)
end

getindex(fps::FlatParamSet, i::Int) = getFlatSetIndex(fps, i)

getindex(fps::FlatParamSet, sector::Symbol, i::Int) = getFlatSetIndex(fps, sector, i)

function setindex!(fps::FlatParamSet, val, i::Int)
    if i == firstindex(fps)
        fps.d0 .= val
    else
        setindex!(fps.d1, val, i+firstindex(fps.d1)-2)
    end
end

function setindex!(fps::FlatParamSet, val, sector::Symbol, i::Int)
    setindex!(getfield(fps, sector), val, i)
end

pushParam!(fps::FlatParamSet, a::ElementalParam) = push!(fps.d0, a)
pushParam!(fps::FlatParamSet, a::FlattenedParam) = push!(fps.d1, a)

function iterate(fps::AbstractFlatParamSet)
    i = firstindex(fps)
    (getindex(fps, i), i+1)
end

function iterate(fps::AbstractFlatParamSet, state)
    if state > length(fps)
        nothing
    else
        (getindex(fps, state), state+1)
    end
end

length(fps::FlatParamSet) = length(fps.d1) + 1

axes(fps::FlatParamSet)	= map(Base.OneTo, size(fps))

function similar(fps::FlatParamSet{T, P1, P2}, shape::Tuple{Int}=size(fps); 
                 innerShape::Tuple{Int}=size(fps.d0)) where 
                {T, P1<:ElementalParam{<:T}, P2<:FlattenedParam{<:T}}
    checkPositivity(shape|>first)
    res = Vector{Union{Vector{P1}, P2}}(undef, shape)
    res[begin] = Vector{P1}(undef, innerShape)
    res
end

getproperty(fps::FlatParamSet, field::Symbol) = getfield(fps, field)

const FlatParamSetMixedVec{T, P1<:ElementalParam{<:T}, P2<:FlattenedParam{<:T}, 
                           P3<:ParamBox{<:T}} = 
      AbstractMiscParamSet{T, FlatParamSet{T, P1, P2}, P3}

function initializeParamSet(::Type{FlatParamSet}, ::Type{T}=Any; 
                            d0Type::MissingOr{Type{<:FlattenedParam{<:T}}}=missing, 
                            d1Type::MissingOr{Type{<:FlattenedParam{<:T}}}=missing) where 
                           {T}
    bl = isconcretetype(T)
    if ismissing(d0Type)
        d0Type = ifelse(bl, ElementalParam{T}, ElementalParam{<:T})
    end
    if ismissing(d1Type)
        d1Type = ifelse(bl, FlattenedParam{T}, FlattenedParam{<:T})
    end
    FlatParamSet{T}(d0Type[], d1Type[])
end

initializeParamSet(::Type{FlatParamSet{T, P1, P2}}) where {T, P1, P2} = 
FlatParamSet(P1[], P2[])

initializeParamSet(f::Function) = 
initializeParamSet(SelectTrait{ParameterizationStyle}()(f))

initializeParamSet(::GenericFunction) = initializeParamSet(FlatParamSet)

initializeParamSet(::TypedParamFunc{T}) where {T} = initializeParamSet(FlatParamSet, T)


# function initializeParamSet(::Type{MiscParamSet}, ::Type{T}=Any; 
#                             d0Type::MissingOr{Type{<:ElementalParam{<:T}}}=missing, 
#                             d1Type::MissingOr{Type{<:FlattenedParam{<:T}}}=missing, 
#                             d2Type::MissingOr{Type{<:ParamBox{<:T}}}=missing) where {T}
#     inner = initializeParamSet(FlatParamSet, T; d0Type, d1Type)
#     if ismissing(d2Type)
#         d2Type = ifelse(isconcretetype(T), ParamBox{T}, ParamBox{<:T})
#     end
#     MiscParamSet(inner, d2Type[])
# end

# initializeParamSet(::Type{MiscParamSet{T, P1, P2, P3}}) where {T, P1, P2, P3} = 
# MiscParamSet(FlatParamSet(P1[], P2[]), P3[])


#!  SingleNestParamSet
#!  DoubleNestParamSet
const FlatPSetInnerPtr{T} = PointPointer{T, 2, Tuple{FirstIndex, Int}}
const FlatPSetOuterPtr{T} = IndexPointer{Volume{T}, 1}
const FlatParamSetIdxPtr{T} = Union{FlatPSetInnerPtr{T}, FlatPSetOuterPtr{T}}

struct FlatParamSetFilter{T} <: PointerStack{1, 2}
    d0::ShapedMemory{FlatPSetInnerPtr{T}, 1}
    d1::Memory{FlatPSetOuterPtr{T}}
end

function FlatParamSetFilter(d0::AbstractVector{FlatPSetInnerPtr{T}}, 
                            d1::AbstractVector{FlatPSetOuterPtr{T}}) where {T}
    d0 isa ShapedMemory || (d0 = ShapedMemory(d0))
    d1 isa Memory || (d1 = getMemory(d1))
    FlatParamSetFilter(d0, d1)
end

const FilteredFlatParamSet{T, N} = 
      FilteredObject{<:TypedFlatParamSet{T}, FlatParamSetFilter{T}}

const TypedParamInput{T} = Union{TypedFlatParamSet{T}, FilteredFlatParamSet{T}}

function FlatParamSetFilter(pSet::FlatParamSet{T}, d0Ids::AbstractVector{Int}, 
                            d1Ids::AbstractVector{Int}) where {T}
    d0Ptr = map(d0Idx->ChainPointer((FirstIndex, d0Idx), TensorType(T)), d0Ids)
    d1Ptr = map(d1Idx->ChainPointer((d1Idx+1,), TensorType(pSet.d1[d1Idx])), d1Ids)
    FlatParamSetFilter(d0Ptr, d1Ptr)
end

size(fps::FlatParamSetFilter) = size(fps.d1) .+ 1

firstindex(::FlatParamSetFilter) = 1

lastindex(fps::FlatParamSetFilter) = length(fps.d1) + 1

getindex(fps::FlatParamSetFilter, i::Int) = getFlatSetIndex(fps, i)

getindex(fps::FlatParamSetFilter, sector::Symbol, i::Int) = getFlatSetIndex(fps, sector, i)

function iterate(fps::FlatParamSetFilter)
    i = firstindex(fps)
    (getindex(fps, i), i+1)
end

function iterate(fps::FlatParamSetFilter, state)
    if state > length(fps)
        nothing
    else
        (getindex(fps, state), state+1)
    end
end

length(fps::FlatParamSetFilter) = length(fps.d1) + 1

getproperty(fps::FlatParamSetFilter, field::Symbol) = getfield(fps, field)


struct FlatParamSubset{T, P1<:ElementalParam{<:T}, P2<:FlattenedParam{<:T}
                       } <: AbstractFlatParamSet{T, ShapedMemory{P1, 1}, P2}
    core::FilteredObject{FlatParamSet{T, P1, P2}, FlatParamSetFilter{T}}
end

const GeneralFlatParamSet{T, P1, P2} = 
      Union{FlatParamSet{T, P1, P2}, FlatParamSubset{T, P1, P2}}

size(fps::FlatParamSubset) = size(fps.core.ptr)

firstindex(fps::FlatParamSubset) = firstindex(fps.core.ptr)

lastindex(fps::FlatParamSubset) = lastindex(fps.core.ptr)

getindex(fps::FlatParamSubset, i::Int) = getField(fps.core.obj, getindex(fps.core.ptr, i))

getindex(fps::FlatParamSubset, sector::Symbol, i::Int) = 
getField(fps.core.obj, getindex(fps.core.ptr, sector, i))

function setindex!(fps::FlatParamSubset, val, i::Int)
    setindex!(fps.core.obj, val, getindex(fps.core.ptr, i))
end

function setindex!(fps::FlatParamSubset, val, sector::Symbol, i::Int)
    setindex!(fps.core.obj, val, getindex(fps.core.ptr, sector, i))
end

pushParam!(::FlatParamSubset, ::Union{ElementalParam, FlattenedParam}) = 
throw(AssertionError("The size of `FlatParamSubset` cannot be changed."))

length(fps::FlatParamSubset) = length(fps.core.ptr)

axes(fps::FlatParamSubset)= map(Base.OneTo, size(fps))

function similar(fps::FlatParamSubset{T, P1, P2}, shape::Tuple{Int}=size(fps); 
                 innerShape::Tuple{Int}=size(fps.code.d0)) where 
                {T, P1<:ElementalParam{<:T}, P2<:FlattenedParam{<:T}}
    checkPositivity(shape|>first)
    res = Memory{Union{ShapedMemory{P1, 1}, P2}}(undef, shape)
    res[begin] = ShapedMemory{P1}(undef, innerShape)
    res
end

getproperty(fps::FlatParamSubset, field::Symbol) = getfield(fps, field)

getField(fps::FlatParamSubset, field::GeneralEntryPointer) = getField(fps.core, field)


#= Additional Method =#
getField(obj::FlatParamSetFilter, ptr::FlatPSetInnerPtr) = 
getindex(obj.d0, last(ptr.chain))

getField(obj::FlatParamSetFilter, ptr::FlatPSetOuterPtr) = 
getFlatSetIndexCore(obj, first(ptr.chain))

function getField(prev::FlatParamSetFilter, here::FlatParamSetFilter)
    d0New = map(here.d0) do ptr
        getField(prev, ptr)
    end
    d1New = map(here.d1) do ptr
        getField(prev, ptr)
    end
    FlatParamSetFilter(d0New, d1New)
end

function refocus!(here::FlatParamSetFilter, prev::FlatParamSetFilter)
    for i in eachindex(here.d0)
        ptr = here.d0[i]
        here.d0[i] = getField(prev, ptr)
    end

    for i in eachindex(here.d1)
        ptr = here.d1[i]
        here.d1[i] = getField(prev, ptr)
    end
    here
end


getParamSector(source::FlatParamSet, ::Type{<:ElementalParam}) = 
( first(source), ChainPointer(FirstIndex()) )

getParamSector(source::AbstractVector, ::Type{<:ParamBox}) = 
( itself(source), ChainPointer() )


function locateParam!(paramSet::AbstractVector, target::P) where {P<:ParamBox}
    returnType = TensorType(target)
    if isempty(paramSet)
        pushParam!(paramSet, target)
        ChainPointer(firstindex(paramSet), returnType)
    else
        paramSector, anchor = getParamSector(paramSet, P)
        idx = findfirst(x->compareObj(x, target), paramSector)
        if idx === nothing
            pushParam!(paramSector, target)
            idx = length(paramSector)
        end
        ChainPointer(anchor, ChainPointer(idx, returnType))
    end
end

function locateParam!(paramSet::AbstractVector, target::AbstractArray{<:ParamBox{T}, N}; 
                      emptyReturnEltype::Type{<:TypedPointer{T}}=TypedPointer{T}) where 
                     {T, N}
    if isempty(target)
        Array{emptyReturnEltype}(undef, ntuple(_->0, Val(N)))
    else
        map(x->locateParam!(paramSet, x), target)
    end
end

function locateParam!(paramSet::AbstractVector, target::NonEmptyTuple{ParamBox})
    locateParam!.(Ref(paramSet), target)
end

locateParam!(paramSet::AbstractVector, target::NamedTuple) = 
locateParam!(paramSet, values(target))

function locateParam!(paramSet::FlatParamSet{T}, target::FlatParamSet{T}) where {T}
    d0ptrs, d1ptrs = map( fieldnames(FlatParamSet), 
                          (FlatPSetInnerPtr{T}, FlatPSetOuterPtr{T}) ) do n, t
        locateParam!(getfield(paramSet, n), getfield(target, n), emptyReturnEltype=t)
    end
    d0ptrs = ChainPointer.(Ref(FirstIndex()), d0ptrs)
    if !isempty(d1ptrs)
        offset = 2 - firstindex(paramSet.d1)
        d1ptrs = map(x->ChainPointer(x.chain .+ offset, x.type), d1ptrs)
    end
    FlatParamSetFilter(d0ptrs, d1ptrs)
end

function locateParam!(paramSet::AbstractVector, target::FlatParamSet{T}) where {T}
    d0ptrs, d1ptrs = map(fieldnames(FlatParamSet)) do n
        locateParam!.(Ref(paramSet), getfield(target, n))
    end
    FlatParamSetFilter(d0ptrs, d1ptrs)
end


const DimensionalSpanMemory{T} = Union{T, ShapedMemory{T}, ShapedMemory{ShapedMemory{T}}}

struct DimSpanDataCacheBox{T} <: QueryBox{DimensionalSpanMemory{T}}
    d0::Dict{Identifier, T}
    d1::Dict{Identifier, ShapedMemory{T}}
    d2::Dict{Identifier, ShapedMemory{ShapedMemory{T}}}
end

DimSpanDataCacheBox(::Type{T}) where {T} = 
DimSpanDataCacheBox( Dict{Identifier, T}(), 
                     Dict{Identifier, ShapedMemory{T}}(), 
                     Dict{Identifier, ShapedMemory{ShapedMemory{T}}}() )

getDimSpanSector(cache::DimSpanDataCacheBox{T}, ::ElementalParam{T}) where {T} = cache.d0

getDimSpanSector(cache::DimSpanDataCacheBox{T}, ::FlattenedParam{T}) where {T} = cache.d1

getDimSpanSector(cache::DimSpanDataCacheBox{T}, ::ParamBox{T}) where {T} = cache.d2

formatDimSpanMemory(::Type{T}, val::T) where {T} = itself(val)

formatDimSpanMemory(::Type{T}, val::AbstractArray{T}) where {T} = ShapedMemory(val)

formatDimSpanMemory(::Type{T}, val::JaggedAbtArray{T}) where {T} = 
ShapedMemory(map(ShapedMemory, val))

function cacheParam!(cache::DimSpanDataCacheBox{T}, param::ParamBox{T}) where {T}
    get!(getDimSpanSector(cache, param), Identifier(param)) do
        formatDimSpanMemory(T, obtain(param))
    end
end

function cacheParam!(cache::DimSpanDataCacheBox{T}, s::FlatParamSubset{T}) where {T}
    cacheParam!(cache, s.core)
end

function cacheParam!(cache::DimSpanDataCacheBox{T}, s::TypedParamInput{T}, 
                     ptr::CompositePointer) where {T}
    evalField(s, ptr) do par
        cacheParam!(cache, par)
    end
end

function cacheParam!(cache::DimSpanDataCacheBox{T}, 
                     s::GeneralParamInput{T, <:TypedParamSetVec{T}}) where {T}
    mapLayout(s) do p
        cacheParam!(cache, p)
    end
end


# Methods for parameterized functions

function evalFunc(func::F, input::T) where {F<:Function, T}
    fCore, pSet, _ = unpackFunc(func)
    evalFunc(fCore, pSet, input)
end

function evalFunc(fCore::F, pSet::Union{DirectParamSource, TypedParamInput}, 
                  input::T) where {F<:Function, T}
    fCore(input, evalParamSource(pSet))
end

function evalFunc(fCore::F, pVals::AbtVecOfAbtArr, input::T) where {F<:Function, T}
    fCore(input, pVals)
end

#! Possibly adding memoization in the future to generate/use the same param set to avoid 
#! bloating `Quiqbox.IdentifierCache` and prevent repeated computation.
unpackFunc(f::F) where {F<:Function} = unpackFunc!(f, initializeParamSet(f))

unpackFunc!(f::F, paramSet::AbstractVector) where {F<:Function} = 
unpackFunc!(SelectTrait{ParameterizationStyle}()(f), f, paramSet)

unpackFunc!(::TypedParamFunc, f::Function, paramSet::AbstractVector) = 
unpackParamFunc!(f, paramSet)

unpackFunc!(::GenericFunction, f::Function, paramSet::AbstractVector) = 
unpackTypedFunc!(f, paramSet)

const FieldPtrPair{T} = Pair{<:ChainPointer, <:FlatParamSetIdxPtr{T}}
const FieldPtrPairs{T} = AbstractVector{<:FieldPtrPair{T}}
const FieldPtrDict{T} = AbstractDict{<:ChainPointer, <:FlatParamSetIdxPtr{T}}
const EmptyFieldPtrDict{T} = TypedEmptyDict{Union{}, FlatPSetInnerPtr{T}}
const FiniteFieldPtrDict{T, N} = FiniteDict{N, <:ChainPointer, <:FlatParamSetIdxPtr{T}}

const FieldValDict{T} = AbstractDict{<:FlatParamSetIdxPtr{T}, <:Union{T, AbstractArray{T}}}
const ParamValOrDict{T} = Union{AbtVecOfAbtArr{T}, FieldValDict{T}}

abstract type FieldParamPointer{R} <: Any end

struct MixedFieldParamPointer{T, R<:FieldPtrDict{T}} <: FieldParamPointer{R}
    core::R
    tag::Identifier
end

function MixedFieldParamPointer(paramPairs::FieldPtrPairs{T}, tag::Identifier) where {T}
    coreDict = buildDict(paramPairs, EmptyFieldPtrDict{T})
    MixedFieldParamPointer(coreDict, tag)
end


# `f` should only take one input.
unpackTypedFunc!(f::Function, paramSet::AbstractVector, 
                 paramSetId::Identifier=Identifier(paramSet)) = 
unpackTypedFunc!(ReturnTyped(f, Any), paramSet, paramSetId)


function unpackTypedFuncCore!(f::ReturnTyped{T}, paramSet::AbstractVector) where {T}
    params, anchors = getFieldParams(f)
    if isempty(params)
        ParamSelectFunc(f), paramSet, FieldPtrPair{T}[]
    else
        ids = locateParam!(paramSet, params)
        fDummy = deepcopy(f.f)
        paramDoubles = getParams(fDummy)
        foreach(p->setScreenLevel!(p, 1), paramDoubles)
        evalCore = function (x, pVals::Vararg)
            for (p, v) in zip(paramDoubles, pVals)
                setVal!(p, v)
            end
            fDummy(x)
        end
        paramPairs = getMemory(ChainPointer(:apply, anchors) .=> ids)
        ParamSelectFunc(ReturnTyped(evalCore, T), ids), paramSet, paramPairs
    end
end

function unpackTypedFunc!(f::ReturnTyped{T}, paramSet::AbstractVector, 
                          paramSetId::Identifier=Identifier(paramSet)) where {T}
    fCore, _, paramPairs = unpackTypedFuncCore!(f, paramSet)
    ptrDict = buildDict(paramPairs, EmptyFieldPtrDict{T})
    paramPtr = MixedFieldParamPointer(ptrDict, paramSetId)
    fCore, paramSet, paramPtr
end