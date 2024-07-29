export TensorVar, CellParam, GridParam, ParamGrid, ParamMesh, setScreenLevel!, 
       setScreenLevel, symOf, inputOf, obtain, fragment, setVal!, screenLevelOf, 
       markParams!, topoSort, getParams, ShapedMemory, directObtain

using Base: Fix2, Threads.Atomic, issingletontype
using Test: @inferred


function checkReshapingAxis(arr::AbstractArray, shape::Tuple{Vararg{Int}})
    isempty(arr) && throw(AssertionError("the target array should not be empty."))
    if any(i <= 0 for i in shape)
        throw(AssertionError("The reshaping axis size should all be positive."))
    end
    len = length(arr)
    if prod(shape) != len
        throw(AssertionError("The product of reshaping axes should be equal to the "*
                             "target array's length."))
    end
    len
end


struct ShapedMemory{T, N} <: AbstractMemory{T, N}
    value::Memory{T}
    shape::NTuple{N, Int}

    function ShapedMemory(value::Memory{T}, shape::Tuple{Vararg{Int}}) where {T}
        checkReshapingAxis(value, shape)
        new{T, length(shape)}(copy(value), shape)
    end
end

ShapedMemory(value::AbstractArray{T}, shape::Tuple{Vararg{Int}}=size(value)) where {T} = 
ShapedMemory(Memory{T}( vec(value) ), shape)

ShapedMemory(::Type{T}, value::AbstractArray{T}) where {T} = ShapedMemory(value)

ShapedMemory(::Type{T}, value::T) where {T} = ShapedMemory( fill(value) )

ShapedMemory(sm::ShapedMemory) = ShapedMemory(sm.value, sm.shape)

import Base: size, getindex, setindex!, iterate, length
size(arr::ShapedMemory) = arr.shape

getindex(arr::ShapedMemory, i::Int) = getindex(arr.value, i)
getindex(arr::ShapedMemory{<:Any, N}, i::Vararg{Int, N}) where {N} = 
getindex(reshape(arr.value, arr.shape), i...)

setindex!(arr::ShapedMemory, val, i::Int) = setindex!(arr.value, val, i)
setindex!(arr::ShapedMemory{<:Any, N}, val, i::Vararg{Int, N}) where {N} = 
setindex!(reshape(arr.value, arr.shape), val, i...)

iterate(arr::ShapedMemory) = iterate(arr.value)
iterate(arr::ShapedMemory, state) = iterate(arr.value, state)

length(arr::ShapedMemory) = length(arr.value)


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


exclude0DimData(::T) where {T} = exclude0DimData(T)
exclude0DimData(::Type) = nothing
exclude0DimData(::Type{<:AbtMemory0D}) = 
throw(ArgumentError("`$(AbstractMemory{<:Any, 0})` is not allowed as the argument."))
exclude0DimData(::Type{<:AbtArray0D}) = 
throw(ArgumentError("`$(AbstractArray{<:Any, 0})` is not allowed as the argument."))
exclude0DimParam(::Type{<:ElementalParam}) = 
throw(ArgumentError("`$(ElementalParam)` is not allowed as the argument."))

excludeAbtArray(::T) where {T} = excludeAbtArray(T)
excludeAbtArray(::Type) = nothing
excludeAbtArray(::Type{<:AbstractArray}) = 
throw(ArgumentError("`AbstractArray` is not allowed as an input argument."))

function checkAbtArrayArg(::Type{V}) where {T, N, V<:AbstractArray{T, N}}
    exclude0DimData(V)
    (T, N)
end
checkAbtArrayArg(::Type{T}) where {T} = (T, 0)
checkAbtArrayArg(::T) where {T} = checkAbtArrayArg(T)


function checkReturnType(f::F, ::Type{T}, args::NonEmptyTuple{Any}) where {F, T}
    @inferred T f(args...)
    f(args...)
end


struct TypedReduction{T, F<:Function} <: ParamOperator{T, F, 0, 0}
    f::F

    function TypedReduction(f::F, arg, args...) where {F}
        allArgs = (arg, args...)
        T = Union{(allArgs.|>checkAbtArrayArg.|>first.|>checkAbtArrayArg.|>first)...}
        excludeAbtArray(T)
        checkReturnType(f, T, allArgs)
        new{T, F}(f)
    end

    function TypedReduction(::Type{T}) where {T}
        excludeAbtArray(T)
        new{T, iT}(itself)
    end
end

TypedReduction(trf::TypedReduction, arg, args...) = TypedReduction(trf.f, arg, args...)

function (sf::TypedReduction{T, F})(arg::AbtArr210L{T}, args::AbtArr210L{T}...) where {T, F}
    sf.f(arg, args...)::T
end

(::TypedReduction{T, iT})(arg::T) where {T} = itself(arg)


struct StableMorphism{T, F<:Function, N} <: ParamOperator{T, F, N, 0}
    f::F

    function StableMorphism(f::F, arg, args...) where {F}
        allArgs = (arg, args...)
        T = Union{(allArgs.|>checkAbtArrayArg.|>first.|>checkAbtArrayArg.|>first)...}
        excludeAbtArray(T)
        val = checkReturnType(f, AbstractArray{T}, allArgs)
        new{T, F, ndims(val)}(f)
    end

    function StableMorphism(::Type{V}) where {N, V<:AbstractArray{<:Any, N}}
        T = checkAbtArrayArg(V) |> first
        excludeAbtArray(T)
        new{T, iT, N}(itself)
    end
end

StableMorphism(srf::StableMorphism, arg, args...) = StableMorphism(srf.f, arg, args...)

function (sf::StableMorphism{T, F, N})(arg::AbtArr210L{T}, args::AbtArr210L{T}...) where 
                                      {T, F, N}
    sf.f(arg, args...)::AbstractArray{T, N}
end

(::StableMorphism{T, iT, N})(arg::AbstractArray{T, N}) where {T, N} = itself(arg)


function genSeqAxis(lens::NTuple{N, Int}, sym::Symbol=:e) where {N}
    Tuple( (Symbol(sym, i), lens[i]) for i in 1:N )
end

FixedShapeLinkAxisType = Union{NonEmptyTuple{Tuple{Symbol, Int}}, Missing}

struct FixedShapeLink{T, F<:Function, N, O} <: ParamOperator{T, F, N, O}
    f::F
    axis::NTuple{O, Tuple{Symbol, Int}}
    extent::Int

    function FixedShapeLink(f::F, ::Type{V}, arg, args...; 
                            axis::FixedShapeLinkAxisType=missing) where 
                           {F<:Function, V<:AbstractArray}
        T, N = checkAbtArrayArg( eltype(V) )
        excludeAbtArray(T)
        allArgs = (arg, args...)
        allArgs .|> checkAbtArrayArg .|> first .|> checkAbtArrayArg
        val = checkReturnType(f, AbstractArray{<:T}, allArgs)
        if ismissing(axis)
            axis = genSeqAxis(val|>size)
        else
            checkReshapingAxis(val, last.(axis))
        end
        O = length(axis)
        exclude0DimData(AbstractArray{T, O})
        new{T, F, N, O}(f, axis, prod( last.(axis) ))
    end

    function FixedShapeLink(arg::V; axis::FixedShapeLinkAxisType=missing) where 
                           {V<:AbstractArray}
        T1, O = checkAbtArrayArg(V)
        T2, N = checkAbtArrayArg(T1)
        excludeAbtArray(T2)
        if ismissing(axis)
            axis = genSeqAxis(arg|>size)
        else
            checkReshapingAxis(arg, last.(axis))
        end
        new{T2, iT, N, O}(itself, axis, prod( last.(axis) ))
    end
end

FixedShapeLink(fslf::FixedShapeLink, arg, args...) = FixedShapeLink(fslf.f, arg, args...)

callFixedShapeLinkCore(ml::FixedShapeLink{T, <:Any, N}, 
                       arg::AbtArr210L{T}, args::AbtArr210L{T}...) where {T, N} = 
ml.f(arg, args...)::AbstractArray{<:AbstractArray{T, N}}

callFixedShapeLinkCore(ml::FixedShapeLink{T, <:Any, 0}, 
                       arg::AbtArr210L{T}, args::AbtArr210L{T}...) where {T} = 
ml.f(arg, args...)::AbstractArray{<:T}

function (ml::FixedShapeLink{T})(arg::AbtArr210L{T}, args::AbtArr210L{T}...) where {T}
    res = callFixedShapeLinkCore(ml, arg, args...)
    iBegin = firstindex(res)
    reshape(res[iBegin : (iBegin + ml.extent - 1)], last.(ml.axis))
end

(::FixedShapeLink{T, iT, 0, O})(arg::AbstractArray{T, O}) where {T, O} = itself(arg)

(::FixedShapeLink{T, iT, N, O})(arg::BiAbtArray{T, N, O}) where {T, N, O} = itself(arg)


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
                            {T, N, O, P<:DoubleDimParam{T, N, O}}
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

function checkCellParamArg(::TypedReduction{T, <:iTalike}, input::I, shifter::S, 
                           memory::Union{ShapedMemory{T, 0}, T, Missing}) where {T, I, S}
    checkParamContainerArgType1(I, Tuple{ElementalParam{T}})
    checkParamInput(input, innerDimMax=0, outerDimMax=0)
    if ismissing(memory)
        memory = ShapedMemory( fill(input[1]|>obtain|>shifter) )
    elseif memory isa T
        memory = ShapedMemory( fill(memory) )
    end
    TypedReduction(T), iT, deepcopy(memory)
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

mutable struct CellParam{T, F<:Function, I<:ParamInputType{T}} <: ParamToken{T, 0, I}
    const lambda::TypedReduction{T, F}
    const input::I
    const symbol::IndexedSym
    const memory::ShapedMemory{T, 0}
    @atomic screen::TernaryNumber
    @atomic offset::T

    function CellParam(lambda::TypedReduction{T, F}, input::I, 
                       symbol::SymOrIndexedSym, 
                       memory::Union{ShapedMemory{T, 0}, T, Missing}=missing, 
                       screen::Union{TernaryNumber, Int}=TUS0, 
                       offset::Union{T, Nothing}=initializeOffset(T)) where 
                      {T, F, I<:ParamInputType{T}}
        levels = getScreenLevelOptions(ParamToken{T, 0})
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
    lambda = TypedReduction(func, obtain.(input)...)
    CellParam(lambda, input, symbol, init, TUS0, initializeOffset(T))
end

CellParam(func::Function, input::DoubleDimParam{T}, symbol::SymOrIndexedSym; 
          init::Union{ShapedMemory{T, 0}, T, Missing}=missing) where {T} = 
CellParam(func, (input,), symbol; init)

CellParam(func::Function, input1::DoubleDimParam{T}, input2::DoubleDimParam{T}, 
          symbol::SymOrIndexedSym; 
          init::Union{ShapedMemory{T, 0}, T, Missing}=missing) where {T} = 
CellParam(func, (input1, input2), symbol; init)

CellParam(func::Function, input1::DoubleDimParam{T}, input2::DoubleDimParam{T}, 
          input3::DoubleDimParam{T}, symbol::SymOrIndexedSym; 
          init::Union{ShapedMemory{T, 0}, T, Missing}=missing) where {T} = 
CellParam(func, (input1, input2, input3), symbol; init)

function CellParam(par::CellParam{T}, symbol::SymOrIndexedSym=symOf(par); 
                   init::Union{ShapedMemory{T, 0}, T, Missing}=par.memory) where {T}
    offset = isOffsetEnabled(par) ? par.offset : nothing
    CellParam(par.lambda, par.input, symbol, init, par.screen, offset)
end

CellParam(input::PrimitiveParam{T, 0}, symbol::SymOrIndexedSym=symOf(input)) where {T} = 
CellParam(TypedReduction(T), (input,), symbol)

CellParam(var, varSym::SymOrIndexedSym, symbol::SymOrIndexedSym=varSym) = 
CellParam(TensorVar(var, varSym), symbol)


function checkGridParamArg(::StableMorphism{T, <:iTalike, N}, input::I, memory::M) where 
                          {T, N, O, I<:DoubleDimParam{T, <:Any, O}, M<:AbstractArray{T, N}}
    checkParamContainerArgType1(I, Tuple{SingleDimParam{T, N}})
    MI = typeof(input[1])
    if !(M <: MI)
        throw(AssertionError("The type of `memory` should be a subtype of `$M1`."))
    end
    checkParamInput(input, innerDimMax=(O==0)*N, outerDimMax=(O==N)*N)
    StableMorphism(AbstractArray{T, N}), iT, deepcopy(memory|>ShapedMemory)
end

function checkGridParamArg(::StableMorphism{T, <:iTalike, N}, input::I, ::Missing) where 
                          {T, N, O, I<:DoubleDimParam{T, <:Any, O}}
    checkParamContainerArgType1(I, Tuple{SingleDimParam{T, N}})
    checkParamInput(input, innerDimMax=(O==0)*N, outerDimMax=(O==N)*N)
    StableMorphism(AbstractArray{T, N}), iT, deepcopy(input[1]|>obtain|>ShapedMemory)
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

mutable struct GridParam{T, F<:Function, I<:ParamInputType{T}, N} <: ParamToken{T, N, I}
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
    lambda = StableMorphism(func, obtain.(input)...)
    GridParam(lambda, input, symbol, init)
end

GridParam(func::Function, input::DoubleDimParam{T}, symbol::SymOrIndexedSym; 
          init::Union{AbstractArray{T}, Missing}=missing) where {T} = 
GridParam(func, (input,), symbol; init)

GridParam(func::Function, input1::DoubleDimParam{T}, input2::DoubleDimParam{T}, 
          symbol::SymOrIndexedSym; 
          init::Union{AbstractArray{T}, Missing}=missing) where {T} = 
GridParam(func, (input1, input2), symbol; init)

GridParam(func::Function, input1::DoubleDimParam{T}, input2::DoubleDimParam{T}, 
          input3::DoubleDimParam{T}, symbol::SymOrIndexedSym; 
          init::Union{AbstractArray{T}, Missing}=missing) where {T} = 
GridParam(func, (input1, input2, input3), symbol; init)

GridParam(par::GridParam{T}, symbol::SymOrIndexedSym=symOf(par); 
          init::Union{AbstractArray{T}, Missing}=par.memory) where {T} = 
GridParam(par.lambda, par.input, symbol, init)

GridParam(input::SingleDimParam{T, N}, symbol::SymOrIndexedSym=symOf(input)) where {T, N} = 
GridParam(StableMorphism(AbstractArray{T, N}), (input,), symbol)

GridParam(val::AbstractArray, valSym::SymOrIndexedSym, symbol::SymOrIndexedSym=valSym) = 
GridParam(TensorVar(val, valSym), symbol)


struct ParamGrid{T, N, I<:SingleDimParam{T, N}, O} <: ParamNest{T, N, I, O}
    input::ShapedMemory{I, O}
    symbol::IndexedSym

    function ParamGrid(input::ShapedMemory{I, O}, symbol::SymOrIndexedSym) where 
                      {T, N, I<:SingleDimParam{T, N}, O}
        exclude0DimData(input)
        isempty(input.value) && throw(AssertionError("`input` should not be empty."))
        new{T, N, I, O}(input, IndexedSym(symbol))
    end
end

function ParamGrid(input::AbstractArray{I, O}, symbol::SymOrIndexedSym) where 
                  {T, N, I<:SingleDimParam{T, N}, O}
    exclude0DimData(input)
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

function checkParamMeshArg(ml::FixedShapeLink{T, F, N}, input::I, 
                           memory::Union{Memory{ShapedMemory{T, N}}, Missing}) where 
                          {T, F, N, I}
    if ismissing(memory)
        memory = Memory{ShapedMemory{T, N}}(ShapedMemory.(T, ml.f(obtain.(input)...))|>vec)
    else
        checkParamContainerArgType2(length(memory), ml.extent)
    end
    checkParamInput(input)
    ml, F, deepcopy(memory)
end

struct ParamMesh{T, F<:Function, I<:ParamInputType{T}, N, O} <: ParamLink{T, N, I, O}
    lambda::FixedShapeLink{T, F, N, O}
    input::I
    symbol::IndexedSym
    memory::Memory{ShapedMemory{T, N}}

    function ParamMesh(lambda::FixedShapeLink{T, F, N, O}, input::I, 
                       symbol::SymOrIndexedSym, 
                       memory::Union{Memory{ShapedMemory{T, N}}, Missing}=missing) where 
                      {T, N, F, O, I<:ParamInputType{T}}
        lambda, funcType, memory = checkParamMeshArg(lambda, input, memory)
        new{T, funcType, I, N, O}(lambda, input, IndexedSym(symbol), memory)
    end
end

function ParamMesh(func::Function, input::ParamInputType, symbol::SymOrIndexedSym; 
                   axis::FixedShapeLinkAxisType=missing)
    inputVal = obtain.(input)
    out = func(inputVal...)
    out isa AbstractArray || throw(AssertionError("`func` should output an AbstractArray."))
    lambda = FixedShapeLink(func, typeof(out), inputVal...; axis)
    ParamMesh(lambda, input, symbol)
end

ParamMesh(func::Function, input::DoubleDimParam{T}, symbol::SymOrIndexedSym; 
          axis::FixedShapeLinkAxisType=missing) where {T} = 
ParamMesh(func, (input,), symbol; axis)

ParamMesh(func::Function, input1::DoubleDimParam{T}, input2::DoubleDimParam{T}, 
          symbol::SymOrIndexedSym; 
          axis::FixedShapeLinkAxisType=missing) where {T} = 
ParamMesh(func, (input1, input2), symbol; axis)

ParamMesh(func::Function, input1::DoubleDimParam{T}, input2::DoubleDimParam{T}, 
          input3::DoubleDimParam{T}, symbol::SymOrIndexedSym; 
          axis::FixedShapeLinkAxisType=missing) where {T} = 
ParamMesh(func, (input1, input2, input3), symbol; axis)

ParamMesh(input::DoubleDimParam, symbol::SymOrIndexedSym; 
          axis::FixedShapeLinkAxisType=missing) = 
ParamMesh(FixedShapeLink(obtain(input); axis), (input,), symbol)


genDefaultRefParSym(input::DoubleDimParam) = IndexedSym(:_, input.symbol)

struct NodeParam{T, F, I, N, O} <: ViewParam{T, N, I}
    input::Tuple{ParamMesh{T, F, I, N, O}}
    index::Int
    symbol::IndexedSym

    function NodeParam(input::Tuple{ParamMesh{T, F, I, N, O}}, index::Int, 
                       symbol::SymOrIndexedSym=genDefaultRefParSym(input|>first)) where 
                      {T, F, I, N, O}
        maxIndex = length(input.memory)
        if index < 1 || index > maxIndex
            throw(DomainError(index, "`index is out of the allowed range: (1, $maxIndex)`"))
        end
        new{T, F, I, N, O}(input, index, symbol)
    end
end

NodeParam(par::ParamMesh{T, F, I, N, O}, index::Int, 
          symbol::SymOrIndexedSym=genDefaultRefParSym(par)) where {T, F, I, N, O} = 
NodeParam(par, index, symbol)


function fragment(pn::ParamMesh{T, F, I, N, O}) where {T, F, I, N, O}
    res = Array{NodeParam{T, F, I, N, O}}(undef, last.(pn.lambda.axis))
    for i in eachindex(res)
        res[i] = NodeParam(pn, i)
    end
    res
end

fragment(pl::ParamGrid) = directObtain(pl.input)


getScreenLevelOptions(::Type{<:NodeParam}) = (0,)

getScreenLevelOptions(::Type{<:ParamToken}) = (0,)

getScreenLevelOptions(::Type{<:ParamToken{T, 0}}) where {T} = 
Tuple(0:(checkTypedOpMethods(T) * 2))

getScreenLevelOptions(::Type{<:ParamGrid}) = (0, 2)

getScreenLevelOptions(::Type{<:ParamBatch}) = (0,)

getScreenLevelOptions(::Type{<:PrimitiveParam}) = (1, 2)

getScreenLevelOptions(::Type{<:DoubleDimParam}) = (0, 1, 2)

getScreenLevelOptions(::T) where {T<:DoubleDimParam} = getScreenLevelOptions(T)

function isScreenLevelChangeable(::Type{T}) where {T<:DoubleDimParam}
    minLevel, maxLevel = extrema( getScreenLevelOptions(T) )
    (maxLevel - minLevel) > 0
end

isOffsetEnabled(::DoubleDimParam) = false

function isOffsetEnabled(pb::T) where {T<:CellParam}
    isScreenLevelChangeable(T) && maximum( getScreenLevelOptions(T) ) > 0 && 
    isdefined(pb, :offset) # Only for safety
end

screenLevelOf(::ParamToken) = 0

screenLevelOf(p::ParamToken{<:Any, 0}) = Int(p.screen)

screenLevelOf(::NodeParam) = 0

screenLevelOf(::ParamBatch) = 0

screenLevelOf(p::PrimitiveParam) = Int(p.screen)

screenLevelOf(p::ParamGrid) = ifelse(all(l==2 for l in screenLevelOf.(p.input.value)), 2, 0)


function setScreenLevelCore!(p::DoubleDimParam, level::Int)
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

setScreenLevel(p::CellParam, level::Int) = setScreenLevel!(CellParam(p), level)

setScreenLevel(p::TensorVar, level::Int) = TensorVar(p.input, p.symbol, TernaryNumber(level))


function memorizeCore!(p::ParamToken{T, N}, newMem::AbstractArray{T, N}) where {T, N}
    safelySetVal!(p.memory.value, newMem)
end

function memorizeCore!(p::ViewParam{T, N}, newMem::AbstractArray{T, N}) where {T, N}
    safelySetVal!(p.input.memory[p.index].value, newMem)
end

function memorize!(p::ParamToken{T, N}, newMem::AbstractArray{T, N}) where {T, N}
    oldMem = directObtain(p.memory)
    if p.memory.shape == size(newMem)
        safelySetVal!(p.memory.value, newMem)
    else
        @atomic p.memory = ShapedMemory(newMem)
    end
    oldMem
end

function memorize!(p::ParamToken{T, 0}, newMem::AbstractArray{T, 0}) where {T}
    oldMem = directObtain(p.memory)
    safelySetVal!(p.memory.value, newMem)
    oldMem
end

memorize!(p::ParamToken{T}, newMem::T) where {T} = memorize!(p, fill(newMem))

memorize!(p::ParamToken) = memorize!(p, obtain(p))


indexedSymOf(p::DoubleDimParam) = p.symbol

symOf(p::DoubleDimParam) = indexedSymOf(p).name

inputOf(p::DoubleDimParam) = p.input


mutable struct NodeMarker{T} <: StorageMarker{T}
    visited::Bool
    data::T

    NodeMarker(init::T, ::Type{U}=T) where {T, U} = new{U}(false, init)
end

const ParamDict0D{T, V} = IdDict{ElementalParam{T}, NodeMarker{V}}
const ParamDictSD{T, V} = IdDict{SingleDimParam{T}, NodeMarker{V}}
const ParamDictDD{T, V} = IdDict{DoubleDimParam{T}, NodeMarker{V}}

const ParamInputIdMarker{T} = NodeMarker{<:Memory{<:DoubleDimParam{T}}}
const ParamDictId{T} = IdDict{DoubleDimParam{T}, ParamInputIdMarker{T}}

struct ParamPointerDict{T, V0, V1, V2} <: MixedContainer{T}
    d0::ParamDict0D{T, V0}
    d1::ParamDictSD{T, V1}
    d2::ParamDictDD{T, V2}
    id::ParamDictId{T}

    ParamPointerDict(::Type{T}, ::Type{V0}, ::Type{V1}, ::Type{V2}) where {T, V0, V1, V2} = 
    new{T, V0, V1, V2}( ParamDict0D{T, V0}(), ParamDictSD{T, V1}(), ParamDictDD{T, V2}(), 
                        ParamDictId{T}() )
end

selectParamPointer(d::ParamPointerDict{T}, ::ElementalParam{T}) where {T} = d.d0
selectParamPointer(d::ParamPointerDict{T}, ::SingleDimParam{T}) where {T} = d.d1
selectParamPointer(d::ParamPointerDict{T}, ::DoubleDimParam{T}) where {T} = d.d2
selectParamPointer(d::ParamPointerDict{T}, ::ParamPointer{T}) where {T} = d.id

getParamDataTypeUB(::ParamPointerDict{T, V0, <:Any, <:Any}, 
                   ::ElementalParam{T}) where {T, V0} = V0
getParamDataTypeUB(::ParamPointerDict{T, <:Any, V1, <:Any}, 
                   ::SingleDimParam{T}) where {T, V1} = V1
getParamDataTypeUB(::ParamPointerDict{T, <:Any, <:Any, V2}, 
                   ::DoubleDimParam{T}) where {T, V2} = V2

function getDataCore1(counter::Int, d::ParamPointerDict{T}, p::DoubleDimParam{T}, 
                      failFlag, maxIter::Int) where {T}
    counter += 1
    if counter > maxIter
        throw( ErrorException("The iterative calling times passed the limit: $maxIter.") )
    end
    res = get(selectParamPointer(d, p), p, failFlag)
    if res !== failFlag && res isa ParamPointer
        getDataCore2(counter, d, res, failFlag, maxIter)
    else
        res.data
    end
end

function getDataCore2(counter::Int, d::ParamPointerDict{T, V0, V1, V2}, p::ParamPointer{T}, 
                      failFlag, maxIter::Int) where {T, V0, V1, V2}
    input = p.input
    container = Memory{Union{V0, V1, V2}}(undef, length(input))
    flag = true
    for i in eachindex(input)
        res = getDataCore1(counter, d, input[i], failFlag, maxIter)
        if res === failFlag
            flag = false
            break
        else
            container[i] = res
        end
    end
    flag ? container : failFlag
end

function getData(d::ParamPointerDict{T}, p::DoubleDimParam{T}, 
                 default; failFlag=nothing, maxIter::Int=10) where {T}
    res = getDataCore1(0, d, p, failFlag, maxIter)
    res===failFlag ? default : res
end

function getParamMarker!(pDict::ParamPointerDict{T}, transformer::F, 
                         p::DoubleDimParam{T}) where {T, F}
    mem = transformer(p)
    tUB = getParamDataTypeUB(pDict, p)
    markerType = NodeMarker{tUB}
    get!(selectParamPointer(pDict, p), p, NodeMarker(mem, tUB))::markerType
end

function getParamMarker!(pDict::ParamPointerDict{T}, transformer::F, 
                         p::ParamPointer{T}) where {T, F}
    input = p.input
    eleT = eltype(input)
    parId = Memory{eleT}(undef, length(input))
    for (i, par) in enumerate(input)
        recursiveTransform!(transformer, pDict, par)
        parId[i] = par
    end
    markerType = NodeMarker{Memory{eleT}}
    get!(selectParamPointer(pDict, p), p, NodeMarker( Memory{eleT}(parId) ))::markerType
end

function recursiveTransformCore1!(generator::F, marker::NodeMarker, 
                                  p::DoubleDimParam) where {F<:Function}
    # Depth-first search by recursive calling
    if !marker.visited
        marker.visited = true
        marker.data = generator(p)
    else
        marker.data
    end
end

function recursiveTransformCore2!(generator::F, marker::NodeMarker, p::ParamPointer{T}, 
                                  pDict::ParamPointerDict{T}) where {F<:Function, T}
    val = map(marker.data) do par
        res = getData(pDict, par, nothing)
        res === nothing && throw(ErrorException("Could note locate the value for $par."))
        res
    end
    generator(val, p)
end

function recursiveTransform!(transformer::F, pDict::ParamPointerDict{T}, 
                             p::PrimitiveParam{T, N}) where {F, T, N}
    marker = getParamMarker!(pDict, transformer, p)
    recursiveTransformCore1!(transformer, marker, p)
    marker.data
end

function recursiveTransform!(transformer::F, pDict::ParamPointerDict{T}, 
                             p::DoubleDimParam{T}) where {F, T}
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

function recursiveTransform!(transformer::F, pDict::ParamPointerDict{T}, 
                             p::ParamPointer{T}) where {F, T}
    marker = getParamMarker!(pDict, transformer, p)
    recursiveTransformCore2!(transformer, marker, p, pDict)
end

const ParamValDict{T} = ParamPointerDict{T, T, AbstractArray{T}, BiAbtArray{T}}

genParamValDict(::Type{T}) where {T} = 
ParamPointerDict(T, T, AbstractArray{T}, BiAbtArray{T})::ParamValDict{T}


obtainCore(p::ParamFunctor) = directObtain(p.memory)
obtainCore(p::PrimitiveParam) = directObtain(p.input)

function obtainCore(inputVal::NTuple{A, AbtArr210L{T}}, 
                    p::ParamToken{T, <:Any, <:ParamInput{T, A}}) where {T, A}
    p.lambda(inputVal...)
end

function obtainCore(inputVal::NTuple{A, AbtArr210L{T}}, 
                    p::ParamMesh{T, <:Any, <:ParamInput{T, A}}) where {T, A}
    f = p.lambda
    valRaw = f.f( inputVal... )
    Memory{eltype(valRaw)}(valRaw[begin:(begin + f.extent - 1)])
end

function obtainCore(inputVal::NTuple{A, AbtArr210L{T}}, 
                    p::CellParam{T, <:Any, <:ParamInput{T, A}}) where {T, A}
    screenLevelOf(p)==0 || throw(AssertionError("The screen level of `p` should be 0."))
    shiftVal = genValShifter(T, (isOffsetEnabled(p) ? p.offset : nothing))
    p.lambda(inputVal...) |> shiftVal
end

obtainCore(::Type{T}, p::CellParam{T}) where {T} = p.offset

obtainCore(val::Memory{T}, p::ParamGrid{T, 0}) where {T} = 
reshape(val, p.input.shape)

obtainCore(val::Memory{<:AbstractArray{T, N}}, p::ParamGrid{T, N}) where {T, N} = 
reshape(val, p.input.shape)

obtainCore(val::Memory{<:AbstractArray{T}}, p::NodeParam{T, <:Any, <:Any, 0}) where {T} = 
getindex(getindex(val), p.idx)

obtainCore(val::Memory{<:BiAbtArray{T, N}}, p::NodeParam{T, <:Any, <:Any, N}) where {T, N} = 
getindex(getindex(val), p.idx)

function searchObtain(pDict::ParamValDict{T}, p::ParamMesh{T}) where {T}
    res = recursiveTransform!(obtainCore, pDict, p)
    reshape(res, last.(p.lambda.axis)) |> collect
end

searchObtain(pDict::ParamValDict{T}, p::CompositeParam{T}) where {T} = 
recursiveTransform!(obtainCore, pDict, p)


obtainINTERNAL(p::PrimitiveParam) = obtainCore(p)

function obtainINTERNAL(p::CompositeParam{T}) where {T}
    searchObtain(genParamValDict(T), p)
end

function obtainINTERNAL(ps::AbstractArray{<:ElementalParam{T}}) where {T}
    map(obtainINTERNAL, ps)
end

function obtainINTERNAL(ps::AbstractArray{<:DoubleDimParam{T}}) where {T}
    pValDict = genParamValDict(T)
    map(p->searchObtain(pValDict, p), ps)
end

function obtain(p::AbtArrayOr{<:DoubleDimParam{T}}) where {T}
    lock( ReentrantLock() ) do
        obtainINTERNAL(p)
    end
end

################################

(pn::DoubleDimParam)() = obtain(pn)

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

isPrimitiveParam(pn::ParamToken) = (screenLevelOf(pn) == 1)

# import Base: iterate, size, length, eltype, broadcastable
# length(::FixedSizeParam{<:Any, N}) where {N} = N
# eltype(np::FixedSizeParam) = eltype(np.input)

# iterate(::FixedSizeParam{<:Any, 1}, args...) = iterate(1, args...)
# size(::FixedSizeParam{<:Any, 1}, args...) = size(1, args...)
# broadcastable(np::FixedSizeParam{<:Any, 1}) = Ref(np)

# iterate(np::FixedSizeParam, args...) = iterate(np.input, args...)
# size(np::FixedSizeParam, args...) = size(np.input, args...)
# broadcastable(np::FixedSizeParam) = Base.broadcastable(np.input)


struct ParamMarker{M<:NonEmptyTuple{IdentityMarker}, MF<:IdentityMarker, 
                   N} <: IdentityMarker{M}
    typeID::UInt
    marker::M
    funcID::MF
    metaID::NTuple{N, UInt}
end

struct ValueMarker <: IdentityMarker{UInt}
    valueID::UInt

    ValueMarker(input) = new(objectid(input))
end

struct CollectionMarker <: IdentityMarker{Union{AbstractArray, Tuple}}
    data::Union{AbstractArray, Tuple}
end

struct ObjectMarker{T} <: IdentityMarker{T}
    data::T
end

markObj(input::PrimitiveParam) = ValueMarker(input)

markObj(input::DoubleDimParam) = ParamMarker(input)

function isPrimVarCollection(arg::AbstractArray{T}) where {T}
    ET = isconcretetype(T) ? T : eltype( map(itself, arg) )
    isbitstype(ET)
end

function isPrimVarCollection(arg::Tuple)
    all(isbits(i) for i in arg)
end

function markObj(input::Union{AbstractArray, Tuple})
    isPrimVarCollection(input) ? ValueMarker(input) : CollectionMarker(input)
end

function markObj(input::T) where {T}
    if isstructtype(input) && !( Base.issingletontype(input) )
        markObj( (objectid(T), getproperty.(input, propertynames(input))...) )
    elseif input isa Function
        markObj( (objectid(input),) )
    else
        ObjectMarker(input)
    end
end

markObj(marker::IdentityMarker) = itself(marker)

markObj(f::ShapedMemory) = markObj( (markObj(f.value), objectid(f.shape)) )

const NothingID = markObj( objectid(nothing) )

function ParamMarker(p::T) where {T<:CellParam}
    offset = isOffsetEnabled(p) ? p.offset : objectid(nothing)
    sl = screenLevelOf(p)
    if sl > 0
        ParamMarker(
            objectid(T), markObj.((objectid(p), offset)), NothingID, (objectid(sl),)
        )
    else
        ParamMarker(
            objectid(T), markObj.((p.input..., offset)), markObj(p.lambda), (objectid(sl),)
        )
    end
end

function ParamMarker(p::T) where {T<:GridParam}
    ParamMarker(objectid(T), markObj.(p.input), markObj(p.lambda), ())
end

function ParamMarker(p::T) where {T<:NodeParam}
    ParamMarker(objectid(T), markObj.(p.input), NothingID, (objectid(p.index),))
end

function ParamMarker(p::T) where {T<:ParamGrid}
    ParamMarker(objectid(T), (markObj(p.input),), NothingID, ())
end

function ParamMarker(p::T) where {T<:ParamMesh}
    ParamMarker(objectid(T), markObj.(pn.input), markObj(p.lambda), ())
end

compareMarker(pm1::IdentityMarker, pm2::IdentityMarker) = false

compareMarker(pm1::ValueMarker, pm2::ValueMarker) = pm1 == pm2

compareMarker(pm1::T, pm2::T) where {T<:ParamMarker{<:Tuple{Vararg{ValueMarker}}}} = 
pm1 == pm2

function compareMarker(pm1::T, pm2::T) where {T<:CollectionMarker}
    if pm1.data === pm2.data
        true
    elseif length(pm1.data) == length(pm2.data)
        isSame = true
        for (i, j) in zip(pm1.data, pm2.data)
            isSame = ( i===j || compareMarker(markObj(i), markObj(j)) )
            isSame || break
        end
        isSame
    else
        false
    end
end

compareMarker(pm1::T, pm2::T) where {T<:ObjectMarker} = pm1.data == pm2.data

function compareMarker(pm1::T, pm2::T) where {T<:ParamMarker}
    isSame = (pm1.metaID == pm2.metaID && compareMarker(pm1.funcID, pm2.funcID))
    if isSame
        if pm1.marker === pm2.marker
        elseif length(pm1.marker) == length(pm2.marker)
            for (marker1, marker2) in zip(pm1.marker, pm2.marker)
                isSame = compareMarker(marker1, marker2)
                isSame || break
            end
        else
            isSame = false
        end
    end
    isSame
end

compareParamContainer(::DoubleDimParam, ::DoubleDimParam) = false

compareParamContainer(::DoubleDimParam, ::Any) = false

compareParamContainer(::Any, ::DoubleDimParam) = false

compareParamContainer(p1::T, p2::T) where {T<:PrimitiveParam} = p1 === p2

function compareParamContainer(p1::CompositeParam{T, N, O}, 
                               p2::CompositeParam{T, N, O}) where {T, N, O}
    p1 === p2 || compareMarker(ParamMarker(p1), ParamMarker(p2))
end


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

getParams(p::DoubleDimParam) = [p]

getParams(p::DoubleDimParam, ::Missing) = [p]

function getParams(p::DoubleDimParam, sym::Symbol)
    res = eltype(p)[]
    inSymbol(sym, symOf(ps)) && push!(res, p)
    res
end

function getParams(v::NonEmptyTupleOrAbtArray{T}, sym::SymOrMiss=missing) where 
                  {T<:Union{ParamContainer, ParamObject}}
    isempty(v) ? ParamContainer{T}[] : reduce(vcat, getParams.(v, sym))
end

function getParams(f::ParamFunction{T}, sym::SymOrMiss=missing) where {T}
    res = map(field->getParams(field, sym), getParamFields(f))
    len = length.(res) |> sum
    len == 0 ? ParamContainer{T}[] : reduce(vcat, res)
end


uniqueParams(ps::AbstractArray{<:DoubleDimParam}) = 
markUnique(ps, compareFunction=compareParamContainer)[end]


function markParamsCore!(indexDict::IdDict{Symbol, Int}, leafPars)
    for i in leafPars
        sym = i.symbol.name
        get!(indexDict, sym, 0)
        i.symbol.index = (indexDict[sym] += 1)
    end
end

function markParams!(pars::AbstractVector{<:DoubleDimParam{T}}) where {T}
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

function topoSortCore!(hbNodesIdSet::Set{UInt}, 
                       orderedNodes::Vector{<:DoubleDimParam{T}}, 
                       haveBranches::Vector{Bool}, connectRoots::Vector{Bool}, 
                       node::DoubleDimParam{T}, recursive::Bool=false) where {T}
    sl = checkScreenLevel(screenLevelOf(node), getScreenLevelOptions(node))

    if sl in (0, 1)
        idx = findfirst(Fix2(compareParamContainer, node), orderedNodes)
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

function topoSortINTERNAL(nodes::AbstractVector{<:DoubleDimParam{T}}) where {T}
    orderedNodes = DoubleDimParam{T}[]
    haveBranches = Bool[]
    connectRoots = Bool[]
    hbNodesIdSet = Set{UInt}()
    for node in nodes
        topoSortCore!(hbNodesIdSet, orderedNodes, haveBranches, connectRoots, node)
    end
    orderedNodes, haveBranches, connectRoots
end

function topoSort(nodes::AbstractVector{<:DoubleDimParam{T}}) where {T}
    uniqueParams(nodes) |> topoSortINTERNAL
end

topoSort(node::CellParam{T}) where {T} = topoSortINTERNAL([node])


# Sever the connection of a node to other nodes
sever(pv::TensorVar) = TensorVar(obtain(pv), pv.symbol)

# function sever(ps::T) where {T<:ParamToken}

#     T(sever.(ps.input), ps.symbol)
# end

sever(obj::Any) = deepcopy(obj)

sever(obj::Union{Tuple, AbstractArray}) = sever.(obj)

function sever(pf::T) where {T<:ParamFunction}
    severedFields = map(field->(sever∘getproperty)(pf, field), fieldnames(pf))
    T(severedFields...)
end