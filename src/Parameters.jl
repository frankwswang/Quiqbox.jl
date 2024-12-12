export TensorVar, CellParam, GridParam, ParamGrid, ParamMesh, setScreenLevel!, 
       setScreenLevel, symOf, inputOf, obtain, fragment, setVal!, screenLevelOf, 
       markParams!, topoSort, getParams, uniqueParams, ShapedMemory, directObtain, 
       memorize!, evalParamSet

using Base: Fix2, Threads.Atomic, issingletontype
using Test: @inferred


function checkReshapingAxis(arr::AbstractArray, shape::Tuple{Vararg{Int}})
    len = checkEmptiness(arr, :arr)
    if any(i <= 0 for i in shape)
        throw(AssertionError("The reshaping axis size should all be positive."))
    end
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
ShapedMemory(getMemory(value), shape)

ShapedMemory(::Type{T}, value::AbstractArray{T}) where {T} = ShapedMemory(value)

ShapedMemory(::Type{T}, value::T) where {T} = ShapedMemory( fill(value) )

ShapedMemory(arr::ShapedMemory) = ShapedMemory(arr.value, arr.shape)


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


struct TypedReduction{T, F<:Function} <: JaggedOperator{T, 0, 0}
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
        new{T, ItsType}(itself)
    end
end

TypedReduction(trf::TypedReduction, arg, args...) = TypedReduction(trf.f, arg, args...)

function (sf::TypedReduction{T, F})(arg::AbtArr210L{T}, args::AbtArr210L{T}...) where {T, F}
    sf.f(arg, args...)::T
end

(::TypedReduction{T, ItsType})(arg::T) where {T} = itself(arg)


struct StableMorphism{T, F<:Function, N} <: JaggedOperator{T, N, 0}
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
        new{T, ItsType, N}(itself)
    end
end

StableMorphism(srf::StableMorphism, arg, args...) = StableMorphism(srf.f, arg, args...)

function (sf::StableMorphism{T, F, N})(arg::AbtArr210L{T}, args::AbtArr210L{T}...) where 
                                      {T, F, N}
    sf.f(arg, args...)::AbstractArray{T, N}
end

(::StableMorphism{T, ItsType, N})(arg::AbstractArray{T, N}) where {T, N} = itself(arg)


function genSeqAxis(lens::NTuple{N, Int}, sym::Symbol=:e) where {N}
    Tuple( (Symbol(sym, i), lens[i]) for i in 1:N )
end

FixedShapeLinkAxisType = Union{NonEmptyTuple{Tuple{Symbol, Int}}, Missing}

struct FixedShapeLink{T, F<:Function, N, O} <: JaggedOperator{T, N, O}
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
        new{T2, ItsType, N, O}(itself, axis, prod( last.(axis) ))
    end
end

FixedShapeLink(fslf::FixedShapeLink, ::Type{V}, arg, args...) where {V<:AbstractArray} = 
FixedShapeLink(fslf.f, arg, args...)

callFixedShapeLinkCore(ml::FixedShapeLink{T, <:Any, N, O}, 
                       arg::AbtArr210L{T}, args::AbtArr210L{T}...) where {T, N, O} = 
ml.f(arg, args...)::AbstractArray{<:AbstractArray{T, N}, O}

callFixedShapeLinkCore(ml::FixedShapeLink{T, <:Any, 0, O}, 
                       arg::AbtArr210L{T}, args::AbtArr210L{T}...) where {T, O} = 
ml.f(arg, args...)::AbstractArray{<:T, O}

function (ml::FixedShapeLink{T})(arg::AbtArr210L{T}, args::AbtArr210L{T}...) where {T}
    res = callFixedShapeLinkCore(ml, arg, args...)
    iBegin = firstindex(res)
    reshape(res[iBegin : (iBegin + ml.extent - 1)], last.(ml.axis))
end

(::FixedShapeLink{T, ItsType, 0, O})(arg::AbstractArray{T, O}) where {T, O} = itself(arg)

(::FixedShapeLink{T, ItsType, N, O})(arg::JaggedAbtArray{T, N, O}) where {T, N, O} = itself(arg)


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
                            {T, N, O, P<:JaggedParam{T, N, O}}
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
    lambda = TypedReduction(func, obtain.(input)...)
    CellParam(lambda, input, symbol, init, TUS0, initializeOffset(T))
end

CellParam(func::Function, input::JaggedParam{T}, symbol::SymOrIndexedSym; 
          init::Union{ShapedMemory{T, 0}, T, Missing}=missing) where {T} = 
CellParam(func, (input,), symbol; init)

CellParam(func::Function, input1::JaggedParam{T}, input2::JaggedParam{T}, 
          symbol::SymOrIndexedSym; 
          init::Union{ShapedMemory{T, 0}, T, Missing}=missing) where {T} = 
CellParam(func, (input1, input2), symbol; init)

CellParam(func::Function, input1::JaggedParam{T}, input2::JaggedParam{T}, 
          input3::JaggedParam{T}, symbol::SymOrIndexedSym; 
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
                          {T, N, O, I<:JaggedParam{T, <:Any, O}, M<:AbstractArray{T, N}}
    checkParamContainerArgType1(I, Tuple{FlattenedParam{T, N}})
    MI = typeof(input[1])
    if !(M <: MI)
        throw(AssertionError("The type of `memory` should be a subtype of `$M1`."))
    end
    checkParamInput(input, innerDimMax=(O==0)*N, outerDimMax=(O==N)*N)
    StableMorphism(AbstractArray{T, N}), ItsType, deepcopy(memory|>ShapedMemory)
end

function checkGridParamArg(::StableMorphism{T, <:ItsTalike, N}, input::I, ::Missing) where 
                          {T, N, O, I<:JaggedParam{T, <:Any, O}}
    checkParamContainerArgType1(I, Tuple{FlattenedParam{T, N}})
    checkParamInput(input, innerDimMax=(O==0)*N, outerDimMax=(O==N)*N)
    StableMorphism(AbstractArray{T, N}), ItsType, deepcopy(input[1]|>obtain|>ShapedMemory)
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
    lambda = StableMorphism(func, obtain.(input)...)
    GridParam(lambda, input, symbol, init)
end

GridParam(func::Function, input::JaggedParam{T}, symbol::SymOrIndexedSym; 
          init::Union{AbstractArray{T}, Missing}=missing) where {T} = 
GridParam(func, (input,), symbol; init)

GridParam(func::Function, input1::JaggedParam{T}, input2::JaggedParam{T}, 
          symbol::SymOrIndexedSym; 
          init::Union{AbstractArray{T}, Missing}=missing) where {T} = 
GridParam(func, (input1, input2), symbol; init)

GridParam(func::Function, input1::JaggedParam{T}, input2::JaggedParam{T}, 
          input3::JaggedParam{T}, symbol::SymOrIndexedSym; 
          init::Union{AbstractArray{T}, Missing}=missing) where {T} = 
GridParam(func, (input1, input2, input3), symbol; init)

GridParam(par::GridParam{T}, symbol::SymOrIndexedSym=symOf(par); 
          init::Union{AbstractArray{T}, Missing}=par.memory) where {T} = 
GridParam(par.lambda, par.input, symbol, init)

GridParam(input::FlattenedParam{T, N}, symbol::SymOrIndexedSym=symOf(input)) where {T, N} = 
GridParam(StableMorphism(AbstractArray{T, N}), (input,), symbol)

GridParam(val::AbstractArray, valSym::SymOrIndexedSym, symbol::SymOrIndexedSym=valSym) = 
GridParam(TensorVar(val, valSym), symbol)


struct ParamGrid{T, N, I<:FlattenedParam{T, N}, O} <: ParamNest{T, N, I, O}
    input::ShapedMemory{I, O}
    symbol::IndexedSym

    function ParamGrid(input::ShapedMemory{I, O}, symbol::SymOrIndexedSym) where 
                      {T, N, I<:FlattenedParam{T, N}, O}
        exclude0DimData(input)
        checkEmptiness(input.value, :input)
        new{T, N, I, O}(input, IndexedSym(symbol))
    end
end

function ParamGrid(input::AbstractArray{I, O}, symbol::SymOrIndexedSym) where 
                  {T, N, I<:FlattenedParam{T, N}, O}
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

ParamMesh(func::Function, input::JaggedParam{T}, symbol::SymOrIndexedSym; 
          axis::FixedShapeLinkAxisType=missing) where {T} = 
ParamMesh(func, (input,), symbol; axis)

ParamMesh(func::Function, input1::JaggedParam{T}, input2::JaggedParam{T}, 
          symbol::SymOrIndexedSym; 
          axis::FixedShapeLinkAxisType=missing) where {T} = 
ParamMesh(func, (input1, input2), symbol; axis)

ParamMesh(func::Function, input1::JaggedParam{T}, input2::JaggedParam{T}, 
          input3::JaggedParam{T}, symbol::SymOrIndexedSym; 
          axis::FixedShapeLinkAxisType=missing) where {T} = 
ParamMesh(func, (input1, input2, input3), symbol; axis)

ParamMesh(input::JaggedParam, symbol::SymOrIndexedSym; 
          axis::FixedShapeLinkAxisType=missing) = 
ParamMesh(FixedShapeLink(obtain(input); axis), (input,), symbol)


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
    CellParam(ChainPointer(idx, TensorType(T)), pb, sym)
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

function indexParam(pb::JaggedParam{T, N}, idx::Int, 
                    sym::MissingOr{Symbol}=missing) where {T, N}
    ismissing(sym) && (sym = Symbol(:_, pb.symbol.name))
    type = TensorType(AbstractArray{T, N}, outputSizeOf(pb))
    GridParam(ChainPointer(idx, type), pb, sym)
end


genDefaultRefParSym(input::JaggedParam) = IndexedSym(:_, input.symbol)

struct KnotParam{T, F, I, N, O} <: LinkParam{T, N, I}
    input::Tuple{ParamMesh{T, F, I, N, O}}
    index::Int
    symbol::IndexedSym

    function KnotParam(input::Tuple{ParamMesh{T, F, I, N, O}}, index::Int, 
                       symbol::SymOrIndexedSym=genDefaultRefParSym(input|>first)) where 
                      {T, F, I, N, O}
        maxIndex = length(first(input).memory)
        if index < 1 || index > maxIndex
            throw(DomainError(index, "`index is out of the allowed range: (1, $maxIndex)`"))
        end
        new{T, F, I, N, O}(input, index, symbol)
    end
end

KnotParam(par::ParamMesh, index::Int, symbol::SymOrIndexedSym=genDefaultRefParSym(par)) = 
KnotParam((par,), index, symbol)


function fragment(pn::ParamMesh{T, F, I, N, O}) where {T, F, I, N, O}
    res = Array{KnotParam{T, F, I, N, O}}(undef, last.(pn.lambda.axis))
    for i in eachindex(res)
        res[i] = KnotParam(pn, i)
    end
    res
end

fragment(pl::ParamGrid) = directObtain(pl.input|>first)


getScreenLevelOptions(::Type{<:ParamGrid}) = (0, 2)

getScreenLevelOptions(::Type{<:LinkParam}) = (0,)

getScreenLevelOptions(::Type{<:BaseParam{T, 0}}) where {T} = 
Tuple(0:(checkTypedOpMethods(T) * 2))

getScreenLevelOptions(::Type{<:ParamToken}) = (0,)

getScreenLevelOptions(::Type{<:ParamBatch}) = (0,)

getScreenLevelOptions(::Type{<:PrimitiveParam}) = (1, 2)

getScreenLevelOptions(::Type{<:JaggedParam}) = (0, 1, 2)

getScreenLevelOptions(::T) where {T<:JaggedParam} = getScreenLevelOptions(T)

function isScreenLevelChangeable(::Type{T}) where {T<:JaggedParam}
    minLevel, maxLevel = extrema( getScreenLevelOptions(T) )
    (maxLevel - minLevel) > 0
end

isOffsetEnabled(::JaggedParam) = false

function isOffsetEnabled(pb::T) where {T<:CellParam}
    isScreenLevelChangeable(T) && maximum( getScreenLevelOptions(T) ) > 0 && 
    isdefined(pb, :offset) # Only for safety
end

screenLevelOf(p::BaseParam{<:Any, 0}) = Int(p.screen)

screenLevelOf(::KnotParam) = 0

screenLevelOf(::ParamToken) = 0

screenLevelOf(::ParamBatch) = 0

screenLevelOf(p::PrimitiveParam) = Int(p.screen)

screenLevelOf(p::ParamGrid) = ifelse(all(l==2 for l in screenLevelOf.(p.input.value)), 2, 0)


function setScreenLevelCore!(p::JaggedParam, level::Int)
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
    if length(p.memory) != length(memNew)
        throw(AssertionError("`memNew` should have the same length as `p`'s memory."))
    end
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


indexedSymOf(p::JaggedParam) = p.symbol

symOf(p::JaggedParam) = indexedSymOf(p).name

inputOf(p::JaggedParam) = p.input

isDependentParam(p::JaggedParam) = (screenLevelOf(p) < 1)

isPrimitiveParam(p::JaggedParam) = (screenLevelOf(p) == 1)

#? Maybe a more general type signature?
outputSizeOf(p::PrimitiveParam) = size(p.input)

outputSizeOf(p::ParamFunctor) = size(p.memory)

outputSizeOf(p::ParamGrid) = size(p.input)

outputSizeOf(p::KnotParam) = size(first(p.input).memory[p.index])



mutable struct NodeMarker{T} <: StorageMarker{T}
    visited::Bool
    data::T

    NodeMarker(init::T, ::Type{U}=T) where {T, U} = new{U}(false, init)
end

const ParamDict0D{T, V} = IdDict{ElementalParam{T}, NodeMarker{V}}
const ParamDictSD{T, V} = IdDict{FlattenedParam{T}, NodeMarker{V}}
const ParamDictDD{T, V} = IdDict{JaggedParam{T}, NodeMarker{V}}

const ParamInputSource{T} = Memory{<:JaggedParam{T}}
const ParamDictId{T} = IdDict{JaggedParam{T}, NodeMarker{<:ParamInputSource{T}}}

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
selectParamPointer(d::ParamPointerBox{T}, ::JaggedParam{T}) where {T} = d.d2
selectParamPointer(d::ParamPointerBox{T}, ::ParamPointer{T}) where {T} = d.id

getParamDataTypeUB(::ParamPointerBox{T, V0, <:Any, <:Any}, 
                   ::ElementalParam{T}) where {T, V0} = V0
getParamDataTypeUB(::ParamPointerBox{T, <:Any, V1, <:Any}, 
                   ::FlattenedParam{T}) where {T, V1} = V1
getParamDataTypeUB(::ParamPointerBox{T, <:Any, <:Any, V2}, 
                   ::JaggedParam{T}) where {T, V2} = V2

function checkGetDataRecNum(counter::Int, maxRec::Int)
    if counter > maxRec
        throw( ErrorException("The recursive calling times passed the limit: $maxRec.") )
    end
    nothing
end

function getDataCore1(counter::Int, d::ParamPointerBox{T}, p::JaggedParam{T}, 
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

function getData(d::ParamPointerBox{T}, p::JaggedParam{T}, default; 
                 failFlag=nothing) where {T}
    res = getDataCore1(0, d, p, failFlag)
    res===failFlag ? default : res
end

function getParamMarker!(pDict::ParamPointerBox{T}, transformer::F, 
                         p::JaggedParam{T}) where {T, F}
    mem = transformer(p)
    tUB = getParamDataTypeUB(pDict, p)
    markerType = NodeMarker{tUB}
    get!(selectParamPointer(pDict, p), p, NodeMarker(mem, tUB))::markerType
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
    get!(selectParamPointer(pDict, p), p, NodeMarker( Memory{eleT}(parId) ))::markerType
end

function recursiveTransformCore1!(generator::F, marker::NodeMarker, 
                                  p::JaggedParam) where {F<:Function}
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
                             p::JaggedParam{T}) where {F, T}
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

function obtainCore(val::Memory{T}, p::ParamGrid{T, 0}) where {T}
    reshape(val, p.input.shape)
end

obtainCore(val::Memory{<:AbstractArray{T, N}}, p::ParamGrid{T, N}) where {T, N} = 
reshape(val, p.input.shape)

obtainCore(val::Memory{<:AbstractArray{T}}, p::KnotParam{T, <:Any, <:Any, 0}) where {T} = 
getindex(getindex(val), p.index)

obtainCore(val::Memory{<:JaggedAbtArray{T, N}}, p::KnotParam{T, <:Any, <:Any, N}) where {T, N} = 
getindex(getindex(val), p.index)

const ParamValDict{T} = ParamPointerBox{ T, T, AbstractArray{T}, JaggedAbtArray{T}, 
                                          typeof(obtainCore) }

genParamValDict(::Type{T}, maxRecursion::Int=DefaultMaxParamPointerLevel) where {T} = 
ParamPointerBox(obtainCore, T, T, AbstractArray{T}, JaggedAbtArray{T}, maxRecursion)

function searchObtain(pDict::ParamValDict{T}, p::ParamMesh{T}) where {T}
    res = recursiveTransform!(obtainCore, pDict, p)
    reshape(res, last.(p.lambda.axis)) |> collect
end

searchObtain(pDict::ParamValDict{T}, p::JaggedParam{T}) where {T} = 
recursiveTransform!(obtainCore, pDict, p)

obtainINTERNALcore(p::JaggedParam{T}, maxRecursion::Int) where {T} = 
searchObtain(genParamValDict(T, maxRecursion), p)

obtainINTERNALcore(ps::AbstractArray{<:JaggedParam{T}}, maxRecursion::Int) where {T} = 
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

function obtainINTERNAL(ps::AbstractArray{<:JaggedParam{T}}, maxRecursion::Int) where {T}
    if any(isDependentParam, ps)
        pValDict = genParamValDict(T, maxRecursion)
        map(p->searchObtain(pValDict, p), ps)
    else
        obtainINTERNALcore(ps, maxRecursion)
    end
end

function obtainINTERNAL(p::AbtArrayOr{<:JaggedParam}, maxRecursion::Int)
    map(i->obtainINTERNAL(i, maxRecursion), p)
end

obtain(p::ParamBox; maxRecursion::Int=DefaultMaxParamPointerLevel) = 
obtainINTERNAL(p, maxRecursion)

obtain(p::ParamTypeArr{<:ParamBox{T}}; 
       maxRecursion::Int=DefaultMaxParamPointerLevel) where {T} = 
obtainINTERNAL(p, maxRecursion)

obtain(p::ParamTypeArr; maxRecursion::Int=DefaultMaxParamPointerLevel) = 
obtainINTERNAL(itself.(p), maxRecursion)

################################

(pn::JaggedParam)() = obtain(pn)

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


# struct CachedJParam{T, N, O, P<:JaggedParam{T, N, O}, 
#                     E<:Union{T, AbstractArray{T, N}}} <: QueryBox{P}
#     source::P
#     cache::ShapedMemory{E, O}

#     function CachedJParam(source::P) where {T, N, O, P<:JaggedParam{T, N, O}}
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

markObj(input::JaggedParam) = ParamMarker(input)

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
    if isstructtype(T) && !( Base.issingletontype(T) )
        markObj( (objectid(T), getproperty.(input, propertynames(input))...) )
    elseif input isa Function
        markObj( (objectid(input),) )
    else
        ObjectMarker(input)
    end
end

markObj(marker::IdentityMarker) = itself(marker)

markObj(arr::ShapedMemory) = markObj( (markObj(arr.value), objectid(arr.shape)) )

markObj(::Nothing) = markObj( objectid(nothing) )

# markObj(p::CachedJParam) = ParamMarker(p.param)

function ParamMarker(p::T) where {T<:CellParam}
    offset = isOffsetEnabled(p) ? p.offset : objectid(nothing)
    sl = screenLevelOf(p)
    if sl > 0
        ParamMarker(
            objectid(T), markObj.((objectid(p), offset)), markObj(nothing), (objectid(sl),)
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

function ParamMarker(p::T) where {T<:KnotParam}
    ParamMarker(objectid(T), markObj.(p.input), markObj(nothing), (objectid(p.index),))
end

function ParamMarker(p::T) where {T<:ParamGrid}
    ParamMarker(objectid(T), (markObj(p.input),), markObj(nothing), ())
end

function ParamMarker(p::T) where {T<:ParamMesh}
    ParamMarker(objectid(T), markObj.(p.input), markObj(p.lambda), ())
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

compareParamBox(p1::T, p2::T) where {T<:PrimitiveParam} = p1 === p2

function compareParamBox(p1::CompositeParam{T, N, O}, 
                         p2::CompositeParam{T, N, O}) where {T, N, O}
    p1 === p2 || compareMarker(ParamMarker(p1), ParamMarker(p2))
end

compareParamBox(::JaggedParam, ::JaggedParam) = false

function compareObj(obj1::T1, obj2::T2) where {T1, T2}
    if T1 <: JaggedParam && T2 <: JaggedParam
        compareParamBox(obj1, obj2)
    elseif T1 == T2
        obj1 === obj2 || compareMarker(markObj(obj1), markObj(obj2))
    else
        false
    end
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
            anchorNew = linkPointer(anchor, ChainPointer(fieldSym, outputValConstraint))
            getFieldParamsCore!(paramPairs, field, anchorNew)
        end
    end
    nothing
end

uniqueParams(ps::AbstractArray{<:JaggedParam}) = 
markUnique(ps, compareFunction=compareParamBox)[end]


function markParamsCore!(indexDict::IdDict{Symbol, Int}, leafPars)
    for i in leafPars
        sym = i.symbol.name
        get!(indexDict, sym, 0)
        i.symbol.index = (indexDict[sym] += 1)
    end
end

function markParams!(pars::AbstractVector{<:JaggedParam{T}}) where {T}
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

function topoSortCore!(hbNodesIdSet::Set{UInt}, 
                       orderedNodes::Vector{<:JaggedParam{T}}, 
                       haveBranches::Vector{Bool}, connectRoots::Vector{Bool}, 
                       node::JaggedParam{T}, recursive::Bool=false) where {T}
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

function topoSortINTERNAL(nodes::AbstractVector{<:JaggedParam{T}}) where {T}
    orderedNodes = JaggedParam{T}[]
    haveBranches = Bool[]
    connectRoots = Bool[]
    hbNodesIdSet = Set{UInt}()
    for node in nodes
        topoSortCore!(hbNodesIdSet, orderedNodes, haveBranches, connectRoots, node)
    end
    orderedNodes, haveBranches, connectRoots
end

function topoSort(nodes::AbstractVector{<:JaggedParam{T}}) where {T}
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


evalParamSet(s::ParamBox) = obtain(s)

evalParamSet(s::ParamTypeArr{<:Any, 1}) = obtain(s)

evalParamSet(s::FlatParamVec{T}) where {T} = map(obtain, s)

evalParamSet(s::MiscParamVec{T}) where {T} = map(evalParamSet, s)


# FlatParamVec: [[d0...], d1...]          # Used for  input-param set

# MiscParamVec: [[[d0...], d1...], d2...] # Used for output-param set

pushParam!(arr::AbstractArray, param::ParamBox) = push!(arr, param)


const AbstractFlatParamSet{T, S1<:ElementalParam{<:T}, S2<:FlattenedParam{<:T}} = 
      AbstractVector{Union{Vector{S1}, S2}}

#? Possibly rename the field names in Flat/MiscParamSet?
## (Also applied for TemporaryStorage, FixedSizeStorage, ParamPointerBox)

struct FlatParamSet{T, S1<:ElementalParam{<:T}, 
                    S2<:FlattenedParam{<:T}} <: AbstractFlatParamSet{T, S1, S2}
    d0::Vector{S1}
    d1::Vector{S2}

    FlatParamSet{T}(d0::Vector{S1}, d1::Vector{S2}) where 
                   {T, S1<:ElementalParam{<:T}, S2<:FlattenedParam{<:T}} = 
    new{T, S1, S2}(d0, d1)
end

const PrimParamSet{T, S1<:ElementalParam{T}, S2<:InnerSpanParam{T}} = 
      FlatParamSet{T, S1, S2}

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

function iterate(fps::FlatParamSet)
    i = firstindex(fps)
    (getindex(fps, i), i+1)
end

function iterate(fps::FlatParamSet, state)
    if state > length(fps)
        nothing
    else
        (getindex(fps, state), state+1)
    end
end

length(fps::FlatParamSet) = length(fps.d1) + 1

getproperty(fps::FlatParamSet, field::Symbol) = getfield(fps, field)

const AbstractMiscParamSet{T, S1<:ElementalParam{<:T}, S2<:FlattenedParam{<:T}, 
                           S3<:JaggedParam{<:T}} = 
      AbstractVector{Union{FlatParamSet{T, S1, S2}, S3}}

struct MiscParamSet{T, S1<:ElementalParam{<:T}, S2<:FlattenedParam{<:T}, 
                    S3<:JaggedParam{<:T}} <: AbstractMiscParamSet{T, S1, S2, S3}
    inner::FlatParamSet{T, S1, S2}
    outer::Vector{S3}
end

const AbstractParamSet{T} = Union{AbstractFlatParamSet{T}, AbstractMiscParamSet{T}}

size(mps::MiscParamSet) = size(mps.outer) .+ 1

firstindex(::MiscParamSet) = 1

lastindex(mps::MiscParamSet) = length(mps.outer) + 1

function getindex(mps::MiscParamSet, i::Int)
    if i == 1
        mps.inner
    else
        getindex(mps.outer, i+firstindex(mps.outer)-2)
    end
end

function getindex(mps::MiscParamSet, sector::Symbol, i::Int)
    getindex(getfield(mps, sector), i)
end

function getindex(mps::MiscParamSet, sec1::Symbol, sec2::Symbol, i::Int)
    getindex(getfield(getfield(mps, sec1), sec2), i)
end

function setindex!(mps::MiscParamSet, val, i::Int)
    if i == firstindex(mps)
        mps.inner .= val
    else
        setindex!(mps.outer, val, i+firstindex(mps.outer)-2)
    end
end

function setindex!(mps::MiscParamSet, val, sector::Symbol, i::Int)
    setindex!(getfield(mps, sector), val, i)
end

function setindex!(mps::MiscParamSet, val, sec1::Symbol, sec2::Symbol, i::Int)
    setindex!(getfield(getfield(mps, sec1), sec2), val, i)
end

pushParam!(fps::MiscParamSet, a::ElementalParam) = push!(fps.inner.d0, a)
pushParam!(fps::MiscParamSet, a::FlattenedParam) = push!(fps.inner.d1, a)
pushParam!(fps::MiscParamSet, a::JaggedParam) = push!(fps.outer, a)

function iterate(mps::MiscParamSet)
    i = firstindex(mps)
    (getindex(mps, i), i+1)
end

function iterate(mps::MiscParamSet, state)
    if state > length(mps)
        nothing
    else
        (getindex(mps, state), state+1)
    end
end

length(mps::MiscParamSet) = length(mps.outer) + 1

getproperty(fps::MiscParamSet, field::Symbol) = getfield(fps, field)


function initializeParamSet(::Type{FlatParamSet}, ::Type{T}=Any; 
                            d0Type::MissingOr{Type{S1}}=missing, 
                            d1Type::MissingOr{Type{S2}}=missing) where 
                           {T, S1<:ElementalParam{<:T}, S2<:FlattenedParam{T}}
    bl = isconcretetype(T)
    if ismissing(d0Type)
        d0Type = ifelse(bl, ElementalParam{T}, ElementalParam{<:T})
    end
    if ismissing(d1Type)
        d1Type = ifelse(bl, FlattenedParam{T}, FlattenedParam{<:T})
    end
    FlatParamSet{T}(d0Type[], d1Type[])
end

initializeParamSet(::Type{FlatParamSet{T, S1, S2}}) where {T, S1, S2} = 
FlatParamSet(S1[], S2[])

initializeParamSet(f::Function) = 
initializeParamSet(SelectTrait{ParameterizationStyle}()(f))

initializeParamSet(::GenericFunction) = initializeParamSet(FlatParamSet)

initializeParamSet(::TypedParamFunc{T}) where {T} = initializeParamSet(FlatParamSet, T)


function initializeParamSet(::Type{MiscParamSet}, ::Type{T}=Any; 
                            d0Type::MissingOr{Type{S1}}=missing, 
                            d1Type::MissingOr{Type{S2}}=missing, 
                            d2Type::MissingOr{Type{S3}}=missing) where 
                           {T, S1<:ElementalParam{<:T}, S2<:FlattenedParam{T}, 
                            S3<:JaggedParam{T}}
    inner = initializeParamSet(FlatParamSet, T; d0Type, d1Type)
    if ismissing(d2Type)
        d2Type = ifelse(isconcretetype(T), JaggedParam{T}, JaggedParam{<:T})
    end
    MiscParamSet(inner, d2Type[])
end

initializeParamSet(::Type{MiscParamSet{T, S1, S2, S3}}) where {T, S1, S2, S3} = 
MiscParamSet(FlatParamSet(S1[], S2[]), S3[])


#!  SingleNestParamSet
#!  DoubleNestParamSet
const FlatPSetInnerPtr{T} = PointPointer{T, 2, Tuple{FirstIndex, Int}}
const FlatPSetOuterPtr{T} = IndexPointer{Volume{T}}
const FlatParamSetIdxPtr{T} = Union{FlatPSetInnerPtr{T}, FlatPSetOuterPtr{T}}

struct FlatParamSetFilter{T} <: BlockPointer{1, 2}
    d0::Memory{FlatPSetInnerPtr{T}} #! Replace by immutable vector
    d1::Memory{FlatPSetOuterPtr{T}} #! Replace by immutable vector
    sourceID::Identifier
end

function FlatParamSetFilter(pSet::FlatParamSet{T}, d0Ids::AbstractVector{Int}, 
                            d1Ids::AbstractVector{Int}) where {T}
    d0Ptr = map(d0Idx->ChainPointer((FirstIndex, d0Idx), TensorType(T)), d0Ids)
    d1Ptr = map(d1Idx->ChainPointer((d1Idx+1,), TensorType(pSet.d1[d1Idx])), d1Ids)
    FlatParamSetFilter(getMemory(d0Ptr), getMemory(d1Ptr), Identifier(pSet))
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

#= Additional Method =#
getField(obj, p::FlatParamSetFilter) = map(x->getField(obj, x), p)

getField(obj::FlatParamSetFilter, ptr::FlatPSetInnerPtr) = 
getindex(obj.d0, last(ptr.chain))

getField(obj::FlatParamSetFilter, ptr::FlatPSetOuterPtr) = 
getFlatSetIndexCore(obj, first(ptr.chain))

function getField(obj::PointedObject{<:Any, <:FlatParamSetFilter}, 
                  ptr::FlatParamSetIdxPtr)
    getField(obj.obj, getField(obj.ptr, ptr))
end

#= Additional Method =#
linkPointer(prev::FlatParamSetFilter{T}, here::FlatParamSetIdxPtr{T}) where {T} = 
getField(prev, here)


getParamSector(source::FlatParamSet, ::Type{<:ElementalParam}) = 
( first(source), ChainPointer(FirstIndex()) )

getParamSector(source::MiscParamSet, ::Type{<:ElementalParam}) = 
( (first∘first)(source), ChainPointer(FirstIndex(), ChainPointer(FirstIndex())) )

getParamSector(source::MiscParamSet, ::Type{<:FlattenedParam}) = 
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
        linkPointer(anchor, ChainPointer(idx, returnType))
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
    d0ptrs = linkPointer.(Ref(FirstIndex()), d0ptrs)
    if !isempty(d1ptrs)
        offset = 2 - firstindex(paramSet.d1)
        d1ptrs = map(x->ChainPointer(x.chain .+ offset, x.type), d1ptrs)
    end
    FlatParamSetFilter(getMemory(d0ptrs), getMemory(d1ptrs), Identifier(paramSet))
end

function locateParam!(paramSet::AbstractVector, target::FlatParamSet{T}) where {T}
    d0ptrs, d1ptrs = map(fieldnames(FlatParamSet)) do n
        locateParam!.(Ref(paramSet), getfield(target, n))
    end
    FlatParamSetFilter(getMemory(d0ptrs), getMemory(d1ptrs), Identifier(paramSet))
end


const DimensionalSpanMemory{T} = Union{T, ShapedMemory{T}, ShapedMemory{ShapedMemory{T}}}

struct DimSpanDataCacheBox{T} <: QueryBox{DimensionalSpanMemory{T}}
    d0::Dict{UInt, T}
    d1::Dict{UInt, ShapedMemory{T}}
    d2::Dict{UInt, ShapedMemory{ShapedMemory{T}}}
end

DimSpanDataCacheBox(::Type{T}) where {T} = 
DimSpanDataCacheBox( Dict{UInt, T}(), 
                     Dict{UInt, ShapedMemory{T}}(), 
                     Dict{UInt, ShapedMemory{ShapedMemory{T}}}() )

getDimSpanSector(cache::DimSpanDataCacheBox{T}, ::ElementalParam{T}) where {T} = cache.d0

getDimSpanSector(cache::DimSpanDataCacheBox{T}, ::FlattenedParam{T}) where {T} = cache.d1

getDimSpanSector(cache::DimSpanDataCacheBox{T}, ::JaggedParam{T}) where {T} = cache.d2

formatDimSpanMemory(::Type{T}, val::T) where {T} = itself(val)

formatDimSpanMemory(::Type{T}, val::AbstractArray{T}) where {T} = ShapedMemory(val)

formatDimSpanMemory(::Type{T}, val::JaggedAbtArray{T}) where {T} = 
ShapedMemory(ShapedMemory.(val))

function cacheParam!(cache::DimSpanDataCacheBox{T}, param::ParamBox{T}) where {T}
    get!(getDimSpanSector(cache, param), objectid(param)) do
        formatDimSpanMemory(T, obtain(param))
    end
end

function cacheParam!(cache::DimSpanDataCacheBox{T}, s::AbstractParamSet{T}, 
                    idx::ChainPointer) where {T}
    pb = getField(s, idx)
    cacheParam!(cache, pb)
end

const ParamCollection{T} = Union{AbstractParamSet{T}, ParamTypeArr{<:ParamBox{T}, 1}}

function cacheParam!(cache::DimSpanDataCacheBox{T}, s::ParamCollection{T}) where {T}
    map(s) do p
        cacheParam!(cache, p)
    end
end