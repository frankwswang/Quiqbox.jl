export NodeVar, NodeVar, CellParam, GridParam, ParamList, ParamNest, setScreenLevel!, 
       setScreenLevel, symOf, inputOf, obtain, setVal!, screenLevelOf, markParams!, 
       topoSort, getParams

using Base: Fix2, Threads.Atomic, issingletontype
using Test: @inferred
using LRUCache


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
        new{T, length(shape)}(vec(value), shape)
    end
end

ShapedMemory(value::AbstractArray{T}, shape::Tuple{Vararg{Int}}=size(value)) where {T} = 
ShapedMemory(Memory{T}( vec(value) ), shape)

ShapedMemory(::Type{T}, value::AbstractArray{T}) where {T} = ShapedMemory(value)

ShapedMemory(::Type{T}, value::T) where {T} = ShapedMemory( fill(value) )

ShapedMemory(sm::ShapedMemory) = itself(sm)

obtainDimVal(sm::ShapedMemory) = itself.( reshape(sm.value, sm.shape) )

obtainDimVal(sm::AbstractArray) = itself(sm)


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
exclude0DimData(::Type{<:AbtArray0D}) = 
throw(ArgumentError("`AbstractArray{<:Any, 0}` is not allowed as the input argument."))
exclude0DimData(::Type{<:DimensionalParam{<:Any, 0}}) = 
throw(ArgumentError("`DimensionalParam{<:Any, 0}` is not allowed as the input argument."))

excludeAbtArray(::T) where {T} = excludeAbtArray(T)
excludeAbtArray(::Type) = nothing
excludeAbtArray(::Type{<:AbstractArray}) = 
throw(ArgumentError("`AbstractArray` is not allowed as an input argument."))

function checkAbtArrayInput(::Type{V}) where {T, V<:AbstractArray{T}}
    exclude0DimData(V)
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
    exclude0DimData.(allArgs)
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

function genSeqAxis(lens::NTuple{N, Int}, sym::Symbol=:e) where {N}
    Tuple( (Symbol(sym, i), lens[i]) for i in 1:N )
end

bundle(arg::T, args::T...) where {T} = collect( themselves(arg, args...) )
const FofBundle = typeof(bundle)

struct FixedShapeLink{T, F<:Function, N, L} <: StructFunction{T, F}
    f::F
    axis::NTuple{L, Tuple{Symbol, Int}}
    extent::Int

    function FixedShapeLink(f::F, ::Type{V}, arg, args...; 
                            axis::Union{NonEmptyTuple{Tuple{Symbol, Int}}, 
                                        Missing}=missing) where {F<:Function, V}
        T, N = checkAbtArrayInput(V)
        allArgs = (arg, args...)
        exclude0DimData.(allArgs)
        val = checkReturnType(f, AbstractArray{<:T}, allArgs)
        if ismissing(axis)
            axis = genSeqAxis(val|>size)
        else
            checkReshapingAxis(val, last.(axis))
        end
        L = length(axis)
        L < 1 && throw(AssertionError("Return dimension of `FixedShapeLink` should at "*
                                      "least be 1."))
        new{T, F, N, L}(f, axis, prod(axis))
    end

    function FixedShapeLink(::V, args::V...; 
                            axis::Union{Tuple{Tuple{Symbol, Int}}, Missing}=missing) where 
                           {V}
        T, N = checkAbtArrayInput(V)
        extent = length(args) + 2
        if ismissing(axis)
            axis = genSeqAxis( (extent,) )
        else
            checkReshapingAxis(ones(extent), last.(axis))
        end
        new{T, FofBundle, N, 1}(bundle, axis, extent)
    end
end

function callFixedShapeLinkCore(ml::FixedShapeLink{T, <:Any, N}, arg, args...) where {T, N}
    allArgs = (arg, args...)
    exclude0DimData.(allArgs)
    ml.f(allArgs...)::AbstractArray{<:assertAbtArrayOutput(T, Val(N))}
end

function (ml::FixedShapeLink{T, F, N, L})(arg::T, args::T...) where {T, F, N, L}
    res = callFixedShapeLinkCore(ml, arg, args...)
    iBegin = firstindex(res)
    reshape(res[iBegin : (iBegin + ml.extent - 1)], getindex.(ml.axis, 2))
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

struct NodeVar{T, N} <: PrimitiveParam{T, N}
    input::ShapedMemory{T, N}
    symbol::IndexedSym
    screen::TernaryNumber

    function NodeVar(input::AbtArrayOrMem{T, N}, symbol::SymOrIndexedSym, 
                     screen::Union{TernaryNumber, Int}=TPS1) where {T, N}
        checkPrimParamElementalType(T)
        checkScreenLevel(screen, getScreenLevelRange(PrimitiveParam))
        input = ShapedMemory(input|>deepcopy)
        new{T, N}(input, IndexedSym(symbol), genTernaryNumber(screen))
    end
end

NodeVar(input::T, symbol::SymOrIndexedSym, 
        screen::Union{TernaryNumber, Int}=TPS1) where {T} = 
NodeVar(fill(input), symbol, screen)


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

getParamBoxArgDim(::Type{<:ParamBoxSingleArg{<:Any, N}}) where {N} = N
getParamBoxArgDim(::T) where {T<:ParamBoxSingleArg} = getParamBoxArgDim(T)

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

function checkCellParamArg(::TypedReduction{T, <:iTalike}, input::I, shifter::S, 
                           memory::Union{ShapedMemory{T, 0}, T, Missing}) where {T, I, S}
    checkParamContainerArgType1(I, Tuple{ElementalParam{T}})
    checkParamBoxInput(input, dimMax=0)
    if ismissing(memory)
        memory = ShapedMemory( fill(input[1]|>obtain|>shifter) )
    elseif memory isa T
        memory = ShapedMemory( fill(memory) )
    end
    TypedReduction(T), iT, deepcopy(memory)
end

function checkCellParamArg(f::TypedReduction{T, F}, input::I, shifter::S, 
                           memory::Union{ShapedMemory{T, 0}, T, Missing}) where {T, F, I, S}
    checkParamBoxInput(input)
    if ismissing(memory)
        memory = ShapedMemory( fill(f(obtain.(input)...)|>shifter) )
    elseif memory isa T
        memory = ShapedMemory( fill(memory) )
    end
    f, F, deepcopy(memory)
end

initializeOffset(::Type) = nothing
initializeOffset(::Type{T}) where {T<:Number} = zero(T)

mutable struct CellParam{T, F<:Function, I<:ParamBoxInputType{T}} <: ParamBox{T, 0, I}
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
            if offset===nothing
                offset = if sl > 0
                    obtainDimVal(memory)
                else
                    memVal = obtainDimVal(memory)
                    typedSub(memVal, memVal)
                end
            end
            new{T, funcType, I}(lambda, input, symbol, memory, screen, offset)
        end
    end
end

function CellParam(func::Function, input::ParamBoxInputType{T}, symbol::SymOrIndexedSym; 
                   init::Union{ShapedMemory{T, 0}, T, Missing}=missing) where {T}
    lambda = TypedReduction(func, obtain.(input)...)
    CellParam(lambda, input, symbol, init, TUS0, initializeOffset(T))
end

CellParam(func::Function, input::ParamBoxSingleArg{T}, symbol::SymOrIndexedSym; 
          init::Union{ShapedMemory{T, 0}, T, Missing}=missing) where {T} = 
CellParam(func, (input,), symbol; init)

CellParam(func::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
          symbol::SymOrIndexedSym; 
          init::Union{ShapedMemory{T, 0}, T, Missing}=missing) where {T} = 
CellParam(func, (input1, input2), symbol; init)

CellParam(func::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
          input3::ParamBoxSingleArg{T}, symbol::SymOrIndexedSym; 
          init::Union{ShapedMemory{T, 0}, T, Missing}=missing) where {T} = 
CellParam(func, (input1, input2, input3), symbol; init)

function CellParam(par::CellParam{T}, symbol::SymOrIndexedSym=symOf(par); 
                   init::Union{ShapedMemory{T, 0}, T, Missing}=par.memory) where {T}
    offset = isOffsetEnabled(par) ? par.offset : nothing
    CellParam(par.lambda, par.input, symbol, init, par.screen, offset)
end

CellParam(input::PrimitiveParam{T, 0}, 
          symbol::SymOrIndexedSym=symOf(input)) where {T} = 
CellParam(TypedReduction(T), (input,), symbol)

CellParam(var, varSym::SymOrIndexedSym, symbol::SymOrIndexedSym=varSym) = 
CellParam(NodeVar(var, varSym), symbol)


function checkGridParamArg(::StableMorphism{T, <:iTalike, N}, input::I, memory::M) where 
                           {T, N, I, M<:AbtArrayOrMem{T, N}}
    checkParamContainerArgType1(I, Tuple{ParamBoxSingleArg{T, N}})
    MI = typeof(input[1])
    if !(M <: MI)
        throw(AssertionError("The type of memory should be the subtype of type of "*
                             "`input[1]` (`::$M1`)"))
    end
    checkParamBoxInput(input, dimMin=1)
    StableMorphism(AbstractArray{T, N}), iT, deepcopy(memory|>ShapedMemory)
end

function checkGridParamArg(::StableMorphism{T, <:iTalike, N}, input::I, ::Missing) where 
                           {T, N, I}
    checkParamContainerArgType1(I, Tuple{ParamBoxSingleArg{T, N}})
    checkParamBoxInput(input, dimMin=1)
    StableMorphism(AbstractArray{T, N}), iT, deepcopy(input[1]|>obtain|>ShapedMemory)
end

function throwGridParamDimErrorMessage()
    throw(ArgumentError("Returned array should have dimension `N` larger than 0. Use "*
                        "`$CellParam` for returning scalar-type output."))
end

function checkGridParamArg(f::StableMorphism{T, F, N}, input::I, memory::M) where 
                           {T, F, N, I, M<:AbtArrayOrMem{T, N}}
    N < 1 && throwGridParamDimErrorMessage()
    checkParamBoxInput(input)
    checkReturnType(f, M, obtain.(input))
    f, F, deepcopy(memory|>ShapedMemory)
end

function checkGridParamArg(f::StableMorphism{T, F, N}, input::I, ::Missing) where 
                           {T, F, N, I}
    N < 1 && throwGridParamDimErrorMessage()
    checkParamBoxInput(input)
    f, F, deepcopy( f(obtain.(input)...)|>ShapedMemory )
end

mutable struct GridParam{T, F<:Function, I<:ParamBoxInputType{T}, N} <: ParamBox{T, N, I}
    const lambda::StableMorphism{T, F, N}
    const input::I
    const symbol::IndexedSym
    @atomic memory::ShapedMemory{T, N}

    function GridParam(lambda::StableMorphism{T, F, N}, input::I, 
                       symbol::SymOrIndexedSym, 
                       memory::Union{AbtArrayOrMem{T, N}, Missing}=missing) where 
                      {T, F, N, I<:ParamBoxInputType{T}}
        lambda, funcType, memory = checkGridParamArg(lambda, input, memory)
        new{T, funcType, I, N}(lambda, input, IndexedSym(symbol), memory)
    end
end

function GridParam(func::Function, input::ParamBoxInputType{T}, symbol::SymOrIndexedSym; 
                   init::Union{AbtArrayOrMem{T}, Missing}=missing) where {T}
    lambda = StableMorphism(func, obtain.(input)...)
    GridParam(lambda, input, symbol, init)
end

GridParam(func::Function, input::ParamBoxSingleArg{T}, symbol::SymOrIndexedSym; 
          init::Union{AbtArrayOrMem{T}, Missing}=missing) where {T} = 
GridParam(func, (input,), symbol; init)

GridParam(func::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
          symbol::SymOrIndexedSym; 
          init::Union{AbtArrayOrMem{T}, Missing}=missing) where {T} = 
GridParam(func, (input1, input2), symbol; init)

GridParam(func::Function, input1::ParamBoxSingleArg{T}, input2::ParamBoxSingleArg{T}, 
          input3::ParamBoxSingleArg{T}, symbol::SymOrIndexedSym; 
          init::Union{AbtArrayOrMem{T}, Missing}=missing) where {T} = 
GridParam(func, (input1, input2, input3), symbol; init)

GridParam(par::GridParam{T}, symbol::SymOrIndexedSym=symOf(par); 
          init::Union{AbtArrayOrMem{T}, Missing}=par.memory) where {T} = 
GridParam(par.lambda, par.input, symbol, init)

function GridParam(input::PrimitiveParam{T, N}, 
          symbol::SymOrIndexedSym=symOf(input)) where {T, N}
    exclude0DimData(input)
    GridParam(StableMorphism(AbstractArray{T, N}), (input,), symbol)
end

function GridParam(input::AbstractArray{<:ElementalParam{T}, N}, 
                   symbol::SymOrIndexedSym) where {T, N}
    exclude0DimData(input)
    GridParam(StableMorphism(AbstractArray{T, N}), (input,), symbol)
end

function GridParam(val::AbstractArray, valSym::SymOrIndexedSym, 
                   symbol::SymOrIndexedSym=valSym)
    exclude0DimData(val)
    GridParam(NodeVar(val, valSym), symbol)
end


struct ParamList{T, N, I<:DimensionalParam{T, N}} <: ParamPile{T, N, I, 1}
    input::Memory{I}
    symbol::IndexedSym

    function ParamList(input::Memory{I}, symbol::SymOrIndexedSym) where 
                      {T, N, I<:DimensionalParam{T, N}}
        isempty(input) && throw(AssertionError("`input` should not be empty."))
        new{T, N, I}(Memory{I}(input), IndexedSym(symbol))
    end
end

ParamList(input::AbstractArray{I}, symbol::SymOrIndexedSym) where 
         {T, N, I<:DimensionalParam{T, N}} = 
ParamList(Memory{I}( vec(input) ), symbol)

ParamList(pl::ParamList, symbol::IndexedSym=pl.symbol) = ParamList(pl.input, symbol)


function checkParamContainerArgType2(::Type{I}, ::Type{T}, ::Val{N}) where  {I, T, N}
    R = NonEmptyTuple{ParamBoxSingleArg{T, N}}
    I <: R || throw(ArgumentError("`I` should be a subtype of `$R`."))
    nothing
end

function checkParamContainerArgType3(name::AbstractString, len::Int, extent::Int)
    if len != extent
        throw(DomainError(len, "The length of `$name` should match `ml::FixedShapeLink`'s "*
                               "specification: $extent."))
    end
    nothing
end

function checkParamNestArg(ml::FixedShapeLink{T, FofBundle, N}, input::I, 
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

function checkParamNestArg(ml::FixedShapeLink{T, F, N}, input::I, 
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

mutable struct LinkParam{T, N, I<:ParamGrid{T, N}} <: ParamBox{T, N, I}
    const input::I
    const index::Int
    @atomic memory::ShapedMemory{T, N}

    function LinkerParam(input::I, index::Int, memory::ShapedMemory{T, N}) where {I, T, N}
        maxIndex = length(input.output)
        if index < 1 || index > maxIndex
            throw(DomainError(index, "`index is out of the allowed range: (1, $maxIndex)`"))
        end
        new{T, N, I}(input, index, memory)
    end
end

struct ParamNest{T, F<:Function, I<:ParamBoxInputType{T}, N, L} <: ParamGrid{T, N, I, L}
    linker::FixedShapeLink{T, F, N, L}
    input::I
    symbol::IndexedSym
    output::Memory{LinkParam{T, N, I}}

    function ParamNest(linker::FixedShapeLink{T, F, N, L}, input::I, 
                       symbol::SymOrIndexedSym, 
                       memory::Union{Memory{ShapedMemory{T, N}}, Missing}=missing) where 
                      {T, N, F, L, I<:ParamBoxInputType{T}}
        linker, funcType, memory = checkParamNestArg(linker, input, memory)
        output = Memory{LinkParam{T, N, I}}(undef, linker.extent)
        pn = new{T, funcType, I, N, L}(linker, input, IndexedSym(symbol), output)
        for (i, val) in zip(eachindex(output), memory)
            pn.output[i] = LinkParam(pn, i, ShapedMemory(val))
        end
        pn
    end
end

function ParamNest(linker::FixedShapeLink{T, F, N, L}, input::I, symbol::SymOrIndexedSym, 
                   output::Memory{LinkParam{T, N, I}}) where 
                  {T, N, F, L, I<:ParamBoxInputType{T}}
    memory = Memory{ShapedMemory{T, N}}(getproperty.(output, :memory))
    ParamNest(linker, input, symbol, memory)
end


function ParamNest(func::Function, input::ParamBoxInputType{T}, 
                   symbol::SymOrIndexedSym) where {T}
    inputVal = obtain.(input)
    out = func(inputVal...)
    out isa AbstractArray || throw(AssertionError("`func` should output an AbstractArray."))
    linker = FixedShapeLink(func, typeof(out), inputVal...)
    ParamNest(linker, input, symbol)
end

ParamNest(input::TwiceThriceNTuple{ParamBoxSingleArg{T, N}}, 
          symbol::SymOrIndexedSym) where {T, N} = 
ParamNest(FixedShapeLink(obtain.(input)...), input, symbol)


getScreenLevelRange(::Type{<:LinkParam}) = (0, 0)

getScreenLevelRange(::Type{<:ParamBox}) = (0, 0)

getScreenLevelRange(::Type{<:ParamBox{T, 0}}) where {T} = (0, checkTypedOpMethods(T) * 2)

getScreenLevelRange(::Type{<:ParamPile}) = (0, 0)

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

screenLevelOf(::LinkParam) = 0

screenLevelOf(::ParamPile) = 0

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

setScreenLevel(p::CellParam, level::Int) = setScreenLevel!(CellParam(p), level)

setScreenLevel(p::NodeVar, level::Int) = NodeVar(p.input, p.symbol, TernaryNumber(level))


function memorize!(p::ParamBox{T, N}, newMem::AbtArrayOrMem{T, N}) where {T, N}
    oldMem = obtainDimVal(p.memory)
    if p.memory.shape == size(newMem)
        safelySetVal!(p.memory.value, newMem)
    else
        @atomic p.memory = ShapedMemory(newMem)
    end
    oldMem
end

function memorize!(p::ParamBox{T, 0}, newMem::AbtArrayOrMem{T, 0}) where {T}
    oldMem = obtainDimVal(p.memory)
    safelySetVal!(p.memory.value, newMem)
    oldMem
end

memorize!(p::ParamBox{T}, newMem::T) where {T} = memorize!(p, fill(newMem))

memorize!(p::ParamBox) = memorize!(p, obtain(p))


indexedSymOf(pc::DimensionalParam) = pc.symbol

symOf(pc::DimensionalParam) = indexedSymOf(pc).name

inputOf(pb::DimensionalParam) = pb.input


mutable struct NodeMarker{T}
    visited::Bool
    value::T

    NodeMarker(init::T, ::Type{U}=T) where {T, U} = new{U}(false, init)
end

obtain(p::PrimitiveParam) = obtainINTERNAL(p)

function obtain(p::Union{CompositeParam{T}, AbstractArray{<:ElementalParam{T}}}) where {T}
    lock( ReentrantLock() ) do
        obtainINTERNAL(p)
    end
end

# Sugar syntax. E.g., for obtaining values of the first element in a parameter set.
obtainINTERNAL(pars::AbstractArray{<:ElementalParam{T}}) where {T} = obtainINTERNAL.(pars)

obtainINTERNAL(p::PrimitiveParam) = directObtain(p)

directObtain(p::PrimitiveParam) = obtainDimVal(p.input)

const ParamBMemDict{T} = IdDict{ParamBox{T}, NodeMarker{<:ShapedMemory{T}}}

const ParamPMemDict{T} = IdDict{ParamPile{T}, NodeMarker{<:VectorOrMem{<:ShapedMemory{T}}}}

function obtainINTERNAL(p::CompositeParam{T}) where {T}
    pbDict = ParamBMemDict{T}()
    ppDict = ParamPMemDict{T}()
    searchObtain(pbDict, ppDict, p)
end

function searchObtain(pbDict::ParamBMemDict{T}, ppDict::ParamPMemDict{T}, 
                      p::ParamList{T}) where {T}
    searchObtain.(Ref(pbDict), Ref(ppDict), p.input)
end

function searchObtain(pbDict::ParamBMemDict{T}, ppDict::ParamPMemDict{T}, 
                      p::ParamNest{T}) where {T}
    inputVal = searchObtain.(Ref(pbDict), Ref(ppDict), p.input)
    p.linker(inputVal...)
end

function searchObtainLoop(pbDict::ParamBMemDict{T}, ppDict::ParamPMemDict{T}, 
                          input::AbstractArray{<:DimensionalParam{T}}) where {T}
    map(input) do child
        searchObtain(pbDict, ppDict, child)
    end
end

function searchObtainLoop(pbDict::ParamBMemDict{T}, ppDict::ParamPMemDict{T}, 
                          input::DimensionalParam{T}) where {T}
    searchObtain(pbDict, ppDict, input)
end

function searchObtainCore(shiftVal::F, pbDict::ParamBMemDict{T}, ppDict::ParamPMemDict{T}, 
                          p::ParamBox{T, N}) where {T, F<:Union{iT, ValShifter{T}}, N}
    # Depth-first search by recursive calling
    valBox = p.memory
    marker::NodeMarker{typeof(valBox)} = get!(pbDict, p, NodeMarker(valBox))
    if !marker.visited
        marker.visited = true
        f = p.lambda
        res = f( (searchObtainLoop(pbDict, ppDict, x) for x in p.input)... ) |> shiftVal
        marker.value = ShapedMemory(T, res)
    else
        marker.value
    end
end

function searchObtainCore(::F, pbDict::ParamBMemDict{T}, ppDict::ParamPMemDict{T}, 
                          p::LinkParam{T, N}) where {T, F<:Union{iT, ValShifter{T}}, N}
    # Depth-first search by recursive calling
    idx = p.index
    input = p.input
    lpMarkVal = p.memory
    lpMarkValType = typeof(lpMarkVal)
    marker::NodeMarker{lpMarkValType} = if haskey(pbDict, p)
        getindex(pbDict, p)
    else
        newMarker = if haskey(ppDict, input)
            inputMarker::NodeMarker{Memory{lpMarkValType}} = getindex(ppDict, input)
            NodeMarker(inputMarker.value[idx])
        else
            inputMarker = NodeMarker(Memory{lpMarkValType}(undef, input.linker.extent))
            NodeMarker(lpMarkVal)
        end
        setindex!(pbDict, newMarker, p)
        newMarker
    end
    if !marker.visited
        marker.visited = true
        if !inputMarker.visited
            res = ShapedMemory.(T, searchObtain(pbDict, ppDict, p.input))
            inputMarker.value .= res
            inputMarker.visited = true
            marker.value = res[idx]
        else
            marker.value = inputMarker.value[idx]
        end
    else
        marker.value
    end
end

function searchObtain(pbDict::ParamBMemDict{T}, ppDict::ParamPMemDict{T}, 
                      p::ParamBox{T, N}) where {T, N}
    sl = checkScreenLevel(screenLevelOf(p), getScreenLevelRange(ParamBox{T, N}))
    if sl == 0
        shiftVal = genValShifter(T, (isOffsetEnabled(p) ? p.offset : nothing))
        obtainDimVal( searchObtainCore(shiftVal, pbDict, ppDict, p) )
    else
        p.offset
    end
end

searchObtain(::ParamBMemDict{T}, ::ParamPMemDict{T}, p::PrimitiveParam{T}) where {T} = 
directObtain(p)

(pn::DimensionalParam)() = obtain(pn)

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


struct ParamMarker{M<:NonEmptyTuple{AbstractMarker}, MF<:AbstractMarker, 
                   N} <: AbstractMarker{M}
    typeID::UInt
    marker::M
    funcID::MF
    metaID::NTuple{N, UInt}
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
    isbitstype(ET)
end

function isPrimVarCollection(arg::Tuple)
    all(isbits(i) for i in arg)
end

function markObj(input::Union{AbstractArray, Tuple})
    isPrimVarCollection(input) ? ValueMarker(input) : CollectionMarker(input)
end

markObj(input) = ObjectMarker(input)

markObj(f::Function) = 
markObj( (objectid(f),) )

markObj(f::TypedFunction) = 
markObj( (objectid(f),) )

markObj(f::StructFunction) = 
markObj( (objectid(f|>typeof), getproperty.(f, propertynames(f))...) )

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

function ParamMarker(p::T) where {T<:LinkParam}
    ParamMarker(objectid(T), (markObj(p.input),), NothingID, (objectid(p.index),))
end

function ParamMarker(p::T) where {T<:ParamList}
    ParamMarker(objectid(T), (markObj(p.input),), NothingID, ())
end

function ParamMarker(p::T) where {T<:ParamNest}
    ParamMarker(objectid(T), markObj.(pn.input), markObj(p.linker), ())
end

compareMarker(pm1::AbstractMarker, pm2::AbstractMarker) = false

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

compareParamContainer(::DimensionalParam, ::DimensionalParam) = false

compareParamContainer(::DimensionalParam, ::Any) = false

compareParamContainer(::Any, ::DimensionalParam) = false

compareParamContainer(p1::T, p2::T) where {T<:PrimitiveParam} = p1 === p2

compareParamContainer(p1::CompositeParam{T, N}, p2::CompositeParam{T, N}) where {T, N} = 
p1 === p2 || compareMarker(ParamMarker(p1), ParamMarker(p2))


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