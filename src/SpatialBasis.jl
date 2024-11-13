export PrimitiveOrb, CompositeOrb, FrameworkOrb, genGaussTypeOrb

function unpackSquareIntNormalizer(f::F) where {F<:Function}
    inv∘sqrt∘genOverlap(f), nothing
end

struct Storage{T} <: CompositeFunction
    val::T
end

(f::Storage)(_) = f.val

function unpackSquareIntNormalizer(num::Number)
    Storage(num)
end


abstract type ComposedOrb{T, D, B} <: OrbitalBasis{T, D, B} end

abstract type EvalComposedOrb{T, D, B} <: Evaluator{B} end

(f::OrbitalBasis)(x) = evalFunc(f, x)

evalFunc(f::EvalComposedOrb, input, param) = f.f(input, param)

function genNormalizer(f::ComposedOrb{T}) where {T}
    (OnlyParam∘unpackSquareIntNormalizer)( ifelse(f.renormalize, f, one(T)) )
end

function unpackParamFunc!(f::ComposedOrb{T, D}, paramSet::PBoxAbtArray) where {T, D}
    fEvalCore, _ = unpackParamFuncCore!(f, paramSet)
    fEval = PairCombine(*, fEvalCore, genNormalizer(f))
    getEvaluator(f)(fEval), paramSet
end


function genCellEncoder(::Type{T}, sym::Symbol) where {T}
    function (input)
        if input isa ElementalParam{T}
            input
        else
            p = CellParam(T(input), sym)
            setScreenLevel!(p, 1)
            p
        end
    end
end


struct PrimitiveOrb{T, D, B<:FieldAmplitude{T, D}, 
                    C<:NTuple{D, ElementalParam{T}}} <: ComposedOrb{T, D, B}
    body::B
    center::C
    renormalize::Bool
end

function PrimitiveOrb(body::B, center::NTuple{D, ParamOrValue{T}}; 
                      renormalize::Bool=false) where {T, D, B<:FieldAmplitude{T, D}}
    length(center)!=D && throw(AssertionError("The length of `center` must match `D=$D`."))
    encoder = genCellEncoder(T, :cen)
    PrimitiveOrb(body, encoder.(center); renormalize)
end

PrimitiveOrb(ob::PrimitiveOrb) = itself(ob)

struct ShiftByArg{T<:Real, D} <: FieldlessFunction end

(::ShiftByArg{T, D})(input::NTuple{D, Real}, arg::Vararg{T, D}) where {T, D} = 
(input .- arg)

const InputShifter{T, D, F} = InsertInward{F, PointerFunc{ShiftByArg{T, D}, NTuple{D, Int}}}

struct EvalPrimitiveOrb{T, D, B<:EvalFieldAmp{T, D}, F} <: EvalComposedOrb{T, D, B}
    f::MulPair{InputShifter{T, D, B}, OnlyParam{F}}
end

function unpackParamFuncCore!(f::PrimitiveOrb{T, D}, paramSet::PBoxAbtArray) where {T, D}
    pSetId = objectid(paramSet)
    parIds = locateParam!(paramSet, f.center)
    fEvalOuter = unpackFunc!(f.body, paramSet) |> first
    fEval = InsertInward(fEvalOuter, PointerFunc(ShiftByArg{T, D}(), parIds, pSetId))
    fEval, paramSet
end

getEvaluator(::PrimitiveOrb) = EvalPrimitiveOrb

struct CompositeOrb{T, D, B<:FieldAmplitude{T, D}, C<:NTuple{D, ElementalParam{T}}, 
                    P<:ElementalParam{T}} <: ComposedOrb{T, D, B}
    basis::Memory{PrimitiveOrb{T, D, <:B, <:C}}
    weight::Memory{P}
    renormalize::Bool

    function CompositeOrb(basis::AbstractVector{<:PrimitiveOrb{T, D}}, 
                          weight::AbstractVector{P}, renormalize::Bool=false) where 
                         {T, D, P<:ElementalParam{T}}
        if checkEmptiness(basis, :basis) != length(weight)
            throw(AssertionError("`basis` and `weight` must have the same length."))
        end
        B = getfield.(basis, :body) |> eltype
        C = getfield.(basis, :center) |> eltype
        basisMem = Memory{PrimitiveOrb{T, D, <:B, <:C}}(basis)
        weightMem = getMemory(weight)
        new{T, D, B, C, eltype(weightMem)}(basisMem, weightMem, renormalize)
    end
end

decompose(b::T) where {T<:PrimitiveOrb} = Memory{T}(b)
decompose(b::CompositeOrb) = b.basis

function CompositeOrb(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                      weight::AbstractVector{<:ElementalParam{T}}, 
                      renormalize::Bool=false) where {T, D}
    checkEmptiness(basis, :basis)
    primOrbs = mapfoldl(decompose, vcat, basis)
    CompositeOrb(primOrbs, weight, renormalize)
end

function CompositeOrb(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                      weight::AbstractVector{<:ParamOrValue{T}}; 
                      renormalize::Bool=false) where {T, D}
    weightParams = genCellEncoder(T, :w).(weight)
    CompositeOrb(basis, weightParams, renormalize)
end

CompositeOrb(ob::CompositeOrb) = itself(ob)

const WeightedPrimField{T, D, F<:EvalPrimitiveOrb{T, D}} = 
      MulPair{<:F, PointOneFunc{OnlyParam{ItsType}}}

struct EvalCompositeOrb{T, D, B<:EvalPrimitiveOrb{T, D}, F} <: EvalComposedOrb{T, D, B}
    f::MulPair{AddChain{Memory{WeightedPrimField{T, D, B}}}, OnlyParam{F}}

    function EvalCompositeOrb(weightedFs::AbstractVector{<:WeightedPrimField{T, D}}, 
                              normalizer::OnlyParam{F}) where {T, D, F}
        fInnerObjs = foldl(∘, Base.Fix2.(getfield, (:f, :left))).(weightedFs)
        fInnerType = eltype(foldl(∘, Base.Fix2.(getfield, (:apply, :left ))).(fInnerObjs))
        nInnerType = eltype(foldl(∘, Base.Fix2.(getfield, (:f,     :right))).(fInnerObjs))
        B = EvalPrimitiveOrb{T, D, <:fInnerType, <:nInnerType}
        weightedFieldMem = Memory{WeightedPrimField{T, D, B}}(weightedFs)
        new{T, D, B, F}(PairCombine(*, ChainReduce(+, weightedFieldMem), normalizer))
    end
end

function unpackParamFunc!(f::CompositeOrb{T, D}, paramSet::PBoxAbtArray) where {T, D}
    pSetId = objectid(paramSet)
    fInnerCores = map(Fix2(unpackFunc!, paramSet), f.basis) .|> first
    init = WeightedPrimField{T, D, eltype(fInnerCores)}[]
    weightedFields = mapfoldl(vcat, zip(fInnerCores, f.weight); init) do (i, w)
        parIds = (locateParam!(paramSet, w),)
        PairCombine(*, i, PointerFunc(OnlyParam(itself), parIds, pSetId))
    end
    EvalCompositeOrb(weightedFields, genNormalizer(f)), paramSet
end

getEvaluator(::CompositeOrb) = EvalCompositeOrb

const PrimGTO{T, D, B<:PolyGaussFunc{T, D}, C} = PrimitiveOrb{T, D, B, C}
const CompGTO{T, D, B<:PolyGaussFunc{T, D}, C, P} = CompositeOrb{T, D, B, C, P}

# concretizeBasisType(basis::AbstractArray{<:CompositeOrb})

abstract type UnpackedOrb{T, D, B} <: OrbitalBasis{T, D, B} end

struct FrameworkOrb{T, D, B<:EvalComposedOrb{T, D}, P} <: UnpackedOrb{T, D, B}
    core::B
    param::Memory{P}

    function FrameworkOrb(dob::ComposedOrb{T, D}, 
                          paramSet::PBoxAbtArray=ParamBox[]) where {T, D}
        core, params = unpackFunc!(dob, paramSet)
        typedParams = getMemory(params .|> itself) # first concretize the type
        new{T, D, typeof(core), eltype(typedParams)}(core, typedParams)
    end
end

const EvalPrimGTO{T, D, B<:EvalPolyGaussFunc{T, D}, F} = EvalPrimitiveOrb{T, D, B, F}
const EvalCompGTO{T, D, B<:EvalPrimGTO{T, D}, F} = EvalCompositeOrb{T, D, B, F}

const FPrimGTO{T, D, B<:EvalPrimGTO{T, D}, P} = FrameworkOrb{T, D, B, P}
const FCompGTO{T, D, B<:EvalCompGTO{T, D}, P} = FrameworkOrb{T, D, B, P}


unpackFunc(b::FrameworkOrb) = (b.core, b.param)


function permitRenormalize!(b::OrbitalBasis)
    b.renormalize = true
end


function forbidRenormalize!(b::OrbitalBasis)
    b.renormalize = false
end


# function unpackSquareIntNormalizer(f::GaussTypeOrb)
#     # inv∘sqrt∘genOverlap(f, f) # input: parValSet
#     # Base.Fix2(polyGaussFuncNormSquare, degree)
# end

function genGaussTypeOrb(center::NonEmptyTuple{ParamOrValue{T}, D}, 
                         xpn::ParamOrValue{T}, 
                         ijk::NonEmptyTuple{Int, D}=ntuple(_->0, Val(D)); 
                         renormalize::Bool=false) where {T, D}
    encoder1 = genCellEncoder(T, :xpn)
    encoder2 = genCellEncoder(T, :cen)
    gf = xpn |> encoder1 |> GaussFunc
    PrimitiveOrb(PolyRadialFunc(gf, ijk), encoder2.(center), renormalize)
end

function genGaussTypeOrb(center::NonEmptyTuple{ParamOrValue{T}, D}, 
                         xpns::AbstractVector{<:ParamOrValue{T}}, 
                         cons::AbstractVector{<:ParamOrValue{T}}, 
                         ijk::NonEmptyTuple{Int, D}=ntuple(_->0, Val(D)); 
                         renormalize::Bool=false) where {T, D}
    pgtoNum = checkEmptiness(xpns, :xpns)
    if pgtoNum != length(cons)
        throw(AssertionError("`xpns` and `cons` must have the same length."))
    end
    if pgtoNum == 1
        genGaussTypeOrb(center, xpns[], cons[], ijk; renormalize)
    else
        conParams = genCellEncoder(T, :con).(cons)
        pgtos = genGaussTypeOrb.(Ref(center), xpns, Ref(ijk); renormalize)
        CompositeOrb(pgtos, conParams; renormalize)
    end
end


# struct GaussTypeSubshell{T, D, L, B, O} <: OrbitalBatch{T, D, PolyGaussProd{T, D, L, B}, O}
#     seed::PolyGaussProd{T, D, L, B}
#     subshell::NTuple{O, CartSHarmonics{D, L}}

#     function GaussTypeSubshell(center::NonEmptyTuple{ParamOrValue{T}, D}, 
#                                xpns::AbstractVector{<:ParamOrValue{T}}, 
#                                cons::AbstractVector{<:ParamOrValue{T}}, 
#                                l::Int; 
#                                renormalize::Bool=false)
#         l < 0
#     end
# end