export PrimitiveOrb, CompositeOrb, FrameworkOrb, genGaussTypeOrb

abstract type ComposedOrb{T, D, B} <: OrbitalBasis{T, D, B} end

abstract type EvalComposedOrb{T, D, B} <: EvalDimensionalKernel{T, D, B} end

abstract type ComposedOrbParamPtr{T, D, R} <: FieldParamPointer{R} end

(f::OrbitalBasis)(x) = evalFunc(f, x)

(f::EvalComposedOrb)(input, param) = f.f(input, param)


struct ScaledOrbital{T, D, C<:EvalComposedOrb{T, D}, 
                     F<:Function} <: EvalComposedOrb{T, D, C}
    f::PairCombine{StableMul{T}, C, F}
end

function ScaledOrbital(orb::EvalComposedOrb{T}, scalar::Function) where {T}
    (ScaledOrbital∘PairCombine)(StableBinary(*, T), orb, scalar)
end


function normalizeOrbital(fCore::EvalComposedOrb{T, D}, paramSet::FlatParamSet, 
                          paramPointer::ComposedOrbParamPtr{T, D}) where {T, D}
    normalizerCore = genNormalizer(fCore, paramPointer)
    ScaledOrbital( fCore, ReturnTyped(normalizerCore, T) )
end

const NormFuncType{T} = Union{ReturnTyped{T, <:ParamSelectFunc}, Storage{T}}


function unpackParamFunc!(f::ComposedOrb{T}, paramSet::FlatParamSet) where {T}
    fEvalCore, _, paramPointer = unpackParamFuncCore!(f, paramSet)
    fEval = if f.renormalize
        normalizeOrbital(fEvalCore, paramSet, paramPointer)
    else
        ScaledOrbital(fEvalCore, (Storage∘one)(T))
    end
    fEval, paramSet, paramPointer
end

#! Change .center to the first field
struct PrimitiveOrb{T, D, B<:FieldAmplitude{T, D}, 
                    C<:NTuple{D, ElementalParam{T}}} <: ComposedOrb{T, D, B}
    body::B
    center::C
    renormalize::Bool
end

const PrimGTO{T, D, B<:PolyGaussProd{T, D}, C<:NTuple{D, ElementalParam{T}}} = 
      PrimitiveOrb{T, D, B, C}

function PrimitiveOrb(body::B, center::NTuple{D, ParamOrValue{T}}; 
                      renormalize::Bool=false) where {T, D, B<:FieldAmplitude{T, D}}
    length(center)!=D && throw(AssertionError("The length of `center` must match `D=$D`."))
    encoder = genCellEncoder(T, :cen)
    PrimitiveOrb(body, encoder.(center), renormalize)
end

PrimitiveOrb(ob::PrimitiveOrb) = itself(ob)

const OrbShifter{T, D} = ParamSelectFunc{ShiftByArg{T, D}, NTuple{ D, FlatPSetInnerPtr{T} }}

struct PrimitiveOrbCore{T, D, B<:EvalFieldAmp{T, D}} <: EvalComposedOrb{T, D, B}
    f::FlatPSetFilterFunc{T, InsertInward{B, OrbShifter{T, D}}}
end

const EvalPrimOrb{T, D, B, F<:NormFuncType{T}} = 
      ScaledOrbital{T, D, PrimitiveOrbCore{T, D, B}, F}

const PrimGTOcore{T, D, B<:EvalPolyGaussProd{T, D}} = 
      PrimitiveOrbCore{T, D, B}

const EvalPrimGTO{T, D, B<:EvalPolyGaussProd{T, D}, F<:NormFuncType{T}} = 
      EvalPrimOrb{T, D, B, F}

struct PrimOrbParamPtr{T, D, R<:FieldPtrDict{T}} <: ComposedOrbParamPtr{T, D, R}
    center::NTuple{D, FlatPSetInnerPtr{T}}
    body::MixedFieldParamPointer{T, R}
    id::Identifier

    function PrimOrbParamPtr(centerParamPtrs::NonEmptyTuple{FlatPSetInnerPtr{T}, D}, 
                             bodyParamPairs::FieldPtrPairs{T}, 
                             paramSet::FlatParamSet{T}) where {T, D}
        bodyPtr = MixedFieldParamPointer(bodyParamPairs, paramSet)
        new{T, D+1, typeof(bodyPtr.core)}(centerParamPtrs, bodyPtr, Identifier(paramSet))
    end
end

#= Additional Method =#
function getPointerPairs(paramPtr::PrimOrbParamPtr)
    vcat(paramPtr.center..., getPointerPairs(paramPtr.body))
end

function unpackParamFuncCore!(f::PrimitiveOrb{T, D}, paramSet::FlatParamSet) where {T, D}
    pSetLocal = initializeParamSet(FlatParamSet, T)
    cenPtrInner = locateParam!(pSetLocal, f.center)
    fEvalCore, _, bodyPairs = unpackParamFuncCore!(f.body, pSetLocal)
    pFilter = locateParam!(paramSet, pSetLocal)
    shifter = ParamSelectFunc(ShiftByArg{T, D}(), cenPtrInner)
    cenPtrs = linkPointer.(Ref(pFilter), cenPtrInner)
    fEval = ParamFilterFunc(InsertInward(fEvalCore, shifter), DelayedPointer(pFilter))
    paramPtr = PrimOrbParamPtr(cenPtrs, bodyPairs, paramSet)
    PrimitiveOrbCore(fEval), paramSet, paramPtr
end


struct CompositeOrb{T, D, B<:FieldAmplitude{T, D}, C<:NTuple{D, ElementalParam{T}}, 
                    W<:FlattenedParam{T, 1}} <: ComposedOrb{T, D, B}
    basis::Memory{PrimitiveOrb{T, D, <:B, <:C}}
    weight::W
    renormalize::Bool

    function CompositeOrb(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                          weight::FlattenedParam{T, 1}, 
                          renormalize::Bool=false) where {T, D}
        basis, weight = getWeightParam(basis, weight)
        B = getfield.(basis, :body) |> eltype
        C = getfield.(basis, :center) |> eltype
        basisMem = Memory{PrimitiveOrb{T, D, <:B, <:C}}(basis)
        new{T, D, B, C, typeof(weight)}(basisMem, weight, renormalize)
    end
end

const CompGTO{T, D, B<:PolyGaussProd{T, D}, C<:NTuple{D, ElementalParam{T}}, 
              W<:FlattenedParam{T, 1}} = CompositeOrb{T, D, B, C, W}

function getWeightParam(basis::AbstractVector{<:PrimitiveOrb{T, D}}, 
                        weight::W) where {T, D, W<:FlattenedParam{T, 1}}
    getWeightParamCore(basis, weight)
end

function getWeightParam(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                        weight::W) where {T, D, W<:FlattenedParam{T, 1}}
    getWeightParamCore(itself.(basis), weight)
end

function getWeightParamCore(basis::AbstractVector{<:PrimitiveOrb{T, D}}, 
                            weight::W) where {T, D, W<:FlattenedParam{T, 1}}
    if checkEmptiness(basis, :basis) != first(outputSizeOf(weight))
        throw(AssertionError("`basis` and `weight` must have the same length."))
    end
    basis, weight
end

function getWeightParamCore(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                            weight::W) where {T, D, W<:FlattenedParam{T, 1}}
    if checkEmptiness(basis, :basis) != first(outputSizeOf(weight))
        throw(AssertionError("`basis` and `weight` must have the same length."))
    end
    weightPieces = getEffectiveWeight(basis, weight)
    expandedWeight = ParamGrid(weightPieces, :weight)
    expandedBasis = mapfoldl(decomposeOrb, vcat, basis)
    expandedBasis, expandedWeight
end

decomposeOrb(b::T) where {T<:PrimitiveOrb} = Memory{T}([b])
decomposeOrb(b::CompositeOrb) = b.basis


function getEffectiveWeight(::PrimitiveOrb{T}, weight::FlattenedParam{T, 1}, 
                            idx::Int) where {T}
    [indexParam(weight, idx)]
end

function getEffectiveWeight(o::CompositeOrb{T}, weight::FlattenedParam{T, 1}, 
                            idx::Int) where {T}
    len = first(outputSizeOf(o.weight))
    outWeight = indexParam(weight, idx)
    map(i->CellParam(StableBinary(*, T), indexParam(o.weight, i), outWeight, :w), 1:len)
end

function getEffectiveWeight(o::AbstractVector{<:ComposedOrb{T, D}}, 
                            weight::FlattenedParam{T, 1}) where {T, D}
    mapreduce(vcat, enumerate(o)) do (i, x)
        getEffectiveWeight(x, weight, i)
    end
end

function CompositeOrb(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                      weight::ParOrValVec{T}; renormalize::Bool=false) where {T, D}
    weightParams = ParamGrid(genCellEncoder(T, :w).(weight), :wBlock)
    CompositeOrb(basis, weightParams, renormalize)
end

CompositeOrb(ob::CompositeOrb) = itself(ob)

const GetPFWeightEntry{T} = GetParamFunc{OnlyBody{GetIdxField{T}}, IndexPointer{Volume{T}}}

const WeightedPF{T, D, U<:EvalPrimOrb{T, D}} = ScaledOrbital{T, D, U, GetPFWeightEntry{T}}

function compressWeightedPF(::Type{B}, ::Type{F}) where {T, D, B<:EvalFieldAmp{T, D}, F}
    boolF = isconcretetype(B)
    boolN = isconcretetype(F)
    U = if boolF
        if boolN
            EvalPrimOrb{T, D, B, F}
        else
            EvalPrimOrb{T, D, B, <:F}
        end
    elseif boolN
        EvalPrimOrb{T, D, <:B, F}
    else
        EvalPrimOrb{T, D, <:B, <:F}
    end
    if boolF && boolN
        Memory{WeightedPF{T, D, U}}
    else
        Memory{WeightedPF{T, D, <:U}}
    end
end

struct CompositeOrbCore{T, D, V<:Memory{<:WeightedPF{T, D}}} <: EvalComposedOrb{T, D, V}
    f::FlatPSetFilterFunc{T, ChainReduce{StableAdd{T}, V}}
end

const EvalCompOrb{T, D, B, F<:NormFuncType{T}} = 
      ScaledOrbital{T, D, CompositeOrbCore{T, D, B}, F}

const EvalCompGTO{T, D, U<:EvalPrimGTO{T, D}, F<:NormFuncType{T}} = 
      EvalCompOrb{T, D, <:Memory{<:WeightedPF{T, D, <:U}}, F}

function restrainEvalOrbType(weightedFs::AbstractVector{<:WeightedPF{T, D}}) where {T, D}
    cPtr1 = ChainPointer((:f, :left, :f))
    fInnerObjs = getField.(weightedFs, Ref(cPtr1))
    cPtr2 = ChainPointer((:left, :f, :apply, :apply))
    fInnerType = eltype( getField.(fInnerObjs, Ref(cPtr2)) )
    nInnerType = eltype( getfield.(fInnerObjs, :right) )
    V = compressWeightedPF(fInnerType, nInnerType)
    ChainReduce(StableBinary(+, T), V(weightedFs))
end

struct CompOrbParamPtr{T, D, R<:FieldPtrDict{T}, 
                       P<:PrimOrbParamPtr{T, D, <:R}} <: ComposedOrbParamPtr{T, D, R}
    basis::Memory{P}
    weight::IndexPointer{Volume{T}}
    id::Identifier

    CompOrbParamPtr{R}(basis::Memory{P}, weight::ChainPointer, id::Identifier) where 
                      {T, D, R<:FieldPtrDict{T}, P<:PrimOrbParamPtr{T, D, <:R}} = 
    new{T, D, R, P}(basis, weight, id)
end

function unpackParamFuncCore!(f::CompositeOrb{T, D}, paramSet::FlatParamSet) where {T, D}
    pSetLocal = initializeParamSet(FlatParamSet, T)
    weightedFields = WeightedPF{T, D, <:EvalPrimOrb{T, D}}[]
    weightLocalPtr = locateParam!(pSetLocal, f.weight)
    i = firstindex(f.basis) - 1
    innerPairGroups = map(f.basis) do b
        i += 1
        fInnerCore, _, basisPtr = unpackParamFunc!(b, pSetLocal)
        getIdx = (OnlyBody∘getField)(ChainPointer( i, TensorType(T) ))
        weight = ParamSelectFunc(getIdx, (weightLocalPtr,))
        push!(weightedFields, ScaledOrbital(fInnerCore, weight))
        (basisPtr.center, collect(basisPtr.body.core))
    end
    pFilter = locateParam!(paramSet, pSetLocal)
    weightPtr = linkPointer(pFilter, weightLocalPtr)
    basisPtr = map(innerPairGroups) do (cenPtr, bodyPairs)
        bodyPairs = anchorRight(bodyPairs, pFilter)
        cenPtr = linkPointer.(Ref(pFilter), cenPtr)
        PrimOrbParamPtr(cenPtr, bodyPairs, paramSet)
    end
    innerDictType = eltype(getField.( basisPtr, Ref(ChainPointer( (:body, :core) )) ))
    paramPtr = CompOrbParamPtr{innerDictType}(basisPtr, weightPtr, Identifier(paramSet))
    fEval = ParamFilterFunc(restrainEvalOrbType(weightedFields), DelayedPointer(pFilter))
    CompositeOrbCore(fEval), paramSet, paramPtr
end


abstract type UnpackedOrb{T, D, B} <: OrbitalBasis{T, D, B} end

struct FrameworkOrb{T, D, B<:EvalComposedOrb{T, D}, P<:FlatParamSet{T}, 
                    A<:FieldParamPointer} <: UnpackedOrb{T, D, B}
    core::B
    param::P
    pointer::A

    function FrameworkOrb(o::ComposedOrb{T, D}, 
                          paramSet::Union{FlatParamSet{T}, Missing}=missing) where {T, D}
        bl = ismissing(paramSet)
        bl && (paramSet = initializeParamSet(FlatParamSet, T))
        core, _, ptr = unpackParamFunc!(o, paramSet)
        d0 = paramSet.d0
        d1 = paramSet.d1
        isempty(paramSet.d0) || (d0 = itself.(d0))
        isempty(paramSet.d1) || (d1 = itself.(d1))
        bl && (paramSet = FlatParamSet{T}(d0, d1))
        new{T, D, typeof(core), typeof(paramSet), typeof(ptr)}(core, paramSet, ptr)
    end
end

const FPrimGTO{T, D, B<:EvalPrimGTO{T, D}, P<:FlatParamSet{T}, A<:FieldParamPointer} = 
      FrameworkOrb{T, D, B, P, A}

const FCompGTO{T, D, B<:EvalCompGTO{T, D}, P<:FlatParamSet{T}, A<:FieldParamPointer} = 
      FrameworkOrb{T, D, B, P, A}


unpackFunc(b::FrameworkOrb) = (b.core, b.param)


function permitRenormalize!(b::OrbitalBasis)
    b.renormalize = true
end


function forbidRenormalize!(b::OrbitalBasis)
    b.renormalize = false
end


function genGaussTypeOrb(center::NonEmptyTuple{ParamOrValue{T}, D}, 
                         xpn::ParamOrValue{T}, 
                         ijk::NonEmptyTuple{Int, D}=ntuple(_->0, Val(D)); 
                         renormalize::Bool=false) where {T, D}
    gf = GaussFunc(xpn)
    PrimitiveOrb(PolyRadialFunc(gf, ijk), center; renormalize)
end

function genGaussTypeOrb(center::NonEmptyTuple{ParamOrValue{T}, D}, 
                         xpns::ParOrValVec{T}, 
                         cons::Union{ParOrValVec{T}, FlattenedParam{T, 1}}, 
                         ijk::NonEmptyTuple{Int, D}=ntuple(_->0, Val(D)); 
                         renormalize::Bool=false) where {T, D}
    consLen = if cons isa FlattenedParam
        len = (first∘outputSizeOf)(cons)
        len < 2 && 
        throw(AssertionError("The length of `cons::FlattenedParam` should be at least 2."))
        len
    else
        cons = genCellEncoder(T, :con).(cons)
        length(cons)
    end

    if consLen != checkEmptiness(xpns, :xpns)
        throw(AssertionError("`xpns` and `cons` must have the same length."))
    end

    cenParams = genCellEncoder(T, :cen).(center)

    if consLen == 1
        genGaussTypeOrb(cenParams, xpns[], cons[], ijk; renormalize)
    else
        pgtos = genGaussTypeOrb.(Ref(cenParams), xpns, Ref(ijk); renormalize)
        CompositeOrb(pgtos, cons; renormalize)
    end
end


# struct GaussTypeSubshell{T, D, L, B, O} <: OrbitalBatch{T, D, PolyGaussProd{T, D, L, B}, O}
#     seed::PolyGaussProd{T, D, L, B}
#     subshell::NTuple{O, CartSHarmonics{D, L}}

#     function GaussTypeSubshell(center::NonEmptyTuple{ParamOrValue{T}, D}, 
#                                xpns::ParOrValVec{T}, 
#                                cons::ParOrValVec{T}, 
#                                l::Int; 
#                                renormalize::Bool=false)
#         l < 0
#     end
# end