export PrimitiveOrb, CompositeOrb, ComponentOrb, genGaussTypeOrb

abstract type ComposedOrb{T, D, B} <: OrbitalBasis{T, D, B} end

abstract type UnpackedOrb{T, D, B} <: OrbitalBasis{T, D, B} end

abstract type EvalComposedOrb{T, D, B} <: EvalFieldFunction{T, D, B} end

abstract type ComposedOrbParamPtr{D, R} <: FieldParamPointer{R} end

(f::OrbitalBasis)(x) = evalFunc(f, x)

(f::EvalComposedOrb)(input, params) = f.f(input, params)


struct NormalizePrimOrb{T, D, F<:OrbitalIntegrator{T, D}} <: OrbitalNormalizer{T, D}
    core::F
end

(f::NormalizePrimOrb)(params::AbstractSpanValueSet) = (AbsSqrtInv∘f.core)(params)


const VariedNormCore{T, D, N, F<:OrbitalIntegrator{T, D}} = 
      ReturnTyped{T, EncodeApply{N, NTuple{N, SpanSetFilter}, F}}

const UnitScalarCore{T} = ReturnTyped{T, Unit{T}}

struct NormalizeCompOrb{T, D, F1<:Union{UnitScalarCore{T}, VariedNormCore{T, D, 1}}, 
                        F2<:VariedNormCore{T, D, 2}} <: OrbitalNormalizer{T, D}
    core::HermitianContract{T, F1, F2}
    weight::Memory{ReturnTyped{T, GetGridEntry}}
end

function (f::NormalizeCompOrb)(input::AbstractSpanValueSet)
    weightVals = map(f->f(input), f.weight)
    res = f.core(input, weightVals)
    AbsSqrtInv(res)
end

const OrbNormalizerCore{T, D} = 
      Union{NormalizePrimOrb{T, D}, NormalizeCompOrb{T, D}, Power{Unit{T}, D}}

struct EvalOrbNormalizer{T, D, F<:OrbNormalizerCore{T, D}} <: DimensionalEvaluator{T, D, F}
    f::ReturnTyped{T, F}
end

EvalOrbNormalizer(f::Function, ::Type{T}) where {T} = 
(EvalOrbNormalizer∘ReturnTyped)(f, T)

EvalOrbNormalizer(::Type{T}, ::Val{D}) where {T, D} = 
EvalOrbNormalizer(Power(Unit(T), Val(D)), T)

(f::EvalOrbNormalizer)(params) = f.f(params)


function unpackParamFunc!(f::ComposedOrb{T, D}, paramSet::AbstractSpanParamSet, 
                          paramSetId::Identifier=Identifier(paramSet)) where {T, D}
    fEvalCore, _, paramPointer = unpackComposedOrbCore!(f, paramSet, paramSetId)
    normalizer = f.renormalize ? buildNormalizer(fEvalCore) : buildNormalizer(T, Val(D))
    fEval = ScaledOrbital(fEvalCore, OnlyBody(normalizer), paramPointer.scope)
    fEval, paramSet, paramPointer
end


const ParamSubsetApply{F} = ParamSelectFunc{F, Tuple{ SpanSetFilter }}

struct ScaledOrbital{T, D, C<:EvalComposedOrb{T, D}, 
                     F<:EvalOrbNormalizer{T, D}} <: EvalComposedOrb{T, D, C}
    f::ParamSubsetApply{PairCombine{ StableMul{T}, C, OnlyBody{F} }}
end

function ScaledOrbital(orb::EvalComposedOrb{T}, scalar::Function, 
                       scope::SpanSetFilter) where {T}
    fCoreLocal = PairCombine(StableMul(T), orb, scalar)
    fCore = ParamSelectFunc(fCoreLocal, scope)
    ScaledOrbital(fCore)
end

#! Change C<:NTuple{D, UnitParam{T}} to C<:NTuple{D, UnitParam{Real}}
mutable struct PrimitiveOrb{T, D, B<:FieldAmplitude{T, D}, 
                            C<:NTuple{D, UnitParam{T}}} <: ComposedOrb{T, D, B}
    const center::C
    const body::B
    @atomic renormalize::Bool
end

const PrimGTO{T, D, B<:PolyGaussProd{T, D}, C<:NTuple{D, UnitParam{T}}} = 
      PrimitiveOrb{T, D, B, C}

function PrimitiveOrb(center::NTuple{D, UnitOrVal{T}}, body::B; 
                      renormalize::Bool=false) where {T, D, B<:FieldAmplitude{T, D}}
    length(center)!=D && throw(AssertionError("The length of `center` must match `D=$D`."))
    encoder = UnitParamEncoder(T, :cen, 1)
    PrimitiveOrb(encoder.(center), body, renormalize)
end

PrimitiveOrb(ob::PrimitiveOrb) = itself(ob)

const OrbShifter{T, D} = ParamSelectFunc{ShiftByArg{T, D}, NTuple{ D, GetIndex{UnitIndex} }}

struct PrimitiveOrbCore{T, D, B<:EvalFieldAmp{T, D}} <: EvalComposedOrb{T, D, B}
    f::InsertInward{B, OrbShifter{T, D}}
end

const EvalPrimOrb{T, D, B, F<:EvalOrbNormalizer{T, D}} = 
      ScaledOrbital{T, D, PrimitiveOrbCore{T, D, B}, F}

const PrimATOcore{T, D, B<:EvalPolyGaussProd{T, D}} = 
      PrimitiveOrbCore{T, D, B}

const PrimGTOcore{T, D, B<:EvalPolyGaussProd{T, D}} = 
      PrimitiveOrbCore{T, D, B}

const TypedPrimGTOcore{T, D, L} = 
      PrimitiveOrbCore{T, D, EvalPolyRadialFunc{T, D, EvalGaussFunc{T}, L}}

const EvalPrimGTO{T, D, B<:EvalPolyGaussProd{T, D}, F<:EvalOrbNormalizer{T, D}} = 
      EvalPrimOrb{T, D, B, F}

struct PrimOrbParamPtr{D, R<:FieldParamPtrDict} <: ComposedOrbParamPtr{D, R}
    center::NTuple{D, GetIndex{UnitIndex}}
    body::MixedFieldParamPointer{R}
    scope::SpanSetFilter
    tag::Identifier

    function PrimOrbParamPtr(centerParamPtrs::NonEmptyTuple{GetIndex{UnitIndex}, D}, 
                             bodyParamPairs::FieldParamPtrPairs, 
                             scope::SpanSetFilter, 
                             tag::Identifier=Identifier(missing)) where {D}
        bodyPtr = MixedFieldParamPointer(bodyParamPairs, tag)
        new{D+1, typeof(bodyPtr.core)}(centerParamPtrs, bodyPtr, scope, tag)
    end

    function PrimOrbParamPtr(ptr::PrimOrbParamPtr{D, R}, 
                             parentScope::SpanSetFilter, 
                             tag::Identifier=ptr.tag) where {D, R<:FieldParamPtrDict}
        newScope = getField(parentScope, ptr.scope)
        new{D, R}(ptr.center, ptr.body, newScope, tag)
    end
end

function unpackComposedOrbCore!(f::PrimitiveOrb{T, D}, paramSet::AbstractSpanParamSet, 
                                paramSetId::Identifier=Identifier(paramSet)) where {T, D}
    pSetLocal = initializeSpanParamSet()
    cenPtrs = locateParam!(pSetLocal, f.center)
    fEvalCore, _, bodyPairs = unpackParamFuncCore!(f.body, pSetLocal)
    pFilter = locateParam!(paramSet, pSetLocal)
    shifter = ParamSelectFunc(ShiftByArg{T, D}(), cenPtrs)
    paramPtr = PrimOrbParamPtr(cenPtrs, bodyPairs, pFilter, paramSetId)
    PrimitiveOrbCore(InsertInward(fEvalCore, shifter)), paramSet, paramPtr
end


mutable struct CompositeOrb{T, D, B<:FieldAmplitude{T, D}, C<:NTuple{D, UnitParam{T}}, 
                            W<:GridParam{T, 1}} <: ComposedOrb{T, D, B}
    const basis::Memory{PrimitiveOrb{T, D, <:B, <:C}}
    const weight::W
    @atomic renormalize::Bool

    function CompositeOrb(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                          weight::GridParam{T, 1}, 
                          renormalize::Bool=false) where {T, D}
        basis, weight = getWeightParam(basis, weight)
        B = getfield.(basis, :body) |> eltype
        C = getfield.(basis, :center) |> eltype
        basisMem = Memory{PrimitiveOrb{T, D, <:B, <:C}}(basis)
        new{T, D, B, C, typeof(weight)}(basisMem, weight, renormalize)
    end
end

const CompGTO{T, D, B<:PolyGaussProd{T, D}, C<:NTuple{D, UnitParam{T}}, 
              W<:GridParam{T, 1}} = CompositeOrb{T, D, B, C, W}

function getWeightParam(basis::AbstractVector{<:PrimitiveOrb{T, D}}, 
                        weight::W) where {T, D, W<:GridParam{T, 1}}
    getWeightParamCore(basis, weight)
end

function getWeightParam(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                        weight::W) where {T, D, W<:GridParam{T, 1}}
    getWeightParamCore(itself.(basis), weight)
end

function getWeightParamCore(basis::AbstractVector{<:PrimitiveOrb{T, D}}, 
                            weight::W) where {T, D, W<:GridParam{T, 1}}
    checkLengthCore(checkEmptiness(basis, :basis), :basis, (first∘getOutputSize)(weight), 
                    "the output length of `weight`")
    basis, weight
end

function getWeightParamCore(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                            weight::W) where {T, D, W<:GridParam{T, 1}}
    checkLengthCore(checkEmptiness(basis, :basis), :basis, (first∘getOutputSize)(weight), 
                    "the output length of `weight`")
    weightPieces = getEffectiveWeight(basis, weight)
    expandedWeight = genHeapParam(weightPieces, :weight)
    expandedBasis = mapfoldl(splitOrb, vcat, basis)
    expandedBasis, expandedWeight
end


function getEffectiveWeight(::PrimitiveOrb{T}, weight::GridParam{T, 1}, idx::Int) where {T}
    [indexParam(weight, idx)]
end

function getEffectiveWeight(o::CompositeOrb{T}, weight::GridParam{T, 1}, idx::Int) where {T}
    o.renormalize && throw(AssertionError("Merging the weight from a renormalized "*
                           "`CompositeOrb` with another value is prohibited."))
    len = first(getOutputSize(o.weight))
    outWeight = indexParam(weight, idx)
    map(i->genCellParam(StableMul(T), (indexParam(o.weight, i), outWeight), :w), 1:len)
end

function getEffectiveWeight(o::AbstractVector{<:ComposedOrb{T, D}}, 
                            weight::GridParam{T, 1}) where {T, D}
    mapreduce(vcat, enumerate(o)) do (i, x)
        getEffectiveWeight(x, weight, i)
    end
end

function CompositeOrb(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                      weight::UnitOrValVec{T}; renormalize::Bool=false) where {T, D}
    weightParams = genHeapParam(UnitParamEncoder(T, :w, 1).(weight), :wBlock)
    CompositeOrb(basis, weightParams, renormalize)
end

CompositeOrb(ob::CompositeOrb) = itself(ob)

const WeightedPF{T, D, U<:EvalPrimOrb{T, D}} = 
      PairCombine{StableMul{T}, U, OnlyBody{GetGridEntry}}

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
    f::ChainReduce{StableAdd{T}, V}
end

const EvalCompOrb{T, D, B, F<:EvalOrbNormalizer{T, D}} = 
      ScaledOrbital{T, D, CompositeOrbCore{T, D, B}, F}

const EvalCompGTO{T, D, U<:EvalPrimGTO{T, D}, F<:EvalOrbNormalizer{T, D}} = 
      EvalCompOrb{T, D, <:Memory{<:WeightedPF{T, D, <:U}}, F}

function restrainEvalOrbType(weightedFs::AbstractVector{<:WeightedPF{T, D}}) where {T, D}
    cPtr = ChainedAccess((:left, :f, :apply))
    fInnerObjs = map(f->getField(f, cPtr), weightedFs)
    # fInnerType = (eltype∘map)(f->getField(f, cPtr), fInnerObjs)
    # nInnerType = (eltype∘map)(f->getField(f, ChainedAccess( (:right, :f) )), fInnerObjs)
    fInnerType = mapreduce(f->(typeof∘getField)(f, cPtr), typejoin, fInnerObjs)
    nPtr = ChainedAccess((:right, :f))
    nInnerType = mapreduce(f->(typeof∘getField)(f, nPtr), typejoin, fInnerObjs)
    V = compressWeightedPF(fInnerType, nInnerType)
    ChainReduce(StableAdd(T), V(weightedFs))
end

struct CompOrbParamPtr{D, R<:FieldParamPtrDict, P<:PrimOrbParamPtr{D, <:R}
                       } <: ComposedOrbParamPtr{D, R}
    basis::Memory{P}
    weight::GetIndex{GridIndex}
    scope::SpanSetFilter
    tag::Identifier

    CompOrbParamPtr{R}(basis::Memory{P}, weight::ChainedAccess, 
                       scope::SpanSetFilter, 
                       tag::Identifier=Identifier(missing)) where 
                      {D, R<:FieldParamPtrDict, P<:PrimOrbParamPtr{D, <:R}} = 
    new{D, R, P}(basis, weight, scope, tag)
end

function unpackComposedOrbCore!(f::CompositeOrb{T, D}, paramSet::AbstractSpanParamSet, 
                                paramSetId::Identifier=Identifier(paramSet)) where {T, D}
    pSetLocal = initializeSpanParamSet()
    weightedFields = WeightedPF{T, D, <:EvalPrimOrb{T, D}}[]
    getWeight = locateParam!(pSetLocal, f.weight)
    i = 0
    basisPtrs = map(f.basis) do b
        i += 1
        fInnerCore, _, basisPtr = unpackParamFunc!(b, pSetLocal, paramSetId)
        getWeightComp = ChainedAccess(getWeight, OneToIndex(i))
        weightedPrimOrb = PairCombine(StableMul(T), fInnerCore, OnlyBody(getWeightComp))
        push!(weightedFields, weightedPrimOrb)
        basisPtr
    end
    pFilter = locateParam!(paramSet, pSetLocal)
    innerDictType = eltype(getField.( basisPtrs, Ref(ChainedAccess( (:body, :core) )) ))
    paramPtr = CompOrbParamPtr{innerDictType}(basisPtrs, getWeight, pFilter, paramSetId)
    CompositeOrbCore(restrainEvalOrbType(weightedFields)), paramSet, paramPtr
end


#! Consider paramSet::FilteredFlatParamSet as an input argument.
#! You cannot. Because `locateParam!` might change upstream (input) paramSet's size by 
#! pushing new parameters while `FilteredFlatParamSet` cannot change its size.
struct ComponentOrb{T, D, B<:EvalComposedOrb{T, D}, P<:AbstractSpanParamSet, 
                    A<:FieldParamPointer} <: UnpackedOrb{T, D, B}
    core::B
    param::P
    pointer::A

    function ComponentOrb(o::ComposedOrb{T, D}, 
                          paramSet::MissingOr{AbstractSpanParamSet}=missing) where {T, D}
        id = Identifier(paramSet)
        bl = ismissing(paramSet)
        bl && (paramSet = initializeSpanParamSet())
        core, _, ptr = unpackParamFunc!(o, paramSet, id)
        bl && (paramSet = map(getMemory, paramSet))
        new{T, D, typeof(core), typeof(paramSet), typeof(ptr)}(core, paramSet, ptr)
    end

    function ComponentOrb(o::ComponentOrb{T, D, <:EvalCompOrb{T, D}, P}, 
                          idx::Int) where {T, D, P<:AbstractSpanParamSet}
        oParams = o.param
        oPointer = o.pointer
        oCore = (getInnerOrb∘getInnerOrb)(o)
        subOrb = oCore.f.chain[idx].left
        subPtr = PrimOrbParamPtr(oPointer.basis[idx], oPointer.scope, oPointer.tag)
        new{T, D, typeof(subOrb), P, typeof(subPtr)}(subOrb, oParams, subPtr)
    end
end

ComponentOrb(o::ComponentOrb) = itself(o)

const FPrimOrb{T, D, B<:EvalPrimOrb{T, D}, P<:AbstractSpanParamSet, A<:FieldParamPointer} = 
      ComponentOrb{T, D, B, P, A}

const FCompOrb{T, D, B<:EvalCompOrb{T, D}, P<:AbstractSpanParamSet, A<:FieldParamPointer} = 
      ComponentOrb{T, D, B, P, A}

const FPrimGTO{T, D, B<:EvalPrimGTO{T, D}, P<:AbstractSpanParamSet, A<:FieldParamPointer} = 
      FPrimOrb{T, D, B, P, A}

const FCompGTO{T, D, B<:EvalCompGTO{T, D}, P<:AbstractSpanParamSet, A<:FieldParamPointer} = 
      FCompOrb{T, D, B, P, A}

unpackFunc(o::ComponentOrb) = (o.core, o.param, o.pointer)


getInnerOrb(o::ComponentOrb) = o.core

getInnerOrb(o::ScaledOrbital) = o.f.apply.left

getInnerOrb(o::PrimitiveOrbCore) = itself(o)


orbSizeOf(::PrimitiveOrb) = 1

orbSizeOf(o::CompositeOrb) = length(o.basis)

orbSizeOf(::FPrimOrb) = 1

orbSizeOf(o::FCompOrb) = length((getInnerOrb∘getInnerOrb)(o).f.chain)


viewOrb(o::CompositeOrb, idx::Int) = o.basis[begin+idx-1]

viewOrb(o::FCompOrb, idx::Int) = ComponentOrb(o, idx)

viewOrb(o::EvalCompOrb, idx::Int) = getInnerOrb(o).f.chain[begin+idx-1]

function viewOrb(o::Union{PrimitiveOrb, FPrimOrb, EvalPrimOrb}, idx::Int)
    idx == 1 ? itself(o) : throw(BoundsError(o, idx))
end


splitOrb(o::Union{PrimitiveOrb, FPrimOrb}) = getMemory(o)

splitOrb(o::CompositeOrb) = o.basis

function splitOrb(o::FCompOrb)
    map(getMemory( 1:orbSizeOf(o) )) do i
        viewOrb(o, i)
    end
end


function getOrbWeight(::Union{PrimitiveOrb{T}, FPrimOrb{T}}) where {T}
    one(T)
end

function getOrbWeight(::Union{CompositeOrb{T}, FCompOrb{T}}) where {T}
    getOrbWeightCore(o) |> obtain
end

function getOrbWeightCore(o::CompositeOrb{T}) where {T}
    o.weight
end

function getOrbWeightCore(o::FCompOrb{T}) where {T}
    getField(o.param, o.pointer.weight)
end


function enforceRenormalize!(b::ComposedOrb)
    @atomic b.renormalize = true
end


function preventRenormalize!(b::ComposedOrb)
    @atomic b.renormalize = false
end


function isRenormalized(orb::ComposedOrb)
    orb.renormalize
end

function isRenormalized(orb::ComponentOrb)
    isRenormalized(orb.core)
end

function isRenormalized(orb::Union{EvalPrimOrb, EvalCompOrb})
    isRenormalizedCore(orb.f.apply.right.f)
end

function isRenormalizedCore(nf::EvalOrbNormalizer)
    isRenormalizedCore(nf.f.f)
end

function isRenormalizedCore(::Union{NormalizePrimOrb, NormalizeCompOrb})
    true
end

function isRenormalizedCore(::Power{<:Unit})
    false
end


function genGaussTypeOrb(center::NonEmptyTuple{UnitOrVal{T}, D}, 
                         xpn::UnitOrVal{T}, 
                         ijk::NonEmptyTuple{Int, D}=ntuple(_->0, Val(D+1)); 
                         renormalize::Bool=false) where {T, D}
    gf = GaussFunc(xpn)
    PrimitiveOrb(center, PolyRadialFunc(gf, ijk); renormalize)
end

function genGaussTypeOrb(center::NonEmptyTuple{UnitOrVal{T}, D}, 
                         xpns::UnitOrValVec{T}, 
                         cons::Union{UnitOrValVec{T}, GridParam{T, 1}}, 
                         ijk::NonEmptyTuple{Int, D}=ntuple(_->0, Val(D+1)); 
                         innerRenormalize::Bool=false, 
                         outerRenormalize::Bool=false) where {T, D}
    nPrimOrbs = if cons isa GridParam
        (first∘getOutputSize)(cons)
    else
        cons = map(UnitParamEncoder(T, :con, 1), cons)
        length(cons)
    end

    checkLengthCore(checkEmptiness(xpns, :xpns), :xpns, nPrimOrbs, 
                    "the output length of `cons`")

    cens = map(UnitParamEncoder(T, :cen, 1), center)

    primGTOs = map(xpns) do xpn
        genGaussTypeOrb(cens, xpn, ijk, renormalize=innerRenormalize)
    end
    CompositeOrb(primGTOs, cons, renormalize=outerRenormalize)
end


function buildNormalizer(f::PrimitiveOrbCore{T}) where {T}
    nfCore = (NormalizePrimOrb∘buildNormalizerCore)(f)
    EvalOrbNormalizer(nfCore, T)
end

function buildNormalizer(f::CompositeOrbCore{T}) where {T}
    nfCore = NormalizeCompOrb(f)
    EvalOrbNormalizer(nfCore, T)
end

function buildNormalizer(::Type{T}, ::Val{D}) where {T, D}
    EvalOrbNormalizer(T, Val(D))
end


function getNormalizerCoreTypeUnion(::Type{T}, ::Val{D}, ::Val{N}, ::Type{F}) where 
                                   {T, D, N, F<:OrbitalIntegrator{T, D}}
    if isconcretetype(F)
        VariedNormCore{T, D, N, F}
    else
        VariedNormCore{T, D, N, <:F}
    end
end

function NormalizeCompOrb(f::CompositeOrbCore{T, D}) where {T, D}
    weightedOrbs = f.f.chain
    nOrbs = length(weightedOrbs)
    oCorePtr = ChainedAccess((:left, :f, :apply, :left))
    oScopePtr = ChainedAccess((:left, :f, :select))
    nIntStyle = OneBodyIntegral{D}()

    hasUnit = false
    diagFuncCoreType = Union{}
    utriFuncCoreType = Union{}

    diagFuncs = map(weightedOrbs) do weightedOrb
        if isRenormalized(weightedOrb.left)
            hasUnit = true
            ReturnTyped(Unit(T), T)
        else
            oCore = getField(weightedOrb, oCorePtr)
            oFilter = getField(weightedOrb, oScopePtr)
            nfCore = buildCoreIntegrator(nIntStyle, Identity(), (oCore,))
            diagFuncCoreType = typejoin(diagFuncCoreType, typeof(nfCore))
            ReturnTyped(EncodeApply(oFilter, nfCore), T)
        end
    end

    if !isconcretetype(eltype(diagFuncs))
        diagFuncType = getNormalizerCoreTypeUnion(T, D, Val(1), diagFuncCoreType)
        hasUnit && (diagFuncType = Union{UnitScalarCore{T}, diagFuncType})
        diagFuncs = Memory{diagFuncType}(diagFuncs)
    end

    if nOrbs > 1
        utriFuncNum = nOrbs * (nOrbs-1) ÷ 2
        utriFuncs = map(1:utriFuncNum) do j
            m, n = convertIndex1DtoTri2D(j)
            weightedOrb1 = weightedOrbs[begin+m-1]
            weightedOrb2 = weightedOrbs[begin+n-1]
            oCore1 = getField(weightedOrb1, oCorePtr)
            oCore2 = getField(weightedOrb2, oCorePtr)
            nfCore = buildCoreIntegrator(nIntStyle, Identity(), (oCore1, oCore2))
            utriFuncCoreType = typejoin(utriFuncCoreType, typeof(nfCore))
            oFilter1 = getField(weightedOrb1, oScopePtr) |> first
            oFilter2 = getField(weightedOrb2, oScopePtr) |> first
            ReturnTyped(EncodeApply((oFilter1, oFilter2), nfCore), T)
        end

        utriFuncType = eltype(utriFuncs)
        if !isconcretetype(utriFuncType)
            utriFuncType = getNormalizerCoreTypeUnion(T, D, Val(2), utriFuncCoreType)
        end
        utriFuncs = Memory{utriFuncType}(utriFuncs)
    else
        oCore = getField(weightedOrbs[begin], oCorePtr)
        utriFuncCoreType = (typeof∘buildCoreIntegrator)(OneBodyIntegral{D}(), Identity(), (oCore, oCore))
        utriFuncs = Memory{VariedNormCore{T, D, 2, utriFuncCoreType}}(undef, 0)
    end

    coreTransform = HermitianContract(diagFuncs, utriFuncs)
    getWeights = map(b->ReturnTyped(b.right.f, T), weightedOrbs)
    NormalizeCompOrb(coreTransform, getWeights)
end