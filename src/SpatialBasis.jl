export PrimitiveOrb, CompositeOrb, genGaussTypeOrb, genOrbitalData

(::SelectTrait{InputStyle})(::OrbitalBasis{C, D}) where {C<:RealOrComplex, D} = 
CartesianInput{D}()


(f::OrbitalBasis)(input) = evalOrbital(f, formatInput(f, input))


mutable struct PrimitiveOrb{T<:Real, D, C<:RealOrComplex{T}, F<:ShiftedField{T, D, C}
                            } <: OrbitalBasis{C, D, F}
    const field::F
    @atomic renormalize::Bool
end

const PrimGTO{T<:Real, D, F<:ShiftedPolyGaussField{T, D}} = PrimitiveOrb{T, D, T, F}

function PrimitiveOrb(center::NTuple{D, UnitOrVal{T}}, body::FieldAmplitude{C, D}; 
                      renormalize::Bool=false) where {T<:Real, C<:RealOrComplex{T}, D}
    PrimitiveOrb(ShiftedField(center, body), renormalize)
end

PrimitiveOrb(o::PrimitiveOrb; renormalize::Bool=o.renormalize) = 
PrimitiveOrb(o.field, renormalize)

getOutputType(::Type{<:PrimitiveOrb{T, D, C}}) where {T<:Real, D, C<:RealOrComplex{T}} = C


function evalOrbital(orb::PrimitiveOrb{T, D, C}, input; 
                     cache!Self::ParamDataCache=initializeParamDataCache()) where 
                    {T<:Real, D, C<:RealOrComplex{T}}
    coreValue = evalFieldAmplitude(orb.field, formatInput(orb, input); cache!Self)
    StableMul(C)(coreValue, getNormFactor(orb))
end


function getPrimitiveOrbType(orbs::AbstractArray{B}) where 
                            {T<:Real, D, B<:PrimitiveOrb{T, D}}
    if isempty(orbs) || isconcretetype(B)
        B
    else
        C = Union{}
        fieldType = Union{}

        for orb in orbs
            C = strictTypeJoin(C, getOutputType(orb))
            fieldType = strictTypeJoin(fieldType, typeof(orb.field))
        end

        pseudoType = genParametricType(PrimitiveOrb, (;T, D, C, F=fieldType))
        typeintersect(PrimitiveOrb, pseudoType)
    end
end

mutable struct CompositeOrb{T<:Real, D, C<:RealOrComplex{T}, B<:PrimitiveOrb{T, D}, 
                            W<:GridParam{C, 1}} <: OrbitalBasis{C, D, B}
    const basis::Memory{B}
    const weight::W
    @atomic renormalize::Bool

    function CompositeOrb(basis::Memory{<:PrimitiveOrb{T, D}}, weight::W, renormalize::Bool
                          ) where {T<:Real, C<:RealOrComplex{T}, D, W<:GridParam{C, 1}}
        nPrim = (first∘getOutputSize)(weight)
        checkLengthCore(checkEmptiness(basis, :basis), :basis, nPrim, 
                        "the output length of `weight`")
        formattedBasis = genMemory(basis)
        basisType = getPrimitiveOrbType(formattedBasis)
        new{T, D, C, basisType, W}(formattedBasis, weight, renormalize)
    end
end

const CompGTO{T<:Real, D, B<:PrimGTO{T, D}, W<:GridParam{T, 1}} = 
      CompositeOrb{T, D, T, B, W}

const ComposedOrb{T<:Real, D, C<:RealOrComplex{T}} = 
      Union{PrimitiveOrb{T, D, C}, CompositeOrb{T, D, C}}

function CompositeOrb(basis::AbstractVector{<:PrimitiveOrb{T, D}}, 
                      weight::GridParam{T, 1}; renormalize::Bool=false) where {T<:Real, D}
    CompositeOrb(extractMemory(basis), weight, renormalize)
end

function CompositeOrb(basis::AbstractVector{<:OrbitalBasis{<:RealOrComplex{T}, D}}, 
                      weight::GridParam{C, 1}; renormalize::Bool=false) where 
                     {T<:Real, C<:RealOrComplex{T}, D}
    formattedBasis = genMemory(basis)
    if !(eltype(formattedBasis) <: PrimitiveOrb)
        weight = mapreduce(vcat, enumerate(formattedBasis)) do (idx, basis)
            getEffectiveWeight(basis, weight, idx)
        end |> Base.Fix2(genHeapParam, :weight)
        formattedBasis = mapfoldl(splitOrb, vcat, basis)
    end
    CompositeOrb(formattedBasis, weight, renormalize)
end

splitOrb(o::PrimitiveOrb) = genMemory(o)

splitOrb(o::CompositeOrb) = copy(o.basis)

viewOrb(o::CompositeOrb, onrToIdx::Int) = o.basis[begin+onrToIdx-1]

function viewOrb(o::PrimitiveOrb, onrToIdx::Int)
    onrToIdx == 1 ? itself(o) : throw(BoundsError(o, onrToIdx))
end

function getEffectiveWeight(::PrimitiveOrb{T}, weight::GridParam{C, 1}, idx::Int) where 
                           {T<:Real, C<:RealOrComplex{T}}
    indexParam(weight, idx) |> fill
end

function getEffectiveWeight(o::CompositeOrb{T, D, C}, weight::GridParam{C, 1}, 
                            idx::Int) where {T<:Real, D, C<:RealOrComplex{T}}
    o.renormalize && throw(AssertionError("Merging the weight from a renormalized "*
                           "`CompositeOrb` with another value is prohibited."))
    len = first(getOutputSize(o.weight))
    outerWeight = indexParam(weight, idx)
    map(1:len) do i
        innerWight = indexParam(o.weight, i)
        genCellParam(StableMul(C), (innerWight, outerWeight), :w)
    end
end

function CompositeOrb(basis::AbstractVector{<:OrbitalBasis{<:RealOrComplex{T}, D}}, 
                      weight::UnitOrValVec{C}; renormalize::Bool=false) where 
                     {T<:Real, C<:RealOrComplex{T}, D}
    encoder = UnitParamEncoder(C, :w, 1)
    weightParams = genHeapParam(map(encoder, weight), :wBlock)
    CompositeOrb(basis, weightParams; renormalize)
end

CompositeOrb(o::CompositeOrb; renormalize::Bool=o.renormalize) = 
CompositeOrb(o.basis, o.weight, renormalize)

getOutputType(::Type{<:CompositeOrb{T, D, C}}) where {T<:Real, D, C<:RealOrComplex{T}} = C


function evalOrbital(orb::CompositeOrb{T, D, C}, input; 
                     cache!Self::ParamDataCache=initializeParamDataCache()) where 
                    {T<:Real, D, C<:RealOrComplex{T}}
    weightVal = cacheParam!(cache!Self, orb.weight)

    bodyVal = mapreduce(StableAdd(C), orb.basis, weightVal) do basis, w
        evalOrbital(basis, input; cache!Self) * w
    end

    StableMul(C)(convert(C, bodyVal), getNormFactor(orb))
end


function isRenormalized(orb::ComposedOrb)
    orb.renormalize
end


function enforceRenormalize!(b::ComposedOrb)
    @atomic b.renormalize = true
end


function disableRenormalize!(b::ComposedOrb)
    @atomic b.renormalize = false
end


function getNormFactor(orb::ComposedOrb{T, D, C}) where {T<:Real, D, C<:RealOrComplex{T}}
    if isRenormalized(orb)
        constructor = getfield(Quiqbox, nameof(orb))
        orbInner = constructor(orb, renormalize=false)
        overlapVal = computeIntegral(OneBodyIntegral{D}(), genOverlapSampler(), (orbInner,))
        convert(C, absSqrtInv(overlapVal))
    else
        one(C)
    end
end


struct PrimOrbData{T<:Real, D, C<:RealOrComplex{T}, F<:AbstractParamFunc, 
                   R<:FieldCenterShifter{T, D}, S<:OptSpanValueSet} <: ConfigBox
    core::StashedShiftedField{T, D, C, F, R, S}
    renormalize::Bool
end

const PGTOrbData{T<:Real, D, F<:PolyGaussFieldCore{T, D}, R<:FieldCenterShifter{T, D}, 
                 S<:OptSpanValueSet} = 
      PrimOrbData{T, D, T, F, R, S}

struct CompOrbData{T<:Real, D, C<:RealOrComplex{T}, B<:PrimOrbData{T, D}} <: ConfigBox
    basis::Memory{B}
    weight::Memory{C}
    renormalize::Bool
end

const OrbitalData{T<:Real, D, C<:RealOrComplex{T}} = 
      Union{PrimOrbData{T, D, C}, CompOrbData{T, D, C}}


const PrimOrbDataVec{T<:Real, D} = AbstractVector{<:PrimOrbData{T, D}}

const OrbDataVec{T<:Real, D} = AbstractVector{<:OrbitalData{T, D}}

const OrbDataTpl{T<:Real, D} = NonEmptyTuple{OrbitalData{T, D}}

const OrbDataCollection{T<:Real, D} = Union{OrbDataVec{T, D}, OrbDataTpl{T, D}}

const OrbDataSource{T<:Real, D} = Union{OrbitalData{T, D}, OrbDataCollection{T, D}}


getPrimOrbDataTypeUnion(::T) where {T<:PrimOrbData} = T

function getPrimOrbDataTypeUnion(data::CompOrbData{T, D}) where {T<:Real, D}
    typeintersect(getMinimalEleType(data.basis), PrimOrbData{T, D})
end

function getPrimOrbDataTypeUnion(data::OrbDataCollection{T, D}) where {T<:Real, D}
    checkEmptiness(data, :data)
    res = mapreduce(strictTypeJoin, data) do ele
        getPrimOrbDataTypeUnion(ele)
    end
    typeintersect(res, PrimOrbData{T, D})
end


getOrbOutputTypeUnion(::Type{<:OrbitalData{T, D, C}}) where {T, D, C<:RealOrComplex{T}} = C

getOrbOutputTypeUnion(::T) where {T<:OrbitalData} = getOrbOutputTypeUnion(T)

getOrbOutputTypeUnion(::AbstractVector{<:OrbitalData{T, D, C}}) where 
                     {T<:Real, D, C<:RealOrComplex{T}} = C

getOrbOutputTypeUnion(::NonEmptyTuple{OrbitalData{T, D, C}}) where 
                     {T<:Real, D, C<:RealOrComplex{T}} = C

function getOrbOutputTypeUnion(orbsData::OrbDataCollection{T, D}) where {T<:Real, D}
    if isempty(orbsData)
        getOrbOutputTypeUnion(orbsData|>eltype)
    else
        mapreduce(getOrbOutputTypeUnion, strictTypeJoin, orbsData)
    end::Union{Type{Complex{T}}, Type{T}}
end


const MixedFieldParamFuncCacheCore{T, D, C<:RealOrComplex{T}} = 
      LRU{EgalBox{ShiftedField{T, D, C}}, ShiftedFieldFunc{T, D, C}}

struct MixedFieldParamFuncCache{T<:Real, D
                                } <: QueryBox{Union{ FieldParamFunc{T, D, T}, 
                                                     FieldParamFunc{T, D, Complex{T}} }}
    real::MixedFieldParamFuncCacheCore{T, D, T}
    complex::MixedFieldParamFuncCacheCore{T, D, Complex{T}}
end

const TypedFieldParamFuncCache{T<:Real, D, C<:RealOrComplex{T}, F<:ShiftedField{T, D, C}} = 
      LRU{EgalBox{F}, ShiftedFieldFunc{T, D, C}}

const FieldParamFuncCache{T<:Real, D} = 
      Union{MixedFieldParamFuncCache{T, D}, TypedFieldParamFuncCache{T, D}}

getBasisInnerType(::Type{F}) where {C<:RealOrComplex, D, B, F<:OrbitalBasis{C, D, B}} = 
(B <: OrbitalBasis ? getBasisInnerType(B) : B)

function genFieldParamFuncCache(::Type{F}, maxSize::Int=200) where 
                               {T<:Real, D, F<:OrbitalBasis{<:RealOrComplex{T}, D}}
    checkPositivity(maxSize)
    if isconcretetype(F)
        fieldType = getBasisInnerType(F)
        valueType = getOutputType(fieldType)
        TypedFieldParamFuncCache{T, D, valueType, fieldType}(maxsize=maxSize)
    else
        rSector = MixedFieldParamFuncCacheCore{T, D, T}(maxsize=maxSize)
        cSector = MixedFieldParamFuncCacheCore{T, D, Complex{T}}(maxsize=maxSize)
        MixedFieldParamFuncCache(rSector, cSector)
    end
end



function genCachedFieldFunc!(fieldCache::MixedFieldParamFuncCache{T, D}, 
                             paramCache::ParamDataCache, 
                             orb::PrimitiveOrb{T, D, C}, 
                             paramSet::OptSpanParamSet, directUnpack::Boolean) where 
                            {T<:Real, D, C<:RealOrComplex{T}}
    sector = C <: Real ? :real : :complex
    f = get!(getfield(fieldCache, sector), EgalBox{ShiftedField{T, D, C}}(orb.field)) do
        unpackFunc!(orb.field, paramSet, directUnpack)
    end
    StashedField(f, paramSet, paramCache)
end #!! Incorporate `genCachedFieldFunc!` into `unpackFunc`

function genCachedFieldFunc!(fieldCache::TypedFieldParamFuncCache{T, D, C, F}, 
                             paramCache::ParamDataCache, 
                             orb::PrimitiveOrb{T, D, C, F}, 
                             paramSet::OptSpanParamSet, directUnpack::Boolean) where 
                            {T<:Real, D, C<:RealOrComplex{T}, F<:ShiftedField{T, D, C}}
    f = get!(fieldCache, EgalBox{F}(orb.field)) do
        unpackFunc!(orb.field, paramSet, directUnpack)
    end
    StashedField(f, paramSet, paramCache)
end


function genOrbitalDataCore!(fieldCache::FieldParamFuncCache{T, D}, 
                             paramCache::ParamDataCache, 
                             paramSet::OptSpanParamSet, 
                             orb::PrimitiveOrb{T, D, C}, 
                             directUnpack::Boolean) where 
                            {T<:Real, D, C<:RealOrComplex{T}}
    coreField = genCachedFieldFunc!(fieldCache, paramCache, orb, paramSet, directUnpack)
    PrimOrbData(coreField, orb.renormalize)
end

function genOrbitalDataCore!(fieldCache::FieldParamFuncCache{T, D}, 
                             paramCache::ParamDataCache, 
                             paramSet::OptSpanParamSet, 
                             orb::CompositeOrb{T, D}, 
                             directUnpack::Boolean) where {T<:Real, D}
    basisData = map(orb.basis) do basis
        genOrbitalDataCore!(fieldCache, paramCache, paramSet, basis, directUnpack)
    end
    weightData = cacheParam!(paramCache, orb.weight)
    CompOrbData(basisData, extractMemory(weightData), orb.renormalize)
end

const OrbBasisVec{T<:Real, D} = AbstractVector{<:OrbitalBasis{<:RealOrComplex{T}, D}}

const OrbBasisTpl{T<:Real, D} = NonEmptyTuple{OrbitalBasis{<:RealOrComplex{T}, D}}

const OrbBasisCollection{T<:Real, D} = Union{OrbBasisVec{T, D}, OrbBasisTpl{T, D}}

const OrbBasisSource{T<:Real, D} = 
      Union{OrbitalBasis{<:RealOrComplex{T}, D}, OrbBasisCollection{T, D}}

function genOrbitalDataCore!(fieldCache::FieldParamFuncCache{T, D}, 
                             paramCache::ParamDataCache, 
                             paramSet::OptSpanParamSet, 
                             orbs::OrbBasisCollection{T, D}, 
                             directUnpack::Boolean) where {T<:Real, D}
    checkEmptiness(orbs, :orbs)
    map(orbs) do orb
        genOrbitalDataCore!(fieldCache, paramCache, paramSet, orb, directUnpack)
    end
end

function genOrbitalDataCore!(fieldCache::FieldParamFuncCache{T, D}, 
                             paramCache::ParamDataCache, 
                             paramSet::OptSpanParamSet, 
                             orbs::N24Tuple{OrbitalBasis{<:RealOrComplex{T}, D}}, 
                             directUnpack::Boolean) where 
                            {T<:Real, D}
    lazyMap(orbs) do orb
        genOrbitalDataCore!(fieldCache, paramCache, paramSet, orb, directUnpack)
    end
end

function genOrbitalData(orbBases::B, directUnpack::Boolean=False(); 
                        cache!Self::ParamDataCache=initializeParamDataCache()) where 
                       {T<:Real, D, B<:OrbBasisCollection{T, D}}
    paramSet = initializeSpanParamSet()
    coreCache = genFieldParamFuncCache(B|>eltype)
    genOrbitalDataCore!(coreCache, cache!Self, paramSet, orbBases, directUnpack)
end

function genOrbitalData(orbBasis::F, directUnpack::Boolean=False(); 
                        cache!Self::ParamDataCache=initializeParamDataCache()) where 
                       {F<:OrbitalBasis}
    paramSet = initializeSpanParamSet()
    coreCache = genFieldParamFuncCache(F)
    genOrbitalDataCore!(coreCache, cache!Self, paramSet, orbBasis, directUnpack)
end


function genGaussTypeOrb(center::NonEmptyTuple{UnitOrVal{T}, D}, 
                         xpn::UnitOrVal{T}, 
                         ijk::NonEmptyTuple{Int, D}=ntuple(_->0, Val(D+1)); 
                         renormalize::Bool=false) where {T<:Real, D}
    gf = GaussFunc(xpn)
    PrimitiveOrb(center, PolyRadialFunc(gf, ijk); renormalize)
end

function genGaussTypeOrb(center::NonEmptyTuple{UnitOrVal{T}, D}, 
                         xpns::UnitOrValVec{T}, 
                         cons::Union{UnitOrValVec{C}, GridParam{C, 1}}, 
                         ijk::NonEmptyTuple{Int, D}=ntuple(_->0, Val(D+1)); 
                         innerRenormalize::Bool=false, outerRenormalize::Bool=false) where 
                        {T<:Real, C<:RealOrComplex{T}, D}
    nPrimOrbs = if cons isa GridParam
        (first∘getOutputSize)(cons)
    else
        cons = map(UnitParamEncoder(C, :con, 1), cons)
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