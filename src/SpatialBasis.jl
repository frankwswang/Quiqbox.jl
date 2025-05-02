export PrimitiveOrb, CompositeOrb, genGaussTypeOrb, genOrbitalData

(::SelectTrait{InputStyle})(::OrbitalBasis{T, D}) where {T, D} = TupleInput{Real, D}()


(f::OrbitalBasis)(input) = evalOrbital(f, formatInput(f, input))


mutable struct PrimitiveOrb{T<:Number, D, B<:FieldAmplitude{T, D}, 
                            C<:NTuple{ D, UnitParam{T} }} <: OrbitalBasis{T, D, B}
    const center::C
    const body::B
    @atomic renormalize::Bool
end

const PrimGTO{T<:Number, D, B<:PolyGaussFunc{T, D}, C<:NTuple{D, UnitParam{T}}} = 
      PrimitiveOrb{T, D, B, C}

function PrimitiveOrb(center::NTuple{D, UnitOrVal{T}}, body::B; 
                      renormalize::Bool=false) where {T, D, B<:FieldAmplitude{T, D}}
    length(center)!=D && throw(AssertionError("The length of `center` must match `D=$D`."))
    encoder = UnitParamEncoder(T, :cen, 1)
    PrimitiveOrb(encoder.(center), body, renormalize)
end

PrimitiveOrb(o::PrimitiveOrb; renormalize::Bool=o.renormalize) = 
PrimitiveOrb(o.center, o.body, renormalize)


function evalOrbital(orb::PrimitiveOrb{T}, input; 
                     cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox()) where 
                    {T<:Number}
    encodedInput = formatInput(orb, input) .- cacheParam!(cache!Self, orb.center)
    StableMul(T)(evalFieldAmplitude(orb.body, encodedInput; cache!Self), getNormFactor(orb))
end


function getPrimitiveOrbType(orbs::AbstractArray{B}) where {T, D, B<:PrimitiveOrb{T, D}}
    if isempty(orbs) || isconcretetype(B)
        B
    else
        bodies = FieldAmplitude{T, D}[]
        centers = NTuple{D, UnitParam{T}}[]
        for orb in orbs
            push!(bodies, orb.body)
            push!(centers, orb.center)
        end
        @show bodies
        bodyType = mapreduce(typeof, strictTypeJoin, bodies)
        @show bodyType
        centerType = mapreduce(typeof, strictTypeJoin, centers)
        bl1 = isconcretetype(bodyType)
        bl2 = isconcretetype(centerType)
        if bl1 && bl2
            PrimitiveOrb{T, D,   bodyType,   centerType}
        elseif !bl1 && bl2
            PrimitiveOrb{T, D, <:bodyType,   centerType}
        elseif bl1 && !bl2
            PrimitiveOrb{T, D,   bodyType, <:centerType}
        else
            PrimitiveOrb{T, D, <:bodyType, <:centerType}
        end
    end
end

mutable struct CompositeOrb{T<:Number, D, B<:PrimitiveOrb{T, D}, W<:GridParam{T, 1}
                            } <: OrbitalBasis{T, D, B}
    const basis::Memory{B}
    const weight::W
    @atomic renormalize::Bool

    function CompositeOrb(basis::Memory{<:PrimitiveOrb{T, D}}, weight::W, 
                          renormalize::Bool) where {T, D, W<:GridParam{T, 1}}
        nPrim = (first∘getOutputSize)(weight)
        checkLengthCore(checkEmptiness(basis, :basis), :basis, nPrim, 
                        "the output length of `weight`")
        formattedBasis = map(itself, basis)
        basisType = getPrimitiveOrbType(formattedBasis)
        new{T, D, basisType, W}(formattedBasis, weight, renormalize)
    end
end

const ComposedOrb{T<:Number, D} = Union{PrimitiveOrb{T, D}, CompositeOrb{T, D}}

const CompGTO{T<:Number, D, B<:PrimGTO{T, D}, W<:GridParam{T, 1}} = CompositeOrb{T, D, B, W}

function CompositeOrb(basis::AbstractVector{<:PrimitiveOrb{T, D}}, 
                      weight::GridParam{T, 1}; renormalize::Bool=false) where {T, D}
    CompositeOrb(getMemory(basis), weight, renormalize)
end

function CompositeOrb(basis::AbstractVector{<:OrbitalBasis{T, D}}, 
                      weight::GridParam{T, 1}; renormalize::Bool=false) where {T, D}
    formattedBasis = getMemory(basis)
    if !(eltype(formattedBasis) <: PrimitiveOrb)
        weight = mapreduce(vcat, enumerate(formattedBasis)) do (idx, basis)
            getEffectiveWeight(basis, weight, idx)
        end |> Base.Fix2(genHeapParam, :weight)
        formattedBasis = mapfoldl(splitOrb, vcat, basis)
    end
    CompositeOrb(formattedBasis, weight, renormalize)
end

splitOrb(o::PrimitiveOrb) = getMemory(o)

splitOrb(o::CompositeOrb) = copy(o.basis)

viewOrb(o::CompositeOrb, onrToIdx::Int) = o.basis[begin+onrToIdx-1]

function viewOrb(o::PrimitiveOrb, onrToIdx::Int)
    onrToIdx == 1 ? itself(o) : throw(BoundsError(o, onrToIdx))
end

function getEffectiveWeight(::PrimitiveOrb{T}, weight::GridParam{T, 1}, idx::Int) where {T}
    indexParam(weight, idx) |> fill
end

function getEffectiveWeight(o::CompositeOrb{T}, weight::GridParam{T, 1}, 
                            idx::Int) where {T}
    o.renormalize && throw(AssertionError("Merging the weight from a renormalized "*
                           "`CompositeOrb` with another value is prohibited."))
    len = first(getOutputSize(o.weight))
    outerWeight = indexParam(weight, idx)
    map(1:len) do i
        innerWight = indexParam(o.weight, i)
        genCellParam(StableMul(T), (innerWight, outerWeight), :w)
    end
end

function CompositeOrb(basis::AbstractVector{<:OrbitalBasis{T, D}}, 
                      weight::UnitOrValVec{T}; renormalize::Bool=false) where {T, D}
    encoder = UnitParamEncoder(T, :w, 1)
    weightParams = genHeapParam(map(encoder, weight), :wBlock)
    CompositeOrb(basis, weightParams; renormalize)
end

CompositeOrb(o::CompositeOrb; renormalize::Bool=o.renormalize) = 
CompositeOrb(o.basis, o.weight, renormalize)


function evalOrbital(orb::CompositeOrb{T}, input; 
                     cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox()) where 
                    {T<:Number}
    multiplier = StableMul(T)
    weightVal = cacheParam!(cache!Self, orb.weight)

    body = mapreduce(StableAdd(T), orb.basis, weightVal) do basis, w
        multiplier(evalOrbital(basis, input; cache!Self), w)
    end

    multiplier(body, getNormFactor(orb))
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


function getNormFactor(orb::ComposedOrb{T}) where {T}
    if isRenormalized(orb)
        constructor = getfield(Quiqbox, nameof(orb))
        orbInner = constructor(orb, renormalize=false)
        convert(T, overlap(orbInner, orbInner)|>AbsSqrtInv)
    else
        one(T)
    end
end

#!! Remove the parameter set.
struct PrimOrbData{T<:Number, D, C<:Real, F<:AbstractParamFunc, 
                   P<:AbstractSpanValueSet} <: ConfigBox
    center::NTuple{D, C}
    body::Pair{FieldParamFunc{T, D, F}, P}
    renormalize::Bool

    function PrimOrbData(center::NonEmptyTuple{C}, 
                         body::Pair{FieldParamFunc{T, D, F}, P}, renormalize::Bool) where 
                        {C<:Real, T<:Number, D, F<:AbstractParamFunc, 
                         P<:AbstractSpanValueSet}
        new{T, D, C, F, P}(center, body, renormalize)
    end
end

struct CompOrbData{T<:Number, D, B<:PrimOrbData{T, D}} <: ConfigBox
    basis::Memory{B}
    weight::Memory{T}
    renormalize::Bool
end

const OrbitalData{T<:Number, D} = Union{PrimOrbData{T, D}, CompOrbData{T, D}}

const OrbitalDataCollection{T<:Number, D} = NonEmpTplOrAbtArr{OrbitalData{T, D}, 1}

const OrbitalDataInput{T<:Number, D} = Union{OrbitalData{T, D}, OrbitalDataCollection{T, D}}


getPrimOrbDataTypeUnion(::T) where {T<:PrimOrbData} = T

function getPrimOrbDataTypeUnion(data::CompOrbData)
    getMinimalEleType(data.basis)
end

function getPrimOrbDataTypeUnion(data::OrbitalDataCollection)
    checkEmptiness(data, :data)
    mapreduce(strictTypeJoin, data) do ele
        getPrimOrbDataTypeUnion(ele)
    end
end


const FieldCoreCache{T<:Number, D} = 
      LRU{EgalBox{FieldAmplitude{T, D}}, FieldParamFunc{T, D}}

FieldCoreCache{T, D}() where {T, D} = FieldCoreCache{T, D}(maxsize=200)

function genOrbitalDataCore!(fieldCoreCache::FieldCoreCache{T, D}, 
                             paramCache::MultiSpanDataCacheBox, 
                             paramSet::AbstractSpanParamSet, 
                             orb::PrimitiveOrb{T, D}) where {T, D}
    centerData = promote(cacheParam!(paramCache, orb.center)...)
    bodyCore = get!(fieldCoreCache, EgalBox{FieldAmplitude{T, D}}(orb.body)) do
        unpackFunc!(orb.body, paramSet)
    end
    PrimOrbData(centerData, bodyCore=>map(obtain, paramSet), orb.renormalize)
end

function genOrbitalDataCore!(fieldCoreCache::FieldCoreCache{T, D}, 
                             paramCache::MultiSpanDataCacheBox, 
                             paramSet::AbstractSpanParamSet, 
                             orb::CompositeOrb{T, D}) where {T, D}
    basisData = map(orb.basis) do basis
        genOrbitalDataCore!(fieldCoreCache, paramCache, paramSet, basis)
    end
    weightData = cacheParam!(paramCache, orb.weight)
    CompOrbData(basisData, getMemory(weightData), orb.renormalize)
end

const OrbitalCollection{T<:Number, D} = NonEmpTplOrAbtArr{OrbitalBasis{T, D}, 1}

const OrbitalBasisSource{T<:Number, D} = Union{ComposedOrb{T, D}, OrbitalCollection{T, D}}

function genOrbitalDataCore!(fieldCoreCache::FieldCoreCache{T, D}, 
                             paramCache::MultiSpanDataCacheBox, 
                             paramSet::AbstractSpanParamSet, 
                             orbs::OrbitalCollection{T, D}) where {T, D}
    checkEmptiness(orbs, :orbs)
    fMap = orbs isa Tuple ? lazyTupleMap : map
    fMap(orbs) do orb
        genOrbitalDataCore!(fieldCoreCache, paramCache, paramSet, orb)
    end
end

function genOrbitalData!(paramSet::AbstractSpanParamSet, 
                         orbSource::OrbitalBasisSource{T, D}; 
                         cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox()
                         ) where {T, D}
    coreCache = FieldCoreCache{T, D}()
    genOrbitalDataCore!(coreCache, cache!Self, paramSet, orbSource)
end

function genOrbitalData(orbSource::OrbitalBasisSource)
    paramSet = initializeSpanParamSet()
    genOrbitalData!(paramSet, orbSource)
end

const PrimGTOData{T<:Number, D, C<:Real, F<:PolyGaussFieldCore{T, D}, 
                  P<:AbstractSpanValueSet} = 
      PrimOrbData{T, D, C, F, P}


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