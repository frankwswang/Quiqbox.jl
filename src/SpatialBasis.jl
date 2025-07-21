export PrimitiveOrb, CompositeOrb, genGaussTypeOrb, genGaussTypeOrbSeq

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
        overlapVal = computeOrbLayoutIntegral(genOverlapSampler(), (orbInner, orbInner))
        convert(C, absSqrtInv(overlapVal))
    else
        one(C)
    end
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


const OrbBasisVector{T<:Real, D} = AbstractVector{<:OrbitalBasis{<:RealOrComplex{T}, D}}

const OrbBasisLayout{T<:Real, D} = N24Tuple{OrbitalBasis{<:RealOrComplex{T}, D}}

const OrbBasisCluster{T<:Real, D} = Union{OrbBasisVector{T, D}, OrbBasisLayout{T, D}}


function getOutputType(orbitals::OrbBasisCluster{T, D}) where {T<:Real, D}
    checkEmptiness(orbitals, :orbitals)
    orbType = eltype(orbitals)
    if isconcretetype(orbType)
        getOutputType(orbType)
    else
        mapreduce(getOutputType, strictTypeJoin, orbitals)
    end::Type{<:RealOrComplex{T}}
end


struct PrimOrbPointer{D, C<:RealOrComplex} <: CustomAccessor
    inner::OneToIndex
    renormalize::Bool
end

struct CompOrbPointer{D, C<:RealOrComplex} <: CustomAccessor
    inner::MemoryPair{PrimOrbPointer{D, C}, C}
    renormalize::Bool
end

const OrbitalPointer{D, C<:RealOrComplex} = 
      Union{PrimOrbPointer{D, C}, CompOrbPointer{D, C}}

getOutputType(::OrbitalPointer{D, C}) where {D, C<:RealOrComplex} = C

strictTypeJoin(::Type{CompOrbPointer{D, C}}, ::Type{PrimOrbPointer{D, C}}) where 
              {D, C<:RealOrComplex}= 
OrbitalPointer{D, C}

strictTypeJoin(::Type{PrimOrbPointer{D, C}}, ::Type{CompOrbPointer{D, C}}) where 
              {D, C<:RealOrComplex}= 
OrbitalPointer{D, C}

const ShiftedFieldConfigDict{T, D, F<:StashedShiftedField{T, D}} = 
      IndexDict{FieldMarker{:StashedField, 2}, F}

const ShiftedFieldMarkerDict{T<:Real, D} = 
      Dict{EgalBox{ShiftedField{T, D}}, FieldMarker{:StashedField, 2}}


function genOrbitalPointer!(::Type{C}, 
                            configDict::ShiftedFieldConfigDict{T, D}, 
                            markerDict::ShiftedFieldMarkerDict{T, D}, 
                            orbital::PrimitiveOrb{T, D}, 
                            directUnpack::Boolean, 
                            paramSet::OptSpanParamSet, 
                            paramCache::ParamDataCache) where 
                           {T<:Real, D, C<:RealOrComplex{T}}
    field = orbital.field
    tracker = EgalBox{ShiftedField{T, D}}(field)

    marker = if haskey(markerDict, tracker)
        markerDict[tracker]
    else
        fieldFunc = unpackFunc!(field, paramSet, directUnpack)
        fieldCore = StashedField(fieldFunc, paramSet, paramCache)
        marker = markObj(fieldCore)
        setindex!(markerDict, marker, tracker)
        get!(configDict, marker, fieldCore)
        marker
    end

    PrimOrbPointer{D, C}(keyIndex(configDict, marker), orbital.renormalize)
end


function genOrbitalPointer!(::Type{C}, 
                            configDict::ShiftedFieldConfigDict{T, D}, 
                            markerDict::ShiftedFieldMarkerDict{T, D}, 
                            orbital::CompositeOrb{T, D}, 
                            directUnpack::Boolean, 
                            paramSet::OptSpanParamSet, 
                            paramCache::ParamDataCache) where 
                           {T<:Real, D, C<:RealOrComplex{T}}
    weightValue = convert(Memory{C}, cacheParam!(paramCache, orbital.weight))
    primOrbPtrs = map(orbital.basis) do o
        genOrbitalPointer!(C, configDict, markerDict, o, directUnpack, paramSet, paramCache)
    end
    CompOrbPointer{D, C}(MemoryPair(primOrbPtrs, weightValue), orbital.renormalize)
end


function initializeOrbitalConfigDict(::Type{T}, ::Count{D}, 
                                     ::Type{F}=StashedShiftedField{T, D}) where 
                                    {T, D, F<:StashedShiftedField{T, D}}
    ShiftedFieldConfigDict{T, D, F}()
end

function cacheOrbitalData!(configDict::ShiftedFieldConfigDict{T, D}, 
                           orbitals::OrbBasisCluster{T, D}, directUnpack::Boolean, 
                           cache!Self::ParamDataCache=initializeParamDataCache()) where 
                          {T<:Real, D}
    checkEmptiness(orbitals, :orbitals)
    paramSet = initializeSpanParamSet()
    outputType = getOutputType(orbitals)
    markerDict = ShiftedFieldMarkerDict{T, D}()
    lazyMap(orbitals) do orb
        genOrbitalPointer!(outputType, configDict, markerDict, orb, 
                           directUnpack, paramSet, cache!Self)
    end
end


struct OrbCorePointer{D, C<:RealOrComplex}
    inner::MemoryPair{OneToIndex, C}

    function OrbCorePointer(::Count{D}, inner::MemoryPair{OneToIndex, C}) where 
                           {D, C<:RealOrComplex}
        new{D, C}(inner)
    end
end

function OrbCorePointer(pointer::PrimOrbPointer{D, C}, weight::C
                        ) where {D, C<:RealOrComplex}
    innerPair = MemoryPair(genMemory(pointer.inner), genMemory(weight))
    OrbCorePointer(Count(D), innerPair)
end

function OrbCorePointer(pointer::CompOrbPointer{D, C}, weight::AbstractArray{C, 1}
                        ) where {D, C<:RealOrComplex}
    innerPair = MemoryPair(map(x->x.inner, pointer.inner.left), extractMemory(weight))
    OrbCorePointer(Count(D), innerPair)
end


const OrbPairLayoutFormat{D, C} = 
      Union{NTuple{2, OrbitalPointer{D, C}}, NTuple{ 2, OrbCorePointer{D, C} }}

const OrbQuadLayoutFormat{D, C} = 
      Union{NTuple{4, OrbitalPointer{D, C}}, NTuple{ 4, OrbCorePointer{D, C} }}

const OrbitalLayoutFormat{D, C} = 
      Union{OrbPairLayoutFormat{D, C}, OrbQuadLayoutFormat{D, C}}

const OrbitalVectorFormat{D, C} = 
      Union{Memory{<:OrbitalPointer{D, C}}, Memory{ OrbCorePointer{D, C} }}

const OrbFormatCollection{D, C<:RealOrComplex} = 
      Union{OrbitalLayoutFormat{D, C}, OrbitalVectorFormat{D, C}}


struct MultiOrbitalData{T<:Real, D, C<:RealOrComplex{T}, F<:StashedShiftedField{T, D}, 
                        P<:OrbFormatCollection{D, C}} <: QueryBox{F}
    config::Memory{F}
    format::P

    function MultiOrbitalData(orbitals::OrbBasisCluster{T, D}, 
                              directUnpack::Boolean=False(); 
                              cache!Self::ParamDataCache=initializeParamDataCache()) where 
                             {T<:Real, D}
        configDict = initializeOrbitalConfigDict(T, Count(D))
        orbPointers = cacheOrbitalData!(configDict, orbitals, directUnpack, cache!Self)
        format = orbitals isa AbstractVector ? genMemory(orbPointers) : orbPointers
        config = genMemory(last.(configDict.storage))
        new{T, D, getOutputType(orbitals), eltype(config), typeof(format)}(config, format)
    end

    function MultiOrbitalData(data::MultiOrbitalData{T, D, C}, 
                              format::OrbFormatCollection{D, C}) where 
                             {T<:Real, D, C<:RealOrComplex{T}}
        config = data.config
        new{T, D, C, eltype(config), typeof(format)}(config, format)
    end
end

const OrbitalLayoutData{T<:Real, D, C<:RealOrComplex{T}, F<:StashedShiftedField{T, D}, 
                        P<:OrbitalLayoutFormat{D, C}} = 
      MultiOrbitalData{T, D, C, F, P}

const OrbitalVectorData{T<:Real, D, C<:RealOrComplex{T}, F<:StashedShiftedField{T, D}, 
                        P<:OrbitalVectorFormat{D, C}} = 
      MultiOrbitalData{T, D, C, F, P}

getOutputType(::MultiOrbitalData{T, D, C}) where {T<:Real, D, C<:RealOrComplex{T}} = C

const OrbBasisData{T<:Real, D} = Union{MultiOrbitalData{T, D}, OrbBasisVector{T, D}}


function get3DimPGTOrbNormFactor(xpn::T, carteAng::NTuple{3, Int}) where {T<:Real}
    for i in carteAng; checkPositivity(i, true) end
    i, j, k = carteAng
    angSum = sum(carteAng)
    xpnPart = xpn^(T(2angSum + 3)*T(0.25))
    angPart = T(PowersOfPi[:n0d75]) * (exp2∘T)(1.5angSum + 0.75) * 
              T(sqrt( factorial(i) * factorial(j) * factorial(k) / 
                      (factorial(2i) * factorial(2j) * factorial(2k)) ))
    xpnPart * angPart
end

function genGaussTypeOrbSeq(center::NTuple{3, UnitOrVal{T}}, 
                            content::AbstractString; unlinkCenter::Bool=false, 
                            innerRenormalize::Bool=false, outerRenormalize::Bool=false
                            ) where {T<:Real}
    cenEncoder = let cenParams=map(UnitParamEncoder(T, :cen, 1), center)
        unlinkCenter ? ()->deepcopy(cenParams) : ()->cenParams
    end
    formattedContent = replaceSciNotation(content)
    data = map((@view formattedContent[begin : end-1]) |> IOBuffer |> readlines) do line
        advancedParse.(T, split(line))
    end
    idxScope = findall(x -> eltype(x)!=T && length(x)>2 && x[begin]!="X", data)
    bfs = CompGTO{T, 3}[] #! Can be replaced by a more type-specific container

    for j in idxScope
        oInfo = data[j]
        nPGTOrb = Int(oInfo[begin + 1])
        coeffPairs = @view data[j+1 : j+nPGTOrb]
        xpns = first.(coeffPairs)
        subshellStr = first(oInfo)
        angNums = subshellStr == "SP" ? (0, 1) : (AngularSubShellDict[subshellStr],)

        for (i, angNum) in enumerate(angNums)
            for ijk in SubshellXYZs[begin+angNum]
                cons = map(xpns, coeffPairs) do xpn, segment
                    getEntry(segment, OneToIndex(1+i)) * get3DimPGTOrbNormFactor(xpn, ijk)
                end
                push!(bfs, genGaussTypeOrb(cenEncoder(), xpns, cons, ijk; 
                                           innerRenormalize, outerRenormalize))
            end
        end
    end

    bfs
end


function genGaussTypeOrbSeq(center::NTuple{3, UnitOrVal{T}}, 
                            atm::Symbol, basisKey::String; unlinkCenter::Bool=false, 
                            innerRenormalize::Bool=false, outerRenormalize::Bool=false
                            ) where {T<:Real}
    hasBasis = true
    basisSetFamily = get(AtomicGTOrbSetDict, basisKey, nothing)
    basisStr = if basisSetFamily === nothing
        hasBasis = false
        ""
    else
        atmCharge = get(NuclearChargeDict, atm, nothing)
        if atmCharge === nothing || atmCharge > length(basisSetFamily)
            hasBasis = false
            ""
        else
            res = getEntry(basisSetFamily, atmCharge)
            if res === nothing
                hasBasis = false
                ""
            else
                res
            end
        end
    end
    hasBasis || throw(DomainError((atm, basisKey), 
                      "Quiqbox does not have this basis-set configuration pre-stored."))
    genGaussTypeOrbSeq(center, basisStr; unlinkCenter, innerRenormalize, outerRenormalize)
end