using LinearAlgebra: dot

const OrbitalDataSet{T, D} = AbstractVector{<:OrbitalData{T, D}}

const OrbitalBasisSet{T, D} = AbstractVector{<:OrbitalBasis{T, D}}

const OrbCoreMarker = FieldMarker{:FieldParamFunc, 1}

const OrbCoreMarkerDict{T, D} = Dict{EgalBox{FieldParamFunc{T, D}}, OrbCoreMarker}

const OrbCoreKey{D, C<:Real, P<:AbstractSpanValueSet} = 
      Tuple{NTuple{D, C}, P, OrbCoreMarker}

const OrbCoreIdxDict{D} = Dict{OrbCoreKey{D}, Int}

const OrbCoreDataSeq{T, D} = AbstractVector{<:PrimOrbData{T, D}}

const OneBodyIdxSymDict = let tempDict=Base.ImmutableDict(true=>:aa)
    Base.ImmutableDict(tempDict, false=>:ab)
end

const TwoBodyIdxSymDict = let
    keyTemp = ((true,  true,  true), (true,  true, false), (false, false,  true), 
               (true, false, false), (false, true, false), (false, false, false))
    valTemp = (:aaaa, :aabb, :abab, :aaxy, :abxx, :abxy)
    mapreduce(Base.ImmutableDict, keyTemp, valTemp, 
              init=Base.ImmutableDict{NTuple{3, Bool}, Symbol}()) do key, val
        key=>val
    end
end

# const GeneralTensorIdxSymDict =  let tempDict=Base.ImmutableDict(2=>:ij)
#     Base.ImmutableDict(tempDict, 4=>:ijkl)
# end

#!? Consider replacing it with LRU-based cache to control memory consumption
struct PrimOrbDataCache{T, D, P<:PrimOrbData{T, D}}
    dict::OrbCoreIdxDict{D}
    list::Vector{P}

    function PrimOrbDataCache(::Type{T}, ::Val{D}, ::Type{P}=PrimOrbData{T, D}) where 
                    {T, D, P<:PrimOrbData{T, D}}
    new{T, D, P}(OrbCoreIdxDict{D}(), P[])
    end
end


struct IndexedWeights{T} <: QueryBox{T}
    list::Memory{Pair{Int, T}}
end


struct OneBodyFullCoreIntegrals{T<:Number, D} <: IntegralData{T, OneBodyIntegral{D}}
    aa::Dict{ Tuple{   Int},  Tuple{   T}}
    ab::Dict{NTuple{2, Int}, NTuple{2, T}}

    OneBodyFullCoreIntegrals(::Type{T}, ::Val{D}) where {T, D} = 
    new{T, D}(Dict{ Tuple{Int},  Tuple{T}}(), Dict{NTuple{2, Int}, NTuple{2, T}}())
end


struct PrimOrbCoreIntegralCache{T, D, S<:MultiBodyIntegral{D}, F<:DirectOperator, 
                                I<:IntegralData{T, S}, P<:PrimOrbData{T, D}
                                } <: SpatialProcessCache{T, D}
    operator::F
    data::I
    basis::PrimOrbDataCache{T, D, P}
end

const POrb1BCoreICache{T, D, F<:DirectOperator, I<:IntegralData{T, OneBodyIntegral{D}}, 
                       P<:PrimOrbData{T, D}} = 
      PrimOrbCoreIntegralCache{T, D, OneBodyIntegral{D}, F, I, P}

const OverlapCoreCache{T, D, I<:IntegralData{T, OneBodyIntegral{D}}, 
                       P<:PrimOrbData{T, D}} = 
      POrb1BCoreICache{T, D, Identity, I, P}


function setIntegralData!(ints::OneBodyFullCoreIntegrals{T}, 
                          pair::Pair{Tuple{Int}, Tuple{T}}) where {T}
    setindex!(getfield(ints, OneBodyIdxSymDict[true ]), pair.second, pair.first)
    ints
end

function setIntegralData!(ints::OneBodyFullCoreIntegrals{T}, 
                          pair::Pair{NTuple{2, Int}, NTuple{2, T}}) where {T}
    setindex!(getfield(ints, OneBodyIdxSymDict[false]), pair.second, pair.first)
    ints
end


function getPrimCoreIntData(cache::OneBodyFullCoreIntegrals, idx::NTuple{2, Int})
    getfield(cache, OneBodyIdxSymDict[false])[idx]
end

function getPrimCoreIntData(cache::OneBodyFullCoreIntegrals, idx::Tuple{Int})
    getfield(cache, OneBodyIdxSymDict[true ])[idx]
end


struct Bar <: Any end

const OrbCoreIntLayoutAllOrbs{T, D, N, O<:PrimOrbData{T, D}} = NonEmptyTuple{O, N}
const OrbCoreIntLayoutOrbBar1{T, D, O<:PrimOrbData{T, D}} = Tuple{O, Bar, O}
const OrbCoreIntLayoutOrbBar2{T, D, O<:PrimOrbData{T, D}} = Tuple{O, O, Bar}
const OrbCoreIntLayoutOrbBar3{T, D, O<:PrimOrbData{T, D}} = Tuple{O, Bar, O, O}
const OrbCoreIntLayoutOrbBar4{T, D, O<:PrimOrbData{T, D}} = Tuple{O, O, Bar, O}

const OneBodyOrbCoreIntLayout{T, D, O<:PrimOrbData{T, D}} = Union{
    OrbCoreIntLayoutAllOrbs{T, D, 0, O}, OrbCoreIntLayoutAllOrbs{T, D, 1, O}
}

const TwoBodyOrbCoreIntLayout{T, D, O<:PrimOrbData{T, D}} = Union{
    OrbCoreIntLayoutAllOrbs{T, D, 0, O}, OrbCoreIntLayoutAllOrbs{T, D, 3, O}, 
    OrbCoreIntLayoutOrbBar1{T, D, O},    OrbCoreIntLayoutOrbBar2{T, D, O}, 
    OrbCoreIntLayoutOrbBar3{T, D, O},    OrbCoreIntLayoutOrbBar4{T, D, O}, 
}

const OrbCoreIntLayout{T, D, O<:PrimOrbData{T, D}} = 
      Union{OneBodyOrbCoreIntLayout{T, D, O}, TwoBodyOrbCoreIntLayout{T, D, O}}

const OneBodyOrbCoreIntLayoutUnion{T, D} = OneBodyOrbCoreIntLayout{T, D, PrimOrbData{T, D}}
const TwoBodyOrbCoreIntLayoutUnion{T, D} = TwoBodyOrbCoreIntLayout{T, D, PrimOrbData{T, D}}
const OrbCoreIntLayoutUnion{T, D} = OrbCoreIntLayout{T, D, PrimOrbData{T, D}}
const GaussTypeOrbIntLayout{T, D} = OrbCoreIntLayout{T, D, PrimGTOData{T, D}}


formatOrbCoreIntConfig(::P) where {P<:OrbCoreIntLayoutUnion} = P

function formatOrbCoreIntConfig(config::GaussTypeOrbIntLayout{T, D}) where {T, D}
    fieldTypes = map(config) do x
        x isa Bar ? Bar : PrimGTOData{T, D}
    end
    Tuple{fieldTypes...}
end

struct OrbCoreIntConfig{T, D, S<:MultiBodyIntegral{D}, 
                        P<:OrbCoreIntLayoutUnion{T, D}} <: StructuredType

    function OrbCoreIntConfig(::OneBodyIntegral{D}, 
                              layout::OneBodyOrbCoreIntLayoutUnion{T, D}) where {D, T}
        new{T, D, OneBodyIntegral{D}, formatOrbCoreIntConfig(layout)}()
    end

    function OrbCoreIntConfig(::TwoBodyIntegral{D}, 
                              layout::TwoBodyOrbCoreIntLayoutUnion{T, D}) where {D, T}
        new{T, D, TwoBodyIntegral{D}, formatOrbCoreIntConfig(layout)}()
    end
end


struct OrbIntCompCache{T, D, S<:MultiBodyIntegral{D}} <: CustomCache{T}
    dict::LRU{OrbCoreIntConfig{T, D, S}, CustomCache{T}}

    function OrbIntCompCache(::S, ::Type{T}) where {D, S<:MultiBodyIntegral{D}, T}
        dict = LRU{OrbCoreIntConfig{T, D, S}, CustomCache{T}}(maxsize=20)
        new{T, D, S}(dict)
    end
end

struct OrbIntNullCache{T, D, S<:MultiBodyIntegral{D}} <: CustomCache{T}

    OrbIntCompCache(::S, ::Type{T}) where {D, S<:MultiBodyIntegral{D}, T} = new{T, D, S}()
end


const OrbCoreIntegralCacheBox{T, D, S<:MultiBodyIntegral{D}} = 
      Union{OrbIntCompCache{T, D, S}, OrbIntNullCache{T, D, S}}

struct CachedOrbCoreIntegrator{T, D, F<:DirectOperator, 
                               C<:OrbCoreIntegralCacheBox{T, D}} <: ConfigBox
    operator::F
    cache::C

    function CachedOrbCoreIntegrator(::Val{true},  ::S, operator::F, ::Type{T}) where 
                                    {F<:DirectOperator, T, D, S<:MultiBodyIntegral{D}}
        new{T, D, F, OrbIntCompCache{T, D, S}}(operator, OrbIntCompCache(S(), T))
    end

    function CachedOrbCoreIntegrator(::Val{false}, ::S, operator::F, ::Type{T}) where 
                                    {F<:DirectOperator, T, D, S<:MultiBodyIntegral{D}}
        new{T, D, F, OrbIntNullCache{T, D, S}}(operator, OrbIntNullCache(S(), T))
    end
end

const DirectOrbCoreIntegrator{T, D, S<:MultiBodyIntegral{D}, F<:DirectOperator} = 
      CachedOrbCoreIntegrator{T, D, F, OrbIntNullCache{T, D, S}}

const ReusedOrbCoreIntegrator{T, D, S<:MultiBodyIntegral{D}, F<:DirectOperator} = 
      CachedOrbCoreIntegrator{T, D, F, OrbIntCompCache{T, D, S}}

const OrbCoreIntegratorConfig{T, D, S<:MultiBodyIntegral{D}, F<:DirectOperator, 
                              C<:OrbCoreIntegralCacheBox{T, D, S}} = 
      CachedOrbCoreIntegrator{T, D, F, C}

const OrbCoreOneBodyIntegrator{T, D} = OrbCoreIntegratorConfig{T, D, OneBodyIntegral{D}}


# getAnalyticIntegralCache(::Union{Identity, MonomialMul{T, D}}, ::Tuple{PrimGTOData{T, D}}
#                          ) where {T, D} = 
# NullCache{T}()

getAnalyticIntegralCache(::Union{Identity, MonomialMul{T, D}}, 
                         ::NTuple{2, PrimGTOData{T, D}}) where {T, D} = 
AxialGaussTypeOverlapCache(T, ntuple( _->Val(true), Val(D) ))

getAnalyticIntegralCache(::DirectOperator, layout::OrbCoreIntLayoutUnion{T}) where {T} = 
NullCache{T}()


function prepareAnalyticIntCache!(f::ReusedOrbCoreIntegrator{T, D, S}, 
                                  data::OrbCoreIntLayoutUnion{T, D}) where 
                                 {T, D, S<:MultiBodyIntegral{D}}
    key = OrbCoreIntConfig(S(), data)
    get!(f.cache.dict, key) do
        getAnalyticIntegralCache(f.operator, data)
    end
end

prepareAnalyticIntCache!(::DirectOrbCoreIntegrator{T, D, S}, 
                         ::OrbCoreIntLayoutUnion{T, D}) where 
                        {T, D, S<:MultiBodyIntegral{D}} = 
NullCache{T}()


function applyIntegralMethod(f::OrbCoreIntegratorConfig{T, D, S}, 
                             orbsData::OrbCoreIntLayoutUnion{T, D}) where 
                            {T, D, S<:MultiBodyIntegral{D}}
    getNumericalIntegral(S(), f.operator, orbsData)
end

function applyIntegralMethod(f::OrbCoreIntegratorConfig{T, D, S}, 
                             orbsData::GaussTypeOrbIntLayout{T, D}) where 
                            {T, D, S<:MultiBodyIntegral{D}}
    IntLayoutCache = prepareAnalyticIntCache!(f, orbsData)
    getAnalyticIntegral!(S(), IntLayoutCache, f.operator, orbsData)
end


# One-Body (i|O|j) hermiticity across O
getHermiticity(::PrimOrbData{T, D}, ::DirectOperator, ::PrimOrbData{T, D}) where {T, D} = 
false

getHermiticity(::PrimOrbData{T, D}, ::Identity, ::PrimOrbData{T, D}) where {T, D} = 
true

# Two-Body (ii|O|jj) (ii|O|jk) (ij|O|kk) (ij|O|kl) hermiticity across O
getHermiticity(::N12Tuple{PrimOrbData{T, D}}, ::Identity, 
               ::N12Tuple{PrimOrbData{T, D}}) where {T, D} = 
true


function genCoreIntTuple(integrator::OrbCoreOneBodyIntegrator{T, D}, 
                         oDataTuple::Tuple{PrimOrbData{T, D}}) where {T, D}
    (applyIntegralMethod(integrator, oDataTuple),)
end

function genCoreIntTuple(integrator::OrbCoreOneBodyIntegrator{T, D}, 
                         oDataTuple::NTuple{2, PrimOrbData{T, D}}) where {T, D}
    ijVal = applyIntegralMethod(integrator, oDataTuple)
    jiVal = if getHermiticity(first(oDataTuple), integrator.operator, last(oDataTuple))
        ijVal'
    else
        applyIntegralMethod(integrator, reverse(oDataTuple))
    end
    (ijVal, jiVal)
end


function sortTensorIndex((i, j)::NTuple{2, Int})
    if i > j
        (j, i)
    else
        (i, j)
    end
end

function sortTensorIndex((i, j, k, l)::NTuple{4, Int})
    pL = sortTensorIndex((i, j))
    pR = sortTensorIndex((k, l))
    if last(pL) > last(pR)
        (pR, pL)
    else
        (pL, pR)
    end
end

sortTensorIndex(arg::Vararg{Int}) = sortTensorIndex(arg)


function genOneBodyIntDataPairs(integrator::CachedOrbCoreIntegrator{T, D}, 
                                (oDataSeq,)::Tuple{OrbCoreDataSeq{T, D}}, 
                                (indexOffset,)::Tuple{Int}=(0,)) where {T, D}
    nOrbs = length(oDataSeq)
    firstIdx = firstindex(oDataSeq)
    offset = indexOffset + firstIdx - 1

    pairs1 = map(1:nOrbs) do i
        iiVal = genCoreIntTuple(integrator, (oDataSeq[begin+i-1],))
        (i + offset,) => iiVal
    end

    pairs2 = map(1:triMatEleNum(nOrbs-1)) do l
        m, n = convertIndex1DtoTri2D(l)
        i, j = ijPair = sortTensorIndex((m, n+1))
        oDataPair = (oDataSeq[begin+i-1], oDataSeq[begin+j-1])
        ijValPair = genCoreIntTuple(integrator, oDataPair)
        (ijPair .+ offset) => ijValPair
    end

    pairs1, pairs2
end

function genOneBodyIntDataPairs(integrator::CachedOrbCoreIntegrator{T, D}, 
                                oDataSeqPair::NTuple{2, OrbCoreDataSeq{T, D}}, 
                                indexOffsets::NTuple{2, Int}) where {T, D}
    mnOrbs = length.(oDataSeqPair)
    firstIdx = firstindex.(oDataSeqPair)
    offsets = indexOffsets .+ firstindex.(oDataSeqPair) .- 1

    map(Iterators.product( Base.OneTo.(mnOrbs)... )) do mnIdx
        idxPairOld = mnIdx .+ offsets
        idxPairNew = sortTensorIndex(idxPairOld)
        ijPair = ifelse(idxPairNew == idxPairOld, mnIdx, reverse(mnIdx))
        oDataPair = getindex.(oDataSeqPair, firstIdx .+ ijPair)
        ijValPair = genCoreIntTuple(integrator, oDataPair)
        idxPairNew => ijValPair
    end |> vec
end


function setTensorEntries!(tensor::AbstractArray{T}, (val,)::Tuple{T}, 
                           (oneBasedIdx,)::Tuple{Int}) where {T}
    idxSet = first.(axes(tensor)) .+ oneBasedIdx .- 1
    tensor[idxSet...] = val
end

function setTensorEntries!(tensor::AbstractMatrix{T}, valPair::NTuple{2, T}, 
                           oneBasedIdxPair::NTuple{2, Int}) where {T}
    i, j = first.(axes(tensor)) .+ oneBasedIdxPair .- 1
    tensor[i, j] = first(valPair)
    tensor[j, i] = last(valPair)
end


function genOneBodyPrimCoreIntTensor(integrator::CachedOrbCoreIntegrator{T, D}, 
                                     (oDataSeq,)::Tuple{OrbCoreDataSeq{T, D}}) where {T, D}
    nOrbs = length(oDataSeq)
    res = ShapedMemory{T}(undef, (nOrbs, nOrbs))

    for i in 1:nOrbs
        temp = genCoreIntTuple(integrator, (oDataSeq[begin+i-1],))
        setTensorEntries!(res, temp, (i,))
    end

    for l in 1:triMatEleNum(nOrbs-1)
        m, n = convertIndex1DtoTri2D(l)
        oDataPair = (oDataSeq[begin+m-1], oDataSeq[begin+n])
        temp = genCoreIntTuple(integrator, oDataPair)
        setTensorEntries!(res, temp, (m, n+1))
    end
    res
end

function genOneBodyPrimCoreIntTensor(integrator::CachedOrbCoreIntegrator{T, D}, 
                                     oDataSeqPair::NTuple{2, OrbCoreDataSeq{T, D}}) where 
                                    {T, D}
    oData1, oData2 = oDataSeqPair
    len1, len2 = length.(oDataSeqPair)
    res = ShapedMemory{T}(undef, (len1, len2))
    for j in 1:len2, i in 1:len1
        oDataPair = (oData1[begin+i-1], oData2[begin+j-1])
        ijVal = applyIntegralMethod(integrator, oDataPair)
        res[begin+i-1, begin+j-1] = ijVal
    end
    res
end


struct BasisIndexList <: QueryBox{Int}
    index::Memory{Int}
    endpoint::Memory{Int}

    function BasisIndexList(basisSizes::NonEmpTplOrAbtArr{Int, 1})
        checkEmptiness(basisSizes, :basisSizes)
        index = Memory{Int}(undef, sum(basisSizes))
        endpoint = Memory{Int}(undef, length(basisSizes)+1)
        endpoint[begin] = firstindex(index)
        i = firstindex(endpoint)
        for s in basisSizes
            endpoint[i+1] = endpoint[i] + s
            i += 1
        end
        new(index, endpoint)
    end

    function BasisIndexList(basisSize::Int)
        index = Memory{Int}(undef, basisSize)
        endpoint = Memory{Int}([firstindex(index), lastindex(index)+1])
        new(index, endpoint)
    end

    function BasisIndexList(idxList::BasisIndexList)
        new(copy(idxList.index), idxList.endpoint)
    end
end

function getBasisIndexRange(list::BasisIndexList, oneToIdx::Int)
    endpointList = list.endpoint
    endpointList[begin+oneToIdx-1] : (endpointList[begin+oneToIdx] - 1)
end


function genOrbCoreKey!(cache::OrbCoreMarkerDict{T, D}, 
                        data::PrimOrbData{T, D}) where {T, D}
    marker = lazyMarkObj!(cache, data.body.first)
    (data.center, data.body.second, marker)
end


function updateOrbCache!(orbCache::PrimOrbDataCache{T, D}, 
                         orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                         orbData::PrimOrbData{T, D}) where {T, D}
    basis = orbCache.list
    get!(orbCache.dict, genOrbCoreKey!(orbMarkerCache, orbData)) do
        push!(basis, orbData)
        lastindex(basis)
    end
end


function indexCacheOrbData!(orbCache::PrimOrbDataCache{T, D}, 
                            orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                            orbData::PrimOrbData{T, D}) where {T, D}
    list = BasisIndexList(1)
    list.index[] = updateOrbCache!(orbCache, orbMarkerCache, orbData)
    list
end

function indexCacheOrbData!(orbCache::PrimOrbDataCache{T, D}, 
                            orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                            orbData::CompOrbData{T, D}) where {T, D}
    orbSize = length(orbData.basis)
    list = BasisIndexList(orbSize)
    for (i, data) in enumerate(orbData.basis)
        idx = updateOrbCache!(orbCache, orbMarkerCache, data)
        list.index[begin+i-1] = idx
    end
    list
end

getOrbDataSize(orbData::PrimOrbData) = 1

getOrbDataSize(orbData::CompOrbData) = length(orbData.basis)

getSubOrbData(orbData::PrimOrbData, ::Int) = itself(orbData)

getSubOrbData(orbData::CompOrbData, i::Int) = orbData.basis[begin+i-1]

function indexCacheOrbData!(orbCache::PrimOrbDataCache{T, D}, 
                            orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                            orbsData::OrbitalDataCollection{T, D}) where {T, D}
    list = (BasisIndexList∘map)(getOrbDataSize, orbsData)
    for (j, orbData) in enumerate(orbsData)
        iRange = getBasisIndexRange(list, j)
        for (n, i) in enumerate(iRange)
            primOrbData = getSubOrbData(orbData, n)
            idx = updateOrbCache!(orbCache, orbMarkerCache, primOrbData)
            list.index[i] = idx
        end
    end
    list
end

function indexCacheOrbData!(targetCache::PrimOrbDataCache{T, D}, 
                            sourceCache::PrimOrbDataCache{T, D}, 
                            orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                            sourceList::BasisIndexList) where {T, D}
    targetList = BasisIndexList(sourceList)
    primOrbIds = targetList.index
    for i in eachindex(primOrbIds)
        orbData = sourceCache.list[sourceList.index[i]]
        primOrbIds[i] = updateOrbCache!(targetCache, orbMarkerCache, orbData)
    end
    targetList
end


function cachePrimCoreIntegrals!(intCache::PrimOrbCoreIntegralCache{T, D}, 
                                 orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                                 orbData::OrbitalDataInput{T, D}) where {T, D}
    orbCache = intCache.basis
    oldMaxIdx = lastindex(orbCache.list)
    orbIdxList = indexCacheOrbData!(orbCache, orbMarkerCache, orbData)
    updatePrimCoreIntCache!(intCache, oldMaxIdx+1)
    orbIdxList
end

function cachePrimCoreIntegrals!(targetIntCache::C, sourceIntCache::C, 
                                 orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                                 sourceOrbList::BasisIndexList) where 
                                {T, D, C<:PrimOrbCoreIntegralCache{T, D}}
    tOrbCache = targetIntCache.basis
    oldMaxIdx = lastindex(tOrbCache.list)
    sOrbCache = sourceIntCache.basis
    orbIdxList = indexCacheOrbData!(tOrbCache, sOrbCache, orbMarkerCache, sourceOrbList)
    updatePrimCoreIntCache!(targetIntCache, oldMaxIdx+1)
    orbIdxList
end


function updateIntCacheCore!(integrator::CachedOrbCoreIntegrator{T, D}, 
                             ints::OneBodyFullCoreIntegrals{T, D}, 
                             basis::Tuple{OrbCoreDataSeq{T, D}}, 
                             offset::Tuple{Int}) where {T, D}
    pairs1, pairs2 = genOneBodyIntDataPairs(integrator, basis, offset)
    foreach(p->setIntegralData!(ints, p), pairs1)
    foreach(p->setIntegralData!(ints, p), pairs2)
    ints
end

function updateIntCacheCore!(integrator::CachedOrbCoreIntegrator{T, D}, 
                             ints::OneBodyFullCoreIntegrals{T, D}, 
                             basis::NTuple{2, OrbCoreDataSeq{T, D}}, 
                             offset::NTuple{2, Int}) where {T, D}
    pairs2 = genOneBodyIntDataPairs(integrator, basis, offset)
    foreach(p->setIntegralData!(ints, p), pairs2)
    ints
end


function updatePrimCoreIntCache!(cache::PrimOrbCoreIntegralCache{T, D, S}, 
                                 startIdx::Int) where {T, D, S<:MultiBodyIntegral{D}}
    basis = cache.basis.list
    ints = cache.data
    firstIdx = firstindex(basis)
    integrator = CachedOrbCoreIntegrator(Val(true), S(), cache.operator, T)

    if startIdx == firstIdx
        updateIntCacheCore!(integrator, ints, (basis,), (0,))
    elseif firstIdx < startIdx <= lastindex(basis)
        boundary = startIdx - 1
        oldBasis = @view basis[begin:boundary]
        newBasis = @view basis[startIdx:  end]
        updateIntCacheCore!(integrator, ints, (newBasis,), (boundary,))
        updateIntCacheCore!(integrator, ints, (oldBasis, newBasis,), (0, boundary))
    end

    cache
end


function decodePrimCoreInt(cache::OneBodyFullCoreIntegrals{T}, ptrPair::NTuple{2, Int}, 
                           (coeff1, coeff2)::NTuple{2, T}) where {T}
    coeffProd = coeff1' * coeff2
    ptr1, ptr2 = ptrPair
    res = if ptr1 == ptr2
        getPrimCoreIntData(cache, (ptr1,))
    else
        ptrPairNew = sortTensorIndex(ptrPair)
        data = getPrimCoreIntData(cache, ptrPairNew)
        ptrPairNew == ptrPair ? data : reverse(data)
    end
    res .* (coeffProd, coeffProd')
end

function decodePrimCoreInt(cache::OneBodyFullCoreIntegrals{T}, ptr::Tuple{Int}, 
                           (coeff1,)::Tuple{T}=( one(T), )) where {T}
    getPrimCoreIntData(cache, ptr) .* coeff1' .* coeff1
end


function buildPrimOrbWeight(normCache::OverlapCoreCache{T, D}, data::PrimOrbData{T, D}, 
                            idx::Int) where {T, D}
    if data.renormalize
        decodePrimCoreInt(normCache.data, (idx,)) |> first |> AbsSqrtInv
    else
        one(T)
    end
end


function buildNormalizedCompOrbWeight!(weight::AbstractVector{T}, 
                                       normCache::OverlapCoreCache{T, D}, 
                                       data::CompOrbData{T, D}, 
                                       idxSeq::AbstractVector{Int}) where {T, D}
    overlapCache = normCache.data
    nPrimOrbs = getOrbDataSize(data)
    innerOverlapSum = zero(T)

    for i in 1:nPrimOrbs
        ptr = idxSeq[begin+i-1]
        innerCoreDiagOverlap = decodePrimCoreInt(overlapCache, (ptr,)) |> first
        wc = weight[begin+i-1]
        innerDiagOverlap = wc' * wc
        if data.basis[begin+i-1].renormalize
            weight[begin+i-1] *= AbsSqrtInv(innerCoreDiagOverlap)
        else
            innerDiagOverlap *= innerCoreDiagOverlap
        end
        innerOverlapSum += innerDiagOverlap
    end

    innerOverlapSum += mapreduce(+, 1:triMatEleNum(nPrimOrbs-1)) do l
        m, n = convertIndex1DtoTri2D(l)
        pointerPair = (idxSeq[begin+m-1], idxSeq[begin+n])
         scalarPair = (weight[begin+m-1], weight[begin+n])
        (sum∘decodePrimCoreInt)(overlapCache, pointerPair, scalarPair)
    end
    weight .*= AbsSqrtInv(innerOverlapSum)
end


function buildOrbWeight!(normCache::OverlapCoreCache{T, D}, orbData::PrimOrbData{T, D}, 
                         idxSeq::AbstractVector{Int}) where {T, D}
    weight = Memory{T}(undef, 1)
    weight[] = buildPrimOrbWeight(normCache, orbData, idxSeq[])
    weight
end

function buildOrbWeight!(normCache::OverlapCoreCache{T, D}, orbData::CompOrbData{T, D}, 
                         idxSeq::AbstractVector{Int}) where {T, D}
    nPrimOrbs = getOrbDataSize(orbData)
    weight = Memory{T}(orbData.weight)
    if orbData.renormalize
        buildNormalizedCompOrbWeight!(weight, normCache, orbData, idxSeq)
    else
        for (i, data) in enumerate(orbData.basis)
            ptr = idxSeq[begin+i-1]
            temp = buildPrimOrbWeight(normCache, data, ptr)
            weight[begin+i-1] *= temp
        end
    end
    weight
end


function buildIndexedOrbWeightsCore!(primOrbPtrs::AbstractVector{Int}, 
                                     primOrbWeight::AbstractVector{T}) where {T}
    nPtrs = length(primOrbWeight)
    list = Memory{Pair{Int, T}}(undef, nPtrs)
    for i in 1:nPtrs
        list[begin+i-1] = primOrbPtrs[begin+i-1] => primOrbWeight[begin+i-1]
    end
    IndexedWeights(list)
end


function buildIndexedOrbWeights!(normCache::OverlapCoreCache{T, D}, 
                                 data::OrbitalDataCollection{T, D}, 
                                 normIdxList::BasisIndexList, 
                                 intIdxList::BasisIndexList=normIdxList) where {T, D}
    i = 0
    map(data) do ele
        iRange = getBasisIndexRange(intIdxList, (i+=1))
        intIdxSeq = view(intIdxList.index, iRange)
        normIdxSeq = view(normIdxList.index, iRange)
        orbWeight = buildOrbWeight!(normCache, ele, normIdxSeq)
        buildIndexedOrbWeightsCore!(intIdxSeq, orbWeight)
    end
end

function buildIndexedOrbWeights!(normCache::OverlapCoreCache{T, D}, 
                                 orbData::OrbitalData{T, D}, 
                                 normIdxList::BasisIndexList, 
                                 intIdxList::BasisIndexList=normIdxList) where {T, D}
    orbWeight = buildOrbWeight!(normCache, orbData, normIdxList.index)
    buildIndexedOrbWeightsCore!(intIdxList.index, orbWeight)
end


function prepareIntegralConfig!(intCache::PrimOrbCoreIntegralCache{T, D}, 
                                normCache::OverlapCoreCache{T, D}, 
                                orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                                input::OrbitalDataInput{T, D}) where {T, D}
    iIdxList = cachePrimCoreIntegrals!(intCache, orbMarkerCache, input)
    if intCache === normCache
        buildIndexedOrbWeights!(intCache, input, iIdxList)
    else
        nIdxList = cachePrimCoreIntegrals!(normCache, intCache, orbMarkerCache, iIdxList)
        buildIndexedOrbWeights!(normCache, input, nIdxList, iIdxList)
    end
end


function buildIntegralEntries(intCache::POrb1BCoreICache{T}, 
                              (intWeights,)::Tuple{IndexedWeights{T}}) where {T}
    idxList = intWeights.list
    len = length(idxList)
    intValCache = intCache.data
    temp = mapreduce(+, eachindex(idxList)) do i
        ptr, coeff = idxList[i]
        (first∘decodePrimCoreInt)(intValCache, (ptr,), (coeff,))
    end
    res = mapreduce(+, 1:triMatEleNum(len-1), init=temp) do l
        m, n = convertIndex1DtoTri2D(l)
        ptr1, weight1 = idxList[begin+m-1]
        ptr2, weight2 = idxList[begin+n]
        (sum∘decodePrimCoreInt)(intValCache, (ptr1, ptr2), (weight1, weight2))
    end
    (res,) # ([1|O|1],)
end

function buildIntegralEntries(intCache::POrb1BCoreICache{T}, 
                              intWeightPair::NTuple{2, IndexedWeights{T}}) where {T}
    list1, list2 = getfield.(intWeightPair, :list)
    intValCache = intCache.data
    idxPairRange = Iterators.product(eachindex(list1), eachindex(list2))
    mapreduce(.+, idxPairRange, init=( zero(T), zero(T) )) do (i, j)
        ptr1, weight1 = list1[i]
        ptr2, weight2 = list2[j]
        decodePrimCoreInt(intValCache, (ptr1, ptr2), (weight1, weight2))
    end # ([1|O|2], [2|O|1])
end


function buildIntegralTensor(intCache::POrb1BCoreICache{T}, 
                             intWeights::AbstractVector{IndexedWeights{T}}) where {T}
    nOrbs = length(intWeights)
    res = ShapedMemory{T}(undef, (nOrbs, nOrbs))
    for i in 1:nOrbs
        iBI = intWeights[begin+i-1]
        temp = buildIntegralEntries(intCache, (iBI,))
        setTensorEntries!(res, temp, (i,))
    end

    for l in 1:triMatEleNum(nOrbs-1)
        m, n = convertIndex1DtoTri2D(l)
        mBI = intWeights[begin+m-1]
        nBI = intWeights[begin+n]
        temp = buildIntegralEntries(intCache, (mBI, nBI))
        setTensorEntries!(res, temp, (m, n+1))
    end
    res
end

function initializeIntegralCache(::OneBodyIntegral{D}, op::DirectOperator, 
                                 input::OrbitalDataInput{T, D}) where {T, D}
    orbCache = PrimOrbDataCache(T, Val(D), getPrimOrbDataTypeUnion(input))
    PrimOrbCoreIntegralCache(op, OneBodyFullCoreIntegrals(T, Val(D)), orbCache)
end

function initializeOverlapCache(input::OrbitalDataInput{T, D}) where {T, D}
    initializeIntegralCache(OneBodyIntegral{D}(), Identity(), input)
end


function initializeOneBodyCachePair(op::F, input::OrbitalDataInput{T, D}) where 
                                   {F<:DirectOperator, T, D}
    intCache = initializeIntegralCache(OneBodyIntegral{D}(), op, input)
    normCache = F <: Identity ? intCache : initializeOverlapCache(input)
    intCache, normCache
end


function computeIntTensor!(intCache::PrimOrbCoreIntegralCache{T, D}, 
                           normCache::OverlapCoreCache{T, D}, 
                           dataSet::OrbitalDataSet{T, D}; 
                           markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}()
                           ) where {T, D}
    ws = prepareIntegralConfig!(intCache, normCache, markerCache, dataSet)
    buildIntegralTensor(intCache, ws)
end


function computeIntTensor(::OneBodyIntegral{D}, op::DirectOperator, 
                          dataSet::OrbitalDataSet{T, D}; 
                          markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}()
                          ) where {T, D}
    intCache, normCache = initializeOneBodyCachePair(op, dataSet)
    computeIntTensor!(intCache, normCache, dataSet; markerCache)
end

function computeIntTensor(style::MultiBodyIntegral{D}, op::DirectOperator, 
                          orbs::OrbitalBasisSet{T, D}; 
                          cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox(), 
                          markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}()
                          ) where {T, D}
    paramSet = initializeSpanParamSet()
    orbsData =  genOrbitalData!(paramSet, orbs; cache!Self)
    computeIntTensor(style, op, orbsData; markerCache)
end


function computeIntegral!(intCache::PrimOrbCoreIntegralCache{T, D}, 
                          normCache::OverlapCoreCache{T, D}, 
                          data::NonEmptyTuple{OrbitalData{T, D}}; 
                          markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}()
                          ) where {T, D}
    ws = prepareIntegralConfig!(intCache, normCache, markerCache, data)
    buildIntegralEntries(intCache, ws) |> first
end


function computeIntegral(::OneBodyIntegral{D}, op::DirectOperator, 
                         (d1,)::Tuple{OrbitalData{T, D}}; 
                         markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}(), 
                         lazyCompute::Bool=false) where {T, D}
    if lazyCompute
        iCache, nCache = initializeOneBodyCachePair(op, d1)
        computeIntegral!(iCache, nCache, (d1,); markerCache)
    else
        coreData, w = decomposeOrbData(d1; markerCache)
        integrator = CachedOrbCoreIntegrator(Val(true), OneBodyIntegral{D}(), op, T)
        tensor = genOneBodyPrimCoreIntTensor(integrator, (coreData,))
        dot(w, tensor, w)
    end
end

function computeIntegral(::OneBodyIntegral{D}, op::DirectOperator, 
                         (d1, d2)::NTuple{2, OrbitalData{T, D}}; 
                         markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}(), 
                         lazyCompute::Bool=false) where {T, D}
    oData1, w1 = decomposeOrbData(d1; markerCache)
    oData2, w2 = decomposeOrbData(d2; markerCache)
    integrator = CachedOrbCoreIntegrator(Val(true), OneBodyIntegral{D}(), op, T)

    tensor = if lazyCompute
        oData1 = Vector(oData1)
        oData2 = Vector(oData2)
        transformation = (b::PrimOrbData{T, D})->lazyMarkObj!(markerCache, b)
        coreDataM = intersectMultisets!(oData1, oData2; transformation)
        block4 = genOneBodyPrimCoreIntTensor(integrator, (oData1, oData2))
        if isempty(coreDataM)
            block4
        else
            block1 = genOneBodyPrimCoreIntTensor(integrator, (coreDataM,))
            block2 = genOneBodyPrimCoreIntTensor(integrator, (oData1, coreDataM))
            block3 = genOneBodyPrimCoreIntTensor(integrator, (coreDataM, oData2))
            hvcat((2, 2), block1, block3, block2, block4)
        end
    else
        genOneBodyPrimCoreIntTensor(integrator, (oData1, oData2))
    end
    dot(w1, tensor, w2)
end


function decomposeOrbData!(normCache::OverlapCoreCache{T, D}, orbData::OrbitalData{T, D}; 
                           markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}()
                           ) where {T, D}
    normIdxList = cachePrimCoreIntegrals!(normCache, markerCache, orbData)
    orbWeight = buildOrbWeight!(normCache, orbData, normIdxList.index)
    normCache.basis.list[normIdxList.index], orbWeight
end

function decomposeOrbData(orbData::OrbitalData{T, D}; 
                          markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}()
                          ) where {T, D}
    normCache = initializeOverlapCache(orbData)
    decomposeOrbData!(normCache, orbData; markerCache)
end


function computeIntegral(::OneBodyIntegral{D}, ::Identity, 
                         orbDataPair::NTuple{2, OrbitalData{T, D}}; 
                         markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}(), 
                         lazyCompute::Bool=false) where {T, D}
    op = Identity()
    d1, d2 = orbDataPair

    if lazyCompute
        if d1 === d2
            computeIntegral(OneBodyIntegral{D}(), op, (d1,); markerCache, lazyCompute)
        else
            normCache = initializeOverlapCache(orbDataPair)
            computeIntegral!(normCache, normCache, orbDataPair; markerCache)
        end
    else
        oData1, w1 = decomposeOrbData(d1; markerCache)
        oData2, w2 = decomposeOrbData(d2; markerCache)
        integrator = CachedOrbCoreIntegrator(Val(true), OneBodyIntegral{D}(), op, T)
        tensor = genOneBodyPrimCoreIntTensor(integrator, (oData1, oData2))
        dot(w1, tensor, w2)
    end
end

function computeIntegral(style::MultiBodyIntegral, op::DirectOperator, 
                         orbs::NonEmptyTuple{OrbitalBasis{T, D}}; 
                         cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox(), 
                         lazyCompute::Bool=false) where {T, D}
    paramSet = initializeSpanParamSet()
    orbsData = genOrbitalData!(paramSet, orbs; cache!Self)
    computeIntegral(style, op, orbsData; lazyCompute)
end