using LinearAlgebra: dot

const OneBodyIdxSymDict = let tempDict=Base.ImmutableDict(false=>:aa)
    Base.ImmutableDict(tempDict, true=>:ab)
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

#? Consider replacing it with LRU-based cache to control memory consumption
struct PrimOrbDataCache{T<:Real, D, P<:PrimOrbData{T, D}}
    dict::Dict{FieldMarker{:StashedField, 2}, OneToIndex}
    list::Vector{P}
    function PrimOrbDataCache(::Type{T}, ::Val{D}, ::Type{P}=PrimOrbData{T, D}) where 
                             {T<:Real, D, P<:PrimOrbData{T, D}}
    new{T, D, P}(Dict{FieldMarker{:StashedField, 2}, OneToIndex}(), P[])
    end
end

#? Consider replacing it with LRU-based cache to control memory consumption
struct OneBodyFullCoreIntegrals{C<:RealOrComplex} <: QueryBox{C}
    aa::Dict{ Tuple{   OneToIndex},  Tuple{   C}}
    ab::Dict{NTuple{2, OneToIndex}, NTuple{2, C}}

    function OneBodyFullCoreIntegrals(::Type{C}) where {C<:RealOrComplex}
        aaSector = Dict{ Tuple{   OneToIndex},  Tuple{   C}}()
        abSector = Dict{NTuple{2, OneToIndex}, NTuple{2, C}}()
        new{C}(aaSector, abSector)
    end
end

const OneBodyCoreIntegrals{C<:RealOrComplex} = Union{OneBodyFullCoreIntegrals{C}}

const MultiBodyCoreIntegrals{C<:RealOrComplex} = Union{OneBodyCoreIntegrals{C}}


struct PrimOrbCoreIntegralCache{T<:Real, D, C<:RealOrComplex{T}, F<:DirectOperator, 
                                I<:MultiBodyCoreIntegrals{C}, P<:PrimOrbData{T, D}
                                } <: IntegralData{C}
    operator::F
    data::I
    basis::PrimOrbDataCache{T, D, P}
end

const POrb1BCoreICache{T<:Real, D, C<:RealOrComplex{T}, F<:DirectOperator, 
                       I<:OneBodyCoreIntegrals{C}, P<:PrimOrbData{T, D}} = 
      PrimOrbCoreIntegralCache{T, D, C, F, I, P}

const OverlapCoreCache{T<:Real, D, C<:RealOrComplex{T}, I<:OneBodyCoreIntegrals{C}, 
                       P<:PrimOrbData{T, D}} = 
      POrb1BCoreICache{T, D, C, OverlapSampler, I, P}


getIntegralStyle(::POrb1BCoreICache{<:Real, D}) where {D} = OneBodyIntegral{D}()


function setPrimCoreIntData!(ints::OneBodyFullCoreIntegrals{C}, 
                             pair::Pair{NonEmptyTuple{OneToIndex, N}, NonEmptyTuple{C, N}}
                             ) where {C<:RealOrComplex, N}
    setindex!(getfield(ints, OneBodyIdxSymDict[Bool(N)]), pair.second, pair.first)
    ints
end


function getPrimCoreIntData(cache::OneBodyFullCoreIntegrals, 
                            idx::NonEmptyTuple{OneToIndex, N}) where {N}
    getfield(cache, OneBodyIdxSymDict[Bool(N)])[idx]
end


const CoreIntegralOrbDataLayout{T<:Real, D} = DualN12Tuple{PrimOrbData{T, D}}


@enum OrbitalCategory::Int8 begin
    PrimGaussTypeOrb
    ArbitraryTypeOrb
end

getOrbitalCategory(::PGTOrbData) = PrimGaussTypeOrb
getOrbitalCategory(::PrimOrbData) = ArbitraryTypeOrb

function genOrbCategoryLayout(data::CoreIntegralOrbDataLayout)
    map(data) do pair
        map(getOrbitalCategory, pair)
    end
end


const OrbIntLayoutInfo{N} = 
      Tuple{TypeBox{<:DirectOperator}, NTuple{N, NTuple{2, OrbitalCategory}}}

const OrbIntLayoutCache{T, C<:RealOrComplex{T}, N, M<:Union{CustomCache{T}, CustomCache{C}}
                        } = LRU{OrbIntLayoutInfo{N}, M}

struct OrbitalCoreIntegralConfig{T<:Real, D, C<:RealOrComplex{T}, N, F<:DirectOperator, 
                                 M<:Union{NullCache{C}, OrbIntLayoutCache{T, C, N}},
                                 E<:MissingOr{ EstimatorConfig{T} }} <: ConfigBox
    operator::F
    cache::M
    estimator::E

    function OrbitalCoreIntegralConfig(::Val{L}, ::S, operator::F, ::Type{C}, 
                                     config::E=missing) where 
                                    {L, N, D, S<:MultiBodyIntegral{N, D}, 
                                     F<:DirectOperator, T<:Real, C<:RealOrComplex{T}, 
                                     E<:MissingOr{ EstimatorConfig{T} }}
        cache = if L
            valueTypeBound = Union{CustomCache{T}, CustomCache{C}}
            LRU{OrbIntLayoutInfo{N}, valueTypeBound}(maxsize=20)
        else
            MultiBodyNullCache(S(), C)
        end
        new{T, D, C, N, F, typeof(cache), E}(operator, cache, config)
    end
end


function evaluateIntegral!(config::OrbitalCoreIntegralConfig{T, D, C, N, F}, 
                           pairwiseData::NTuple{N, NTuple{ 2, PrimOrbData{T, D} }}) where 
                          {T, C<:RealOrComplex{T}, D, N, F<:DirectOperator}
    formattedOp = TypedOperator(config.operator, C)
    estimateOrbIntegral(config.estimator, formattedOp, pairwiseData)::C
end


#> One-Body (i|O|j) and two-body (ij|O|kl) hermiticity across O
getHermiticity(::DirectOperator, ::DualN12Tuple{OrbitalCategory}) = false

getHermiticity(::OverlapSampler, ::DualN12Tuple{OrbitalCategory}) = true

getHermiticity(::MultipoleMomentSampler, ::DualN12Tuple{OrbitalCategory}) = true

function getHermiticity(::DiagDirectionalDiffSampler, layout::N12Tuple{OrbitalCategory})
    all(layout) do c
        c == PrimGaussTypeOrb
    end
end


function genCoreIntTuple(config::OrbitalCoreIntegralConfig{T, D, C, 1}, 
                         (data,)::Tuple{PrimOrbData{T, D}}) where 
                        {T<:Real, D, C<:RealOrComplex{T}}
    integralRes = evaluateIntegral!(config, ((data, data),))
    (convert(C, integralRes),)
end

function genCoreIntTuple(config::OrbitalCoreIntegralConfig{T, D, C, 1}, 
                         pairData::NTuple{2, PrimOrbData{T, D}}) where 
                        {T<:Real, D, C<:RealOrComplex{T}}
    ijVal = evaluateIntegral!(config, (pairData,))
    jiVal = if getHermiticity(config.operator, genOrbCategoryLayout(pairData|>tuple))
        conj(ijVal)
    else
        evaluateIntegral!(config, (reverse(pairData),))
    end
    (convert(C, ijVal), convert(C, jiVal))
end


function genOneBodyIntDataPairs(config::OrbitalCoreIntegralConfig{T, D, C}, 
                                (oDataSeq,)::Tuple{PrimOrbDataVec{T, D}}, 
                                (offset,)::Tuple{Int}=(0,)) where 
                               {T<:Real, D, C<:RealOrComplex{T}}
    nOrbs = length(oDataSeq)

    pairs1 = Memory{Pair{ Tuple{OneToIndex}, Tuple{C} }}(undef, nOrbs)
    for (i, oData) in enumerate(oDataSeq)
        iiVal = genCoreIntTuple(config, (oData,))
        pairs1[begin+i-1] = (OneToIndex(i + offset, False()),) => iiVal
    end

    nTri = triMatEleNum(nOrbs-1)
    pairs2 = Memory{Pair{ NTuple{2, OneToIndex}, NTuple{2, C} }}(undef, nTri)
    for l in 1:nTri
        m, n = convertIndex1DtoTri2D(l)
        i, j = ijPair = sortTensorIndex((m, n+1))
        ijValPair = genCoreIntTuple(config, (oDataSeq[begin+i-1], oDataSeq[begin+j-1]))
        pairs2[begin+l-1] = OneToIndex.(ijPair .+ offset, False()) => ijValPair
    end

    pairs1, pairs2
end

function genOneBodyIntDataPairs(config::OrbitalCoreIntegralConfig{T, D}, 
                                oDataSeqPair::NTuple{2, PrimOrbDataVec{T, D}}, 
                                offsets::NTuple{2, Int}) where {T<:Real, D}
    seqL, seqR = oDataSeqPair
    nOrbsL = length(seqL)
    nOrbsR = length(seqR)
    pairs = Memory{Pair{ NTuple{2, OneToIndex}, NTuple{2, C} }}(undef, nOrbsL*nOrbsR)
    k = firstindex(pairs) - 1

    for j in 1:nOrbsR, i in 1:nOrbsL
        k += 1
        ijIdx = (i, j)
        idxPairOld = ijIdx .+ offsets
        idxPairNew = sortTensorIndex(idxPairOld)
        m, n = ifelse(idxPairNew==idxPairOld, ijIdx, reverse(ijIdx))
        ijValPair = genCoreIntTuple(config, (seqL[begin+m-1], seqR[begin+n-1]))
        pairs[k] = OneToIndex.(idxPairNew, False()) => ijValPair
    end

    pairs
end


function setTensorEntries!(tensor::AbstractArray{C}, (val,)::Tuple{C}, 
                           (oneBasedIdx,)::Tuple{Int}) where {C<:RealOrComplex}
    idxSet = first.(axes(tensor)) .+ oneBasedIdx .- 1
    tensor[idxSet...] = val
end

function setTensorEntries!(tensor::AbstractMatrix{C}, valPair::NTuple{2, C}, 
                           oneBasedIdxPair::NTuple{2, Int}) where {C<:RealOrComplex}
    i, j = first.(axes(tensor)) .+ oneBasedIdxPair .- 1
    tensor[i, j] = first(valPair)
    tensor[j, i] = last(valPair)
end


function genOneBodyPrimCoreIntTensor(config::OrbitalCoreIntegralConfig{T, D, C}, 
                                     (oDataSeq,)::Tuple{PrimOrbDataVec{T, D}}
                                     ) where {T<:Real, D, C<:RealOrComplex{T}}
    nOrbs = length(oDataSeq)
    res = ShapedMemory{C}(undef, (nOrbs, nOrbs))
    checkAxialIndexStep(res)

    for i in 1:nOrbs
        temp = genCoreIntTuple(config, (oDataSeq[begin+i-1],))
        setTensorEntries!(res, temp, (i,))
    end

    for l in 1:triMatEleNum(nOrbs-1)
        m, n = convertIndex1DtoTri2D(l)
        temp = genCoreIntTuple(config, (oDataSeq[begin+m-1], oDataSeq[begin+n]))
        setTensorEntries!(res, temp, (m, n+1))
    end
    res
end

function genOneBodyPrimCoreIntTensor(config::OrbitalCoreIntegralConfig{T, D}, 
                                     oDataSeqPair::NTuple{2, PrimOrbDataVec{T, D}}
                                     ) where {T<:Real, D}
    oData1, oData2 = oDataSeqPair
    len1, len2 = length.(oDataSeqPair)
    res = ShapedMemory{T}(undef, (len1, len2))
    checkAxialIndexStep(res)

    for j in 1:len2, i in 1:len1
        ijVal = evaluateIntegral!(config, ((oData1[begin+i-1], oData2[begin+j-1]),))
        res[begin+i-1, begin+j-1] = ijVal
    end

    res
end


struct BasisIndexList <: QueryBox{OneToIndex}
    index::Memory{OneToIndex}    #> Flattened sectors of primitive indices for all bases
    endpoint::Memory{OneToIndex} #> Endpoints to separate the sectors of primitive indices

    function BasisIndexList(sectorSizes::LinearSequence{Int})
        checkEmptiness(sectorSizes, :sectorSizes)
        index = Memory{OneToIndex}(undef, sum(sectorSizes))
        endpoint = Memory{OneToIndex}(undef, length(sectorSizes)+1)
        endpoint[begin] = OneToIndex() #> First endpoint is 1 as the anchor
        i = firstindex(endpoint)
        for s in sectorSizes
            endpoint[i+1] = endpoint[i] + s
            i += 1
        end
        new(index, endpoint)
    end

    function BasisIndexList(sectorSize::Int)
        checkPositivity(sectorSize)
        index = Memory{OneToIndex}(undef, sectorSize)
        endpoint = Memory{OneToIndex}(undef, 2)
        endpoint[begin] = OneToIndex()
        endpoint[ end ] = OneToIndex(sectorSize+1, False())
        new(index, endpoint)
    end

    function BasisIndexList(idxList::BasisIndexList)
        new(copy(idxList.index), idxList.endpoint) #> Assuming .endpoint is read-only
    end
end

function getBasisIndexRange(list::BasisIndexList, idx::OneToIndex)
    endpointList = list.endpoint
    i = shiftLinearIndex(endpointList, idx)
    (endpointList[i], (endpointList[i+1] - 1))
end


const OrbCoreMarkerDict{T<:Real, D} = 
      Dict{EgalBox{StashedField{T, D}}, FieldMarker{:StashedField, 2}}

function genOrbCoreKey!(cache::OrbCoreMarkerDict{T, D}, 
                        data::PrimOrbData{T, D}) where {T<:Real, D}
    innerField = data.core
    lazyMarkObj!(cache, innerField)
end #? Replace `PrimOrbData` with `StashedField`?


function updateOrbCache!(orbCache::PrimOrbDataCache{T, D}, 
                         orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                         orbData::PrimOrbData{T, D}) where {T<:Real, D}
    basis = orbCache.list
    get!(orbCache.dict, genOrbCoreKey!(orbMarkerCache, orbData)) do
        push!(basis, orbData)
        OneToIndex(length(basis), False())
    end
end


function indexCacheOrbData!(orbCache::PrimOrbDataCache{T, D}, 
                            orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                            orbData::PrimOrbData{T, D}) where {T<:Real, D}
    list = BasisIndexList(1)
    list.index[] = updateOrbCache!(orbCache, orbMarkerCache, orbData)
    list
end

function indexCacheOrbData!(orbCache::PrimOrbDataCache{T, D}, 
                            orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                            orbData::CompOrbData{T, D}) where {T<:Real, D}
    orbSize = length(orbData.basis)
    list = BasisIndexList(orbSize)
    listIndices = list.index
    for (i, data) in zip(eachindex(listIndices), orbData.basis)
        oneToIdx = updateOrbCache!(orbCache, orbMarkerCache, data)
        listIndices[i] = oneToIdx
    end
    list
end

getOrbDataSize(orbData::PrimOrbData) = 1

getOrbDataSize(orbData::CompOrbData) = length(orbData.basis)

getSubOrbData(orbData::PrimOrbData, ::OneToIndex) = itself(orbData)

getSubOrbData(orbData::CompOrbData, i::OneToIndex) = getEntry(orbData.basis, i)

function indexCacheOrbData!(orbCache::PrimOrbDataCache{T, D}, 
                            orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                            orbsData::OrbDataCollection{T, D}) where {T<:Real, D}
    list = map(getOrbDataSize, orbsData) |> BasisIndexList
    listIndices = list.index
    for (j, orbData) in enumerate(orbsData)
        oneToStart, oneToFinal = getBasisIndexRange(list, OneToIndex( j, False() ))
        for (n, i) in enumerate(oneToStart.idx : oneToFinal.idx)
            primOrbData = getSubOrbData(orbData, OneToIndex( n, False() ))
            oneToIdx = updateOrbCache!(orbCache, orbMarkerCache, primOrbData)
            listIndices[shiftLinearIndex(listIndices, i)] = oneToIdx
        end
    end
    list
end

function indexCacheOrbData!(targetCache::PrimOrbDataCache{T, D}, 
                            sourceCache::PrimOrbDataCache{T, D}, 
                            orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                            sourceList::BasisIndexList) where {T<:Real, D}
    targetList = BasisIndexList(sourceList) #> targetList is a copy of sourceList
    primOrbIds = targetList.index
    for i in eachindex(primOrbIds)
        orbData = getEntry(sourceCache.list, sourceList.index[i])
        primOrbIds[i] = updateOrbCache!(targetCache, orbMarkerCache, orbData)
    end
    targetList
end


#> Compute and cache `PrimOrbData` and core integrals
function cachePrimCoreIntegrals!(intCache::PrimOrbCoreIntegralCache{T, D}, 
                                 orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                                 orbData::OrbDataSource{T, D}) where {T<:Real, D}
    orbCache = intCache.basis
    nCacheOld = length(orbCache.list)
    orbIdxList = indexCacheOrbData!(orbCache, orbMarkerCache, orbData)
    updatePrimCoreIntCache!(intCache, OneToIndex( nCacheOld+1, False() ))
    orbIdxList
end
#> First try loading existing `PrimOrbData` from `sourceIntCache` to `targetIntCache`
function cachePrimCoreIntegrals!(targetIntCache::PrimOrbCoreIntegralCache{T, D}, 
                                 sourceIntCache::PrimOrbCoreIntegralCache{T, D}, 
                                 orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                                 sourceOrbList::BasisIndexList) where {T<:Real, D}
    tOrbCache = targetIntCache.basis
    nCacheOld = length(tOrbCache.list)
    sOrbCache = sourceIntCache.basis
    orbIdxList = indexCacheOrbData!(tOrbCache, sOrbCache, orbMarkerCache, sourceOrbList)
    updatePrimCoreIntCache!(targetIntCache, OneToIndex( nCacheOld+1, False() ))
    orbIdxList
end


function updateIntCacheCore!(config::OrbitalCoreIntegralConfig{T, D, C, 1}, 
                             ints::OneBodyCoreIntegrals{C}, 
                             basis::Tuple{PrimOrbDataVec{T, D}}, 
                             offset::Tuple{Int}) where {T<:Real, D, C<:RealOrComplex{T}}
    pairs1, pairs2 = genOneBodyIntDataPairs(config, basis, offset)
    foreach(p->setPrimCoreIntData!(ints, p), pairs1)
    foreach(p->setPrimCoreIntData!(ints, p), pairs2)
    ints
end

function updateIntCacheCore!(config::OrbitalCoreIntegralConfig{T, D, C, 1}, 
                             ints::OneBodyCoreIntegrals{C}, 
                             basis::NTuple{2, PrimOrbDataVec{T, D}}, 
                             offset::NTuple{2, Int}) where {T<:Real, D, C<:RealOrComplex{T}}
    pairs2 = genOneBodyIntDataPairs(config, basis, offset)
    foreach(p->setPrimCoreIntData!(ints, p), pairs2)
    ints
end


function updatePrimCoreIntCache!(cache::PrimOrbCoreIntegralCache{T, D, C}, 
                                 startIdx::OneToIndex) where 
                                {T<:Real, D, C<:RealOrComplex{T}}
    ints = cache.data
    basis = cache.basis.list
    intStyle = getIntegralStyle(cache)
    config = OrbitalCoreIntegralConfig(Val(true), intStyle, cache.operator, C)

    startNum = Int(startIdx)
    offset = startNum - 1

    if offset == 0
        updateIntCacheCore!(config, ints, (basis,), (0,))
    elseif offset < length(basis)
        idxStart = firstindex(basis) + offset
        oldBasis = @view basis[begin:(idxStart-1)]
        newBasis = @view basis[idxStart:      end]
        updateIntCacheCore!(config, ints, (newBasis,), (offset,))
        updateIntCacheCore!(config, ints, (oldBasis, newBasis,), (0, offset))
    end

    cache
end


function decodePrimCoreInt(cache::OneBodyFullCoreIntegrals{C1}, 
                           ptrPair::NTuple{2, OneToIndex}, (coeff1, coeff2)::NTuple{2, C2}
                           ) where {T<:Real, C1<:RealOrComplex{T}, C2<:RealOrComplex{T}}
    coeffProd = conj(coeff1) * coeff2
    ptr1, ptr2 = ptrPair
    res = if ptr1 == ptr2
        getPrimCoreIntData(cache, (ptr1,))
    else
        ptrPairNew = OneToIndex.(sortTensorIndex(getfield.(ptrPair, :idx)), False())
        data = getPrimCoreIntData(cache, ptrPairNew)
        ptrPairNew == ptrPair ? data : reverse(data)
    end
    res .* (coeffProd, conj(coeffProd))
end

function decodePrimCoreInt(cache::OneBodyFullCoreIntegrals{C1}, ptr::Tuple{OneToIndex}, 
                           (coeff1,)::Tuple{C2}=(one(C1),)) where 
                          {T<:Real, C1<:RealOrComplex{T}, C2<:RealOrComplex{T}}
    getPrimCoreIntData(cache, ptr) .* conj(coeff1) .* coeff1
end


function buildPrimOrbWeight(normCache::OverlapCoreCache{T, D, C}, data::PrimOrbData{T, D}, 
                            idx::OneToIndex) where {T<:Real, D, C<:RealOrComplex{T}}
    if data.renormalize
        convert(C, decodePrimCoreInt(normCache.data, (idx,)) |> first |> absSqrtInv)
    else
        one(C)
    end
end


function buildNormalizedCompOrbWeight!(weight::AbstractVector{C}, 
                                       primOrbsNormCache::OverlapCoreCache{T, D, C}, 
                                       primOrbsData::AbstractVector{<:PrimOrbData{T, D}}, 
                                       idxSeq::AbstractVector{OneToIndex}) where 
                                      {T<:Real, D, C<:RealOrComplex{T}}
    overlapCache = primOrbsNormCache.data
    innerOverlapSum = zero(T)

    for (innerIdx, data, i) in zip(idxSeq, primOrbsData, eachindex(weight))
        w = weight[i]
        innerDiagOverlap = conj(w) * w
        innerCoreDiagOverlap = decodePrimCoreInt(overlapCache, (innerIdx,)) |> first

        if data.renormalize
            weight[i] *= absSqrtInv(innerCoreDiagOverlap)
        else
            innerDiagOverlap *= innerCoreDiagOverlap
        end

        innerOverlapSum += innerDiagOverlap
    end

    for l in 1:triMatEleNum(length(primOrbsData)-1)
        m, n = convertIndex1DtoTri2D(l)
        pointerPair = (idxSeq[begin+m-1], idxSeq[begin+n])
         scalarPair = (weight[begin+m-1], weight[begin+n])
        innerOverlapSum += (decodePrimCoreInt(overlapCache, pointerPair, scalarPair) |> sum)
    end

    weight .*= absSqrtInv(innerOverlapSum)
end


function buildOrbWeight!(normCache::OverlapCoreCache{T, D, C}, orbData::PrimOrbData{T, D}, 
                         idxSeq::AbstractVector{OneToIndex}) where 
                        {T<:Real, D, C<:RealOrComplex{T}}
    weight = Memory{C}(undef, 1)
    weight[] = buildPrimOrbWeight(normCache, orbData, idxSeq[])
    weight
end

function buildOrbWeight!(normCache::OverlapCoreCache{T, D, C}, orbData::CompOrbData{T, D}, 
                         idxSeq::AbstractVector{OneToIndex}) where 
                        {T<:Real, D, C<:RealOrComplex{T}}
    weight = Memory{C}(orbData.weight)
    if orbData.renormalize
        buildNormalizedCompOrbWeight!(weight, normCache, orbData.basis, idxSeq)
    else
        for (i, idx, data) in zip(eachindex(weight), idxSeq, orbData.basis)
            temp = buildPrimOrbWeight(normCache, data, idx)
            weight[i] *= temp
        end
    end
    weight
end


const IndexedMemory{T} = Memory{Pair{OneToIndex, T}}

function buildIndexedOrbWeightsCore!(primOrbPtrs::AbstractVector{OneToIndex}, 
                                     primOrbWeight::AbstractVector{C}) where 
                                    {C<:RealOrComplex}
    nPtrs = length(primOrbWeight)
    list = IndexedMemory{C}(undef, nPtrs)
    for (i, p, w) in zip(eachindex(list), primOrbPtrs, primOrbWeight)
        list[i] = (p => w)
    end
    list
end


function buildIndexedOrbWeights!(normCache::OverlapCoreCache{T, D}, 
                                 data::OrbDataCollection{T, D}, 
                                 normIdxList::BasisIndexList, 
                                 intIdxList::BasisIndexList=normIdxList) where {T<:Real, D}
    i = 0
    map(data) do ele
        oneToStart, oneToFinal = getBasisIndexRange(intIdxList, OneToIndex( i+=1, False() ))
        oneToRange = oneToStart.idx : oneToFinal.idx
         iIdxRange = shiftLinearIndex( intIdxList.index, oneToRange)
         nIdxRange = shiftLinearIndex(normIdxList.index, oneToRange)
         intIdxSeq = view( intIdxList.index, iIdxRange)
        normIdxSeq = view(normIdxList.index, nIdxRange)
        orbWeight = buildOrbWeight!(normCache, ele, normIdxSeq)
        buildIndexedOrbWeightsCore!(intIdxSeq, orbWeight)
    end
end

function buildIndexedOrbWeights!(normCache::OverlapCoreCache{T, D}, 
                                 orbData::OrbitalData{T, D}, 
                                 normIdxList::BasisIndexList, 
                                 intIdxList::BasisIndexList=normIdxList) where {T<:Real, D}
    orbWeight = buildOrbWeight!(normCache, orbData, normIdxList.index)
    buildIndexedOrbWeightsCore!(intIdxList.index, orbWeight)
end


function prepareIntegralConfig!(intCache::PrimOrbCoreIntegralCache{T, D}, 
                                normCache::OverlapCoreCache{T, D}, 
                                orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                                input::OrbDataSource{T, D}) where {T<:Real, D}
    iIdxList = cachePrimCoreIntegrals!(intCache, orbMarkerCache, input)
    if intCache === normCache
        buildIndexedOrbWeights!(intCache, input, iIdxList)
    else
        nIdxList = cachePrimCoreIntegrals!(normCache, intCache, orbMarkerCache, iIdxList)
        buildIndexedOrbWeights!(normCache, input, nIdxList, iIdxList)
    end
end


function buildIntegralEntries(intCache::POrb1BCoreICache{T, D, C}, 
                              (intWeights,)::Tuple{IndexedMemory{C}}) where 
                             {T<:Real, D, C<:RealOrComplex{T}}
    intValCache = intCache.data
    res = zero(C)

    for (ptr, coeff) in intWeights
        res += (decodePrimCoreInt(intValCache, (ptr,), (coeff,)) |> first)
    end

    for l in 1:triMatEleNum(length(intWeights)-1)
        m, n = convertIndex1DtoTri2D(l)
        ptr1, weight1 = intWeights[begin+m-1]
        ptr2, weight2 = intWeights[begin+n  ]
        res += (decodePrimCoreInt(intValCache, (ptr1, ptr2), (weight1, weight2)) |> sum)
    end

    (res,) # ([1|O|1],)
end

function buildIntegralEntries(intCache::POrb1BCoreICache{T, D, C}, 
                              intWeightPair::NTuple{2, IndexedMemory{C}}) where 
                             {T<:Real, D, C<:RealOrComplex{T}}
    pairs1, pairs2 = intWeightPair
    intValCache = intCache.data
    res = (zero(C), zero(C))

    for pair2 in pairs2, pair1 in pairs1
        ptr1, weight1 = pair1
        ptr2, weight2 = pair2
        res = res .+ decodePrimCoreInt(intValCache, (ptr1, ptr2), (weight1, weight2))
    end # ([1|O|2], [2|O|1])

    res
end


function buildIntegralTensor(intCache::POrb1BCoreICache{T, D, C}, 
                             intWeights::AbstractVector{IndexedMemory{C}}) where 
                            {T<:Real, D, C<:RealOrComplex{T}}
    nOrbs = length(intWeights)
    res = ShapedMemory{C}(undef, (nOrbs, nOrbs))
    checkAxialIndexStep(res)

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
                                 input::OrbDataSource{T, D}) where {T<:Real, D}
    C = getOrbOutputTypeUnion(input)
    orbCache = PrimOrbDataCache(T, Val(D), getPrimOrbDataTypeUnion(input))
    PrimOrbCoreIntegralCache(op, OneBodyFullCoreIntegrals(C), orbCache)
end

function initializeOverlapCache(input::OrbDataSource{T, D}) where {T<:Real, D}
    initializeIntegralCache(OneBodyIntegral{D}(), genOverlapSampler(), input)
end


function initializeOneBodyCachePair(op::F, input::OrbDataSource{T, D}) where 
                                   {F<:DirectOperator, T<:Real, D}
    intCache = initializeIntegralCache(OneBodyIntegral{D}(), op, input)
    normCache = op isa OverlapSampler ? intCache : initializeOverlapCache(input)
    intCache, normCache
end


function computeIntTensor!(intCache::PrimOrbCoreIntegralCache{T, D}, 
                           normCache::OverlapCoreCache{T, D}, 
                           dataSet::OrbDataVec{T, D}; 
                           markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}()
                           ) where {T<:Real, D}
    ws = prepareIntegralConfig!(intCache, normCache, markerCache, dataSet)
    buildIntegralTensor(intCache, ws)
end


function computeIntTensor(::OneBodyIntegral{D}, op::DirectOperator, 
                          dataSet::OrbDataVec{T, D}; 
                          markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}()
                          ) where {T<:Real, D}
    intCache, normCache = initializeOneBodyCachePair(op, dataSet)
    computeIntTensor!(intCache, normCache, dataSet; markerCache)
end

function computeIntTensor(style::MultiBodyIntegral{N, D}, op::DirectOperator, 
                          orbs::OrbBasisVec{T, D}; 
                          cache!Self::ParamDataCache=initializeParamDataCache(), 
                          markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}()
                          ) where {T<:Real, N, D}
    orbsData = genOrbitalData(orbs, isParamIndependent(op); cache!Self)
    computeIntTensor(style, op, orbsData; markerCache)
end


function computeIntegral!(intCache::PrimOrbCoreIntegralCache{T, D}, 
                          normCache::OverlapCoreCache{T, D}, 
                          data::NonEmptyTuple{OrbitalData{T, D}}; 
                          markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}()
                          ) where {T<:Real, D}
    ws = prepareIntegralConfig!(intCache, normCache, markerCache, data)
    buildIntegralEntries(intCache, ws) |> first
end


function computeIntegral(::OneBodyIntegral{D}, op::DirectOperator, 
                         (d1,)::Tuple{OrbitalData{T, D, C}}; 
                         markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}(), 
                         lazyCompute::Bool=false) where {T<:Real, D, C<:RealOrComplex{T}}
    if lazyCompute
        iCache, nCache = initializeOneBodyCachePair(op, d1)
        computeIntegral!(iCache, nCache, (d1,); markerCache)
    else
        coreData, w = decomposeOrbData(d1; markerCache)
        config = OrbitalCoreIntegralConfig(Val(true), OneBodyIntegral{D}(), op, C)
        tensor = genOneBodyPrimCoreIntTensor(config, (coreData,))
        dot(w, tensor, w)
    end
end

function computeIntegral(::OneBodyIntegral{D}, op::DirectOperator, 
                         (d1, d2)::NTuple{2, OrbitalData{T, D}}; 
                         markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}(), 
                         lazyCompute::Bool=false) where {T<:Real, D}
    oPairType = getOrbOutputTypeUnion((d1, d2))
    oData1, w1 = decomposeOrbData(d1; markerCache)
    oData2, w2 = decomposeOrbData(d2; markerCache)
    config = OrbitalCoreIntegralConfig(Val(true), OneBodyIntegral{D}(), op, oPairType)

    tensor = if lazyCompute
        oData1 = Vector(oData1)
        oData2 = Vector(oData2)
        #> Find the shared `PrimOrbData` (excluding the renormalization information)
        transformation = (b::PrimOrbData{T, D})->genOrbCoreKey!(markerCache, b)
        coreDataM = intersectMultisets!(oData1, oData2; transformation)
        block4 = genOneBodyPrimCoreIntTensor(config, (oData1, oData2))
        if isempty(coreDataM)
            block4
        else
            block1 = genOneBodyPrimCoreIntTensor(config, (coreDataM,))
            block2 = genOneBodyPrimCoreIntTensor(config, (oData1, coreDataM))
            block3 = genOneBodyPrimCoreIntTensor(config, (coreDataM, oData2))
            hvcat((2, 2), block1, block3, block2, block4)
        end
    else
        genOneBodyPrimCoreIntTensor(config, (oData1, oData2))
    end
    dot(w1, tensor, w2)
end


function decomposeOrbData!(normCache::OverlapCoreCache{T, D}, orbData::OrbitalData{T, D}; 
                           markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}()
                           ) where {T<:Real, D}
    normIdxList = cachePrimCoreIntegrals!(normCache, markerCache, orbData)
    orbWeight = buildOrbWeight!(normCache, orbData, normIdxList.index)
    map(i->getEntry(normCache.basis.list, i), normIdxList.index), orbWeight
end

function decomposeOrbData(orbData::OrbitalData{T, D}; 
                          markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}()
                          ) where {T<:Real, D}
    normCache = initializeOverlapCache(orbData)
    decomposeOrbData!(normCache, orbData; markerCache)
end


function computeIntegral(::OneBodyIntegral{D}, op::OverlapSampler, 
                         orbDataPair::NTuple{2, OrbitalData{T, D}}; 
                         markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}(), 
                         lazyCompute::Bool=false) where {T<:Real, D}
    d1, d2 = orbDataPair

    if lazyCompute
        if d1 === d2
            computeIntegral(OneBodyIntegral{D}(), op, (d1,); markerCache, lazyCompute)
        else
            normCache = initializeOverlapCache(orbDataPair)
            computeIntegral!(normCache, normCache, orbDataPair; markerCache)
        end
    else
        oPairType = getOrbOutputTypeUnion((d1, d2))
        oData1, w1 = decomposeOrbData(d1; markerCache)
        oData2, w2 = decomposeOrbData(d2; markerCache)
        config = OrbitalCoreIntegralConfig(Val(true), OneBodyIntegral{D}(), op, oPairType)
        tensor = genOneBodyPrimCoreIntTensor(config, (oData1, oData2))
        dot(w1, tensor, w2)
    end
end

function computeIntegral(style::MultiBodyIntegral{N, D}, op::DirectOperator, 
                         orbs::NonEmptyTuple{OrbitalBasis{<:RealOrComplex{T}, D}}; 
                         cache!Self::ParamDataCache=initializeParamDataCache(), 
                         lazyCompute::Bool=false) where {N, T<:Real, D}
    orbsData = genOrbitalData(orbs, isParamIndependent(op); cache!Self)
    computeIntegral(style, op, orbsData; lazyCompute)
end