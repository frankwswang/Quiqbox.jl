using LinearAlgebra: dot

const OrbitalBasisSet{T, D} = AbstractVector{<:OrbitalBasis{T, D}}
const OrbitalCollection{T, D} = NonEmpTplOrAbtArr{FrameworkOrb{T, D}, 1}
const OrbitalInput{T, D} = Union{FrameworkOrb{T, D}, OrbitalCollection{T, D}}
const FPrimOrbSet{T, D} = AbstractVector{<:FPrimOrb{T, D}}

const OrbCoreData{T, D, F<:PrimitiveOrbCore{T, D}, V<:Vector{<:ShapedMemory{T}}} = 
      Tuple{F, V}

const FlatParamValMatcher{T, V<:AbstractVector{ <:AbstractArray{T} }} = 
      ElementWiseMatcher{ItsType, V}

const ConfinedParamInput{T} = Union{ShapedMemory{<:ElementalParam{T}, 1}, FlattenedParam{T}}

const OrbCoreMarker = FieldMarker{:PrimitiveOrbCore, 1}

const OrbCoreKey{T, V<:Vector{<:ShapedMemory{T}}} = 
      Tuple{OrbCoreMarker, FlatParamValMatcher{T, V}}

const OrbCoreMarkerDict = Dict{BlackBox, OrbCoreMarker}

const OrbCoreIdxDict{T} = Dict{OrbCoreKey{T}, Int}

const OrbCoreDataSeq{T, D, F<:PrimitiveOrbCore{T, D}, V<:AbtVecOfAbtArr{T}, N} = 
      AbstractVector{OrbCoreData{T, D, F, V}}

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

const GeneralTensorIdxSymDict =  let tempDict=Base.ImmutableDict(2=>:ij)
    Base.ImmutableDict(tempDict, 4=>:ijkl)
end


struct PrimOrbCoreCache{T, D, F<:PrimitiveOrbCore{T, D}}
    dict::OrbCoreIdxDict{T}
    list::Vector{OrbCoreData{ T, D, F, Vector{ShapedMemory{T}} }}
end

PrimOrbCoreCache(::Type{T}, ::Val{D}, ::Type{F}=PrimitiveOrbCore{T, D}) where 
                {T, D, F<:PrimitiveOrbCore{T, D}} = 
PrimOrbCoreCache(OrbCoreIdxDict{T}(), OrbCoreData{ T, D, F, Vector{ShapedMemory{T}} }[])


struct IndexedWeights{T} <: QueryBox{T}
    list::Memory{Pair{Int, T}}
end


abstract type IntegralData{T, S} <: QueryBox{T} end

struct CompleteOneBodyIntegrals{T<:Number} <: IntegralData{T, OneBodyIntegral}
    aa::Dict{ Tuple{   Int},  Tuple{   T}}
    ab::Dict{NTuple{2, Int}, NTuple{2, T}}
end

CompleteOneBodyIntegrals(::Type{T}) where {T} = 
CompleteOneBodyIntegrals(Dict{ Tuple{   Int},  Tuple{   T}}(), 
                         Dict{NTuple{2, Int}, NTuple{2, T}}())

struct OneBodySymmetricDiffData{T<:Number} <: IntegralData{T, OneBodyIntegral}
    ab::Dict{NTuple{2, Int}, T}
end


struct IntegralCache{T, D, F<:DirectOperator, B<:PrimitiveOrbCore{T, D}, 
                     I<:IntegralData{T}} <: QueryBox{T}
    operator::F
    basis::PrimOrbCoreCache{T, D, B}
    data::I
end

const OneBodyIntCache{T, D, F<:DirectOperator, B<:PrimitiveOrbCore{T, D}, 
                      I<:IntegralData{T, OneBodyIntegral}} = 
      IntegralCache{T, D, F, B, I}

const OverlapCache{T, D, B<:PrimitiveOrbCore{T, D}, 
                   I<:IntegralData{T, OneBodyIntegral}} = 
      OneBodyIntCache{T, D, Identity, B, I}


function setIntegralData!(ints::CompleteOneBodyIntegrals{T}, 
                          pair::Pair{Tuple{Int}, Tuple{T}}) where {T}
    setindex!(getfield(ints, OneBodyIdxSymDict[true ]), pair.second, pair.first)
    ints
end

function setIntegralData!(ints::CompleteOneBodyIntegrals{T}, 
                          pair::Pair{NTuple{2, Int}, NTuple{2, T}}) where {T}
    setindex!(getfield(ints, OneBodyIdxSymDict[false]), pair.second, pair.first)
    ints
end


function getPrimCoreIntData(cache::CompleteOneBodyIntegrals, idx::NTuple{2, Int})
    getfield(cache, OneBodyIdxSymDict[false])[idx]
end

function getPrimCoreIntData(cache::CompleteOneBodyIntegrals, idx::Tuple{Int})
    getfield(cache, OneBodyIdxSymDict[true ])[idx]
end


isHermitian(::PrimitiveOrbCore{T, D}, ::DirectOperator, 
            ::PrimitiveOrbCore{T, D}) where {T, D} = 
false

isHermitian(::PrimitiveOrbCore{T, D}, ::Identity, 
            ::PrimitiveOrbCore{T, D}) where {T, D} = 
true


buildOneBodyCoreIntegrator(op::DirectOperator, 
                           orbs::N12Tuple{PrimitiveOrbCore{T, D}}) where {T, D} = 
OneBodyNumIntegrate(op, orbs)

buildOneBodyCoreIntegrator(::Identity, orbs::N12Tuple{PrimGTOcore{T, D}}) where {T, D} = 
genGTOrbOverlapFunc(orbs)


prepareOneBodyCoreIntCache(::Identity, orbs::N12Tuple{PrimGTOcore{T, D}}) where {T, D} = 
getGTOrbOverlapCache(orbs)

prepareOneBodyCoreIntCache(::Identity, ::N12Tuple{PrimitiveOrbCore{T, D}}) where {T, D} = 
NullCache{T}()


function lazyEvalCoreIntegral(integrator::OrbitalIntegrator{T}, 
                              paramData::N12Tuple{FilteredVecOfArr{T}}, 
                              cache::C) where {T, C}
    integrator(paramData...; cache)::T
end

function lazyEvalCoreIntegral(integrator::OrbitalIntegrator{T}, 
                              paramData::N12Tuple{FilteredVecOfArr{T}}, 
                              ::NullCache{T}) where {T}
    integrator(paramData...)::T
end


function evalOneBodyPrimCoreIntegral(op::DirectOperator, 
                                     data::N12Tuple{OrbCoreData{T, D}}) where {T, D}
    orbData = first.(data)
    parData =  last.(data)
    fCore = buildOneBodyCoreIntegrator(op, orbData)
    cache = prepareOneBodyCoreIntCache(op, orbData)
    lazyEvalCoreIntegral(fCore, parData, cache)
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
    if i+j > k+l
        (pR, pL)
    else
        (pL, pR)
    end
end

sortTensorIndex(arg::Vararg{Int}) = sortTensorIndex(arg)


function genOneBodyIntDataPairsCore(op::DirectOperator, 
                                    (oData,)::Tuple{OrbCoreDataSeq{T}}, 
                                    (oneBasedIdx,)::Tuple{Int}) where {T}
    iiVal = evalOneBodyPrimCoreIntegral(op, (oData[begin+oneBasedIdx-1],))
    (iiVal,)
end

function genOneBodyIntDataPairsCore(op::DirectOperator, 
                                    oDataPair::NTuple{2, OrbCoreDataSeq{T}}, 
                                    oneBasedIdxPair::NTuple{2, Int}) where {T}
    orbPars1, orbPars2 = map(oDataPair, oneBasedIdxPair) do data, idx
        data[begin+idx-1]
    end
    ijVal = evalOneBodyPrimCoreIntegral(op, (orbPars1, orbPars2))
    jiVal = if isHermitian(first(orbPars1), op, first(orbPars2))
        ijVal'
    else
        evalOneBodyPrimCoreIntegral(op, (orbPars2, orbPars1))
    end
    (ijVal, jiVal)
end


function genOneBodyIntDataPairs(op::DirectOperator, 
                                (oData,)::Tuple{OrbCoreDataSeq{T, D}}, 
                                (indexOffset,)::Tuple{Int}=(0,)) where {T, D}
    nOrbs = length(oData)
    offset = indexOffset + firstindex(oData) - 1

    pairs1 = map(1:nOrbs) do i
        iiVal = genOneBodyIntDataPairsCore(op, (oData,), (i,))
        (i + offset,) => iiVal
    end

    pairs2 = map(1:triMatEleNum(nOrbs-1)) do l
        n, m = convert1DidxTo2D(nOrbs-1, l)
        ijPair = sortTensorIndex((m, n+1))
        ijValPair = genOneBodyIntDataPairsCore(op, (oData, oData), ijPair)
        (ijPair .+ offset) => ijValPair
    end

    pairs1, pairs2
end

function genOneBodyIntDataPairs(op::DirectOperator, 
                                oDataPair::NTuple{2, OrbCoreDataSeq{T, D}}, 
                                indexOffsets::NTuple{2, Int}) where {T, D}
    mnOrbs = length.(oDataPair)
    offsets = indexOffsets .+ firstindex.(oDataPair) .- 1

    map(Iterators.product( Base.OneTo.(mnOrbs)... )) do mnIdx
        idxPairOld = mnIdx .+ offsets
        idxPairNew = sortTensorIndex(idxPairOld)
        ijPair = ifelse(idxPairNew == idxPairOld, mnIdx, reverse(mnIdx))
        ijValPair = genOneBodyIntDataPairsCore(op, oDataPair, ijPair)
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


function computeOneBodyPrimCoreIntTensor(op::DirectOperator, 
                                         (oData,)::Tuple{OrbCoreDataSeq{T}}) where {T}
    nOrbs = length(oData)
    res = ShapedMemory{T}(undef, (nOrbs, nOrbs))

    for i in 1:nOrbs
        temp = genOneBodyIntDataPairsCore(op, (oData,), (i,))
        setTensorEntries!(res, temp, (i,))
    end

    for l in 1:triMatEleNum(nOrbs-1)
        n, m = convert1DidxTo2D(mBasis-1, l)
        temp = genOneBodyIntDataPairsCore(op, (oData, oData), (m, n+1))
        setTensorEntries!(res, temp, (m, n+1))
    end
    res
end

function computeOneBodyPrimCoreIntTensor(op::DirectOperator, 
                                         oDataPair::NTuple{2, OrbCoreDataSeq{T, D}}) where 
                                        {T, D}
    oData1, oData2 = oDataPair
    len1, len2 = length.(oDataPair)
    res = ShapedMemory{T}(undef, (len1, len2))
    for j in 1:len2, i in 1:len1
        ijVal = evalOneBodyPrimCoreIntegral(op, (oData1[begin+i-1], oData2[begin+j-1]))
        res[begin+i-1, begin+j-1] = ijVal
    end
    res
end


struct BasisIndexList
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


function updateOrbCache!(orbCache::PrimOrbCoreCache{T, D}, 
                         paramCache::DimSpanDataCacheBox{T}, 
                         orbMarkerCache::OrbCoreMarkerDict, 
                         orb::FPrimOrb{T}) where {T, D}
    basis = orbCache.list
    idxDict = orbCache.dict
    orbCore = (getInnerOrb∘getInnerOrb)(orb)
    objCoreMarker = lazyMarkObj!(orbMarkerCache, orbCore)
    paramSubset = FilteredObject(orb.param, orb.pointer.scope) |> FlatParamSubset
    marker = (p::ConfinedParamInput{T}) -> cacheParam!(paramCache, p)

    get(idxDict, ( objCoreMarker, ElementWiseMatcher(paramSubset, marker) )) do
        paramVals = cacheParam!(paramCache, paramSubset)
        push!(basis, (orbCore, paramVals))
        idx = lastindex(basis)
        setindex!(idxDict, idx, ( objCoreMarker, ElementWiseMatcher(paramVals) ))
        idx
    end
end


function updateOrbCache!(orbCache::PrimOrbCoreCache{T, D}, 
                         orbMarkerCache::OrbCoreMarkerDict, 
                         orbData::OrbCoreData{T, D}) where {T, D}
    basis = orbCache.list
    orbCore, orbPars = orbData
    objCoreMarker = lazyMarkObj!(orbMarkerCache, orbCore)
    get!(orbCache.dict, ( objCoreMarker, ElementWiseMatcher(orbPars) )) do
        push!(basis, orbData)
        lastindex(basis)
    end
end


function indexCacheOrbData!(orbCache::PrimOrbCoreCache{T, D}, 
                            paramCache::DimSpanDataCacheBox{T}, 
                            orbMarkerCache::OrbCoreMarkerDict, 
                            orb::FrameworkOrb{T, D}) where {T, D}
    orbSize = orbSizeOf(orb)
    list = BasisIndexList(orbSize)
    for i in 1:orbSize
        idx = updateOrbCache!(orbCache, paramCache, orbMarkerCache, viewOrb(orb, i))
        list.index[begin+i-1] = idx
    end
    list
end

function indexCacheOrbData!(orbCache::PrimOrbCoreCache{T, D}, 
                            paramCache::DimSpanDataCacheBox{T}, 
                            orbMarkerCache::OrbCoreMarkerDict, 
                            orbs::OrbitalCollection{T, D}) where {T, D}
    list = (BasisIndexList∘map)(orbSizeOf, orbs)
    for (j, orb) in enumerate(orbs)
        iRange = getBasisIndexRange(list, j)
        for (n, i) in enumerate(iRange)
            idx = updateOrbCache!(orbCache, paramCache, orbMarkerCache, viewOrb(orb, n))
            list.index[i] = idx
        end
    end
    list
end

function indexCacheOrbData!(targetCache::PrimOrbCoreCache{T, D}, 
                            sourceCache::PrimOrbCoreCache{T, D}, 
                            orbMarkerCache::OrbCoreMarkerDict, 
                            sourceList::BasisIndexList) where {T, D}
    targetList = BasisIndexList(sourceList)
    primOrbIds = targetList.index
    for i in eachindex(primOrbIds)
        orbData = sourceCache.list[sourceList.index[i]]
        primOrbIds[i] = updateOrbCache!(targetCache, orbMarkerCache, orbData)
    end
    targetList
end


function cachePrimCoreIntegrals!(intCache::IntegralCache{T, D}, 
                                 paramCache::DimSpanDataCacheBox{T}, 
                                 orbMarkerCache::OrbCoreMarkerDict, 
                                 orbs::OrbitalCollection{T, D}) where {T, D}
    orbCache = intCache.basis
    oldMaxIdx = lastindex(orbCache.list)
    orbIdxList = indexCacheOrbData!(orbCache, paramCache, orbMarkerCache, orbs)
    updatePrimCoreIntCache!(intCache, oldMaxIdx+1)
    orbIdxList
end

function cachePrimCoreIntegrals!(intCache::IntegralCache{T, D}, 
                                 paramCache::DimSpanDataCacheBox{T}, 
                                 orbMarkerCache::OrbCoreMarkerDict, 
                                 orb::FrameworkOrb{T, D}) where {T, D}
    orbCache = intCache.basis
    oldMaxIdx = lastindex(orbCache.list)
    orbIdxList = indexCacheOrbData!(orbCache, paramCache, orbMarkerCache, orb)
    updatePrimCoreIntCache!(intCache, oldMaxIdx+1)
    orbIdxList
end

function cachePrimCoreIntegrals!(targetIntCache::IntegralCache{T, D}, 
                                 sourceIntCache::IntegralCache{T, D}, 
                                 orbMarkerCache::OrbCoreMarkerDict, 
                                 sourceOrbList::BasisIndexList) where {T, D}
    tOrbCache = targetIntCache.basis
    oldMaxIdx = lastindex(tOrbCache.list)
    sOrbCache = sourceIntCache.basis
    orbIdxList = indexCacheOrbData!(tOrbCache, sOrbCache, orbMarkerCache, sourceOrbList)
    updatePrimCoreIntCache!(targetIntCache, oldMaxIdx+1)
    orbIdxList
end


function updateIntCacheCore!(op::DirectOperator, ints::CompleteOneBodyIntegrals{T}, 
                             basis::Tuple{OrbCoreDataSeq{T, D}}, 
                             offset::Tuple{Int}) where {T, D}
    pairs1, pairs2 = genOneBodyIntDataPairs(op, basis, offset)
    foreach(p->setIntegralData!(ints, p), pairs1)
    foreach(p->setIntegralData!(ints, p), pairs2)
    ints
end

function updateIntCacheCore!(op::DirectOperator, ints::CompleteOneBodyIntegrals{T}, 
                             basis::NTuple{2, OrbCoreDataSeq{T, D}}, 
                             offset::NTuple{2, Int}) where {T, D}
    pairs2 = genOneBodyIntDataPairs(op, basis, offset)
    foreach(p->setIntegralData!(ints, p), pairs2)
    ints
end


function updatePrimCoreIntCache!(cache::IntegralCache, startIdx::Int)
    op = cache.operator
    basis = cache.basis.list
    ints = cache.data
    firstIdx = firstindex(basis)

    if startIdx == firstIdx
        updateIntCacheCore!(op, ints, (basis,), (0,))
    elseif firstIdx < startIdx <= lastindex(basis)
        boundary = startIdx - 1
        oldBasis = @view basis[begin:boundary]
        newBasis = @view basis[startIdx:  end]
        updateIntCacheCore!(op, ints, (newBasis,), (boundary,))
        updateIntCacheCore!(op, ints, (oldBasis, newBasis,), (0, boundary))
    end

    cache
end


function decodePrimCoreInt(cache::CompleteOneBodyIntegrals{T}, ptrPair::NTuple{2, Int}, 
                           (coeff1, coeff2)::NTuple{2, T}=( one(T), one(T) )) where {T}
    coeffProd = coeff1' * coeff2
    ptrPairNew = sortTensorIndex(ptrPair)
    data = getPrimCoreIntData(cache, ptrPairNew)
    (ptrPairNew == ptrPair ? data : reverse(data)) .* (coeffProd, coeffProd')
end

function decodePrimCoreInt(cache::CompleteOneBodyIntegrals{T}, ptr::Tuple{Int}, 
                           (coeff1,)::Tuple{T}=( one(T), )) where {T}
    getPrimCoreIntData(cache, ptr) .* coeff1' .* coeff1
end


function buildPrimOrbWeight(normCache::OverlapCache{T, D}, orb::EvalPrimOrb{T, D}, 
                            idx::Int) where {T, D}
    if isRenormalized(orb)
        decodePrimCoreInt(normCache.data, (idx,)) |> first |> AbsSqrtInv
    else
        one(T)
    end
end


function buildNormalizedCompOrbWeight!(weight::AbstractVector{T}, 
                                       normCache::OverlapCache{T, D}, orb::FCompOrb{T, D}, 
                                       idxSeq::AbstractVector{Int}) where {T, D}
    overlapCache = normCache.data
    nPrimOrbs = length(weight)
    innerOverlapSum = zero(T)

    for i in 1:nPrimOrbs
        ptr = idxSeq[begin+i-1]
        innerCoreDiagOverlap = decodePrimCoreInt(overlapCache, (ptr,)) |> first
        wc = weight[begin+i-1]
        innerDiagOverlap = wc' * wc
        if (isRenormalized∘viewOrb)(orb, i)
            weight[begin+i-1] *= AbsSqrtInv(innerCoreDiagOverlap)
        else
            innerDiagOverlap *= innerCoreDiagOverlap
        end
        innerOverlapSum += innerDiagOverlap
    end

    innerOverlapSum += mapreduce(+, 1:triMatEleNum(nPrimOrbs-1)) do l
        n, m = convert1DidxTo2D(nPrimOrbs-1, l)
        pointerPair = (idxSeq[begin+m-1], idxSeq[begin+n])
         scalarPair = (weight[begin+m-1], weight[begin+n])
        (sum∘decodePrimCoreInt)(overlapCache, pointerPair, scalarPair)
    end
    weight .*= AbsSqrtInv(innerOverlapSum)
end


function buildOrbWeight!(paramCache::DimSpanDataCacheBox{T}, 
                         normCache::OverlapCache{T, D}, orb::FrameworkOrb{T, D}, 
                         idxSeq::AbstractVector{Int}) where {T, D}
    nPrimOrbs = orbSizeOf(orb)
    weight = Memory{T}(undef, nPrimOrbs)

    if orb isa FCompOrb
        weight .= cacheParam!(paramCache, getOrbWeightCore(orb))
        if isRenormalized(orb)
            buildNormalizedCompOrbWeight!(weight, normCache, orb, idxSeq)
        else
            for i in 1:nPrimOrbs
                ptr = idxSeq[begin+i-1]
                temp = buildPrimOrbWeight(normCache, (getInnerOrb∘viewOrb)(orb, i), ptr)
                weight[begin+i-1] *= temp
            end
        end
    else
        weight[] = buildPrimOrbWeight(normCache, getInnerOrb(orb), idxSeq[])
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


function buildIndexedOrbWeights!(paramCache::DimSpanDataCacheBox{T}, 
                                normCache::OverlapCache{T, D}, 
                                orbs::OrbitalCollection{T, D}, 
                                normIdxList::BasisIndexList, 
                                intIdxList::BasisIndexList=normIdxList) where {T, D}
    i = 0
    map(orbs) do orb
        iRange = getBasisIndexRange(intIdxList, (i+=1))
        intIdxSeq = view(intIdxList.index, iRange)
        normIdxSeq = view(normIdxList.index, iRange)
        orbWeight = buildOrbWeight!(paramCache, normCache, orb, normIdxSeq)
        buildIndexedOrbWeightsCore!(intIdxSeq, orbWeight)
    end
end

function buildIndexedOrbWeights!(paramCache::DimSpanDataCacheBox{T}, 
                                normCache::OverlapCache{T, D}, 
                                orb::FrameworkOrb{T, D}, 
                                normIdxList::BasisIndexList, 
                                intIdxList::BasisIndexList=normIdxList) where {T, D}
    orbWeight = buildOrbWeight!(paramCache, normCache, orb, normIdxList.index)
    buildIndexedOrbWeightsCore!(intIdxList.index, orbWeight)
end


function prepareIntegralConfig!(intCache::IntegralCache{T, D}, 
                                normCache::OverlapCache{T, D}, 
                                paramCache::DimSpanDataCacheBox{T}, 
                                orbMarkerCache::OrbCoreMarkerDict, 
                                orbInput::OrbitalInput{T, D}) where {T, D}
    iIdxList = cachePrimCoreIntegrals!(intCache, paramCache, orbMarkerCache, orbInput)
    if intCache === normCache
        buildIndexedOrbWeights!(paramCache, intCache, orbInput, iIdxList)
    else
        nIdxList = cachePrimCoreIntegrals!(normCache, intCache, orbMarkerCache, iIdxList)
        buildIndexedOrbWeights!(paramCache, normCache, orbInput, nIdxList, iIdxList)
    end
end


function buildIntegralEntries(intCache::OneBodyIntCache{T}, 
                              (intWeights,)::Tuple{IndexedWeights{T}}) where {T}
    idxList = intWeights.list
    len = length(idxList)
    intValCache = intCache.data
    temp = mapreduce(+, eachindex(idxList)) do i
        ptr, coeff = idxList[i]
        (first∘decodePrimCoreInt)(intValCache, (ptr,), (coeff,))
    end
    res = mapreduce(+, 1:triMatEleNum(len-1), init=temp) do l
        n, m = convert1DidxTo2D(len-1, l)
        ptr1, weight1 = idxList[begin+m-1]
        ptr2, weight2 = idxList[begin+n]
        (sum∘decodePrimCoreInt)(intValCache, (ptr1, ptr2), (weight1, weight2))
    end
    (res,) # ([1|O|1],)
end

function buildIntegralEntries(intCache::OneBodyIntCache{T}, 
                              intWeightPair::NTuple{2, IndexedWeights{T}}) where {T}
    list1, list2 = getfield.(intWeightPair, :list)
    intValCache = intCache.data
    idxPairRange = Iterators.product(eachindex(list1), eachindex(list2))
    mapreduce(.+, idxPairRange, init=( zero(T), zero(T) )) do (i, j)
        ptr1, weight1 = list1[i]
        ptr2, weight2 = list2[j]
        if ptr1 == ptr2
            temp = (first∘decodePrimCoreInt)(intValCache, (ptr1,), (weight1,))
            (temp, temp')
        else
            decodePrimCoreInt(intValCache, (ptr1, ptr2), (weight1, weight2))
        end
    end # ([1|O|2], [2|O|1])
end


function buildIntegralTensor(intCache::OneBodyIntCache{T}, 
                             intWeights::AbstractVector{IndexedWeights{T}}) where {T}
    nOrbs = length(intWeights)
    res = ShapedMemory{T}(undef, (nOrbs, nOrbs))
    for i in 1:nOrbs
        iBI = intWeights[begin+i-1]
        temp = buildIntegralEntries(intCache, (iBI,))
        setTensorEntries!(res, temp, (i,))
    end

    for l in 1:triMatEleNum(nOrbs-1)
        n, m = convert1DidxTo2D(nOrbs-1, l)
        mBI = intWeights[begin+m-1]
        nBI = intWeights[begin+n]
        temp = buildIntegralEntries(intCache, (mBI, nBI))
        setTensorEntries!(res, temp, (m, n+1))
    end
    res
end


function getPrimOrbCoreTypeUnion(orb::FPrimOrb)
    (typeof∘getInnerOrb∘getInnerOrb)(orb)
end

function getPrimOrbCoreTypeUnion(orb::FCompOrb)
    mapreduce(typejoin, 1:orbSizeOf(orb), init=Union{}) do i
        viewOrb(orb, i) |> getPrimOrbCoreTypeUnion
    end
end

function getPrimOrbCoreTypeUnion(orbs::OrbitalCollection)
    orbs isa AbstractVector && checkEmptiness(orbs, :orbs)
    mapreduce(typejoin, orbs, init=Union{}) do orb
        getPrimOrbCoreTypeUnion(orb)
    end
end


function initializeIntegralCache!(::OneBodyIntegral, op::DirectOperator, 
                                  paramCache::DimSpanDataCacheBox{T}, 
                                  orbInput::OrbitalInput{T, D}) where {T, D}
    coreType = getPrimOrbCoreTypeUnion(orbInput)
    orbCache = PrimOrbCoreCache(T, Val(D), coreType)
    IntegralCache(op, orbCache, CompleteOneBodyIntegrals(T))
end

function initializeOverlapCache!(paramCache::DimSpanDataCacheBox{T}, 
                                 orbInput::OrbitalInput{T}) where {T}
    initializeIntegralCache!(OneBodyIntegral(), Identity(), paramCache, orbInput)
end


function initializeOneBodyCachePair!(op::F, paramCache::DimSpanDataCacheBox{T}, 
                                     orbInput::OrbitalInput{T}) where {F<:DirectOperator, T}
    intCache = initializeIntegralCache!(OneBodyIntegral(), op, paramCache, orbInput)
    normCache = F <: Identity ? intCache : initializeOverlapCache!(paramCache, orbInput)
    intCache, normCache
end


function computeIntTensor!(intCache::IntegralCache{T, D}, 
                           normCache::OverlapCache{T, D}, 
                           basisSet::AbstractVector{<:FrameworkOrb{T, D}}; 
                           paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                           basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict()) where 
                          {T, D}
    ws = prepareIntegralConfig!(intCache, normCache, paramCache, basisMarkerCache, basisSet)
    buildIntegralTensor(intCache, ws)
end

function computeIntTensor(::OneBodyIntegral, op::F, 
                          basisSet::AbstractVector{<:FrameworkOrb{T, D}}; 
                          paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                          basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict()) where 
                         {F<:DirectOperator, T, D}
    intCache, normCache = initializeOneBodyCachePair!(op, paramCache, basisSet)
    computeIntTensor!(intCache, normCache, basisSet; paramCache, basisMarkerCache)
end

computeIntTensor(style::MultiBodyIntegral, op::DirectOperator, orbs::OrbitalBasisSet{T}; 
                 paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                 basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict()) where {T} = 
computeIntTensor(style, op, map(FrameworkOrb, orbs); paramCache, basisMarkerCache)


function decomposeOrb!(normCache::OverlapCache{T, D}, orb::FrameworkOrb{T, D}; 
                       paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                       markerCache::OrbCoreMarkerDict=OrbCoreMarkerDict()) where {T, D}
    normIdxList = cachePrimCoreIntegrals!(normCache, paramCache, markerCache, orb)
    orbWeight = buildOrbWeight!(paramCache, normCache, orb, normIdxList.index)
    normCache.basis.list[normIdxList.index], orbWeight
end

function decomposeOrb(orb::FrameworkOrb{T}; 
                      paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                      markerCache::OrbCoreMarkerDict=OrbCoreMarkerDict()) where {T}
    normCache = initializeOverlapCache!(paramCache, orb)
    decomposeOrb!(normCache, orb; paramCache, markerCache)
end


function computeIntegral!(intCache::IntegralCache{T, D}, 
                          normCache::OverlapCache{T, D}, 
                          bfs::NonEmptyTuple{FrameworkOrb{T, D}}; 
                          paramCache::DimSpanDataCacheBox{T}, 
                          basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict()) where 
                         {T, D}
    ws = prepareIntegralConfig!(intCache, normCache, paramCache, basisMarkerCache, bfs)
    buildIntegralEntries(intCache, ws) |> first
end

function computeIntegral(::OneBodyIntegral, op::DirectOperator, 
                         (bf1,)::Tuple{FrameworkOrb{T, D}}; 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                         basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict(), 
                         lazyCompute::Bool=false) where {T, D}
    if lazyCompute
        iCache, nCache = initializeOneBodyCachePair!(op, paramCache, bf1)
        computeIntegral!(iCache, nCache, (bf1,); paramCache, basisMarkerCache)
    else
        coreData, w = decomposeOrb(bf1; paramCache, markerCache=basisMarkerCache)
        tensor = computeOneBodyPrimCoreIntTensor(op, (coreData,))
        dot(w, tensor, w)
    end
end

function computeIntegral(::OneBodyIntegral, op::DirectOperator, 
                         (bf1, bf2)::NTuple{2, FrameworkOrb{T, D}}; 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                         basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict(), 
                         lazyCompute::Bool=false) where {T, D}
    coreData1, w1 = decomposeOrb(bf1; paramCache, markerCache=basisMarkerCache)
    coreData2, w2 = decomposeOrb(bf2; paramCache, markerCache=basisMarkerCache)

    tensor = if lazyCompute
        coreData1 = Vector(coreData1)
        coreData2 = Vector(coreData2)
        transformation = (b::PrimitiveOrbCore{T, D})->lazyMarkObj!(basisMarkerCache, b)
        coreDataM = intersectMultisets!(coreData1, coreData2; transformation)
        block4 = computeOneBodyPrimCoreIntTensor(op, (coreData1, coreData2))
        if isempty(coreDataM)
            block4
        else
            block1 = computeOneBodyPrimCoreIntTensor(op, (coreDataM,))
            block2 = computeOneBodyPrimCoreIntTensor(op, (coreData1, coreDataM))
            block3 = computeOneBodyPrimCoreIntTensor(op, (coreDataM, coreData2))
            hvcat((2, 2), block1, block3, block2, block4)
        end
    else
        computeOneBodyPrimCoreIntTensor(op, (coreData1, coreData2))
    end
    dot(w1, tensor, w2)
end

function computeIntegral(::OneBodyIntegral, ::Identity, 
                         bfPair::NTuple{2, FrameworkOrb{T, D}}; 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                         basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict(), 
                         lazyCompute::Bool=false) where {T, D}
    bf1, bf2 = bfPair
    if lazyCompute
        if bf1 === bf2
            computeIntegral(OneBodyIntegral(), Identity(), (bf1,); paramCache, 
                            basisMarkerCache, lazyCompute)
        else
            normCache = initializeOverlapCache!(paramCache, bfPair)
            computeIntegral!(normCache, normCache, bfPair; paramCache, basisMarkerCache)
        end
    else
        coreData1, w1 = decomposeOrb(bf1; paramCache, markerCache=basisMarkerCache)
        coreData2, w2 = decomposeOrb(bf2; paramCache, markerCache=basisMarkerCache)
        tensor = computeOneBodyPrimCoreIntTensor(Identity(), (coreData1, coreData2))
        dot(w1, tensor, w2)
    end
end

function computeIntegral(style::MultiBodyIntegral, op::DirectOperator, 
                         orbs::NonEmptyTuple{OrbitalBasis{T, D}}; 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                         basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict(), 
                         lazyCompute::Bool=false) where {T, D}
    fOrbs = lazyTupleMap(FrameworkOrb, orbs)
    computeIntegral(style, op, fOrbs; paramCache, basisMarkerCache, lazyCompute)
end