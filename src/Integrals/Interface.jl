using LinearAlgebra: dot

const N12Tuple{T} = Union{Tuple{T}, NTuple{2, T}}
const N24Tuple{T} = Union{NTuple{2, T}, NTuple{4, T}}

const OrbitalBasisSet{T, D} = AbstractVector{<:OrbitalBasis{T, D}}
const OrbitalCollection{T, D} = NonEmpTplOrAbtArr{FrameworkOrb{T, D}, 1}
const OrbitalInput{T, D} = Union{FrameworkOrb{T, D}, OrbitalCollection{T, D}}
const FPrimOrbSet{T, D} = AbstractVector{<:FPrimOrb{T, D}}

const OrbCoreIdxDict{T} = 
      Dict{Tuple{FieldMarker{:PrimitiveOrbCore, 1}, 
                 ElementWiseMatcher{ ItsType, <:Vector{<:ShapedMemory{T}} }}, Int}

const OrbCoreData{T, D, F<:PrimitiveOrbCore{T, D}, V<:AbtVecOfAbtArr{T}} = Tuple{F, V}
const OrbCoreDataSeq{T, D, F<:PrimitiveOrbCore{T, D}, V<:AbtVecOfAbtArr{T}, N} = 
      AbstractVector{OrbCoreData{T, D, F, V}}

const OneBodyIdxSymDict = let tempDict=Base.ImmutableDict(true=>:aa)
    Base.ImmutableDict(tempDict, false=>:ab)
end

const TwoBodyIdxSymDict = let
    valTemp = (:aaaa, :aabb, :abab, :aaxy, :abxx, :abxy)
    keyTemp = ((true,  true,  true), (true,  true, false), (false, false,  true), 
               (true, false, false), (false, true, false), (false, false, false))
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


struct BasisWeightIndexer{T} <: QueryBox{T}
    list::Memory{Pair{Int, T}}
end


abstract type IntegralIndexer{T, S} <: QueryBox{T} end

struct OneBodyCompleteGraphIndexer{T<:Number} <: IntegralIndexer{T, OneBodyIntegral}
    aa::Dict{ Tuple{   Int},  Tuple{   T}}
    ab::Dict{NTuple{2, Int}, NTuple{2, T}}
end

OneBodyCompleteGraphIndexer(::Type{T}) where {T} = 
OneBodyCompleteGraphIndexer(Dict{ Tuple{   Int},  Tuple{   T}}(), 
                            Dict{NTuple{2, Int}, NTuple{2, T}}())

struct OneBodySymmetricDiffIndexer{T<:Number} <: IntegralIndexer{T, OneBodyIntegral}
    ab::Dict{NTuple{2, Int}, T}
end


struct IntegralCache{T, D, F<:DirectOperator, V<:PrimitiveOrbCore{T, D}, 
                     I<:IntegralIndexer{T}} <: QueryBox{T}
    operator::F
    basis::PrimOrbCoreCache{T, D, V}
    data::I
end

const OneBodyIntCache{T, D, F<:DirectOperator, V<:PrimitiveOrbCore{T, D}, 
                      I<:IntegralIndexer{T, OneBodyIntegral}} = 
      IntegralCache{T, D, F, V, I}

const OverlapCache{T, D, V<:PrimitiveOrbCore{T, D}, 
                   I<:IntegralIndexer{T, OneBodyIntegral}} = 
      OneBodyIntCache{T, D, Identity, V, I}


function setIntegralIndexer!(idxer::OneBodyCompleteGraphIndexer{T}, 
                             pair::Pair{Tuple{Int}, Tuple{T}}) where {T}
    setindex!(getfield(idxer, OneBodyIdxSymDict[true ]), pair.second, pair.first)
    idxer
end

function setIntegralIndexer!(idxer::OneBodyCompleteGraphIndexer{T}, 
                             pair::Pair{NTuple{2, Int}, NTuple{2, T}}) where {T}
    setindex!(getfield(idxer, OneBodyIdxSymDict[false]), pair.second, pair.first)
    idxer
end


function getPrimCoreIntData(cache::OneBodyCompleteGraphIndexer, idx::NTuple{2, Int})
    getfield(cache, OneBodyIdxSymDict[false])[idx]
end

function getPrimCoreIntData(cache::OneBodyCompleteGraphIndexer, idx::Tuple{Int})
    getfield(cache, OneBodyIdxSymDict[true ])[idx]
end


genOneBodyCoreIntegrator(::Identity, orbs::N12Tuple{PrimGTOcore{T, D}}) where {T, D} = 
genGTOrbOverlapFunc(orbs)


isHermitian(::PrimitiveOrbCore{T, D}, ::DirectOperator, 
            ::PrimitiveOrbCore{T, D}) where {T, D} = 
false

isHermitian(::PrimitiveOrbCore{T, D}, ::Identity, 
            ::PrimitiveOrbCore{T, D}) where {T, D} = 
true


## evalOneBodyPrimCoreIntConfig
function evalOneBodyPrimCoreIntegral(op::DirectOperator, 
                                     (orb1, pars1)::OrbCoreData{T, D}, 
                                     (orb2, pars2)::OrbCoreData{T, D}) where {T, D}
    f = ReturnTyped(genOneBodyCoreIntegrator(op, (orb1, orb2)), T)
    f(pars1, pars2)
end

function evalOneBodyPrimCoreIntegral(op::DirectOperator, 
                                     (orb, pars)::OrbCoreData{T, D}) where {T, D}
    f = ReturnTyped(genOneBodyCoreIntegrator(op, (orb,)), T)
    f(pars)
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


function genOneBodyCGIndexerPairsCore(op::DirectOperator, 
                                      (oData,)::Tuple{OrbCoreDataSeq{T}}, 
                                      (oneBasedIdx,)::Tuple{Int}) where {T}
    iiVal = evalOneBodyPrimCoreIntegral(op, oData[begin+oneBasedIdx-1])
    (iiVal,)
end

function genOneBodyCGIndexerPairsCore(op::DirectOperator, 
                                      oDataPair::NTuple{2, OrbCoreDataSeq{T}}, 
                                      oneBasedIdxPair::NTuple{2, Int}) where {T}
    orbPars1, orbPars2 = map(oDataPair, oneBasedIdxPair) do data, idx
        data[begin+idx-1]
    end
    ijVal = evalOneBodyPrimCoreIntegral(op, orbPars1, orbPars2)
    jiVal = if isHermitian(first(orbPars1), op, first(orbPars2))
        ijVal'
    else
        evalOneBodyPrimCoreIntegral(op, orbPars2, orbPars1)
    end
    (ijVal, jiVal)
end


function genOneBodyCGIndexerPairs(op::DirectOperator, 
                                  (oData,)::Tuple{OrbCoreDataSeq{T, D}}, 
                                  (indexOffset,)::Tuple{Int}=(0,)) where {T, D}
    nOrbs = length(oData)
    offset = indexOffset + firstindex(oData) - 1

    pairs1 = map(1:nOrbs) do i
        iiVal = genOneBodyCGIndexerPairsCore(op, (oData,), (i,))
        (i + offset,) => iiVal
    end

    pairs2 = map(1:triMatEleNum(nOrbs-1)) do l
        n, m = convert1DidxTo2D(nOrbs-1, l)
        ijPair = sortTensorIndex((m, n+1))
        ijValPair = genOneBodyCGIndexerPairsCore(op, (oData, oData), ijPair)
        (ijPair .+ offset) => ijValPair
    end

    pairs1, pairs2
end

function genOneBodyCGIndexerPairs(op::DirectOperator, 
                                  oDataPair::NTuple{2, OrbCoreDataSeq{T, D}}, 
                                  indexOffsets::NTuple{2, Int}) where {T, D}
    mnOrbs = length.(oDataPair)
    offsets = indexOffsets .+ firstindex.(oDataPair) .- 1

    map(Iterators.product( Base.OneTo.(mnOrbs)... )) do mnIdx
        idxPairOld = mnIdx .+ offsets
        idxPairNew = sortTensorIndex(idxPairOld)
        ijPair = ifelse(idxPairNew == idxPairOld, mnIdx, reverse(mnIdx))
        ijValPair = genOneBodyCGIndexerPairsCore(op, oDataPair, ijPair)
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
        temp = genOneBodyCGIndexerPairsCore(op, (oData,), (i,))
        setTensorEntries!(res, temp, (i,))
    end

    for l in 1:triMatEleNum(nOrbs-1)
        n, m = convert1DidxTo2D(mBasis-1, l)
        temp = genOneBodyCGIndexerPairsCore(op, (oData, oData), (m, n+1))
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
        ijVal = evalOneBodyPrimCoreIntegral(op, oData1[begin+i-1], oData2[begin+j-1])
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


function genOrbCoreData!(paramCache::DimSpanDataCacheBox{T}, orb::FPrimOrb{T}) where {T}
    pVal = cacheParam!(paramCache, orb.param, orb.pointer.scope)
    ((getInnerOrb∘getInnerOrb)(orb), pVal)
end

function updateOrbCache!(basisCache::PrimOrbCoreCache{T, D}, 
                         paramCache::DimSpanDataCacheBox{T}, orb::FPrimOrb{T}) where {T, D}
    basis = basisCache.list
    idxDict = basisCache.dict
    orbCore = first(orbData)
    objCoreMarker = markObj(orbCore)
    paramSubset = FilteredObject(orb.param, orb.pointer.scope) |> FlatParamSubset
    marker = (p::Union{ElementalParam{T}, FlattenedParam{T}}) -> cacheParam!(paramCache, p)

    get(idxDict, ( objCoreMarker, ElementWiseMatcher(paramSubset, marker) )) do
        paramVals = cacheParam!(paramCache, paramSubset)
        push!(basis, (orbCore, paramVals))
        idx = lastindex(basis)
        setindex!(idxDict, idx, ( objCoreMarker, ElementWiseMatcher(paramVals) ))
        idx
    end
end


function updateOrbCache!(basisCache::PrimOrbCoreCache{T, D}, 
                         orbData::OrbCoreData{T, D}) where {T, D}
    basis = basisCache.list
    orbCore, orbPars = orbData
    get!(basisCache.dict, ( markObj(orbCore), ElementWiseMatcher(orbPars) )) do
        push!(basis, orbData)
        lastindex(basis)
    end
end


function indexCacheOrbData!(orbCache::PrimOrbCoreCache{T, D}, 
                            paramCache::DimSpanDataCacheBox{T}, 
                            orb::FrameworkOrb{T, D}) where {T, D}
    orbSize = orbSizeOf(orb)
    list = BasisIndexList(orbSize)
    for i in 1:orbSize
        orbData = genOrbCoreData!(paramCache, viewOrb(orb, i))
        list.index[begin+i-1] = updateOrbCache!(orbCache, orbData)
    end
    list
end

function indexCacheOrbData!(orbCache::PrimOrbCoreCache{T, D}, 
                            paramCache::DimSpanDataCacheBox{T}, 
                            orbs::OrbitalCollection{T, D}) where {T, D}
    list = (BasisIndexList∘map)(orbSizeOf, orbs)
    for (j, orb) in enumerate(orbs)
        iRange = getBasisIndexRange(list, j)
        for (n, i) in enumerate(iRange)
            orbData = genOrbCoreData!(paramCache, viewOrb(orb, n))
            list.index[i] = updateOrbCache!(orbCache, orbData)
        end
    end
    list
end

function indexCacheOrbData!(targetCache::PrimOrbCoreCache{T, D}, 
                            sourceCache::PrimOrbCoreCache{T, D}, 
                            sourceList::BasisIndexList) where {T, D}
    targetList = BasisIndexList(sourceList)
    primOrbIds = targetList.index
    for i in eachindex(primOrbIds)
        orbData = sourceCache.list[sourceList.index[i]]
        primOrbIds[i] = updateOrbCache!(targetCache, orbData)
    end
    targetList
end


function cachePrimCoreIntegrals!(intCache::IntegralCache{T, D}, 
                                 paramCache::DimSpanDataCacheBox{T}, 
                                 orbs::OrbitalCollection{T, D}) where {T, D}
    orbCache = intCache.basis
    oldMaxIdx = lastindex(orbCache.list)
    orbIdxList = indexCacheOrbData!(orbCache, paramCache, orbs)
    updatePrimCoreIntCache!(intCache, oldMaxIdx+1)
    orbIdxList
end

function cachePrimCoreIntegrals!(intCache::IntegralCache{T, D}, 
                                 paramCache::DimSpanDataCacheBox{T}, 
                                 orb::FrameworkOrb{T, D}) where {T, D}
    orbCache = intCache.basis
    oldMaxIdx = lastindex(orbCache.list)
    orbIdxList = indexCacheOrbData!(orbCache, paramCache, orb)
    updatePrimCoreIntCache!(intCache, oldMaxIdx+1)
    orbIdxList
end

function cachePrimCoreIntegrals!(targetIntCache::IntegralCache{T, D}, 
                                 sourceIntCache::IntegralCache{T, D}, 
                                 sourceOrbList::BasisIndexList) where {T, D}
    targetOrbCache = targetIntCache.basis
    oldMaxIdx = lastindex(targetOrbCache.list)
    sourceOrbCache = sourceIntCache.basis
    targetOrbIdxList = indexCacheOrbData!(targetOrbCache, sourceOrbCache, sourceOrbList)
    updatePrimCoreIntCache!(targetIntCache, oldMaxIdx+1)
    targetOrbIdxList
end


function updateIntCacheCore!(op::DirectOperator, idxer::OneBodyCompleteGraphIndexer{T}, 
                             basis::Tuple{OrbCoreDataSeq{T, D}}, 
                             offset::Tuple{Int}) where {T, D}
    pairs1, pairs2 = genOneBodyCGIndexerPairs(op, basis, offset)
    foreach(p->setIntegralIndexer!(idxer, p), pairs1)
    foreach(p->setIntegralIndexer!(idxer, p), pairs2)
    idxer
end

function updateIntCacheCore!(op::DirectOperator, idxer::OneBodyCompleteGraphIndexer{T}, 
                             basis::NTuple{2, OrbCoreDataSeq{T, D}}, 
                             offset::NTuple{2, Int}) where {T, D}
    pairs2 = genOneBodyCGIndexerPairs(op, basis, offset)
    foreach(p->setIntegralIndexer!(idxer, p), pairs2)
    idxer
end


function updatePrimCoreIntCache!(cache::IntegralCache, startIdx::Int)
    op = cache.operator
    basis = cache.basis.list
    intIdxer = cache.data
    firstIdx = firstindex(basis)

    if startIdx == firstIdx
        updateIntCacheCore!(op, intIdxer, (basis,), (0,))
    elseif firstIdx < startIdx <= lastindex(basis)
        boundary = startIdx - 1
        oldBasis = @view basis[begin:boundary]
        newBasis = @view basis[startIdx:  end]
        updateIntCacheCore!(op, intIdxer, (newBasis,), (boundary,))
        updateIntCacheCore!(op, intIdxer, (oldBasis, newBasis,), (0, boundary))
    end

    cache
end


function decodePrimCoreInt(cache::OneBodyCompleteGraphIndexer{T}, ptrPair::NTuple{2, Int}, 
                           (coeff1, coeff2)::NTuple{2, T}=( one(T), one(T) )) where {T}
    coeffProd = coeff1' * coeff2
    ptrPairNew = sortTensorIndex(ptrPair)
    data = getPrimCoreIntData(cache, ptrPairNew)
    (ptrPairNew == ptrPair ? data : reverse(data)) .* (coeffProd, coeffProd')
end

function decodePrimCoreInt(cache::OneBodyCompleteGraphIndexer{T}, ptr::Tuple{Int}, 
                           (coeff1,)::Tuple{T}=( one(T), )) where {T}
    getPrimCoreIntData(cache, ptr) .* coeff1' .* coeff1
end


function buildPrimOrbWeight(normCache::OverlapCache{T, D}, orb::EvalPrimOrb{T, D}, 
                            idx::Int, scalar::T=one(T)) where {T, D}
    if isRenormalized(orb)
        decodePrimCoreInt(normCache.data, (idx,)) |> first |> AbsSqrtInv
    else
        one(T)
    end * scalar
end


function buildNormalizedCompOrbWeight(weight::AbstractVector{T}, 
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
    if orb isa FCompOrb
        weight = cacheParam!(paramCache, getOrbWeightCore(orb))
        if isRenormalized(orb)
            buildNormalizedCompOrbWeight(weight, normCache, orb, idxSeq)
        else
            for (i, wc) in enumerate(weight)
                ptr = idxSeq[begin+i-1]
                temp = buildPrimOrbWeight(normCache, (getInnerOrb∘viewOrb)(orb, i), ptr, wc)
                weight[begin+i-1] = temp
            end
        end
    else
        weight = getMemory(buildPrimOrbWeight(normCache, getInnerOrb(orb), idxSeq[]))
    end
    weight
end


function buildOrbWeightIndexerCore!(primOrbPtrs::AbstractVector{Int}, 
                                    primOrbWeight::AbstractVector{T}) where {T}
    nPtrs = length(primOrbWeight)
    list = Memory{Pair{Int, T}}(undef, nPtrs)
    for i in 1:nPtrs
        list[begin+i-1] = primOrbPtrs[begin+i-1] => primOrbWeight[begin+i-1]
    end
    BasisWeightIndexer(list)
end


function buildOrbWeightIndexer!(paramCache::DimSpanDataCacheBox{T}, 
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
        buildOrbWeightIndexerCore!(intIdxSeq, orbWeight)
    end
end

function buildOrbWeightIndexer!(paramCache::DimSpanDataCacheBox{T}, 
                                normCache::OverlapCache{T, D}, 
                                orb::FrameworkOrb{T, D}, 
                                normIdxList::BasisIndexList, 
                                intIdxList::BasisIndexList=normIdxList) where {T, D}
    orbWeight = buildOrbWeight!(paramCache, normCache, orb, normIdxList.index)
    buildOrbWeightIndexerCore!(intIdxList.index, orbWeight)
end


function prepareIntegralConfig!(intCache::IntegralCache{T, D}, 
                                normCache::OverlapCache{T, D}, 
                                paramCache::DimSpanDataCacheBox{T}, 
                                orbInput::OrbitalInput{T, D}) where {T, D}
    intIdxList = cachePrimCoreIntegrals!(intCache, paramCache, orbInput)
    if intCache === normCache
        buildOrbWeightIndexer!(paramCache, intCache, orbInput, intIdxList)
    else
        normIdxList = cachePrimCoreIntegrals!(normCache, intCache, intIdxList)
        buildOrbWeightIndexer!(paramCache, normCache, orbInput, normIdxList, intIdxList)
    end
end


function buildIntegralEntries(intCache::OneBodyIntCache{T}, 
                              (intIdxer1,)::Tuple{BasisWeightIndexer{T}}) where {T}
    IdxerList = intIdxer1.list
    len = length(IdxerList)
    intValCache = intCache.data
    temp = mapreduce(+, eachindex(IdxerList)) do i
        ptr, coeff = IdxerList[i]
        (first∘decodePrimCoreInt)(intValCache, (ptr,), (coeff,))
    end
    res = mapreduce(+, 1:triMatEleNum(len-1), init=temp) do l
        n, m = convert1DidxTo2D(len-1, l)
        ptr1, weight1 = IdxerList[begin+m-1]
        ptr2, weight2 = IdxerList[begin+n]
        (sum∘decodePrimCoreInt)(intValCache, (ptr1, ptr2), (weight1, weight2))
    end
    (res,) # ([1|O|1],)
end

function buildIntegralEntries(intCache::OneBodyIntCache{T}, 
                              intIdxerPair::NTuple{2, BasisWeightIndexer{T}}) where {T}
    list1, list2 = getfield.(intIdxerPair, :list)
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
                             intIdxers::AbstractVector{BasisWeightIndexer{T}}) where {T}
    nOrbs = length(intIdxers)
    res = ShapedMemory{T}(undef, (nOrbs, nOrbs))
    for i in 1:nOrbs
        iBI = intIdxers[begin+i-1]
        temp = buildIntegralEntries(intCache, (iBI,))
        setTensorEntries!(res, temp, (i,))
    end

    for l in 1:triMatEleNum(nOrbs-1)
        n, m = convert1DidxTo2D(nOrbs-1, l)
        mBI = intIdxers[begin+m-1]
        nBI = intIdxers[begin+n]
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
    basisCache = PrimOrbCoreCache(T, Val(D), coreType)
    IntegralCache(op, basisCache, OneBodyCompleteGraphIndexer(T))
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
                           paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where 
                          {T, D}
    idxers = prepareIntegralConfig!(intCache, normCache, paramCache, basisSet)
    buildIntegralTensor(intCache, idxers)
end

function computeIntTensor(::OneBodyIntegral, op::F, 
                          basisSet::AbstractVector{<:FrameworkOrb{T, D}}; 
                          paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where 
                         {F<:DirectOperator, T, D}
    intCache, normCache = initializeOneBodyCachePair!(op, paramCache, basisSet)
    computeIntTensor!(intCache, normCache, basisSet; paramCache)
end

computeIntTensor(style::MultiBodyIntegral, op::DirectOperator, orbs::OrbitalBasisSet{T}; 
                 paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where {T} = 
computeIntTensor(style, op, map(FrameworkOrb, orbs); paramCache)


function decomposeOrb!(normCache::OverlapCache{T, D}, paramCache::DimSpanDataCacheBox{T}, 
                       orb::FrameworkOrb{T, D}) where {T, D}
    normIdxList = cachePrimCoreIntegrals!(normCache, paramCache, orb)
    orbWeight = buildOrbWeight!(paramCache, normCache, orb, normIdxList.index)
    normCache.basis.list[normIdxList.index], orbWeight
end

function decomposeOrb!(paramCache::DimSpanDataCacheBox{T}, orb::FrameworkOrb{T}) where {T}
    normCache = initializeOverlapCache!(paramCache, orb)
    decomposeOrb!(normCache, paramCache, orb)
end


function computeIntegral!(intCache::IntegralCache{T, D}, 
                          normCache::OverlapCache{T, D}, 
                          bfs::NonEmptyTuple{FrameworkOrb{T, D}}; 
                          paramCache::DimSpanDataCacheBox{T}) where {T, D}
    bfIndexers = prepareIntegralConfig!(intCache, normCache, paramCache, bfs)
    buildIntegralEntries(intCache, bfIndexers) |> first
end

function computeIntegral(::OneBodyIntegral, op::DirectOperator, 
                         (bf1,)::Tuple{FrameworkOrb{T, D}}; 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                         lazyCompute::Bool=false) where {T, D}
    if lazyCompute
        iCache, nCache = initializeOneBodyCachePair!(op, paramCache, bf1)
        computeIntegral!(iCache, nCache, (bf1,); paramCache)
    else
        coreData, w = decomposeOrb!(paramCache, bf1)
        tensor = computeOneBodyPrimCoreIntTensor(op, (coreData,))
        dot(w, tensor, w)
    end
end

function computeIntegral(::OneBodyIntegral, op::DirectOperator, 
                         (bf1, bf2)::NTuple{2, FrameworkOrb{T, D}}; 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                         lazyCompute::Bool=false) where {T, D}
    coreData1, w1 = decomposeOrb!(paramCache, bf1)
    coreData2, w2 = decomposeOrb!(paramCache, bf2)

    tensor = if lazyCompute
        coreData1 = Vector(coreData1)
        coreData2 = Vector(coreData2)
        coreDataM = intersectMultisets!(coreData1, coreData2, transformation=markObj)
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
                         lazyCompute::Bool=false) where {T, D}
    bf1, bf2 = bfPair
    if lazyCompute
        if bf1 === bf2
            computeIntegral(OneBodyIntegral(), Identity(), (bf1,); paramCache, lazyCompute)
        else
            normCache = initializeOverlapCache!(paramCache, bfPair)
            computeIntegral!(normCache, normCache, bfPair; paramCache)
        end
    else
        coreData1, w1 = decomposeOrb!(paramCache, bf1)
        coreData2, w2 = decomposeOrb!(paramCache, bf2)
        tensor = computeOneBodyPrimCoreIntTensor(Identity(), (coreData1, coreData2))
        dot(w1, tensor, w2)
    end
end

function computeIntegral(style::MultiBodyIntegral, op::DirectOperator, 
                         orbs::NonEmptyTuple{OrbitalBasis{T, D}}; 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                         lazyCompute::Bool=false) where {T, D}
    fOrbs = lazyTupleMap(FrameworkOrb, orbs)
    computeIntegral(style, op, fOrbs; paramCache, lazyCompute)
end


function getOverlapN(orb1::OrbitalBasis{T, D}, orb2::OrbitalBasis{T, D}; 
                     paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                     lazyCompute::Bool=false) where {T, D}
    if orb1 === orb2 && isRenormalized(orb1)
        one(T)
    else
        computeIntegral(OneBodyIntegral(), Identity(), (orb1, orb2); 
                        paramCache, lazyCompute)
    end
end

getOverlapsN(basisSet::OrbitalBasisSet{T}; 
             paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where {T} = 
computeIntTensor(OneBodyIntegral(), Identity(), basisSet; paramCache)