using LinearAlgebra: dot

const N12Tuple{T} = Union{Tuple{T}, NTuple{2, T}}
const N24Tuple{T} = Union{NTuple{2, T}, NTuple{4, T}}

const OrbitalBasisSet{T, D} = AbstractVector{<:OrbitalBasis{T, D}}
const OrbitalSet{T, D} = AbstractVector{<:FrameworkOrb{T, D}}
const FPrimOrbSet{T, D} = AbstractVector{<:FPrimOrb{T, D}}

const OrbCoreIdxDict{T} = 
      Dict{Tuple{FieldMarker{:PrimitiveOrbCore, 1}, AbtVecOfAbtArr{T}}, Int}

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


function getPrimOrbCores(orb::ScaledOrbital)
    getMemory(orb.f.apply.left)
end

function getPrimOrbCores(orb::FPrimOrb)
    getPrimOrbCores(orb.core)
end

function getPrimOrbCores(orb::FCompOrb)
    map(getPrimOrbCores(orb.core)[].f.chain) do wf
        getPrimOrbCores(wf.left)[]
    end
end


function genOrbCoreData!(paramCache::DimSpanDataCacheBox{T}, orb::FPrimOrb{T}) where {T}
    oCore = getPrimOrbCores(orb)[]
    pVal = cacheParam!(paramCache, orb.param, orb.pointer.scope)
    (oCore, pVal)
end

function genOrbCoreData!(paramCache::DimSpanDataCacheBox{T}, orbs::FPrimOrbSet{T}) where {T}
    map(orbs) do orb
        genOrbCoreData!(paramCache, orb)
    end
end


function updatePrimOrbCoreCache!(basisCache::PrimOrbCoreCache{T, D}, 
                                 paramCache::DimSpanDataCacheBox{T}, 
                                 orb::FPrimOrb{T, D}, 
                                 anchor::Int=firstindex(basisCache.list)) where {T, D}
    data = genOrbCoreData!(paramCache, orb)
    basis = basisCache.list
    idx = get!(basisCache.dict, ( (markObj∘first)(data), last(data) )) do
        push!(basis, data)
        lastindex(basis)
    end
    idx, max(idx, anchor)
end

function cachePrimOrbCoreDataCore!(basisCache::PrimOrbCoreCache{T, D}, 
                                   paramCache::DimSpanDataCacheBox{T}, 
                                   orb::FPrimOrb{T, D}) where {T, D}
    data = genOrbCoreData!(paramCache, orb)
    basis = basisCache.list
    get!(basisCache.dict, ( (markObj∘first)(data), last(data) )) do
        push!(basis, data)
        lastindex(basis)
    end
end

function cachePrimOrbCoreData!(basisCache::PrimOrbCoreCache{T, D}, 
                               paramCache::DimSpanDataCacheBox{T}, 
                               orb::FPrimOrb{T, D}) where {T, D}
    orb => (getMemory∘cachePrimOrbCoreDataCore!)(basisCache, paramCache, orb)
end

function cachePrimOrbCoreData!(basisCache::PrimOrbCoreCache{T, D}, 
                               paramCache::DimSpanDataCacheBox{T}, 
                               orb::FCompOrb{T, D}) where {T, D}
    ids = map(orb|>splitOrb) do pOrb
        cachePrimOrbCoreDataCore!(basisCache, paramCache, pOrb)
    end
    orb => getMemory(ids)
end


function cachePrimCoreIntegrals!(intCache::IntegralCache{T, D}, 
                                 paramCache::DimSpanDataCacheBox{T}, 
                                 orbs::OrbitalSet{T, D}) where {T, D}
    checkEmptiness(orbs, :orbs)
    orbCache = intCache.basis
    oldMaxIdx = lastindex(orbCache.list)
    orbPairList = map(orbs) do orb
        cachePrimOrbCoreData!(orbCache, paramCache, orb)
    end
    updatePrimCoreIntCache!(intCache, oldMaxIdx+1)
    orbPairList
end

function cachePrimCoreIntegrals!(intCache::IntegralCache{T, D}, 
                                 paramCache::DimSpanDataCacheBox{T}, 
                                 orb::FrameworkOrb{T, D}) where {T, D}
    orbCache = intCache.basis
    oldMaxIdx = lastindex(orbCache.list)
    orbPair = cachePrimOrbCoreData!(orbCache, paramCache, orb)
    updatePrimCoreIntCache!(intCache, oldMaxIdx+1)
    orbPair
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

function buildOrbWeightInfoCore(orbPtrPair::Pair{<:FPrimOrb{T, D}, Memory{Int}}, 
                                normCache::OverlapCache{T, D}) where {T, D}
    orb = orbPtrPair.first
    ptr = orbPtrPair.second[]
    if isRenormalized(orb)
        decodePrimCoreInt(normCache.data, (ptr,)) |> first |> AbsSqrtInv
    else
        one(T)
    end
end


function buildOrbWeightInfoCore(::Val{true}, weights::AbstractVector{T}, 
                                orbs::FPrimOrbSet{T, D}, 
                                ptrs::AbstractVector{Int}, 
                                normCache::OverlapCache{T, D}) where {T, D}
    nOrbs = length(orbs)
    innerOverlapSum = zero(T)
    overlapCache = normCache.data
    pOrbNormCoeffs = map(weights, orbs, ptrs) do w, pOrb, ptr
        innerCoreDiagOverlap = decodePrimCoreInt(overlapCache, (ptr,)) |> first
        innerDiagOverlap = w' * w
        if isRenormalized(pOrb)
            w *= AbsSqrtInv(innerCoreDiagOverlap)
        else
            innerDiagOverlap *= innerCoreDiagOverlap
        end
        innerOverlapSum += innerDiagOverlap
        w
    end

    innerOverlapSum += mapreduce(+, 1:triMatEleNum(nOrbs-1)) do l
        n, m = convert1DidxTo2D(nOrbs-1, l)
        ptrPair = (ptrs[begin+m-1], ptrs[begin+n])
        coeffPair = (pOrbNormCoeffs[begin+m-1], pOrbNormCoeffs[begin+n])
        (sum∘decodePrimCoreInt)(overlapCache, ptrPair, coeffPair)
    end

    outerNormCoeff = AbsSqrtInv(innerOverlapSum)
    pOrbNormCoeffs .*= outerNormCoeff
    pOrbNormCoeffs
end

function buildOrbWeightInfoCore(::Val{false}, weights::AbstractVector{T}, 
                                orbs::FPrimOrbSet{T, D}, 
                                ptrs::AbstractVector{Int}, 
                                normCache::OverlapCache{T, D}) where {T, D}
    map(weights, orbs, ptrs) do w, pOrb, ptr
        w * buildOrbWeightInfoCore(pOrb=>getMemory(ptr), normCache)
    end
end

function cacheOrbBareWeight!(paramCache::DimSpanDataCacheBox{T}, orb::FCompOrb{T}) where {T}
    cacheParam!(paramCache, getOrbWeightCore(orb))
end

function cacheOrbBareWeight!(::DimSpanDataCacheBox{T}, ::FPrimOrb{T}) where {T}
    one(T)
end

function buildOrbWeightInfo!(orbPtrPair::Pair{<:FCompOrb{T, D}, Memory{Int}}, 
                             normCache::OverlapCache{T, D}, 
                             paramCache::DimSpanDataCacheBox{T}) where {T, D}
    orb = orbPtrPair.first
    normBool = isRenormalized(orb)
    wVal = cacheOrbBareWeight!(paramCache, orb)
    pOrbs = splitOrb(orb)
    ptrs = orbPtrPair.second

    res = buildOrbWeightInfoCore(Val(normBool), wVal, pOrbs, ptrs, normCache)
    getMemory(res)
end

#? Maybe a better signature?
buildOrbWeightInfo!(orbPtrPair::Pair{<:FPrimOrb{T, D}, Memory{Int}}, 
                    normCache::OverlapCache{T, D}, 
                    ::DimSpanDataCacheBox{T}) where {T, D} = 
getMemory(buildOrbWeightInfoCore(orbPtrPair, normCache))


function extractOrbWeightData!(normCache::OverlapCache{T, D}, 
                               paramCache::DimSpanDataCacheBox{T}, 
                               orb::FrameworkOrb{T, D}) where {T, D}
    normIdxPair = cachePrimCoreIntegrals!(normCache, paramCache, orb)
    buildOrbWeightInfo!(normIdxPair, normCache, paramCache)
end

function extractOrbWeightData!(normCache::OverlapCache{T, D}, 
                               paramCache::DimSpanDataCacheBox{T}, 
                               orbs::OrbitalSet{T, D}) where {T, D}
    map(orbs) do orb
        extractOrbWeightData!(normCache, paramCache, orb)
    end
end


function prepareIntegralConfigCore(orbPtrs::AbstractVector{Int}, 
                                   orbWeights::AbstractVector{T}) where {T}
    map(orbPtrs, orbWeights) do ptr, w
        ptr => w
    end |> getMemory |> BasisWeightIndexer
end

function prepareIntegralConfig!(intCache::IntegralCache{T, D}, 
                                normCache::OverlapCache{T, D}, 
                                paramCache::DimSpanDataCacheBox{T}, 
                                orbs::OrbitalSet{T, D}) where {T, D}
    intIdxPairs = cachePrimCoreIntegrals!(intCache, paramCache, orbs)
    map(orbs, intIdxPairs) do orb, intIdxPair
        orbWeights = extractOrbWeightData!(normCache, paramCache, orb)
        prepareIntegralConfigCore(intIdxPair.second, orbWeights)
    end
end

function prepareIntegralConfig!(intCache::IntegralCache{T, D}, 
                                normCache::OverlapCache{T, D}, 
                                paramCache::DimSpanDataCacheBox{T}, 
                                orb::FrameworkOrb{T, D}) where {T, D}
    intIdxPair = cachePrimCoreIntegrals!(intCache, paramCache, orb)
    orbWeights = extractOrbWeightData!(normCache, paramCache, orb)
    prepareIntegralConfigCore(intIdxPair.second, orbWeights)
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


function initializeIntegralCache!(::OneBodyIntegral, op::F, 
                                  paramCache::DimSpanDataCacheBox{T}, 
                                  orbs::OrbitalSet{T, D}) where {F<:DirectOperator, T, D}
    checkEmptiness(orbs, :orbs)
    orbCoreType = mapreduce(typejoin, orbs, init=Union{}) do orb
        (eltype∘getPrimOrbCores)(orb)
    end
    basisCache = PrimOrbCoreCache(T, Val(D), orbCoreType)
    IntegralCache(op, basisCache, OneBodyCompleteGraphIndexer(T))
end

initializeOverlapCache!(paramCache::DimSpanDataCacheBox{T}, orbs::OrbitalSet{T}) where {T} = 
initializeIntegralCache!(OneBodyIntegral(), Identity(), paramCache, orbs)


function initializeOneBodyCachePair!(op::F, paramCache::DimSpanDataCacheBox{T}, 
                                     basisSet::OrbitalSet{T}) where {F<:DirectOperator, T}
    intCache = initializeIntegralCache!(OneBodyIntegral(), op, paramCache, basisSet)
    normCache = F <: Identity ? intCache : initializeOverlapCache!(paramCache, basisSet)
    intCache, normCache
end


function computeIntTensor!(intCache::IntegralCache{T, D}, 
                           normCache::OverlapCache{T, D}, 
                           basisSet::OrbitalSet{T, D}; 
                           paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where 
                          {T, D}
    idxers = prepareIntegralConfig!(intCache, normCache, paramCache, basisSet)
    buildIntegralTensor(intCache, idxers)
end

function computeIntTensor(::OneBodyIntegral, op::F, basisSet::OrbitalSet{T}; 
                          paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where 
                         {F<:DirectOperator, T}
    intCache, normCache = initializeOneBodyCachePair!(op, paramCache, basisSet)
    computeIntTensor!(intCache, normCache, basisSet; paramCache)
end

computeIntTensor(style::MultiBodyIntegral, op::DirectOperator, orbs::OrbitalBasisSet{T}; 
                 paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where {T} = 
computeIntTensor(style, op, map(FrameworkOrb, orbs); paramCache)


function decomposeOrb!(paramCache::DimSpanDataCacheBox{T}, orb::FrameworkOrb{T}) where {T}
    normCache = initializeOverlapCache!(paramCache, getMemory(orb))
    weight = extractOrbWeightData!(normCache, paramCache, orb)
    coreDataSet = genOrbCoreData!(paramCache, splitOrb(orb))
    coreDataSet, weight
end


function computeIntegral!(intCache::IntegralCache{T, D}, 
                          normCache::OverlapCache{T, D}, 
                          bfs::NonEmptyTuple{FrameworkOrb{T, D}}; 
                          paramCache::DimSpanDataCacheBox{T}) where {T, D}
    bfIndexers = map(bfs) do bf
        prepareIntegralConfig!(intCache, normCache, paramCache, bf)
    end
    buildIntegralEntries(intCache, bfIndexers) |> first
end

function computeIntegral(::OneBodyIntegral, op::DirectOperator, 
                         (bf1,)::Tuple{FrameworkOrb{T, D}}; 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                         lazyCompute::Bool=false) where {T, D}
    if lazyCompute
        iCache, nCache = initializeOneBodyCachePair!(op, paramCache, getMemory(bf1))
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
            normCache = initializeOverlapCache!(paramCache, getMemory(bfPair))
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