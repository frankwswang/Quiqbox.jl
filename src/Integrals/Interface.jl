using LinearAlgebra: dot

const N12Tuple{T} = Union{Tuple{T}, NTuple{2, T}}
const N24Tuple{T} = Union{NTuple{2, T}, NTuple{4, T}}

const OrbitalBasisSet{T, D} = AbstractVector{<:OrbitalBasis{T, D}}

const OrbitalSet{T, D} = AbstractVector{<:FrameworkOrb{T, D}}

const FPrimOrbSet{T, D} = AbstractVector{<:FPrimOrb{T, D}}

const OrbCoreIdxDict{T} = 
      Dict{Tuple{FieldMarker{:PrimitiveOrbCore, 1}, AbtVecOfAbtArr{T}}, Int}

const OrbCoreData{T, D, F<:PrimitiveOrbCore{T, D}, V<:AbtVecOfAbtArr{T}} = Tuple{F, V}

const FPrimOrbData{T, D, B<:AbstractVector{<:FPrimOrb{T, D}}, W<:AbstractVector{T}} = 
      Tuple{B, W}

const OrbCoreInfoVec{T, D, F<:PrimitiveOrbCore{T, D}} = 
      Vector{ OrbCoreData{T, D, F, Vector{ ShapedMemory{T} }} }

const AbtOrbCoreInfoArr{T, D, F<:PrimitiveOrbCore{T, D}, V<:AbtVecOfAbtArr{T}, N} = 
      AbstractArray{OrbCoreData{T, D, F, V}, N}

const AbtOrbCoreInfoVec{T, D, F<:PrimitiveOrbCore{T, D}, V<:AbtVecOfAbtArr{T}} = 
      AbtOrbCoreInfoArr{T, D, F, V, 1}

const OneBodyIntOrbInfo{T, D} = NTuple{2, AbtOrbCoreInfoVec{T, D}}
const TwoBodyIntOrbInfo{T, D} = NTuple{4, AbtOrbCoreInfoVec{T, D}}

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
    list::OrbCoreInfoVec{T, D, F}
end

PrimOrbCoreCache(::Type{T}, ::Val{D}, ::Type{F}=PrimitiveOrbCore{T, D}) where 
                {T, D, F<:PrimitiveOrbCore{T, D}} = 
PrimOrbCoreCache(OrbCoreIdxDict{T}(), eltype(OrbCoreInfoVec{T, D, F})[])

struct BasisIndexer{T}
    list::Memory{Tuple{Int, T, Bool}} # index, coeff, normalization
    renormalize::Bool
end

const BasisIdxerVec{T} = AbstractVector{BasisIndexer{T}}

abstract type IntegralIndexer{T, S} <: QueryBox{T} end

struct GenericOneBodyIndexer{T<:Number} <: IntegralIndexer{T, OneBodyIntegral}
    aa::Dict{ Tuple{   Int},  Tuple{   T}}
    ab::Dict{NTuple{2, Int}, NTuple{2, T}}
end

function GenericOneBodyIndexer(::Type{T}) where {T}
    GenericOneBodyIndexer(Dict{ Tuple{   Int},  Tuple{     T}}(), 
                           Dict{NTuple{2, Int}, NTuple{2,   T}}())
end

initializeIntegralIndexer(::OneBodyIntegral, ::Type{T}) where {T} = GenericOneBodyIndexer(T)

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

function setIntegralIndexer!(idxer::GenericOneBodyIndexer{T}, 
                             pair::Pair{Tuple{Int}, Tuple{T}}) where {T}
    setindex!(getfield(idxer, OneBodyIdxSymDict[true ]), pair.second, pair.first)
    idxer
end

function setIntegralIndexer!(idxer::GenericOneBodyIndexer{T}, 
                             pair::Pair{NTuple{2, Int}, NTuple{2, T}}) where {T}
    setindex!(getfield(idxer, OneBodyIdxSymDict[false]), pair.second, pair.first)
    idxer
end

function getPrimCoreIntData(cache::GenericOneBodyIndexer, formattedIdx::NTuple{2, Int})
    getfield(cache, OneBodyIdxSymDict[false])[formattedIdx]
end

function getPrimCoreIntData(cache::GenericOneBodyIndexer, formattedIdx::Tuple{Int})
    getfield(cache, OneBodyIdxSymDict[true ])[formattedIdx]
end


genOneBodyCoreIntegrator(::Identity, orbs::N12Tuple{PrimGTOcore{T, D}}) where {T, D} = 
genGTOrbOverlapFunc(orbs)


isHermitian(::PrimitiveOrbCore{T, D}, ::DirectOperator, 
            ::PrimitiveOrbCore{T, D}) where {T, D} = 
false

isHermitian(::PrimitiveOrbCore{T, D}, ::Identity, 
            ::PrimitiveOrbCore{T, D}) where {T, D} = 
true

function genOneBodyPrimCoreIntPairs(op::DirectOperator, 
                                    (oData,)::Tuple{AbtOrbCoreInfoVec{T, D}}, 
                                    (indexOffset,)::Tuple{Int}=(0,)) where {T, D}
    iFirst = firstindex(oData)
    nOrbs = length(oData)
    dm = Plus(iFirst - 1)

    pairs1 = Memory{Pair{Tuple{Int}, Tuple{T}}}(undef, nOrbs)
    pairs2 = map(1:triMatEleNum(nOrbs)) do l
        n, m = convert1DidxTo2D(nOrbs, l)
        if m == n
            i = dm(m)
            iiVal = computeOneBodyPrimCoreIntVals(op, oData, i)
            pairs1[begin+n-1] = (i+indexOffset,) => iiVal
        else
            i, j = sortTensorIndex((m, n)) .|> dm
            ijValPair = computeOneBodyPrimCoreIntVals(op, oData, (i, j))
            (i+indexOffset, j+indexOffset) => ijValPair
        end
    end

    pairs1, pairs2
end

function computeOneBodyPrimCoreIntVals(op::DirectOperator, oData::AbtOrbCoreInfoArr{T}, 
                                       i::Int) where {T}
    orb, pars = oData[i]
    f = ReturnTyped(genOneBodyCoreIntegrator(op, (orb,)), T)
    (f(pars),)
end

function evalOneBodyPrimCoreIntConfig(op::DirectOperator, 
                                      (orb1, pars1)::OrbCoreData{T, D}, 
                                      (orb2, pars2)::OrbCoreData{T, D}) where {T, D}
    f = ReturnTyped(genOneBodyCoreIntegrator(op, (orb1, orb2)), T)
    f(pars1, pars2)
end

function computeOneBodyPrimCoreIntVals(op::DirectOperator, 
                                       (oData1, oData2)::OneBodyIntOrbInfo{T, D}, 
                                       (i, j)::NTuple{2, Int}) where {T, D}
    orbPars1 = oData1[i]
    orbPars2 = oData2[j]
    ijVal = evalOneBodyPrimCoreIntConfig(op, orbPars1, orbPars2)
    jiVal = if isHermitian(first(orbPars1), op, first(orbPars2))
        ijVal'
    else
        evalOneBodyPrimCoreIntConfig(op, orbPars2, orbPars1)
    end
    (ijVal, jiVal)
end

function computeOneBodyPrimCoreIntVals(op::DirectOperator, oData::AbtOrbCoreInfoVec, 
                                       idx::NTuple{2, Int})
    computeOneBodyPrimCoreIntVals(op, (oData, oData), idx)
end

function genOneBodyPrimCoreIntPairs(op::DirectOperator, data::OneBodyIntOrbInfo{T, D}, 
                                    (dIdx1, dIdx2)::NTuple{2, Int}) where {T, D}
    map(Iterators.product( eachindex.(data) )) do (i, j)
        ijValPair = computeOneBodyPrimCoreIntVals(op, data, (i, j))
        (i+dIdx1, j+dIdx2) => ijValPair
    end |> vec
end

#!! Add index offset and double-check index alignment.
function computeOneBodyPrimCoreIntTensor(op::DirectOperator, 
                                         (oData1,)::Tuple{AbtOrbCoreInfoVec{T}}) where {T}
    nBasis = length(oData1)
    res = ShapedMemory{T}(undef, (nBasis, nBasis))
    for i in 1:nBasis
        temp = computeOneBodyPrimCoreIntVals(op, oData1, (i, i))
        setTensorEntries!(res, temp, (i,))
    end

    for l in 1:triMatEleNum(nBasis-1)
        n, m = convert1DidxTo2D(mBasis-1, l)
        temp = computeOneBodyPrimCoreIntVals(op, oData1, (m, n+1))
        setTensorEntries!(res, temp, (m, n+1))
    end
    res
end



function computeOneBodyPrimCoreIntTensor(op::DirectOperator, 
                                         data::OneBodyIntOrbInfo{T, D}) where {T, D}
    oData1, oData2 = data
    len1, len2 = length.(data)
    res = ShapedMemory{T}(undef, (len1, len2))
    for j in 1:len2, i in 1:len1
        ijVal = evalOneBodyPrimCoreIntConfig(op, oData1[begin+i-1], oData2[begin+j-1])
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

function updatePrimCoreIntCache!(cache::IntegralCache, startIdx::Int)
    op = cache.operator
    basis = cache.basis.list
    intIdxer = cache.data
    firstIdx = firstindex(basis)

    if startIdx == firstIdx
        updateIntCacheCore!(op, intIdxer, (basis,), (0,))
    elseif firstIdx < startIdx <= lastindex(basis)
        oldBasis = @view basis[begin:startIdx-1]
        newBasis = @view basis[startIdx:    end]
        updateIntCacheCore!(op, intIdxer, (newBasis,), (boundary,))
        updateIntCacheCore!(op, intIdxer, (oldBasis, newBasis,), (0, boundary))
    end

    cache
end

function updateIntCacheCore!(op::DirectOperator, idxer::GenericOneBodyIndexer{T}, 
                             basis::Tuple{AbtOrbCoreInfoVec{T, D}}, 
                             offset::Tuple{Int}) where {T, D}
    pairs1, pairs2 = genOneBodyPrimCoreIntPairs(op, basis, offset)
    foreach(p->setIntegralIndexer!(idxer, p), pairs1)
    foreach(p->setIntegralIndexer!(idxer, p), pairs2)
    idxer
end

function updateIntCacheCore!(op::DirectOperator, idxer::GenericOneBodyIndexer{T}, 
                             basis::OneBodyIntOrbInfo{T, D}, 
                             offset::NTuple{2, Int}) where {T, D}
    pairs2 = genOneBodyPrimCoreIntPairs(op, basis, offset)
    foreach(p->setIntegralIndexer!(idxer, p), pairs2)
    idxer
end

struct BasisWeightIndexer{T}
    list::Memory{Pair{Int, T}}
end

function decodePrimCoreInt(cache::GenericOneBodyIndexer{T}, ptrPair::NTuple{2, Int}, 
                           (coeff1, coeff2)::NTuple{2, T}=( one(T), one(T) )) where {T}
    coeffProd = coeff1' * coeff2
    ptrPairNew = sortTensorIndex(ptrPair)
    data = getPrimCoreIntData(cache, ptrPairNew)
    (ptrPairNew == ptrPair ? data : reverse(data)) .* (coeffProd, coeffProd')
end

function decodePrimCoreInt(cache::GenericOneBodyIndexer{T}, ptr::Tuple{Int}, 
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

#? Maybe a better signature?
buildOrbWeightInfo!(orbPtrPair::Pair{<:FPrimOrb{T, D}, Memory{Int}}, 
                    normCache::OverlapCache{T, D}, 
                    ::DimSpanDataCacheBox{T}) where {T, D} = 
getMemory(buildOrbWeightInfoCore(orbPtrPair, normCache))


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

function setTensorEntries!(tensor::AbstractMatrix{T}, valPair::NTuple{2, T}, 
                           idxPair::NTuple{2, Int}) where {T}
    i, j = first.(axes(tensor)) .+ idxPair .- 1
    tensor[i, j] = first(valPair)
    tensor[j, i] = last(valPair)
end

function setTensorEntries!(tensor::AbstractArray{T}, (val,)::Tuple{T}, 
                           (idx,)::Tuple{Int}) where {T}
    idxSet = first.(axes(tensor)) .+ idx .- 1
    tensor[idxSet...] = val
end

function buildIntegralTensor(intCache::OneBodyIntCache{T}, 
                             intIdxers::AbstractVector{BasisWeightIndexer{T}}) where {T}
    nBasis = length(intIdxers)
    res = ShapedMemory{T}(undef, (nBasis, nBasis))
    for i in 1:nBasis
        iBI = intIdxers[begin+i-1]
        temp = buildIntegralEntries(intCache, (iBI,))
        setTensorEntries!(res, temp, (i,))
    end

    for l in 1:triMatEleNum(nBasis-1)
        n, m = convert1DidxTo2D(nBasis-1, l)
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
    IntegralCache(op, basisCache, initializeIntegralIndexer(OneBodyIntegral(), T))
end

initializeOverlapCache!(paramCache::DimSpanDataCacheBox{T}, orbs::OrbitalSet{T}) where {T} = 
initializeIntegralCache!(OneBodyIntegral(), Identity(), paramCache, orbs)

function computeIntTensor!(intCache::IntegralCache{T, D}, 
                           normCache::OverlapCache{T, D}, 
                           basisSet::OrbitalSet{T, D}; 
                           paramCache::DimSpanDataCacheBox{T}) where {T, D}
    idxers = prepareIntegralConfig!(intCache, normCache, paramCache, basisSet)
    buildIntegralTensor(intCache, idxers)
end

function initializeOneBodyCachePair!(op::F, paramCache::DimSpanDataCacheBox{T}, 
                                     basisSet::OrbitalSet{T}) where {F<:DirectOperator, T}
    intCache = initializeIntegralCache!(OneBodyIntegral(), op, paramCache, basisSet)
    normCache = F <: Identity ? intCache : initializeOverlapCache!(paramCache, basisSet)
    intCache, normCache
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

getOverlapsN(basisSet::OrbitalBasisSet{T}; 
             paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where {T} = 
computeIntTensor(OneBodyIntegral(), Identity(), basisSet; paramCache)

function getPrimOrbsNormCoeff!(paramCache::DimSpanDataCacheBox{T}, 
                               pOrbCluster::FPrimOrbSet{T}) where {T}
    normCache = initializeOverlapCache!(paramCache, pOrbCluster)
    extractOrbWeightData!(normCache, paramCache, pOrbCluster) .|> getindex
end

# function computePrimIntTensor!(::OneBodyIntegral, op::DirectOperator, 
#                                (poc1, poc2)::NTuple{2, FPrimOrbSet{T, D}}; 
#                                paramCache::DimSpanDataCacheBox{T}) where {T, D}
#     w1 = getPrimOrbsNormCoeff!(paramCache, poc1)
#     w2 = getPrimOrbsNormCoeff!(paramCache, poc2)
#     coreData1 = genOrbCoreData!(paramCache, poc1)
#     coreData2 = genOrbCoreData!(paramCache, poc2)
#     tensor = computeOneBodyPrimCoreIntTensor(op, (coreData1, coreData2))

#     dot(w1, tensor, w2)
# end

function computePrimIntTensor!(::OneBodyIntegral, op::DirectOperator, 
                               (pOrbCluster,)::Tuple{FPrimOrbSet{T, D}}; 
                               paramCache::DimSpanDataCacheBox{T}) where {T, D}
    w1 = getPrimOrbsNormCoeff!(paramCache, pOrbCluster)
    coreData = genOrbCoreData!(paramCache, pOrbCluster)

    tensor = computeOneBodyPrimCoreIntTensor(op, (coreData,))

    dot(w1, tensor, w1)
end

function decomposeOrb!(paramCache::DimSpanDataCacheBox{T}, orb::FrameworkOrb{T}) where {T}
    normCache = initializeOverlapCache!(paramCache, getMemory(orb))
    weight = extractOrbWeightData!(normCache, paramCache, orb)
    coreDataSet = genOrbCoreData!(paramCache, splitOrb(orb))
    coreDataSet, weight
end

function decomposeOrbs!(paramCache::DimSpanDataCacheBox{T}, 
                        (b1, b2)::NTuple{2, FrameworkOrb{T, D}}) where {T, D}
    b1Config = decomposeOrb!(paramCache, b1)
    b2Config = b2 === b1 ? b1Config : decomposeOrb!(paramCache, b2)
    (b1Config, b2Config)
end

function decomposeOrbs!(paramCache::DimSpanDataCacheBox{T}, 
                        (b1, b2, b3, b4)::NTuple{4, FrameworkOrb{T, D}}) where {T, D}
    b1Config = decomposeOrb!(paramCache, b1)
    b2Config = b2 === b1 ? b1Config : decomposeOrb!(paramCache, b2)
    b3Config = b3 === b1 ? b1Config : (b3 === b2 ? b2Config : decomposeOrb!(paramCache, b3))
    b4Config = if b4 === b1
        b1Config
    else
        b4 === b2 ? b2Config : (b4 === b3 ? b3Config : decomposeOrb!(paramCache, b4))
    end
    (b1Config, b2Config, b3Config, b4Config)
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

function getPrimOrbsNCoeff!(paramCache::DimSpanDataCacheBox{T}, 
                               pOrbCluster::FPrimOrbSet{T}) where {T}
    normCache = initializeOverlapCache!(paramCache, pOrbCluster)
    extractOrbWeightData!(normCache, paramCache, pOrbCluster) .|> getindex
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
    if lazyCompute
        normCache = initializeOverlapCache!(paramCache, getMemory(bfPair))
        computeIntegral!(normCache, normCache, bfPair; paramCache)
    else
        coreData1, w1 = decomposeOrb!(paramCache, first(bfPair))
        coreData2, w2 = decomposeOrb!(paramCache,  last(bfPair))
        tensor = computeOneBodyPrimCoreIntTensor(Identity(), (coreData1, coreData2))
        dot(w1, tensor, w2)
    end
end

computeIntegral(style::MultiBodyIntegral, op::DirectOperator, 
                orbs::NonEmptyTuple{OrbitalBasis{T, D}}; 
                paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                lazyCompute::Bool=false) where {T, D} = 
computeIntegral(style, op, FrameworkOrb.(orbs); paramCache, lazyCompute)


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