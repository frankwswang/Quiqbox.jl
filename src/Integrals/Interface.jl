using LinearAlgebra: dot

const OneTwoTpl{T} = Union{Tuple{T}, NTuple{2, T}}

const FrameworkOrbSet{T, D} = AbstractVector{<:FrameworkOrb{T, D}}

const FPrimOrbSet{T, D} = AbstractVector{<:FPrimOrb{T, D}}

const OrbCoreIdxDict{T} = 
      Dict{Tuple{FieldMarker{:PrimitiveOrbCore, 1}, AbtVecOfAbtArr{T}}, Int}

const OrbCoreData{T, D, F<:PrimitiveOrbCore{T, D}, V<:AbtVecOfAbtArr{T}} = Tuple{F, V}

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
    list::Memory{Pair{Int, T}}
    renormalize::Bool
end

const BasisIdxerVec{T} = AbstractVector{BasisIndexer{T}}

abstract type IntegralIndexer{T, N} <: QueryBox{T} end

struct OneBodyIntegralIndexer{T<:Number} <: IntegralIndexer{T, 1}
    aa::Dict{ Tuple{   Int},  Tuple{   T}}
    ab::Dict{NTuple{2, Int}, NTuple{2, T}}
end

function OneBodyIntegralIndexer(::Type{T}) where {T}
    OneBodyIntegralIndexer(Dict{ Tuple{   Int},  Tuple{     T}}(), 
                           Dict{NTuple{2, Int}, NTuple{2,   T}}())
end

initializeIntIndexer(::Type{T}, ::Val{1}) where {T} = OneBodyIntegralIndexer(T)

struct IntegralCache{T, D, N, F<:DirectOperator, V, I<:IntegralIndexer{T, N}} <: QueryBox{T}
    operator::F
    basis::PrimOrbCoreCache{T, D, V}
    data::I
end

function setIntegralIndexer!(idxer::OneBodyIntegralIndexer{T}, 
                             pair::Pair{Tuple{Int}, Tuple{T}}) where {T}
    setindex!(getfield(idxer, OneBodyIdxSymDict[true ]), pair.second, pair.first)
    idxer
end

function setIntegralIndexer!(idxer::OneBodyIntegralIndexer{T}, 
                             pair::Pair{NTuple{2, Int}, NTuple{2, T}}) where {T}
    setindex!(getfield(idxer, OneBodyIdxSymDict[false]), pair.second, pair.first)
    idxer
end


genPrimIntegrator(::Identity, orb::PrimGTOcore) = genOverlapFunc(orb)

genPrimIntegrator(::Identity, orb1::PrimGTOcore{T, D}, orb2::PrimGTOcore{T, D}) where 
                 {T, D} = 
genOverlapFunc(orb1, orb2)

isHermitian(::DirectOperator, 
            ::PrimitiveOrbCore{T, D}, ::PrimitiveOrbCore{T, D}) where {T, D} = 
false

isHermitian(::Identity, 
            ::PrimitiveOrbCore{T, D}, ::PrimitiveOrbCore{T, D}) where {T, D} = 
true

function genOneBodyPrimIntPairs(op::DirectOperator, 
                                (oData,)::Tuple{AbtOrbCoreInfoVec{T, D}}, 
                                (indexOffset,)::Tuple{Int}=(0,)) where {T, D}
    iFirst = firstindex(oData)
    nOrbs = length(oData)
    dIdx = Plus(iFirst - 1)

    pairs1 = Memory{Pair{Tuple{Int}, Tuple{T}}}(undef, nOrbs)
    pairs2 = map(1:triMatEleNum(nOrbs)) do l
        n, m = convert1DidxTo2D(nOrbs, l)
        if m == n
            i = dIdx(m)
            iiVal = computeOneBodyPrimIntVals(op, oData, i)
            pairs1[begin+n-1] = (i+indexOffset,) => iiVal
        else
            i, j = sortTensorIndex((m, n)) .|> dIdx
            ijValPair = computeOneBodyPrimIntVals(op, oData, (i, j))
            (i+indexOffset, j+indexOffset) => ijValPair
        end
    end

    pairs1, pairs2
end

function computeOneBodyPrimIntVals(op::DirectOperator, oData::AbtOrbCoreInfoArr{T}, 
                                   i::Int) where {T}
    orb, pars = oData[i]
    f = ReturnTyped(genPrimIntegrator(op, orb), T)
    (f(pars),)
end

function computeOneBodyPrimIntCore(op::DirectOperator, 
                                  (orb1, pars1)::OrbCoreData{T, D}, 
                                  (orb2, pars2)::OrbCoreData{T, D}) where {T, D}
    f = ReturnTyped(genPrimIntegrator(op, orb1, orb2), T)
    f(pars1, pars2)
end

function computeOneBodyPrimIntVals(op::DirectOperator, 
                                   (oData1, oData2)::OneBodyIntOrbInfo{T, D}, 
                                   (i, j)::NTuple{2, Int}) where {T, D}
    orbPars1 = oData1[i]
    orbPars2 = oData2[j]
    ijVal = computeOneBodyPrimIntCore(op, orbPars1, orbPars2)
    jiVal = if isHermitian(op, first(orbPars1), first(orbPars2))
        ijVal'
    else
        computeOneBodyPrimIntCore(op, orbPars2, orbPars1)
    end
    (ijVal, jiVal)
end

function computeOneBodyPrimIntVals(op::DirectOperator, oData::AbtOrbCoreInfoVec, 
                                   idx::NTuple{2, Int})
    computeOneBodyPrimIntVals(op, (oData, oData), idx)
end

function genOneBodyPrimIntPairs(op::DirectOperator, data::OneBodyIntOrbInfo{T, D}, 
                                (dIdx1, dIdx2)::NTuple{2, Int}) where {T, D}
    map(Iterators.product( eachindex.(data) )) do (i, j)
        ijValPair = computeOneBodyPrimIntVals(op, data, (i, j))
        (i+dIdx1, j+dIdx2) => ijValPair
    end |> vec
end

function computePrimCoreIntTensor(op::DirectOperator, 
                                   data::OneBodyIntOrbInfo{T, D}) where {T, D}
    oData1, oData2 = data
    res = ShapedMemory{T}(undef, length.(data))
    di, dj = Plus.(first.(axes(res)) .- firstindex.(data))
    for j in eachindex(oData2), i in eachindex(oData1)
        ijVal = computeOneBodyPrimIntCore(op, oData1[i], oData2[j])
        res[di(i), dj(j)] = ijVal
    end
    res
end


function getOrbitalCores(orb::ScaledOrbital)
    getMemory([orb.f.apply.left])
end

function getOrbitalCores(orb::FPrimOrb)
    getOrbitalCores(orb.core)
end

function getOrbitalCores(orb::FCompOrb)
    map(getOrbitalCores(orb.core)[].f.chain) do wf
        getOrbitalCores(wf.left)[]
    end
end

function genOrbCoreData!(paramCache::DimSpanDataCacheBox{T}, orb::FPrimOrb{T}) where {T}
    oCore = getOrbitalCores(orb)[]
    pVal = cacheParam!(paramCache, orb.param, orb.pointer.scope)
    (oCore, pVal)
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


function cacheIntComponentCore!(intCache::IntegralCache{T, D}, 
                                paramCache::DimSpanDataCacheBox{T}, 
                                orb::FPrimOrb{T, D}) where {T, D}
    idx, maxIdx = updatePrimOrbCoreCache!(intCache.basis, paramCache, orb)
    BasisIndexer(getMemory((idx => one(T),)), isRenormalized(orb)), maxIdx
end

function cacheIntComponentCore!(intCache::IntegralCache{T, D}, 
                                paramCache::DimSpanDataCacheBox{T}, 
                                orb::FCompOrb{T, D}) where {T, D}
    basis = intCache.basis
    pOrbs = decomposeOrb(orb)
    wVal = cacheOrbWeight!(paramCache, orb)
    maxIdx = lastindex(basis.list)
    i = firstindex(wVal) - 1
    pairs = map(pOrbs) do pOrb
        idx, maxIdx = updatePrimOrbCoreCache!(basis, paramCache, pOrb, maxIdx)
        idx => wVal[i += 1]
    end
    BasisIndexer(pairs, isRenormalized(orb)), maxIdx
end

function cacheIntComponents!(intCache::IntegralCache{T, D}, 
                             paramCache::DimSpanDataCacheBox{T}, 
                             orbs::FrameworkOrbSet{T, D}) where {T, D}
    checkEmptiness(orbs, :orbs)
    basis = intCache.basis
    offset = lastindex(basis.list)
    maxIdx = offset
    idxerList = map(orbs) do orb
        idxer, maxIdx = cacheIntComponentCore!(intCache, paramCache, orb)
        idxer
    end
    if maxIdx > offset
        updateIntCache!(intCache, offset+1)
    end
    idxerList
end


function initializeIntCache!(::Val{N}, op::DirectOperator, 
                             paramCache::DimSpanDataCacheBox{T}, 
                             orbs::FrameworkOrbSet{T, D}) where {N, T, D}
    checkEmptiness(orbs, :orbs)
    orbCoreType = mapreduce(typejoin, orbs, init=Union{}) do orb
        (eltype∘getOrbitalCores)(orb)
    end
    basisCache = PrimOrbCoreCache(T, Val(D), orbCoreType)
    intCache = IntegralCache(op, basisCache, initializeIntIndexer( T, Val(N) ))
    intCache, cacheIntComponents!(intCache, paramCache, orbs)
end

initializeOverlapCache!(paramCache::DimSpanDataCacheBox{T}, 
                        orbs::FrameworkOrbSet{T}) where {T} = 
initializeIntCache!(Val(1), Identity(), paramCache, orbs)

function updateIntCacheCore!(op::DirectOperator, idxer::OneBodyIntegralIndexer{T}, 
                             basis::Tuple{AbtOrbCoreInfoVec{T, D}}, 
                             offset::Tuple{Int}) where {T, D}
    pairs1, pairs2 = genOneBodyPrimIntPairs(op, basis, offset)
    foreach(p->setIntegralIndexer!(idxer, p), pairs1)
    foreach(p->setIntegralIndexer!(idxer, p), pairs2)
    idxer
end

function updateIntCacheCore!(op::DirectOperator, idxer::OneBodyIntegralIndexer{T}, 
                             basis::OneBodyIntOrbInfo{T, D}, 
                             offset::NTuple{2, Int}) where {T, D}
    pairs2 = genOneBodyPrimIntPairs(op, basis, offset)
    foreach(p->setIntegralIndexer!(idxer, p), pairs2)
    idxer
end

function updateIntCache!(cache::IntegralCache, startIdx::Int)
    op = cache.operator
    basis = cache.basis.list
    intIdxer = cache.data
    firstIdx = firstindex(basis)

    if startIdx == firstIdx
        updateIntCacheCore!(op, intIdxer, (basis,), (0,))
    else
        firstIdx < startIdx <= lastindex(basis) || throw(BoundsError(basis, startIdx))
        oldBasis = @view basis[begin:startIdx-1]
        newBasis = @view basis[startIdx:    end]
        updateIntCacheCore!(op, intIdxer, (newBasis,), (boundary,))
        updateIntCacheCore!(op, intIdxer, (oldBasis, newBasis,), (0, boundary))
    end

    cache
end

const OverlapCache{T, D} = IntegralCache{T, D, 1, Identity}


function getNormCoeffCore(cache::OverlapCache{T}, cacheIdxer::BasisIndexer{T}) where {T}
    if cacheIdxer.renormalize
        res = (first∘buildOneBodyEleCore)(cache, (cacheIdxer,))
        AbsSqrtInv(res)
    else
        one(T)
    end
end

function getNormCoeffCore(isNormalized::Bool, data::OrbCoreData{T}) where {T}
    if isNormalized
        res = (first∘computeOneBodyPrimIntVals)(Identity(), fill(data), firstindex(data))
        AbsSqrtInv(res)
    else
        one(T)
    end
end

# Chemist notation
function getNBodyScalarProd((a, b)::NTuple{2, T}) where {T<:Number}
    a' * b
end

# function getNBodyScalarProd((a, b, c, d)::NTuple{4, T}) where {T<:Number}
#     a' * b * c' * d
# end

getNBodyScalarProd(args::Vararg) = getNBodyScalarProd(args)


abstract type MarkedTensorIndex{N} <: Any end

abstract type MultiBodyTensorIndex{N} <: MarkedTensorIndex{N} end

struct OneBodyTensorIndex <: MultiBodyTensorIndex{2}
    index::NTuple{2, Int} # (i,j)
    hermiticity::Bool
end

struct TwoBodyTensorIndex <: MultiBodyTensorIndex{4}
    index::NTuple{2, OneBodyTensorIndex} # ((i,j),(k,l))
    hermiticity::Bool
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

struct SortedTensorIndex{N, M} <: MarkedTensorIndex{N}
    index::NTuple{N, Int}
    relation::NTuple{M, Bool}
    hermiticity::NTuple{M, Bool}
    permutation::NTuple{M, Bool}

    function SortedTensorIndex(idx::OneBodyTensorIndex)
        hermiticity = idx.hermiticity
        i, j = index = idx.index
        permutation = if hermiticity && i > j
            index = (j, i)
            true
        else
            false
        end
        new{2, 1}(index, (i==j,), (hermiticity,), (permutation,))
    end

    function SortedTensorIndex(idx::TwoBodyTensorIndex)
        subIdxL, subIdxR = subIndices = idx.index
        iL, iR = SortedTensorIndex.(subIndices)
        permutation = if idx.hermiticity && sum(subIdxL.index) > sum(subIdxR.index)
            iL, iR = iR, iL
            (first(iL.permutation), first(iR.permutation), true)
        else
            (first(iL.permutation), first(iR.permutation), false)
        end
        index = (iL.index..., iR.index...)
        relation = (first(iL.relation), first(iR.relation), subIdxL.index==subIdxR.index)
        hermiticity = (first(iL.hermiticity), first(iR.hermiticity), idx.hermiticity)
        new{4, 3}(index, relation, hermiticity, permutation)
    end
end


function buildOneBodyEleCore(intCache::IntegralCache{T, D, 1, F}, 
                             (intIdxer,)::Tuple{BasisIndexer{T}}) where {T, D, F}
    len = length(intIdxer.list)
    intValCache = intCache.data
    if len == 1
        idx = firstindex(intIdxer.list)
        getOneBodyIntValPairs(intValCache, intIdxer, (idx,))
    else
        mapreduce(+, 1:triMatEleNum(len)) do k
            j, i = convert1DidxTo2D(len, k)
            pairs = getOneBodyIntValPairs(intValCache, intIdxer, (i, j))
            prod.(pairs) |> sum
        end |> tuple
    end
end

function getOneBodyIntValPairs(cache::OneBodyIntegralIndexer, indexer::BasisIndexer, 
                               (i,)::Tuple{Int})
    iPtr, iWeight = indexer.list[i]
    getOneBodyIntValCore(cache, (iPtr,)) .* getNBodyScalarProd(iWeight, iWeight)
end

function getOneBodyIntValCore(cache::OneBodyIntegralIndexer, formattedIdx::NTuple{2, Int})
    getfield(cache, OneBodyIdxSymDict[false])[formattedIdx]
end

function getOneBodyIntValCore(cache::OneBodyIntegralIndexer, formattedIdx::Tuple{Int})
    getfield(cache, OneBodyIdxSymDict[true ])[formattedIdx]
end

function getOneBodyIntValPairs(cache::OneBodyIntegralIndexer, 
                               (idxer1, idxer2)::NTuple{2, BasisIndexer{T}}, 
                               (i, j)::NTuple{2, Int}) where {T}
    iPtr, iWeight = idxer1.list[i]
    jPtr, jWeight = idxer2.list[j]
    reverseBack = false

    index, weights = if iPtr == jPtr
        (iPtr,), (getNBodyScalarProd(iWeight, iWeight),)
    else
        temp = getNBodyScalarProd(iWeight, jWeight)
        if iPtr > jPtr
            reverseBack = true
            (jPtr, iPtr), (temp', temp)
            else
            (iPtr, jPtr), (temp, temp')
        end
    end

    res = getOneBodyIntValCore(cache, index) .=> weights
    reverseBack ? reverse(res) : res
end

function getOneBodyIntValPairs(cache::OneBodyIntegralIndexer, indexer::BasisIndexer, 
                               (i, j)::NTuple{2, Int})
    getOneBodyIntValPairs(cache, (indexer, indexer), (i, j))
end


function buildOneBodyEleCore(cache::IntegralCache{T, D, 1}, 
                             (idxer1, idxer2)::NTuple{2, BasisIndexer{T}}, 
                             ) where {T, D}
    intValCache = cache.data
    idxPairRange = Iterators.product(eachindex(idxer1.list), eachindex(idxer2.list))
    mapreduce(.+, idxPairRange, init=(zero(T), zero(T))) do (i, j)
        pairs = getOneBodyIntValPairs(intValCache, (idxer1, idxer2), (i, j))
        prod.(pairs)
    end
end

function extendOneBodyBasis((a,)::Tuple{Any})
    (a, a)
end

function extendOneBodyBasis(t::NTuple{2, Any})
    itself(t)
end

# function extendTwoBodyBasis(tpl::Tuple{Vararg{Any}}, indexer::NTuple{4, TetraLevel})
#     map(indexer) do idx
#         getindex(tpl, idx)
#     end
# end

function buildOneBodyEleTuple((iCache, iIdxer)::Tuple{IntegralCache{T, D, 1}, I}, 
                              (nCache, nIdxer)::Tuple{ OverlapCache{T, D},    I}) where 
                             {T, D, I<:OneTwoTpl{BasisIndexer{T}}}
    resInner = buildOneBodyEleCore(iCache, iIdxer)
    nCoeffs = getNormCoeffCore.(Ref(nCache), nIdxer) |> extendOneBodyBasis
    resInner .* getNBodyScalarProd(nCoeffs)
end

function buildOneBodyEleTuple(intConfig::Tuple{ OverlapCache{T, D}, 
                                               Tuple{BasisIndexer{T}} }) where {T, D}
    intCache, (idxer,) = intConfig
    if idxer.renormalize
        (one(T),)
    else
        buildOneBodyEleCore(intCache, (idxer,))
    end
end

function buildOneBodyEleTuple(intConfig::Tuple{ OverlapCache{T, D}, 
                                               NTuple{2, BasisIndexer{T}} }) where {T, D}
    buildOneBodyEleTuple(intConfig, intConfig)
end

function buildOneBodyTensor(intConfig::Tuple{IntegralCache{T, D, 1}, BasisIdxerVec{T}}, 
                            nlzConfig::Tuple{ OverlapCache{T, D},    BasisIdxerVec{T}}
                            ) where {T, D}
    iCache, iIdxers = intConfig
    nCache, nIdxers = nlzConfig
    nBasis = length(iIdxers)
    checkLength(nIdxers, Symbol("nlzConfig[2]"), nBasis, "`length(intConfig[2])`")
    res = ShapedMemory{T}(undef, (nBasis, nBasis))
    dh = Plus(firstindex(nIdxers) - firstindex(iIdxers))
    di, dj = Plus.(first.(axes(res)) .- firstindex(iIdxers))
    for j in eachindex(iIdxers), i in firstindex(iIdxers):j
        m = di(i)
        if i == j
            iConfigOrb = (iCache, (iIdxers[   i ],))
            nConfigOrb = (nCache, (nIdxers[dh(i)],))
            temp = buildOneBodyEleTuple(iConfigOrb, nConfigOrb)
            setTensorEntry!(res, temp, m)
        else
            n = dj(j)
            iConfigOrb = (iCache, (iIdxers[   i ], iIdxers[   j ]))
            nConfigOrb = (nCache, (nIdxers[dh(i)], nIdxers[dh(j)]))
            temp = buildOneBodyEleTuple(iConfigOrb, nConfigOrb)
            setTensorEntry!(res, temp, (m, n))
        end
    end
    res
end

function setTensorEntry!(tensor::AbstractMatrix{T}, val::NTuple{2, T}, 
                         (m, n)::NTuple{2, Int}) where {T}
    tensor[m, n] = first(val)
    tensor[n, m] = last(val)
end

function setTensorEntry!(tensor::AbstractArray{T, N}, val::Tuple{T}, 
                         m::Int) where {T, N}
    tensor[fill(m, N)...] = first(val)
end

function buildOneBodyTensor(intConfig::Tuple{OverlapCache{T, D}, BasisIdxerVec{T}}) where 
                           {T, D}
    iCache, iIdxers = intConfig
    nBasis = length(iIdxers)
    res = ShapedMemory{T}(undef, (nBasis, nBasis))
    di, dj = Plus.(first.(axes(res)) .- firstindex(iIdxers))
    for j in eachindex(iIdxers), i in firstindex(iIdxers):j
        m = di(i)
        if i == j
            temp = buildOneBodyEleTuple(( iCache, (iIdxers[i],) ))
            setTensorEntry!(res, temp, m)
        else
            n = dj(j)
            temp = buildOneBodyEleTuple(( iCache, (iIdxers[i], iIdxers[j]) ))
            setTensorEntry!(res, temp, (m, n))
        end
    end
    res
end

buildNBodyTensor(::Val{1}, args...) = buildOneBodyTensor(args...)


function extractFPrimOrbSetData!(cache::DimSpanDataCacheBox{T}, 
                                 pOrbs::FPrimOrbSet{T}) where {T}
    normCoeffs = Memory{T}(undef, checkEmptiness(pOrbs, :pOrbs))
    coreData = map(pOrbs, eachindex(normCoeffs)) do o, i
        d = genOrbCoreData!(cache, o)
        normCoeffs[i] = getNormCoeffCore(isRenormalized(o), d)
        d
    end
    coreData, normCoeffs
end

function computePrimIntTensor(cache::DimSpanDataCacheBox{T}, op::DirectOperator, 
                              pOrbClusters::NTuple{2, FPrimOrbSet{T, D}}) where {T, D}
    lens = length.(pOrbClusters)
    if any(i==0 for i in lens)
        ShapedMemory{T}(undef, lens)
    else
        data1, nCoeffs1 = extractFPrimOrbSetData!(cache, first(pOrbClusters))
        data2, nCoeffs2 = extractFPrimOrbSetData!(cache,  last(pOrbClusters))
        tensor = computePrimCoreIntTensor(op, (data1, data2))
        di, dj = Plus.(first.(axes(tensor)) .- firstindex(nCoeffs1)) #!! Need a function wrapper
        for j in eachindex(nCoeffs2), i in eachindex(nCoeffs1)
            tensor[di(i), dj(j)] *= nCoeffs1[i] * nCoeffs2[j]
        end
        tensor
    end
end

function cacheIntegrateCore!(intCache::IntegralCache{T, D, N}, 
                             paramCache::IntegralCache{T, D, N}, 
                             basisSet::FrameworkOrbSet{T, D}, 
                             ::Missing) where {T, D, N}
    intIdxers = cacheIntComponents!(intCache, paramCache, basisSet)
    if intCache isa OverlapCache{T, D}
        buildNBodyTensor(Val(N), (intCache, intIdxers))
    else
        normCache, normIdxers = initializeOverlapCache!(paramCache, basisSet)
        buildNBodyTensor(Val(N), (intCache, intIdxers), (normCache, normIdxers))
    end
end

function cacheIntegrateCore!(intCache::IntegralCache{T, D, N}, 
                             paramCache::IntegralCache{T, D, N}, 
                             basisSet::FrameworkOrbSet{T, D}, 
                             normCache::OverlapCache{T, D}) where {T, D, N}
    intIdxers = cacheIntComponents!(intCache, paramCache, basisSet)
    normIdxers = cacheIntComponents!(normCache, paramCache, basisSet)
    buildNBodyTensor(Val(N), (intCache, intIdxers), (normCache, normIdxers))
end


function cacheIntegrate!(intCache::IntegralCache{T, D}, basisSet::FrameworkOrbSet{T, D}; 
                         normCache::Union{OverlapCache{T, D}, Missing}=missing, 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where 
                        {T, D}
    cacheIntegrateCore!(intCache, paramCache, basisSet, normCache)
end

#! Think about if `::Val{N}` is a good trait switcher.
function integrateNBody(::Val{N}, op::DirectOperator, basisSet::FrameworkOrbSet{T, D}; 
                        paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where 
                       {N, T, D}
    intCache, intIdxers = initializeIntCache!(Val(N), op, paramCache, basisSet)
    normCache, normIdxers = initializeOverlapCache!(paramCache, basisSet)
    buildNBodyTensor(Val(N), (intCache, intIdxers), (normCache, normIdxers))
end

integrateNBody(::Val{1}, ::Identity, basisSet::FrameworkOrbSet{T, D}; 
               paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where {T, D} = 
buildNBodyTensor(Val(1), initializeOverlapCache!(paramCache, basisSet))

getOverlapsN(basisSet::FrameworkOrbSet{T}; 
             paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where {T} = 
integrateNBody(Val(1), Identity(), basisSet; paramCache)

function getOverlapN(bf1::FrameworkOrb{T, D}, bf2::FrameworkOrb{T, D}; 
                     paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                     lazyCompute::Bool=false) where {T, D}
    integrateNBody(Identity(), (bf1, bf2); paramCache, lazyCompute)
end


function cacheOrbWeight!(paramCache::DimSpanDataCacheBox{T}, orb::FCompOrb{T}) where {T}
    cacheParam!(paramCache, getOrbWeightCore(orb))
end

function cacheOrbWeight!(::DimSpanDataCacheBox{T}, ::FPrimOrb{T}) where {T}
    one(T)
end


function integrateNBody(::Val{1}, op::DirectOperator, 
                        (bf1, bf2)::NTuple{2, FrameworkOrb{T, D}}; 
                        paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                        lazyCompute::Bool=false) where {T, D}
    pOrbs1 = decomposeOrb(bf1)
    w1 = cacheOrbWeight!(paramCache, bf1)
    if bf1 === bf2
            w2 = w1
        tensor = integrateNBody(Val(1), op, pOrbs1; paramCache)
    else
        pOrbs2 = decomposeOrb(bf2)
            w2 = cacheOrbWeight!(paramCache, bf2)
        tensor = if lazyCompute
            pOrbs1 = Vector(pOrbs1)
            pOrbs2 = Vector(pOrbs2)
            transformation = (bf::FPrimOrb{T, D})->(markObj∘genOrbCoreData!)(paramCache, bf)
            pOrbsMutual = intersectMultisets!(pOrbs1, pOrbs2; transformation)
            block4 = computePrimIntTensor(paramCache, op, (pOrbs1, pOrbs2))
            if isempty(pOrbsMutual)
                block4
            else
                block1 = integrateNBody(Val(1), op, pOrbsMutual; paramCache)
                block2 = computePrimIntTensor(paramCache, op, (pOrbs1, pOrbsMutual))
                block3 = computePrimIntTensor(paramCache, op, (pOrbsMutual, pOrbs2))
                hvcat((2, 2), block1, block3, block2, block4)
            end
        else
            computePrimIntTensor(paramCache, op, (pOrbs1, pOrbs2))
        end
    end

    dot(w1, tensor, w2)
end

function getOverlapN(bf1::FrameworkOrb{T, D}, bf2::FrameworkOrb{T, D}; 
                     paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                     lazyCompute::Bool=false) where {T, D}
    integrateNBody(Identity(), (bf1, bf2); paramCache, lazyCompute)
end

function getOverlapN(orb1::OrbitalBasis{T, D}, orb2::OrbitalBasis{T, D}; 
                     paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                     lazyCompute::Bool=false) where {T, D}
    bf1 = FrameworkOrb(orb1)
    bf2 = orb1 === orb2 ? bf1 : FrameworkOrb(orb2)
    getOverlapN(bf1, bf2; paramCache, lazyCompute)
end