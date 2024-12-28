const OneTwoTpl{T} = Union{Tuple{T}, NTuple{2, T}}

const FrameworkOrbSet{T, D} = AbstractVector{<:FrameworkOrb{T, D}}

const OrbCoreIdxDict{T} = 
      Dict{Tuple{FieldMarker{:PrimitiveOrbCore, 1}, AbtVecOfAbtArr{T}}, Int}

const OrbCoreData{T, D, F<:PrimitiveOrbCore{T, D}, V<:AbtVecOfAbtArr{T}} = Tuple{F, V}

const OrbCoreInfoVec{T, D, F<:PrimitiveOrbCore{T, D}} = 
      Vector{ OrbCoreData{T, D, F, Vector{ ShapedMemory{T} }} }

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

function computeOneBodyInt(op::DirectOperator, (oData,)::Tuple{OrbCoreInfoVec{T, D}}, 
                           (indexOffset,)::Tuple{Int}=(0,)) where {T, D}
    iFirst = firstindex(oData)
    nOrbs = length(oData)
    dIdx = Plus(iFirst - 1)

    pairs1 = map(eachindex(oData)) do i
        iiVal = genPrimOneBodyIntVal(op, oData, i)
        (i+indexOffset,) => iiVal
    end
    pairs2 = map(1:triMatEleNum(nOrbs)) do l
        i, j = (sortTensorIndex∘convert1DidxTo2D)(nOrbs, l) .|> dIdx
        ijValPair = genPrimOneBodyIntVal(op, oData, (i, j))
        (i+indexOffset, j+indexOffset) => ijValPair
    end

    pairs1, pairs2
end

function genPrimOneBodyIntVal(op::DirectOperator, oData::OrbCoreInfoVec{T}, 
                              i::Int) where {T}
    orb, pars = oData[i]
    f = ReturnTyped(genPrimIntegrator(op, orb), T)
    (f(pars),)
end

function genPrimOneBodyIntValCore(op::DirectOperator, 
                                  (orb1, pars1)::OrbCoreData{T, D}, 
                                  (orb2, pars2)::OrbCoreData{T, D}) where {T, D}
    f = ReturnTyped(genPrimIntegrator(op, orb1, orb2), T)
    f(pars1, pars2)
end

# function genPrimOneBodyIntVal(::Val{false}, op::DirectOperator, 
#                               (oData1, oData2)::NTuple{2, OrbCoreInfoVec{T, D}}, 
#                               (i, j)::NTuple{2, Int}) where {T, D}
#     (genPrimOneBodyIntValCore(op, oData1[i], oData2[j]),)
# end

function genPrimOneBodyIntVal(op::DirectOperator, 
                              (oData1, oData2)::NTuple{2, OrbCoreInfoVec{T, D}}, 
                              (i, j)::NTuple{2, Int}) where {T, D}
    orbPars1 = oData1[i]
    orbPars2 = oData2[j]
    ijVal = genPrimOneBodyIntValCore(op, orbPars1, orbPars2)
    jiVal = if isHermitian(op, first(orbPars1), first(orbPars2))
        ijVal'
    else
        genPrimOneBodyIntValCore(op, orbPars2, orbPars1)
    end
    (ijVal, jiVal)
end

genPrimOneBodyIntVal(op::DirectOperator, oData::OrbCoreInfoVec, idx::NTuple{2, Int}) = 
genPrimOneBodyIntVal(op, (oData, oData), idx)

function computeOneBodyInt(op::DirectOperator, 
                           (oData1, oData2)::NTuple{2, OrbCoreInfoVec{T, D}}, 
                           (dIdx1, dIdx2)::NTuple{2, Int}) where {T, D}
    map(Iterators.product( eachindex(oData1), eachindex(oData2) )) do (i, j)
        ijValPair = genPrimOneBodyIntVal(op, (oData1, oData2), (i, j))
        (i+dIdx1, j+dIdx2) => ijValPair
    end |> vec
end

# function computeOneBodyInt(op::DirectOperator, 
#                            (oData1, oData2)::NTuple{2, OrbCoreInfoVec{T, D}}) where {T, D}
#     pairs1 = Memory{Pair{Tuple{Int}, T}}([])

#     pairs2 = map(Iterators.product( eachindex(oData1), eachindex(oData2) )) do (i, j)
#         ijVal = genPrimOneBodyIntVal(Val(false), op, (oData1, oData2), (i, j))
#         (i+dIdx1, j+dIdx2) => ijVal
#     end |> vec

#     pairs1, pairs2
# end


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


function tryAddPrimOrbCoreCache!(basisCache::PrimOrbCoreCache{T, D}, 
                                 paramCache::DimSpanDataCacheBox{T}, 
                                 orb::FPrimOrb{T, D}, 
                                 anchor::Int=firstindex(basisCache.list)) where {T, D}
    oCore = getOrbitalCores(orb)[]
    basis = basisCache.list
    pVal = cacheParam!(paramCache, orb.param, orb.pointer.scope)
    idx = get!(basisCache.dict, (markObj(oCore), pVal)) do
        push!(basis, (oCore, pVal))
        lastindex(basis)
    end
    idx, max(idx, anchor)
end

function getScalar(orb::FrameworkOrb)
    orb.core.f.apply.right
end

#!! Designed a better wrapper for the normalizer
#!! Designed a one(T) like function: SpecVal{T, F} -> F(T)

function isRenormalized(orb::ComposedOrb)
    orb.renormalize
end

function isRenormalized(orb::FrameworkOrb)
    (isRenormalizedCore∘getScalar)(orb)
end

function isRenormalizedCore(orb::Storage)
    false
end

function isRenormalizedCore(orb::ReturnTyped)
    true
end

function cacheIntComponents!(intCache::IntegralCache{T, D}, 
                             paramCache::DimSpanDataCacheBox{T}, 
                             orb::FPrimOrb{T, D}) where {T, D}
    idx, maxIdx = tryAddPrimOrbCoreCache!(intCache.basis, paramCache, orb)
    BasisIndexer(getMemory((idx => one(T),)), isRenormalized(orb)), maxIdx
end

function cacheIntComponents!(intCache::IntegralCache{T, D}, 
                             paramCache::DimSpanDataCacheBox{T}, 
                             orb::FCompOrb{T, D}) where {T, D}
    basis = intCache.basis
    pOrbs = decomposeOrb(orb)
    weightParam = getOrbitalWeight(orb)
    wVal = cacheParam!(paramCache, weightParam)
    maxIdx = lastindex(basis.list)
    i = firstindex(wVal) - 1
    pairs = map(pOrbs) do pOrb
        idx, maxIdx = tryAddPrimOrbCoreCache!(basis, paramCache, pOrb, maxIdx)
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
        idxer, maxIdx = cacheIntComponents!(intCache, paramCache, orb)
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
    intCache = IntegralCache(op, basisCache, initializeIntIndexer(T, Val(N)))
    intCache, cacheIntComponents!(intCache, paramCache, orbs)
end

initializeOverlapCache!(paramCache::DimSpanDataCacheBox{T}, 
                        orbs::FrameworkOrbSet{T}) where {T} = 
initializeIntCache!(Val(1), Identity(), paramCache, orbs)

function updateIntCacheCore!(op::DirectOperator, idxer::OneBodyIntegralIndexer{T}, 
                             basis::Tuple{OrbCoreInfoVec{T, D}}, 
                             offset::Tuple{Int}) where {T, D}
    pairs1, pairs2 = computeOneBodyInt(op, basis, offset)
    foreach(p->setIntegralIndexer!(idxer, p), pairs1)
    foreach(p->setIntegralIndexer!(idxer, p), pairs2)
    idxer
end

function updateIntCacheCore!(op::DirectOperator, idxer::OneBodyIntegralIndexer{T}, 
                             basis::NTuple{2, OrbCoreInfoVec{T, D}}, 
                             offset::NTuple{2, Int}) where {T, D}
    pairs2 = computeOneBodyInt(op, basis, offset)
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

function getNormCoeff(cache::OverlapCache{T}, 
                      cacheIdxer::NonEmptyTuple{BasisIndexer{T}}) where {T}
    map(cacheIdxer) do idxer
        getNormCoeffCore(cache, idxer)
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

function buildOneBodyEleTuple(intConfig::Tuple{IntegralCache{T, D, 1}, I}, 
                              nlzConfig::Tuple{ OverlapCache{T, D},    I}) where 
                             {T, D, I<:OneTwoTpl{BasisIndexer{T}}}
    resInner = buildOneBodyEleCore(intConfig...)
    nCoeff = getNormCoeff(nlzConfig...) |> extendOneBodyBasis |> getNBodyScalarProd
    resInner .* nCoeff
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

# function getOverlapN(bf1::FrameworkOrb{T, D}, bf2::FrameworkOrb{T, D}; 
#                      paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where {T, D}
#     integrateNBody(Identity(), bf1, bf2; paramCache)
# end

# function integrateNBody(op::DirectOperator, 
#                         bf1::FrameworkOrb{T, D}, bf2::FrameworkOrb{T, D}; 
#                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where 
#                        {T, D}
#     if bf1 === bf2
        
#     else

#     end
# end

#!! Add element-mise integral interfaces