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


struct OneBodyFullCoreIntegrals{T<:Number, D} <: IntegralData{T, OneBodyIntegral{D}}
    aa::Dict{ Tuple{   Int},  Tuple{   T}}
    ab::Dict{NTuple{2, Int}, NTuple{2, T}}

    OneBodyFullCoreIntegrals(::Type{T}, ::Val{D}) where {T, D} = 
    new{T, D}(Dict{ Tuple{Int},  Tuple{T}}(), Dict{NTuple{2, Int}, NTuple{2, T}}())
end

# struct OneBodySymmetricDiffData{T<:Number} <: IntegralData{T, OneBodyIntegral}
#     ab::Dict{NTuple{2, Int}, T}
# end


struct PrimOrbCoreIntegralCache{T, D, F<:DirectOperator, 
                                I<:IntegralData{T, <:MultiBodyIntegral{D}}, 
                                B<:PrimitiveOrbCore{T, D}} <: SpatialIntegralCache{T, D}
    operator::F
    data::I
    basis::PrimOrbCoreCache{T, D, B}
end

const OneBodyCoreIntCache{T, D, F<:DirectOperator, I<:IntegralData{T, OneBodyIntegral{D}}, 
                          B<:PrimitiveOrbCore{T, D}} = 
      PrimOrbCoreIntegralCache{T, D, F, I, B}

const OverlapCoreCache{T, D, I<:IntegralData{T, OneBodyIntegral{D}}, 
                       B<:PrimitiveOrbCore{T, D}} = 
      OneBodyCoreIntCache{T, D, Identity, I, B}


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

buildOneBodyCoreIntegrator(op::MonomialMul{T, D}, 
                           orbs::N12Tuple{PrimGTOcore{T, D}}) where {T, D} = 
genGTOrbMultiMomentFunc(op, orbs)

const OneBodyCoreIntData{T, D} = Tuple{DirectOperator, Vararg{PrimitiveOrbCore{T, D}, 2}}

const OneBodyCICacheDict{T, D} = 
      Dict{Type{<:OneBodyCoreIntData{T, D}}, OneBodyIntProcessCache{T, D}}
#! Consider a new type union when two-body computation cache is needed
const IntCompCacheDict{T, D} = Union{OneBodyCICacheDict{T, D}, TypedEmptyDict{Val{D}, T}}

IntCompCacheDict{T, D}() where {T, D} = TypedEmptyDict{Val{D}, T}()

IntCompCacheDict(::Type{T}, ::Type{OneBodyIntegral{D}}) where {T, D} = 
OneBodyCICacheDict{T, D}()

IntCompCacheDict(::Type{T}, ::Type{TwoBodyIntegral{D}}) where {T, D} = 
TypedEmptyDict{Val{D}, T}()

IntCompCacheDict(::Type{<:IntegralData{T, S}}) where {T, S<:MultiBodyIntegral} = 
IntCompCacheDict(T, S)

function prepareOneBodyIntCompCache!(cacheDict::OneBodyCICacheDict{T, D}, 
                                     op::DirectOperator, 
                                     orbs::NTuple{2, PrimGTOcore{T, D}}) where {T, D}
    get!(cacheDict, typeof( (op, orbs...) )) do
        genGTOrbIntCompCache(op, orbs)
    end
end

function prepareOneBodyIntCompCache!(::IntCompCacheDict{T, D}, ::DirectOperator, 
                                     ::N12Tuple{PrimitiveOrbCore{T, D}}) where {T, D}
    NullCache{T}()
end

# adjustOneBodyCoreIntConfig


function lazyEvalCoreIntegral!(cache::IntegralProcessCache{T}, 
                               integrator::OrbitalIntegrator{T}, 
                               paramData::N12Tuple{FilteredVecOfArr{T}}) where {T}
    integrator(paramData...; cache)::T
end

function lazyEvalCoreIntegral!(::NullCache{T}, integrator::OrbitalIntegrator{T}, 
                               paramData::N12Tuple{FilteredVecOfArr{T}}) where {T}
    integrator(paramData...)::T
end


function evalOneBodyPrimCoreIntegral(op::DirectOperator, 
                                     data::N12Tuple{OrbCoreData{T, D}}; 
                                     computeCache::IntCompCacheDict{T, D}=
                                                   IntCompCacheDict{T, D}()) where {T, D}
    orbData = first.(data)
    parData =  last.(data)
    fCore = buildOneBodyCoreIntegrator(op, orbData)
    intCompCache = prepareOneBodyIntCompCache!(computeCache, op, orbData)
    lazyEvalCoreIntegral!(intCompCache, fCore, parData)
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
                                    (oData,)::Tuple{OrbCoreDataSeq{T, D}}, 
                                    (oneBasedIdx,)::Tuple{Int}; 
                                    computeCache::IntCompCacheDict{T, D}=
                                                  IntCompCacheDict{T, D}()) where {T, D}
    iiVal = evalOneBodyPrimCoreIntegral(op, (oData[begin+oneBasedIdx-1],); computeCache)
    (iiVal,)
end

function genOneBodyIntDataPairsCore(op::DirectOperator, 
                                    oDataPair::NTuple{2, OrbCoreDataSeq{T, D}}, 
                                    oneBasedIdxPair::NTuple{2, Int}; 
                                    computeCache::IntCompCacheDict{T, D}=
                                                  IntCompCacheDict{T, D}()) where {T, D}
    orbPars1, orbPars2 = map(oDataPair, oneBasedIdxPair) do data, idx
        data[begin+idx-1]
    end
    ijVal = evalOneBodyPrimCoreIntegral(op, (orbPars1, orbPars2); computeCache)
    jiVal = if isHermitian(first(orbPars1), op, first(orbPars2))
        ijVal'
    else
        evalOneBodyPrimCoreIntegral(op, (orbPars2, orbPars1); computeCache)
    end
    (ijVal, jiVal)
end


function genOneBodyIntDataPairs(op::DirectOperator, 
                                (oData,)::Tuple{OrbCoreDataSeq{T, D}}, 
                                (indexOffset,)::Tuple{Int}=(0,); 
                                computeCache::IntCompCacheDict{T, D}=
                                              IntCompCacheDict{T, D}()) where {T, D}
    nOrbs = length(oData)
    offset = indexOffset + firstindex(oData) - 1

    pairs1 = map(1:nOrbs) do i
        iiVal = genOneBodyIntDataPairsCore(op, (oData,), (i,); computeCache)
        (i + offset,) => iiVal
    end

    pairs2 = map(1:triMatEleNum(nOrbs-1)) do l
        n, m = convert1DidxTo2D(nOrbs-1, l)
        ijPair = sortTensorIndex((m, n+1))
        ijValPair = genOneBodyIntDataPairsCore(op, (oData, oData), ijPair; computeCache)
        (ijPair .+ offset) => ijValPair
    end

    pairs1, pairs2
end

function genOneBodyIntDataPairs(op::DirectOperator, 
                                oDataPair::NTuple{2, OrbCoreDataSeq{T, D}}, 
                                indexOffsets::NTuple{2, Int}; 
                                computeCache::IntCompCacheDict{T, D}=
                                              IntCompCacheDict{T, D}()) where {T, D}
    mnOrbs = length.(oDataPair)
    offsets = indexOffsets .+ firstindex.(oDataPair) .- 1

    map(Iterators.product( Base.OneTo.(mnOrbs)... )) do mnIdx
        idxPairOld = mnIdx .+ offsets
        idxPairNew = sortTensorIndex(idxPairOld)
        ijPair = ifelse(idxPairNew == idxPairOld, mnIdx, reverse(mnIdx))
        ijValPair = genOneBodyIntDataPairsCore(op, oDataPair, ijPair; computeCache)
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
                                         (oData,)::Tuple{OrbCoreDataSeq{T, D}}; 
                                         computeCache::IntCompCacheDict{T, D}=
                                                       IntCompCacheDict{T, D}()
                                         ) where {T, D}
    nOrbs = length(oData)
    res = ShapedMemory{T}(undef, (nOrbs, nOrbs))

    for i in 1:nOrbs
        temp = genOneBodyIntDataPairsCore(op, (oData,), (i,); computeCache)
        setTensorEntries!(res, temp, (i,))
    end

    for l in 1:triMatEleNum(nOrbs-1)
        n, m = convert1DidxTo2D(mBasis-1, l)
        temp = genOneBodyIntDataPairsCore(op, (oData, oData), (m, n+1); computeCache)
        setTensorEntries!(res, temp, (m, n+1))
    end
    res
end

function computeOneBodyPrimCoreIntTensor(op::DirectOperator, 
                                         oDataPair::NTuple{2, OrbCoreDataSeq{T, D}}; 
                                         computeCache::IntCompCacheDict{T, D}=
                                                       IntCompCacheDict{T, D}()
                                         ) where {T, D}
    oData1, oData2 = oDataPair
    len1, len2 = length.(oDataPair)
    res = ShapedMemory{T}(undef, (len1, len2))
    for j in 1:len2, i in 1:len1
        oCorePair = (oData1[begin+i-1], oData2[begin+j-1])
        ijVal = evalOneBodyPrimCoreIntegral(op, oCorePair; computeCache)
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


function cachePrimCoreIntegrals!(intCache::PrimOrbCoreIntegralCache{T, D, F, I}, 
                                 paramCache::DimSpanDataCacheBox{T}, 
                                 orbMarkerCache::OrbCoreMarkerDict, 
                                 orbs::OrbitalCollection{T, D}) where {T, D, 
                                 F<:DirectOperator, 
                                 I<:IntegralData{ T, <:MultiBodyIntegral{D} }}
    orbCache = intCache.basis
    computeCache = IntCompCacheDict(I)
    oldMaxIdx = lastindex(orbCache.list)
    orbIdxList = indexCacheOrbData!(orbCache, paramCache, orbMarkerCache, orbs)
    updatePrimCoreIntCache!(intCache, oldMaxIdx+1; computeCache)
    orbIdxList
end

function cachePrimCoreIntegrals!(intCache::PrimOrbCoreIntegralCache{T, D, F, I}, 
                                 paramCache::DimSpanDataCacheBox{T}, 
                                 orbMarkerCache::OrbCoreMarkerDict, 
                                 orb::FrameworkOrb{T, D}) where {T, D, F<:DirectOperator, 
                                 I<:IntegralData{ T, <:MultiBodyIntegral{D} }}
    orbCache = intCache.basis
    computeCache = IntCompCacheDict(I)
    oldMaxIdx = lastindex(orbCache.list)
    orbIdxList = indexCacheOrbData!(orbCache, paramCache, orbMarkerCache, orb)
    updatePrimCoreIntCache!(intCache, oldMaxIdx+1; computeCache)
    orbIdxList
end

function cachePrimCoreIntegrals!(targetIntCache::PrimOrbCoreIntegralCache{T, D, F, I}, 
                                 sourceIntCache::PrimOrbCoreIntegralCache{T, D, F, I}, 
                                 orbMarkerCache::OrbCoreMarkerDict, 
                                 sourceOrbList::BasisIndexList) where {T, D, 
                                 F<:DirectOperator, 
                                 I<:IntegralData{ T, <:MultiBodyIntegral{D} }}
    tOrbCache = targetIntCache.basis
    computeCache = IntCompCacheDict(I)
    oldMaxIdx = lastindex(tOrbCache.list)
    sOrbCache = sourceIntCache.basis
    orbIdxList = indexCacheOrbData!(tOrbCache, sOrbCache, orbMarkerCache, sourceOrbList)
    updatePrimCoreIntCache!(targetIntCache, oldMaxIdx+1; computeCache)
    orbIdxList
end


function updateIntCacheCore!(op::DirectOperator, ints::OneBodyFullCoreIntegrals{T, D}, 
                             basis::Tuple{OrbCoreDataSeq{T, D}}, 
                             offset::Tuple{Int}; 
                             computeCache::IntCompCacheDict{T, D}=
                                           IntCompCacheDict{T, D}()) where {T, D}
    pairs1, pairs2 = genOneBodyIntDataPairs(op, basis, offset; computeCache)
    foreach(p->setIntegralData!(ints, p), pairs1)
    foreach(p->setIntegralData!(ints, p), pairs2)
    ints
end

function updateIntCacheCore!(op::DirectOperator, ints::OneBodyFullCoreIntegrals{T, D}, 
                             basis::NTuple{2, OrbCoreDataSeq{T, D}}, 
                             offset::NTuple{2, Int}; 
                             computeCache::IntCompCacheDict{T, D}=
                                           IntCompCacheDict{T, D}()) where {T, D}
    pairs2 = genOneBodyIntDataPairs(op, basis, offset; computeCache)
    foreach(p->setIntegralData!(ints, p), pairs2)
    ints
end


function updatePrimCoreIntCache!(cache::PrimOrbCoreIntegralCache{T, D}, startIdx::Int; 
                                 computeCache::IntCompCacheDict{T, D}=
                                               IntCompCacheDict{T, D}()) where {T, D}
    op = cache.operator
    basis = cache.basis.list
    ints = cache.data
    firstIdx = firstindex(basis)

    if startIdx == firstIdx
        updateIntCacheCore!(op, ints, (basis,), (0,); computeCache)
    elseif firstIdx < startIdx <= lastindex(basis)
        boundary = startIdx - 1
        oldBasis = @view basis[begin:boundary]
        newBasis = @view basis[startIdx:  end]
        updateIntCacheCore!(op, ints, (newBasis,), (boundary,); computeCache)
        updateIntCacheCore!(op, ints, (oldBasis, newBasis,), (0, boundary); computeCache)
    end

    cache
end


function decodePrimCoreInt(cache::OneBodyFullCoreIntegrals{T}, ptrPair::NTuple{2, Int}, 
                           (coeff1, coeff2)::NTuple{2, T}=( one(T), one(T) )) where {T}
    coeffProd = coeff1' * coeff2
    ptrPairNew = sortTensorIndex(ptrPair)
    data = getPrimCoreIntData(cache, ptrPairNew)
    (ptrPairNew == ptrPair ? data : reverse(data)) .* (coeffProd, coeffProd')
end

function decodePrimCoreInt(cache::OneBodyFullCoreIntegrals{T}, ptr::Tuple{Int}, 
                           (coeff1,)::Tuple{T}=( one(T), )) where {T}
    getPrimCoreIntData(cache, ptr) .* coeff1' .* coeff1
end


function buildPrimOrbWeight(normCache::OverlapCoreCache{T, D}, orb::EvalPrimOrb{T, D}, 
                            idx::Int) where {T, D}
    if isRenormalized(orb)
        decodePrimCoreInt(normCache.data, (idx,)) |> first |> AbsSqrtInv
    else
        one(T)
    end
end


function buildNormalizedCompOrbWeight!(weight::AbstractVector{T}, 
                                       normCache::OverlapCoreCache{T, D}, 
                                       orb::FCompOrb{T, D}, 
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
                         normCache::OverlapCoreCache{T, D}, orb::FrameworkOrb{T, D}, 
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
                                normCache::OverlapCoreCache{T, D}, 
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
                                normCache::OverlapCoreCache{T, D}, 
                                orb::FrameworkOrb{T, D}, 
                                normIdxList::BasisIndexList, 
                                intIdxList::BasisIndexList=normIdxList) where {T, D}
    orbWeight = buildOrbWeight!(paramCache, normCache, orb, normIdxList.index)
    buildIndexedOrbWeightsCore!(intIdxList.index, orbWeight)
end


function prepareIntegralConfig!(intCache::PrimOrbCoreIntegralCache{T, D}, 
                                normCache::OverlapCoreCache{T, D}, 
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


function buildIntegralEntries(intCache::OneBodyCoreIntCache{T}, 
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

function buildIntegralEntries(intCache::OneBodyCoreIntCache{T}, 
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


function buildIntegralTensor(intCache::OneBodyCoreIntCache{T}, 
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


function initializeIntegralCache!(::OneBodyIntegral{D}, op::DirectOperator, 
                                  paramCache::DimSpanDataCacheBox{T}, 
                                  orbInput::OrbitalInput{T, D}) where {T, D}
    coreType = getPrimOrbCoreTypeUnion(orbInput)
    orbCache = PrimOrbCoreCache(T, Val(D), coreType)
    PrimOrbCoreIntegralCache(op, OneBodyFullCoreIntegrals(T, Val(D)), orbCache)
end

function initializeOverlapCache!(paramCache::DimSpanDataCacheBox{T}, 
                                 orbInput::OrbitalInput{T, D}) where {T, D}
    initializeIntegralCache!(OneBodyIntegral{D}(), Identity(), paramCache, orbInput)
end


function initializeOneBodyCachePair!(op::F, paramCache::DimSpanDataCacheBox{T}, 
                                     orbInput::OrbitalInput{T, D}) where 
                                    {F<:DirectOperator, T, D}
    intCache = initializeIntegralCache!(OneBodyIntegral{D}(), op, paramCache, orbInput)
    normCache = F <: Identity ? intCache : initializeOverlapCache!(paramCache, orbInput)
    intCache, normCache
end


function computeIntTensor!(intCache::PrimOrbCoreIntegralCache{T, D}, 
                           normCache::OverlapCoreCache{T, D}, 
                           basisSet::AbstractVector{<:FrameworkOrb{T, D}}; 
                           paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                           basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict()) where 
                          {T, D}
    ws = prepareIntegralConfig!(intCache, normCache, paramCache, basisMarkerCache, basisSet)
    buildIntegralTensor(intCache, ws)
end

function computeIntTensor(::OneBodyIntegral{D}, op::F, 
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


function decomposeOrb!(normCache::OverlapCoreCache{T, D}, orb::FrameworkOrb{T, D}; 
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


function computeIntegral!(intCache::PrimOrbCoreIntegralCache{T, D}, 
                          normCache::OverlapCoreCache{T, D}, 
                          bfs::NonEmptyTuple{FrameworkOrb{T, D}}; 
                          paramCache::DimSpanDataCacheBox{T}, 
                          basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict()) where 
                         {T, D}
    ws = prepareIntegralConfig!(intCache, normCache, paramCache, basisMarkerCache, bfs)
    buildIntegralEntries(intCache, ws) |> first
end

function computeIntegral(::OneBodyIntegral{D}, op::DirectOperator, 
                         (bf1,)::Tuple{FrameworkOrb{T, D}}; 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                         basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict(), 
                         lazyCompute::Bool=false) where {T, D}
    if lazyCompute
        iCache, nCache = initializeOneBodyCachePair!(op, paramCache, bf1)
        computeIntegral!(iCache, nCache, (bf1,); paramCache, basisMarkerCache)
    else
        coreData, w = decomposeOrb(bf1; paramCache, markerCache=basisMarkerCache)
        computeCache = IntCompCacheDict(T, OneBodyIntegral{D})
        tensor = computeOneBodyPrimCoreIntTensor(op, (coreData,); computeCache)
        dot(w, tensor, w)
    end
end

function computeIntegral(::OneBodyIntegral{D}, op::DirectOperator, 
                         (bf1, bf2)::NTuple{2, FrameworkOrb{T, D}}; 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                         basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict(), 
                         lazyCompute::Bool=false) where {T, D}
    oData1, w1 = decomposeOrb(bf1; paramCache, markerCache=basisMarkerCache)
    oData2, w2 = decomposeOrb(bf2; paramCache, markerCache=basisMarkerCache)
    computeCache = IntCompCacheDict(T, OneBodyIntegral{D})

    tensor = if lazyCompute
        oData1 = Vector(oData1)
        oData2 = Vector(oData2)
        transformation = (b::PrimitiveOrbCore{T, D})->lazyMarkObj!(basisMarkerCache, b)
        coreDataM = intersectMultisets!(oData1, oData2; transformation)
        block4 = computeOneBodyPrimCoreIntTensor(op, (oData1, oData2); computeCache)
        if isempty(coreDataM)
            block4
        else
            block1 = computeOneBodyPrimCoreIntTensor(op, (coreDataM,); computeCache)
            block2 = computeOneBodyPrimCoreIntTensor(op, (oData1, coreDataM); computeCache)
            block3 = computeOneBodyPrimCoreIntTensor(op, (coreDataM, oData2); computeCache)
            hvcat((2, 2), block1, block3, block2, block4)
        end
    else
        computeOneBodyPrimCoreIntTensor(op, (oData1, oData2); computeCache)
    end
    dot(w1, tensor, w2)
end

function computeIntegral(::OneBodyIntegral{D}, ::Identity, 
                         bfPair::NTuple{2, FrameworkOrb{T, D}}; 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                         basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict(), 
                         lazyCompute::Bool=false) where {T, D}
    bf1, bf2 = bfPair
    if lazyCompute
        if bf1 === bf2
            computeIntegral(OneBodyIntegral{D}(), Identity(), (bf1,); paramCache, 
                            basisMarkerCache, lazyCompute)
        else
            normCache = initializeOverlapCache!(paramCache, bfPair)
            computeIntegral!(normCache, normCache, bfPair; paramCache, basisMarkerCache)
        end
    else
        oData1, w1 = decomposeOrb(bf1; paramCache, markerCache=basisMarkerCache)
        oData2, w2 = decomposeOrb(bf2; paramCache, markerCache=basisMarkerCache)
        computeCache = IntCompCacheDict(T, OneBodyIntegral{D})
        tensor = computeOneBodyPrimCoreIntTensor(Identity(), (oData1, oData2); computeCache)
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