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


struct PrimOrbCoreIntegralCache{T, D, S<:MultiBodyIntegral{D}, F<:DirectOperator, 
                                I<:IntegralData{T, S}, B<:PrimitiveOrbCore{T, D}
                                } <: SpatialProcessCache{T, D}
    operator::F
    data::I
    basis::PrimOrbCoreCache{T, D, B}
end

const POrb1BCoreICache{T, D, F<:DirectOperator, I<:IntegralData{T, OneBodyIntegral{D}}, 
                       B<:PrimitiveOrbCore{T, D}} = 
      PrimOrbCoreIntegralCache{T, D, OneBodyIntegral{D}, F, I, B}

const OverlapCoreCache{T, D, I<:IntegralData{T, OneBodyIntegral{D}}, 
                       B<:PrimitiveOrbCore{T, D}} = 
      POrb1BCoreICache{T, D, Identity, I, B}


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

const OrbCoreIntLayoutAllOrbs{T, D, N} = NonEmptyTuple{PrimitiveOrbCore{T, D}, N}
const OrbCoreIntLayoutOrbBar1{T, D} = 
      Tuple{PrimitiveOrbCore{T, D}, Bar, PrimitiveOrbCore{T, D}}
const OrbCoreIntLayoutOrbBar2{T, D} = 
      Tuple{PrimitiveOrbCore{T, D}, PrimitiveOrbCore{T, D}, Bar}
const OrbCoreIntLayoutOrbBar3{T, D} = 
      Tuple{PrimitiveOrbCore{T, D}, Bar, PrimitiveOrbCore{T, D}, PrimitiveOrbCore{T, D}}
const OrbCoreIntLayoutOrbBar4{T, D} = 
      Tuple{PrimitiveOrbCore{T, D}, PrimitiveOrbCore{T, D}, Bar, PrimitiveOrbCore{T, D}}

const OneBodyOrbCoreIntLayout{T, D} = Union{
    OrbCoreIntLayoutAllOrbs{T, D, 0}, OrbCoreIntLayoutAllOrbs{T, D, 1}
}

const TwoBodyOrbCoreIntLayout{T, D} = Union{
    OrbCoreIntLayoutAllOrbs{T, D, 0}, OrbCoreIntLayoutAllOrbs{T, D, 3}, 
    OrbCoreIntLayoutOrbBar1{T, D},    OrbCoreIntLayoutOrbBar2{T, D}, 
    OrbCoreIntLayoutOrbBar3{T, D},    OrbCoreIntLayoutOrbBar4{T, D}, 
}

const OrbCoreIntLayout{T, D} = 
      Union{OneBodyOrbCoreIntLayout{T, D}, TwoBodyOrbCoreIntLayout{T, D}}


struct OrbCoreIntConfig{T, D, S<:MultiBodyIntegral{D}, 
                        P<:OrbCoreIntLayout{T, D}} <: StructuredType

    function OrbCoreIntConfig(::OneBodyIntegral{D}, ::P) where 
                             {D, T, P<:OneBodyOrbCoreIntLayout{T, D}}
        new{T, D, OneBodyIntegral{D}, P}()
    end

    function OrbCoreIntConfig(::TwoBodyIntegral{D}, ::P) where 
                             {D, T, P<:TwoBodyOrbCoreIntLayout{T, D}}
        new{T, D, TwoBodyIntegral{D}, P}()
    end
end

const OneBodyOrbCoreIntConfig{T, D, P<:OneBodyOrbCoreIntLayout{T, D}} = 
      OrbCoreIntConfig{T, D, OneBodyIntegral{D}, P}

const TwoBodyOrbCoreIntConfig{T, D, P<:TwoBodyOrbCoreIntLayout{T, D}} = 
      OrbCoreIntConfig{T, D, TwoBodyIntegral{D}, P}


abstract type OrbIntegralCompCacheBox{T, D, S<:MultiBodyIntegral{D}} <: QueryBox{T} end


struct OrbIntCompCacheBox{T, D, S<:MultiBodyIntegral{D}} <: OrbIntegralCompCacheBox{T, D, S}
    dict::Dict{OrbCoreIntConfig{T, D, S}, OrbIntegralComputeCache{T, D, S}}

    function OrbIntCompCacheBox(::S, ::Type{T}) where {D, S<:MultiBodyIntegral{D}, T}
        dict = Dict{OrbCoreIntConfig{T, D, S}, OrbIntegralComputeCache{T, D, S}}()
        new{T, D, S}(dict)
    end
end


struct OrbIntNullCacheBox{T, D, S<:MultiBodyIntegral{D}} <: OrbIntegralCompCacheBox{T, D, S}

    OrbIntCompCacheBox(::S, ::Type{T}) where {D, S<:MultiBodyIntegral{D}, T} = 
    new{T, D, S}()
end


abstract type CachedComputeSpatialIntegral{T, D} <: StatefulFunction{T} end


struct CachedComputeOrbCoreIntegral{T, D, F<:DirectOperator, 
                                    C<:OrbIntegralCompCacheBox{T, D}
                                    } <: CachedComputeSpatialIntegral{T, D}
    operator::F
    cache::C

    function CachedComputeOrbCoreIntegral(::Val{true},  ::S, operator::F, ::Type{T}) where 
                                         {F<:DirectOperator, T, D, S<:MultiBodyIntegral{D}}
        new{T, D, F, OrbIntCompCacheBox{T, D, S}}(operator, OrbIntCompCacheBox(S(), T))
    end

    function CachedComputeOrbCoreIntegral(::Val{false}, ::S, operator::F, ::Type{T}) where 
                                         {F<:DirectOperator, T, D, S<:MultiBodyIntegral{D}}
        new{T, D, F, OrbIntNullCacheBox{T, D, S}}(operator, OrbIntNullCacheBox(S(), T))
    end
end

const OrbCoreIntegralComputeConfig{T, D, S<:MultiBodyIntegral{D}, F<:DirectOperator, 
                                   C<:OrbIntegralCompCacheBox{T, D, S}} = 
      CachedComputeOrbCoreIntegral{T, D, F, C}

const Orb1BCCIntComputeConfigUnion{T, D, F<:DirectOperator} = 
      OrbCoreIntegralComputeConfig{T, D, OneBodyIntegral{D}, F}

const Orb2BCCIntComputeConfigUnion{T, D, F<:DirectOperator} = 
      OrbCoreIntegralComputeConfig{T, D, TwoBodyIntegral{D}, F}

const DirectComputeOrbCoreIntegral{T, D, S<:MultiBodyIntegral{D}, F<:DirectOperator, 
                                   C<:OrbIntNullCacheBox{T, D, S}} = 
      CachedComputeOrbCoreIntegral{T, D, F, C}

const ReusedComputeOrbCoreIntegral{T, D, S<:MultiBodyIntegral{D}, F<:DirectOperator, 
                                   C<:OrbIntCompCacheBox{T, D, S}} = 
      CachedComputeOrbCoreIntegral{T, D, F, C}


function buildCoreIntegrator(::OneBodyIntegral{D}, op::DirectOperator, 
                             orbs::N12Tuple{PrimitiveOrbCore{T, D}}
                             ) where {T, D}
    OneBodyNumIntegrate(op, orbs)
end

function buildCoreIntegrator(::OneBodyIntegral{D}, ::Identity, 
                             orbs::N12Tuple{PrimGTOcore{T, D}}) where {T, D}
    genGTOrbOverlapFunc(orbs)
end

function buildCoreIntegrator(::OneBodyIntegral{D}, op::MonomialMul{T, D}, 
                             orbs::N12Tuple{PrimGTOcore{T, D}}) where {T, D}
    genGTOrbMultiMomentFunc(op, orbs)
end


function (f::DirectComputeOrbCoreIntegral{T, D, S}
          )(data::N12Tuple{OrbCoreData{T, D}}) where {T, D, S<:MultiBodyIntegral{D}}
    orbData = first.(data)
    parData =  last.(data)
    fCore = buildCoreIntegrator(S(), f.operator, orbData)
    fCore(parData...)
end

const OrbCoreIntegralComputeCacheTypes{T, D} = Union{
    OrbCoreIntConfig{T, D, <:MultiBodyIntegral{D}, TetraTupleUnion{PrimGTOcore{T, D}}}
}

function prepareOneBodyIntCompCache!(f::ReusedComputeOrbCoreIntegral{T, D, S}, 
                                     orbs::TetraTupleUnion{PrimitiveOrbCore{T, D}}) where 
                                    {T, D, S<:MultiBodyIntegral{D}}
    key = OrbCoreIntConfig(S(), orbs)
    if key isa OrbCoreIntegralComputeCacheTypes
        get!(f.cache.dict, key) do
            genGTOrbIntCompCache(f.operator, orbs)
        end
    else
        nothing
    end
end

prepareOneBodyIntCompCache!(::ReusedComputeOrbCoreIntegral{T, D, OneBodyIntegral{D}}, 
                            ::Tuple{PrimGTOcore{T, D}}) where {T, D} = 
nothing

function (f::OrbCoreIntegralComputeConfig{T, D, S}
          )(data::TetraTupleUnion{OrbCoreData{T, D}}) where {T, D, S<:MultiBodyIntegral{D}}
    orbData = first.(data)
    parData =  last.(data)
    fCore = buildCoreIntegrator(S(), f.operator, orbData)
    intSectorCache = prepareOneBodyIntCompCache!(f, orbData)
    if intSectorCache === nothing
        fCore(parData...)
    else
        fCore(parData..., cache!Self=intSectorCache)
    end
end


# One-Body (i|O|j) hermiticity across O
getHermiticity(::PrimitiveOrbCore{T, D}, ::DirectOperator, 
               ::PrimitiveOrbCore{T, D}) where {T, D} = 
false

getHermiticity(::PrimitiveOrbCore{T, D}, ::Identity, 
               ::PrimitiveOrbCore{T, D}) where {T, D} = 
true

# Two-Body (ii|O|jj) (ii|O|jk) (ij|O|kk) (ij|O|kl) hermiticity across O
getHermiticity(::N12Tuple{PrimitiveOrbCore{T, D}}, ::Identity, 
               ::N12Tuple{PrimitiveOrbCore{T, D}}) where {T, D} = 
true


function genCoreIntTuple(integrator::Orb1BCCIntComputeConfigUnion{T, D}, 
                         oDataTuple::Tuple{OrbCoreData{T, D}}) where {T, D}
    (integrator(oDataTuple),)
end

function genCoreIntTuple(integrator::Orb1BCCIntComputeConfigUnion{T, D}, 
                         oDataTuple::NTuple{2, OrbCoreData{T, D}}) where {T, D}
    d1, d2 = oDataTuple
    ijVal = integrator(oDataTuple)
    jiVal = if getHermiticity(first(d1), integrator.operator, first(d2))
        ijVal'
    else
        integrator(oDataTuple|>reverse)
    end
    (ijVal, jiVal)
end

# function genCoreIntTuple(integrator::Orb2BCCIntComputeConfigUnion{T, D}, 
#                          oDataTuple::Tuple{OrbCoreDataSeq{T, D}}, 
#                          oneBasedIds::Tuple{Int}) where {T, D}
# end

# function genCoreIntTuple(integrator::Orb2BCCIntComputeConfigUnion{T, D}, 
#                          oDataTuple::Tuple{OrbCoreDataSeq{T, D}, Bar, OrbCoreDataSeq{T, D}}, 
#                          oneBasedIds::NTuple{2, Int}) where {T, D}
# end

# function genCoreIntTuple(integrator::Orb2BCCIntComputeConfigUnion{T, D}, 
#                          oDataTuple::Tuple{OrbCoreDataSeq{T, D}, OrbCoreDataSeq{T, D}, Bar}, 
#                          oneBasedIds::NTuple{2, Int}) where {T, D}
# end

# function genCoreIntTuple(integrator::Orb2BCCIntComputeConfigUnion{T, D}, 
#                          oDataTuple::Tuple{OrbCoreDataSeq{T, D}, Bar, OrbCoreDataSeq{T, D}, 
#                                            OrbCoreDataSeq{T, D}}, 
#                          oneBasedIds::NTuple{3, Int}) where {T, D}
# end

# function genCoreIntTuple(integrator::Orb2BCCIntComputeConfigUnion{T, D}, 
#                          oDataTuple::Tuple{OrbCoreDataSeq{T, D}, OrbCoreDataSeq{T, D}, Bar, 
#                                            OrbCoreDataSeq{T, D}}, 
#                          oneBasedIds::NTuple{3, Int}) where {T, D}
# end

# function genCoreIntTuple(integrator::Orb2BCCIntComputeConfigUnion{T, D}, 
#                          oDataTuple::NTuple{4, OrbCoreDataSeq{T, D}}, 
#                          oneBasedIds::NTuple{4, Int}) where {T, D}
# end


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


function genOneBodyIntDataPairs(integrator::CachedComputeOrbCoreIntegral{T, D}, 
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
        n, m = convert1DidxTo2D(nOrbs-1, l)
        i, j = ijPair = sortTensorIndex((m, n+1))
        oDataPair = (oDataSeq[begin+i-1], oDataSeq[begin+j-1])
        ijValPair = genCoreIntTuple(integrator, oDataPair)
        (ijPair .+ offset) => ijValPair
    end

    pairs1, pairs2
end

function genOneBodyIntDataPairs(integrator::CachedComputeOrbCoreIntegral{T, D}, 
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


function genOneBodyPrimCoreIntTensor(integrator::CachedComputeOrbCoreIntegral{T, D}, 
                                     (oDataSeq,)::Tuple{OrbCoreDataSeq{T, D}}) where {T, D}
    nOrbs = length(oDataSeq)
    res = ShapedMemory{T}(undef, (nOrbs, nOrbs))

    for i in 1:nOrbs
        temp = genCoreIntTuple(integrator, (oDataSeq[begin+i-1],))
        setTensorEntries!(res, temp, (i,))
    end

    for l in 1:triMatEleNum(nOrbs-1)
        n, m = convert1DidxTo2D(mBasis-1, l)
        oDataPair = (oDataSeq[begin+m-1], oDataSeq[begin+n])
        temp = genCoreIntTuple(integrator, oDataPair)
        setTensorEntries!(res, temp, (m, n+1))
    end
    res
end

function genOneBodyPrimCoreIntTensor(integrator::CachedComputeOrbCoreIntegral{T, D}, 
                                     oDataSeqPair::NTuple{2, OrbCoreDataSeq{T, D}}) where 
                                    {T, D}
    oData1, oData2 = oDataSeqPair
    len1, len2 = length.(oDataSeqPair)
    res = ShapedMemory{T}(undef, (len1, len2))
    for j in 1:len2, i in 1:len1
        oDataPair = (oData1[begin+i-1], oData2[begin+j-1])
        ijVal = integrator(oDataPair)
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


function cachePrimCoreIntegrals!(intCache::PrimOrbCoreIntegralCache{T, D}, 
                                 paramCache::DimSpanDataCacheBox{T}, 
                                 orbMarkerCache::OrbCoreMarkerDict, 
                                 orbData::Union{OrbitalCollection{T, D}, FrameworkOrb{T, D}}
                                 ) where {T, D}
    orbCache = intCache.basis
    oldMaxIdx = lastindex(orbCache.list)
    orbIdxList = indexCacheOrbData!(orbCache, paramCache, orbMarkerCache, orbData)
    updatePrimCoreIntCache!(intCache, oldMaxIdx+1)
    orbIdxList
end

function cachePrimCoreIntegrals!(targetIntCache::C, sourceIntCache::C, 
                                 orbMarkerCache::OrbCoreMarkerDict, 
                                 sourceOrbList::BasisIndexList) where 
                                {C<:PrimOrbCoreIntegralCache}
    tOrbCache = targetIntCache.basis
    oldMaxIdx = lastindex(tOrbCache.list)
    sOrbCache = sourceIntCache.basis
    orbIdxList = indexCacheOrbData!(tOrbCache, sOrbCache, orbMarkerCache, sourceOrbList)
    updatePrimCoreIntCache!(targetIntCache, oldMaxIdx+1)
    orbIdxList
end


function updateIntCacheCore!(integrator::CachedComputeOrbCoreIntegral{T, D}, 
                             ints::OneBodyFullCoreIntegrals{T, D}, 
                             basis::Tuple{OrbCoreDataSeq{T, D}}, 
                             offset::Tuple{Int}) where {T, D}
    pairs1, pairs2 = genOneBodyIntDataPairs(integrator, basis, offset)
    foreach(p->setIntegralData!(ints, p), pairs1)
    foreach(p->setIntegralData!(ints, p), pairs2)
    ints
end

function updateIntCacheCore!(integrator::CachedComputeOrbCoreIntegral{T, D}, 
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
    integrator = CachedComputeOrbCoreIntegral(Val(true), S(), cache.operator, T)

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
        n, m = convert1DidxTo2D(len-1, l)
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
        integrator = CachedComputeOrbCoreIntegral(Val(true), OneBodyIntegral{D}(), op, T)
        tensor = genOneBodyPrimCoreIntTensor(integrator, (coreData,))
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
    integrator = CachedComputeOrbCoreIntegral(Val(true), OneBodyIntegral{D}(), op, T)

    tensor = if lazyCompute
        oData1 = Vector(oData1)
        oData2 = Vector(oData2)
        transformation = (b::PrimitiveOrbCore{T, D})->lazyMarkObj!(basisMarkerCache, b)
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

function computeIntegral(::OneBodyIntegral{D}, ::Identity, 
                         bfPair::NTuple{2, FrameworkOrb{T, D}}; 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                         basisMarkerCache::OrbCoreMarkerDict=OrbCoreMarkerDict(), 
                         lazyCompute::Bool=false) where {T, D}
    op = Identity()
    bf1, bf2 = bfPair

    if lazyCompute
        if bf1 === bf2
            computeIntegral(OneBodyIntegral{D}(), op, (bf1,); paramCache, 
                            basisMarkerCache, lazyCompute)
        else
            normCache = initializeOverlapCache!(paramCache, bfPair)
            computeIntegral!(normCache, normCache, bfPair; paramCache, basisMarkerCache)
        end
    else
        oData1, w1 = decomposeOrb(bf1; paramCache, markerCache=basisMarkerCache)
        oData2, w2 = decomposeOrb(bf2; paramCache, markerCache=basisMarkerCache)
        integrator = CachedComputeOrbCoreIntegral(Val(true), OneBodyIntegral{D}(), op, T)
        tensor = genOneBodyPrimCoreIntTensor(integrator, (oData1, oData2))
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