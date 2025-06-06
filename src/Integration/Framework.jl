using LinearAlgebra: dot

const OrbCoreMarker{T<:Real, D} = Union{
    FieldMarker{:TypedReturn, 1}, 
    ValueMarker{TypedCarteFunc{C, D, F}} where {C<:RealOrComplex{T}, F<:AbstractParamFunc}
}

const OrbCoreMarkerDict{T<:Real, D} = 
      Dict{EgalBox{TypedCarteFunc{<:RealOrComplex{T}, D}}, OrbCoreMarker{T, D}}

const OrbCoreKey{T<:Real, D, M<:OrbCoreMarker{T, D}, P<:OptSpanValueSet} = 
      Tuple{NTuple{D, T}, M, P}

const OrbCoreIdxDict{T<:Real, D} = Dict{OrbCoreKey{T, D}, Int}


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

#? Consider replacing it with LRU-based cache to control memory consumption
struct PrimOrbDataCache{T<:Real, D, P<:PrimOrbData{T, D}}
    dict::OrbCoreIdxDict{T, D}
    list::Vector{P}

    function PrimOrbDataCache(::Type{T}, ::Val{D}, ::Type{P}=PrimOrbData{T, D}) where 
                    {T<:Real, D, P<:PrimOrbData{T, D}}
    new{T, D, P}(OrbCoreIdxDict{T, D}(), P[])
    end
end


struct OneBodyFullCoreIntegrals{C<:RealOrComplex} <: QueryBox{C}
    aa::Dict{ Tuple{   Int},  Tuple{   C}}
    ab::Dict{NTuple{2, Int}, NTuple{2, C}}

    OneBodyFullCoreIntegrals(::Type{C}) where {C<:RealOrComplex} = 
    new{C}(Dict{Tuple{Int},  Tuple{C}}(), Dict{NTuple{2, Int}, NTuple{2, C}}())
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


function setIntegralData!(ints::OneBodyFullCoreIntegrals{C}, 
                          pair::Pair{Tuple{Int}, Tuple{C}}) where {C<:RealOrComplex}
    setindex!(getfield(ints, OneBodyIdxSymDict[true ]), pair.second, pair.first)
    ints
end

function setIntegralData!(ints::OneBodyFullCoreIntegrals{C}, 
                          pair::Pair{NTuple{2, Int}, NTuple{2, C}}) where {C<:RealOrComplex}
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

const PrimOrbDataOrType{T<:PrimOrbData} = Union{T, TypeBox{T}}

const OneBodyOrbIntLayout{O<:PrimOrbDataOrType} = Union{Tuple{O}, Tuple{O, O}}

const OrbBarLayout1{O<:PrimOrbDataOrType} = Tuple{O, Bar}
const OrbBarLayout2{O<:PrimOrbDataOrType} = Tuple{O, Bar, O}
const OrbBarLayout3{O<:PrimOrbDataOrType} = Tuple{O, O, Bar}
const OrbBarLayout4{O<:PrimOrbDataOrType} = Tuple{O, Bar, O, O}
const OrbBarLayout5{O<:PrimOrbDataOrType} = Tuple{O, O, Bar, O}
const OrbBarLayout6{O<:PrimOrbDataOrType} = Tuple{O, O, Bar, O, O}

const TwoBodyOrbIntLayout{O<:PrimOrbDataOrType} = Union{OrbBarLayout1{O}, OrbBarLayout2{O}, 
                                                        OrbBarLayout3{O}, OrbBarLayout4{O}, 
                                                        OrbBarLayout5{O}, OrbBarLayout6{O}}

const OrbCoreIntegralLayout{O<:PrimOrbDataOrType} = 
      Union{OneBodyOrbIntLayout{O}, TwoBodyOrbIntLayout{O}}

const OneBodyOrbCoreIntLayoutUnion{T<:Real, D} = OneBodyOrbIntLayout{PrimOrbData{T, D}}
const TwoBodyOrbCoreIntLayoutUnion{T<:Real, D} = TwoBodyOrbIntLayout{PrimOrbData{T, D}}
const OrbCoreIntLayoutUnion{T<:Real, D} = 
      Union{OneBodyOrbCoreIntLayoutUnion{T, D}, OneBodyOrbCoreIntLayoutUnion{T, D}}
const GaussTypeOrbIntLayout{T<:Real, D} = OrbCoreIntegralLayout{PGTOrbData{T, D}}

const OneBodyOrbCoreIntTypeLayout{T<:Real, D} = 
      OneBodyOrbIntLayout{TypeBox{ <:PrimOrbData{T, D} }}
const TwoBodyOrbCoreIntTypeLayout{T<:Real, D} = 
      TwoBodyOrbIntLayout{TypeBox{ <:PrimOrbData{T, D} }}
const OrbCoreIntTypeLayout{T<:Real, D} = 
      Union{OneBodyOrbCoreIntTypeLayout{T, D}, TwoBodyOrbCoreIntTypeLayout{T, D}}

#= Additional Method =#
function getOrbOutputTypeUnion(layout::TwoBodyOrbCoreIntLayoutUnion{T}) where {T<:Real}
    mapreduce(strictTypeJoin, layout) do ele
        ele isa Bar ? Union{} : getOrbOutputTypeUnion(ele)
    end::Union{Type{Complex{T}}, Type{T}}
end

formatOrbLayoutType(::Bar) = Bar()
formatOrbLayoutType(::PGTOrbData{T, D}) where {T<:Real, D} = TypeBox(PGTOrbData{T, D})
formatOrbLayoutType(::PrimOrbData{T, D}) where {T<:Real, D} = TypeBox(PrimOrbData{T, D})

getOrbCoreIntStyle(::OneBodyOrbCoreIntLayoutUnion) = OneBodyIntegral
getOrbCoreIntStyle(::TwoBodyOrbCoreIntLayoutUnion) = TwoBodyIntegral

struct OrbCoreIntConfig{T<:Real, D, S<:MultiBodyIntegral{D}, L} <: StructuredType
    config::NTuple{L, Union{Bar, TypeBox{<:PrimOrbData{T, D}} }}

    function OrbCoreIntConfig(layout::OrbCoreIntLayoutUnion{T, D}) where {T<:Real, D}
        IntStyle = getOrbCoreIntStyle(layout)
        config = formatOrbLayoutType.(layout)
        new{T, D, IntStyle{D}, length(config)::Int}(config)
    end
end


struct OrbIntCompCache{T<:Real, D, C<:RealOrComplex{T}, S<:MultiBodyIntegral{D}, 
                       M<:Union{ CustomCache{T}, CustomCache{C} }} <: CustomCache{C}
    dict::LRU{OrbCoreIntConfig{T, D, S}, M}

    function OrbIntCompCache(::S, ::Type{C}, 
                             ::Type{M}=Union{CustomCache{T}, CustomCache{C}}) where 
                            {D, S<:MultiBodyIntegral{D}, T<:Real, C<:RealOrComplex{T}, 
                             M<:Union{ CustomCache{T}, CustomCache{C} }}
        dict = LRU{OrbCoreIntConfig{T, D, S}, M}(maxsize=20)
        new{T, D, C, S, M}(dict)
    end
end

struct OrbIntNullCache{T<:Real, D, C<:RealOrComplex{T}, 
                       S<:MultiBodyIntegral{D}} <: CustomCache{C}

    function OrbIntNullCache(::S, ::Type{C}) where 
                            {D, S<:MultiBodyIntegral{D}, T<:Real, C<:RealOrComplex{T}}
        new{T, D, C, S}()
    end
end


const OrbCoreIntegralCache{T<:Real, D, C<:RealOrComplex{T}, S<:MultiBodyIntegral{D}} = 
      Union{OrbIntCompCache{T, D, C, S}, OrbIntNullCache{T, D, C, S}}

struct CachedOrbCoreIntegrator{T<:Real, D, F<:DirectOperator, 
                               M<:OrbCoreIntegralCache{T, D}} <: ConfigBox
    operator::F
    cache::M

    function CachedOrbCoreIntegrator(::Val{L}, ::S, operator::F, ::Type{C}) where 
                                    {L, F<:DirectOperator, T<:Real, C<:RealOrComplex{T}, D, 
                                     S<:MultiBodyIntegral{D}}
        cache = ifelse(L::Bool, OrbIntCompCache, OrbIntNullCache)(S(), C)
        new{T, D, F, typeof(cache)}(operator, cache)
    end
end

const DirectOrbCoreIntegrator{T<:Real, D, C<:RealOrComplex{T}, S<:MultiBodyIntegral{D}, 
                              F<:DirectOperator} = 
      CachedOrbCoreIntegrator{T, D, F, OrbIntNullCache{T, D, C, S}}

const ReusedOrbCoreIntegrator{T<:Real, D, C<:RealOrComplex{T}, S<:MultiBodyIntegral{D}, 
                              F<:DirectOperator, M<:CustomCache{<:Union{T, C}}} = 
      CachedOrbCoreIntegrator{T, D, F, OrbIntCompCache{T, D, C, S, M}}

const OrbCoreIntegratorConfig{T<:Real, D, C<:RealOrComplex{T}, S<:MultiBodyIntegral{D}, 
                              F<:DirectOperator, M<:OrbCoreIntegralCache{T, D, C, S}} = 
      CachedOrbCoreIntegrator{T, D, F, M}

const OrbCoreOneBodyIntConfig{T<:Real, D, C<:RealOrComplex{T}} = 
      OrbCoreIntegratorConfig{T, D, C, OneBodyIntegral{D}}


getIntegralOutputType(::Type{<:OrbCoreIntegralCache{T, D, C}}) where 
                     {T<:Real, D, C<:RealOrComplex{T}} = 
C

getIntegralOutputType(::CachedOrbCoreIntegrator{T, D, F, M}) where 
                     {T<:Real, D, F<:DirectOperator, M<:OrbCoreIntegralCache{T, D}} = 
getIntegralOutputType(M)

getIntegralOutputType(::MultiBodyCoreIntegrals{C}) where {C<:RealOrComplex} = C

getIntegralOutputType(::IntegralData{C}) where {C<:RealOrComplex} = C


function prepareAnalyticIntCache!(f::ReusedOrbCoreIntegrator{T, D, C, S, F}, 
                                  config::OrbCoreIntConfig{T, D}) where 
                                 {T<:Real, D, C<:RealOrComplex{T}, S<:MultiBodyIntegral{D}, 
                                  F<:DirectOperator}
    get!(f.cache.dict, config) do
        genAnalyticIntegralCache(TypeBox(F), config.config)
    end
end

prepareAnalyticIntCache!(::DirectOrbCoreIntegrator{T, D, C}, ::OrbCoreIntConfig{T, D}) where 
                        {T<:Real, D, C<:RealOrComplex{T}} = 
NullCache{C}()

function supportAnalyticIntegral(::TypeBox{F}, ::L) where 
                                {F<:DirectOperator, L<:OrbCoreIntTypeLayout}
    hasmethod(genAnalyticIntegralCache, (TypeBox{F}, L)) ? true : false
end

function selectIntegralMethod(f::OrbCoreIntegratorConfig{T, D, C, S, F}, 
                              config::OrbCoreIntConfig{T, D, S}) where 
                             {T<:Real, D, C<:RealOrComplex{T}, S<:MultiBodyIntegral{D}, 
                              F<:DirectOperator}
    if supportAnalyticIntegral(TypeBox(F), config.config)
        cache = prepareAnalyticIntCache!(f, config)
        genAnalyticIntegrator!(S(), cache, f.operator)
    else
        LPartial(getNumericalIntegral, (S(), f.operator))
    end
end


#> One-Body (i|O|j) hermiticity across O
getHermiticity(::PrimOrbData{T, D}, ::DirectOperator, 
               ::PrimOrbData{T, D}) where {T<:Real, D} = 
false

getHermiticity(::PrimOrbData{T, D}, ::OverlapSampler, 
               ::PrimOrbData{T, D}) where {T<:Real, D} = 
true

getHermiticity(::PrimOrbData{T, D}, ::MultipoleMomentSampler, 
               ::PrimOrbData{T, D}) where {T<:Real, D} = 
true

getHermiticity(::PGTOrbData{T, D}, ::DiagDirectionalDiffSampler, 
               ::PGTOrbData{T, D}) where {T<:Real, D} = 
true

#> Two-Body (ii|O|jj) (ii|O|jk) (ij|O|kk) (ij|O|kl) hermiticity across O
getHermiticity(::N12Tuple{PrimOrbData{T, D}}, ::OverlapSampler, 
               ::N12Tuple{PrimOrbData{T, D}}) where {T<:Real, D} = 
true


function genCoreIntTuple(integrator::OrbCoreOneBodyIntConfig{T, D}, 
                         oDataTuple::Tuple{PrimOrbData{T, D}}) where {T<:Real, D}
    layoutConfig = OrbCoreIntConfig(oDataTuple)
    integralRes = selectIntegralMethod(integrator, layoutConfig)(oDataTuple)
    orbSetType = getIntegralOutputType(integrator)
    (convert(orbSetType, integralRes),)
end

function genCoreIntTuple(integrator::OrbCoreOneBodyIntConfig{T, D}, 
                         oDataTuple::NTuple{2, PrimOrbData{T, D}}) where {T<:Real, D}
    layoutConfig = OrbCoreIntConfig(oDataTuple)
    f = selectIntegralMethod(integrator, layoutConfig)
    ijVal = f(oDataTuple)
    jiVal = if getHermiticity(first(oDataTuple), integrator.operator, last(oDataTuple))
        conj(ijVal)
    else
        f(reverse(oDataTuple))
    end
    orbSetType = getIntegralOutputType(integrator)
    (convert(orbSetType, ijVal), convert(orbSetType, jiVal))
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
                                (oDataSeq,)::Tuple{PrimOrbDataVec{T, D}}, 
                                (indexOffset,)::Tuple{Int}=(0,)) where {T<:Real, D}
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
                                oDataSeqPair::NTuple{2, PrimOrbDataVec{T, D}}, 
                                indexOffsets::NTuple{2, Int}) where {T<:Real, D}
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


function genOneBodyPrimCoreIntTensor(integrator::CachedOrbCoreIntegrator{T, D}, 
                                     (oDataSeq,)::Tuple{PrimOrbDataVec{T, D}}
                                     ) where {T<:Real, D}
    nOrbs = length(oDataSeq)
    orbSetType = getIntegralOutputType(integrator)
    res = ShapedMemory{orbSetType}(undef, (nOrbs, nOrbs))

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
                                     oDataSeqPair::NTuple{2, PrimOrbDataVec{T, D}}
                                     ) where {T<:Real, D}
    oData1, oData2 = oDataSeqPair
    len1, len2 = length.(oDataSeqPair)
    res = ShapedMemory{T}(undef, (len1, len2))
    for j in 1:len2, i in 1:len1
        oDataPair = (oData1[begin+i-1], oData2[begin+j-1])
        layoutConfig = OrbCoreIntConfig(oDataPair)
        ijVal = selectIntegralMethod(integrator, layoutConfig)(oDataPair)
        res[begin+i-1, begin+j-1] = ijVal
    end
    res
end

#? Try simplify the indexing complexity by incorporating `OneToIndex`
struct BasisIndexList <: QueryBox{Int}
    index::Memory{Int}
    endpoint::Memory{Int}

    function BasisIndexList(basisSizes::Union{NonEmptyTuple{Int}, AbstractVector{Int}})
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
                        data::PrimOrbData{T, D}) where {T<:Real, D}
    innerField = data.core
    marker = lazyMarkObj!(cache, innerField.core)
    (innerField.center, marker, innerField.param)
end


function updateOrbCache!(orbCache::PrimOrbDataCache{T, D}, 
                         orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                         orbData::PrimOrbData{T, D}) where {T<:Real, D}
    basis = orbCache.list
    get!(orbCache.dict, genOrbCoreKey!(orbMarkerCache, orbData)) do
        push!(basis, orbData)
        lastindex(basis)
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
                            orbsData::OrbDataCollection{T, D}) where {T<:Real, D}
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
                            sourceList::BasisIndexList) where {T<:Real, D}
    targetList = BasisIndexList(sourceList)
    primOrbIds = targetList.index
    for i in eachindex(primOrbIds)
        orbData = sourceCache.list[sourceList.index[i]]
        primOrbIds[i] = updateOrbCache!(targetCache, orbMarkerCache, orbData)
    end
    targetList
end


#> Compute and cache `PrimOrbData` and core integrals
function cachePrimCoreIntegrals!(intCache::PrimOrbCoreIntegralCache{T, D}, 
                                 orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                                 orbData::OrbDataSource{T, D}) where {T<:Real, D}
    orbCache = intCache.basis
    oldMaxIdx = lastindex(orbCache.list)
    orbIdxList = indexCacheOrbData!(orbCache, orbMarkerCache, orbData)
    updatePrimCoreIntCache!(intCache, oldMaxIdx+1)
    orbIdxList
end
#> First try loading existing `PrimOrbData` from `sourceIntCache` to `targetIntCache`
function cachePrimCoreIntegrals!(targetIntCache::PrimOrbCoreIntegralCache{T, D}, 
                                 sourceIntCache::PrimOrbCoreIntegralCache{T, D}, 
                                 orbMarkerCache::OrbCoreMarkerDict{T, D}, 
                                 sourceOrbList::BasisIndexList) where {T<:Real, D}
    tOrbCache = targetIntCache.basis
    oldMaxIdx = lastindex(tOrbCache.list)
    sOrbCache = sourceIntCache.basis
    orbIdxList = indexCacheOrbData!(tOrbCache, sOrbCache, orbMarkerCache, sourceOrbList)
    updatePrimCoreIntCache!(targetIntCache, oldMaxIdx+1)
    orbIdxList
end


function updateIntCacheCore!(integrator::CachedOrbCoreIntegrator{T, D}, 
                             ints::OneBodyCoreIntegrals{C}, 
                             basis::Tuple{PrimOrbDataVec{T, D}}, 
                             offset::Tuple{Int}) where {T<:Real, D, C<:RealOrComplex{T}}
    pairs1, pairs2 = genOneBodyIntDataPairs(integrator, basis, offset)
    foreach(p->setIntegralData!(ints, p), pairs1)
    foreach(p->setIntegralData!(ints, p), pairs2)
    ints
end

function updateIntCacheCore!(integrator::CachedOrbCoreIntegrator{T, D}, 
                             ints::OneBodyCoreIntegrals{C}, 
                             basis::NTuple{2, PrimOrbDataVec{T, D}}, 
                             offset::NTuple{2, Int}) where {T<:Real, D, C<:RealOrComplex{T}}
    pairs2 = genOneBodyIntDataPairs(integrator, basis, offset)
    foreach(p->setIntegralData!(ints, p), pairs2)
    ints
end


function updatePrimCoreIntCache!(cache::PrimOrbCoreIntegralCache{T, D}, startIdx::Int
                                 ) where {T<:Real, D}
    basis = cache.basis.list
    ints = cache.data
    firstIdx = firstindex(basis)
    intStyle = getIntegralStyle(cache)
    orbSetType = getIntegralOutputType(cache)
    integrator = CachedOrbCoreIntegrator(Val(true), intStyle, cache.operator, orbSetType)

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


function decodePrimCoreInt(cache::OneBodyFullCoreIntegrals{C1}, ptrPair::NTuple{2, Int}, 
                           (coeff1, coeff2)::NTuple{2, C2}) where 
                          {T<:Real, C1<:RealOrComplex{T}, C2<:RealOrComplex{T}}
    coeffProd = conj(coeff1) * coeff2
    ptr1, ptr2 = ptrPair
    res = if ptr1 == ptr2
        getPrimCoreIntData(cache, (ptr1,))
    else
        ptrPairNew = sortTensorIndex(ptrPair)
        data = getPrimCoreIntData(cache, ptrPairNew)
        ptrPairNew == ptrPair ? data : reverse(data)
    end
    res .* (coeffProd, conj(coeffProd))
end

function decodePrimCoreInt(cache::OneBodyFullCoreIntegrals{C1}, ptr::Tuple{Int}, 
                           (coeff1,)::Tuple{C2}=(one(C1),)) where 
                          {T<:Real, C1<:RealOrComplex{T}, C2<:RealOrComplex{T}}
    getPrimCoreIntData(cache, ptr) .* conj(coeff1) .* coeff1
end


function buildPrimOrbWeight(normCache::OverlapCoreCache{T, D, C}, data::PrimOrbData{T, D}, 
                            idx::Int) where {T<:Real, D, C<:RealOrComplex{T}}
    if data.renormalize
        convert(C, decodePrimCoreInt(normCache.data, (idx,)) |> first |> absSqrtInv)
    else
        one(C)
    end
end


function buildNormalizedCompOrbWeight!(weight::AbstractVector{C}, 
                                       normCache::OverlapCoreCache{T, D, C}, 
                                       data::CompOrbData{T, D}, 
                                       idxSeq::AbstractVector{Int}) where 
                                      {T<:Real, D, C<:RealOrComplex{T}}
    overlapCache = normCache.data
    nPrimOrbs = getOrbDataSize(data)
    innerOverlapSum = zero(T)

    for i in 1:nPrimOrbs
        ptr = idxSeq[begin+i-1]
        innerCoreDiagOverlap = decodePrimCoreInt(overlapCache, (ptr,)) |> first
        wc = weight[begin+i-1]
        innerDiagOverlap = conj(wc) * wc
        if data.basis[begin+i-1].renormalize
            weight[begin+i-1] *= absSqrtInv(innerCoreDiagOverlap)
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
    weight .*= absSqrtInv(innerOverlapSum)
end


function buildOrbWeight!(normCache::OverlapCoreCache{T, D, C}, orbData::PrimOrbData{T, D}, 
                         idxSeq::AbstractVector{Int}) where {T<:Real, D, C<:RealOrComplex{T}}
    weight = Memory{C}(undef, 1)
    weight[] = buildPrimOrbWeight(normCache, orbData, idxSeq[])
    weight
end

function buildOrbWeight!(normCache::OverlapCoreCache{T, D, C}, orbData::CompOrbData{T, D}, 
                         idxSeq::AbstractVector{Int}) where {T<:Real, D, C<:RealOrComplex{T}}
    nPrimOrbs = getOrbDataSize(orbData)
    weight = Memory{C}(orbData.weight)
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
                                     primOrbWeight::AbstractVector{C}) where 
                                    {C<:RealOrComplex}
    nPtrs = length(primOrbWeight)
    list = TypedIdxerMemory{C}(undef, nPtrs)
    for i in 1:nPtrs
        list[begin+i-1] = primOrbPtrs[begin+i-1] => primOrbWeight[begin+i-1]
    end
    list
end


function buildIndexedOrbWeights!(normCache::OverlapCoreCache{T, D}, 
                                 data::OrbDataCollection{T, D}, 
                                 normIdxList::BasisIndexList, 
                                 intIdxList::BasisIndexList=normIdxList) where {T<:Real, D}
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
                              (intWeights,)::Tuple{TypedIdxerMemory{C}}) where 
                             {T<:Real, D, C<:RealOrComplex{T}}
    len = length(intWeights)
    intValCache = intCache.data
    temp = mapreduce(+, eachindex(intWeights)) do i
        ptr, coeff = intWeights[i]
        (first∘decodePrimCoreInt)(intValCache, (ptr,), (coeff,))
    end
    res = mapreduce(+, 1:triMatEleNum(len-1), init=temp) do l
        m, n = convertIndex1DtoTri2D(l)
        ptr1, weight1 = intWeights[begin+m-1]
        ptr2, weight2 = intWeights[begin+n]
        (sum∘decodePrimCoreInt)(intValCache, (ptr1, ptr2), (weight1, weight2))
    end
    (res,) # ([1|O|1],)
end

function buildIntegralEntries(intCache::POrb1BCoreICache{T, D, C}, 
                              intWeightPair::NTuple{2, TypedIdxerMemory{C}}) where 
                             {T<:Real, D, C<:RealOrComplex{T}}
    pairs1, pairs2 = intWeightPair
    intValCache = intCache.data
    idxPairRange = Iterators.product(eachindex(pairs1), eachindex(pairs2))
    mapreduce(.+, idxPairRange, init=( zero(C), zero(C) )) do (i, j)
        ptr1, weight1 = pairs1[i]
        ptr2, weight2 = pairs2[j]
        decodePrimCoreInt(intValCache, (ptr1, ptr2), (weight1, weight2))
    end # ([1|O|2], [2|O|1])
end


function buildIntegralTensor(intCache::POrb1BCoreICache{T, D, C}, 
                             intWeights::AbstractVector{TypedIdxerMemory{C}}) where 
                            {T<:Real, D, C<:RealOrComplex{T}}
    nOrbs = length(intWeights)
    res = ShapedMemory{C}(undef, (nOrbs, nOrbs))
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

function computeIntTensor(style::MultiBodyIntegral{D}, op::DirectOperator, 
                          orbs::OrbBasisVec{T, D}; 
                          cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox(), 
                          markerCache::OrbCoreMarkerDict{T, D}=OrbCoreMarkerDict{T, D}()
                          ) where {T<:Real, D}
    orbsData = genOrbitalData(orbs; cache!Self)
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
        integrator = CachedOrbCoreIntegrator(Val(true), OneBodyIntegral{D}(), op, C)
        tensor = genOneBodyPrimCoreIntTensor(integrator, (coreData,))
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
    integrator = CachedOrbCoreIntegrator(Val(true), OneBodyIntegral{D}(), op, oPairType)

    tensor = if lazyCompute
        oData1 = Vector(oData1)
        oData2 = Vector(oData2)
        #> Find the shared `PrimOrbData` (excluding the renormalization information)
        transformation = (b::PrimOrbData{T, D})->genOrbCoreKey!(markerCache, b)
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
                           ) where {T<:Real, D}
    normIdxList = cachePrimCoreIntegrals!(normCache, markerCache, orbData)
    orbWeight = buildOrbWeight!(normCache, orbData, normIdxList.index)
    normCache.basis.list[normIdxList.index], orbWeight
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
        integrator = CachedOrbCoreIntegrator(Val(true), OneBodyIntegral{D}(), op, oPairType)
        tensor = genOneBodyPrimCoreIntTensor(integrator, (oData1, oData2))
        dot(w1, tensor, w2)
    end
end

function computeIntegral(style::MultiBodyIntegral, op::DirectOperator, 
                         orbs::NonEmptyTuple{OrbitalBasis{<:RealOrComplex{T}, D}}; 
                         cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox(), 
                         lazyCompute::Bool=false) where {T<:Real, D}
    orbsData = genOrbitalData(orbs; cache!Self)
    computeIntegral(style, op, orbsData; lazyCompute)
end