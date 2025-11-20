@enum OrbitalCategory::Int8 begin
    PrimGaussTypeOrb
    ArbitraryTypeOrb
end

const OrbIntLayoutInfo{N} = 
      Tuple{TypeBox{<:DirectOperator}, NTuple{N, NTuple{2, OrbitalCategory}}}

const OrbIntLayoutCache{T<:Real, C<:RealOrComplex{T}, N, 
                        M<:Union{OptionalCache{T}, OptionalCache{C}}} = 
      LRU{OrbIntLayoutInfo{N}, M}

const OptOrbIntLayoutCache{T<:Real, C<:RealOrComplex{T}, N} = 
      Union{EmptyDict{OrbIntLayoutInfo{N}, C}, OrbIntLayoutCache{T, C, N}}

const OptEstimatorConfig{T} = MissingOr{EstimatorConfig{T}}


struct OrbitalIntegrationConfig{T<:Real, D, C<:RealOrComplex{T}, N, F<:DirectOperator, 
                                M<:OptOrbIntLayoutCache{T, C, N}, E<:OptEstimatorConfig{T}
                                } <: ConfigBox
    operator::F
    cache::M
    estimator::E

    function OrbitalIntegrationConfig(::MultiBodyIntegral{D, C, N}, operator::F, 
                                      cache::OptOrbIntLayoutCache{T, C, N}, 
                                      config::OptEstimatorConfig{T}) where 
                                     {D, T<:Real, C<:RealOrComplex{T}, N, F<:DirectOperator}
        new{T, D, C, N, F, typeof(cache), typeof(config)}(operator, cache, config)
    end
end

function OrbitalIntegrationConfig(style::MultiBodyIntegral{D, C, N}, operator::F, 
                                  caching::Boolean, config::OptEstimatorConfig{T}=missing
                                  ) where {D, T<:Real, C<:RealOrComplex{T}, N, 
                                           F<:DirectOperator}
    cache = if evalTypedData(caching)
        valueTypeBound = Union{OptionalCache{T}, OptionalCache{C}}
        LRU{OrbIntLayoutInfo{N}, valueTypeBound}(maxsize=20)
    else
        EmptyDict{OrbIntLayoutInfo{N}, C}()
    end
    OrbitalIntegrationConfig(style, operator, cache, config)
end

const OrbitalOverlapConfig{T<:Real, D, C<:RealOrComplex{T}, 
                           M<:OptOrbIntLayoutCache{T, C, 1}, E<:OptEstimatorConfig{T}} = 
      OrbitalIntegrationConfig{T, D, C, 1, OverlapSampler, M, E}

struct OneBodyIntegralValCache{C<:RealOrComplex} <: QueryBox{C}
    aa::LRU{N1N2Tuple{OneToIndex}, C}
    ab::LRU{N1N2Tuple{OneToIndex}, C}
    dimension::Int
    threshold::Int

    function OneBodyIntegralValCache(::OneBodyIntegral{D, C}, 
                                     threshold::Int=1024) where {D, C<:RealOrComplex}
        checkPositivity(threshold)
        maxPairNum = threshold * (threshold - 1)
        aaSector = LRU{N1N2Tuple{OneToIndex}, C}(maxsize=threshold )
        abSector = LRU{N1N2Tuple{OneToIndex}, C}(maxsize=maxPairNum)
        new{C}(aaSector, abSector, Int(D), threshold)
    end
end


struct TwoBodyIntegralValCache{C<:RealOrComplex} <: QueryBox{C}
    aaaa::LRU{N2N2Tuple{OneToIndex}, C}
    aabb::LRU{N2N2Tuple{OneToIndex}, C}
    half::LRU{N2N2Tuple{OneToIndex}, C} # aaxy or xyaa
    misc::LRU{N2N2Tuple{OneToIndex}, C} # abxy
    dimension::Int
    threshold::Int

    function TwoBodyIntegralValCache(::TwoBodyIntegral{D, C}, 
                                     threshold::Int=128) where {D, C<:RealOrComplex}
        checkPositivity(threshold)
        threshold2 = threshold * (threshold - 1)
        threshold3 = threshold * threshold2 * 2
        threshold4 = threshold^4 - threshold - threshold2 - threshold3
        aaaaSector = LRU{N2N2Tuple{OneToIndex}, C}(maxsize=threshold )
        aabbSector = LRU{N2N2Tuple{OneToIndex}, C}(maxsize=threshold2)
        halfSector = LRU{N2N2Tuple{OneToIndex}, C}(maxsize=threshold3)
        miscSector = LRU{N2N2Tuple{OneToIndex}, C}(maxsize=threshold4)
        new{C}(aaaaSector, aabbSector, halfSector, miscSector, Int(D), threshold)
    end
end

const FauxIntegralValCache{N, C<:RealOrComplex} = 
      EmptyDict{NTuple{N, NTuple{2, OneToIndex}}, C}

genFauxIntegralValCache(::Count{N}, ::Type{C}) where {N, C<:RealOrComplex} = 
EmptyDict{NTuple{N, NTuple{2, OneToIndex}}, C}()::FauxIntegralValCache{N, C}

const OneBodyInteValCacheUnion{C<:RealOrComplex} = Union{
    OneBodyIntegralValCache{C}, 
    FauxIntegralValCache{1, C}
}

const TwoBodyInteValCacheUnion{C<:RealOrComplex} = Union{
    TwoBodyIntegralValCache{C}, 
    FauxIntegralValCache{2, C}
}

const MultiBodyIntegralValCache{C<:RealOrComplex} = Union{
    OneBodyInteValCacheUnion{C}, 
    TwoBodyInteValCacheUnion{C}
}
abstract type FieldIntegralInfo{D, C, N} <: ConfigBox end

struct OrbitalIntegralInfo{T<:Real, D, C<:RealOrComplex{T}, N, F<:DirectOperator, 
                           M<:OrbitalIntegrationConfig{T, D, C, N, F}, 
                           V<:MultiBodyIntegralValCache{C}, P<:MultiOrbitalData{T, D, C}
                           } <: FieldIntegralInfo{D, C, N}
    method::M
    memory::V
    basis::P

    OrbitalIntegralInfo(method::M, memory::V, basis::P) where {T<:Real, D, 
                        C<:RealOrComplex{T}, F<:DirectOperator, 
                        M<:OrbitalIntegrationConfig{T, D, C, 1, F}, 
                        V<:OneBodyInteValCacheUnion{C}, P<:MultiOrbitalData{T, D, C}} = 
    new{T, D, C, 1, F, M, V, P}(method, memory, basis)

    OrbitalIntegralInfo(method::M, memory::V, basis::P) where {T<:Real, D, 
                        C<:RealOrComplex{T}, F<:DirectOperator, 
                        M<:OrbitalIntegrationConfig{T, D, C, 2, F}, 
                        V<:TwoBodyInteValCacheUnion{C}, P<:MultiOrbitalData{T, D, C}} = 
    new{T, D, C, 2, F, M, V, P}(method, memory, basis)
end

const OneBodyOrbIntegralInfo{T<:Real, D, C<:RealOrComplex{T}, F<:DirectOperator, 
                             M<:OrbitalIntegrationConfig{T, D, C, 1, F}, 
                             V<:OneBodyInteValCacheUnion{C}, P<:MultiOrbitalData{T, D, C}} = 
      OrbitalIntegralInfo{T, D, C, 1, F, M, V, P}

const OrbitalOverlapInfo{T<:Real, D, C<:RealOrComplex{T}, M<:OrbitalOverlapConfig{T, D, C}, 
                         V<:OneBodyInteValCacheUnion{C}, P<:MultiOrbitalData{T, D, C}} = 
      OneBodyOrbIntegralInfo{T, D, C, OverlapSampler, M, V, P}

const TwoBodyOrbIntegralInfo{T<:Real, D, C<:RealOrComplex{T}, F<:DirectOperator, 
                             M<:OrbitalIntegrationConfig{T, D, C, 2, F}, 
                             V<:TwoBodyInteValCacheUnion{C}, P<:MultiOrbitalData{T, D, C}} = 
      OrbitalIntegralInfo{T, D, C, 2, F, M, V, P}

const OrbitalLayoutInteInfo{T<:Real, D, C<:RealOrComplex{T}, N, F<:DirectOperator, 
                            M<:OrbitalIntegrationConfig{T, D, C, N, F}, 
                            V<:MultiBodyIntegralValCache{C}, P<:OrbitalLayoutData{T, D}} = 
      OrbitalIntegralInfo{T, D, C, N, F, M, V, P}

const OrbitalVectorInteInfo{T<:Real, D, C<:RealOrComplex{T}, N, F<:DirectOperator, 
                            M<:OrbitalIntegrationConfig{T, D, C, N, F}, 
                            V<:MultiBodyIntegralValCache{C}, P<:OrbitalVectorData{T, D}} = 
      OrbitalIntegralInfo{T, D, C, N, F, M, V, P}


function initializeOrbIntegral(::MultiBodyIntegral{D, C, N}, op::DirectOperator, 
                               data::MultiOrbitalData{T, D, C}, 
                               cacheConfig::Union{Boolean, OptOrbIntLayoutCache{T, C, N}}, 
                               estimatorConfig::OptEstimatorConfig{T}=missing) where 
                              {D, T<:Real, C<:RealOrComplex{T}, N}
    inteStyle = MultiBodyIntegral{D, C, N}()
    methodConfig = OrbitalIntegrationConfig(inteStyle, op, cacheConfig, estimatorConfig)
    if N==1
        resultConfig = OneBodyIntegralValCache(inteStyle)
    elseif N==2
        resultConfig = TwoBodyIntegralValCache(inteStyle)
    else
        throw(AssertionError("$(MultiBodyIntegral{D, C, N}) is not supported."))
    end
    OrbitalIntegralInfo(methodConfig, resultConfig, data)
end

function reformatOrbIntegral(info::OrbitalIntegralInfo{T, D, C, N}, 
                             op::F) where {T, D, C<:RealOrComplex{T}, N, F<:DirectOperator}
    inteStyle = MultiBodyIntegral{D, C, N}()
    method = info.method
    defaultCache = method.cache
    cacheConfig = if method.operator isa F; defaultCache else
                     defaultCache isa EmptyDict ? False() : True() end
    initializeOrbIntegral(inteStyle, op, info.basis, cacheConfig, method.estimator)
end

function reformatOrbIntegral(info::OrbitalIntegralInfo{T, D, C, N}, 
                             ptrFormat::OrbFormatCollection{D, C}) where 
                            {T, D, C<:RealOrComplex{T}, N}
    if ptrFormat isa Tuple && !(length(ptrFormat) == 2N)
        throw(AssertionError("The format of orbital pointers does not match `info`."))
    end
    newBasis = MultiOrbitalData(info.basis, ptrFormat)
    OrbitalIntegralInfo(info.method, info.memory, newBasis)
end

function initializeOrbNormalization(inteInfo::OrbitalIntegralInfo{T, D, C, N}, 
                                    caching::Boolean) where 
                                   {T<:Real, D, C<:RealOrComplex{T}, N}
    inteMethod = inteInfo.method
    inteMemory = inteInfo.memory
     basisData = inteInfo.basis
     estConfig = inteMethod.estimator
     isOverlap = N==1 && inteMethod.operator isa OverlapSampler

    op = genOverlapSampler()
    style = OneBodyIntegral{D, C}()

    if evalTypedData(caching)
        normConfig = if isOverlap && !(inteMethod.cache isa EmptyDict); inteMethod else
                        OrbitalIntegrationConfig(style, op, True(), estConfig) end
        normMemory = if isOverlap && !(inteMemory isa FauxIntegralValCache); inteMemory else
                        OneBodyIntegralValCache(style) end
    else
        normConfig = OrbitalIntegrationConfig(style, op, False(), estConfig)
        normMemory = genFauxIntegralValCache(Count(N), C)
    end
    OrbitalIntegralInfo(normConfig, normMemory, basisData)
end


getOrbitalCategory(::TypeBox{<:FloatingPolyGaussField}) = PrimGaussTypeOrb
getOrbitalCategory(::TypeBox{<:StashedShiftedField}) = ArbitraryTypeOrb

function genOrbCategoryLayout(data::N12N2Tuple{StashedShiftedField{T, D}}
                              ) where {T<:Real, D}
    map(data) do pair
        map(pair) do field
            getOrbitalCategory(field|>typeof|>TypeBox)
        end
    end
end

const MultiBodyOrbCorePair{T<:Real, D, N} = 
      Pair{ <:NTuple{N, NTuple{ 2, StashedShiftedField{T, D} }}, 
              NTuple{N, NTuple{ 2, OneToIndex                }} }


toSingleBool(num::OctalNumber) = Bool(num)

function toTripleBool(num::OctalNumber)
    val = Int(num)
    bit1 = (val & 1) != 0
    bit2 = (val & 2) != 0
    bit3 = (val & 4) != 0
    (bit1, bit2, bit3)
end

toOctalNumber(val::Bool) = OctalNumber(val)

function toOctalNumber(val::NTuple{3, Bool})
    bit1, bit2, bit3 = val
    OctalNumber(Int(bit1) + 2Int(bit2) + 4Int(bit3))
end


#>> Operator-orbital layout symmetry
#> One-Body (i|O|j) symmetry across O: (i|O|j)' == (j|O|i) when i != j
getIntegralOpOrbSymmetry(::DirectOperator, ::N1N2Tuple{OrbitalCategory}) = false
getIntegralOpOrbSymmetry(::OverlapSampler, ::N1N2Tuple{OrbitalCategory}) = true
getIntegralOpOrbSymmetry(::MultipoleMomentSampler, ::N1N2Tuple{OrbitalCategory}) = true
function getIntegralOpOrbSymmetry(::DiagDirectionalDiffSampler, 
                                  layout::N1N2Tuple{OrbitalCategory})
    all(layout|>first) do c; (c == PrimGaussTypeOrb) end
end
getIntegralOpOrbSymmetry(::CoulombMultiPointSampler, ::N1N2Tuple{OrbitalCategory}) = true
#> Two-body (ij|O|kl) symmetry between ij: (ij|O|kl)' == (ji|O|kl) when  i != j
#> Two-body (ij|O|kl) symmetry between kl: (ij|O|kl)' == (ij|O|lk) when  k != l
#> Two-body (ij|O|kl) symmetry across O:   (ij|O|kl)  == (kl|O|ij) when ij != kl
getIntegralOpOrbSymmetry(::DirectOperator, ::N2N2Tuple{OrbitalCategory}) = 
(false, false, false)

getIntegralOpOrbSymmetry(::CoulombInteractionSampler, ::N2N2Tuple{OrbitalCategory}) = 
(true, true, true)

#>> Index layout symmetry
getIntegralIndexSymmetry((part,)::N1N2Tuple{OneToIndex}) = first(part) == last(part)
function getIntegralIndexSymmetry((partL, partR)::N2N2Tuple{OneToIndex})
    idxL1, idxR1 = partL
    idxL2, idxR2 = partR
    (idxL1==idxR1, idxL2==idxR2, partL==partR)
end


function prepareInteValCache(cache::FauxIntegralValCache{1}, layout::N1N2Tuple{OneToIndex})
    layout=>cache, getIntegralIndexSymmetry(layout)
end

function prepareInteValCache(cache::OneBodyIntegralValCache, layout::N1N2Tuple{OneToIndex})
    indexSymmetry = getIntegralIndexSymmetry(layout)
    keySectorPair = layout => ifelse(indexSymmetry, cache.aa, cache.ab)
    keySectorPair, indexSymmetry
end

function prepareInteValCache(cache::FauxIntegralValCache{2}, layout::N2N2Tuple{OneToIndex})
    layout=>cache, getIntegralIndexSymmetry(layout)
end

function prepareInteValCache(cache::TwoBodyIntegralValCache, layout::N2N2Tuple{OneToIndex})
    symmetryL, symmetryR, _ = indexSymmetry = getIntegralIndexSymmetry(layout)

    keySector = if all(indexSymmetry)
        cache.aaaa
    elseif symmetryL && symmetryR
        cache.aabb
    elseif symmetryL
        cache.half #> aaxy
    elseif symmetryR
        cache.half #> xyaa
    else
        cache.misc #> abxy
    end

    (layout => keySector), indexSymmetry
end

#> ab
function formatIntegralCacheKey(key::N1N2Tuple{OneToIndex}, permuteControl::Bool)
    (i, j), = key
    if i > j && permuteControl
        ((j, i),) => true
    else
        key => false
    end
end

#> abcd
function formatIntegralCacheKey(key::N2N2Tuple{OneToIndex}, permuteControl::NTuple{3, Bool})
    needToConjugate = false
    permuteL, permuteR, permuteLR = permuteControl

    (i, j), partR = key
    if i > j && permuteL
        key = ((j, i), partR)
        needToConjugate = !needToConjugate
    end

    partL, (k, l) = key
    if k > l && permuteR
        key = (partL, (l, k))
        needToConjugate = !needToConjugate
    end

    partL, partR = key
    if partL > partR && permuteLR
        key = (partR, partL)
    end

    key => needToConjugate
end
#> `indexSymmetry` should be index based symmetry -> find correct layout sector
#> `layoutSymmetry` should be operator--orbital based symmetry -> reuse integral result
function getIntegralValue!(cache::MultiBodyIntegralValCache{C}, 
                           method::OrbitalIntegrationConfig{T, D, C, N}, 
                           pair::MultiBodyOrbCorePair{T, D, N}) where 
                          {T<:Real, D, C<:RealOrComplex{T}, N}
    fieldLayout, indexLayout = pair
    (key, sector), indexSymmetry = prepareInteValCache(cache, indexLayout)
    categoryLayout = genOrbCategoryLayout(fieldLayout)
    layoutSymmetry = getIntegralOpOrbSymmetry(method.operator, categoryLayout)
    permuteControl = .!(indexSymmetry) .&& layoutSymmetry
    orderedIdsKey, needToConjugate = formatIntegralCacheKey(key, permuteControl)

    res = get(sector, orderedIdsKey, nothing)::NothingOr{C}
    if res === nothing
        inteVal = evaluateIntegral!(method, fieldLayout)::C
        res = needToConjugate ? conj(inteVal) : inteVal
        setindex!(sector, res, orderedIdsKey)
    end #> Fewer allocations than using `get!`

    needToConjugate ? conj(res) : res
end


function evaluateIntegral!(config::OrbitalIntegrationConfig{T, D, C, N, F}, 
                           layout::NTuple{N, NTuple{ 2, StashedShiftedField{T, D} }}, 
                           ) where {T, C<:RealOrComplex{T}, D, N, F<:DirectOperator}
    component = prepareInteComponent!(config, layout)
    evaluateIntegralCore!(TypedOperator(config.operator, C), component, layout)
end
#> Adaptive integration interface 1
function prepareInteComponent!(config::OrbitalIntegrationConfig{T, D}, 
                               ::N12N2Tuple{StashedShiftedField{T, D}}) where {T<:Real, D}
    config.estimator
end
#> Adaptive integration interface 2
function evaluateIntegralCore!(formattedOp::TypedOperator{C}, 
                               config::OptEstimatorConfig{T}, 
                               layout::N12N2Tuple{StashedShiftedField{T, D}}) where 
                              {T, C<:RealOrComplex{T}, D}
    estimateOrbIntegral(config, formattedOp, layout)::C
end


function formatOrbCorePair(configSource::AbstractVector{<:StashedShiftedField{T, D}}, 
                           indexer::N12N2Tuple{OneToIndex}) where {T<:Real, D}
    fields = map(indexer) do pair
        idxL, idxR = pair
        (getEntry(configSource, idxL), getEntry(configSource, idxR))
    end
    fields => indexer
end


getOrbDataPair(::Type{C}, idx::OneToIndex) where {C<:RealOrComplex} = (idx => one(C))

getOrbDataPair(::Type{C}, pair::Pair{OneToIndex, C}) where {C<:RealOrComplex} = itself(pair)


const OrbDataPtrPairLayout{D, C<:RealOrComplex} = 
      Union{NTuple{2, PrimOrbPointer{D, C}}, NTuple{ 2, OrbCorePointer{D, C} }}

const OrbDataPtrQuadLayout{D, C<:RealOrComplex} = 
      Union{NTuple{4, PrimOrbPointer{D, C}}, NTuple{ 4, OrbCorePointer{D, C} }}

const OrbDataPtrLayout{D, C<:RealOrComplex} = 
      Union{OrbDataPtrPairLayout{D, C}, OrbDataPtrQuadLayout{D, C}}

const OrbDataPtrVector{D, C<:RealOrComplex} = 
      Union{AbstractVector{<:PrimOrbPointer{D, C}}, AbstractVector{ OrbCorePointer{D, C} }}

const OrbCorePtrCollection{D, C<:RealOrComplex} = 
      Union{Memory{ OrbCorePointer{D, C} }, N24Tuple{ OrbCorePointer{D, C} }}


function getOrbLayoutIntegralCore!(inteInfo::OneBodyOrbIntegralInfo{T, D, C}, 
                                   ptrLayout::OrbDataPtrPairLayout{D, C}) where 
                                  {T<:Real, D, C<:RealOrComplex{T}}
    method = inteInfo.method
    config = inteInfo.basis.config
    memory = inteInfo.memory
    ptrL, ptrR = ptrLayout

    res = zero(C)

    for eleR in ptrR.inner
        idxR, weightR = getOrbDataPair(C, eleR)

        for eleL in ptrL.inner
            idxL, weightL = getOrbDataPair(C, eleL)

            weightProd = conj(weightL) * weightR
            orbCorePair = formatOrbCorePair(config, ((idxL, idxR),))
            ijVal = getIntegralValue!(memory, method, orbCorePair) * weightProd

            res += ijVal
        end
    end

    res::C
end


function getOrbLayoutIntegralCore!(inteInfo::TwoBodyOrbIntegralInfo{T, D, C}, 
                                   ptrLayout::OrbDataPtrQuadLayout{D, C}) where 
                                  {T<:Real, D, C<:RealOrComplex{T}}
    method = inteInfo.method
    config = inteInfo.basis.config
    memory = inteInfo.memory
    ptrL1, ptrR1, ptrL2, ptrR2 = ptrLayout

    res = zero(C)

    for eleR2 in ptrR2.inner
        idxR2, weightR2 = getOrbDataPair(C, eleR2)

        for eleL2 in ptrL2.inner
            idxL2, weightL2 = getOrbDataPair(C, eleL2)
            weightProd2 = conj(weightL2) * weightR2

            for eleR1 in ptrR1.inner
                idxR1, weightR1 = getOrbDataPair(C, eleR1)

                for eleL1 in ptrL1.inner
                    idxL1, weightL1 = getOrbDataPair(C, eleL1)
                    weightProd = conj(weightL1) * weightR1 * weightProd2
                    formattedIndex = ((idxL1, idxR1), (idxL2, idxR2))
                    orbCorePair = formatOrbCorePair(config, formattedIndex)
                    ijklVal = getIntegralValue!(memory, method, orbCorePair) * weightProd
                    res += ijklVal
                end
            end
        end
    end

    res::C
end


function getOrbInteTensorSymmetry(::MultiBodyIntegral{D, C, N}, op::DirectOperator, 
                                  type::TypeBox{<:StashedShiftedField{T, D, C}}) where 
                                 {D, T, C<:RealOrComplex{T}, N}
    orbCate = getOrbitalCategory(type)
    layout = ntuple(_->(orbCate, orbCate), Val(N))
    getIntegralOpOrbSymmetry(op, layout)
end

function setInteTensorEntry!(tensor::AbstractArray{C, N}, val::C, 
                             idxTuple::NTuple{N, OneToIndex}) where {C<:RealOrComplex, N}
    ids = shiftAxialIndex(tensor, idxTuple)
    tensor[ids...] = val
end

function evalSetInteTensorEntry!(tensor::AbstractArray{C, N}, 
                                 inteInfo::OrbitalIntegralInfo{T, D, C}, 
                                 ptrVector::OrbDataPtrVector{D, C}, 
                                 idxTpl::NTuple{N, OneToIndex}) where 
                                {T<:Real, D, C<:RealOrComplex{T}, N}
    ptrTpl = getEntry.(Ref(ptrVector), idxTpl)
    val = getOrbLayoutIntegralCore!(inteInfo, ptrTpl)
    setInteTensorEntry!(tensor, val, idxTpl)
    val
end

function getOrbVectorIntegralCore!(inteInfo::OneBodyOrbIntegralInfo{T, D, C}, 
                                   ptrVector::OrbDataPtrVector{D, C}
                                   ) where {T<:Real, D, C<:RealOrComplex{T}}
    len = length(ptrVector)
    op = inteInfo.method.operator
    style = OneBodyIntegral{D, C}()
    tensor = Array{C}(undef, (len, len))
    typeInfo = (TypeBox∘eltype)(inteInfo.basis.config)
    symmetry = getOrbInteTensorSymmetry(style, op, typeInfo)

    if symmetry
        for n in 1:symmetric2DArrEleNum(len)
            ijIdx = OneToIndex.(n|>convertIndex1DtoTri2D)
            ijVal = evalSetInteTensorEntry!(tensor, inteInfo, ptrVector, ijIdx)
            if !getIntegralIndexSymmetry(ijIdx|>tuple)
                setInteTensorEntry!(tensor, conj(ijVal), reverse(ijIdx))
            end
        end
    else
        idxRange = OneToRange(len)
        for j in idxRange, i in idxRange
            evalSetInteTensorEntry!(tensor, inteInfo, ptrVector, (i, j))
        end
    end

    tensor
end


function getOrbVectorIntegralCore!(inteInfo::TwoBodyOrbIntegralInfo{T, D, C}, 
                                   ptrVector::OrbDataPtrVector{D, C}
                                   ) where {T<:Real, D, C<:RealOrComplex{T}}
    len = length(ptrVector)
    op = inteInfo.method.operator
    style = TwoBodyIntegral{D, C}()
    tensor = Array{C}(undef, (len, len, len, len))
    typeInfo = (TypeBox∘eltype)(inteInfo.basis.config)
    symL, symR, symO = getOrbInteTensorSymmetry(style, op, typeInfo)

    if symL && symR && symO
        for n in 1:symmetric4DArrEleNum(len)
            i, j, k, l = ijklIdx = OneToIndex.(n|>convertIndex1DtoTri4D)
            sym1, sym2, sym3 = getIntegralIndexSymmetry(( (i, j), (k, l) ))
            sym12 = sym1 && sym2

            ijklVal = evalSetInteTensorEntry!(tensor, inteInfo, ptrVector, ijklIdx)
            ijklValConj = conj(ijklVal)

            sym1  || setInteTensorEntry!(tensor, ijklValConj, (j, i, k, l))
            sym2  || setInteTensorEntry!(tensor, ijklValConj, (i, j, l, k))
            sym12 || setInteTensorEntry!(tensor, ijklVal,     (j, i, l, k))
            sym3  || setInteTensorEntry!(tensor, ijklVal,     (k, l, i, j))
            sym3 && sym1  || setInteTensorEntry!(tensor, ijklValConj, (k, l, j, i))
            sym3 && sym2  || setInteTensorEntry!(tensor, ijklValConj, (l, k, i, j))
            sym3 && sym12 || setInteTensorEntry!(tensor, ijklVal,     (l, k, j, i))
        end
    elseif symL && symR && !symO
        for n in 1:symmetric2DArrEleNum(len)
            k, l = OneToIndex.(n|>convertIndex1DtoTri2D)
            sym2 = k == l

            for m in 1:symmetric2DArrEleNum(len)
                i, j = OneToIndex.(m|>convertIndex1DtoTri2D)
                sym1 = i == j

                ijklVal = evalSetInteTensorEntry!(tensor, inteInfo, ptrVector, (i, j, k, l))
                ijklValConj = conj(ijklVal)

                sym1 || setInteTensorEntry!(tensor, ijklValConj, (j, i, k, l))
                sym2 || setInteTensorEntry!(tensor, ijklValConj, (i, j, l, k))
                sym1 && sym2 || setInteTensorEntry!(tensor, ijklVal,  (j, i, l, k))
            end
        end
    elseif symO
        idxRange = OneToRange(len)
        for l in idxRange, k in idxRange, j in OneToRange(l), i in OneToRange(k)
            ijklVal = evalSetInteTensorEntry!(tensor, inteInfo, ptrVector, (i, j, k, l))
            (i==k && j==l) || setInteTensorEntry!(tensor, ijklVal, (i, j, l, k))
        end
    else #> It is possible but not necessary to add more branches
        idxRange = OneToRange(len)
        for l in idxRange, k in idxRange, j in idxRange, i in idxRange
            evalSetInteTensorEntry!(tensor, inteInfo, ptrVector, (i, j, k, l))
        end
    end

    tensor
end


function getOrbCoreOverlap!(info::OrbitalOverlapInfo{T, D, C}, pointer::PrimOrbPointer{D, C}
                            ) where {T<:Real, D, C<:RealOrComplex{T}}
    getOrbLayoutIntegralCore!(info, (pointer, pointer))
end

function getOrbCoreOverlap!(info::OrbitalOverlapInfo{T, D, C}, pointer::CompOrbPointer{D, C}
                            ) where {T<:Real, D, C<:RealOrComplex{T}}
    primPtrs = pointer.inner.left
    overlapSum = zero(C)

    normalizedWeights = map(pointer.inner) do (primPtr, weightOld)
        weightNew = weightOld
        diagOverlap = conj(weightOld) * weightOld
        coreOverlap = getOrbCoreOverlap!(info, primPtr)

        if primPtr.renormalize
            weightNew *= absSqrtInv(coreOverlap)
        else
            diagOverlap *= coreOverlap
        end

        overlapSum += diagOverlap

        weightNew
    end

    for n in 1:symmetric2DArrEleNum(length(pointer.inner) - 1)
        i, j = convertIndex1DtoTri2D(n)
        idxPair = OneToIndex.((i, j+1))
        weightL, weightR = getEntry.(Ref(normalizedWeights), idxPair)
        weightProd = conj(weightL) * weightR
        primPtrPair = getEntry.(Ref(primPtrs), idxPair)
        offDiagOverlap = getOrbLayoutIntegralCore!(info, primPtrPair) * weightProd
        overlapSum += offDiagOverlap + conj(offDiagOverlap)
    end

    overlapSum::C
end


function buildOrbCoreWeight!(normInfo::OrbitalOverlapInfo{T, D, C}, 
                             primOrbPtr::PrimOrbPointer{D, C}) where 
                            {T<:Real, D, C<:RealOrComplex{T}}
    if primOrbPtr.renormalize
        absSqrtInv(getOrbCoreOverlap!(normInfo, primOrbPtr))::C
    else
        one(C)
    end
end

function buildOrbCoreWeight!(normInfo::OrbitalOverlapInfo{T, D, C}, 
                             compOrbPtr::CompOrbPointer{D, C}) where 
                            {T<:Real, D, C<:RealOrComplex{T}}
    primPtrs = compOrbPtr.inner.left
    weight = ShapedMemory(compOrbPtr.inner.right)

    for (i, primPtr) in zip(eachindex(weight), primPtrs)
        weight[i] *= buildOrbCoreWeight!(normInfo, primPtr)
    end

    if compOrbPtr.renormalize
        weight .*= absSqrtInv(getOrbCoreOverlap!(normInfo, compOrbPtr))
    end

    weight
end

function getOrbCorePointers(inteInfo::OrbitalIntegralInfo, lazyNormalize::Boolean)
    ptrs = inteInfo.basis.format
    if ptrs isa OrbCorePtrCollection
        ptrs
    else
        normInfo = initializeOrbNormalization(inteInfo, lazyNormalize)
        lazyMap(ptrs) do pointer
            weightHolder = buildOrbCoreWeight!(normInfo, pointer)
            OrbCorePointer(pointer, weightHolder)
        end
    end
end


function checkOrbIntegralInfo(inteInfo::OrbitalIntegralInfo)
    weightInfo = inteInfo.basis.format
    if !(weightInfo isa OrbCorePtrCollection)
        throw(AssertionError("The basis pointers inside `inteInfo` (at `.basis.config`) "*
                             "must all be $OrbCorePointer."))
    end
    weightInfo
end

function evalOrbIntegralInfo!(inteInfo::OrbitalLayoutInteInfo{T, D, C}) where 
                             {T<:Real, D, C<:RealOrComplex{T}}
    weightInfo = checkOrbIntegralInfo(inteInfo)
    getOrbLayoutIntegralCore!(inteInfo, weightInfo)
end

function evalOrbIntegralInfo!(inteInfo::OrbitalVectorInteInfo{T, D, C}) where 
                             {T<:Real, D, C<:RealOrComplex{T}}
    weightInfo = checkOrbIntegralInfo(inteInfo)
    getOrbVectorIntegralCore!(inteInfo, weightInfo)
end

function evalOrbIntegralInfo!(op::F, inteInfo::OrbitalLayoutInteInfo{T, D, C}) where 
                             {T<:Real, D, C<:RealOrComplex{T}, F<:DirectOperator}
    if F <: Summator
        op1, op2 = op.dresser
        res1 = evalOrbIntegralInfo!(op1, inteInfo)
        res2 = evalOrbIntegralInfo!(op2, inteInfo)
        op.bundler(res1, res2)
    else
        finalInfo = reformatOrbIntegral(inteInfo, op)
        evalOrbIntegralInfo!(finalInfo)
    end
end

function evalOrbIntegralInfo!(op::F, inteInfo::OrbitalVectorInteInfo{T, D, C}) where 
                             {T<:Real, D, C<:RealOrComplex{T}, F<:DirectOperator}
    if F <: Summator
        op1, op2 = op.dresser
        bundler = op.bundler
        res1 = evalOrbIntegralInfo!(op1, inteInfo)
        res2 = evalOrbIntegralInfo!(op2, inteInfo)
        for (i, j) in zip(eachindex(res1), eachindex(res2))
            val = res1[i]
            res1[i] = bundler(val, res2[j])
        end
        res1
    else
        finalInfo = reformatOrbIntegral(inteInfo, op)
        evalOrbIntegralInfo!(finalInfo)
    end
end


function computeOrbDataIntegral(style::MultiBodyIntegral{D, C}, op::F, 
                                data::MultiOrbitalData{T, D, C}; 
                                lazyCompute::Boolean=True(), 
                                estimatorConfig::OptEstimatorConfig{T}=missing) where 
                               {T<:Real, C<:RealOrComplex{T}, D, F<:Summator}
    opStart = first(op.dresser)
    initInfo = initializeOrbIntegral(style, opStart, data, lazyCompute, estimatorConfig)
    weightInfo = getOrbCorePointers(initInfo, lazyCompute)
    coreInfo = reformatOrbIntegral(initInfo, weightInfo)
    infoData = evalOrbIntegralInfo!(op, coreInfo)
    coreInfo => infoData
end

function computeOrbDataIntegral(style::MultiBodyIntegral{D, C}, op::F, 
                                data::MultiOrbitalData{T, D, C}; 
                                lazyCompute::Boolean=True(), 
                                estimatorConfig::OptEstimatorConfig{T}=missing) where 
                               {T<:Real, C<:RealOrComplex{T}, D, F<:DirectOperator}
    initInfo = initializeOrbIntegral(style, op, data, lazyCompute, estimatorConfig)
    weightInfo = getOrbCorePointers(initInfo, lazyCompute)
    coreInfo = reformatOrbIntegral(initInfo, weightInfo)
    infoData = evalOrbIntegralInfo!(coreInfo)
    coreInfo => infoData
end


function computeOrbLayoutIntegral(op::DirectOperator, orbs::OrbBasisLayout{T, D}; 
                                  lazyCompute::Boolean=True(), 
                                  estimatorConfig::OptEstimatorConfig{T}=missing, 
                                  cache!Self::OptParamDataCache=initializeParamDataCache()
                                  ) where {T<:Real, D}
    orbsData = MultiOrbitalData(orbs, isParamIndependent(op); cache!Self)
    style = MultiBodyIntegral{D, getOutputType(orbsData), length(orbs)÷2}()
    computeOrbDataIntegral(style, op, orbsData; estimatorConfig, lazyCompute).second
end


function computeOrbVectorIntegral(::MultiBodyIntegral{D, T, N}, op::DirectOperator, 
                                  orbs::OrbBasisVector{T, D}; 
                                  lazyCompute::Boolean=True(), 
                                  estimatorConfig::OptEstimatorConfig{T}=missing, 
                                  cache!Self::OptParamDataCache=initializeParamDataCache()
                                  ) where {T<:Real, D, N}
    orbsData = MultiOrbitalData(orbs, isParamIndependent(op); cache!Self)
    style = MultiBodyIntegral{D, getOutputType(orbsData), N}()
    computeOrbDataIntegral(style, op, orbsData; estimatorConfig, lazyCompute).second
end