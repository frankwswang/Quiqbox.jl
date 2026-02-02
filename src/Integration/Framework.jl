const OrbIntLayoutInfo{N} = 
      Tuple{TypeUnion{DirectOperator}, NTuple{ N, NTuple{2, OrbitalCategory} }}

const OrbIntLayoutCache{T<:Real, C<:RealOrComplex{T}, N, 
                        M<:Union{OptionalCache{T}, OptionalCache{C}}} = 
      PseudoLRU{OrbIntLayoutInfo{N}, M}

const OptOrbIntLayoutCache{T<:Real, C<:RealOrComplex{T}, N} = 
      Union{EmptyDict{OrbIntLayoutInfo{N}, C}, OrbIntLayoutCache{T, C, N}}

const OptEstimatorConfig{T} = MissingOr{EstimatorConfig{T}}

const CONSTVAR_inteLayoutCacheScale::Int = 4

const CONSTVAR_inteValCacheSize::Int = 100


struct OrbitalIntegrationConfig{T<:Real, D, C<:RealOrComplex{T}, N, O<:DirectOperator, 
                                M<:OptOrbIntLayoutCache{T, C, N}, E<:OptEstimatorConfig{T}
                                } <: ConfigBox
    operator::O
    cache::M
    estimator::E

    function OrbitalIntegrationConfig(::MultiBodyIntegral{D, C, N}, operator::O, 
                                      cache::OptOrbIntLayoutCache{T, C, N}, 
                                      config::OptEstimatorConfig{T}) where 
                                     {D, T<:Real, C<:RealOrComplex{T}, N, O<:DirectOperator}
        new{T, D, C, N, O, typeof(cache), typeof(config)}(operator, cache, config)
    end
end

function OrbitalIntegrationConfig(style::MultiBodyIntegral{D, C, N}, operator::O, 
                                  caching::Boolean, config::OptEstimatorConfig{T}=missing
                                  ) where {D, T<:Real, C<:RealOrComplex{T}, N, 
                                           O<:DirectOperator}
    cache = if evalTypedData(caching)
        valueTypeBound = Union{OptionalCache{T}, OptionalCache{C}}
        partition = 2^(2N)
        maxCapacity = checkPositivity(CONSTVAR_inteLayoutCacheScale) * partition
        PseudoLRU{OrbIntLayoutInfo{N}, valueTypeBound}(maxCapacity, partition)
    else
        EmptyDict{OrbIntLayoutInfo{N}, C}()
    end
    OrbitalIntegrationConfig(style, operator, cache, config)
end

const OrbitalOverlapConfig{T<:Real, D, C<:RealOrComplex{T}, 
                           M<:OptOrbIntLayoutCache{T, C, 1}, E<:OptEstimatorConfig{T}} = 
      OrbitalIntegrationConfig{T, D, C, 1, OverlapSampler, M, E}

struct OneBodyIntegralValCache{C<:RealOrComplex} <: QueryBox{C}
    aa::PseudoLRU{N1N2Tuple{OneToIndex}, C}
    ab::PseudoLRU{N1N2Tuple{OneToIndex}, C}
    dimension::Int
    threshold::Int

    function OneBodyIntegralValCache(::OneBodyIntegral{D, C}, 
                                     threshold::Int=10CONSTVAR_inteValCacheSize) where 
                                    {D, C<:RealOrComplex}
        checkPositivity(threshold)
        maxPairNum = threshold * (threshold - 1)
        aaSector = PseudoLRU{N1N2Tuple{OneToIndex}, C}(threshold,  threshold )
        abSector = PseudoLRU{N1N2Tuple{OneToIndex}, C}(maxPairNum, maxPairNum)
        new{C}(aaSector, abSector, Int(D), threshold)
    end
end


struct TwoBodyIntegralValCache{C<:RealOrComplex} <: QueryBox{C}
    aaaa::PseudoLRU{N2N2Tuple{OneToIndex}, C}
    aabb::PseudoLRU{N2N2Tuple{OneToIndex}, C}
    half::PseudoLRU{N2N2Tuple{OneToIndex}, C} # aaxy or xyaa
    misc::PseudoLRU{N2N2Tuple{OneToIndex}, C} # abxy
    dimension::Int
    threshold::Int

    function TwoBodyIntegralValCache(::TwoBodyIntegral{D, C}, 
                                     threshold::Int=CONSTVAR_inteValCacheSize) where 
                                    {D, C<:RealOrComplex}
        checkPositivity(threshold)
        threshold2 = threshold * (threshold - 1)
        threshold3 = threshold * threshold2 * 2
        threshold4 = threshold^4 - threshold - threshold2 - threshold3
        aaaaSector = PseudoLRU{N2N2Tuple{OneToIndex}, C}(threshold,  threshold )
        aabbSector = PseudoLRU{N2N2Tuple{OneToIndex}, C}(threshold2, threshold2)
        halfSector = PseudoLRU{N2N2Tuple{OneToIndex}, C}(threshold3, threshold3)
        miscSector = PseudoLRU{N2N2Tuple{OneToIndex}, C}(threshold4, threshold4)
        new{C}(aaaaSector, aabbSector, halfSector, miscSector, Int(D), threshold)
    end
end

const FauxIntegralValCache{N, C<:RealOrComplex} = 
      EmptyDict{NTuple{N, NTuple{2, OneToIndex}}, C}

function genMultiBodyIntegralValCache(::MultiBodyIntegral{D, C, N}, caching::Boolean=True()
                                      ) where {D, C<:RealOrComplex, N}
    if !(N in (1, 2))
        throw(AssertionError("`$(MultiBodyIntegral{D, C, N})` is not supported."))
    end

    if evalTypedData(caching)
        if N == 1
            OneBodyIntegralValCache(OneBodyIntegral{D, C}())
        else
            TwoBodyIntegralValCache(TwoBodyIntegral{D, C}())
        end
    else
        EmptyDict{NTuple{N, NTuple{2, OneToIndex}}, C}()::FauxIntegralValCache{N, C}
    end
end

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


struct OrbCorePointer{D, C<:RealOrComplex}
    inner::MemoryPair{OneToIndex, C}

    function OrbCorePointer(::Count{D}, inner::MemoryPair{OneToIndex, C}) where 
                           {D, C<:RealOrComplex}
        new{D, C}(inner)
    end
end

function OrbCorePointer(orbPointer::PrimOrbPointer{D, C}, weight::C
                        ) where {D, C<:RealOrComplex}
    innerPair = MemoryPair(genMemory(orbPointer.inner), genMemory(weight))
    OrbCorePointer(Count(D), innerPair)
end

function OrbCorePointer(orbPointer::CompOrbPointer{D, C}, weight::AbstractArray{C, 1}
                        ) where {D, C<:RealOrComplex}
    innerPair = MemoryPair(map(x->x.inner, orbPointer.inner.left), extractMemory(weight))
    OrbCorePointer(Count(D), innerPair)
end


const OrbCorePointerLayout{D, C<:RealOrComplex} = N24Tuple{OrbCorePointer{D, C}}

const OrbCorePointerVector{D, C<:RealOrComplex} = Memory{OrbCorePointer{D, C}}

const OrbCoreWeightFormat{D, C<:RealOrComplex} = 
      Union{OrbCorePointerLayout{D, C}, OrbCorePointerVector{D, C}}


#> Potential interface for more simplified basis-set data
struct OrbitalCoreData{T<:Real, D, C<:RealOrComplex{T}, P<:OrbCoreWeightFormat{D, C}, 
                       F<:StashedShiftedField{T, D}}
    source::MemoryPair{F, OrbitalCategory}
    format::P
end


const OrbPointerData{T<:Real, D, C<:RealOrComplex{T}} = 
      Union{OrbitalCoreData{T, D, C}, MultiOrbitalData{T, D, C}}


struct OrbitalInteCoreInfo{T<:Real, D, C<:RealOrComplex{T}, N, O<:DirectOperator, 
                           M<:OrbitalIntegrationConfig{T, D, C, N, O}, 
                           V<:MultiBodyIntegralValCache{C}, F<:StashedShiftedField{T, D}
                           } <: FieldIntegralInfo{D, C, N}
    method::M
    memory::V
    source::MemoryPair{F, OrbitalCategory}

    OrbitalInteCoreInfo(method::M, memory::V, source::MemoryPair{F, OrbitalCategory}) where 
                       {T<:Real, D, C<:RealOrComplex{T}, O<:DirectOperator, 
                        M<:OrbitalIntegrationConfig{T, D, C, 1, O}, 
                        V<:OneBodyInteValCacheUnion{C}, F<:StashedShiftedField{T, D}} = 
    new{T, D, C, 1, O, M, V, F}(method, memory, source)

    OrbitalInteCoreInfo(method::M, memory::V, source::MemoryPair{F, OrbitalCategory}) where 
                       {T<:Real, D, C<:RealOrComplex{T}, O<:DirectOperator, 
                        M<:OrbitalIntegrationConfig{T, D, C, 2, O}, 
                        V<:TwoBodyInteValCacheUnion{C}, F<:StashedShiftedField{T, D}} = 
    new{T, D, C, 2, O, M, V, F}(method, memory, source)
end

const OneBodyOrbIntegralInfo{T<:Real, D, C<:RealOrComplex{T}, O<:DirectOperator, 
                             M<:OrbitalIntegrationConfig{T, D, C, 1, O}, 
                             V<:OneBodyInteValCacheUnion{C}, F<:StashedShiftedField{T, D}} = 
      OrbitalInteCoreInfo{T, D, C, 1, O, M, V, F}

const OrbitalOverlapInfo{T<:Real, D, C<:RealOrComplex{T}, M<:OrbitalOverlapConfig{T, D, C}, 
                         V<:OneBodyInteValCacheUnion{C}, F<:StashedShiftedField{T, D}} = 
      OneBodyOrbIntegralInfo{T, D, C, OverlapSampler, M, V, F}

const TwoBodyOrbIntegralInfo{T<:Real, D, C<:RealOrComplex{T}, O<:DirectOperator, 
                             M<:OrbitalIntegrationConfig{T, D, C, 2, O}, 
                             V<:TwoBodyInteValCacheUnion{C}, F<:StashedShiftedField{T, D}} = 
      OrbitalInteCoreInfo{T, D, C, 2, O, M, V, F}


function initializeOrbIntegral(::MultiBodyIntegral{D, C, N}, op::DirectOperator, 
                               orbCoreSource::MemoryPair{F, OrbitalCategory}, 
                               cachingMethod::Union{Boolean, OptOrbIntLayoutCache{T, C, N}}, 
                               cachingResult::Union{Boolean, MultiBodyIntegralValCache{C}}, 
                               estimatorConfig::OptEstimatorConfig{T}=missing) where 
                              {D, T<:Real, C<:RealOrComplex{T}, N, 
                               F<:StashedShiftedField{T, D}}
    inteStyle = MultiBodyIntegral{D, C, N}()

    methodConfig = OrbitalIntegrationConfig(inteStyle, op, cachingMethod, estimatorConfig)
    resultConfig = if !(cachingResult isa Boolean); cachingResult
                   else; genMultiBodyIntegralValCache(inteStyle, cachingResult) end

    OrbitalInteCoreInfo(methodConfig, resultConfig, orbCoreSource)
end

initializeOrbIntegral(::MultiBodyIntegral{D, C, N}, op::DirectOperator, 
                      orbCoreSource::MemoryPair{F, OrbitalCategory}, caching::Boolean, 
                      estimatorConfig::OptEstimatorConfig{T}=missing) where 
                     {D, T<:Real, C<:RealOrComplex{T}, N, F<:StashedShiftedField{T, D}} = 
initializeOrbIntegral(MultiBodyIntegral{D, C, N}(), op, orbCoreSource, caching, caching, 
                      estimatorConfig)


function reformatOrbIntegral(op::O, info::OrbitalInteCoreInfo{T, D, C, N}, 
                             activeCaching::Boolean, ::Count{NO}=Count(N)) where 
                            {T<:Real, D, C<:RealOrComplex{T}, NO, N, O<:DirectOperator}
    method = info.method
    methodCache = method.cache
    resultCache = info.memory
    activeMCache = !(methodCache isa EmptyDict)
    activeRCache = !(resultCache isa EmptyDict)
    equalPtclNum = NO == N
    sameIntegral = equalPtclNum && compareObj(op, method.operator)

    if evalTypedData(activeCaching)
        cachingMethod = (equalPtclNum && activeMCache) ? methodCache : True()
        cachingResult = (sameIntegral && activeRCache) ? resultCache : True()
    else
        cachingMethod = equalPtclNum ? methodCache : toBoolean(activeMCache)
        cachingResult = sameIntegral ? resultCache : toBoolean(activeRCache)
    end

    initializeOrbIntegral(MultiBodyIntegral{D, C, NO}(), op, info.source, 
                          cachingMethod, cachingResult, method.estimator)
end

initializeOrbNormalization(info::OrbitalInteCoreInfo, caching::Boolean) = 
reformatOrbIntegral(genOverlapSampler(), info, caching, Count(1))


#> Operator-orbital layout symmetry
#>> One-Body (i|O|j) symmetry across O: (i|O|j)' == (j|O|i) when i != j
getIntegralOpOrbSymmetry(::DirectOperator, ::N1N2Tuple{OrbitalCategory}) = false
getIntegralOpOrbSymmetry(::OverlapSampler, ::N1N2Tuple{OrbitalCategory}) = true
getIntegralOpOrbSymmetry(::MultipoleMomentSampler, ::N1N2Tuple{OrbitalCategory}) = true
function getIntegralOpOrbSymmetry(::DiagDirectionalDiffSampler, 
                                  layout::N1N2Tuple{OrbitalCategory})
    all(layout|>first) do c; (c == PrimGaussTypeOrb) end
end
getIntegralOpOrbSymmetry(::CoulombMultiPointSampler, ::N1N2Tuple{OrbitalCategory}) = true
#>> Two-body (ij|O|kl) symmetry between ij: (ij|O|kl)' == (ji|O|kl) when  i != j
#>> Two-body (ij|O|kl) symmetry between kl: (ij|O|kl)' == (ij|O|lk) when  k != l
#>> Two-body (ij|O|kl) symmetry across O:   (ij|O|kl)  == (kl|O|ij) when ij != kl
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


#>> `indexSymmetry` should be index layout symmetry -> find correct layout sector
#>> `layoutSymmetry` should be operator-orbital layout symmetry -> reuse integral result
getInteValCacheSector(cache::FauxIntegralValCache{1}, ::Bool) = itself(cache)

getInteValCacheSector(cache::FauxIntegralValCache{2}, ::NTuple{3, Bool}) = itself(cache)

function getInteValCacheSector(cache::OneBodyIntegralValCache{C}, 
                               indexSymmetry::Bool) where {C<:RealOrComplex}
    ifelse(indexSymmetry, cache.aa, cache.ab)::PseudoLRU{N1N2Tuple{OneToIndex}, C}
end

function getInteValCacheSector(cache::TwoBodyIntegralValCache{C}, 
                               indexSymmetry::NTuple{3, Bool}) where {C<:RealOrComplex}
    symmetryL, symmetryR, _ = indexSymmetry

    if all(indexSymmetry)
        cache.aaaa
    elseif symmetryL && symmetryR
        cache.aabb
    elseif symmetryL
        cache.half #> aaxy
    elseif symmetryR
        cache.half #> xyaa
    else
        cache.misc #> abxy
    end::PseudoLRU{N2N2Tuple{OneToIndex}, C}
end


#> Reordered-integral-index-layout => whether-need-to-needToConjugate
#>> One-body
function getIntegralIndexPair(key::N1N2Tuple{OneToIndex}, permuteControl::Bool)
    (i, j), = key
    if permuteControl && i < j
        Pair(tuple((j, i)), true)
    else
        Pair(key, false)
    end #>> Resulting order: `i >= j`
end

#>> Two-body
function getIntegralIndexPair(key::N2N2Tuple{OneToIndex}, permuteControl::NTuple{3, Bool})
    needToConjugate::Bool = false
    permuteL, permuteR, permuteLR = permuteControl

    (i, j), partR = key
    if permuteL && i < j
        key = ((j, i), partR)
        needToConjugate = !needToConjugate
    end

    partL, (k, l) = key
    if permuteR && k < l
        key = (partL, (l, k))
        needToConjugate = !needToConjugate
    end

    partL, partR = key
    if permuteLR && partL < partR
        key = (partR, partL)
    end

    Pair(key, needToConjugate) #>> Resulting order: `i >= j; k >= l; (i, j) >= (k, l)`
end


struct OrbPointerLayoutConfig{N} <: ConfigBox
    idx::NTuple{N, NTuple{2, OneToIndex}}
    orb::NTuple{N, NTuple{2, OrbitalCategory}}

    function OrbPointerLayoutConfig(idxLayout::NonEmptyTuple{NTuple{2, OneToIndex}, N}, 
                                    orbLayout::NonEmptyTuple{NTuple{2, OrbitalCategory}, N}
                                    ) where {N}
        new{N+1}(idxLayout, orbLayout)
    end
end

function OrbPointerLayoutConfig(categorySource::AbstractVector{OrbitalCategory}, 
                                idxLayout::NonEmptyTuple{NTuple{2, OneToIndex}})
    orbLayout = getPairTuple(categorySource, idxLayout)
    OrbPointerLayoutConfig(idxLayout, orbLayout)
end

function prepareInteValCache(op::DirectOperator, layoutInfoOld::OrbPointerLayoutConfig, 
                             cache::MultiBodyIntegralValCache)
    idxLayoutOld = layoutInfoOld.idx
    orbLayoutOld = layoutInfoOld.orb
    indexSymmOld = getIntegralIndexSymmetry(idxLayoutOld)
    opOrbSymmOld = getIntegralOpOrbSymmetry(op, orbLayoutOld)
    inteCacheSec = getInteValCacheSector(cache, indexSymmOld)
    permuteControl = .!(indexSymmOld) .&& opOrbSymmOld
    idxLayoutNew, needToConjugate = getIntegralIndexPair(idxLayoutOld, permuteControl)

    (idxLayoutNew => inteCacheSec), needToConjugate
end

function prepareOrbPointerLayoutCache(info::OrbitalInteCoreInfo{T, D, C, N}, 
                                      idxLayoutOld::NTuple{N, NTuple{2, OneToIndex}}
                                      ) where {T<:Real, D, C<:RealOrComplex{T}, N}
    operator = info.method.operator
    fieldCateSource = info.source.right
    layoutInfoOld = OrbPointerLayoutConfig(fieldCateSource, idxLayoutOld)
    config, needToConjugate = prepareInteValCache(operator, layoutInfoOld, info.memory)
    layoutInfoNew = OrbPointerLayoutConfig(fieldCateSource, config.first)
    (layoutInfoNew => needToConjugate), config.second
end


struct NumericalIntegration{D, C<:RealOrComplex, N} <: IntegralStyle end
struct GaussTypeIntegration{D, C<:RealOrComplex, N} <: IntegralStyle end

const IntegrationMethod{D, C<:RealOrComplex, N} = 
      Union{NumericalIntegration{D, C, N}, GaussTypeIntegration{D, C, N}}

#>> Consistent with the reverse-ordered index layout such that the left index contributes 
#>> less than the right index to incrementing the linear index. This results in a better 
#>> locality of the hash index.
function indexLayoutHash(layout::N1N2Tuple{OneToIndex}, orbCoreNum::Int)
    (i, j), = layout

    if i == j
        i.idx
    else
        linearIds = LinearIndices((orbCoreNum, orbCoreNum))
        linearIds[i.idx, j.idx]
    end |> UInt
end

function indexLayoutHash(layout::N2N2Tuple{OneToIndex}, orbCoreNum::Int)
    (i, j), (k, l) = L, R = layout
    bl1 = (i == j)
    bl2 = (k == l)
    bl3 = (L == R)

    if bl1 && bl2 && bl3
        i.idx
    elseif bl1 && bl2
        linearIds = LinearIndices((orbCoreNum, orbCoreNum))
        linearIds[i.idx, k.idx]
    else
        linearIds = LinearIndices((orbCoreNum, orbCoreNum, orbCoreNum, orbCoreNum))
        linearIds[i.idx, j.idx, k.idx, l.idx]
    end |> UInt
end

function getIntegralValue!(info::OrbitalInteCoreInfo{T, D, C, N}, 
                           indexLayout::NTuple{N, NTuple{2, OneToIndex}}) where 
                          {T<:Real, D, C<:RealOrComplex{T}, N}
    (layoutInfo, needToConjugate), sector = prepareOrbPointerLayoutCache(info, indexLayout)
    reorderedIdsKey = layoutInfo.idx
    orbSource = info.source.left
    orbCoreNum = length(orbSource)

    #> No noticeable performance hit from the closure
    val = get!(sector, reorderedIdsKey, RPartial( indexLayoutHash, (orbCoreNum,) )) do
        inteConfig = info.method
        component = configureIntegration!(inteConfig, layoutInfo.orb)
        switcher = genInteMethodSwitcher(Count(D), C, Count(N), component)
        evaluateIntegralCore!(switcher::IntegrationMethod{D, C, N}, inteConfig.operator, 
                              component, orbSource, reorderedIdsKey)::C
    end

    ifelse(needToConjugate, conj(val), val)::C
end


function configureIntegration!(config::OrbitalIntegrationConfig{T, D, C, N}, 
                               layout::NTuple{N, NTuple{2, OrbitalCategory}}) where 
                              {T<:Real, D, C<:RealOrComplex{T}, N}
    if all(x->all(isequal(PrimGaussTypeOrb), x), layout)
        getInteComponentCore!(Val(PrimGaussTypeOrb), config)
    else
        getInteComponentCore!(Val(missing), config)
    end
end

#> Adaptive integration interface 1 (Also need if-else branch in `configureIntegration!`)
function getInteComponentCore!(::Val, config::OrbitalIntegrationConfig)
    config.estimator
end

#> Adaptive integration interface 2
function genInteMethodSwitcher(::Count{D}, ::Type{C}, ::Count{N}, ::Any) where 
                              {D, C<:RealOrComplex, N}
    NumericalIntegration{D, C, N}()
end

#> Adaptive integration interface 3
function evaluateIntegralCore!(::NumericalIntegration{D, C, N}, op::DirectOperator, 
                               component::Any, source::AbstractVector{F}, 
                               layout::NTuple{N, NTuple{2, OneToIndex}}) where 
                              {T, D, C<:RealOrComplex{T}, N, F<:StashedShiftedField{T, D}}
    formattedOp = TypedOperator(op, C)
    fieldLayout = getPairTuple(source, layout)
    convert(C, estimateOrbIntegral(formattedOp, fieldLayout, component))
end


function getOrbLayoutIntegralCore!(inteInfo::OneBodyOrbIntegralInfo{T, D, C}, 
                                   ptrLayout::NTuple{2, OrbCorePointer{D, C}}) where 
                                  {T<:Real, D, C<:RealOrComplex{T}}
    ptrL, ptrR = ptrLayout
    res = zero(C)

    for eleR in ptrR.inner
        idxR, weightR = eleR

        for eleL in ptrL.inner
            idxL, weightL = eleL

            weightProd = conj(weightL) * weightR
            res += getIntegralValue!(inteInfo, ((idxL, idxR),)) * weightProd
        end
    end

    res::C
end


function getOrbLayoutIntegralCore!(inteInfo::TwoBodyOrbIntegralInfo{T, D, C}, 
                                   ptrLayout::NTuple{4, OrbCorePointer{D, C}}) where 
                                  {T<:Real, D, C<:RealOrComplex{T}}
    ptrL1, ptrR1, ptrL2, ptrR2 = ptrLayout
    res = zero(C)

    for eleR2 in ptrR2.inner
        idxR2, weightR2 = eleR2

        for eleL2 in ptrL2.inner
            idxL2, weightL2 = eleL2
            weightProd2 = conj(weightL2) * weightR2

            for eleR1 in ptrR1.inner
                idxR1, weightR1 = eleR1

                for eleL1 in ptrL1.inner
                    idxL1, weightL1 = eleL1
                    formattedIndex = ((idxL1, idxR1), (idxL2, idxR2))

                    weightProd = conj(weightL1) * weightR1 * weightProd2
                    res += getIntegralValue!(inteInfo, formattedIndex) * weightProd
                end
            end
        end
    end

    res::C
end


function getOrbInteTensorSymmetry(::MultiBodyIntegral{D, C, N}, op::DirectOperator, 
                                  orbCate::OrbitalCategory) where 
                                 {D, T, C<:RealOrComplex{T}, N}
    layout = ntuple(_->(orbCate, orbCate), Val(N))
    getIntegralOpOrbSymmetry(op, layout)
end

function setInteTensorEntry!(tensor::AbstractArray{C, N}, val::C, 
                             idxTuple::NTuple{N, OneToIndex}) where {C<:RealOrComplex, N}
    ids = shiftAxialIndex(tensor, idxTuple)
    tensor[ids...] = val
end

function evalSetInteTensorEntry!(tensor::AbstractArray{C, N}, 
                                 inteInfo::OrbitalInteCoreInfo{T, D, C}, 
                                 ptrVector::OrbCorePointerVector{D, C}, 
                                 idxTpl::NTuple{N, OneToIndex}) where 
                                {T<:Real, D, C<:RealOrComplex{T}, N}
    ptrTpl = getTuple(ptrVector, idxTpl)
    val = getOrbLayoutIntegralCore!(inteInfo, ptrTpl)
    setInteTensorEntry!(tensor, val, idxTpl)
    val
end

function getOrbVectorIntegralCore!(inteInfo::OneBodyOrbIntegralInfo{T, D, C}, 
                                   ptrVector::OrbCorePointerVector{D, C}
                                   ) where {T<:Real, D, C<:RealOrComplex{T}}
    len = length(ptrVector)
    op = inteInfo.method.operator
    style = OneBodyIntegral{D, C}()
    tensor = Array{C}(undef, (len, len))
    orbCate = (getOrbitalCategory∘eltype)(inteInfo.source.left)
    symmetry = getOrbInteTensorSymmetry(style, op, orbCate)

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

# #> Potential interface for cross-basis integral computation
# function getOrbVectorIntegralCore!(inteInfo::OneBodyOrbIntegralInfo{T, D, C}, 
#                                    ptrVector::OrbCorePointerVector{D, C}, 
#                                    cutIdx::OneToIndex #> separate the basis set into two subsets and only compute integrals between those two subsets
#                                    ) where {T<:Real, D, C<:RealOrComplex{T}}
#     len = length(ptrVector)
#     op = inteInfo.method.operator
#     style = OneBodyIntegral{D, C}()
#     tensor = Array{C}(undef, (len, len))
#     orbCate = (getOrbitalCategory∘eltype)(inteInfo.source.left)
#     symmetry = getOrbInteTensorSymmetry(style, op, orbCate)

#     if symmetry
#         for n in 1:symmetric2DArrEleNum(len)
#             ijIdx = OneToIndex.(n|>convertIndex1DtoTri2D)
#             ijVal = evalSetInteTensorEntry!(tensor, inteInfo, ptrVector, ijIdx)
#             if !getIntegralIndexSymmetry(ijIdx|>tuple)
#                 setInteTensorEntry!(tensor, conj(ijVal), reverse(ijIdx))
#             end
#         end
#     else
#         idxRange = OneToRange(len)
#         for j in idxRange, i in idxRange
#             evalSetInteTensorEntry!(tensor, inteInfo, ptrVector, (i, j))
#         end
#     end

#     tensor
# end


function getOrbVectorIntegralCore!(inteInfo::TwoBodyOrbIntegralInfo{T, D, C}, 
                                   ptrVector::OrbCorePointerVector{D, C}
                                   ) where {T<:Real, D, C<:RealOrComplex{T}}
    len = length(ptrVector)
    op = inteInfo.method.operator
    style = TwoBodyIntegral{D, C}()
    tensor = Array{C}(undef, (len, len, len, len))
    orbCate = (getOrbitalCategory∘eltype)(inteInfo.source.left)
    symL, symR, symO = getOrbInteTensorSymmetry(style, op, orbCate)

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


function getOrbCoreOverlap!(info::OrbitalOverlapInfo{T, D, C}, 
                            orbPointerL::PrimOrbPointer{D, C}, 
                            orbPointerR::PrimOrbPointer{D, C}=orbPointerL
                            ) where {T<:Real, D, C<:RealOrComplex{T}}
    tempPtrPair = lazyMap((orbPointerL, orbPointerR)) do orbPointer
        ptrCore = MemoryPair(genMemory(orbPointer.inner), (genMemory∘one)(C))
        OrbCorePointer(Count(D), ptrCore)
    end
    getOrbLayoutIntegralCore!(info, tempPtrPair)
end

function getOrbCoreOverlap!(info::OrbitalOverlapInfo{T, D, C}, 
                            orbPointer::CompOrbPointer{D, C}
                            ) where {T<:Real, D, C<:RealOrComplex{T}}
    primPtrs = orbPointer.inner.left
    overlapSum = (fill∘zero)(C)

    normalizedWeights = map(orbPointer.inner) do (primPtr, weightOld)
        weightNew = weightOld
        diagOverlap = conj(weightOld) * weightOld
        coreOverlap = getOrbCoreOverlap!(info, primPtr)

        if primPtr.renormalize
            weightNew *= absSqrtInv(coreOverlap)
        else
            diagOverlap *= coreOverlap
        end

        overlapSum[] += diagOverlap::C

        weightNew::C
    end

    for n in 1:symmetric2DArrEleNum(length(orbPointer.inner) - 1)
        i, j = convertIndex1DtoTri2D(n)
        idxPair = OneToIndex.((i, j+1))
        weightL, weightR = getTuple(normalizedWeights, idxPair)
        weightProd = conj(weightL) * weightR
        primPtrPair = getTuple(primPtrs, idxPair)
        offDiagOverlap = getOrbCoreOverlap!(info, primPtrPair...) * weightProd
        overlapSum[] += offDiagOverlap + conj(offDiagOverlap)
    end

    overlapSum[]::C
end


function buildOrbCoreWeight!(normInfo::OrbitalOverlapInfo{T, D, C}, 
                             orbPointer::PrimOrbPointer{D, C}) where 
                            {T<:Real, D, C<:RealOrComplex{T}}
    if orbPointer.renormalize
        absSqrtInv(getOrbCoreOverlap!(normInfo, orbPointer))::C
    else
        one(C)
    end
end

function buildOrbCoreWeight!(normInfo::OrbitalOverlapInfo{T, D, C}, 
                             orbPointer::CompOrbPointer{D, C}) where 
                            {T<:Real, D, C<:RealOrComplex{T}}
    primPtrs = orbPointer.inner.left
    weight = ShapedMemory(orbPointer.inner.right)

    for (i, primPtr) in zip(eachindex(weight), primPtrs)
        weight[i] *= buildOrbCoreWeight!(normInfo, primPtr)
    end

    if orbPointer.renormalize
        weight .*= absSqrtInv(getOrbCoreOverlap!(normInfo, orbPointer))
    end

    weight
end


function checkInteLayoutLength(::Count{L}, layoutLength::Int, layoutName::Symbol) where {L}
    if layoutLength != L
        throw(AssertionError("The length of `$layoutName` must equal `$L` to match the "*
                             "layout of the integral."))
    end
end


function getOrbCorePointers(inteInfo::OrbitalInteCoreInfo{T, D, C, N}, 
                            format::Pair{OrbitalPointerBox{D, C}, True}, 
                            lazyNormalize::Boolean) where 
                           {T<:Real, D, C<:RealOrComplex{T}, N}
    ptrs = format.first
    nPtr = checkEmptiness(ptrs, Symbol("format.first"))
    checkInteLayoutLength(Count(2N), nPtr, Symbol("format.first"))

    let normInfo=initializeOrbNormalization(inteInfo, lazyNormalize)
        map(ntuple( itself, Val(2N) )) do idx
            ptr = getindex(ptrs, idx)
            weightHolder = buildOrbCoreWeight!(normInfo, ptr)
            OrbCorePointer(ptr, weightHolder)
        end
    end
end

function getOrbCorePointers(inteInfo::OrbitalInteCoreInfo{T, D, C, N}, 
                            format::Pair{OrbitalPointerBox{D, C}, False}, 
                            lazyNormalize::Boolean) where 
                           {T<:Real, D, C<:RealOrComplex{T}, N}
    ptrs = format.first
    nPtr = checkEmptiness(ptrs, Symbol("format.first"))
    res = Memory{OrbCorePointer{D, C}}(undef, nPtr)

    let normInfo=initializeOrbNormalization(inteInfo, lazyNormalize)
        for (i, ptr) in zip(eachindex(res), ptrs)
            weightHolder = buildOrbCoreWeight!(normInfo, ptr)
            res[i] = OrbCorePointer(ptr, weightHolder)
        end
    end

    res
end

function getOrbCorePointers(::OrbitalInteCoreInfo{T, D, C, N}, 
                            format::OrbCoreWeightFormat{D, C}, ::Boolean
                            ) where {T<:Real, D, C<:RealOrComplex{T}, N}
    nPtr = checkEmptiness(format, :format)
    format isa Tuple && checkInteLayoutLength(Count(2N), nPtr, Symbol("format"))
    format
end


function evalOrbIntegralInfo!(inteInfo::OrbitalInteCoreInfo{T, D, C, N}, 
                              weightInfo::OrbCorePointerLayout{D, C}) where 
                             {T<:Real, D, C<:RealOrComplex{T}, N}
    checkInteLayoutLength(Count(2N), length(weightInfo), :weightInfo)
    getOrbLayoutIntegralCore!(inteInfo, weightInfo)
end

function evalOrbIntegralInfo!(inteInfo::OrbitalInteCoreInfo{T, D, C, N}, 
                              weightInfo::OrbCorePointerVector{D, C}) where 
                             {T<:Real, D, C<:RealOrComplex{T}, N}
    checkEmptiness(weightInfo, :weightInfo)
    getOrbVectorIntegralCore!(inteInfo, weightInfo)
end

function evalOrbIntegralInfo!(op::O, inteInfo::OrbitalInteCoreInfo{T, D, C}, 
                              weightInfo::OrbCoreWeightFormat{D, C}) where 
                             {T<:Real, D, C<:RealOrComplex{T}, O<:DirectOperator}
    if O <: Summator
        op1, op2 = op.dresser
        bundler = op.bundler
        res1 = evalOrbIntegralInfo!(op1, inteInfo, weightInfo)
        res2 = evalOrbIntegralInfo!(op2, inteInfo, weightInfo)
        if weightInfo isa AbstractVector
            for (i, j) in zip(eachindex(res1), eachindex(res2))
                val = res1[i]
                res1[i] = bundler(val, res2[j])
            end
            res1
        else
            bundler(res1, res2)
        end
    else
        finalInfo = reformatOrbIntegral(op, inteInfo, False())
        evalOrbIntegralInfo!(finalInfo, weightInfo)
    end
end

struct OrbitalSetIntegralInfo{T<:Real, D, C<:RealOrComplex{T}, N, 
                              M<:OrbitalInteCoreInfo{T, D, C, N}
                              } <: FieldIntegralInfo{D, C, N}
    config::M
    weight::OrbCorePointerVector{D, C}
    memory::LRU{NTuple{N, NTuple{2, OneToIndex}}, C}

    function OrbitalSetIntegralInfo(coreInfo::M, weightInfo::OrbCorePointerVector{D, C}, 
                                    maxSize::Int=(10^N)*CONSTVAR_inteValCacheSize) where 
                                   {T<:Real, D, C<:RealOrComplex{T}, N, 
                                    M<:OrbitalInteCoreInfo{T, D, C, N}}
        memory = LRU{NTuple{N, NTuple{2, OneToIndex}}, C}(maxsize=maxSize)
        new{T, D, C, N, M}(coreInfo, weightInfo, memory)
    end
end

const OrbitalSetOverlapInfo{T, D, C, M<:OrbitalOverlapInfo{T, D, C}} = 
      OrbitalSetIntegralInfo{T, D, C, 1, M}

function evalOrbIntegralInfo!(basisInteInfo::OrbitalSetIntegralInfo{T, D, C}) where 
                             {T<:Real, D, C<:RealOrComplex{T}}
    evalOrbIntegralInfo!(basisInteInfo.config, basisInteInfo.weight)
end #! Future interface extension to utilize .memory

function evalOrbIntegralInfo!(op::O, basisInteInfo::OrbitalSetIntegralInfo{T, D, C}) where 
                             {T<:Real, D, C<:RealOrComplex{T}, O<:Summator}
    evalOrbIntegralInfo!(op, basisInteInfo.config, basisInteInfo.weight)
end #! Future interface extension to utilize .memory


function computeOrbDataIntegral(style::MultiBodyIntegral{D, C}, op::O, 
                                data::OrbPointerData{T, D, C}; 
                                lazyCompute::Boolean=True(), 
                                estimatorConfig::OptEstimatorConfig{T}=missing) where 
                               {T<:Real, C<:RealOrComplex{T}, D, O<:Summator}
    head = first(op.dresser)
    initInfo = initializeOrbIntegral(style, head, data.source, lazyCompute, estimatorConfig)
    weightInfo = getOrbCorePointers(initInfo, data.format, lazyCompute)
    infoData = evalOrbIntegralInfo!(op, initInfo, weightInfo)
    (initInfo, weightInfo) => infoData
end

function computeOrbDataIntegral(style::MultiBodyIntegral{D, C}, op::O, 
                                data::OrbPointerData{T, D, C}; 
                                lazyCompute::Boolean=True(), 
                                estimatorConfig::OptEstimatorConfig{T}=missing) where 
                               {T<:Real, C<:RealOrComplex{T}, D, O<:DirectOperator}
    initInfo = initializeOrbIntegral(style, op, data.source, lazyCompute, estimatorConfig)
    weightInfo = getOrbCorePointers(initInfo, data.format, lazyCompute)
    infoData = evalOrbIntegralInfo!(initInfo, weightInfo)
    (initInfo, weightInfo) => infoData
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