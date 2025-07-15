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

    function OrbitalIntegrationConfig(::S, operator::F, caching::Boolean, config::E=missing
                                      ) where {D, T<:Real, C<:RealOrComplex{T}, N, 
                                               S<:MultiBodyIntegral{D, C, N}, 
                                               F<:DirectOperator, E<:OptEstimatorConfig{T}}
        cache = if evalTypedData(caching)
            valueTypeBound = Union{OptionalCache{T}, OptionalCache{C}}
            LRU{OrbIntLayoutInfo{N}, valueTypeBound}(maxsize=20)
        else
            EmptyDict{OrbIntLayoutInfo{N}, C}()
        end
        new{T, D, C, N, F, typeof(cache), E}(operator, cache, config)
    end
end

const OrbitalOverlapConfig{T<:Real, D, C<:RealOrComplex{T}, 
                           M<:OptOrbIntLayoutCache{T, C, 1}, E<:OptEstimatorConfig{T}} = 
      OrbitalIntegrationConfig{T, D, C, 1, OverlapSampler, M, E}

struct OneBodyIntegralValCache{C<:RealOrComplex} <: QueryBox{C}
    aa::LRU{ Tuple{   OneToIndex}, C}
    ab::LRU{NTuple{2, OneToIndex}, C}
    dimension::Int
    threshold::Int

    function OneBodyIntegralValCache(::OneBodyIntegral{D, C}, 
                                     threshold::Int=1024) where {D, C<:RealOrComplex}
        checkPositivity(threshold)
        maxPairNum = threshold * (threshold - 1)
        aaSector = LRU{ Tuple{   OneToIndex}, C}(maxsize=threshold )
        abSector = LRU{NTuple{2, OneToIndex}, C}(maxsize=maxPairNum)
        new{C}(aaSector, abSector, Int(D), threshold)
    end
end


struct TwoBodyIntegralValCache{C<:RealOrComplex} <: QueryBox{C}
    aaaa::LRU{ Tuple{   OneToIndex}, C}
    aabb::LRU{NTuple{2, OneToIndex}, C}
    half::LRU{NTuple{4, OneToIndex}, C} # aaxy or xyaa
    misc::LRU{NTuple{4, OneToIndex}, C} # abxy
    dimension::Int
    threshold::Int

    function TwoBodyIntegralValCache(::TwoBodyIntegral{D, C}, 
                                     threshold::Int=128) where {D, C<:RealOrComplex}
        checkPositivity(threshold)
        threshold2 = threshold * (threshold - 1)
        threshold3 = threshold * threshold2 * 2
        threshold4 = threshold^4 - threshold - threshold2 - threshold3
        aaaaSector = LRU{ Tuple{   OneToIndex}, C}(maxsize=threshold )
        aabbSector = LRU{NTuple{2, OneToIndex}, C}(maxsize=threshold2)
        halfSector = LRU{NTuple{4, OneToIndex}, C}(maxsize=threshold3)
        miscSector = LRU{NTuple{4, OneToIndex}, C}(maxsize=threshold4)
        new{C}(aaaaSector, aabbSector, halfSector, miscSector, Int(D), threshold)
    end
end


const FauxIntegralValCache{C<:RealOrComplex} = EmptyDict{Tuple{Vararg{OneToIndex}}, C}

genFauxIntegralValCache(::Type{C}) where {C<:RealOrComplex} = 
EmptyDict{Tuple{Vararg{OneToIndex}}, C}()

const OneBodyInteValCacheUnion{C<:RealOrComplex} = Union{
    OneBodyIntegralValCache{C}, 
    FauxIntegralValCache{C}
}

const TwoBodyInteValCacheUnion{C<:RealOrComplex} = Union{ 
    FauxIntegralValCache{C}
}

const MultiBodyIntegralValCache{C<:RealOrComplex} = Union{
    OneBodyInteValCacheUnion{C}, 
    TwoBodyInteValCacheUnion{C}
}
abstract type FieldIntegralInfo{D, C, N} <: ConfigBox end

struct OrbitalIntegralInfo{T<:Real, D, C<:RealOrComplex{T}, N, F<:DirectOperator, 
                           M<:OrbitalIntegrationConfig{T, D, C, N, F}, 
                           V<:MultiBodyIntegralValCache{C}, P<:MultiOrbitalData{T, D}
                           } <: FieldIntegralInfo{D, C, N}
    method::M
    memory::V
    basis::P
end

const OneBodyOrbIntegralInfo{T<:Real, D, C<:RealOrComplex{T}, F<:DirectOperator, 
                             M<:OrbitalIntegrationConfig{T, D, C, 1, F}, 
                             V<:OneBodyInteValCacheUnion{C}, P<:MultiOrbitalData{T, D}} = 
      OrbitalIntegralInfo{T, D, C, 1, F, M, V, P}

const OrbitalOverlapInfo{T<:Real, D, C<:RealOrComplex{T}, M<:OrbitalOverlapConfig{T, D, C}, 
                         V<:OneBodyInteValCacheUnion{C}, P<:MultiOrbitalData{T, D}} = 
      OneBodyOrbIntegralInfo{T, D, C, OverlapSampler, M, V, P}

const OrbitalLayoutInteInfo{T<:Real, D, C<:RealOrComplex{T}, N, F<:DirectOperator, 
                            M<:OrbitalIntegrationConfig{T, D, C, N, F}, 
                            V<:MultiBodyIntegralValCache{C}, P<:OrbitalLayoutData{T, D}} = 
      OrbitalIntegralInfo{T, D, C, N, F, M, V, P}

const OrbitalVectorInteInfo{T<:Real, D, C<:RealOrComplex{T}, N, F<:DirectOperator, 
                            M<:OrbitalIntegrationConfig{T, D, C, N, F}, 
                            V<:MultiBodyIntegralValCache{C}, P<:OrbitalVectorData{T, D}} = 
      OrbitalIntegralInfo{T, D, C, N, F, M, V, P}


function initializeOrbIntegral(::OneBodyIntegral{D, C}, op::DirectOperator, 
                               data::MultiOrbitalData{T, D, C}, caching::Boolean, 
                               estimatorConfig::OptEstimatorConfig{T}=missing) where 
                              {T<:Real, D, C<:RealOrComplex{T}}
    inteStyle = OneBodyIntegral{D, C}()
    methodConfig = OrbitalIntegrationConfig(inteStyle, op, caching, estimatorConfig)
    resultConfig = OneBodyIntegralValCache(inteStyle)
    OrbitalIntegralInfo(methodConfig, resultConfig, data)
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
        normMemory = genFauxIntegralValCache(C)
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

const SesquiOrbCore{T<:Real, D} = N1N2Tuple{StashedShiftedField{T, D}}
const OneBodyOrbCorePair{T<:Real, D} = Pair{<:SesquiOrbCore{T, D}, N1N2Tuple{OneToIndex}}


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
#> Two-body (ij|O|kl) symmetry between ij: (ij|O|kl)' == (ji|O|kl) when  i != j
#> Two-body (ij|O|kl) symmetry between kl: (ij|O|kl)' == (ij|O|lk) when  k != l
#> Two-body (ij|O|kl) symmetry across O:   (ij|O|kl)  == (kl|O|ij) when ij != kl
getIntegralOpOrbSymmetry(::DirectOperator, ::N2N2Tuple{OrbitalCategory}) = 
(false, false, false)

#>> Index layout symmetry
getIntegralIndexSymmetry((part,)::N1N2Tuple{OneToIndex}) = first(part) == last(part)
function getIntegralIndexSymmetry((partL, partR)::N2N2Tuple{OneToIndex})
    idxL1, idxR1 = partL
    idxL2, idxR2 = partR
    (idxL1==idxR1, idxL2==idxR2, partL==partR)
end

function prepareInteValCache(cache::FauxIntegralValCache, layout::N1N2Tuple{OneToIndex})
    first(layout)=>cache, getIntegralIndexSymmetry(layout)
end

function prepareInteValCache(cache::FauxIntegralValCache, layout::N2N2Tuple{OneToIndex})
    partL, partR = layout
    (partL..., partR...)=>cache, getIntegralIndexSymmetry(layout)
end

function prepareInteValCache(cache::OneBodyIntegralValCache, layout::N1N2Tuple{OneToIndex})
    idxL, idxR = first(layout)
    indexSymmetry = getIntegralIndexSymmetry(layout)
    keySectorPair = indexSymmetry ? (idxL,)=>cache.aa : (idxL, idxR)=>cache.ab
    keySectorPair, indexSymmetry
end

function prepareInteValCache(cache::TwoBodyIntegralValCache, layout::N2N2Tuple{OneToIndex})
    (idxL1, idxR1), (idxL2, idxR2) = layout
    symmetryL, symmetryR, _ = indexSymmetry = getIntegralIndexSymmetry(layout)

    keySectorPair = if all(indexSymmetry)
        (idxL1,                    )=>cache.aaaa
    elseif symmetryL && symmetryR
        (idxL1, idxL2              )=>cache.aabb
    elseif symmetryL
        (idxL1, idxR1, idxL2, idxR2)=>cache.half #> aaxy
    elseif symmetryR
        (idxL1, idxR1, idxL2, idxR2)=>cache.half #> xyaa
    else
        (idxL1, idxR1, idxL2, idxR2)=>cache.misc #> abxy
    end

    keySectorPair, indexSymmetry
end

#> aa, aaaa
function formatIntegralCacheKey(key::Tuple{OneToIndex}, ::Union{Bool, NTuple{3, Bool}})
    key => false # orderedKey => needToConjugate
end
#> ab
function formatIntegralCacheKey(key::NTuple{2, OneToIndex}, permuteControl::Bool)
    i, j = key
    if permuteControl && i > j
        (j, i) => true
    else
        key => false
    end
end
#> aabb
function formatIntegralCacheKey(key::NTuple{2, OneToIndex}, permuteControl::NTuple{3, Bool})
    i, j = key
    if last(permuteControl) && (i > j)
        (j, i) => false
    else
        key => false
    end
end
#> aaxy, xyaa, abxy
function formatIntegralCacheKey(key::NTuple{4, OneToIndex}, permuteControl::NTuple{3, Bool})
    needToConjugate = false
    permuteL, permuteR, permuteLR = permuteControl

    i, j, k, l = key
    if permuteL && i > j
        key = (j, i, k, l)
        needToConjugate = !needToConjugate
    end

    i, j, k, l = key
    if permuteR && k > l
        key = (i, j, l, k)
        needToConjugate = !needToConjugate
    end

    i, j, k, l = key
    if permuteLR && (i, j) > (k, l)
        key = (k, l, i, j)
    end

    key => needToConjugate
end
#> `indexSymmetry` should be index based symmetry -> to simplify layout
#> `layoutSymmetry` should be operator--orbital based symmetry -> to reuse result
function getIntegralValue!(cache::OneBodyInteValCacheUnion{C}, 
                           method::OrbitalIntegrationConfig{T, D, C}, 
                           pair::OneBodyOrbCorePair{T, D}) where 
                          {T<:Real, D, C<:RealOrComplex{T}}
    fieldLayout, indexLayout = pair
    (key, sector), indexSymmetry = prepareInteValCache(cache, indexLayout)
    categoryLayout = genOrbCategoryLayout(fieldLayout)
    layoutSymmetry = getIntegralOpOrbSymmetry(method.operator, categoryLayout)
    permuteControl = .!(indexSymmetry) .&& layoutSymmetry
    orderedKey, needToConjugate = formatIntegralCacheKey(key, permuteControl)
    res = get!(sector, orderedKey) do
        inteVal = evaluateIntegral!(method, fieldLayout)
        needToConjugate ? conj(inteVal) : inteVal
    end::C
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
                           indexer::N1N2Tuple{OneToIndex}) where {T<:Real, D}
    fields = getEntry.(Ref(configSource), first(indexer))
    tuple(fields) => indexer
end


struct OrbCorePointer{D, C<:RealOrComplex}
    inner::MemoryPair{OneToIndex, C}

    function OrbCorePointer(::Count{D}, inner::MemoryPair{OneToIndex, C}) where 
                           {D, C<:RealOrComplex}
        new{D, C}(inner)
    end
end

function OrbCorePointer(pointer::PrimOrbPointer{D, C}, weight::C
                        ) where {D, C<:RealOrComplex}
    innerPair = MemoryPair(genMemory(pointer.inner), genMemory(weight))
    OrbCorePointer(Count(D), innerPair)
end

function OrbCorePointer(pointer::CompOrbPointer{D, C}, weight::AbstractArray{C, 1}
                        ) where {D, C<:RealOrComplex}
    innerPair = MemoryPair(map(x->x.inner, pointer.inner.left), extractMemory(weight))
    OrbCorePointer(Count(D), innerPair)
end


getOrbDataPair(::Type{C}, idx::OneToIndex) where {C<:RealOrComplex} = (idx => one(C))
getOrbDataPair(::Type{C}, pair::Pair{OneToIndex, C}) where {C<:RealOrComplex} = itself(pair)

const OrbDataPtrPairLayout{D, C<:RealOrComplex} = Union{
    NTuple{2, PrimOrbPointer{D, C}}, NTuple{2, OrbCorePointer{D, C}}
}

const OrbDataPtrQuadLayout{D, C<:RealOrComplex} = Union{
    NTuple{4, PrimOrbPointer{D, C}}, NTuple{4, OrbCorePointer{D, C}}
}

const OrbDataPtrVector{D, C<:RealOrComplex} = Union{
    AbstractVector{<:PrimOrbPointer{D, C}}, AbstractVector{OrbCorePointer{D, C}}
}


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
    tensor = ShapedMemory{C}(undef, (len, len))
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

function genOrbCorePointers(inteInfo::OrbitalIntegralInfo, lazyNormalize::Boolean)
    normInfo = initializeOrbNormalization(inteInfo, lazyNormalize)
    lazyMap(inteInfo.basis.format) do pointer
        weightHolder = buildOrbCoreWeight!(normInfo, pointer)
        OrbCorePointer(pointer, weightHolder)
    end
end

function evalIntegralInfo!(inteInfo::OrbitalLayoutInteInfo{T, D, C}, 
                           lazyNormalize::Boolean) where 
                          {T<:Real, D, C<:RealOrComplex{T}}
    corePointers = genOrbCorePointers(inteInfo, lazyNormalize)
    getOrbLayoutIntegralCore!(inteInfo, corePointers)
end


function evalIntegralInfo!(inteInfo::OrbitalVectorInteInfo{T, D, C}, 
                           lazyNormalize::Boolean) where 
                          {T<:Real, D, C<:RealOrComplex{T}}
    corePointers = genOrbCorePointers(inteInfo, lazyNormalize)
    getOrbVectorIntegralCore!(inteInfo, corePointers)
end


function computeLayoutIntegral(op::DirectOperator, orbs::OrbBasisLayout{T, D}; 
                               cache!Self::ParamDataCache=initializeParamDataCache(), 
                               estimatorConfig::OptEstimatorConfig{T}=missing, 
                               lazyCompute::Boolean=True()) where {T<:Real, D}
    orbsData = MultiOrbitalData(orbs, isParamIndependent(op); cache!Self)
    style = OneBodyIntegral{D, getOutputType(orbsData)}()
    inteInfo = initializeOrbIntegral(style, op, orbsData, lazyCompute, estimatorConfig)
    evalIntegralInfo!(inteInfo, lazyCompute)
end


function computeVectorIntegral(::MultiBodyIntegral{D, T, N}, op::DirectOperator, 
                               orbs::OrbBasisVector{T, D}; 
                               cache!Self::ParamDataCache=initializeParamDataCache(), 
                               estimatorConfig::OptEstimatorConfig{T}=missing, 
                               lazyCompute::Boolean=True()) where {T<:Real, D, N}
    orbsData = MultiOrbitalData(orbs, isParamIndependent(op); cache!Self)
    style = MultiBodyIntegral{D, getOutputType(orbsData), N}()
    inteInfo = initializeOrbIntegral(style, op, orbsData, lazyCompute, estimatorConfig)
    evalIntegralInfo!(inteInfo, lazyCompute)
end