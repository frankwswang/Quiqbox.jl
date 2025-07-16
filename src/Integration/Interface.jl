export overlap, overlaps, multipoleMoment, multipoleMoments, eKinetic, eKinetics, 
       neAttraction, neAttractions, eeInteraction, eeInteractions

function overlap(orbL::OrbitalBasis{CL, D}, orbR::OrbitalBasis{CR, D}; 
                 cache!Self::MissingOr{ParamDataCache}=missing, 
                 estimatorConfig::OptEstimatorConfig{T}=missing, 
                 lazyCompute::AbstractBool=True()) where 
                {T<:Real, CL<:RealOrComplex{T}, CR<:RealOrComplex{T}, D}
    if orbL === orbR && isRenormalized(orbL)
        one(T)
    else
        ismissing(cache!Self) && (cache!Self = initializeParamDataCache())
        lazyCompute = toBoolean(lazyCompute)
        computeLayoutIntegral(genOverlapSampler(), (orbL, orbR); 
                              cache!Self, estimatorConfig, lazyCompute)
    end
end

function overlaps(basisSet::OrbBasisVector{T, D}; 
                  cache!Self::ParamDataCache=initializeParamDataCache(), 
                  estimatorConfig::OptEstimatorConfig{T}=missing, 
                  lazyCompute::AbstractBool=True()) where {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    computeVectorIntegral(OneBodyIntegral{D, T}(), genOverlapSampler(), basisSet; 
                          cache!Self, estimatorConfig, lazyCompute)
end


function multipoleMoment(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                         orbL::OrbitalBasis{CL, D}, orbR::OrbitalBasis{CR, D}; 
                         cache!Self::ParamDataCache=initializeParamDataCache(), 
                         estimatorConfig::OptEstimatorConfig{T}=missing, 
                         lazyCompute::AbstractBool=True()) where 
                        {T<:Real, CL<:RealOrComplex{T}, CR<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    mmOp = (genMultipoleMomentSampler∘FloatingMonomial)(T.(center), degrees)
    computeLayoutIntegral(mmOp, (orbL, orbR); cache!Self, estimatorConfig, lazyCompute)
end

function multipoleMoments(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                          basisSet::OrbBasisVector{T, D}; 
                          cache!Self::ParamDataCache=initializeParamDataCache(), 
                          estimatorConfig::OptEstimatorConfig{T}=missing, 
                          lazyCompute::AbstractBool=True()) where {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    mmOp = (genMultipoleMomentSampler∘FloatingMonomial)(T.(center), degrees)
    computeVectorIntegral(OneBodyIntegral{D, T}(), mmOp, basisSet; 
                          cache!Self, estimatorConfig, lazyCompute)
end


function eKinetic(orbL::OrbitalBasis{CL, D}, orbR::OrbitalBasis{CR, D}, 
                  config::KineticEnergySampler{T, D}=genKineticEnergySampler(T, Count(D)); 
                  cache!Self::ParamDataCache=initializeParamDataCache(), 
                  estimatorConfig::OptEstimatorConfig{T}=missing, 
                  lazyCompute::AbstractBool=True()) where 
                 {T<:Real, CL<:RealOrComplex{T}, CR<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    computeLayoutIntegral(config.core, (orbL, orbR); 
                          cache!Self, estimatorConfig, lazyCompute)
end

function eKinetics(basisSet::OrbBasisVector{T, D}, 
                   config::KineticEnergySampler{T, D}=genKineticEnergySampler(T, Count(D)); 
                   cache!Self::ParamDataCache=initializeParamDataCache(), 
                   estimatorConfig::OptEstimatorConfig{T}=missing, 
                   lazyCompute::AbstractBool=True()) where {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    computeVectorIntegral(OneBodyIntegral{D, T}(), config.core, basisSet; 
                          cache!Self, estimatorConfig, lazyCompute)
end


function neAttraction(nucs::AbstractVector{Symbol}, 
                      nucCoords::AbstractVector{NTuple{D, T}}, 
                      orbL::OrbitalBasis{CL, D}, orbR::OrbitalBasis{CR, D}; 
                      cache!Self::ParamDataCache=initializeParamDataCache(), 
                      estimatorConfig::OptEstimatorConfig{T}=missing, 
                      lazyCompute::AbstractBool=True()) where 
                     {T<:Real, CL<:RealOrComplex{T}, CR<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    neOp = genCoulombMultiPointSampler(map(T∘getCharge, nucs), nucCoords)
    computeLayoutIntegral(neOp, (orbL, orbR); cache!Self, estimatorConfig, lazyCompute)
end

function neAttractions(nucs::AbstractVector{Symbol}, 
                       nucCoords::AbstractVector{NTuple{D, T}}, 
                       basisSet::OrbBasisVector{T, D}; 
                       cache!Self::ParamDataCache=initializeParamDataCache(), 
                       estimatorConfig::OptEstimatorConfig{T}=missing, 
                       lazyCompute::AbstractBool=True()) where {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    neOp = genCoulombMultiPointSampler(map(T∘getCharge, nucs), nucCoords)
    computeVectorIntegral(OneBodyIntegral{D, T}(), neOp, basisSet; 
                          cache!Self, estimatorConfig, lazyCompute)
end


function eeInteraction(orbL1::OrbitalBasis{CL1, D}, orbR1::OrbitalBasis{CR1, D}, 
                       orbL2::OrbitalBasis{CL2, D}, orbR2::OrbitalBasis{CR2, D}; 
                       cache!Self::ParamDataCache=initializeParamDataCache(), 
                       estimatorConfig::OptEstimatorConfig{T}=missing, 
                       lazyCompute::AbstractBool=True()) where 
                      {T<:Real, CL1<:RealOrComplex{T}, CR1<:RealOrComplex{T}, 
                                CL2<:RealOrComplex{T}, CR2<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    eeOp = genCoulombInteractionSampler(T, Count(D))
    layout = (orbL1, orbR1, orbL2, orbR2)
    computeLayoutIntegral(eeOp, layout; cache!Self, estimatorConfig, lazyCompute)
end

function eeInteractions(basisSet::OrbBasisVector{T, D}; 
                        cache!Self::ParamDataCache=initializeParamDataCache(), 
                        estimatorConfig::OptEstimatorConfig{T}=missing, 
                        lazyCompute::AbstractBool=True()) where {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    eeOp = genCoulombInteractionSampler(T, Count(D))
    computeVectorIntegral(TwoBodyIntegral{D, T}(), eeOp, basisSet; 
                          cache!Self, estimatorConfig, lazyCompute)
end