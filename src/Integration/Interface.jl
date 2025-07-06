export overlap, overlaps, multipoleMoment, multipoleMoments, eKinetic, eKinetics

function overlap(orb1::OrbitalBasis{C1, D}, orb2::OrbitalBasis{C2, D}; 
                 cache!Self::MissingOr{ParamDataCache}=missing, 
                 estimatorConfig::OptEstimatorConfig{T}=missing, 
                 lazyCompute::AbstractBool=True()) where 
                {T<:Real, C1<:RealOrComplex{T}, C2<:RealOrComplex{T}, D}
    if orb1 === orb2 && isRenormalized(orb1)
        one(T)
    else
        ismissing(cache!Self) && (cache!Self = initializeParamDataCache())
        lazyCompute = toBoolean(lazyCompute)
        computeLayoutIntegral(genOverlapSampler(), (orb1, orb2); 
                              cache!Self, estimatorConfig, lazyCompute)
    end
end

function overlaps(basisSet::OrbBasisVector{T, D}; 
                  cache!Self::ParamDataCache=initializeParamDataCache(), 
                  estimatorConfig::OptEstimatorConfig{T}=missing, 
                  lazyCompute::AbstractBool=True()) where {T<:Real, D}
    computeVectorIntegral(OneBodyIntegral{D, T}(), genOverlapSampler(), basisSet; 
                          cache!Self, estimatorConfig, lazyCompute)
end


function multipoleMoment(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                         orb1::OrbitalBasis{C1, D}, orb2::OrbitalBasis{C2, D}; 
                         cache!Self::ParamDataCache=initializeParamDataCache(), 
                         estimatorConfig::OptEstimatorConfig{T}=missing, 
                         lazyCompute::AbstractBool=True()) where 
                        {T<:Real, C1<:RealOrComplex{T}, C2<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    mmOp = (genMultipoleMomentSampler∘FloatingMonomial)(T.(center), degrees)
    computeLayoutIntegral(mmOp, (orb1, orb2); cache!Self, estimatorConfig, lazyCompute)
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


function eKinetic(orb1::OrbitalBasis{C1, D}, orb2::OrbitalBasis{C2, D}, 
                  config::KineticEnergySampler{T, D}=genKineticEnergySampler(T, Count(D)); 
                  cache!Self::ParamDataCache=initializeParamDataCache(), 
                  estimatorConfig::OptEstimatorConfig{T}=missing, 
                  lazyCompute::AbstractBool=True()) where 
                 {T<:Real, C1<:RealOrComplex{T}, C2<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    computeLayoutIntegral(config.core, (orb1, orb2); 
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