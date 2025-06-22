export overlap, overlaps, multipoleMoment, multipoleMoments, eKinetic, eKinetics

function overlap(orb1::OrbitalBasis{C1, D}, orb2::OrbitalBasis{C2, D}; 
                 cache!Self::MissingOr{ParamDataCache}=missing, 
                 lazyCompute::Bool=false) where {T<:Real, C1<:RealOrComplex{T}, 
                                                 C2<:RealOrComplex{T}, D}
    if orb1 === orb2 && isRenormalized(orb1)
        one(T)
    else
        ismissing(cache!Self) && (cache!Self = initializeParamDataCache())
        computeIntegral(OneBodyIntegral{D}(), genOverlapSampler(), (orb1, orb2); 
                        cache!Self, lazyCompute)
    end
end

function overlaps(basisSet::OrbBasisVec{<:Real, D}; 
                  cache!Self::ParamDataCache=initializeParamDataCache()) where {D}
    computeIntTensor(OneBodyIntegral{D}(), genOverlapSampler(), basisSet; cache!Self)
end


function multipoleMoment(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                         orb1::OrbitalBasis{C1, D}, orb2::OrbitalBasis{C2, D}; 
                         cache!Self::ParamDataCache=initializeParamDataCache(), 
                         lazyCompute::Bool=false) where {T<:Real, C1<:RealOrComplex{T}, 
                                                         C2<:RealOrComplex{T}, D}
    mmOp = (genMultipoleMomentSampler∘FloatingMonomial)(T.(center), degrees)
    computeIntegral(OneBodyIntegral{D}(), mmOp, (orb1, orb2); cache!Self, lazyCompute)
end

function multipoleMoments(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                          basisSet::OrbBasisVec{<:Real, D}; 
                          cache!Self::ParamDataCache=initializeParamDataCache()
                          ) where {D}
    mmOp = (genMultipoleMomentSampler∘FloatingMonomial)(T.(center), degrees)
    computeIntTensor(OneBodyIntegral{D}(), mmOp, basisSet; cache!Self)
end


function eKinetic(orb1::OrbitalBasis{C1, D}, orb2::OrbitalBasis{C2, D}, 
                  operator::KineticEnergySampler{T, D}=genKineticEnergySampler(T, Count(D)); 
                  cache!Self::MissingOr{ParamDataCache}=missing, 
                  lazyCompute::Bool=false) where 
                 {T<:Real, C1<:RealOrComplex{T}, C2<:RealOrComplex{T}, D}
    ismissing(cache!Self) && (cache!Self = initializeParamDataCache())
    computeIntegral(OneBodyIntegral{D}(), operator.core, (orb1, orb2); 
                    cache!Self, lazyCompute)
end

eKinetics(basisSet::OrbBasisVec{T, D}, 
          operator::KineticEnergySampler{T, D}=genKineticEnergySampler(T, Count(D)); 
          cache!Self::ParamDataCache=initializeParamDataCache()) where {T<:Real, D} = 
computeIntTensor(OneBodyIntegral{D}(), operator.core, basisSet; cache!Self)