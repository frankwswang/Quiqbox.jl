export overlap, overlaps, multipoleMoment, multipoleMoments

function overlap(orb1::OrbitalBasis{T, D}, orb2::OrbitalBasis{T, D}; 
                 cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox(), 
                 lazyCompute::Bool=false) where {T, D}
    if orb1 === orb2 && isRenormalized(orb1)
        one(T)
    else
        computeIntegral(OneBodyIntegral{D}(), Identity(), (orb1, orb2); 
                        cache!Self, lazyCompute)
    end
end

overlaps(basisSet::OrbitalBasisSet{T, D}; 
         cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox()) where {T, D} = 
computeIntTensor(OneBodyIntegral{D}(), Identity(), basisSet; cache!Self)


function multipoleMoment(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                         orb1::OrbitalBasis{T, D}, orb2::OrbitalBasis{T, D}; 
                         cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox(), 
                         lazyCompute::Bool=false) where {T, D}
    mmOp = MonomialMul(T.(center), degrees)
    orbData = orb1 === orb2 ? (orb1,) : (orb1, orb2)
    computeIntegral(OneBodyIntegral{D}(), mmOp, (orb1, orb2); cache!Self, lazyCompute)
end

function multipoleMoments(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                          basisSet::OrbitalBasisSet{T, D}; 
                          cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox()
                          ) where {T, D}
    mmOp = MonomialMul(T.(center), degrees)
    computeIntTensor(OneBodyIntegral{D}(), mmOp, basisSet; cache!Self)
end