function overlap(orb1::OrbitalBasis{T, D}, orb2::OrbitalBasis{T, D}; 
                 paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                 lazyCompute::Bool=false) where {T, D}
    if orb1 === orb2 && isRenormalized(orb1)
        one(T)
    else
        computeIntegral(OneBodyIntegral{D}(), Identity(), (orb1, orb2); 
                        paramCache, lazyCompute)
    end
end

overlaps(basisSet::OrbitalBasisSet{T, D}; 
         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where {T, D} = 
computeIntTensor(OneBodyIntegral{D}(), Identity(), basisSet; paramCache)


function multipoleMoment(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                         orb1::OrbitalBasis{T, D}, orb2::OrbitalBasis{T, D}; 
                         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                         lazyCompute::Bool=false) where {T, D}
    mmOp = MonomialMul(T.(center), degrees)
    orbData = orb1 === orb2 ? (orb1,) : (orb1, orb2)
    computeIntegral(OneBodyIntegral{D}(), mmOp, (orb1, orb2); paramCache, lazyCompute)
end

function multipoleMoments(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                          basisSet::OrbitalBasisSet{T, D}; 
                          paramCache::DimSpanDataCacheBox{T}=
                          DimSpanDataCacheBox(T)) where {T, D}
    mmOp = MonomialMul(T.(center), degrees)
    computeIntTensor(OneBodyIntegral{D}(), mmOp, basisSet; paramCache)
end