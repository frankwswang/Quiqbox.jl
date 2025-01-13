function overlap(orb1::OrbitalBasis{T, D}, orb2::OrbitalBasis{T, D}; 
                 paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T), 
                 lazyCompute::Bool=false) where {T, D}
    if orb1 === orb2 && isRenormalized(orb1)
        one(T)
    else
        computeIntegral(OneBodyIntegral(), Identity(), (orb1, orb2); 
                        paramCache, lazyCompute)
    end
end

overlaps(basisSet::OrbitalBasisSet{T}; 
         paramCache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where {T} = 
computeIntTensor(OneBodyIntegral(), Identity(), basisSet; paramCache)