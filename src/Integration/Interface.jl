export overlap, overlaps, multipoleMoment, multipoleMoments, eKinetic, eKinetics, 
       neAttraction, neAttractions, coreHamiltonian, eeInteraction, eeInteractions, 
       changeOrbitalBasis

using TensorOperations: @tensor as @TOtensor

function overlap(orbL::OrbitalBasis{CL, D}, orbR::OrbitalBasis{CR, D}; 
                 lazyCompute::AbstractBool=True(), 
                 estimatorConfig::OptEstimatorConfig{T}=missing, 
                 cache!Self::MissingOr{ParamDataCache}=missing) where 
                {T<:Real, CL<:RealOrComplex{T}, CR<:RealOrComplex{T}, D}
    if orbL === orbR && isRenormalized(orbL)
        one(T)
    else
        ismissing(cache!Self) && (cache!Self = initializeParamDataCache())
        lazyCompute = toBoolean(lazyCompute)
        computeOrbLayoutIntegral(genOverlapSampler(), (orbL, orbR); 
                                 lazyCompute, estimatorConfig, cache!Self)
    end
end

function overlaps(basisSet::OrbBasisVector{T, D}; 
                  lazyCompute::AbstractBool=True(), 
                  estimatorConfig::OptEstimatorConfig{T}=missing, 
                  cache!Self::ParamDataCache=initializeParamDataCache()) where {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    computeOrbVectorIntegral(OneBodyIntegral{D, T}(), genOverlapSampler(), basisSet; 
                             lazyCompute, estimatorConfig, cache!Self)
end


function multipoleMoment(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                         orbL::OrbitalBasis{CL, D}, orbR::OrbitalBasis{CR, D}; 
                         lazyCompute::AbstractBool=True(), 
                         estimatorConfig::OptEstimatorConfig{T}=missing, 
                         cache!Self::ParamDataCache=initializeParamDataCache()) where 
                        {T<:Real, CL<:RealOrComplex{T}, CR<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    mmOp = (genMultipoleMomentSampler∘FloatingMonomial)(T.(center), degrees)
    computeOrbLayoutIntegral(mmOp, (orbL, orbR); lazyCompute, estimatorConfig, cache!Self)
end

function multipoleMoments(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                          basisSet::OrbBasisVector{T, D}; 
                          lazyCompute::AbstractBool=True(), 
                          estimatorConfig::OptEstimatorConfig{T}=missing, 
                          cache!Self::ParamDataCache=initializeParamDataCache()) where 
                         {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    mmOp = (genMultipoleMomentSampler∘FloatingMonomial)(T.(center), degrees)
    computeOrbVectorIntegral(OneBodyIntegral{D, T}(), mmOp, basisSet; 
                             lazyCompute, estimatorConfig, cache!Self)
end


function eKinetic(orbL::OrbitalBasis{CL, D}, orbR::OrbitalBasis{CR, D}, 
                  config::KineticEnergySampler{T, D}=genKineticEnergySampler(T, Count(D)); 
                  lazyCompute::AbstractBool=True(), 
                  estimatorConfig::OptEstimatorConfig{T}=missing, 
                  cache!Self::ParamDataCache=initializeParamDataCache()) where 
                 {T<:Real, CL<:RealOrComplex{T}, CR<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    computeOrbLayoutIntegral(config.core, (orbL, orbR); 
                             lazyCompute, estimatorConfig, cache!Self)
end

function eKinetics(basisSet::OrbBasisVector{T, D}, 
                   config::KineticEnergySampler{T, D}=genKineticEnergySampler(T, Count(D)); 
                   lazyCompute::AbstractBool=True(), 
                   estimatorConfig::OptEstimatorConfig{T}=missing, 
                   cache!Self::ParamDataCache=initializeParamDataCache()) where 
                  {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    computeOrbVectorIntegral(OneBodyIntegral{D, T}(), config.core, basisSet; 
                             lazyCompute, estimatorConfig, cache!Self)
end


function neAttraction(nucs::AbstractVector{Symbol}, 
                      nucCoords::AbstractVector{NTuple{D, T}}, 
                      orbL::OrbitalBasis{CL, D}, orbR::OrbitalBasis{CR, D}; 
                      lazyCompute::AbstractBool=True(), 
                      estimatorConfig::OptEstimatorConfig{T}=missing, 
                      cache!Self::ParamDataCache=initializeParamDataCache()) where 
                     {T<:Real, CL<:RealOrComplex{T}, CR<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    neOp = genCoulombMultiPointSampler(map(T∘getCharge, nucs), nucCoords)
    computeOrbLayoutIntegral(neOp, (orbL, orbR); lazyCompute, estimatorConfig, cache!Self)
end

function neAttractions(nucs::AbstractVector{Symbol}, 
                       nucCoords::AbstractVector{NTuple{D, T}}, 
                       basisSet::OrbBasisVector{T, D}; 
                       lazyCompute::AbstractBool=True(), 
                       estimatorConfig::OptEstimatorConfig{T}=missing, 
                       cache!Self::ParamDataCache=initializeParamDataCache()) where 
                      {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    neOp = genCoulombMultiPointSampler(map(T∘getCharge, nucs), nucCoords)
    computeOrbVectorIntegral(OneBodyIntegral{D, T}(), neOp, basisSet; 
                             lazyCompute, estimatorConfig, cache!Self)
end


function coreHamiltonian(nucs::AbstractVector{Symbol}, 
                         nucCoords::AbstractVector{NTuple{D, T}}, 
                         basisSet::OrbBasisVector{T, D}, 
                         kineticOperator::KineticEnergySampler{T, D}=
                                          genKineticEnergySampler(T, Count(D)), 
                         lazyCompute::AbstractBool=True(), 
                         estimatorConfig::OptEstimatorConfig{T}=missing, 
                         cache!Self::ParamDataCache=initializeParamDataCache()) where 
                        {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    coreHop = genCoreHamiltonianSampler(nucs, nucCoords, kineticOperator)
    computeOrbVectorIntegral(OneBodyIntegral{D, T}(), coreHop, basisSet; 
                             lazyCompute, estimatorConfig, cache!Self)
end


function eeInteraction(orbL1::OrbitalBasis{CL1, D}, orbR1::OrbitalBasis{CR1, D}, 
                       orbL2::OrbitalBasis{CL2, D}, orbR2::OrbitalBasis{CR2, D}; 
                       lazyCompute::AbstractBool=True(), 
                       estimatorConfig::OptEstimatorConfig{T}=missing, 
                       cache!Self::ParamDataCache=initializeParamDataCache()) where 
                      {T<:Real, CL1<:RealOrComplex{T}, CR1<:RealOrComplex{T}, 
                                CL2<:RealOrComplex{T}, CR2<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    eeOp = genCoulombInteractionSampler(T, Count(D))
    layout = (orbL1, orbR1, orbL2, orbR2)
    computeOrbLayoutIntegral(eeOp, layout; lazyCompute, estimatorConfig, cache!Self)
end

function eeInteractions(basisSet::OrbBasisVector{T, D}; 
                        lazyCompute::AbstractBool=True(), 
                        estimatorConfig::OptEstimatorConfig{T}=missing, 
                        cache!Self::ParamDataCache=initializeParamDataCache()) where 
                       {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    eeOp = genCoulombInteractionSampler(T, Count(D))
    computeOrbVectorIntegral(TwoBodyIntegral{D, T}(), eeOp, basisSet; 
                             lazyCompute, estimatorConfig, cache!Self)
end


"""

    changeOrbitalBasis(DbodyInt::AbstractArray{T, D}, C::AbstractMatrix{T}) where {T} -> 
    AbstractArray{T, D}

Change the orbital basis of the input one-body / two-body integrals `DbodyInt` based on the 
orbital coefficient matrix `C`.
"""
changeOrbitalBasis(oneBodyInt::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T} = 
@TOtensor ij[i,j] := oneBodyInt[a,b] * C[a,i] * C[b,j]

changeOrbitalBasis(twoBodyInt::AbstractArray{T, 4}, C::AbstractMatrix{T}) where {T} = 
@TOtensor ijkl[i,j,k,l] := twoBodyInt[a,b,c,d] * C[a,i] * C[b,j] * C[c,k] * C[d,l]

function getJᵅᵝ(twoBodyInt::AbstractArray{T, 4}, 
                (C1, C2)::NTuple{2, AbstractMatrix{T}}) where {T}
    m = axes(C1, 2)
    n = axes(C2, 2)
    map(Iterators.product(m, n)) do idx
        C1c = view(C1, :, idx[begin])
        C2c = view(C2, :, idx[end])
        @TOtensor twoBodyInt[a,b,c,d] * C1c[a] * C1c[b] * C2c[c] * C2c[d]
    end
end

"""

    changeOrbitalBasis(twoBodyInt::AbstractArray{T, 4}, 
                 C1::AbstractMatrix{T}, C2::AbstractMatrix{T}) where {T} -> 
    AbstractArray{T, 4}

Change the orbital basis of the input two-body integrals `twoBodyInt` based on two orbital 
coefficient matrices `C1` and `C2` for different spin configurations (e.g., the 
unrestricted case). The output is a 3-element `Tuple` of which the first 2 elements are the 
spatial integrals of each spin configurations respectively, while the last element is the 
Coulomb interactions between orbitals with different spins.
"""
changeOrbitalBasis(twoBodyInt::AbstractArray{T, 4}, 
                   C::Vararg{AbstractMatrix{T}, 2}) where {T} = 
(changeOrbitalBasis.(Ref(twoBodyInt), C)..., getJᵅᵝ(twoBodyInt, C))