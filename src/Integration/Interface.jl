export overlap, overlaps, multipoleMoment, multipoleMoments, elecKinetic, elecKinetics, 
       nucAttraction, nucAttractions, coreHamiltonian, elecRepulsion, elecRepulsions, 
       changeOrbitalBasis

using TensorOperations: @tensor as @TOtensor

const CONSTVAR!!TypeStrOfOrbBasisVector = shortUnionAllString(OrbBasisVector)

"""

    overlap(orbL::$OrbitalBasis{CL, D}, orbR::$OrbitalBasis{CR, D}; 
            lazyCompute::Bool=true
            ) where {T<:Real, CL<:$CONSTVAR!!TypeStrOfRealOrComplex, 
                              CR<:$CONSTVAR!!TypeStrOfRealOrComplex, D} -> 
    $CONSTVAR!!TypeStrOfRealOrComplex

Compute the overlap integral between two orbital basis functions `orbL` and `orbR`. If 
`lazyCompute` is set to `true`, the integral will be computed in a lazy manner to avoid 
repetitive primitive integration.
"""
function overlap(orbL::OrbitalBasis{CL, D}, orbR::OrbitalBasis{CR, D}; 
                 lazyCompute::AbstractBool=True(), 
                 estimatorConfig::OptEstimatorConfig{T}=missing, 
                 cache!Self::OptParamDataCache=initializeParamDataCache()) where 
                {T<:Real, CL<:RealOrComplex{T}, CR<:RealOrComplex{T}, D}
    if orbL === orbR && isRenormalized(orbL)
        one(T)
    else
        lazyCompute = toBoolean(lazyCompute)
        computeOrbLayoutIntegral(genOverlapSampler(), (orbL, orbR); 
                                 lazyCompute, estimatorConfig, cache!Self)
    end
end

"""

    overlaps(basisSet::$CONSTVAR!!TypeStrOfOrbBasisVector; 
             lazyCompute::Bool=true
             ) where {T<:Real, D} -> 
    AbstractMatrix{<:$CONSTVAR!!TypeStrOfRealOrComplex}

Compute the overlap integrals for all pairs of orbital basis functions in `basisSet`. If 
`lazyCompute` is set to `true`, the integrals will be computed in a lazy manner to avoid 
repetitive primitive integration.
"""
function overlaps(basisSet::OrbBasisVector{T, D}; 
                  lazyCompute::AbstractBool=True(), 
                  estimatorConfig::OptEstimatorConfig{T}=missing, 
                  cache!Self::OptParamDataCache=initializeParamDataCache()) where 
                 {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    computeOrbVectorIntegral(OneBodyIntegral{D, T}(), genOverlapSampler(), basisSet; 
                             lazyCompute, estimatorConfig, cache!Self)
end


"""

    multipoleMoment(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                    orbL::$OrbitalBasis{CL, D}, orbR::$OrbitalBasis{CR, D}; 
                    lazyCompute::Bool=true
                    ) where {T<:Real, CL<:$CONSTVAR!!TypeStrOfRealOrComplex, 
                                      CR<:$CONSTVAR!!TypeStrOfRealOrComplex, D} -> 
    $CONSTVAR!!TypeStrOfRealOrComplex

Compute the multipole-moment integral between two orbital basis functions `orbL` and `orbR` 
at the `D`-dimensional `center` with `degrees` specified for its axes. If `lazyCompute` is 
set to `true`, the integral will be computed in a lazy manner to avoid repetitive primitive 
integration.
"""
function multipoleMoment(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                         orbL::OrbitalBasis{CL, D}, orbR::OrbitalBasis{CR, D}; 
                         lazyCompute::AbstractBool=True(), 
                         estimatorConfig::OptEstimatorConfig{T}=missing, 
                         cache!Self::OptParamDataCache=initializeParamDataCache()) where 
                        {T<:Real, CL<:RealOrComplex{T}, CR<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    mmOp = (genMultipoleMomentSampler∘FloatingMonomial)(T.(center), degrees)
    computeOrbLayoutIntegral(mmOp, (orbL, orbR); lazyCompute, estimatorConfig, cache!Self)
end

"""

    multipoleMoments(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                     basisSet::$CONSTVAR!!TypeStrOfOrbBasisVector; 
                     lazyCompute::Bool=true
                     ) where {T<:Real, D} -> 
    AbstractMatrix{<:$CONSTVAR!!TypeStrOfRealOrComplex}

Compute the multipole-moment integrals for all pairs of orbital basis functions in 
`basisSet` at the `D`-dimensional `center` with `degrees` specified for its axes. If 
`lazyCompute` is set to `true`, the integrals will be computed in a lazy manner to avoid 
repetitive primitive integration.
"""
function multipoleMoments(center::NTuple{D, Real}, degrees::NTuple{D, Int}, 
                          basisSet::OrbBasisVector{T, D}; 
                          lazyCompute::AbstractBool=True(), 
                          estimatorConfig::OptEstimatorConfig{T}=missing, 
                          cache!Self::OptParamDataCache=initializeParamDataCache()) where 
                         {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    mmOp = (genMultipoleMomentSampler∘FloatingMonomial)(T.(center), degrees)
    computeOrbVectorIntegral(OneBodyIntegral{D, T}(), mmOp, basisSet; 
                             lazyCompute, estimatorConfig, cache!Self)
end


"""

    elecKinetic(orbL::$OrbitalBasis{CL, D}, orbR::$OrbitalBasis{CR, D}; 
                lazyCompute::Bool=true
                ) where {T<:Real, CL<:$CONSTVAR!!TypeStrOfRealOrComplex, 
                                  CR<:$CONSTVAR!!TypeStrOfRealOrComplex, D} -> 
    $CONSTVAR!!TypeStrOfRealOrComplex

Compute the electronic kinetic-energy integral between two orbital basis functions `orbL` 
and `orbR`. If `lazyCompute` is set to `true`, the integral will be computed in a lazy 
manner to avoid repetitive primitive integration.
"""
function elecKinetic(orbL::OrbitalBasis{CL, D}, orbR::OrbitalBasis{CR, D}, 
                     config::KineticEnergySampler{T, D}=
                             genKineticEnergySampler(T, Count(D)); 
                     lazyCompute::AbstractBool=True(), 
                     estimatorConfig::OptEstimatorConfig{T}=missing, 
                     cache!Self::OptParamDataCache=initializeParamDataCache()) where 
                    {T<:Real, CL<:RealOrComplex{T}, CR<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    computeOrbLayoutIntegral(config.core, (orbL, orbR); 
                             lazyCompute, estimatorConfig, cache!Self)
end

"""

    elecKinetics(basisSet::$CONSTVAR!!TypeStrOfOrbBasisVector; 
                 lazyCompute::Bool=true
                 ) where {T<:Real, D} -> 
    AbstractMatrix{<:$CONSTVAR!!TypeStrOfRealOrComplex}

Compute the electronic kinetic-energy integrals for all pairs of orbital basis functions in 
`basisSet`. If `lazyCompute` is set to `true`, the integrals will be computed in a lazy 
manner to avoid repetitive primitive integration.
"""
function elecKinetics(basisSet::OrbBasisVector{T, D}, 
                      config::KineticEnergySampler{T, D}=
                              genKineticEnergySampler(T, Count(D)); 
                      lazyCompute::AbstractBool=True(), 
                      estimatorConfig::OptEstimatorConfig{T}=missing, 
                      cache!Self::OptParamDataCache=initializeParamDataCache()) where 
                     {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    computeOrbVectorIntegral(OneBodyIntegral{D, T}(), config.core, basisSet; 
                             lazyCompute, estimatorConfig, cache!Self)
end


"""

    nucAttraction(nucs::AbstractVector{Symbol}, nucCoords::AbstractVector{NTuple{D, T}}, 
                  orbL::$OrbitalBasis{CL, D}, orbR::$OrbitalBasis{CR, D}; 
                  lazyCompute::Bool=true
                  ) where {T<:Real, CL<:$CONSTVAR!!TypeStrOfRealOrComplex, 
                                    CR<:$CONSTVAR!!TypeStrOfRealOrComplex, D} ->
    $CONSTVAR!!TypeStrOfRealOrComplex

    nucAttraction(nucInfo::$NuclearCluster{T, D}, 
                  orbL::$OrbitalBasis{CL, D}, orbR::$OrbitalBasis{CR, D}; 
                  lazyCompute::Bool=true
                  ) where {T<:Real, CL<:$CONSTVAR!!TypeStrOfRealOrComplex, 
                                    CR<:$CONSTVAR!!TypeStrOfRealOrComplex, D} ->
    $CONSTVAR!!TypeStrOfRealOrComplex

Compute the nuclear-attraction integral between two orbital basis functions `orbL` and 
`orbR`. If `lazyCompute` is set to `true`, the integral will be computed in a lazy manner 
to avoid repetitive primitive integration.

≡≡≡ Positional argument(s) ≡≡≡

`nucs::AbstractVector{Symbol}`: A list of nuclear species.

`nucCoords::AbstractVector{NTuple{D, T}}`: The list of Cartesian nuclear coordinates in 
the order respective to `nucs`.

`nucInfo::NuclearCluster{T, D}`: A container storing the nuclear species and their 
respective coordinates.
"""
function nucAttraction(nucs::AbstractVector{Symbol}, 
                       nucCoords::AbstractVector{NTuple{D, T}}, 
                       orbL::OrbitalBasis{CL, D}, orbR::OrbitalBasis{CR, D}; 
                       lazyCompute::AbstractBool=True(), 
                       estimatorConfig::OptEstimatorConfig{T}=missing, 
                       cache!Self::OptParamDataCache=initializeParamDataCache()) where 
                      {T<:Real, CL<:RealOrComplex{T}, CR<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    neOp = genCoulombMultiPointSampler(map(T∘getCharge, nucs), nucCoords)
    computeOrbLayoutIntegral(neOp, (orbL, orbR); lazyCompute, estimatorConfig, cache!Self)
end

nucAttraction(nucInfo::NuclearCluster{T, D}, 
              orbL::OrbitalBasis{CL, D}, orbR::OrbitalBasis{CR, D}; 
              lazyCompute::AbstractBool=True(), 
              estimatorConfig::OptEstimatorConfig{T}=missing, 
              cache!Self::OptParamDataCache=initializeParamDataCache()) where 
             {T<:Real, CL<:RealOrComplex{T}, CR<:RealOrComplex{T}, D} = 
nucAttraction(nucInfo.layout.left, nucInfo.layout.right, orbL, orbR; 
              lazyCompute, estimatorConfig, cache!Self)


"""

    nucAttractions(nucs::AbstractVector{Symbol}, nucCoords::AbstractVector{NTuple{D, T}}, 
                   basisSet::$CONSTVAR!!TypeStrOfOrbBasisVector; 
                   lazyCompute::Bool=true
                   ) where {T<:Real, D} -> 
    AbstractMatrix{<:$CONSTVAR!!TypeStrOfRealOrComplex}

    nucAttractions(nucInfo::NuclearCluster{T, D}, 
                   basisSet::$CONSTVAR!!TypeStrOfOrbBasisVector; 
                   lazyCompute::Bool=true
                   ) where {T<:Real, D} -> 
    AbstractMatrix{<:$CONSTVAR!!TypeStrOfRealOrComplex}

Compute the nuclear-attraction integrals for all pairs of orbital basis functions in 
`basisSet`. If `lazyCompute` is set to `true`, the integrals will be computed in a lazy 
manner to avoid repetitive primitive integration.

≡≡≡ Positional argument(s) ≡≡≡

`nucs::AbstractVector{Symbol}`: A list of nuclear species.

`nucCoords::AbstractVector{NTuple{D, T}}`: The list of Cartesian nuclear coordinates in 
the order respective to `nucs`.

`nucInfo::NuclearCluster{T, D}`: A container storing the nuclear species and their 
respective coordinates.
"""
function nucAttractions(nucs::AbstractVector{Symbol}, 
                        nucCoords::AbstractVector{NTuple{D, T}}, 
                        basisSet::OrbBasisVector{T, D}; 
                        lazyCompute::AbstractBool=True(), 
                        estimatorConfig::OptEstimatorConfig{T}=missing, 
                        cache!Self::OptParamDataCache=initializeParamDataCache()) where 
                       {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    neOp = genCoulombMultiPointSampler(map(T∘getCharge, nucs), nucCoords)
    computeOrbVectorIntegral(OneBodyIntegral{D, T}(), neOp, basisSet; 
                             lazyCompute, estimatorConfig, cache!Self)
end

nucAttractions(nucInfo::NuclearCluster{T, D},
               basisSet::OrbBasisVector{T, D}; 
               lazyCompute::AbstractBool=True(), 
               estimatorConfig::OptEstimatorConfig{T}=missing, 
               cache!Self::OptParamDataCache=initializeParamDataCache()) where 
              {T<:Real, D} = 
nucAttractions(nucInfo.layout.left, nucInfo.layout.right, basisSet; 
               lazyCompute, estimatorConfig, cache!Self)


"""

    coreHamiltonian(nucs::AbstractVector{Symbol}, nucCoords::AbstractVector{NTuple{D, T}}, 
                    basisSet::$CONSTVAR!!TypeStrOfOrbBasisVector; 
                    lazyCompute::Bool=true
                    ) where {T<:Real, D} -> 
    AbstractMatrix{<:$CONSTVAR!!TypeStrOfRealOrComplex}

    coreHamiltonian(nucInfo::NuclearCluster{T, D}, 
                    basisSet::$CONSTVAR!!TypeStrOfOrbBasisVector; 
                    lazyCompute::Bool=true
                    ) where {T<:Real, D} -> 
    AbstractMatrix{<:$CONSTVAR!!TypeStrOfRealOrComplex}

Compute the core-Hamiltonian integrals for all pairs of orbital basis functions in 
`basisSet`, which is the matrix addition of the electronic kinetic-energy integrals and the 
nuclear-electron attraction integrals. If `lazyCompute` is set to `true`, the integrals 
will be computed in a lazy manner to avoid repetitive primitive integration.

≡≡≡ Positional argument(s) ≡≡≡

`nucs::AbstractVector{Symbol}`: A list of nuclear species.

`nucCoords::AbstractVector{NTuple{D, T}}`: The list of Cartesian nuclear coordinates in 
the order respective to `nucs`.

`nucInfo::NuclearCluster{T, D}`: A container storing the nuclear species and their 
respective coordinates.
"""
function coreHamiltonian(nucs::AbstractVector{Symbol}, 
                         nucCoords::AbstractVector{NTuple{D, T}}, 
                         basisSet::OrbBasisVector{T, D}, 
                         kineticOperator::KineticEnergySampler{T, D}=
                                          genKineticEnergySampler(T, Count(D)); 
                         lazyCompute::AbstractBool=True(), 
                         estimatorConfig::OptEstimatorConfig{T}=missing, 
                         cache!Self::OptParamDataCache=initializeParamDataCache()) where 
                        {T<:Real, D}
    lazyCompute = toBoolean(lazyCompute)
    coreHop = genCoreHamiltonianSampler(nucs, nucCoords, kineticOperator)
    computeOrbVectorIntegral(OneBodyIntegral{D, T}(), coreHop, basisSet; 
                             lazyCompute, estimatorConfig, cache!Self)
end

coreHamiltonian(nucInfo::NuclearCluster{T, D}, 
                basisSet::OrbBasisVector{T, D}, 
                kineticOperator::KineticEnergySampler{T, D}=
                                 genKineticEnergySampler(T, Count(D)); 
                lazyCompute::AbstractBool=True(), 
                estimatorConfig::OptEstimatorConfig{T}=missing, 
                cache!Self::OptParamDataCache=initializeParamDataCache()) where 
               {T<:Real, D} = 
coreHamiltonian(nucInfo.layout.left, nucInfo.layout.right, basisSet, kineticOperator; 
                lazyCompute, estimatorConfig, cache!Self)


"""

    elecRepulsion(orbL1::$OrbitalBasis{CL1, D}, orbR1::$OrbitalBasis{CR1, D}, 
                  orbL2::$OrbitalBasis{CL2, D}, orbR2::$OrbitalBasis{CR2, D}; 
                  lazyCompute::Bool=true
                  ) where {T<:Real, CL1<:$CONSTVAR!!TypeStrOfRealOrComplex, 
                                    CR1<:$CONSTVAR!!TypeStrOfRealOrComplex, 
                                    CL2<:$CONSTVAR!!TypeStrOfRealOrComplex, 
                                    CR2<:$CONSTVAR!!TypeStrOfRealOrComplex, D} -> 
    $CONSTVAR!!TypeStrOfRealOrComplex

Compute the electron-repulsion integral between two pairs of orbital basis functions
`(orbL1, orbR1)` and `(orbL2, orbR2)` (ordered by the chemists' notation). If `lazyCompute` 
is set to `true`, the integral will be computed in a lazy manner to avoid repetitive 
primitive integration.
"""
function elecRepulsion(orbL1::OrbitalBasis{CL1, D}, orbR1::OrbitalBasis{CR1, D}, 
                       orbL2::OrbitalBasis{CL2, D}, orbR2::OrbitalBasis{CR2, D}; 
                       lazyCompute::AbstractBool=True(), 
                       estimatorConfig::OptEstimatorConfig{T}=missing, 
                       cache!Self::OptParamDataCache=initializeParamDataCache()) where 
                      {T<:Real, CL1<:RealOrComplex{T}, CR1<:RealOrComplex{T}, 
                                CL2<:RealOrComplex{T}, CR2<:RealOrComplex{T}, D}
    lazyCompute = toBoolean(lazyCompute)
    eeOp = genCoulombInteractionSampler(T, Count(D))
    layout = (orbL1, orbR1, orbL2, orbR2)
    computeOrbLayoutIntegral(eeOp, layout; lazyCompute, estimatorConfig, cache!Self)
end

"""

    elecRepulsions(basisSet::$CONSTVAR!!TypeStrOfOrbBasisVector; 
                   lazyCompute::Bool=true
                   ) where {T<:Real, D} -> 
    AbstractArray{<:$CONSTVAR!!TypeStrOfRealOrComplex, 4}

Compute the electron-repulsion integrals for all double pairs of orbital basis functions 
in `basisSet` (tensor indices are ordered by the chemists' notation). If `lazyCompute` is 
set to `true`, the integrals will be computed in a lazy manner to avoid repetitive 
primitive integration.
"""
function elecRepulsions(basisSet::OrbBasisVector{T, D}; 
                        lazyCompute::AbstractBool=True(), 
                        estimatorConfig::OptEstimatorConfig{T}=missing, 
                        cache!Self::OptParamDataCache=initializeParamDataCache()) where 
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
orbital coefficient matrix.
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