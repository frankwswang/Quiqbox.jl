const OneBodySampler{O<:MonoTermOperator} = Multiplier{ComplexConj, O}

const OverlapSampler = OneBodySampler{Identity}

function genOneBodySampler(rightOp::MonoTermOperator)::OneBodySampler
    Correlate(*, (ComplexConj(), rightOp))
end

function genOverlapSampler()::OverlapSampler
    rightOp = genIdentity()
    genOneBodySampler(rightOp)
end


const MonomialMul{T<:Real, D} = Associate{Left, typeof(*), FloatingMonomial{T, D}}

const MultipoleMomentSampler{T<:Real, D} = OneBodySampler{MonomialMul{T, D}}

function genMultipoleMomentSampler(fm::FloatingMonomial)::MultipoleMomentSampler
    rightOp = Associate(Left(), *, fm)
    genOneBodySampler(rightOp)
end


struct CoulombMultiPointPotential{T<:Real, D} <: MonoTermOperator
    approx::GaussApproxInverse{T}
    source::MemoryPair{T, NTuple{D, T}}
    charge::T

    function CoulombMultiPointPotential(sourceCharges::AbstractVector{T}, 
                                        sourceCoords::AbstractVector{NonEmptyTuple{T, D}}, 
                                        testCharge::T) where {T<:Real, D}
        checkEmptiness(sourceCharges, :sourceCharges)
        approxField = GaussApproxInverse(T)
        new{T, D+1}(approxField, MemoryPair(sourceCharges, sourceCoords), testCharge)
    end
end

function modifyFunction(op::CoulombMultiPointPotential{T, D}, 
                        f::AbstractTypedCarteFunc{C, D}) where 
                       {T<:Real, C<:RealOrComplex{T}, D}
    function CoulombPointPotentialAccumulator(input::NTuple{D, T})
        res = zero(strictTypeJoin(T, C))
        factor = f(input) * op.charge
        for (charge, coord) in op.source
            res += op.approx(norm(coord .- input)) * factor * charge
        end
        res
    end
end

const CoulombMultiPointSampler{T<:Real, D} = OneBodySampler{CoulombMultiPointPotential{T, D}}

function genCoulombMultiPointSampler(sourceCharges::AbstractVector{T}, 
                                     sourceCoords::AbstractVector{NonEmptyTuple{T, D}}, 
                                     testCharge::T=-one(T)) where {T<:Real, D}
    coreOp = CoulombMultiPointPotential(sourceCharges, sourceCoords, testCharge)
    genOneBodySampler(coreOp)
end

const CoulombInteractionSamplerCore{T<:Real, D} = 
      ComposedApply{2, GaussApproxInverse{T}, ComposedApply{ 2, typeof(LinearAlgebra.norm), 
                                                             StableTupleSub{NTuple{D, T}} }}

const CoulombInteractionSampler{T<:Real, D} = 
      Sandwich{Left, NTuple{2, typeof(*)}, CoulombInteractionSamplerCore{T, D}}

function genCoulombInteractionSampler(::Type{T}, ::Count{D}) where {T, D}
    checkPositivity(D)
    encoder = ComposedApply(StableTupleSub(T, Count(D)), LinearAlgebra.norm, Count(2))
    f = ComposedApply(encoder, GaussApproxInverse(T), Count(2))
    Sandwich(Left(), (*, *), f)
end



const DiagDirectionalDiffSampler{T, D, M, N} = OneBodySampler{DiagonalDiff{T, D, M, N}}

struct KineticEnergySampler{T<:Real, D, N} <: DualTermOperator
    core::DiagDirectionalDiffSampler{T, D, 2, N}

    function KineticEnergySampler{T, D}(::Count{N}) where {T<:Real, D, N}
        checkPositivity(D)
        rightOp = DiagonalDiff(Count(2), ntuple(_->(-one(T)/2), Val(D)), Count(N))
        new{T, D, N}(rightOp|>genOneBodySampler)
    end
end


"""

    genKineticEnergySampler(::Type{T}, ::Count{D}, ::Count{N}=Count(0)) where {T<:Real} -> 
    KineticEnergySampler{T<:Real, D, N}

`T` is the data type for the precision of the operator. `D` is the dimension of the target 
basis function. `N` must be even number to set the accuracy order for the finite difference 
approximation of the kinetic energy operator in case no analytical form of its action on a 
target basis function. When `N` is set to `0`, a preset order will be assigned with respect 
to the form of the target basis function adaptively on the fly.
"""
function genKineticEnergySampler(::Type{T}, ::Count{D}, ::Count{N}=Nil()
                                 ) where {T<:Real, D, N}
    iseven(N) || throw(AssertionError("`N` must be an even number."))
    KineticEnergySampler{T, D}(Count(N))
end


const TypedSampler{T, D} = Union{
    MonomialMul{T, D}, MultipoleMomentSampler{T, D}, DiagDirectionalDiffSampler{T, D}, 
    CoulombMultiPointSampler{T, D}, CoulombInteractionSampler{T, D}
}

const BottomTypedSampler = Union{OverlapSampler}


isParamIndependent(::DirectOperator) = False()

isParamIndependent(::OverlapSampler) = True()

isParamIndependent(::MultipoleMomentSampler) = True()

isParamIndependent(::DiagDirectionalDiffSampler) = True()

isParamIndependent(::CoulombMultiPointSampler) = True()

isParamIndependent(::CoulombInteractionSampler) = True()