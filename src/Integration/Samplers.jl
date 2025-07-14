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


function inverseDistance(coord1::NonEmptyTuple{Real, D}, 
                         coord2::NonEmptyTuple{Real, D}) where {D}
    (coord1 .- coord2) |> LinearAlgebra.norm |> inv
end


const PointInverseDistance{T<:Real, D} = RPartial{NTuple{D, T}, typeof(inverseDistance)}

const CoulombPointField{T<:Real, D} = 
      PairCoupler{StableMul{T}, Storage{T}, PointInverseDistance{T, D}}

const CoulombPointFieldSampler{T<:Real, D} = 
      OneBodySampler{Associate{ Left, typeof(*), CoulombPointField{T, D} }}

function genCoulombPointFieldSampler(chargePair::NTuple{2, T}, pointCoord::NTuple{D, T}) where 
                                    {T<:Real, D}
    pid = RPartial(inverseDistance, (pointCoord,))
    cpf = PairCoupler(StableMul(T), Storage(prod(chargePair), :chargePair), pid)
    genOneBodySampler(Associate(Left(), *, cpf))
end


const CoulombInteractionSampler = 
      Sandwich{Left, NTuple{2, typeof(*)}, typeof(inverseDistance)}

function genCoulombInteractionSampler()
    Sandwich(Left(), (*, *), inverseDistance)
end

const DiagDirectionalDiffSampler{T, D, M, N} = OneBodySampler{DiagonalDiff{T, D, M, N}}

struct KineticEnergySampler{T<:Real, D, N} <: MonoTermOperator
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


const ReturnTypedSampler{T} = Union{
    MonomialMul{T}, MultipoleMomentSampler{T}, DiagDirectionalDiffSampler{T}, 
    CoulombPointFieldSampler{T}
}

const StableTypedSampler = Union{OneBodySampler, CoulombInteractionSampler}


isParamIndependent(::DirectOperator) = False()

isParamIndependent(::OverlapSampler) = True()

isParamIndependent(::MultipoleMomentSampler) = True()

isParamIndependent(::DiagDirectionalDiffSampler) = True()

isParamIndependent(::CoulombPointFieldSampler) = True()

isParamIndependent(::CoulombInteractionSampler) = True()