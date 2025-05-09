const OverlapSampler = Multiplier{ComplexConj, Identity}

function genOverlapSampler()::OverlapSampler
    rightOp = genIdentity()
    Correlate(*, (ComplexConj(), rightOp))
end


const MonomialMul{T<:Real, D} = Associate{Left, typeof(*), FloatingMonomial{T, D}}

const MultipoleMomentSampler{T<:Real, D} = Multiplier{ComplexConj, MonomialMul{T, D}}

function genMultipoleMomentSampler(fm::FloatingMonomial)::MultipoleMomentSampler
    rightOp = Associate(Left(), *, fm)
    Correlate(*, (ComplexConj(), rightOp))
end


function inverseDistance(coord1::NonEmptyTuple{Real, D}, 
                         coord2::NonEmptyTuple{Real, D}) where {D}
    (coord1 .- coord2) |> LinearAlgebra.norm |> inv
end

const CoulombRepulsionSampler = 
      Sandwich{Left, NTuple{2, typeof(*)}, typeof(inverseDistance)}

function genCoulombRepulsionSampler()
    Sandwich(Left(), (*, *), inverseDistance)
end