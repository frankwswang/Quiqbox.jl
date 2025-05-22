using LinearAlgebra

const MonoTermOperator = DirectOperator{1}
const DualTermOperator = DirectOperator{2}


(op::DirectOperator{N})(fs::Vararg{Function, N}) where {N} = 
modifyFunction(op, fs...)::Function


struct ComplexConj <: MonoTermOperator end

modifyFunction(::ComplexConj, f::Function) = conj ∘ f


struct MultiMonoApply{N, O<:NTuple{N, MonoTermOperator}} <: MonoTermOperator
    chain::O
end

const Identity = MultiMonoApply{0, Tuple{}}

genIdentity() = MultiMonoApply{0, Tuple{}}(())

function modifyFunction(op::MultiMonoApply, f::Function)
    term = f
    for operator in op.chain
        term = operator(term)
    end
    term
end


struct Associate{A<:Lateral, J<:Function, F<:Function} <: MonoTermOperator
    position::A
    coupler::J
    term::F
end


function modifyFunction(op::Associate, term::Function)
    fL, fR = op.position isa Left ? (op.term, term) : (term, op.term)
    PairCoupler(op.coupler, fL, fR)
end


struct Correlate{J<:Function, F<:NTuple{2, MonoTermOperator}} <: DualTermOperator
    coupler::J
    dresser::F
end

const Multiplier{OL<:MonoTermOperator, OR<:MonoTermOperator} = 
      Correlate{typeof(*), Tuple{OL, OR}}

function modifyFunction(op::Correlate, termPair::Vararg{Function, 2})
    fL, fR = termPair .|> op.dresser
    PairCoupler(op.coupler, fL, fR)
end


struct Sandwich{A<:Lateral, J<:NTuple{2, Function}, F<:Function} <: DualTermOperator
    associativity::A
    coupler::J
    core::F
end

function modifyFunction(op::Sandwich, termL::Function, termR::Function)
    jL, jR = op.coupler

    if op.associativity isa Left
        fL = PairCoupler(jL, SelectHeader{2, 1}(termL), op.core)
        PairCoupler(jR, fL, SelectHeader{2, 2}(termR))
    else
        fR = PairCoupler(jR, op.core, SelectHeader{2, 2}(termR))
        PairCoupler(jL, SelectHeader{2, 1}(termL), fR)
    end
end


struct DiagonalDiff{C<:RealOrComplex, D, M, N} <: MonoTermOperator
    direction::NTuple{D, C}

    function DiagonalDiff(::Val{M}, direction::NonEmptyTuple{C}, 
                             ::Val{N}=Val(4)) where {M, C<:RealOrComplex, N}
        checkPositivity(M::Int, true)
        checkPositivity(M::Int, true)
        new{C, length(direction), M, N::Int}(direction)
    end
end

function modifyFunction(op::DiagonalDiff{C1, D, M, N}, term::TypedCarteFunc{C2, D}) where 
                       {T<:Real, C1<:RealOrComplex{T}, C2<:RealOrComplex{T}, D, M, N}
    C = ifelse(C1==C2==T, T, Complex{T})
    mapper = ntuple(Val(D)) do i
        AxialFiniteDiff(TypedReturn(term, C), Val(M), i, Val(N))
    end |> ChainMapper
    PairCoupler(Contract(C, C1, C2), Storage(op.direction), mapper)
end


#! Future development for computation of Hessian
# struct MixedPartialDiff{D} <: MonoTermOperator
#     degree::WeakComp{D}
# end