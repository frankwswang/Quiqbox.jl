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
    combiner::J
    term::F
end

function modifyFunction(op::Associate, term::Function)
    fL, fR = op.position isa Left ? (op.term, term) : (term, op.term)
    PairCoupler(op.combiner, fL, fR)
end


struct Correlate{J<:Function, F<:NTuple{2, MonoTermOperator}} <: DualTermOperator
    combiner::J
    dresser::F
end

const Multiplier{OL<:MonoTermOperator, OR<:MonoTermOperator} = 
      Correlate{typeof(*), Tuple{OL, OR}}

function modifyFunction(op::Correlate, termPair::Vararg{Function, 2})
    fL, fR = termPair .|> op.dresser
    PairCoupler(op.combiner, fL, fR)
end


struct Sandwich{A<:Lateral, J<:NTuple{2, Function}, F<:Function} <: DualTermOperator
    associativity::A
    combiner::J
    core::F
end

function modifyFunction(op::Sandwich, termL::Function, termR::Function)
    jL, jR = op.combiner

    if op.associativity isa Left
        fL = PairCoupler(jL, SelectHeader{2, 1}(termL), op.core)
        PairCoupler(jR, fL, SelectHeader{2, 2}(termR))
    else
        fR = PairCoupler(jR, op.core, SelectHeader{2, 2}(termR))
        PairCoupler(jL, SelectHeader{2, 1}(termL), fR)
    end
end