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
end #! Optimize the composition when coupler is Multiplier and fL === fR


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


# `N` must be even number to set the accuracy order for the finite difference acting on a 
# target function. When `N` is set to `0`, a preset order will be assigned with respect to 
# the form of the target function adaptively on the fly.
struct DiagonalDiff{C<:RealOrComplex, D, M, N} <: MonoTermOperator
    direction::NTuple{D, C}

    function DiagonalDiff(::Count{M}, direction::NonEmptyTuple{C}, ::Count{N}=Nil()) where 
                         {M, C<:RealOrComplex, N}
        checkPositivity(M)
        iseven(N) || throw(AssertionError("`N` must be an even number."))
        new{C, length(direction), M, N}(direction)
    end
end

const GeneralFieldAmplitude{C<:RealOrComplex, D} = 
      Union{TypedReturn{C, <:ParticleFunction{D, 1}}, FieldAmplitude{C, D}}

const AbstractTypedCarteFunc{C<:RealOrComplex, D} = 
      Union{TypedCarteFunc{C, D}, GeneralFieldAmplitude{C, D}}

function modifyFunction(op::DiagonalDiff{C1, D, M, N}, term::AbstractTypedCarteFunc{C2, D}
                        ) where {T<:Real, C1<:RealOrComplex{T}, C2<:RealOrComplex{T}, D, M, 
                                 N}
    C = ifelse(C1==C2==T, T, Complex{T})
    formattedTerm = TypedCarteFunc(term, C, Count(D))
    order = Count(ifelse(N==0, getFiniteDiffApproxOrder(term), N))
    mapper = ntuple(Val(D)) do i
        AxialFiniteDiff(formattedTerm, Count(M), i, order)
    end |> ChainMapper
    PairCoupler(Contract(C, C1, C2), Storage(op.direction, :diffDirection), mapper)
end

getFiniteDiffApproxOrder(::Function) = 6


struct TypedOperator{T, F<:DirectOperator}
    core::F
    type::Type{T}
end

TypedOperator(op::TypedOperator, ::Type{T}) where {T} = TypedOperator(op.core, T)

function modifyFunction(op::TypedOperator{T}, term::Function) where {T}
    TypedReturn(modifyFunction(op.core, term), T)
end


#! Future development for computation of Hessian
# struct MixedPartialDiff{D} <: MonoTermOperator
#     degree::WeakComp{D}
# end