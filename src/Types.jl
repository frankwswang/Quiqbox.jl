abstract type Box <: Any end
abstract type AnyInterface <: Any end

abstract type EqualityDict{K, T} <: AbstractDict{K, T} end

abstract type CompositeFunction <: Function end # composite-type function

abstract type AbstractMemory{T, N} <: AbstractArray{T, N} end

abstract type ConfigBox <: Box end
abstract type MarkerBox <: Box end
abstract type StateBox{T} <: Box end
abstract type GraphBox{T} <: Box end
abstract type QueryBox{T} <: Box end

# N: Inner dim, size mutable; O: Outer dim, size immutable
abstract type ParticleFunction{D, N} <: CompositeFunction end # N: Particle number
abstract type StatefulFunction{T} <: CompositeFunction end
abstract type GraphEvaluator{G} <: CompositeFunction end
abstract type TypedEvaluator{T} <: CompositeFunction end
abstract type TraitAction{I} <: CompositeFunction end
abstract type Modifier <: CompositeFunction end
abstract type Mapper <: CompositeFunction end
abstract type Getter <: CompositeFunction end
const Encoder = Union{Getter, Mapper}

abstract type IntegralData{T} <: QueryBox{T} end
abstract type CustomCache{T} <: QueryBox{T} end

abstract type DirectOperator{N} <: Modifier end # N: Number of input functions

abstract type StructuredInfo <: ConfigBox end
abstract type StructuredType <: ConfigBox end

abstract type IdentityMarker{T} <: MarkerBox end
abstract type StorageMarker{T} <: MarkerBox end

abstract type OrbitalBasis{T, D, F} <: ParticleFunction{D, 1} end
abstract type FieldAmplitude{T, D} <: ParticleFunction{D, 1} end

# abstract type ManyParticleState{T} <: Any end

# abstract type SpinfulState{T, N} <: ManyParticleState{T} end

# abstract type FermionicState{T, 1} <: SpinfulState{T, 1} end

@enum TernaryNumber::Int8 begin
    TUS0 = 0
    TPS1 = 1
    TPS2 = 2
end

const RealOrComplex{T<:Real} = Union{Complex{T}, T}

const AVectorOrNTuple{T, NNMO} = Union{Tuple{T, Vararg{T, NNMO}}, AbstractVector{<:T}}
const NonEmptyTuple{T, NMO} = Tuple{T, Vararg{T, NMO}}
const N12Tuple{T} = Union{Tuple{T}, NTuple{2, T}}
const GeneralTupleUnion{T<:Tuple} = Union{T, NamedTuple{<:Any, <:T}}

const MissingOr{T} = Union{Missing, T}
const NothingOr{T} = Union{Nothing, T}
const AbtArray0D{T} = AbstractArray{T, 0}

const TypedIdxerMemory{T} = Memory{Pair{Int, T}}

const RefVal = Base.RefValue

const NonEmpTplOrAbtArr{T, N, A<:AbstractArray{<:T, N}} = Union{NonEmptyTuple{T}, A}
const AbtBottomArray{N} = AbstractArray{Union{}, N}
const AbtBottomVector = AbtBottomArray{1}
const AbtVecOfAbtArr{T} = AbstractVector{<:AbstractArray{T}}
const TriTupleUnion{T} = Union{(NTuple{N, T} for N in 1:3)...}

const AbstractEqualityDict = Union{EqualityDict, Dict}

const FunctionChainUnion{F<:Function} = Union{
    AbstractMemory{<:F}, GeneralTupleUnion{NonEmptyTuple{F}}
}

const BoolVal = Union{Val{true}, Val{false}}

import Base: size, firstindex, lastindex, getindex, setindex!, iterate, length, similar

import Base: isempty, collect, get, keys, values, getproperty, ==, hash

import Base: +, -