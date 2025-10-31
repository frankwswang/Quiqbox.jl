abstract type Box <: Any end
abstract type AnyInterface <: Any end
abstract type CallableObject <: Any end

abstract type EqualityDict{K, V} <: AbstractDict{K, V} end

abstract type CompositeFunction <: Function end # composite-type function

abstract type CustomMemory{T, N} <: AbstractArray{T, N} end

abstract type ConfigBox <: Box end
abstract type MarkerBox <: Box end
abstract type StateBox{T} <: Box end
abstract type GraphBox{T} <: Box end
abstract type QueryBox{T} <: Box end

# N: Inner dim, size mutable; O: Outer dim, size immutable
abstract type ParticleFunction{D, N} <: CompositeFunction end # N: Particle number
abstract type AbstractParamFunc <: CompositeFunction end
abstract type GraphEvaluator{G} <: CompositeFunction end
abstract type TypedEvaluator{T} <: CompositeFunction end
abstract type TraitAction{I} <: CompositeFunction end
abstract type Modifier <: CompositeFunction end
abstract type Mapper <: CompositeFunction end

abstract type IntegralData{T} <: QueryBox{T} end
abstract type QueryCache{K, T} <: QueryBox{T} end

abstract type DirectOperator{N} <: Modifier end # N: Number of input functions

abstract type EstimatorConfig{T} <: ConfigBox end
abstract type CustomAccessor <: ConfigBox end
abstract type StructuredType <: ConfigBox end
abstract type CustomRange <: ConfigBox end

abstract type IdentityMarker{T} <: MarkerBox end
abstract type StorageMarker{T} <: MarkerBox end

abstract type ValueType <: StructuredType end

abstract type TypedParamFunc{T} <: AbstractParamFunc end

abstract type OrbitalBasis{T, D, F} <: ParticleFunction{D, 1} end
abstract type FieldAmplitude{T, D} <: ParticleFunction{D, 1} end

abstract type SymbolType <: ValueType end

# abstract type ManyParticleState{T} <: Any end

# abstract type SpinfulState{T, N} <: ManyParticleState{T} end

# abstract type FermionicState{T, 1} <: SpinfulState{T, 1} end


@enum TernaryNumber::Int8 begin
    TUS0 = 0
    TPS1 = 1
    TPS2 = 2
end


@enum OctalNumber::Int8 begin
    OUS0 = 0 # (false, false, false)
    OPS1 = 1 # (true,  false, false)
    OPS2 = 2 # (false, true,  false)
    OPS3 = 3 # (true,  true,  false)
    OPS4 = 4 # (false, false, true )
    OPS5 = 5 # (true,  false, true )
    OPS6 = 6 # (false, true,  true )
    OPS7 = 7 # (true,  true,  true )
end


const N12Tuple{T} = Union{NTuple{1, T}, NTuple{2, T}}
const N24Tuple{T} = Union{NTuple{2, T}, NTuple{4, T}}
const N1N2Tuple{T} = NTuple{1, NTuple{2, T}}
const N2N2Tuple{T} = NTuple{2, NTuple{2, T}}
const N12N2Tuple{T} = N12Tuple{NTuple{2, T}}
const NonEmptyTuple{T, NMO} = Tuple{T, Vararg{T, NMO}}
const TriTupleUnion{T} = Union{(NTuple{N, T} for N in 1:3)...}
const GeneralTupleUnion{T<:Tuple} = Union{T, NamedTuple{<:Any, <:T}}

const AbtArray0D{T} = AbstractArray{T, 0}
const AbtBottomVector = AbstractArray{Union{}, 1}
const AbstractMemory{T} = Union{CustomMemory{T}, Memory{T}}
const AbtVecOfAbtArr{T} = AbstractVector{<:AbstractArray{T}}

const GeneralCollection = Union{AbstractArray, Tuple, NamedTuple}

const LinearSequence{T} = Union{AbstractVector{T}, NonEmptyTuple{T}}
const NumberSequence{T<:Number} = LinearSequence{T}


const AbstractEqualityDict = Union{EqualityDict, Dict}

const RealOrComplex{T<:Real} = Union{Complex{T}, T}

const MissingOr{T} = Union{Missing, T}
const NothingOr{T} = Union{Nothing, T}

const FunctionChainUnion{F<:Function} = Union{
    AbstractMemory{<:F}, GeneralTupleUnion{NonEmptyTuple{F}}
}

const AbstractRealCoordVector{T<:Real} = Union{
    AbstractVector{<:AbstractVector{T}}, (AbstractVector{NonEmptyTuple{T, D}} where {D})
}


const RefVal = Base.RefValue

const ArithmeticOperator = Union{typeof(+), typeof(-), typeof(*), typeof(/)}


import Base: iterate, size, getindex, setindex!, IndexStyle, zero, similar

import Base: get, get!, haskey, hash, collect, length, eltype

import Base: eachindex, firstindex, lastindex

import Base: broadcastable

import Base: +, -, ==, isless

import Base: Int

import Base: show