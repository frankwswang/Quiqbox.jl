abstract type Box <: Any end

abstract type EqualityDict{K, T} <: AbstractDict{K, T} end

abstract type CompositeFunction <: Function end # composite-type function
abstract type FieldlessFunction <: Function end # singleton function

abstract type AbstractMemory{T, N} <: AbstractArray{T, N} end

abstract type ConfigBox <: Box end
abstract type MarkerBox <: Box end
abstract type StateBox{T} <: Box end
abstract type GraphBox{T} <: Box end
abstract type QueryBox{T} <: Box end

# N: Inner dim, size mutable; O: Outer dim, size immutable
abstract type DualSpanFunction{T, N, O} <: CompositeFunction end
abstract type StatefulFunction{T} <: CompositeFunction end
abstract type FunctionComposer <: CompositeFunction end
abstract type Evaluator{F} <: CompositeFunction end
abstract type Mapper <: CompositeFunction end
abstract type Getter <: CompositeFunction end
const Encoder = Union{Getter, Mapper}

abstract type SpatialProcessCache{T, D} <: QueryBox{T} end
abstract type IntegralData{T, S} <: QueryBox{T} end
abstract type ViewedObject{T, P} <: QueryBox{T} end
abstract type CustomCache{T} <: QueryBox{T} end

abstract type AmplitudeNormalizer{T, D, N} <: StatefulFunction{T} end
abstract type AmplitudeIntegrator{T, D, N} <: StatefulFunction{T} end
abstract type ParamBoxFunction{T} <: StatefulFunction{T} end

abstract type FunctionModifier <: FunctionComposer end # Modify a function
abstract type FunctionCombiner <: FunctionComposer end # Combine functions together

abstract type TypedEvaluator{T, F} <: Evaluator{F} end

abstract type DirectOperator <: FunctionModifier end

abstract type ParamFuncBuilder{F} <: FunctionCombiner end

abstract type StructuredInfo <: ConfigBox end
abstract type StructuredType <: ConfigBox end

abstract type FieldParamPointer <: StructuredInfo end

abstract type IntegralProcessCache{T, D} <: SpatialProcessCache{T, D} end

abstract type DimensionalEvaluator{T, D, F} <: TypedEvaluator{T, F} end

abstract type OrbitalNormalizer{T, D} <: AmplitudeNormalizer{T, D, 1} end
abstract type OrbitalIntegrator{T, D} <: AmplitudeIntegrator{T, D, 1} end
# M: Particle number
abstract type SpatialAmplitude{T, D, M} <: ParamBoxFunction{T} end

abstract type IdentityMarker{T} <: MarkerBox end
abstract type StorageMarker{T} <: MarkerBox end

abstract type EvalDimensionalFunc{T, D, F} <: DimensionalEvaluator{T, D, F} end

abstract type OrbitalBasis{T, D, F} <: SpatialAmplitude{T, D, 1} end
abstract type FieldAmplitude{T, D} <: SpatialAmplitude{T, D, 1} end

# abstract type ManyParticleState{T} <: Any end

# abstract type SpinfulState{T, N} <: ManyParticleState{T} end

# abstract type FermionicState{T, 1} <: SpinfulState{T, 1} end

@enum TernaryNumber::Int8 begin
    TUS0 = 0
    TPS1 = 1
    TPS2 = 2
end


const AVectorOrNTuple{T, NNMO} = Union{Tuple{T, Vararg{T, NNMO}}, AbstractVector{<:T}}
const NonEmptyTuple{T, NMO} = Tuple{T, Vararg{T, NMO}}
const N12Tuple{T} = Union{Tuple{T}, NTuple{2, T}}
const N24Tuple{T} = Union{NTuple{2, T}, NTuple{4, T}}

const MissingOr{T} = Union{Missing, T}
const NothingOr{T} = Union{Nothing, T}
const AbtArray0D{T} = AbstractArray{T, 0}

const RefVal = Base.RefValue

const NonEmpTplOrAbtArr{T, N, A<:AbstractArray{<:T, N}} = Union{NonEmptyTuple{T}, A}

const AbtVecOfAbtArr{T} = AbstractVector{<:AbstractArray{T}}
const JaggedAbtArray{T, N, O} = AbstractArray{<:AbstractArray{T, N}, O}
const AbtArrayOr{T} = Union{T, AbstractArray{T}}
const AbtArr210L{T} = Union{T, AbstractArray{T}, JaggedAbtArray{T}}
const AbtMemory0D{T} = AbstractMemory{T, 0}

const TriTupleUnion{T} = Union{(NTuple{N, T} for N in 1:3)...}
const TetraTupleUnion{T} = Union{(NTuple{N, T} for N in 1:4)...}

const MissSymInt = MissingOr{Union{Symbol, Int}}

const AbstractEqualityDict = Union{EqualityDict, Dict}


import Base: size, firstindex, lastindex, getindex, setindex!, iterate, length, similar

import Base: isempty, collect, get, keys, values, getproperty, ==, hash

import Base: +, -