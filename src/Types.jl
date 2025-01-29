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

abstract type CompositePointer <: ConfigBox end
abstract type StructuredType <: ConfigBox end

abstract type IntegralProcessCache{T, D} <: SpatialProcessCache{T, D} end

abstract type ActivePointer <: CompositePointer end
abstract type StaticPointer{P<:ActivePointer} <: CompositePointer end

abstract type PointerStack{L, U} <: ActivePointer end
abstract type EntryPointer <: ActivePointer end

abstract type DimensionalEvaluator{T, D, F} <: TypedEvaluator{T, F} end

abstract type OrbitalNormalizer{T, D} <: AmplitudeNormalizer{T, D, 1} end
abstract type OrbitalIntegrator{T, D} <: AmplitudeIntegrator{T, D, 1} end
# M: Particle number
abstract type SpatialAmplitude{T, D, M} <: ParamBoxFunction{T} end

abstract type GraphNode{T, N, O} <: GraphBox{T} end

abstract type ParamBox{T, N, O} <: StateBox{T} end

abstract type IdentityMarker{T} <: MarkerBox end
abstract type StorageMarker{T} <: MarkerBox end

abstract type CompositeParam{T, N, O} <: ParamBox{T, N, O} end
abstract type PrimitiveParam{T, N} <: ParamBox{T, N, 0} end

abstract type OperationNode{T, N, O, I} <: GraphNode{T, N, O} end
abstract type ReferenceNode{T, N, O, I} <: GraphNode{T, N, O} end
abstract type ContainerNode{T, N, O}    <: GraphNode{T, N, O} end

abstract type ParamBatch{T, N, I, O} <: CompositeParam{T, N, O} end
abstract type ParamToken{T, N, I} <: CompositeParam{T, N, 0} end

abstract type ParamLink{T, N, I, O} <: ParamBatch{T, N, I, O} end
abstract type ParamNest{T, N, I, O} <: ParamBatch{T, N, I, O} end

abstract type BaseParam{T, N, I} <: ParamToken{T, N, I} end
abstract type LinkParam{T, N, I} <: ParamToken{T, N, I} end

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

const EffectivePtrStack{P<:PointerStack} = Union{P, StaticPointer{P}}

const Dim0GNode{T} = GraphNode{T, 0, 0}
const DimIGNode{T, N} = GraphNode{T, N, 0}
const DimOGNode{T, N} = GraphNode{T, 0, N}
const DimSGNode{T, N} = Union{DimIGNode{T, N}, DimOGNode{T, N}}

const ElementalParam{T} = ParamBox{T, 0, 0}
const InnerSpanParam{T, N} = ParamBox{T, N, 0}
const OuterSpanParam{T, O} = ParamBox{T, 0, O}
const FlattenedParam{T, N} = Union{InnerSpanParam{T, N}, OuterSpanParam{T, N}}

const AVectorOrNTuple{T, NNMO} = Union{Tuple{T, Vararg{T, NNMO}}, AbstractVector{<:T}}
const NonEmptyTuple{T, NMO} = Tuple{T, Vararg{T, NMO}}
const N12Tuple{T} = Union{Tuple{T}, NTuple{2, T}}
const N24Tuple{T} = Union{NTuple{2, T}, NTuple{4, T}}

const MissingOr{T} = Union{Missing, T}
const AbtArray0D{T} = AbstractArray{T, 0}

const RefVal = Base.RefValue

const RealOrComplex{T<:Real} = Union{T, Complex{T}}
const ParamOrValue{T} = Union{ElementalParam{T}, T}
const ParOrValVec{T} = AbstractVector{<:ParamOrValue{T}}

const NonEmpTplOrAbtArr{T, N, A<:AbstractArray{<:T, N}} = Union{NonEmptyTuple{T}, A}

const AbtVecOfAbtArr{T} = AbstractVector{<:AbstractArray{T}}
const JaggedAbtArray{T, N, O} = AbstractArray{<:AbstractArray{T, N}, O}
const AbtArrayOr{T} = Union{T, AbstractArray{T}}
const AbtArr210L{T} = Union{T, AbstractArray{T}, JaggedAbtArray{T}}
const AbtMemory0D{T} = AbstractMemory{T, 0}

const TriTupleUnion{T} = Union{(NTuple{N, T} for N in 1:3)...}
const TetraTupleUnion{T} = Union{(NTuple{N, T} for N in 1:4)...}
const ParamInputType{T} = TriTupleUnion{ParamBox{T}}
const ParamInput{T, N} = NTuple{N, ParamBox{T}}

const EleParamSpanVecTyped{T} = AbstractVector{<:ElementalParam{T}}
const EleParamSpanVecUnion{T} = AbstractVector{<:ElementalParam{<:T}}

const PrimParamEle{T} = Union{EleParamSpanVecTyped{T}, InnerSpanParam{T}}
const PrimParamVec{T} = AbstractVector{<:PrimParamEle{T}}

const FlatParamEle{T} = Union{EleParamSpanVecTyped{T}, FlattenedParam{T}}
const FlatParamVec{T} = AbstractVector{<:FlatParamEle{T}}

const MiscParamEle{T} = Union{FlatParamVec{T}, ParamBox{T}}
const MiscParamVec{T} = AbstractVector{<:MiscParamEle{T}}

const ParamBoxTypedArr{T, N} = AbstractArray{<:ParamBox{T}, N}
const ParamBoxUnionArr{P<:ParamBox, N} = AbstractArray{<:P, N}

const AbstractFlatParamSet{T, V<:EleParamSpanVecUnion{<:T}, P<:FlattenedParam{<:T}} = 
      AbstractVector{Union{V, P}}
const AbstractMiscParamSet{T, S<:AbstractFlatParamSet{T}, P<:ParamBox{<:T}} = 
      AbstractVector{Union{S, P}}

const AbstractParamSet{T} = Union{AbstractFlatParamSet{T}, AbstractMiscParamSet{T}}
const TypedParamSetVec{T} = Union{FlatParamVec{T}, ParamBoxTypedArr{T, 1}}

const DirectParamSource{T} = Union{AbstractParamSet{T}, TypedParamSetVec{T}}
const ViewedParamSource{T, S<:DirectParamSource{T}, P} = ViewedObject{S, P}
const GeneralParamInput{T, S<:DirectParamSource{T}} = Union{S, ViewedParamSource{T, S}}

const ParamFunctor{T, N, I} = Union{BaseParam{T, N, I}, ParamLink{T, N, I}}
const ParamPointer{T, N, I} = Union{LinkParam{T, N, I}, ParamNest{T, N, I}}

const MissSymInt = MissingOr{Union{Symbol, Int}}

const AbstractEqualityDict = Union{EqualityDict, Dict}


import Base: size, firstindex, lastindex, getindex, setindex!, iterate, length, similar

import Base: isempty, collect, get, keys, values, getproperty, ==, hash