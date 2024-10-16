using Base: Slice, OneTo

abstract type Box <: Any end

abstract type StructuredFunction <: Function end

abstract type AbstractMemory{T, N} <: AbstractArray{T, N} end

abstract type ConfigBox <: Box end
abstract type MarkerBox <: Box end
abstract type ParamBox{T} <: Box end
abstract type GraphBox{T} <: Box end
abstract type QueryBox{T} <: Box end

# N: Inner dim, size mutable; O: Outer dim, size immutable
abstract type JaggedOperator{T, N, O} <: StructuredFunction end
abstract type TaggedOperator <: StructuredFunction end
abstract type Evaluator{S} <: StructuredFunction end

abstract type AbstractAmpTensor{T, N} <: JaggedOperator{T, N, 0} end
abstract type AbstractAmplitude{T} <: JaggedOperator{T, 0, 0} end

abstract type SpatialAmpTensor{T, D, N, M} <: AbstractAmpTensor{T, M} end

abstract type SpatialAmplitude{T, D, N} <: AbstractAmplitude{T} end

abstract type GraphNode{T, N, O} <: GraphBox{T} end

abstract type JaggedParam{T, N, O} <: ParamBox{T} end

abstract type IdentityMarker{T} <: MarkerBox end
abstract type StorageMarker{T} <: MarkerBox end

abstract type CompositeParam{T, N, O} <: JaggedParam{T, N, O} end
abstract type PrimitiveParam{T, N} <: JaggedParam{T, N, 0} end

abstract type OperationNode{T, N, O, I} <: GraphNode{T, N, O} end
abstract type ReferenceNode{T, N, O, I} <: GraphNode{T, N, O} end
abstract type ContainerNode{T, N, O}    <: GraphNode{T, N, O} end

abstract type ParamBatch{T, N, I, O} <: CompositeParam{T, N, O} end
abstract type ParamToken{T, N, I} <: CompositeParam{T, N, 0} end

abstract type ParamLink{T, N, I, O} <: ParamBatch{T, N, I, O} end
abstract type ParamNest{T, N, I, O} <: ParamBatch{T, N, I, O} end

abstract type BaseParam{T, N, I} <: ParamToken{T, N, I} end
abstract type LinkParam{T, N, I} <: ParamToken{T, N, I} end

abstract type SpatialOrbital{T, D, F} <: SpatialAmplitude{T, D, 1} end

abstract type SpatialOrbPlex{T, D, F, O} <: SpatialAmpTensor{T, D, 1, O} end

const SpatialOrbitals{T, D, F} = Union{SpatialOrbital{T, D, F}, SpatialOrbPlex{T, D, F}}

# abstract type ManyParticleState{T} <: Any end

# abstract type SpinfulState{T, N} <: ManyParticleState{T} end

# abstract type FermionicState{T, 1} <: SpinfulState{T, 1} end

@enum TernaryNumber::Int8 begin
    TUS0 = 0
    TPS1 = 1
    TPS2 = 2
end

const Dim0GNode{T} = GraphNode{T, 0, 0}
const DimIGNode{T, N} = GraphNode{T, N, 0}
const DimOGNode{T, N} = GraphNode{T, 0, N}
const DimSGNode{T, N} = Union{DimIGNode{T, N}, DimOGNode{T, N}}

const ElementalParam{T} = JaggedParam{T, 0, 0}
const InnerSpanParam{T, N} = JaggedParam{T, N, 0}
const OuterSpanParam{T, O} = JaggedParam{T, 0, O}
const SingleDimParam{T, N} = Union{InnerSpanParam{T, N}, OuterSpanParam{T, N}}

const AVectorOrNTuple{T, NNMO} = Union{Tuple{T, Vararg{T, NNMO}}, AbstractVector{<:T}}
const NonEmptyTuple{T, NMO} = Tuple{T, Vararg{T, NMO}}

const IntOrMiss = Union{Int, Missing}
const SymOrMiss = Union{Symbol, Missing}
const AbtArray0D{T} = AbstractArray{T, 0}

const RefVal = Base.RefValue

const RealOrComplex{T<:Real} = Union{T, Complex{T}}

const NonEmptyTupleOrAbtArray{T, A<:AbstractArray{T}} = Union{NonEmptyTuple{T}, A}

const AbtVecOfAbtArr{T} = AbstractVector{<:AbstractArray{T}}
const JaggedAbtArray{T, N, O} = AbstractArray{<:AbstractArray{T, N}, O}
const AbtArrayOr{T} = Union{T, AbstractArray{T}}
const AbtArr210L{T} = Union{T, AbstractArray{T}, JaggedAbtArray{T}}
const AbtMemory0D{T} = AbstractMemory{T, 0}

const TernaryNTupleUnion{T} = Union{(NTuple{N, T} for N in 1:3)...}
const ParamInputType{T} = TernaryNTupleUnion{JaggedParam{T}}
const ParamInput{T, N} = NTuple{N, JaggedParam{T}}

const ParamSetEle{T} = Union{AbstractVector{<:ElementalParam{T}}, InnerSpanParam{T}}
const DimParamSet{T} = AbstractVector{<:ParamSetEle{T}}

const ParamFunctor{T, N, I} = Union{BaseParam{T, N, I}, ParamLink{T, N, I}}
const ParamPointer{T, N, I} = Union{LinkParam{T, N, I}, ParamNest{T, N, I}}