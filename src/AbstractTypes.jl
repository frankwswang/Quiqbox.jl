export FloatingGTBasisFuncs, CompositeGTBasisFuncs, AbstractGaussFunc, GTBasisFuncs, 
       AbstractGTBasisFuncs, AbstractSpatialPoint

abstract type AbstractQuiqboxContainer <: Any end
abstract type MetaParameter <: Any end

abstract type StructuredFunction <:Function end


abstract type StructFunction{F} <: StructuredFunction end

abstract type QuiqboxContainer{T} <: AbstractQuiqboxContainer end

abstract type MetaParam{T} <: MetaParameter end


abstract type CompositeFunction{F1, F2} <: StructFunction{F1} end

abstract type QuiqboxVariableBox{T} <: QuiqboxContainer{T} end


abstract type ChainedFunction{F, O, C} <: CompositeFunction{F, C} end

abstract type QuiqboxParameter{T, ParameterT, ContainerT} <: QuiqboxVariableBox{T} end
abstract type QuiqboxDataBox{T} <: QuiqboxVariableBox{T} end

abstract type ImmutableParameter{DataT, ContainerT} <: QuiqboxParameter{DataT, ImmutableParameter, ContainerT} end
abstract type SemiMutableParameter{DataT, ContainerT} <: QuiqboxParameter{DataT, SemiMutableParameter, ContainerT} end
abstract type MutableParameter{DataT, ContainerT} <: QuiqboxParameter{DataT, MutableParameter, ContainerT} end

abstract type ImmutableDataBox{T} <: QuiqboxDataBox{T} end
abstract type MutableDataBox{T} <: QuiqboxDataBox{T} end

abstract type AbstractHartreeFockFinalValue{T} <: ImmutableDataBox{T} end
abstract type AbstractBasisSetData{T} <: ImmutableDataBox{T} end
abstract type ManyFermionDataBox{T, D, N} <: ImmutableDataBox{T} end

abstract type HartreeFockintermediateData{T} <: MutableDataBox{T} end

abstract type DifferentiableParameter{DataT, ContainerT} <: SemiMutableParameter{DataT, ContainerT} end
abstract type ConfigBox{T, ContainerT, MethodT} <: MutableParameter{ContainerT, T} end

abstract type HartreeFockFinalValue{T, HFT} <: AbstractHartreeFockFinalValue{T} end

abstract type BasisSetData{T, D, BFT} <: AbstractBasisSetData{T} end

abstract type MatterData{T, D, N} <: ManyFermionDataBox{T, D, N} end

abstract type ParameterizedContainer{T} <: QuiqboxContainer{T} end

abstract type DimensionlessParamContainer{T} <: ParameterizedContainer{T} end
abstract type DimensionalParamContainer{T, D} <: ParameterizedContainer{T} end

abstract type PrimitiveBasisFunc{T} <: DimensionlessParamContainer{T} end

abstract type QuiqboxBasis{T, D, ON} <: DimensionalParamContainer{T, D} end
abstract type SpatialStructure{T, D} <: DimensionalParamContainer{T, D} end
abstract type AbstractSpatialPoint{T, D} <: DimensionalParamContainer{T, D} end

abstract type SpatialBasis{T, D, ON} <: QuiqboxBasis{T, D, ON} end
abstract type FermionState{T, D, MaxOccupation} <: QuiqboxBasis{T, D, MaxOccupation} end

abstract type AbstractSpinOrbital{T, D} <: FermionState{T, D, 2} end

abstract type AbstractGaussFunc{T} <: PrimitiveBasisFunc{T} end
abstract type AbstractGTBasisFuncs{T, D, OrbitalN} <: SpatialBasis{T, D, OrbitalN} end

abstract type GTBasisFuncs{NumberT, D, OrbitalN} <: AbstractGTBasisFuncs{NumberT, D, OrbitalN} end

abstract type CompositeGTBasisFuncs{NumberT, D, NofLinearlyCombinedBasis, OrbitalN} <: GTBasisFuncs{NumberT, D, OrbitalN} end

abstract type FloatingGTBasisFuncs{NumberT, D, 𝑙, GaussFuncN, PointT, OrbitalN} <: CompositeGTBasisFuncs{NumberT, D, 1, OrbitalN} end


const CGTBasisFuncsON{ON} = CompositeGTBasisFuncs{<:Any, <:Any, <:Any, ON}
const CGTBasisFuncs1O{T, D, BN} = CompositeGTBasisFuncs{T, D, BN, 1}

const FGTBasisFuncsON{ON} = FloatingGTBasisFuncs{<:Any, <:Any, <:Any, <:Any, <:Any, ON}
const FGTBasisFuncs1O{T, D, 𝑙, GN, PT} = FloatingGTBasisFuncs{T, D, 𝑙, GN, PT, 1}


const SpatialCoordType{T, D, N} = Union{ AbstractVector{<:AbstractVector{<:T}}, 
                                         AbstractVector{NTuple{D, T}}, 
                                         NTuple{N, <:AbstractVector{<:T}}, 
                                         NTuple{N, NTuple{D, T}} }

const VectorOrNTuple{T, N} = Union{NTuple{N, T}, AbstractVector{<:T}}

const ArrayOrNTuple{T, N} = Union{NTuple{N, T}, AbstractArray{<:T}}