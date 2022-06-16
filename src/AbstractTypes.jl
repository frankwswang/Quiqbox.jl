export FloatingGTBasisFuncs, CompositeGTBasisFuncs, AbstractGaussFunc, GTBasisFuncs, 
       AbstractGTBasisFuncs, AbstractSpatialPoint

abstract type QuiqboxContainer <: Any end

abstract type MetaParameter <: Any end

abstract type StructFunction{F} <:Function end


abstract type QuiqboxVariableBox <: QuiqboxContainer end

abstract type MetaParam{T} <: MetaParameter end

abstract type ParameterizedFunction{P, F} <: StructFunction{F} end


abstract type QuiqboxParameter{ParameterT, ContainerT} <: QuiqboxVariableBox end
abstract type QuiqboxDataBox <: QuiqboxVariableBox end

abstract type ImmutableParameter{ContainerT, DataT} <: QuiqboxParameter{ImmutableParameter, ContainerT} end
abstract type SemiMutableParameter{ContainerT, DataT} <: QuiqboxParameter{SemiMutableParameter, ContainerT} end
abstract type MutableParameter{ContainerT, DataT} <: QuiqboxParameter{MutableParameter, ContainerT} end

abstract type ImmutableDataBox <: QuiqboxDataBox end
abstract type MutableDataBox <: QuiqboxDataBox end

abstract type AbstractHartreeFockFinalValue <: ImmutableDataBox end
abstract type AbstractBasisSetData <: ImmutableDataBox end
abstract type MolecularDataBox <: ImmutableDataBox end

abstract type HartreeFockintermediateData <: MutableDataBox end

abstract type DifferentiableParameter{ContainerT, DataT} <: MutableParameter{ContainerT, DataT} end
abstract type ConfigBox{ContainerT, MethodT} <: MutableParameter{ContainerT, Any} end

abstract type AbstractSpatialPoint{D, T} <: DifferentiableParameter{AbstractSpatialPoint, T} end

abstract type HartreeFockFinalValue{T} <: AbstractHartreeFockFinalValue end

abstract type BasisSetData{D, T, BT} <: AbstractBasisSetData end

abstract type MolecularCoefficients <: MolecularDataBox end


abstract type QuiqboxBasis{T} <: QuiqboxContainer end

abstract type SpatialOrbital{T} <: QuiqboxBasis{T} end
abstract type SpinOrbital{T} <: QuiqboxBasis{T} end

abstract type StructSpatialBasis{T} <: SpatialOrbital{T} end

abstract type AbstractMolOrbital{T} <: SpinOrbital{T} end

abstract type AbstractGaussFunc{T} <: StructSpatialBasis{T} end
abstract type AbstractGTBasisFuncs{D, T} <: StructSpatialBasis{T} end

abstract type MolecularHartreeFockCoefficient{NN, N} <: MolecularCoefficients end

abstract type GTBasisFuncs{OrbitalN, D, NumberT}  <: AbstractGTBasisFuncs{D, NumberT} end

abstract type CompositeGTBasisFuncs{NofLinearlyCombinedBasis, NofOrbital, D, NumberT}  <: GTBasisFuncs{NofOrbital, D, NumberT} end

abstract type FloatingGTBasisFuncs{𝑙, GaussFuncN, OrbitalN, PointT, D, NumberT} <: CompositeGTBasisFuncs{1, OrbitalN, D, NumberT} end