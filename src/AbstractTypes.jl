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

abstract type BasisSetData{BT} <: AbstractBasisSetData end

abstract type MolecularCoefficients <: MolecularDataBox end


abstract type QuiqboxBasis <: QuiqboxContainer end

abstract type SpatialOrbital <: QuiqboxBasis end
abstract type SpinOrbital <: QuiqboxBasis end

abstract type StructSpatialBasis <: SpatialOrbital end

abstract type AbstractMolOrbital <: SpinOrbital end

abstract type AbstractGaussFunc{T} <: StructSpatialBasis end
abstract type AbstractGTBasisFuncs <: StructSpatialBasis end

abstract type MolecularHartreeFockCoefficient{NN, N} <: MolecularCoefficients end

abstract type GTBasisFuncs{OrbitalN}  <: AbstractGTBasisFuncs end

abstract type CompositeGTBasisFuncs{NofLinearlyCombinedBasis, NofOrbital, D, NumberT}  <: GTBasisFuncs{NofOrbital} end

abstract type FloatingGTBasisFuncs{𝑙, GaussFuncN, OrbitalN, PointT, D, NumberT} <: CompositeGTBasisFuncs{1, OrbitalN, D, NumberT} end