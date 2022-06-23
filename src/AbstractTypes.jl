export FloatingGTBasisFuncs, CompositeGTBasisFuncs, AbstractGaussFunc, GTBasisFuncs, 
       AbstractGTBasisFuncs, AbstractSpatialPoint

abstract type QuiqboxContainer <: Any end

abstract type MetaParameter <: Any end

abstract type StructFunction{F} <:Function end


abstract type QuiqboxVariableBox{T} <: QuiqboxContainer end

abstract type MetaParam{T} <: MetaParameter end

abstract type ParameterizedFunction{P, F} <: StructFunction{F} end


abstract type QuiqboxParameter{T, ParameterT, ContainerT} <: QuiqboxVariableBox{T} end
abstract type QuiqboxDataBox{T} <: QuiqboxVariableBox{T} end

abstract type ImmutableParameter{DataT, ContainerT} <: QuiqboxParameter{DataT, ImmutableParameter, ContainerT} end
abstract type SemiMutableParameter{DataT, ContainerT} <: QuiqboxParameter{DataT, SemiMutableParameter, ContainerT} end
abstract type MutableParameter{DataT, ContainerT} <: QuiqboxParameter{DataT, MutableParameter, ContainerT} end

abstract type ImmutableDataBox{T} <: QuiqboxDataBox{T} end
abstract type MutableDataBox{T} <: QuiqboxDataBox{T} end

abstract type AbstractHartreeFockFinalValue{T} <: ImmutableDataBox{T} end
abstract type AbstractBasisSetData{T} <: ImmutableDataBox{T} end
abstract type ManyFermionDataBox{T, N} <: ImmutableDataBox{T} end

abstract type HartreeFockintermediateData{T} <: MutableDataBox{T} end

abstract type DifferentiableParameter{DataT, ContainerT} <: SemiMutableParameter{DataT, ContainerT} end
abstract type ConfigBox{T, ContainerT, MethodT} <: MutableParameter{ContainerT, T} end

abstract type AbstractSpatialPoint{T, D} <: DifferentiableParameter{T, AbstractSpatialPoint} end

abstract type HartreeFockFinalValue{T, HFT} <: AbstractHartreeFockFinalValue{T} end

abstract type BasisSetData{T, D, BT} <: AbstractBasisSetData{T} end

abstract type MatterHFData{T, NN, N} <: ManyFermionDataBox{T, N} end


abstract type QuiqboxBasis{T} <: QuiqboxContainer end

abstract type SpatialOrbital{T} <: QuiqboxBasis{T} end
abstract type SpinOrbital{T} <: QuiqboxBasis{T} end

abstract type StructSpatialBasis{T} <: SpatialOrbital{T} end

abstract type AbstractMolOrbital{T} <: SpinOrbital{T} end

abstract type AbstractGaussFunc{T} <: StructSpatialBasis{T} end
abstract type AbstractGTBasisFuncs{T, D, OrbitalN} <: StructSpatialBasis{T} end

abstract type GTBasisFuncs{NumberT, D, OrbitalN}  <: AbstractGTBasisFuncs{NumberT, D, OrbitalN} end

abstract type CompositeGTBasisFuncs{NumberT, D, NofLinearlyCombinedBasis, OrbitalN}  <: GTBasisFuncs{NumberT, D, OrbitalN} end

abstract type FloatingGTBasisFuncs{NumberT, D, 𝑙, GaussFuncN, PointT, OrbitalN} <: CompositeGTBasisFuncs{NumberT, D, 1, OrbitalN} end


const CGTBasisFuncsON{ON} = CompositeGTBasisFuncs{<:Any, <:Any, <:Any, ON}
const CGTBasisFuncs1O{T, D, BN} = CompositeGTBasisFuncs{T, D, BN, 1}

const FGTBasisFuncsON{ON} = FloatingGTBasisFuncs{<:Any, <:Any, <:Any, <:Any, <:Any, ON}
const FGTBasisFuncs1O{T, D, 𝑙, GN, PT} = FloatingGTBasisFuncs{T, D, 𝑙, GN, PT, 1}