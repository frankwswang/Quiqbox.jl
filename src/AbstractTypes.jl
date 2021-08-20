abstract type QuiqboxContainer <: Any end


abstract type QuiqboxVariableBox <: QuiqboxContainer end

abstract type QuiqboxParameter{ParameterT, ContainerT} <: QuiqboxVariableBox end
abstract type QuiqboxDataBox <: QuiqboxVariableBox end

abstract type ImmutableParameter{ContainerT, DataT} <: QuiqboxParameter{ImmutableParameter, ContainerT} end
abstract type MutableParameter{ContainerT, DataT} <: QuiqboxParameter{MutableParameter, ContainerT} end

abstract type ImmutableDataBox <: QuiqboxDataBox end
abstract type MutableDataBox <: QuiqboxDataBox end

abstract type AbstractHartreeFockFinalValue <: ImmutableDataBox end
abstract type AbstractBasisSetData <: ImmutableDataBox end
abstract type MolecularDataBox <: ImmutableDataBox end

abstract type HartreeFockintermediateData <: MutableDataBox end

abstract type DifferentiableParameter{ContainerT, DataT} <: MutableParameter{ContainerT, DataT} end

abstract type HartreeFockFinalValue{T} <: AbstractHartreeFockFinalValue end

abstract type BasisSetData{N} <: AbstractBasisSetData end

abstract type MolecularCoefficients <: MolecularDataBox end


abstract type QuiqboxBasis <: QuiqboxContainer end

abstract type SpatialOrbital <: QuiqboxBasis end
abstract type SpinOrbital <: QuiqboxBasis end

abstract type StructSpatialBasis <: SpatialOrbital end

abstract type AbstractMolOrbital <: SpinOrbital end

abstract type NucleusCenteredBasis <: StructSpatialBasis end
abstract type FloatingBasis <: StructSpatialBasis end

abstract type AbstractGaussFunc <: NucleusCenteredBasis end

abstract type AbstractFloatingGTBasisFunc <: FloatingBasis end

abstract type MolecularHartreeFockCoefficient{NucleiC, ElectronC} <: MolecularCoefficients end

abstract type FloatingGTBasisFunc{Subshell, GaussFuncN, OrbitalN} <: AbstractFloatingGTBasisFunc end