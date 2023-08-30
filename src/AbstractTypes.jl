using Base: Slice, OneTo

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
abstract type ManyFermionDataBox{T, D} <: ImmutableDataBox{T} end

abstract type HartreeFockintermediateData{T} <: MutableDataBox{T} end

abstract type VariableBox{T, ContainerT} <: SemiMutableParameter{T, ContainerT} end
abstract type ConfigBox{T, ContainerT, MethodT} <: MutableParameter{T, ContainerT} end

abstract type ParamBox{T, V, F} <: VariableBox{T, ParamBox} end

abstract type HartreeFockFinalValue{T, HFT} <: AbstractHartreeFockFinalValue{T} end

abstract type BasisSetData{T, D, BFT} <: AbstractBasisSetData{T} end

abstract type MatterData{T, D} <: ManyFermionDataBox{T, D} end

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


const SpatialBasis1O{T, D} = SpatialBasis{T, D, 1}
const CGTBasisFuncsON{ON} = CompositeGTBasisFuncs{<:Any, <:Any, <:Any, ON}
const CGTBasisFuncs1O{T, D, BN} = CompositeGTBasisFuncs{T, D, BN, 1}

const FGTBasisFuncsON{ON} = FloatingGTBasisFuncs{<:Any, <:Any, <:Any, <:Any, <:Any, ON}
const FGTBasisFuncs1O{T, D, 𝑙, GN, PT} = FloatingGTBasisFuncs{T, D, 𝑙, GN, PT, 1}


const SpatialCoordType{T, D, NNMO} = Union{ 
    AbstractVector{<:AbstractVector{<:T}}, 
    AbstractVector{NTuple{D, T}}, 
    Tuple{TL, Vararg{TL, NNMO}} where TL<:AbstractVector{<:T}, 
    Tuple{NTuple{D, T}, Vararg{NTuple{D, T}, NNMO}} }

const AVectorOrNTuple{T, NNMO} = Union{Tuple{T, Vararg{T, NNMO}}, AbstractVector{<:T}}

const AArrayOrNTuple{T, NNMO} = Union{Tuple{T, Vararg{T, NNMO}}, AbstractArray{<:T}}


const NTupleOfFGTBF{BN, T, D} = NTuple{BN, FGTBasisFuncs1O{T, D}}
const NTupleOfSB1{BN, T, D} = NTuple{BN, SpatialBasis1O{T, D}}
const NTupleOfSBN{BN, T, D} = NTuple{BN, SpatialBasis{T, D}}


const StrOrSym = Union{String, Symbol}


const MatrixCol{T} = Vector{SubArray{T, 1, Matrix{T}, Tuple{Slice{OneTo{Int}}, Int}, true}}

const Array0D{T} = Array{T, 0}

const Union0D{T} = Union{T, Array0D{T}}

const MTuple{M, T} = NTuple{M, Array0D{T}} # M: mutable

const UTuple{U, T} = NTuple{U, Union0D{T}} # U: union

const NMOTuple{NMO, T} = Tuple{T, Vararg{T, NMO}}

const MMOTuple{MMO, T} = Tuple{Array0D{T}, Vararg{Array0D{T}, MMO}}

const UMOTuple{UMO, T} = Tuple{Union0D{T}, Vararg{Union0D{T}, UMO}}

const NMTTuple{NMO, T} = Tuple{T, T, Vararg{T, NMO}}

const MMTTuple{MMO, T} = Tuple{Array0D{T}, Array0D{T}, Vararg{Array0D{T}, MMO}}

const UMTTuple{UMO, T} = Tuple{Union0D{T}, Union0D{T}, Vararg{Union0D{T}, UMO}}

const TorTuple{T} = Union{T, Tuple{T}}

const TorTupleLong{T} = Union{T, Tuple{T, Vararg{T}}}

const TorAbstractVec{T} = Union{T, AbstractVector{<:T}}

const IntOrNone = Union{Int, Nothing}
const IntOrMiss = Union{Int, Missing}

const MutableIndex = Array0D{IntOrNone}


const MonoParam{T} = OneParam{T}

const DualParam{T} = Tuple{OneParam{T}, OneParam{T}}

const MoreParam{T} = Tuple{OneParam{T}, OneParam{T}, OneParam{T}, Vector{OneParam{T}}}