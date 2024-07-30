using Base: Slice, OneTo

export FloatingBasisFuncs, CompositeBasisFuncs, AbstractGaussFunc, AbstractBasisFuncs, 
       AbstractBasisFuncs, AbstractSpatialPoint


abstract type TypeObject <: Any end
abstract type StructuredContainer <: Any end

abstract type StructuredFunction <:Function end
abstract type AbstractMemory{T, N} <: AbstractArray{T, N} end

abstract type StructFunction{T, F} <: StructuredFunction end
abstract type TypedFunction{T, F} <: StructuredFunction end
abstract type ParamFunction{T} <: StructuredFunction end

abstract type Singleton <: TypeObject end

# abstract type CompositeFunction{F1, F2} <: StructFunction{F1} end

abstract type ParamOperator{T, F, N, O} <: StructFunction{T, F} end

abstract type CompositeContainer{T} <: StructuredContainer end
abstract type NoTypeParContainer <: StructuredContainer end

abstract type MixedContainer{T} <: CompositeContainer{T} end
abstract type ParamContainer{T} <: CompositeContainer{T} end
abstract type ValueContainer{T} <: CompositeContainer{T} end

# abstract type ChainedFunction{F, O, C} <: CompositeFunction{F, C} end

abstract type DimensionalValue{T, N} <: ValueContainer{T} end
abstract type HartreeFockintermediateData{T} <: ValueContainer{T} end
abstract type ConfigBox{T, ContainerT, MethodT} <: ValueContainer{T} end
abstract type ComputableGraph{T} <: ValueContainer{T} end
abstract type AbstractMarker{T} <: ValueContainer{T} end
abstract type ValueStorage{T} <: ValueContainer{T} end

abstract type IdentityMarker{T} <: AbstractMarker{T} end
abstract type StorageMarker{T} <: AbstractMarker{T} end

abstract type HartreeFockFinalValue{T, HFT} <: MixedContainer{T} end
abstract type BasisSetData{T, D, BS} <: MixedContainer{T} end
abstract type MatterData{T, D} <: MixedContainer{T} end

abstract type DoubleDimParam{T, N, O} <: ParamContainer{T} end

abstract type CompositeParam{T, N, O} <: DoubleDimParam{T, N, O} end
abstract type PrimitiveParam{T, N} <: DoubleDimParam{T, N, 0} end

abstract type GraphNode{T, N, O} <: ComputableGraph{T} end

abstract type OperationNode{T, N, O, I} <: GraphNode{T, N, O} end
abstract type ReferenceNode{T, N, O, I} <: GraphNode{T, N, O} end
abstract type ContainerNode{T, N, O}    <: GraphNode{T, N, O} end

abstract type ParamBatch{T, N, I, O} <: CompositeParam{T, N, O} end
abstract type ParamToken{T, N, I} <: CompositeParam{T, N, 0} end

abstract type ParamLink{T, N, I, O} <: ParamBatch{T, N, I, O} end
abstract type ParamNest{T, N, I, O} <: ParamBatch{T, N, I, O} end
abstract type ViewParam{T, N, I} <: ParamToken{T, N, I} end

abstract type SingleVarFunction{T} <: ParamFunction{T} end
abstract type MultiDimFunction{T, D} <: ParamFunction{T} end

abstract type RadialFunc{T} <: SingleVarFunction{T} end
abstract type AbstractBasis{T, D, ON} <: MultiDimFunction{T, D} end
abstract type SpatialStructure{T, D} <: MultiDimFunction{T, D} end
abstract type AbstractSpatialPoint{T, D} <: MultiDimFunction{T, D} end

abstract type SpatialBasis{T, D, ON} <: AbstractBasis{T, D, ON} end
abstract type FermionState{T, D, MaxOccupation} <: AbstractBasis{T, D, MaxOccupation} end

abstract type AbstractSpinOrbital{T, D} <: FermionState{T, D, 2} end

abstract type AbstractGaussFunc{T} <: RadialFunc{T} end
abstract type AbstractBasisFuncs{T, D, OrbitalN} <: SpatialBasis{T, D, OrbitalN} end

abstract type CompositeBasisFuncs{NumberT, D, FBasisFuncN, OrbitalN} <: AbstractBasisFuncs{NumberT, D, OrbitalN} end

abstract type FloatingBasisFuncs{NumberT, D, 𝑙, PointT, RadialV, OrbitalN} <: CompositeBasisFuncs{NumberT, D, 1, OrbitalN} end


const Dim0GNode{T} = GraphNode{T, 0, 0}
const DimIGNode{T, N} = GraphNode{T, N, 0}
const DimOGNode{T, N} = GraphNode{T, 0, N}
const DimSGNode{T, N} = Union{DimIGNode{T, N}, DimOGNode{T, N}}

const ParamObject{T} = Union{ParamContainer{T}, ParamFunction{T}}
const ElementalParam{T} = DoubleDimParam{T, 0, 0}
const InnerSpanParam{T, N} = DoubleDimParam{T, N, 0}
const OuterSpanParam{T, N} = DoubleDimParam{T, 0, N}
const SingleDimParam{T, N} = Union{InnerSpanParam{T, N}, OuterSpanParam{T, N}}
const PrimParCandidate{T} = Union{ElementalParam{T}, PrimitiveParam{T}}
# const PNodeIn1{T} = Union{T, ElementalParam{T}}
# const PNodeIn2{T, F<:Function} = Tuple{F, ElementalParam{T}}
# const PNodeIn3{T, F<:Function} = Tuple{F, AbstractArray{<:ElementalParam{T}}}
# const PNodeIn4{T, F} = Union{PNodeIn2{T, F}, PNodeIn3{T, F}}
const PNodeInput{T} = Union{T, ElementalParam{T}}

const SParamAbtArray{T, N} = AbstractArray{<:ElementalParam{T}, N}
const PParamAbtArray{T, N} = AbstractArray{<:PrimitiveParam{T}, N}
const DParamAbtArray{T, N} = AbstractArray{<:DoubleDimParam{T, <:Any, 0}, N}

const SpatialBasis1O{T, D} = SpatialBasis{T, D, 1}
const CBasisFuncsON{ON} = CompositeBasisFuncs{<:Any, <:Any, <:Any, ON}
const CBasisFuncs1O{T, D, BN} = CompositeBasisFuncs{T, D, BN, 1}

const FBasisFuncsON{ON} = FloatingBasisFuncs{<:Any, <:Any, <:Any, <:Any, <:Any, ON}
const FBasisFuncs1O{T, D, 𝑙, PT, RV} = FloatingBasisFuncs{T, D, 𝑙, PT, RV, 1}


##! To be disposed
const SpatialCoordType{T, D, NNMO} = Union{ 
    AbstractVector{<:AbstractVector{<:T}}, 
    AbstractVector{NTuple{D, T}}, 
    Tuple{TL, Vararg{TL, NNMO}} where TL<:AbstractVector{<:T}, 
    Tuple{NTuple{D, T}, Vararg{NTuple{D, T}, NNMO}} }
const AVectorOrNTuple{T, NNMO} = Union{Tuple{T, Vararg{T, NNMO}}, AbstractVector{<:T}}
const AArrayOrNTuple{T, NNMO} = Union{Tuple{T, Vararg{T, NNMO}}, AbstractArray{<:T}}
const NTupleOfFBF{BN, T, D} = NTuple{BN,  FBasisFuncs1O{T, D}}
const NTupleOfSB1{BN, T, D} = NTuple{BN, SpatialBasis1O{T, D}}
const NTupleOfSBN{BN, T, D} = NTuple{BN,   SpatialBasis{T, D}}
##!


const StrOrSym = Union{String, Symbol}

const MatrixCol{T} = Vector{SubArray{T, 1, Matrix{T}, Tuple{Slice{OneTo{Int}}, Int}, true}}

const RFuncAbtVector{T} = AbstractVector{<:RadialFunc{T}}

const NonEmptyTuple{T, NMO} = Tuple{T, Vararg{T, NMO}}

const TupleOrAbtVec = Union{Tuple, AbstractVector}

const RefVal = Base.RefValue

const ParamFunctions = Union{ ParamFunction, 
                              AbstractArray{<:ParamFunction}, 
                              Tuple{Vararg{ParamFunction}} }

const ABFuncAbtVector{T, D} = AbstractVector{<:AbstractBasisFuncs{T, D, 1}}


const IntOrNone = Union{Int, Nothing}
const IntOrMiss = Union{Int, Missing}
const SymOrMiss = Union{Symbol, Missing}
const Array0D{T} = Array{T, 0}
const AbtArray0D{T} = AbstractArray{T, 0}

const NonEmptyTupleOrAbtArray{T, A<:AbstractArray{T}} = Union{NonEmptyTuple{T}, A}
const NonEmptyTupleOrAbtVector{T, A<:AbstractVector{T}} = Union{NonEmptyTuple{T}, A}

@enum TernaryNumber::Int8 begin
    TUS0= 0
    TPS1= 1
    TPS2= 2
end

const AbtArrayOr{T} = Union{T, AbstractArray{T}}
const BiAbtArray{T, NI, NO} = AbstractArray{<:AbstractArray{T, NI}, NO}
const AbtArr210L{T} = Union{T, AbstractArray{T}, BiAbtArray{T}}
const AbtArr21L{T} = Union{AbstractArray{T}, BiAbtArray{T}}

const RefOrA0D{T, V<:AbstractArray{T, 0}} =  Union{RefVal{T}, V}

# const ElemParamAbtArray{T, N} = AbstractArray{<:DoubleDimParam{T, 0}, N}
# const SingleDimParArg{T, N} = Union{ElemParamAbtArray{T, N}, DoubleDimParam{T, N}}
# const MultiNDParTuple{T, N, M} = NTuple{M, SingleDimParArg{T, N}}
# const MonoDimParTuple{T, N} = MultiNDParTuple{T, N, 1}
# const DualDimParTuple{T, N} = MultiNDParTuple{T, N, 2}
# const ParamInputType{T, N} = Union{MonoDimParTuple{T, N}, DualDimParTuple{T, N}}

# const AbtArrTypeTuple{T, N, M} = NTuple{M, Type{<:AbstractArray{T, N}}}
# const ArgTypeOfNDPVal{T, N} = Union{AbtArrTypeTuple{T, N, 1}, AbtArrTypeTuple{T, N, 2}}

# const TypeNTuple{T, N} = NTuple{N, Type{T}}

const TernaryNTupleUnion{T} = Union{(NTuple{N, T} for N in 1:3)...}
const ElemParamAbtArray{T, N} = AbstractArray{<:ElementalParam{T}, N}
const ParamInputType{T} = TernaryNTupleUnion{DoubleDimParam{T}}
const ParamInput{T, N} = NTuple{N, DoubleDimParam{T}}
const ParamSetEle{T} = Union{AbstractVector{<:ElementalParam{T}}, InnerSpanParam{T}}
const DimParamSet{T} = AbstractVector{<:ParamSetEle{T}}
const ArgTypeOfNDPIVal{T} = Union{
    Tuple{Type{T}}, 
    Tuple{Type{T}, Type{T}}, 
    Tuple{Type{<:AbstractArray{T}}, Type{T}}, 
    Tuple{Type{T}, Type{<:AbstractArray{T}}}, 
    Tuple{Type{<:AbstractArray{T}}, Type{<:AbstractArray{T}}}
}


# const NETupleOfAOOType{T, NMO} = NonEmptyTuple{AbtArrOOType{T}, NMO}
# const ArgTypeOfNDPIVal{T} = Union{NETupleOfAOOType{T, 0}, NETupleOfAOOType{T, 1}}

const NETupleOfPBoxVal{T, NMO} = NonEmptyTuple{AbtArrayOr{T}, NMO}
const PBoxInputValType{T} = Union{NETupleOfPBoxVal{T, 0}, NETupleOfPBoxVal{T, 1}}

const AbtVecOfAbtArr{T} = AbstractVector{<:AbstractArray{T}}

const GraphArgDataType{T} = Union{AbtVecOfAbtArr{T}, NonEmptyTuple{AbstractArray{T}}}

const ParBTypeArgNumOutDim{T, N, A} = ParamToken{T, N, <:NTuple{A, DoubleDimParam{T}}}

const AbtMemory0D{T} = AbstractMemory{T, 0}

const ParamFunctor{T, N, I} = Union{ParamLink{T, N, I}, ParamToken{T, N, I}}
const ParamPointer{T, N, I} = Union{ViewParam{T, N, I}, ParamNest{ T, N, I}}