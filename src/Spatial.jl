export FieldFunc, GaussFunc, AxialProdFunc, PolyRadialFunc

using LinearAlgebra

(f::FieldAmplitude)(x) = evalFunc(f, x)

abstract type EvalDimensionalKernel{T, D, F} <: TypedEvaluator{T, F} end

abstract type EvalFieldAmp{T, D, F} <: EvalDimensionalKernel{T, D, F} end

(f::EvalFieldAmp)(input, param) = 
f.f(formatInput(SelectTrait{InputStyle}()(f), input), param)


struct FieldFunc{T<:Number, F<:Function} <: FieldAmplitude{T, 0}
    f::ReturnTyped{T, F}

    function FieldFunc(f::F, ::Type{T}) where {F, T}
        new{T, F}( ReturnTyped(f, T) )
    end
end

struct EvalFieldFunc{T, F<:ReturnTyped{T}} <: EvalFieldAmp{T, 0, FieldFunc}
    f::F
end

function unpackParamFunc!(f::FieldFunc{T}, paramSet::FlatParamSet) where {T}
    fEval, _, fieldParamPointer = unpackFunc!(f.f, paramSet)
    EvalFieldFunc(ReturnTyped(fEval, T)), paramSet, fieldParamPointer
end


struct GaussFunc{T<:Real, P<:ElementalParam{T}} <: FieldAmplitude{T, 0}
    xpn::P
end

GaussFunc(xpn::Real) = GaussFunc(CellParam(xpn, :xpn))

struct ComputeGFunc{T} <: FieldlessFunction end

function (f::ComputeGFunc{T})(r::Real, xpnVal::T) where {T}
    exp(-xpnVal * r * r)
end

const GetEleParamInSNPSet{T} = ChainIndexer{T, 0, Tuple{FirstIndex, Int}}

struct EvalGaussFunc{T} <: EvalFieldAmp{T, 0, GaussFunc}
    f::SinglePtrFunc{ComputeGFunc{T}, OneLayerAllPass{T}, GetEleParamInSNPSet{T}}
end

function unpackParamFunc!(f::GaussFunc{T, P}, paramSet::FlatParamSet) where {T, P}
    pSetId = Identifier(paramSet)
    anchor = ChainPointer(:xpn, TensorType(T))
    parIdx = locateParam!(paramSet, getField(f, anchor))
    fEval = PointerFunc(ComputeGFunc{T}(), OneLayerAllPass{T}(), (parIdx,), pSetId)
    EvalGaussFunc(fEval), paramSet, MixedFieldParamPointer(anchor=>parIdx, pSetId)
end


struct AxialProdFunc{T, D, B<:NTuple{D, FieldAmplitude{T, 0}}} <: FieldAmplitude{T, D}
    axis::B

    AxialProdFunc(axis::B) where {T, B<:NonEmptyTuple{FieldAmplitude{T, 0}}} = 
    new{T, length(axis), B}(axis)
end

AxialProdFunc(compos::AbstractVector{<:FieldAmplitude{T, 0}}) where {T} = 
AxialProdFunc( Tuple(compos) )

function AxialProdFunc(b::FieldAmplitude{<:Any, 0}, dim::Int)
    dim < 1 && throw(AssertionError("`dim` must be a positive integer."))
    (AxialProdFunc∘Tuple∘fill)(b, dim)
end

const EvalAxialField{T, F<:EvalFieldAmp{T, 0}} = InsertInward{F, OnlyHead{GetIndex{T, 0}}}

struct EvalAxialProdFunc{T, D, F<:EvalFieldAmp{T, 0}} <: EvalFieldAmp{T, D, AxialProdFunc}
    f::ChainReduce{StableMul{T}, VectorMemory{EvalAxialField{T, F}, D}}
end

function unpackParamFunc!(f::AxialProdFunc{T, D}, paramSet::FlatParamSet) where {T, D}
    fEvalComps = Memory{Function}(undef, D)
    fieldParamDict = mapfoldl(vcat, 1:D) do i
        anchor = ChainPointer(:axis, ChainPointer(i))
        fInner, _, pointerInner = unpackFunc!(f.axis[i], paramSet)
        dictInner = pointerInner.core
        ptr = ChainPointer(i, TensorType(T))
        fEvalComps[i] = InsertInward(fInner, (OnlyHead∘getField)(ptr))
        anchorFieldPointerDictCore(dictInner, anchor)
    end |> buildDict
    fieldParamPointer = MixedFieldParamPointer(fieldParamDict, Identifier(paramSet))
    fEval = Tuple(fEvalComps) |> (ChainReduce∘StableBinary)(*, T)
    EvalAxialProdFunc(fEval), paramSet, fieldParamPointer
end


struct PolyRadialFunc{T, D, F<:FieldAmplitude{T, 0}, L} <: FieldAmplitude{T, D}
    radial::F
    angular::CartSHarmonics{D, L}
end

PolyRadialFunc(radial::FieldAmplitude{T, 0}, angular::NonEmptyTuple{Int}) where {T} = 
PolyRadialFunc(radial, CartSHarmonics(angular))

const MagnitudeConverter{F} = InsertInward{F, OnlyHead{typeof(LinearAlgebra.norm)}}

const TypedAngularFunc{T, D, L} = OnlyHead{ReturnTyped{T, CartSHarmonics{D, L}}}

struct EvalPolyRadialFunc{T, D, F<:EvalFieldAmp{T, 0}, 
                          L} <: EvalFieldAmp{T, D, PolyRadialFunc}
    f::PairCombine{StableMul{T}, MagnitudeConverter{F}, TypedAngularFunc{T, D, L}}
end

function unpackParamFunc!(f::PolyRadialFunc{T, D}, paramSet::FlatParamSet) where 
                         {T, D}
    fInner, _, fieldParamPointerInner = unpackParamFunc!(f.radial, paramSet)
    dictInner = fieldParamPointerInner.core
    fieldParamDict = anchorFieldPointerDict(dictInner, ChainPointer(:radial))
    coordEncoder = InsertInward(fInner, OnlyHead(LinearAlgebra.norm))
    angularFunc = (OnlyHead∘ReturnTyped)(f.angular, T)
    fEval = PairCombine(StableBinary(*, T), coordEncoder, angularFunc)
    fieldParamPointer = MixedFieldParamPointer(fieldParamDict, Identifier(paramSet))
    EvalPolyRadialFunc(fEval), paramSet, fieldParamPointer
end

const PolyGaussProd{T, D, L} = PolyRadialFunc{T, D, <:GaussFunc{T}, L}
const EvalPolyGaussProd{T, D, L} = EvalPolyRadialFunc{T, D, EvalGaussFunc{T}, L}

getAngTuple(f::PolyRadialFunc) = f.angular.m.tuple