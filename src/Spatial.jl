export FieldFunc, GaussFunc, AxialProdFunc, PolyRadialFunc

using LinearAlgebra

(f::FieldAmplitude)(x) = evalFunc(f, x)

abstract type EvalDimensionalKernel{T, D, F} <: TypedEvaluator{T, F} end

abstract type EvalFieldAmp{T, D, F} <: EvalDimensionalKernel{T, D, F} end

(f::EvalFieldAmp)(input, param) = 
f.f(formatInput(SelectTrait{InputStyle}()(f), input), param)


function unpackParamFunc!(f::FieldAmplitude{T}, paramSet::FlatParamSet) where {T}
    fCore, _, paramPairs = unpackParamFuncCore!(f, paramSet)
    paramPtr = MixedFieldParamPointer(paramPairs, paramSet)
    fCore, paramSet, paramPtr
end


struct FieldFunc{T<:Number, F<:Function} <: FieldAmplitude{T, 0}
    f::ReturnTyped{T, F}

    function FieldFunc(f::F, ::Type{T}) where {F, T}
        new{T, F}( ReturnTyped(f, T) )
    end
end

struct EvalFieldFunc{T, F<:ParamSelectFunc{<:ReturnTyped{T}}
                     } <: EvalFieldAmp{T, 0, FieldFunc}
    f::F
end

function unpackParamFuncCore!(f::FieldFunc{T}, paramSet::FlatParamSet) where {T}
    fEvalCore, _, pairs = unpackTypedFuncCore!(f, paramSet)
    EvalFieldFunc(fEvalCore), paramSet, pairs
end


struct GaussFunc{T<:Real, P<:ElementalParam{T}} <: FieldAmplitude{T, 0}
    xpn::P
end

GaussFunc(xpn::T) where {T<:Real} = (GaussFunc∘genCellEncoder(T, :xpn))(xpn)


struct ComputeGFunc{T} <: FieldlessFunction end

function (f::ComputeGFunc{T})(r::Real, xpnVal::T) where {T}
    exp(-xpnVal * r * r)
end

struct EvalGaussFunc{T} <: EvalFieldAmp{T, 0, GaussFunc}
    f::GetParamFunc{ComputeGFunc{T}, FlatPSetInnerPtr{T}}
end

function unpackParamFuncCore!(f::GaussFunc{T, P}, paramSet::FlatParamSet) where {T, P}
    anchor = ChainPointer(:xpn, TensorType(T))
    parIdx = locateParam!(paramSet, getField(f, anchor))
    fEvalCore = ParamSelectFunc(ComputeGFunc{T}(), (parIdx,))
    EvalGaussFunc(fEvalCore), paramSet, [anchor=>parIdx]
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

struct EvalAxialProdFunc{T, D, F<:EvalFieldAmp{T, 0}} <: EvalFieldAmp{T, D, AxialProdFunc}
    f::CountedChainReduce{StableMul{T}, InsertInward{F, OnlyHead{GetIdxField{T}}}, D}
end

function unpackParamFuncCore!(f::AxialProdFunc{T, D}, paramSet::FlatParamSet) where {T, D}
    fEvalComps = Memory{Function}(undef, D)
    pairs = mapfoldl(vcat, 1:D) do i
        anchor = ChainPointer((:axis, i))
        fInner, _, axialPairs = unpackParamFuncCore!(f.axis[i], paramSet)
        ptr = ChainPointer(i, TensorType(T))
        fEvalComps[i] = InsertInward(fInner, (OnlyHead∘getField)(ptr))
        anchorLeft(axialPairs, anchor)
    end
    fEvalCore = Tuple(fEvalComps) |> (ChainReduce∘StableBinary)(*, T)
    EvalAxialProdFunc(fEvalCore), paramSet, pairs
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

function unpackParamFuncCore!(f::PolyRadialFunc{T, D}, paramSet::FlatParamSet) where {T, D}
    fInner, _, radialPairs = unpackParamFuncCore!(f.radial, paramSet)
    pairs = anchorLeft(radialPairs, ChainPointer(:radial))
    coordEncoder = InsertInward(fInner, OnlyHead(LinearAlgebra.norm))
    angularFunc = (OnlyHead∘ReturnTyped)(f.angular, T)
    fEvalCore = PairCombine(StableBinary(*, T), coordEncoder, angularFunc)
    EvalPolyRadialFunc(fEvalCore), paramSet, pairs
end

const PolyGaussProd{T, D, L} = PolyRadialFunc{T, D, <:GaussFunc{T}, L}
const EvalPolyGaussProd{T, D, L} = EvalPolyRadialFunc{T, D, EvalGaussFunc{T}, L}

getAngTuple(f::PolyRadialFunc) = f.angular.m.tuple