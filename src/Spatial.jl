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

struct EvalFieldFunc{T, F<:FlatPSetFilterFunc{T, <:ParamSelectFunc{<:ReturnTyped{T}}}
                     } <: EvalFieldAmp{T, 0, FieldFunc}
    f::F
end

function unpackParamFuncCore!(f::FieldFunc{T}, paramSet::FlatParamSet) where {T}
    pSetLocal = initializeParamSet(FlatParamSet, T)
    fEvalCore, _, innerPairs = unpackTypedFuncCore!(f, pSetLocal)
    pFilter = locateParam!(paramSet, pSetLocal)
    pairs = anchorRight(innerPairs, pFilter)
    EvalFieldFunc(ParamFilterFunc(fEvalCore, pFilter)), paramSet, pairs
end


struct GaussFunc{T<:Real, P<:ElementalParam{T}} <: FieldAmplitude{T, 0}
    xpn::P
end

GaussFunc(xpn::Real) = GaussFunc(CellParam(xpn, :xpn))

struct ComputeGFunc{T} <: FieldlessFunction end

function (f::ComputeGFunc{T})(r::Real, xpnVal::T) where {T}
    exp(-xpnVal * r * r)
end

const EvalGaussFuncCore{T} = GetParamFunc{ComputeGFunc{T}, AllPassPtr{Flavor{T}}}

struct EvalGaussFunc{T} <: EvalFieldAmp{T, 0, GaussFunc}
    f::ParamFilterFunc{EvalGaussFuncCore{T}, FlatPSetInnerPtr{T}}
end

function unpackParamFuncCore!(f::GaussFunc{T, P}, paramSet::FlatParamSet) where {T, P}
    anchor = ChainPointer(:xpn, TensorType(T))
    parIdxInner = ChainPointer((), TensorType(T))
    parIdxOuter = locateParam!(paramSet, getField(f, anchor))
    fEvalCore = ParamSelectFunc(ComputeGFunc{T}(), (parIdxInner,))
    fEval = ParamFilterFunc(fEvalCore, parIdxOuter)
    EvalGaussFunc(fEval), paramSet, [anchor=>parIdxOuter]
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

const EvalAxialField{T, F<:EvalFieldAmp{T, 0}} = InsertInward{F, OnlyHead{GetScalarIdx{T}}}

struct EvalAxialProdFunc{T, D, F<:EvalFieldAmp{T, 0}} <: EvalFieldAmp{T, D, AxialProdFunc}
    f::FlatPSetFilterFunc{T, CountedChainReduce{StableMul{T}, EvalAxialField{T, F}, D}}
end

function unpackParamFuncCore!(f::AxialProdFunc{T, D}, paramSet::FlatParamSet) where {T, D}
    fEvalComps = Memory{Function}(undef, D)
    pSetLocal = initializeParamSet(FlatParamSet, T)
    innerPairs = mapfoldl(vcat, 1:D) do i
        anchor = ChainPointer((:axis, i))
        fInner, _, axialPairs = unpackParamFuncCore!(f.axis[i], pSetLocal)
        ptr = ChainPointer(i, TensorType(T))
        fEvalComps[i] = InsertInward(fInner, (OnlyHead∘getField)(ptr))
        anchorLeft(axialPairs, anchor)
    end
    pFilter = locateParam!(paramSet, pSetLocal)
    pairs = anchorRight(innerPairs, pFilter)
    fEvalCore = Tuple(fEvalComps) |> (ChainReduce∘StableBinary)(*, T)
    EvalAxialProdFunc(ParamFilterFunc(fEvalCore, pFilter)), paramSet, pairs
end


struct PolyRadialFunc{T, D, F<:FieldAmplitude{T, 0}, L} <: FieldAmplitude{T, D}
    radial::F
    angular::CartSHarmonics{D, L}
end

PolyRadialFunc(radial::FieldAmplitude{T, 0}, angular::NonEmptyTuple{Int}) where {T} = 
PolyRadialFunc(radial, CartSHarmonics(angular))

const MagnitudeConverter{F} = InsertInward{F, OnlyHead{typeof(LinearAlgebra.norm)}}

const TypedAngularFunc{T, D, L} = OnlyHead{ReturnTyped{T, CartSHarmonics{D, L}}}

const EvalPolyRadialFuncCore{T, D, F, L} = 
      PairCombine{StableMul{T}, MagnitudeConverter{F}, TypedAngularFunc{T, D, L}}

struct EvalPolyRadialFunc{T, D, F<:EvalFieldAmp{T, 0}, 
                          L} <: EvalFieldAmp{T, D, PolyRadialFunc}
    f::FlatPSetFilterFunc{T, EvalPolyRadialFuncCore{T, D, F, L}}
end

function unpackParamFuncCore!(f::PolyRadialFunc{T, D}, paramSet::FlatParamSet) where {T, D}
    pSetLocal = initializeParamSet(FlatParamSet, T)
    fInner, _, radialPairs = unpackParamFuncCore!(f.radial, pSetLocal)
    pFilter = locateParam!(paramSet, pSetLocal)
    pairs = anchorRight(anchorLeft(radialPairs, ChainPointer(:radial)), pFilter)
    coordEncoder = InsertInward(fInner, OnlyHead(LinearAlgebra.norm))
    angularFunc = (OnlyHead∘ReturnTyped)(f.angular, T)
    fEvalCore = PairCombine(StableBinary(*, T), coordEncoder, angularFunc)
    EvalPolyRadialFunc(ParamFilterFunc(fEvalCore, pFilter)), paramSet, pairs
end

const PolyGaussProd{T, D, L} = PolyRadialFunc{T, D, <:GaussFunc{T}, L}
const EvalPolyGaussProd{T, D, L} = EvalPolyRadialFunc{T, D, EvalGaussFunc{T}, L}

getAngTuple(f::PolyRadialFunc) = f.angular.m.tuple