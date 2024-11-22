export FieldFunc, GaussFunc, AxialProdFunc, PolyRadialFunc

using LinearAlgebra

(f::FieldAmplitude)(x) = evalFunc(f, x)

abstract type EvalFieldAmp{T, D, F} <: Evaluator{F} end

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

function unpackParamFunc!(f::FieldFunc{T}, paramSet::AbstractFlatParamSet) where {T}
    fEval, _, dict = unpackFunc!(f.f, paramSet)
    EvalFieldFunc(ReturnTyped(fEval, T)), paramSet, dict
end


struct GaussFunc{T<:Real, P<:ElementalParam{T}} <: FieldAmplitude{T, 0}
    xpn::P
end

GaussFunc(xpn::Real) = GaussFunc(CellParam(xpn, :xpn))

struct ComputeGFunc{T} <: FieldlessFunction end

function (f::ComputeGFunc{T})(r::Real, xpnVal::T) where {T}
    exp(-xpnVal * r * r)
end

struct EvalGaussFunc{T} <: EvalFieldAmp{T, 0, GaussFunc}
    f::PointOneFunc{ComputeGFunc{T}}
end

function unpackParamFunc!(f::GaussFunc{T}, paramSet::AbstractFlatParamSet) where {T}
    anchor = (FieldLinker∘FieldSymbol)(:xpn, outputTypeOf(f.xpn))
    parIdx = locateParam!(paramSet, getField(f, anchor))
    fEval = PointerFunc(ComputeGFunc{T}(), (parIdx,), objectid(paramSet))
    EvalGaussFunc(fEval), paramSet, Dict(anchor=>parIdx)
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

const EvalAxialField{T, F<:EvalFieldAmp{T, 0}} = 
InsertInward{F, OnlyHead{IndexPointer{DataXD}}}

struct EvalAxialProdFunc{T, D, F<:EvalFieldAmp{T, 0}} <: EvalFieldAmp{T, D, AxialProdFunc}
    f::ChainReduce{StableMul{T}, VectorMemory{EvalAxialField{T, F}, D}}
end


function unpackParamFunc!(f::AxialProdFunc{T, D}, paramSet::AbstractFlatParamSet) where 
                         {T, D}
    fEvalComps = Memory{Function}(undef, D)
    paramFieldDict = mapfoldl(vcat, 1:D) do i
        anchor = FieldLinker(:axis, FieldSymbol(i))
        fInner, _, dictInner = unpackFunc!(f.axis[i], paramSet)
        fEvalComps[i] = InsertInward(fInner, (OnlyHead∘IndexPointer)(i))
        anchorFieldPointerDictCore(dictInner, anchor)
    end |> Dict
    fEval = Tuple(fEvalComps) |> (ChainReduce∘StableBinary)(*, T)
    EvalAxialProdFunc(fEval), paramSet, paramFieldDict
end


struct PolyRadialFunc{T, D, F<:FieldAmplitude{T, 0}, L} <: FieldAmplitude{T, D}
    radial::F
    angular::CartSHarmonics{D, L}
end

PolyRadialFunc(radial::FieldAmplitude{T, 0}, angular::NonEmptyTuple{Int}) where {T} = 
PolyRadialFunc(radial, CartSHarmonics(angular))

const MagnitudeConverter{F} = InsertInward{F, OnlyHead{typeof(LinearAlgebra.norm)}}

struct EvalPolyRadialFunc{T, D, F<:EvalFieldAmp{T, 0}, 
                          L} <: EvalFieldAmp{T, D, PolyRadialFunc}
    f::PairCombine{StableMul{T}, MagnitudeConverter{F}, OnlyHead{CartSHarmonics{D, L}}}
end

function unpackParamFunc!(f::PolyRadialFunc{T, D}, paramSet::AbstractFlatParamSet) where 
                         {T, D}
    fInner, _, paramFieldDictInner = unpackParamFunc!(f.radial, paramSet)
    paramFieldDict = anchorFieldPointerDict(paramFieldDictInner, FieldSymbol(:radial))
    coordEncoder = InsertInward(fInner, OnlyHead(LinearAlgebra.norm))
    fEval = PairCombine(StableBinary(*, T), coordEncoder, OnlyHead(f.angular))
    EvalPolyRadialFunc(fEval), paramSet, paramFieldDict
end

const PolyGaussProd{T, D, L} = PolyRadialFunc{T, D, <:GaussFunc{T}, L}
const EvalPolyGaussProd{T, D, L} = EvalPolyRadialFunc{T, D, EvalGaussFunc{T}, L}

getAngTuple(f::PolyRadialFunc) = f.angular.m.tuple