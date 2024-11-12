export FieldFunc, GaussFunc, AxialProdFunc, PolyRadialFunc

using LinearAlgebra: norm

(f::FieldAmplitude)(x) = evalFunc(f, x)

abstract type EvalFieldAmp{T, D, F} <: Evaluator{F} end

evalFunc(f::EvalFieldAmp, input, param) = 
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

function unpackParamFunc!(f::FieldFunc{T}, paramSet::PBoxAbtArray) where {T}
    fEval, _ = unpackFunc!(f.f, paramSet)
    EvalFieldFunc(ReturnTyped(fEval, T)), paramSet
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

function unpackParamFunc!(f::GaussFunc{T, <:ElementalParam{T}}, 
                          paramSet::PBoxAbtArray) where {T}
    parIds = (locateParam!(paramSet, f.xpn),)
    fEval = PointerFunc(ComputeGFunc{T}(), parIds, objectid(paramSet))
    EvalGaussFunc(fEval), paramSet
end


struct AxialProdFunc{T, D, B<:NTuple{D, FieldAmplitude{T, 0}}} <: FieldAmplitude{T, D}
    component::B

    AxialProdFunc(component::B) where {T, B<:NonEmptyTuple{FieldAmplitude{T, 0}}} = 
    new{T, length(component), B}(component)
end

AxialProdFunc(compos::AbstractVector{<:FieldAmplitude{T, 0}}) where {T} = 
AxialProdFunc( Tuple(compos) )

function AxialProdFunc(b::FieldAmplitude{<:Any, 0}, dim::Int)
    dim < 1 && throw(AssertionError("`dim` must be a positive integer."))
    (AxialProdFunc∘Tuple∘fill)(b, dim)
end

# const AxialFieldProd{T, D, A<:EvalFieldAmp{T, 0}} = 
#       NTuple{D, InsertInward{<:A, OnlyInput{Base.Fix2{typeof(getindex), Int}}}}

const EvalAxialField{T, F<:EvalFieldAmp{T, 0}} = 
InsertInward{F, OnlyInput{Base.Fix2{typeof(getindex), Int}}}

struct EvalAxialProdFunc{T, D, F<:EvalFieldAmp{T, 0}} <: EvalFieldAmp{T, D, AxialProdFunc}
    f::MulChain{VectorMemory{EvalAxialField{T, F}, D}}
end

function unpackParamFunc!(f::AxialProdFunc{T, D}, paramSet::PBoxAbtArray) where {T, D}
    fEvalComps = map(Fix2(unpackFunc!, paramSet), f.component) .|> first
    fEval = map(fEvalComps, Tuple(1:D)) do efc, idx
        InsertInward(efc, (OnlyInput∘Base.Fix2)(getindex, idx))
    end |> ChainReduce(*)
    EvalAxialProdFunc(fEval), paramSet
end


struct PolyRadialFunc{T, D, F<:FieldAmplitude{T, 0}, L} <: FieldAmplitude{T, D}
    radial::F
    angular::CartSHarmonics{D, L}
end

PolyRadialFunc(radial::FieldAmplitude{T, 0}, angular::NonEmptyTuple{Int}) where {T} = 
PolyRadialFunc(radial, CartSHarmonics(angular))

const MagnitudeConverter{F} = InsertInward{F, OnlyInput{typeof(norm)}}

struct EvalPolyRadialFunc{T, D, F<:EvalFieldAmp{T, 0}, 
                          L} <: EvalFieldAmp{T, D, PolyRadialFunc}
    f::MulPair{MagnitudeConverter{F}, OnlyInput{CartSHarmonics{D, L}}}
end

function unpackParamFunc!(f::PolyRadialFunc{<:Any, D}, paramSet::PBoxAbtArray) where {D}
    fInner, _ = unpackParamFunc!(f.radial, paramSet)
    fEval = PairCombine(*, InsertInward(fInner, OnlyInput(norm)), OnlyInput(f.angular))
    EvalPolyRadialFunc(fEval), paramSet
end

const PolyGaussFunc{T, D, L} = PolyRadialFunc{T, D, <:GaussFunc{T}, L}
const EvalPolyGaussFunc{T, D, L} = EvalPolyRadialFunc{T, D, EvalGaussFunc{T}, L}