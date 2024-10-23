(f::SpatialAmplitude{<:Any, <:Any, 1})(x) = 
formatInputCall(SelectTrait{InputStyle}()(f), f, x)

evalFunc(f::Evaluator{<:SpatialAmplitude{<:Any, <:Any, 1}}, input, param) = 
f.f(input, param)


# API: Evaluator, unpackParamFunc!
struct GaussFunc{T<:Real, P1<:ElementalParam{T}, 
                 P2<:ElementalParam{<:RealOrComplex{T}}} <: SpatialAmplitude{T, 0, 1}
    xpn::P1
    con::P2

    GaussFunc(xpn::P1, con::P2) where {T, P1<:ElementalParam{T}, P2<:ElementalParam} = 
    new{T, P1, P2}(xpn, con)
end

struct ComputeGaussFunc{T1, T2} <: FieldlessFunction end

struct EvalGaussFunc{F<:PointerFunc{<:ComputeGaussFunc}} <: Evaluator{GaussFunc}
    f::F
end

function (::ComputeGaussFunc{T1, T2})(r::Real, xpnVal::T1, conVal::T2) where {T1, T2}
    exp(-xpnVal * r * r) * conVal
end

function unpackParamFunc!(f::GaussFunc{T1, <:ElementalParam{T1}, <:ElementalParam{T2}}, 
                          paramSet::PBoxCollection) where {T1, T2}
    evalFuncCore = ComputeGaussFunc{T1, T2}()
    parIds = locateParam!(paramSet, getParamsCore(f))
    evalFunc = PointerFunc(evalFuncCore, parIds, objectid(paramSet))
    EvalGaussFunc(evalFunc), paramSet
end

(f::GaussFunc)(r::Real) = evalFunc(f, r)


struct AxialProd{T<:Real, D, F<:SpatialAmplitude{T, 0, 1}, 
                 L<:NTuple{D, Memory{<:ElementalParam{T}}}} <: SpatialAmplitude{T, D, 1}
    series::Memory{F}
    layout::L

    function AxialProd(series::AbstractVector{F}, layout::L) where 
                      {T, F<:SpatialAmplitude{T, 0, 1}, D, 
                       L<:NonEmptyTuple{AbstractVector{<:ElementalParam{T}}, D}}
        nFunc = checkEmptiness(series, :series)
        if any(l != nFunc for l in length.(layout))
            throw(AssertionError("`series` and each `layout` element must have equal "*
                                 "length."))
        end
        formattedLayout = getMemory.(layout)
        new{T, D+1, F, typeof(formattedLayout)}(getMemory(series), formattedLayout)
    end
end

function AxialProd(series::AbstractVector{F}, dim::Int=3) where {T, F<:SpatialAmplitude{T, 0, 1}}
    dim < 1 && throw(AssertionError("`dim` should be a positive integer."))
    cell = CellParam(zero(T), :r)
    setScreenLevel!(cell, 2)
    layout = fill(Memory{ItselfCParam{T}}( fill(cell, length(series)) ), dim) |> Tuple
    AxialProd(getMemory(series), layout)
end

struct EvalAxialProd{F<:JoinParallel} <: Evaluator{AxialProd}
    f::F
end

function unpackParamFunc!(f::AxialProd, paramSet::PBoxCollection)
    pSetId = objectid(paramSet)
    innerEvalFuncs = map(Fix2(unpackFunc!, paramSet), f.series) .|> first
    evalFunc = mapreduce(JoinParallel(*), enumerate(f.layout)) do (axisIndex, axisOffset)
        encoder = (x, y) -> (getindex(x, axisIndex) - y)
        mapreduce(JoinParallel(+), innerEvalFuncs, axisOffset) do ief, oPar
            parIds = (locateParam!(paramSet, oPar),)
            InsertInward(ief, PointerFunc(encoder, parIds, pSetId))
        end
    end
    EvalAxialProd(evalFunc), paramSet
end

(f::AxialProd{<:Any, D})(r::NTuple{D, Real}) where {D} = evalFunc(f, r)


struct BundleSum{T, D, F<:SpatialAmplitude{T, D, 1}} <: SpatialAmplitude{T, D, 1}
    bundle::Memory{F}

    function BundleSum(bundle::AbstractVector{F}) where {T, D, F<:SpatialAmplitude{T, D, 1}}
        checkEmptiness(bundle, :bundle)
        new{T, D, F}(getMemory(bundle))
    end
end

struct EvalBundleSum{F<:JoinParallel} <: Evaluator{BundleSum}
    f::F
end

function unpackParamFunc!(f::BundleSum, paramSet::PBoxCollection)
    innerEvalFuncs = map(Fix2(unpackFunc!, paramSet), f.bundle) .|> first
    evalFunc = reduce(JoinParallel(+), innerEvalFuncs)
    EvalBundleSum(evalFunc), paramSet
end

(f::BundleSum{<:Any, D})(r::NTuple{D, Real}) where {D} = evalFunc(f, r)


struct Angularizer{T, D, A<:SphericalHarmonics{D}, 
                   B<:SpatialAmplitude{T, D, 1}} <:SpatialAmplitude{T, D, 1}
    angular::A
    multiplier::B
end

struct EvalAngularizer{F<:JoinParallel} <: Evaluator{Angularizer}
    f::F
end

function unpackParamFunc!(f::Angularizer, paramSet::PBoxCollection)
    evalMulFunc = unpackFunc!(f.multiplier, paramSet) |> first
    evalFunc = JoinParallel(evalMulFunc, OnlyInput(f.angular), *)
    EvalBundleSum(evalFunc), paramSet
end

(f::Angularizer{<:Any, D})(r::NTuple{D, Real}) where {D} = evalFunc(f, r)