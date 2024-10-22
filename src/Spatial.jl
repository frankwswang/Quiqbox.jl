(f::SpatialAmplitude{<:Any, <:Any, 1})(x) = 
formatInputCall(SelectTrait{InputStyle}()(f), f, x)


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

(gf::GaussFunc)(r::Real) = evalFunc(gf, r)


struct AxialProd{T<:Real, D, F<:SpatialAmplitude{T, 0, 1}, 
                 L<:NTuple{D, Memory{<:ElementalParam{T}}}} <: SpatialAmplitude{T, D, 1}
    series::Memory{F}
    layout::L

    function AxialProd(series::Memory{F}, layout::L) where 
                      {T, F<:SpatialAmplitude{T, 0, 1}, D, 
                       L<:NonEmptyTuple{Memory{<:ElementalParam{T}}, D}}
        if isempty(series)
            throw(AssertionError("`series` must not be empty."))
        else
            nFunc = length(series)
            if any(l != nFunc for l in length.(layout))
                throw(AssertionError("`series` and each `layout` element must have equal "*
                                     "length."))
            end
        end
        new{T, D+1, F, L}(series, layout)
    end
end

function AxialProd(series::Memory{F}, dim::Int=3) where {T, F<:SpatialAmplitude{T, 0, 1}}
    dim < 1 && throw(AssertionError("`dim` should be a positive integer."))
    cell = CellParam(zero(T), :r)
    setScreenLevel!(cell, 2)
    layout = fill(Memory{ItselfCParam{T}}( fill(cell, length(series)) ), dim) |> Tuple
    AxialProd(series, layout)
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
            InsertInward(ief.f, PointerFunc(encoder, parIds, pSetId))
        end
    end
    EvalAxialProd(evalFunc), paramSet
end

(gf::AxialProd{<:Any, D})(r::NTuple{D, Real}) where {D} = evalFunc(gf, r)


struct BundleSum{T, D, F<:SpatialAmplitude{T, D, 1}} <: SpatialAmplitude{T, D, 1}
    bundle::Memory{F}
end

struct EvalBundleSum{F<:JoinParallel} <: Evaluator{BundleSum}
    f::F
end

function unpackParamFunc!(f::BundleSum, paramSet::PBoxCollection)
    innerEvalFuncs = map(Fix2(unpackFunc!, paramSet), f.bundle) .|> first
    evalFunc = mapreduce(JoinParallel(+), innerEvalFuncs) do ief; ief.f end
    EvalBundleSum(evalFunc), paramSet
end

(gf::BundleSum{<:Any, D})(r::NTuple{D, Real}) where {D} = evalFunc(gf, r)