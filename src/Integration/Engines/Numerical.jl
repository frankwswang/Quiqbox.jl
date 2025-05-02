using HCubature
using LinearAlgebra: dot

struct OrbitalFunc{D, F<:DirectOperator, C<:OrbitalData{<:Number, D}
                   } <: SpatialAmplitude{D, 1}
    action::F
    config::C

    function OrbitalFunc(action::F, config::C) where {D, C<:OrbitalData{<:Number, D}, 
                                                      F<:DirectOperator}
        new{D, F, C}(action, config)
    end
end

OrbitalFunc(config) = OrbitalFunc(Identity(), config)

function (f::OrbitalFunc{D})(coord::NTuple{D, Real}) where {D}
    f.action(Base.Fix1(evalOrbitalConfig, f.config))(coord)
end

function evalOrbitalConfig(config::PrimOrbData{T, D}, coord::NTuple{D, Real}) where {T, D}
    input = coord .- config.center
    fCore, params = config.body
    fCore.core.f(input, params)
end


struct OrbitalInnerProd{D, P<:N12Tuple{ SpatialAmplitude{D, 1} }} <: SpatialAmplitude{D, 1}
    term::P
end

(::SelectTrait{InputStyle})(::OrbitalInnerProd{D}) where {D} = CoordInput{D}()

const SelfOrbInnerProd{D, F<:SpatialAmplitude{D, 1}} = OrbitalInnerProd{D, Tuple{F}}

function (f::SelfOrbInnerProd{D})(coord::NTuple{D, Real}) where {D}
    val = first(f.term)(coord)
    val' * val
end

function (f::OrbitalInnerProd{D})(coord::NTuple{D, Real}) where {D}
    fL, fR = f.term
    valL = fL(coord)
    valR = fR(coord)
    valL' * valR
end


struct DoubleOrbProduct{D, F1<:OrbitalInnerProd{D}, F2<:OrbitalInnerProd{D}, 
                        C<:DirectOperator} <: SpatialAmplitude{D, 2}
    one::F1
    two::F2
    coupler::C
end

(::SelectTrait{InputStyle})(::DoubleOrbProduct{D}) where {D} = CoordInput{2D}()

function (f::DoubleOrbProduct{D})(coord1::NTuple{D, Real}, 
                                  coord2::NTuple{D, Real}) where {D}
    f.coupler(f.one, f.two)(coord1, coord2)
end

function (f::DoubleOrbProduct{D})(coord::NTuple{DD, Real}) where {D, DD}
    f.coupler(f.one, f.two)(coord[begin:begin+D-1], coord[begin+D:end])
end


struct ConfinedInfIntegrand{T, L, F<:Function} <: FunctionModifier
    core::F

    function ConfinedInfIntegrand(f::F, ::Type{T}) where {F<:Function, T}
        new{T, getDimension(f), F}(f)
    end
end

function (f::ConfinedInfIntegrand{T})(x) where {T}
    val = f.core( formatInput(f.core, x ./ (one(T) .- x .* x)) )
    mapreduce(*, x) do t
        tSquare = t * t
        (1 + tSquare) / (1 - tSquare)^2
    end * val
end


struct HCubatureConfig{T} <: ConfigBox
    maxEvalNum::Int
                                                # 1000 per one of six dimensions
    function HCubatureConfig(::Type{T}, maxEvalNum::Int=1000_000_000_000_000_000
                             ) where {T}
        checkPositivity(maxEvalNum)
        new{T}(maxEvalNum)
    end
end


function getIntegratorConfig(::Type{T}, ::SpatialAmplitude) where {T}
    HCubatureConfig(T)
end

function getConfinedInterval(::ConfinedInfIntegrand{T, L}) where {T, L}
    bound = ntuple(_->one(T), Val(L))
    (.-(bound), bound)
end


function genIntegrant(::OneBodyIntegral{D}, op::Identity, 
                      (data,)::Tuple{PrimOrbData{T, D}}) where {T, D}
    orbFunc = OrbitalFunc(op, data)
    OrbitalInnerProd((orbFunc,))
end

function genIntegrant(::OneBodyIntegral{D}, op::DirectOperator, 
                      (dataL, dataR)::NTuple{2, PrimOrbData{T, D}}) where {T, D}
    orbFuncL = OrbitalFunc(dataL)
    orbFuncR = OrbitalFunc(op, dataR)
    OrbitalInnerProd((orbFuncL, orbFuncR))
end


function genIntegrantSubspaces(intStyle::MultiBodyIntegral{D}, op::DirectOperator, 
                               data::NonEmptyTuple{PrimOrbData{T, D}}) where {T, D}
    tuple(genIntegrant(intStyle, op, data))
end


function boundIntegralReturnType(::DirectOperator, 
                                 ::NonEmptyTuple{PrimOrbData{T, D}}) where {T, D}
    Number
end


function getNumericalIntegral(::OneBodyIntegral{D}, op::F, data::N12Tuple{PrimOrbData{T, D}}
                              ) where {F<:DirectOperator, T, D}
    spaces = genIntegrantSubspaces(OneBodyIntegral{D}(), op, data)
    mapreduce(*, spaces) do kernel
        getNumericalIntegral(kernel, T)
    end::boundIntegralReturnType(op, data)
end

function getNumericalIntegral(integrand::SpatialAmplitude, ::Type{T}) where {T<:Number}
    config = getIntegratorConfig(extractRealNumberType(T), integrand)
    convert(T, numericalIntegrateCore(config, integrand))
end

function numericalIntegrateCore(config::HCubatureConfig{T}, integrand::Function) where {T}
    formattedInt = ConfinedInfIntegrand(integrand, T)
    interval = getConfinedInterval(formattedInt)
    fullRes = hcubature(formattedInt, interval..., maxevals=config.maxEvalNum)
    first(fullRes)
end