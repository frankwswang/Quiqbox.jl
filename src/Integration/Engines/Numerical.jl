using HCubature
using LinearAlgebra: dot

struct SesquiFieldProd{T<:Real, D, O<:Multiplier, 
                       F<:N12Tuple{FieldAmplitude{<:RealOrComplex{T}, D}}
                       } <: ParticleFunction{D, 1}
    layout::F
    dresser::O

    function SesquiFieldProd(fields::N12Tuple{FieldAmplitude{<:RealOrComplex{T}, D}}, 
                             dresser::O=genOverlapSampler()) where 
                            {T<:Real, D, O<:Multiplier}
        new{T, D, O, typeof(fields)}(fields, dresser)
    end
end

SesquiFieldProd(data::N12Tuple{PrimOrbData{T, D}}, dresser::O=genOverlapSampler()
                ) where {T<:Real, D, O<:Multiplier} = 
SesquiFieldProd(getfield.(data, :core), dresser)

(::SelectTrait{InputStyle})(::SesquiFieldProd{<:Real, D}) where {D} = EuclideanInput{D}()

(f::SesquiFieldProd)(coord) = evalSesquiFieldProd(f, formatInput(f, coord))

const SelfModeOverlap{T<:Real, D, F<:FieldAmplitude{<:RealOrComplex{T}, D}} = 
      SesquiFieldProd{T, D, OverlapSampler, Tuple{F}}

function evalSesquiFieldProd(f::SelfModeOverlap{<:Real, D}, 
                             coord::NTuple{D, Real}) where {D}
    field, = f.layout
    val = field(coord)
    conj(val) * val
end

function evalSesquiFieldProd(f::SesquiFieldProd{<:Real, D, O}, coord::NTuple{D, Real}
                             ) where {D, O<:Multiplier}
    fields = ifelse(length(f.layout)==1, ntuple(_->first(f.layout), Val(2)), f.layout)
    f.dresser(fields...)(coord)
end


struct DoubleFieldProd{T<:Real, D, O<:DualTermOperator, 
                       F<:N12Tuple{SesquiFieldProd{<:RealOrComplex{T}, D}}
                       } <: ParticleFunction{D, 2}
    layout::F
    coupler::O
end

function DoubleFieldProd(layout::OrbBarLayout1{PrimOrbData{T, D}}, 
                         coupler::DualTermOperator) where {T<:Real, D}
    sfProd = SesquiFieldProd(layout)
    DoubleFieldProd((sfProd,), coupler)
end

function DoubleFieldProd(layout::OrbBarLayout2{PrimOrbData{T, D}}, 
                         coupler::DualTermOperator) where {T<:Real, D}
    sfProdL = SesquiFieldProd((first(layout),))
    sfProdR = SesquiFieldProd((last(layout),))
    DoubleFieldProd((sfProdL, sfProdR), coupler)
end

function DoubleFieldProd(layout::OrbBarLayout3{PrimOrbData{T, D}}, 
                         coupler::DualTermOperator) where {T<:Real, D}
    pair..., _ = layout
    sfProd = SesquiFieldProd(pair)
    DoubleFieldProd((sfProd,), coupler)
end

function DoubleFieldProd(layout::OrbBarLayout4{PrimOrbData{T, D}}, 
                         coupler::DualTermOperator) where {T<:Real, D}
    sfProdL = SesquiFieldProd((first(layout),))
    sfProdR = SesquiFieldProd((layout[end-1], layout[end]))
    DoubleFieldProd((sfProdL, sfProdR), coupler)
end

function DoubleFieldProd(layout::OrbBarLayout5{PrimOrbData{T, D}}, 
                         coupler::DualTermOperator) where {T<:Real, D}
    sfProdL = SesquiFieldProd((layout[begin], layout[begin+1]))
    sfProdR = SesquiFieldProd((last(layout),))
    DoubleFieldProd((sfProdL, sfProdR), coupler)
end

function DoubleFieldProd(layout::OrbBarLayout6{PrimOrbData{T, D}}, 
                         coupler::DualTermOperator) where {T<:Real, D}
    a, b, c, d = layout
    sfProdL = SesquiFieldProd((a, b))
    sfProdR = SesquiFieldProd((c, d))
    DoubleFieldProd((sfProdL, sfProdR), coupler)
end


(::SelectTrait{InputStyle})(::DoubleFieldProd{<:Real, D}) where {D} = EuclideanInput{2D}()

(f::DoubleFieldProd)(coord) = evalDoubleFieldProd(f, formatInput(f, coord))

function evalDoubleFieldProd(f::DoubleFieldProd{<:Real, D, O}, 
                             coord1::NTuple{D, Real}, coord2::NTuple{D, Real}) where 
                            {D, O<:DualTermOperator}
    lp, rp = f.layout
    f.coupler(lp, rp)(coord1, coord2)
end

function evalDoubleFieldProd(f::DoubleFieldProd{<:Real, D}, 
                             coord::NonEmptyTuple{Real}) where {D}
    evalDoubleFieldProd(f, coord[begin:begin+D-1], coord[begin+D:end])
end


function genIntegrant(::OneBodyIntegral{D}, op::Multiplier, 
                      layout::OneBodyOrbCoreIntLayoutUnion{T, D}) where {T<:Real, D}
    SesquiFieldProd(layout, op)
end

function genIntegrant(::TwoBodyIntegral{D}, op::DualTermOperator, 
                      layout::TwoBodyOrbCoreIntLayoutUnion{T, D}) where {T<:Real, D}
    DoubleFieldProd(layout, op)
end


function genIntegralSectors(intStyle::MultiBodyIntegral{D}, op::DirectOperator, 
                            layout::OrbCoreIntLayoutUnion{T, D}) where {T<:Real, D}
    tuple(genIntegrant(intStyle, op, layout))
end


function getIntegratorConfig(::Type{T}, ::ParticleFunction) where {T<:Real}
    HCubatureConfig(T)
end


function getNumericalIntegral(intStyle::MultiBodyIntegral{D}, op::DirectOperator, 
                              layout::OrbCoreIntLayoutUnion{T, D}) where {T<:Real, D}
    sectors = genIntegralSectors(intStyle, op, layout)
    mapreduce(*, sectors) do kernel
        config = getIntegratorConfig(getOrbOutputTypeUnion(layout), kernel)
        numericalIntegrateCore(config, kernel)
    end::RealOrComplex{T}
end


struct ConfinedInfIntegrand{T<:Real, L, F<:Function} <: Modifier
    core::F

    function ConfinedInfIntegrand(f::F, ::Type{T}) where {F<:Function, T<:Real}
        new{T, getDimension(f)::Int, F}(f)
    end
end


(f::ConfinedInfIntegrand)(x) = evalConfinedInfIntegrand(f, x)

function evalConfinedInfIntegrand(f::ConfinedInfIntegrand{T, L, F}, 
                                  x::Union{NTuple{L, T}, AbstractVector{T}}) where 
                                 {T<:Real, L, F<:Function}
    val = f.core(x ./ (one(T) .- x .* x))
    mapreduce(*, x) do t
        tSquare = t * t
        (1 + tSquare) / (1 - tSquare)^2
    end * val
end

function getConfinedInterval(::ConfinedInfIntegrand{T, L}) where {T<:Real, L}
    bound = ntuple(_->one(T), Val(L))
    (.-(bound), bound)
end


struct HCubatureConfig{T<:Real} <: ConfigBox
    tolerance::T
    maxEvalNum::Int

    function HCubatureConfig(::Type{T}, tolerance::T=zero(T); 
                             maxEvalNum::Int=1000_000_000_000_000_000) where {T<:Real}
        checkPositivity(tolerance, true)
        checkPositivity(maxEvalNum) # 1000 per one of six dimensions
        new{T}(tolerance, maxEvalNum)
    end
end

function numericalIntegrateCore(config::HCubatureConfig{T}, 
                                integrand::Function) where {T<:Real}
    formattedInt = ConfinedInfIntegrand(integrand, T)
    interval = getConfinedInterval(formattedInt)
    atol = config.tolerance
    rtol = ifelse(iszero(atol), sqrt(eps(T)), zero(T))
    fullRes = hcubature(formattedInt, interval...; rtol, atol, maxevals=config.maxEvalNum)
    first(fullRes)
end