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

getOutputType(::Type{<:SesquiFieldProd{T, D, <:StableTypedSampler}}) where {T<:Real, D} = 
T

getOutputType(::Type{<:SesquiFieldProd{T, D, O}}) where 
             {T<:Real, D, C<:RealOrComplex{T}, O<:ReturnTypedSampler{C}} = 
strictTypeJoin(T, C)

(::SelectTrait{InputStyle})(::SesquiFieldProd{<:Real, D}) where {D} = CartesianInput{D}()

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
                       L<:SesquiFieldProd{<:RealOrComplex{T}, D}, 
                       R<:SesquiFieldProd{<:RealOrComplex{T}, D}} <: ParticleFunction{D, 2}
    layout::Tuple{L, R}
    coupler::O
end

function DoubleFieldProd(pairL::N12Tuple{PrimOrbData{T, D}}, 
                         pairR::N12Tuple{PrimOrbData{T, D}}, 
                         coupler::DualTermOperator) where {T, D}
    sfProdL = SesquiFieldProd(pairL)
    sfProdR = SesquiFieldProd(pairR)
    DoubleFieldProd((sfProdL, sfProdR), coupler)
end

getOutputType(::Type{<:DoubleFieldProd{T, D, <:StableTypedSampler}}) where {T<:Real, D} = 
T

getOutputType(::Type{<:DoubleFieldProd{T, D, O}}) where 
             {T<:Real, D, C<:RealOrComplex{T}, O<:ReturnTypedSampler{C}} = 
strictTypeJoin(T, C)


(::SelectTrait{InputStyle})(::DoubleFieldProd{<:Real, D}) where {D} = CartesianInput{2D}()

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


function genIntegrant(op::Multiplier, 
                      (pair,)::Tuple{N12Tuple{ PrimOrbData{T, D} }}) where {T<:Real, D}
    SesquiFieldProd(getfield.(pair, :core), op)
end

function genIntegrant(op::DualTermOperator, 
                      (pL, pR)::NTuple{2, N12Tuple{ PrimOrbData{T, D} }}) where {T<:Real, D}
    DoubleFieldProd(getfield.(pL, :core), getfield.(pR, :core), op)
end


function genIntegralSectors(op::DirectOperator, layout::CoreIntegralOrbDataLayout)
    tuple(genIntegrant(op, layout))
end


function getIntegratorConfig(::Type{T}, ::ParticleFunction) where {T<:Real}
    HCubatureConfig(T)
end


function estimateOrbIntegral(config::MissingOr{EstimatorConfig{T}}, 
                             op::TypedOperator{C}, layout::CoreIntegralOrbDataLayout{T, D}
                             ) where {T<:Real, C<:RealOrComplex{T}, D}
    sectors = genIntegralSectors(op.core, layout)
    noGlobalConfig = ismissing(config)
    res = one(C)
    for kernel in sectors
        configLocal = noGlobalConfig ? getIntegratorConfig(T, kernel) : config
        res *= numericalIntegrateCore(configLocal, kernel)
    end
    convert(C, res)
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
    res = one(T)
    for t in x
        tSquare = t * t
        res *= (1 + tSquare) / (1 - tSquare)^2
    end
    res * f.core(x ./ (one(T) .- x .* x))
end

function getConfinedInterval(::ConfinedInfIntegrand{T, L}) where {T<:Real, L}
    bound = ntuple(_->one(T), Val(L))
    (.-(bound), bound)
end


struct HCubatureConfig{T<:Real} <: EstimatorConfig{T}
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