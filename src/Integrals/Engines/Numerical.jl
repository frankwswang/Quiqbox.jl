using HCubature

struct ConfineInterval{T, F<:ReturnTyped{T}} <: FunctionModifier
    f::F

    ConfineInterval(f::F) where {T, F<:ReturnTyped{T}} = new{T, F}(f)
end

ConfineInterval(::Type{T}, f::Function) where {T} = ConfineInterval(ReturnTyped(f, T))

function (f::ConfineInterval{T})(x::AbstractVector{T}) where {T}
    val = f.f(x ./ (one(T) .- x.^2))
    mapreduce(*, x) do t
        tSquare = t * t
        (1 + tSquare) / (1 - tSquare)^2
    end * val
end


struct ModSquaredMap{T, F<:ReturnTyped{T}} <: FunctionModifier
    f::F

    ModSquaredMap(f::F) where {T, F<:ReturnTyped{T}} = new{T, F}(f)
end

ModSquaredMap(::Type{T}, f::Function) where {T} = ModSquaredMap(ReturnTyped(f, T))

function (f::ModSquaredMap{T})(input::AbstractVector{T}) where {T}
    val = f.f(input)
    val' * val
end


function numericalIntegrateCore(integrand::ConfineInterval{T}, 
                                interval::NTuple{2, NonEmptyTuple{T, D}}) where {T, D}
    fullRes = hcubature(integrand, first(interval), last(interval), maxevals=typemax(Int))
    first(fullRes)
end


function composeOneBodyKernel(op::O, ::Type{T}, (termL, termR)::Tuple{F1, F2}) where 
                             {O<:Function, T, F1<:Function, F2<:Function}
    PairCombine(StableMul(T), adjoint∘termL, op(termR))
end

function composeOneBodyKernel(op::O, ::Type{T}, (term,)::Tuple{F}) where 
                             {O<:Function, T, F<:Function}
    composeOneBodyKernel(op, T, (term, term))
end

function composeOneBodyKernel(::Identity, ::Type{T}, (term,)::Tuple{F}) where 
                             {T, F<:Function}
    ModSquaredMap(T, term)
end


function composeIntegralKernel(::OneBodyIntegral, op::O, ::Type{T}, 
                               terms::N12Tuple{Function}) where {O<:Function, T}
    composeOneBodyKernel(op, T, terms)
end

function numericalIntegrate(::OneBodyIntegral{D}, op::F, 
                            orbs::NonEmptyTuple{EvalFieldFunction{T, D}, N}, 
                            pVals::NonEmptyTuple{FilteredVecOfArr{T}, N}) where 
                            {F<:DirectOperator, T, D, N}
    terms = Base.Fix2.(orbs, pVals)
    integralKernel = composeIntegralKernel(OneBodyIntegral{D}(), op, T, terms)
    integrand = ConfineInterval(T, integralKernel)
    bound = ntuple(_->one(T), Val(D))
    numericalIntegrateCore(integrand, (.-(bound), bound))
end


function buildNormalizerCore(o::PrimitiveOrbCore{T, D}) where {T, D}
    buildOneBodyCoreIntegrator(Identity(), (o,))
end


struct OneBodyNumIntegrate{T, D, F<:DirectOperator, 
                           P<:N12Tuple{PrimitiveOrbCore{T, D}}} <: OrbitalIntegrator{T, D}
    op::F
    basis::P
end

const OneBodySelfNumInt{T, D, F<:DirectOperator, P<:PrimitiveOrbCore{T, D}} = 
      OneBodyNumIntegrate{T, D, F, Tuple{P}}
const OnyBodyPairNumInt{T, D, F<:DirectOperator, P1<:PrimitiveOrbCore{T, D}, 
                        P2<:PrimitiveOrbCore{T, D}} = 
      OneBodyNumIntegrate{T, D, F, Tuple{P1, P2}}

function (f::OneBodySelfNumInt{T, D})(pVal::FilteredVecOfArr{T}) where {T, D}
    numericalIntegrate(OneBodyIntegral{D}(), f.op, f.basis, (pVal,))
end

function (f::OnyBodyPairNumInt{T, D})(pVal1::FilteredVecOfArr{T}, 
                                      pVal2::FilteredVecOfArr{T}) where {T, D}
    numericalIntegrate(OneBodyIntegral{D}(), f.op, f.basis, (pVal1, pVal2))
end