using HCubature

function changeFuncRange(f::F, ::Type{T}) where {F, T}
    fCore = function (x)
        x = Tuple(x)
        f(x ./ (one(T) .- x.^2)) * mapreduce(*, x) do t
            (1 + t^2) / (1 - t^2)^2
        end
    end
    ReturnTyped(fCore, T)
end


function numericalIntegration(integrand::ReturnTyped{T}, 
                              interval::NTuple{2, NonEmptyTuple{T, D}}) where {T, D}
    (first∘hcubature)(integrand, first(interval), last(interval); maxevals=typemax(Int))
end


function composeOneBodyKernel(op::O, ::Type{T}, termL::F1, termR::F2) where 
                             {O<:Function, T, F1<:Function, F2<:Function}
    PairCombine(StableBinary(*, T), adjoint∘termL, op(termR))
end

function composeOneBodyKernel(op::O, ::Type{T}, term::F) where {O<:Function, T, F<:Function}
    composeOneBodyKernel(op, T, term, term)
end

function composeOneBodyKernel(::Identity, ::Type{T}, term::F) where {T, F<:Function}
    function (input::I) where {I}
        val = term(input)::T
        val' * val
    end
end

function numericalOneBodyInt(op::F, (orb,)::Tuple{EvalFieldFunction{T, D}}, 
                             (pVal,)::Tuple{FilteredVecOfArr{T}}) where 
                            {F<:DirectOperator, T, D}
    term = Base.Fix2(orb, pVal)
    integrand = changeFuncRange(composeOneBodyKernel(op, T, term), T)
    bound = ntuple(_->one(T), Val(D))
    numericalIntegration(integrand, (.-(bound), bound))
end

function numericalOneBodyInt(op::F, orbs::NTuple{2, EvalFieldFunction{T, D}}, 
                             pValPair::NTuple{2, FilteredVecOfArr{T}}) where 
                            {F<:DirectOperator, T, D}
    termL, termR = Base.Fix2.(orbs, pValPair)
    integrand = changeFuncRange(composeOneBodyKernel(op, T, termL, termR), T)
    bound = ntuple(_->one(T), Val(D))
    numericalIntegration(integrand, (.-(bound), bound))
end


struct NumOverlapOrbSelf{T, D, B<:PrimitiveOrbCore{T, D}} <: OrbitalIntegrator{T, D}
    orb::Tuple{B}
end

function (f::NumOverlapOrbSelf{T})(pars::FilteredVecOfArr{T}) where {T}
    numericalOneBodyInt(Identity(), f.orb, (pars,))
end

struct NumOverlapOrbPair{T, D, B1<:PrimitiveOrbCore{T, D}, 
                         B2<:PrimitiveOrbCore{T, D}} <: OrbitalIntegrator{T, D}
    orb::Tuple{B1, B2}
end

function (f::NumOverlapOrbPair{T})(pars::Vararg{FilteredVecOfArr{T}, 2}) where {T}
    numericalOneBodyInt(Identity(), f.orb, pars)
end

genOneBodyCoreIntegrator(::Identity, orbs::Tuple{PrimitiveOrbCore{T, D}}) where {T, D} = 
NumOverlapOrbSelf(orbs)

genOneBodyCoreIntegrator(::Identity, orbs::NTuple{2, PrimitiveOrbCore{T, D}}) where {T, D} = 
NumOverlapOrbPair(orbs)

function buildNormalizerCore(o::PrimitiveOrbCore{T, D}) where {T, D}
    genOneBodyCoreIntegrator(Identity(), (o,))
end


#!! Implement a functional struct (<:OrbitalIntegrator{T, D}) for the closure
function genOneBodyCoreIntegrator(op::F, 
                                  orbs::NonEmptyTuple{PrimitiveOrbCore{T, D}, N}) where 
                                 {F<:DirectOperator, N, T, D}
    function (pVal::AbtVecOfAbtArr{T}, pVals::Vararg{AbtVecOfAbtArr{T}, N}) where {T}
        numericalOneBodyInt(op, orbs, (pVal, pVals...))
    end
end