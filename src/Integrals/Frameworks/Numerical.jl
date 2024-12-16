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

function numericalOneBodyInt(op::F, (orb,)::Tuple{EvalDimensionalKernel{T, D}}, 
                             (pValDict,)::Tuple{ParamValOrDict{T}}) where 
                            {F<:DirectOperator, T, D}
    term = Base.Fix2(orb, pValDict)
    integrand = changeFuncRange(composeOneBodyKernel(op, T, term), T)
    bound = ntuple(_->one(T), Val(D))
    numericalIntegration(integrand, (.-(bound), bound))
end

function numericalOneBodyInt(op::F, orbs::NTuple{2, EvalDimensionalKernel{T, D}}, 
                             pValGroups::NTuple{2, ParamValOrDict{T}}) where 
                            {F<:DirectOperator, T, D}
    termL, termR = Base.Fix2.(orbs, pValGroups)
    integrand = changeFuncRange(composeOneBodyKernel(op, T, termL, termR), T)
    bound = ntuple(_->one(T), Val(D))
    numericalIntegration(integrand, (.-(bound), bound))
end