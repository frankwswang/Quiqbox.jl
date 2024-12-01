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


numericalOneBodyInt(op::F, orb::EvalDimensionalKernel{T, D}, 
                    pValDict::ParamValOrDict{T}) where {F<:DirectOperator, T, D} = 
numericalOneBodyInt(op, (orb, orb), (pValDict, pValDict))

function numericalOneBodyInt(op::F, orbs::NTuple{2, EvalDimensionalKernel{T, D}}, 
                             pValGroups::NTuple{2, ParamValOrDict{T}}) where 
                            {F<:DirectOperator, T, D}
    termL, termR = Base.Fix2.(orbs, pValGroups)
    integrand = changeFuncRange(PairCombine(StableBinary(*, T), termL, op(termR)), T)
    bound = ntuple(_->one(T), Val(D))
    numericalIntegration(integrand, (.-(bound), bound))
end