
function gaussProdCore1(R1, R2, α1, α2)
    α = α1 + α2
    exp(-α1 / α * α2* sum(abs2, R1 .- R2))
end

function gaussProdCore2(R1, R2, α1, α2)
    α = α1 + α2
    (α1 .* R1 .+ α2 .* R2) ./ α
end

function gaussProdCore3(x1, x2, x, l1, l2, l)
    lb = max(-l, l-2l2)
    ub = min( l, 2l1-l)
    map(lb:2:ub) do q
        i = (l + q) ÷ 2
        j = (l - q) ÷ 2
        binomial(l1,  i) * binomial(l2,  j) * (x - x1)^(l1 -i) * (x - x2)^(l2 -j)
    end |> sum
end

# function gaussProdCore4(cen1, cen2, α1, α2, ijk1, ijk2, lmn)
#     cen = gaussProdCore2(α1, R1, α2, R2)
#     ijk = ijk1 + ijk2
#     lmnRange =  Iterators.product(Base.OneTo.(ijk)...)
#     map(lmnRange) do (l, m, n)
#         lmn = (l, m, n)
#         mapreduce(*, gaussProdCore3, cen1, cen2, cen, ijk1, ijk2, lmn)
#     end
# end

gaussProdCore4(cen1, cen2, cen, ijk1, ijk2, lmn) = 
mapreduce(*, gaussProdCore3, cen1, cen2, cen, ijk1, ijk2, lmn)

# function gaussProdCore3(α₁, α₂)
#     α₁ * α₂ / (α₁ + α₂)
# end

# function gaussProdCore4(R₁, R₂)
#     sum((R₁ .+ R₂).^2)
# end

function doubleFactorial(a::Int)
    if a < 0
        throw(AssertionError("`a` must be non-negative."))
    elseif isodd(a)
        oddFactorialCore(a)
    else
        evenFactorialCore(a)
    end
end

function oddFactorialCore(a::Int)
    factorial(2a) ÷ (2^a * factorial(a))
end

function evenFactorialCore(a::Int)
    2^a * factorial(a)
end



function polyGaussFuncNormFactor(α::T, i::Int) where {T<:Real}
    factor = i > 0 ? sqrt((4α)^i / oddFactorialCore(2i - 1)) : one(T)
    T((sqrt∘sqrt)(2α/πPowers[:p1d0])) * factor
end