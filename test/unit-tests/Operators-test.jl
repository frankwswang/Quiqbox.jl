using Test
using Quiqbox
using Quiqbox: DiagonalDiff, Summator, MultiMonoApply, ComplexConj, genIdentity

@testset "Operators.jl" begin

f = function (tpl::NTuple{3, Real})
    x, y, z = tpl
    x*y + log(x^2 + z^4) / (y^2 + 1) - (z + x)^3
end

g_sd = function (tpl::NTuple{3, Real})
    x, y, z = tpl
    tmp1 = (y^2 + 1)
    tmp2 = (x^2 + z^4) * tmp1
    tmp3 = 3(z + x)^2
    ((y + 2x / tmp2 - tmp3), (x - 2y * log(x^2 + z^4) / tmp1^2), (4z^3/tmp2 - tmp3))
end

l_sd = function (tpl::NTuple{3, Real})
    x, y, z = tpl
    tmp1 = (y^2 + 1)
    tmp2 = (x^2 + z^4)
    ((2(z^4 - x^2) / (tmp1 * tmp2^2) - 6(x + z)), (2log(tmp2) * (3y^2 - 1) / tmp1^3), 
     ( (12z^2 * tmp2 - 16z^6) / (tmp1 * tmp2^2) - 6(x + z) ))
end

coord1 = (1.1, 2.2, 3.3)
f_typed = Quiqbox.TypedCarteFunc(f, Float64, Count(3))

∇v = Quiqbox.DiagonalDiff(Count(1), (1.0, 1.0, 1.0))
∇vf = ∇v(f_typed)
@test isapprox(∇vf(coord1), (sum∘g_sd)(coord1), atol=1e-12)
@test all(isapprox.(∇vf.right(coord1), g_sd(coord1), atol=1e-12))

Δ = Quiqbox.DiagonalDiff(Count(2), (1.0, 1.0, 1.0))
Δf = Δ(f_typed)
@test isapprox(Δf(coord1), (sum∘l_sd)(coord1), atol=5e-9)
@test all(isapprox.(Δf.right(coord1), l_sd(coord1), atol=5e-9))

op = Summator(MultiMonoApply(ComplexConj()|>tuple), genIdentity())
f = op(x->x^3)
@test f(1 - im) == let x=(1 - im)^3; conj(x) + x end

end