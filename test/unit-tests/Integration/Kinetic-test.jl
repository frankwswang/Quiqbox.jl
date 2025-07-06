using Test
using Quiqbox

@testset "Kinetic-Based Features" begin

cen_sto1 = (1.1, 2.2)

stf1Core = Quiqbox.ModularField(x->exp(-first(x)), Float64, Count(1))
stf1 = Quiqbox.PolyRadialFunc(stf1Core, (1, 1))
sto1 = PrimitiveOrb(cen_sto1, stf1; renormalize=false)
ke1 = eKinetic(sto1, sto1)

genSymKE_sto1Core = function (cen::NTuple{2, Float64})
    let a1 = first(cen), a2=last(cen)
        function symKE_stoCore((x, y)::NTuple{2, Float64})
            r = sqrt((a1 - x)^2 + (a2 - y)^2)
            k = 0.5 * (x - a1) * (a2 - y) * exp(-r) / r^3
            k * ((r*a1^2 - 2a1^2 + r*x^2 - 2a1*x*r + 4a1*x - 3a2^2 + 6a2*y - 2x^2 - 3y^2) + 
                 (r*a2^2 - 2a2^2 + r*y^2 - 2a2*y*r + 4a2*y - 3a1^2 + 6a1*x - 2y^2 - 3x^2))
        end
    end
end
symKE_stf1Core = genSymKE_sto1Core(cen_sto1)
symKE_stf1 = Quiqbox.ModularField(symKE_stf1Core, Float64, Count(2))
symKE_sto1 = PrimitiveOrb((0., 0.), symKE_stf1; renormalize=false)
ke1_t = overlap(sto1, symKE_sto1)

@test ke1 ≈ 0.7363107784289
@test isapprox(ke1, ke1_t, atol=1e-12)

ke1mat = eKinetics([sto1])
@test size(ke1mat) == (1, 1)
@test ke1mat[] == ke1
@test unique(eKinetics([sto1, sto1]))[] == ke1

pgf1 = genGaussTypeOrb((0.1, 0.2, 0.3), 2.0, (1, 0, 0))
@test eKinetic(pgf1, pgf1) ≈ eKinetic(pgf1, pgf1, lazyCompute=Quiqbox.False()) ≈ 
      0.4350256247524772
pgf1_masked = PrimitiveOrb((0., 0., 0.), Quiqbox.EncodedField( pgf1, Float64, Count(3) ))
@test eKinetic(pgf1_masked, pgf1) ≈ eKinetic(pgf1, pgf1) ≈ 0.43502562475512524
@test eKinetic(pgf1, pgf1, lazyCompute=Quiqbox.False()) ≈ 0.43502562475512524

cgf1 = genGaussTypeOrb((1.1, 0.5, 1.1), [1.2, 0.6], [1.5, -0.3], (1, 2, 2))
@test eKinetic(cgf1, cgf1) ≈ eKinetic(cgf1, cgf1, lazyCompute=Quiqbox.False()) ≈ 
      0.06737210531634309

end