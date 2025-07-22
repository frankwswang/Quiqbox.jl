using Test
using Quiqbox
using LinearAlgebra

@testset "FieldFunctions.jl" begin

xpn1 = genCellParam(1.5, :xpn)
xpn2 = genCellParam(2.5, :xpn)
xpn3 = genCellParam(0.5, :xpn)

xpns = [xpn1, xpn2, xpn3]

gf1, gf2, gf3 = gfs1 = Quiqbox.GaussFunc.(xpns)

@test uniqueParams(gf1)[] === xpn1
@test all(uniqueParams(gfs1) .=== [xpn1, xpn2, xpn3])
@test gf1(0) == exp(0)

gf1_core, gf1_par = Quiqbox.unpackFunc(gf1)
fixedPoint = 1.1
gf1Ofpar = Base.Fix1(gf1_core, 1.1)
gf1_parVal = map(obtain, gf1_par)
@test 0.1628379083901969 ≈ gf1(fixedPoint) ≈ gf1Ofpar(gf1_parVal)

f1 = x -> exp(-xpn1() * x^2)
f2 = x -> exp(-xpn2() * x^2)
f3 = x -> exp(-xpn3() * x^2)
g1 = x -> mapreduce((f, i)->f(i), *, (f1, f2, f3), x)
gfs1Prod = AxialProdField(gfs1 |> Tuple)
gfs1Prod_core, gfs1Prod_par = Quiqbox.unpackFunc(gfs1Prod)
pt = (1.1, 2.0, -3.0)
@test g1(pt) == gfs1Prod(pt) ≈ gfs1Prod_core(pt, obtain(gfs1Prod_par))

gf4 = GaussFunc(1.0)
gf4_val1 = 0.36787944117144233

for a_x in 0:3
    ang = (a_x, 0, 0)
    pgf = Quiqbox.PolyRadialFunc(gf4, ang)
    k = (0.1, -0.3, 0.5)
    pgf_core, pgf_par = Quiqbox.unpackFunc(pgf)
    @test pgf(k) == gf4(k|>norm) * prod(k .^ ang) ≈ pgf_core(k, obtain(pgf_par))
    @test pgf((-1., 0., 0.)) == (-1)^isodd(a_x) * gf4_val1
    @test pgf(( 0., 0., 0.)) == iszero(a_x)
    @test pgf(( 1., 0., 0.)) == gf4_val1
end

pgto1 = genGaussTypeOrb((0., 1., 0.), 2.5, (1, 1, 0))
pgto1Core, paramSet = Quiqbox.unpackFunc(pgto1.field)
Quiqbox.StashedField(pgto1Core, paramSet) isa Quiqbox.FloatingPolyGaussField

end