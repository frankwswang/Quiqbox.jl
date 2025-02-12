using Test
using Quiqbox
using LinearAlgebra

@testset "Spatial.jl" begin

xpn1 = CellParam(1.5, :xpn)
xpn2 = CellParam(2.5, :xpn)
xpn3 = CellParam(0.5, :xpn)

xpns = [xpn1, xpn2, xpn3]

gf1, gf2, gf3 = gfs1 = Quiqbox.GaussFunc.(xpns)

@test getParams(gf1)[] === xpn1
@test all(getParams(gfs1) .=== [xpn1, xpn2, xpn3])
@test gf1(0) == exp(0)

gf1_core, gf1_par = Quiqbox.unpackFunc(gf1)
fixedPoint = 1.1
gf1Ofpar = Base.Fix1(gf1_core, 1.1)
gf1_parVal = evalParamSource(gf1_par)
@test -0.1970338691521383 ≈ gf1(fixedPoint) * (-fixedPoint^2)

f1 = x -> exp(-xpn1() * x^2)
f2 = x -> exp(-xpn2() * x^2)
f3 = x -> exp(-xpn3() * x^2)
g1 = x -> mapreduce((f, i)->f(i), *, (f1, f2, f3), x)
gfs1Prod = Quiqbox.AxialProdFunc(gfs1)
pt = (1.1, 2.0, -3.0)
@test g1(pt) == gfs1Prod(pt)

gf4 = GaussFunc(1.0)
gf4_val1 = 0.36787944117144233

for a_x in 0:3
    ang = (a_x, 0, 0)
    pgf = Quiqbox.PolyRadialFunc(gf4, ang)
    k = rand(3)
    @test pgf(k) == gf4(k|>norm) * prod(k .^ ang)
    @test pgf((-1., 0., 0.)) == (-1)^isodd(a_x) * gf4_val1
    @test pgf(( 0., 0., 0.)) == iszero(a_x)
    @test pgf(( 1., 0., 0.)) == gf4_val1
end

end