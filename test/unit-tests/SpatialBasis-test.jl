using Test
using Quiqbox
using Quiqbox: markObj
using LinearAlgebra

@testset "SpatialBasis.jl" begin

gf1 = GaussFunc(1.1)
gf1_3d = Quiqbox.ProductField((gf1, gf1, gf1))

gf1_3dCore, par_gf1 = Quiqbox.unpackFunc(gf1_3d)
@test par_gf1 isa Quiqbox.TypedSpanParamSet

cen1 = (0.0, 0.0, 0.0)
ijk1 = (0, 1, 1)
pgto1 = genGaussTypeOrb(cen1, 1.2, ijk1)
pgto1Data = genOrbitalData(pgto1)
@test pgto1Data isa Quiqbox.PrimOrbData
@test pgto1Data isa Quiqbox.PrimGTOData

pgto1core, par_pgto1 = Quiqbox.unpackFunc(pgto1);
@test pgto1 isa Quiqbox.PrimGTO
@test pgto1core isa Quiqbox.ParamBindFunc
par_pgto1Val = obtain(par_pgto1)
@test pgto1core((1.1, 1.0, -0.3), par_pgto1Val) == pgto1((1.1, 1.0, -0.3))

ijk2 = (1, 1, 2)
pgto2 = genGaussTypeOrb(cen1, Quiqbox.flatVectorize(par_pgto1)[end], ijk2)
cgto1 = CompositeOrb([pgto1, pgto2], [1.1, 1.2])
@test cgto1((1., 0., 0.1)) |> iszero
@test cgto1((0., 0., 1.0)) |> iszero
pt1 = (1.2, 1.1, -1.0)
@test cgto1(pt1) == 1.1pgto1(pt1) + 1.2pgto2(pt1) == 
                    1.1pgto1.body(pt1 .- cen1) + 1.2pgto2.body(pt1 .- cen1)
cgto1n = CompositeOrb([pgto1, pgto2], [1.1, 1.2], renormalize=true)
@test cgto1n(pt1) != cgto1(pt1)

cgto1core, par_cgto1 = Quiqbox.unpackFunc(cgto1);
@test cgto1 isa Quiqbox.CompGTO
@test cgto1core isa Quiqbox.ParamBindFunc

coord2 = (0.2, 1.1, 2.1)
cgto1Data = genOrbitalData(cgto1)
@test cgto1Data isa Quiqbox.CompOrbData

xpns1 = [1.2, 2.2, 3.1]
pgf = genGaussTypeOrb(cen1, xpns1[1], ijk1)
@test !compareParamBox(pgf.center[1], pgf.center[2])
cons1 = [1.0, 0.2, 0.1]
cgto2 = genGaussTypeOrb(cen1, xpns1, cons1, ijk1)
@test cgto2 isa Quiqbox.CompGTO
cgto2core, par_cgto2 = Quiqbox.unpackFunc(cgto2);

compute_cgto = function (dr, xpn, con, ijk)
    prod(dr .^ ijk) * exp(-xpn*norm(dr)^2) * con
end
cgto2_val1 = compute_cgto.(Ref(coord2 .- cen1), xpns1, cons1, Ref(ijk1))
@test cgto2(coord2) ≈ sum(cgto2_val1)

cgto3 = CompositeOrb([pgto1, cgto2, pgto2], [2.1, 2.2, 0.3])
@test cgto3 isa Quiqbox.CompGTO
cgto3core, par_cgto3 = Quiqbox.unpackFunc(cgto3);

cen2 = (1.1, 0.5, 1.1)
xpns2 = [1.2, 0.6]
cons2 = [1.5, -0.3]
ijk3 = (1, 0, 0)
cgf1 = genGaussTypeOrb(cen2, xpns2, cons2, ijk3)
cgf1s = Quiqbox.splitOrb(cgf1)
pgf1 = Quiqbox.viewOrb(cgf1, 1)
@test pgf1 isa Quiqbox.PrimitiveOrb
@test pgf1 === Quiqbox.viewOrb(pgf1, 1)
pgf1_rc = PrimitiveOrb(pgf1)
@test pgf1.center === pgf1_rc.center && pgf1.body === pgf1_rc.body && 
      pgf1.renormalize === pgf1_rc.renormalize
@test pgf1 != pgf1_rc

bf1 = genGaussTypeOrb((0., 0., 0.), 1.5)
bf2 = genGaussTypeOrb((0., 0., 0.), [1.5], [2.3])
@test bf2((1.1, 0.3, 0.4)) / bf1((1.1, 0.3, 0.4)) ≈ 2.3

@test !Quiqbox.isRenormalized(bf1)
Quiqbox.enforceRenormalize!(bf1)
@test Quiqbox.isRenormalized(bf1)
Quiqbox.disableRenormalize!(bf1)
@test !Quiqbox.isRenormalized(bf1)

stf1Core = Quiqbox.TypedReturn(x->exp(-norm(x)), Float64)
stf1 = Quiqbox.EncodedField(stf1Core, Val(1))
sto1 = Quiqbox.PolyRadialFunc(stf1, (1, 1, 0))
sto1((1.1, 0.2, 1.2))

# Complex orbital testing
cen3 = (1.0, 2.0, 3.0)
xpns3 = [2.0, 1.0]
complexWeights = [-1.5im, 0.7 + 0.2im]
ccgto1 = genGaussTypeOrb(cen3, xpns3, complexWeights)
@test ccgto1(cen3) == sum(complexWeights)
coord1 = (0.2, 0.5, 1.1)
ccgto1_pos = Quiqbox.splitOrb(ccgto1)
@test ccgto1(coord1) ≈ dot((Ref(coord1) .|> ccgto1.basis), complexWeights)
ccgto1_2 = CompositeOrb(ccgto1_pos, complexWeights)
coord2 = (0.0, 1.0, 0.0)
@test ccgto1_2(coord2) == ccgto1(coord2)
ccgto2 = CompositeOrb([ccgto1, ccgto1_2], ComplexF64[0.4, 0.6])
@test ccgto1(coord2) ≈ ccgto2(coord2)

end