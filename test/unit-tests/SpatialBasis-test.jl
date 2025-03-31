using Test
using Quiqbox
using Quiqbox: getInnerOrb, markObj
using LinearAlgebra

@testset "SpatialBasis.jl" begin

gf1 = GaussFunc(1.1)
@test Quiqbox.SelectTrait{Quiqbox.ParameterizationStyle}()(gf1) == 
      Quiqbox.TypedParamFunc{Float64}()

gf1_3d = Quiqbox.AxialProdFunc((gf1, gf1, gf1))
@test Quiqbox.SelectTrait{Quiqbox.ParameterizationStyle}()(gf1_3d) == 
      Quiqbox.TypedParamFunc{Float64}()

gf1_3dCore, par_gf1, par_gf1_Pointer = Quiqbox.unpackFunc(gf1_3d)
@test par_gf1 isa Quiqbox.TypedSpanParamSet

cen1 = (0.0, 0.0, 0.0)
ijk1 = (0, 1, 1)
pgto1 = genGaussTypeOrb(cen1, 1.2, ijk1)
pgto1f = FrameworkOrb(pgto1)
@test (pgto1f |> getInnerOrb |> getInnerOrb) isa Quiqbox.PrimitiveOrbCore

@test Quiqbox.SelectTrait{Quiqbox.ParameterizationStyle}()(pgto1) == 
      Quiqbox.TypedParamFunc{Float64}()
pgto1core, par_pgto1, ptr_pgto1 = Quiqbox.unpackFunc(pgto1);
@test pgto1 isa Quiqbox.PrimGTO
@test pgto1core isa Quiqbox.EvalPrimGTO
par_pgto1Val = map(obtain, par_pgto1)

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

cgto1core, par_cgto1, _ = Quiqbox.unpackFunc(cgto1);
@test cgto1 isa Quiqbox.CompGTO
@test cgto1core isa Quiqbox.EvalCompGTO

ao1 = FrameworkOrb(pgto1)
@test ao1.core isa Quiqbox.EvalPrimGTO
@test ao1 isa Quiqbox.FPrimGTO

coord2 = (0.2, 1.1, 2.1)
ao2 = FrameworkOrb(cgto1)
@test ao2.core isa Quiqbox.EvalCompGTO
@test ao2 isa Quiqbox.FCompGTO
@test ao2(coord2) == cgto1(coord2)

xpns1 = [1.2, 2.2, 3.1]
pgf = genGaussTypeOrb(cen1, xpns1[1], ijk1)
@test !compareParamBox(pgf.center[1], pgf.center[2])
cons1 = [1.0, 0.2, 0.1]
cgto2 = genGaussTypeOrb(cen1, xpns1, cons1, ijk1)
@test cgto2 isa Quiqbox.CompGTO
cgto2core, par_cgto2, pointer_cgto2 = Quiqbox.unpackFunc(cgto2);
cgto2a = FrameworkOrb(cgto2)
typeof(cgto2core) == typeof(cgto2a.core)
par_cgto2 == cgto2a.param
@test cgto2a.core isa Quiqbox.EvalCompGTO
cgto2_pVal = (unit=[cen1..., xpns1...], grid=[cons1])
@test map(obtain, par_cgto2) == cgto2_pVal

compute_cgto = function (dr, xpn, con, ijk)
    prod(dr .^ ijk) * exp(-xpn*norm(dr)^2) * con
end
cgto2_val1 = compute_cgto.(Ref(coord2 .- cen1), xpns1, cons1, Ref(ijk1))
@test cgto2a.core(coord2, cgto2_pVal) == cgto2(coord2) ≈ sum(cgto2_val1)

pgto1_a = FrameworkOrb(pgto1)
coord1 = (0.0, 1.0, 2.0)
@test pgto1_a(coord1) == pgto1(coord1)

cgto3 = CompositeOrb([pgto1, cgto2, pgto2], [2.1, 2.2, 0.3])
@test cgto3 isa Quiqbox.CompGTO
cgto3core, par_cgto3, _ = Quiqbox.unpackFunc(cgto3);
cgto3a = FrameworkOrb(cgto3)

typeof(cgto3core) == typeof(cgto3a.core)
par_cgto3 == cgto3a.param
@test cgto3a.core isa Quiqbox.EvalCompGTO

cen2 = (1.1, 0.5, 1.1)
xpns2 = [1.2, 0.6]
cons2 = [1.5, -0.3]
ijk3 = (1, 0, 0)
cgf1 = genGaussTypeOrb(cen2, xpns2, cons2, ijk3)
cgf1s = Quiqbox.splitOrb(cgf1)
cgf1f = FrameworkOrb(cgf1)
bs1 = Quiqbox.splitOrb(cgf1)
bs1f = Quiqbox.splitOrb(cgf1f)
pgf1 = Quiqbox.viewOrb(cgf1, 1)
@test pgf1 isa Quiqbox.PrimitiveOrb
@test pgf1 === PrimitiveOrb(pgf1) === Quiqbox.viewOrb(pgf1, 1)
@test pgf1 === bs1[1]

pgf1f = FrameworkOrb(pgf1)
@test pgf1f === Quiqbox.viewOrb(pgf1f, 1)
pgf1i = getInnerOrb(pgf1f)
@test pgf1i isa Quiqbox.ScaledOrbital
pgf1i_2 = getInnerOrb(Quiqbox.viewOrb(cgf1f, 1))
pgf1i_3 = getInnerOrb(Quiqbox.viewOrb(bs1f[1], 1))
@test markObj(pgf1i) == markObj(pgf1i_2) == markObj(pgf1i_3)
pgf1c = getInnerOrb(pgf1i)
@test pgf1c isa Quiqbox.PrimitiveOrbCore
@test pgf1c === getInnerOrb(pgf1i_2) === getInnerOrb(pgf1i_3)

bf1 = genGaussTypeOrb((0., 0., 0.), 1.5)
bf2 = genGaussTypeOrb((0., 0., 0.), [1.5], [2.3])
@test bf2((1.1, 0.3, 0.4)) / bf1((1.1, 0.3, 0.4)) ≈ 2.3

@test !Quiqbox.isRenormalized(bf1)
Quiqbox.enforceRenormalize!(bf1)
@test Quiqbox.isRenormalized(bf1)
Quiqbox.preventRenormalize!(bf1)
@test !Quiqbox.isRenormalized(bf1)

cen3 = (1.0, 1.5, 1.1)
xpns3 = [1.2, 0.6]
cons3 = [1.5, -0.3]
cgf2 = genGaussTypeOrb(cen3, xpns3, cons3, (1, 0, 0))
cgf2f = FrameworkOrb(cgf2)
cgf2f1 = FrameworkOrb(cgf2f, 1)
cgf2f2 = FrameworkOrb(cgf2f, 2)
cgf2fComps = Quiqbox.splitOrb(cgf2f)
pgf1_2 = FrameworkOrb(cgf2.basis[1])
pgf2_2 = FrameworkOrb(cgf2.basis[2])
pgfs_2_cores = [pgf1_2.core, pgf2_2.core]
cgf2fCores = getfield.(cgf2fComps, :core)
@test cgf2fComps[1].core == cgf2f1.core
@test cgf2fComps[1].param == cgf2f1.param
@test cgf2fComps[1].pointer.tag == cgf2f1.pointer.tag
@test markObj(cgf2fComps[1].pointer.scope) == markObj(cgf2f1.pointer.scope)
@test markObj(cgf2fComps) == markObj([cgf2f1, cgf2f2])
@test typeof.(cgf2fCores) == typeof.(pgfs_2_cores)

mk1 = markObj(cgf2fCores[1].f.select)
mk2 = markObj(pgfs_2_cores[1].f.select)
@test mk1 == mk2
@test markObj(cgf2fCores[1]) == markObj(pgfs_2_cores[1])
@test markObj(cgf2fCores[1].f) == markObj(pgfs_2_cores[1].f)
@test markObj(cgf2fCores[1].f.apply) == markObj(pgfs_2_cores[1].f.apply)
@test typeof(markObj(cgf2fCores[1].f.apply)) == Quiqbox.FieldMarker{:PairCombine, 3}
typeof(cgf2fCores[1].f.select) == typeof(pgfs_2_cores[1].f.select)

mkr1 = markObj(cgf2fComps[1]);
mkr1c = markObj(cgf2fComps[1].core);
mkr1p = markObj(cgf2fComps[1].pointer);
@test !Quiqbox.compareObj(cgf2fComps[1], pgf1_2)
@test Quiqbox.compareObj(cgf2fCores[1], pgfs_2_cores[1])

ptr1 = first(cgf2fCores[1].f.select)
ptr2 = last(pgfs_2_cores[1].f.select)
@test mapreduce(==, *, ptr1.scope, ptr2.scope)

end