using Test
using Quiqbox
using Quiqbox: overlap, overlaps
using LinearAlgebra: norm

@testset "Overlap-Based Features" begin

pf1 = genGaussTypeOrb((0.5, 0.3, 0.4), 0.9)
pf1c = genGaussTypeOrb((0.5, 0.3, 0.4), 0.9)
pfs1 = [pf1, pf1c]
bs1 = [CompositeOrb(pfs1, [0.6,  0.3]), CompositeOrb(pfs1, [0.9, -0.2])]

s1 = overlaps(bs1)
s1_t = [overlap(i, j) for i in bs1, j in bs1]
@test s1 ≈ s1_t

pf2 = genGaussTypeOrb((0.1, 0.2, -0.4), 1.2)
pfs2 = [pf1, pf2]
bs2 = [CompositeOrb(pfs2, [0.6,  0.3]), CompositeOrb(pfs2, [0.9, -0.2])]

s2 = overlaps(bs2)
s2_t = [overlap(i, j) for i in bs2, j in bs2]
@test s2 ≈ s2_t

pgf1 =  genGaussTypeOrb((0.1, -0.2, -0.3), 1.5, (1, 0, 0))
pgf1n = genGaussTypeOrb((0.1, -0.2, -0.3), 1.5, (1, 0, 0), renormalize=true)
pgf1_val1 = pgf1n((0., 0., 0.)) * sqrt(overlap(pgf1, pgf1))
@test pgf1((0., 0., 0.)) ≈ pgf1.body((-0.1, 0.2, 0.3)) ≈ pgf1_val1

s1 = overlap(pgf1, pgf1n)
@test overlap(pgf1n, pgf1) ≈ s1
@test overlap(pgf1, pgf1) ≈ s1^2
@test overlap(pgf1n, pgf1n) ≈ 1
pgf1n_c = deepcopy(pgf1n)
@test overlap(pgf1n, pgf1n_c) ≈ 1

cen1 = (1.1, 0.5, 1.1)
cons1 = [1.5, -0.3]
xpns1 = [1.2, 0.6]
cgf1 = genGaussTypeOrb(cen1, xpns1, cons1, (1, 0, 0))
stf1Core = Quiqbox.ReturnTyped(x->exp(-norm(x)), Float64)
stf1 = Quiqbox.EncodedField(stf1Core, Val(1))
sto1 = Quiqbox.PolyRadialFunc(stf1, (1, 1, 0))
stoBasis1 = Quiqbox.PrimitiveOrb((1.0, 2.0, 3.0), sto1; renormalize=false)
@test overlap(stoBasis1, stoBasis1) ≈ 4.7123889802878764

sto2 = Quiqbox.PolyRadialFunc(stf1, (2,))
stoBasis2 = Quiqbox.PrimitiveOrb((1.0,), sto2; renormalize=false)
overlap(stoBasis2, stoBasis2)

stoBasis1n = Quiqbox.PrimitiveOrb((1.0, 2.0, 3.0), sto1; renormalize=true)
stoBasis1n_c, stoBasis1n_s = Quiqbox.unpackFunc(stoBasis1n);
overlap(stoBasis1n, stoBasis1n) ≈ 1
s2 = sqrt(overlap(stoBasis1, stoBasis1)) # 2.170803763600724
@test overlap(stoBasis1n, stoBasis1) ≈ overlap(stoBasis1, stoBasis1n) ≈ s2
@test overlap(stoBasis1n, stoBasis1, lazyCompute=true) ≈ 
      overlap(stoBasis1, stoBasis1n, lazyCompute=true) ≈ 
      sqrt(overlap(stoBasis1, stoBasis1, lazyCompute=true)) ≈ s2

cen2 = (1.0, 1.5, 1.1)
xpns2 = [1.5, 0.6]
cons2 = [1.0,  0.8]
cgf2 = genGaussTypeOrb(cen2, xpns2, cons2, (1, 0, 0))
cgf2n1 = genGaussTypeOrb(cen2, xpns2, cons2, (1, 0, 0), innerRenormalize=true, 
                         outerRenormalize=false)
cgf2n2 = genGaussTypeOrb(cen2, xpns2, cons2, (1, 0, 0), innerRenormalize=false, 
                         outerRenormalize=true)
cgf2n3 = genGaussTypeOrb(cen2, xpns2, cons2, (1, 0, 0), innerRenormalize=true, 
                         outerRenormalize=true)
s3 = overlap(cgf2n2, cgf2)
@test overlap(cgf2, cgf2n2) ≈ s3
@test overlap(cgf2, cgf2) ≈ s3^2
@test overlap(cgf2n1, cgf2n2) ≈ overlap(cgf2n2, cgf2n1)
s4 = overlap(cgf2n1, cgf2n3)
@test overlap(cgf2n3, cgf2n1) ≈ s4
@test overlap(cgf2n1, cgf2n1) ≈ s4^2
@test !(overlap(cgf2n1, cgf2n1) ≈ 1)
@test overlap(cgf2n2, cgf2n2) ≈ 1
@test overlap(cgf2n3, cgf2n3) ≈ 1
@test !(overlap(cgf2n2, cgf2n3) ≈ 1)
Quiqbox.decomposeOrbData(cgf2n2|>genOrbitalData)

consH = [0.1543289673, 0.5353281423, 0.4446345422]
bfH = genGaussTypeOrb((0., 0., 0.), [3.425250914, 0.6239137298, 0.1688554040], 
                      consH, innerRenormalize=true)
bfH_t = genGaussTypeOrb((0., 0., 0.), [3.425250914, 0.6239137298, 0.1688554040], consH)
bfH_t2 = genGaussTypeOrb((0., 0., 0.), [3.425250914, 0.6239137298, 0.1688554040], consH, 
                         outerRenormalize=true)
bfLi1 = genGaussTypeOrb((1.4, 0., 0.), [16.11957475, 2.93620066, 0.7946504870], 
                        [0.1543289673, 0.5353281423, 0.4446345422], innerRenormalize=true)
bfLi2 = genGaussTypeOrb((1.4, 0., 0.), [0.6362897469, 0.1478600533, 0.0480886784], 
                        [-0.09996722919, 0.3995128261, 0.7001154689], innerRenormalize=true)
bfLi3 = genGaussTypeOrb((1.4, 0., 0.), [0.6362897469, 0.1478600533, 0.0480886784], 
                        [0.1559162750, 0.6076837186, 0.3919573931], (1, 0, 0), 
                        innerRenormalize=true)
bfLi4 = genGaussTypeOrb((1.4, 0., 0.), [0.6362897469, 0.1478600533, 0.0480886784], 
                        [0.1559162750, 0.6076837186, 0.3919573931], (0, 1, 0), 
                        innerRenormalize=true)
bfLi5 = genGaussTypeOrb((1.4, 0., 0.), [0.6362897469, 0.1478600533, 0.0480886784], 
                        [0.1559162750, 0.6076837186, 0.3919573931], (0, 0, 1), 
                        innerRenormalize=true)
bsLiH = [bfH, bfLi1, bfLi2, bfLi3, bfLi4, bfLi5]
@test overlaps(bsLiH) ≈ 
[1.0000000000699911 0.36853233891350523 0.5815110893922335 -0.48820809433363677 0.0 0.0; 
 0.36853233891350523 1.000000000086529 0.24113657387766615 0.0 0.0 0.0; 
 0.5815110893922335 0.24113657387766615 1.0000000000680496 0.0 0.0 0.0; 
 -0.48820809433363677 0.0 0.0 1.0000000000245315 0.0 0.0; 
 0.0 0.0 0.0 0.0 1.0000000000245315 0.0; 0.0 0.0 0.0 0.0 0.0 1.0000000000245315]

@test overlaps([bsLiH[1]]) ≈ [1;;]
@test !Quiqbox.isRenormalized(bsLiH[1])
@test all(Quiqbox.isRenormalized(o) for o in Quiqbox.splitOrb(bsLiH[1]))

@test overlap(bsLiH[1], bsLiH[1]) ≈ 1
@test overlap(bsLiH[1], bsLiH[1], lazyCompute=true) ≈ 1
@test overlap(bsLiH[1], deepcopy(bsLiH[1]), lazyCompute=true) ≈ 1

gfHs = Quiqbox.splitOrb(bfH_t)
s_gfHs = [0.3105569331128749 0.6834026444177144 0.8172189509285114; 
          0.6834026444177144 3.994781236834846 7.888686729849412; 
          0.8172189509285114 7.888686729849412 28.373137923911862]
@test overlaps(gfHs) ≈ s_gfHs
@test overlap(bfH_t2, bfH_t2) ≈ 1

bfHt = genGaussTypeOrb((0., 0., 0.), [3.425250914, 0.6239137298, 0.1688554040], 
                      [1.0, 1.0, 1.0], innerRenormalize=true)
bfLi1t = genGaussTypeOrb((1.4, 0., 0.), [16.11957475, 2.93620066, 0.7946504870], 
                        [1.0, 1.0, 1.0], innerRenormalize=true)
bs2 = [bfHt, bfLi1t]
@test overlaps(bs2[1].basis) .* (consH  * consH') ≈ 
[0.023817430147884476 0.05069055715296341 0.01889142761833038; 
 0.05069055715296341 0.286576219938369 0.17637125216187333; 
 0.01889142761833038 0.17637125216187333 0.19769987611740347]

end