using Test
using Quiqbox
using LinearAlgebra: dot

@testset "Interface.jl" begin

cen1 = (1.1, 0.5, 1.1)
cen2 = (1.0, 1.5, 1.1)

cons1 = [1.5, -0.3]
xpns1 = [1.2, 0.6]

xpns2 = [1.5, 0.6]
cons2 = [1.0,  0.8]

cgf1 = genGaussTypeOrb(cen1, xpns1, cons1, (1, 0, 0))
cgf2 = genGaussTypeOrb(cen2, xpns2, cons2, (1, 0, 0))
bfs1 = [cgf1, cgf2]

s1 = Quiqbox.overlaps(cgf1.basis)
ovlp1 = dot(cons1, s1, cons1)
@test ovlp1 ≈ 0.2844258928014478

cgf1n = genGaussTypeOrb(cen1, xpns1, cons1, (1, 0, 0), 
                        innerRenormalize=true, outerRenormalize=true)
cgf2n = genGaussTypeOrb(cen2, xpns2, cons2, (1, 0, 0), 
                        innerRenormalize=true, outerRenormalize=true)
ovlp1_2 = Quiqbox.overlap(cgf1, cgf1)
@test ovlp1_2 ≈ 0.2844258928014479

ovlp1_3 = Quiqbox.overlap(cgf1, cgf1, lazyCompute=Quiqbox.False())
@test ovlp1_2 ≈ ovlp1_3

s2 = [0.2844258928014478 0.2894349248354434; 0.2894349248354434 2.0052505884348175]
@test Quiqbox.overlap(cgf1, cgf2) ≈ s2[2]
@test Quiqbox.overlaps(bfs1) ≈ s2
@test Quiqbox.overlap(cgf1, cgf2) ≈ Quiqbox.overlap(cgf1, cgf2, lazyCompute=false)

mmCen = (1.0, 2.0, 3.0)
mmDeg = (1, 2, 3)
Quiqbox.multipoleMoment(mmCen, mmDeg, cgf1, cgf2)
@test Quiqbox.multipoleMoment(cen1, (0, 0, 0), cgf1, cgf2) == 
      Quiqbox.overlap(cgf1, cgf2)

cgf3 = genGaussTypeOrb(cen1, xpns1, cons1, (3, 0, 0))
@test Quiqbox.multipoleMoment(cen1, (2, 0, 0), cgf2, cgf1) == Quiqbox.overlap(cgf2, cgf3)
@test Quiqbox.multipoleMoment(cen1, (2, 0, 0), cgf1, cgf2) ≈ Quiqbox.overlap(cgf2, cgf3)
@test multipoleMoment((1.0, 0.0, 0.0), (2, 2, 2), cgf1, cgf1) == 
      multipoleMoment((1.0, 0.0, 0.0), (2, 2, 2), cgf1, cgf1, lazyCompute=false)

nucs1 = [:H, :Li]
coords1 = [(-0.7, 0., 0.), (0.7, 0., 0.)]

@test nucAttractions(nucs1, coords1, bfs1) + elecKinetics(bfs1) == 
      coreHamiltonian(nucs1, coords1, bfs1)

end