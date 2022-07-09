using Test
using Quiqbox
using LinearAlgebra: norm, diag

@testset "MatterByHF.jl tests" begin

# struct MatterByHF
get1spinHcore = function (C, Hc)
    r = 1 : size(C)[1]
    [ sum([C[m,i]*C[n,j]*Hc[m,n] for m=r, n=r]) for i=r, j=r ]
end

get1spin2eI = function (C, eeI)
    r = 1 : size(C)[1]
    [ sum([C[m,i]*C[n,j]*C[p,k]*C[q,l]*eeI[m,n,p,q] for m=r, n=r, p=r, q=r]) 
      for i=r, j=r, k=r, l=r ]
end

nuc1 = ["H", "H"]
nHalf = getCharge(nuc1) ÷ 2
nucCoords1 = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]
bs1 = genBasisFunc.(nucCoords1, ("STO-3G", "H") |> Ref) |> flatten
nbs1 = basisSize.(bs1) |> sum
basis1 = GTBasis(bs1)
Hc1 = coreH(basis1, nuc1, nucCoords1)
HFres1 = runHF(basis1, nuc1, nucCoords1, printInfo=false)
H2 = MatterByHF(HFres1)
C_RHF = HFres1.C[1]

@test all(nHalf .== H2.spin)
@test H2.Ehf == Quiqbox.getEᵀ(Hc1, basis1.eeI, H2.occuC, (H2.spin[1],))
@test H2.Hcore[1] == changeHbasis(Hc1, C_RHF)
@test isapprox.(H2.Hcore[1], get1spinHcore(C_RHF, Hc1), atol=1e-15) |> all
@test H2.eeI[1] == changeHbasis(basis1.eeI, C_RHF)
@test isapprox.(H2.eeI[1], get1spin2eI(C_RHF, basis1.eeI)) |> all
@test isapprox(H2.Ehf, 2(diag(H2.Hcore[1])[1:nHalf] |> sum) + 
                          sum( [(2H2.eeI[1][i,i,j,j] - H2.eeI[1][i,j,j,i]) 
                               for j in 1:nHalf, i in 1:nHalf] ), atol=1e-15)

nuc2 = ["H", "H", "O"]
nucCoords2 = [[-0.7,0.0,0.0], [0.6,0.0,0.0], [0.0, 0.0, 0.0]]
bs2 = genBasisFunc.(nucCoords2, [("STO-3G", i) for i in nuc2]) |> flatten
nbs2 = basisSize.(bs2) |> sum
basis2 = GTBasis(bs2)
Hc2 = coreH(basis2, nuc2, nucCoords2)
HFres2 = runHF(basis2, nuc2, nucCoords2, HFconfig((HF=:UHF,)), printInfo=false)
H2O = MatterByHF(HFres2)
nα, nβ = H2O.spin
C_UHF1, C_UHF2 = HFres2.C

Ehf_H2O = Quiqbox.getEᵀ(Hc2, basis2.eeI, H2O.occuC, H2O.spin)
@test isapprox(H2O.Ehf, Ehf_H2O, atol=1e-9)
@test H2O.Hcore == changeHbasis.(Ref(Hc2), HFres2.C)
@test isapprox.(H2O.Hcore[1], get1spinHcore(C_UHF1, Hc2), atol=1e-12) |> all
@test isapprox.(H2O.Hcore[2], get1spinHcore(C_UHF2, Hc2), atol=1e-12) |> all
@test H2O.eeI == changeHbasis.(Ref(basis2.eeI), HFres2.C)
@test isapprox.(H2O.eeI[1], get1spin2eI(C_UHF1, basis2.eeI)) |> all
@test isapprox.(H2O.eeI[2], get1spin2eI(C_UHF2, basis2.eeI)) |> all

Jαβ = [eeInteraction(i.func, i.func, j.func, j.func) for (i,j) in Iterators.product(H2O.occuOrbital...)]
EαandEβ = map(H2O.spin, H2O.Hcore, H2O.eeI) do n, Hc, eeI
    sum(diag(Hc)[1:n]) + 0.5*sum([(eeI[i,i,j,j] - eeI[i,j,j,i]) for j in 1:n, i in 1:n])
end |> sum
@test isapprox(H2O.Ehf, EαandEβ + sum([Jαβ[i,j] for i=1:nα, j=1:nβ]), atol=1e-9)
@test isapprox(Ehf_H2O, EαandEβ + sum([Jαβ[i,j] for i=1:nα, j=1:nβ]), atol=1e-12)


# function nnRepulsions
@test nnRepulsions(nuc2, nucCoords2) == 1*1/norm(nucCoords2[1] - nucCoords2[2]) + 
                                        1*8/norm(nucCoords2[1] - nucCoords2[3]) + 
                                        1*8/norm(nucCoords2[2] - nucCoords2[3])

end