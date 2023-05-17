using Test
using Quiqbox
using LinearAlgebra: norm, diag

isdefined(Main, :SharedTestFunctions) || include("../../test/test-functions/Shared.jl")

@testset "MatterByHF.jl tests" begin

# struct MatterByHF and function changeHbasis
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
bs1 = genBasisFunc.(nucCoords1, "STO-3G", "H") |> flatten
nbs1 = orbitalNumOf.(bs1) |> sum
basis1 = GTBasis(bs1)
Hc1 = coreH(basis1, nuc1, nucCoords1)
HFres1 = runHF(basis1, nuc1, nucCoords1, printInfo=false)
H2 = MatterByHF(HFres1)
C_RHF = HFres1.C[1]
C_H2 = hcat(H2.occuC[1], H2.unocC[1])

t1 = 2e-15
@test C_H2 == C_RHF
@test all(nHalf .== H2.Ns)
@test hasEqual(genCanOrbitals(HFres1), 
               (vcat(collect.(H2.occuOrbital)...), vcat(collect.(H2.unocOrbital)...)))
@test H2.Ehf == Quiqbox.getEhf(Hc1, basis1.eeI, H2.occuC, (H2.Ns[1],))
@test H2.coreHsameSpin[1] == changeHbasis(Hc1, C_RHF)
@test compr2Arrays3((H2_cH1=H2.coreHsameSpin[1], 
                     H2_cH2=get1spinHcore(C_RHF, Hc1)), t1)
@test H2.eeIsameSpin[1] == changeHbasis(basis1.eeI, C_RHF)
@test compr2Arrays3((H2_eeI1=H2.eeIsameSpin[1], 
                     H2_eeI2=get1spin2eI(C_RHF, basis1.eeI)), t1)
@test isapprox(H2.Ehf, 2(diag(H2.coreHsameSpin[1])[1:nHalf] |> sum) + 
                          sum( [(2H2.eeIsameSpin[1][i,i,j,j] - H2.eeIsameSpin[1][i,j,j,i]) 
                               for j in 1:nHalf, i in 1:nHalf] ), atol=t1)
@test compr2Arrays3((H2_eeIds=H2.eeIdiffSpin, 
                     Jαβ=Quiqbox.getJᵅᵝ(H2.basis.eeI, (C_H2, C_H2))), t1)
H2Ehf = Quiqbox.getEhf(H2.coreHsameSpin, H2.eeIsameSpin, Quiqbox.splitSpins(Val(1), H2.Ns))
@test isapprox(H2.Ehf, H2Ehf, atol=t1)

nuc2 = ["H", "H", "O"]
nucCoords2 = [[-0.7,0.0,0.0], [0.6,0.0,0.0], [0.0, 0.0, 0.0]]
bs2 = genBasisFunc.(nucCoords2, "STO-3G", nuc2) |> flatten
nbs2 = orbitalNumOf.(bs2) |> sum
basis2 = GTBasis(bs2)
Hc2 = coreH(basis2, nuc2, nucCoords2)
HFres2 = runHF(basis2, nuc2, nucCoords2, HFconfig((HF=:UHF,)), printInfo=false)
H2O = MatterByHF(HFres2)
nα, nβ = H2O.Ns
C_UHF1, C_UHF2 = HFres2.C
Cα_H2O = hcat(H2O.occuC[1], H2O.unocC[1])
Cβ_H2O = hcat(H2O.occuC[2], H2O.unocC[2])
MOα_H2O = (H2O.occuOrbital[1]..., H2O.unocOrbital[1]...)
MOβ_H2O = (H2O.occuOrbital[2]..., H2O.unocOrbital[2]...)
Jαβ = [ eeInteraction(i.orbital, i.orbital, j.orbital, j.orbital) 
        for (i,j) in Iterators.product(MOα_H2O, MOβ_H2O) ]

t2 = 1e-12
@test C_UHF1 == Cα_H2O
@test C_UHF2 == Cβ_H2O
@test hasEqual(genCanOrbitals(HFres2), 
               (vcat(collect.(H2O.occuOrbital)...), vcat(collect.(H2O.unocOrbital)...)))
@test H2O.Ehf == Quiqbox.getEhf(Hc2, basis2.eeI, H2O.occuC, H2O.Ns)
@show H2O.Ehf
@test H2O.coreHsameSpin == changeHbasis.(Ref(Hc2), HFres2.C)
@test compr2Arrays3((H2O_cHα1=H2O.coreHsameSpin[1], 
                     H2O_cHα2=get1spinHcore(C_UHF1, Hc2)), t2)
@test compr2Arrays3((H2O_cHβ1=H2O.coreHsameSpin[2], 
                     H2O_cHβ2=get1spinHcore(C_UHF2, Hc2)), t2)
@test H2O.eeIsameSpin == changeHbasis.(Ref(basis2.eeI), HFres2.C)
@test compr2Arrays3((H2O_eeIα1=H2O.eeIsameSpin[1], 
                     H2O_eeIα2=get1spin2eI(C_UHF1, basis2.eeI)), t2)
@test compr2Arrays3((H2O_eeIβ1=H2O.eeIsameSpin[2], 
                     H2O_eeIβ2=get1spin2eI(C_UHF2, basis2.eeI)), t2)
@test compr2Arrays3((H2O_eeIds=H2O.eeIdiffSpin, Jαβ=Jαβ), t2)

ids = 1:length(H2O.occuOrbital[1])
Jαβ_occu = Jαβ[ids, ids]
EαandEβ = map(H2O.Ns, H2O.coreHsameSpin, H2O.eeIsameSpin) do n, Hc, eeI
    sum(diag(Hc)[1:n]) + 0.5*sum([(eeI[i,i,j,j] - eeI[i,j,j,i]) for j in 1:n, i in 1:n])
end |> sum
@test isapprox(H2O.Ehf, EαandEβ + sum([Jαβ_occu[i,j] for i=1:nα, j=1:nβ]), atol=t2)
H2OEhf = Quiqbox.getEhf(H2O.coreHsameSpin, H2O.eeIsameSpin, H2O.eeIdiffSpin, Quiqbox.splitSpins(Val(2), H2O.Ns))
@test isapprox(H2O.Ehf, H2OEhf, atol=t2)


# function nnRepulsions
@test nnRepulsions(nuc2, nucCoords2) == 1*1/norm(nucCoords2[1] - nucCoords2[2]) + 
                                        1*8/norm(nucCoords2[1] - nucCoords2[3]) + 
                                        1*8/norm(nucCoords2[2] - nucCoords2[3])

end