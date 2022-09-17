using Test
using Quiqbox
using Quiqbox: Fγ, F0
using QuadGK: quadgk

include("../../../test/test-functions/Shared.jl")

@testset "Core.jl tests" begin

tolerance1 = 5e-17
tolerance2 = 5Quiqbox.getAtolVal(Float64)
perturbStep = rand(-1e-1:2e-3:1e-1)
fNumInt = (γ, u) -> quadgk(t -> t^(2γ)*exp(-u*t^2), 0, 1; order=25, rtol=tolerance1)[1]
rng = -20:(0.2+perturbStep):9
for γ in 0:24
    f = ifelse(γ==0, (_, u)->F0(u), Fγ)
    ints1 = [f(γ, 10.0^e) for e in rng]
    ints2 = [fNumInt(γ, 10.0^e) for e in rng]
    compr2Arrays3((Fγ=ints1, FγTest=ints2), tolerance2; 
                  additionalInfo="γ=$γ step=$(0.2+perturbStep) rng=$(rng)")
end

nuc = ["H", "F"]
nucCoords = [[-0.8664394281409157, 0.0, 0.0], [0.8664394281409157, 0.0, 0.0]]
b1 = genBasisFunc(nucCoords[1], GaussFunc(2.0, 1.0), "d", (true,))
b2 = genBasisFunc(nucCoords[2], "STO-3G", "F")
bfm1, bfm2 = b1 .+ b2[1:2]
bs_bf_bfs_bfm = [b1, bfm1, b2..., bfm2]

eeI = eeInteractions(bs_bf_bfs_bfm)
cH = coreH(bs_bf_bfs_bfm, nuc, nucCoords)

@test @isdefined eeI
@test @isdefined cH

# function getCompositeIntCore
tolerance3 = 1e-15
for i in Quiqbox.SubshellNames[2:4]
    bfs1 = genBasisFunc(rand(3), (rand(0.5:0.1:2.0, 2), rand(-0.5:0.1:0.5, 2)), "S")
    bfs2 = genBasisFunc(rand(3), (rand(0.5:0.1:2.0, 2), rand(-0.5:0.1:0.5, 2)), i)
    bs1 = [bfs1, bfs2]
    bs2 = [bfs1, bfs2...]
    bss = (bs1, bs2)

    S1, S2 = overlaps.(bss)
    compr2Arrays3((S1=S1, S2=S2), tolerance3, true)

    T1, T2 = eKinetics.(bss)
    compr2Arrays3((T1=T1, T2=T2), tolerance3, true)

    V1, V2 = neAttractions.(bss, Ref(nuc), Ref(nucCoords))
    compr2Arrays3((V1=V1, V2=V2), tolerance3, true)

    eeI1, eeI2 = eeInteractions.(bss)
    compr2Arrays3((eeI1=eeI1, eeI2=eeI2), tolerance3)
end

end