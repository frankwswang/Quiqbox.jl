using Test
using Quiqbox
using Suppressor: @suppress_out

@testset "Optimization.jl" begin

errorThreshold1 = 1e-10
errorThreshold2 = 1e-6

# Floating basis set
nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]
nuc = ["H", "H"]

configs = [POconfig((maxStep=200, error=NaN)), 
           POconfig((maxStep=200, error=NaN, config=HFconfig((HF=:UHF,))))]

Eend = Float64[]
Ebegin = Float64[]

for c in configs, (i,j) in zip((1,2,7,8,9,10), (2,2,7,9,9,10))
    # 1->X₁, 2->X₂, 7->α₁, 8->α₂, 9->d₁, 10->d₂
    gf1 = GaussFunc(1.7, 0.8)
    gf2 = GaussFunc(0.45, 0.25)
    cens = makeCenter.(nucCoords)
    bs1 = genBasisFunc.(cens, Ref((gf1, gf2)), normalizeGTO=true)
    pars1 = markParams!(bs1, true)

    local Es1L
    @suppress_out begin
        Es1L, _, _ = optimizeParams!(pars1[i:j], bs1, nuc, nucCoords, c, printInfo=false)
        push!(Ebegin, Es1L[1])
        push!(Eend, Es1L[end])
    end
end

@test all(Ebegin .> Eend)
@test all(Eend[1:6] .<= Eend[7:end])
@test all(isapprox.(Eend[1:6], Eend[7:end], atol=errorThreshold2))


# Grid-based basis set
grid = GridBox(1, 3.0)
gf2 = GaussFunc(0.7,1)
bs2 = genBasisFunc.(grid.box, Ref([gf2]))

pars2 = markParams!(bs2, true)[1:2]

local Es2L, ps2L, grads2L
@suppress_out begin
    Es2L, ps2L, grads2L = optimizeParams!(pars2, bs2, nuc, nucCoords, 
                                            POconfig((maxStep=200,)))
end

E_t2 = -1.1665258292058682
# L, α
par_t2  = [2.8465051231251404, 0.2255010453287884]
grad_t2 = [0.3752225248706008, 0.6830952135402155]

@test Es2L[end] < Es2L[1]
@test isapprox(Es2L[end], E_t2, atol=errorThreshold2)
@test isapprox(ps2L[end, :], par_t2, atol=errorThreshold2)
@test isapprox(grads2L[end, :], grad_t2, atol=errorThreshold2)


# BasisFuncMix basis set
gf2_2 = GaussFunc(0.7,1)
bs2_2 = genBasisFunc.(grid.box, Ref([gf2_2]))
gf3 = GaussFunc(0.5,1)
bs3 = bs2_2 .+ genBasisFunc([0,0,0], gf3)
pars3 = markParams!(bs3, true)[[1,5:end...]]
local Es3L, ps3L, grads3L

@suppress_out begin
    Es3L, ps3L, grads3L = optimizeParams!(pars3, bs3, nuc, nucCoords, 
                                          POconfig((maxStep=50,)))
end

E_t3 = -1.660135125602929
# L, α₁, α₂, d₁, d₂
par_t3  = [2.8438717380095864, 0.6912590580923045, 0.48913645109117465, 
           0.9974149356990022, 1.0025785319279148]
grad_t3 = [0.040670718930460946, 0.16979735243389227, 0.18942366614583084, 
           0.05227968716424935, -0.052010430256302635]

@test Es3L[end] < Es3L[1]
@test isapprox(Es3L[end], E_t3, atol=errorThreshold2)
@test isapprox(ps3L[end, :], par_t3, atol=errorThreshold2)
@test isapprox(grads3L[end, :], grad_t3, atol=errorThreshold2)

end