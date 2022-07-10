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
    cens = genSpatialPoint.(nucCoords)
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
@test all(isapprox.(Eend[1:6], Eend[7:end], atol=100*errorThreshold2))


# Grid-based basis set
grid = GridBox(1, 3.0)
gf2 = GaussFunc(0.7, 1.0)
bs2 = genBasisFunc.(grid.box, Ref([gf2]))

pars2 = markParams!(bs2, true)[1:2]

local Es2L, ps2L, grads2L
@suppress_out begin
    Es2L, ps2L, grads2L = optimizeParams!(pars2, bs2, nuc, nucCoords, 
                                          POconfig((maxStep=200,)))
end

E_t2 = -1.1665258293062977
# L, α
par_t2  = [2.8465051230989435, 0.22550104532759083]
grad_t2 = [0.37522252486564855, 0.683095213465126]

@test Es2L[1] > Es2L[end]
@test isapprox(Es2L[end], E_t2, atol=errorThreshold2)
@test isapprox(ps2L[:, end], par_t2, atol=errorThreshold2)
@test isapprox(grads2L[:, end], grad_t2, atol=errorThreshold2)


# BasisFuncMix basis set
gf2_2 = GaussFunc(0.7, 1.0)
grid2 = GridBox(1, 3.0)
bs2_2 = genBasisFunc.(grid2.box, Ref([gf2_2]))
gf3 = GaussFunc(0.5, 1.0)
bs3 = bs2_2 .+ genBasisFunc(fill(0.0, 3), gf3)
pars3 = markParams!(bs3, true)[[1,5:end...]]
local Es3L, ps3L, grads3L

@suppress_out begin
    Es3L, ps3L, grads3L = optimizeParams!(pars3, bs3, nuc, nucCoords, 
                                          POconfig((maxStep=50,)))
end

E_t3 = -1.653859783670078
# L, α₁, α₂, d₁, d₂
par_t3  = [2.996646686997478, 0.6913223149667996, 0.4835057214802305, 0.996686357834139, 
           1.0033029163221774]
grad_t3 = [0.0595635921759659, 0.16518443157274584, 0.2853998439170006, 0.0666603115041208, 
           -0.0662207016487781]

@test Es3L[1] > Es3L[end]
@test isapprox(Es3L[end], E_t3, atol=errorThreshold2)
@test isapprox(ps3L[:, end], par_t3, atol=errorThreshold2)
@test isapprox(grads3L[:, end], grad_t3, atol=errorThreshold2)

end