using Quiqbox
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
    pars1 = uniqueParams!(bs1, filterMapping=true)

    local Es1L, pars1L, grads1L
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

pars2 = uniqueParams!(bs2, filterMapping=true)[[1,4]]

local Es2L, pars2L, grads2L
@suppress_out begin
    Es2L, pars2L, grads2L = optimizeParams!(pars2, bs2, nuc, nucCoords, 
                                            POconfig((maxStep=200,)))
end

E_t2 = -1.1665258292058682
# L, α
par_t2  = [2.8465051231251404, 0.2255010453287884]
grad_t2 = [0.3752225248706008, 0.6830952135402155]

@test isapprox(Es2L[end], E_t2, atol=errorThreshold2)
@test isapprox(pars2L[end, :], par_t2, atol=errorThreshold2)
@test isapprox(grads2L[end, :], grad_t2, atol=errorThreshold2)

end