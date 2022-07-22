using Test
using Quiqbox
using Suppressor: @suppress_out

@testset "Optimization.jl" begin

errorThreshold = 1e-10

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
@test all(Eend[2] >= Eend[8])
@test all(Eend[[3:6...]] .<= Eend[[9:end...]])
@test all(isapprox.(Eend[1:6], Eend[7:end], atol=1e-5))


# Grid-based basis set
grid = GridBox(1, 3.0)
gf2 = GaussFunc(0.7, 1.0)
bs2 = genBasisFunc.(grid.point, Ref([gf2]))

pars2 = markParams!(bs2, true)[1:2]

local Es2L, ps2L, grads2L
@suppress_out begin
    Es2L, ps2L, grads2L = optimizeParams!(pars2, bs2, nuc, nucCoords, 
                                          POconfig((maxStep=200,)))
end

E_t2 = -1.16652582930629
# L, α
par_t2  = [2.846505123098946, 0.225501045327590]
grad_t2 = [0.375222524865646, 0.683095213465142]
@test Es2L[1] > Es2L[end]
@test isapprox(Es2L[end], E_t2, atol=errorThreshold)
@test isapprox(ps2L[:, end], par_t2, atol=errorThreshold)
@test isapprox(grads2L[:, end], grad_t2, atol=errorThreshold)


# BasisFuncMix basis set
gf2_2 = GaussFunc(0.7, 1.0)
grid2 = GridBox(1, 3.0)
bs2_2 = genBasisFunc.(grid2.point, Ref([gf2_2]))
gf3 = GaussFunc(0.5, 1.0)
bs3 = bs2_2 .+ genBasisFunc(fill(0.0, 3), gf3)
pars3 = markParams!(bs3, true)[[1,5:end...]]
local Es3L, ps3L, grads3L

@suppress_out begin
    Es3L, ps3L, grads3L = optimizeParams!(pars3, bs3, nuc, nucCoords, 
                                          POconfig((maxStep=50,)))
end

E_t3 = -1.653859783670083
# L, α₁, α₂, d₁, d₂
par_t3  = [ 2.996646686997478,  0.691322314966799,  0.483505721480230,  0.996686357834139, 
            1.003302916322178]
grad_t3 = [ 0.059563592175966,  0.165184431572721,  0.285399843917005,  0.066660311504127, 
           -0.066220701648778]
@test Es3L[1] > Es3L[end]
@test isapprox(Es3L[end], E_t3, atol=errorThreshold)
@test isapprox(ps3L[:, end], par_t3, atol=errorThreshold)
@test isapprox(grads3L[:, end], grad_t3, atol=errorThreshold)

end