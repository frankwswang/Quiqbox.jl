using Test
using Quiqbox
using Quiqbox: formatTunableParams!, makeAbsLayerForXpnParams, compareParamBox, Absolute
using LinearAlgebra: norm
using Suppressor: @suppress_out

include("../../test/test-functions/Shared.jl")

@testset "Optimization.jl" begin

errorThreshold = 1e-10

grid0 = GridBox(1, 3.0)
gf0 = GaussFunc(0.65, 1.2)
enableDiff!(gf0.xpn)
bs0 = genBasisFunc.(grid0.point, Ref([gf0])) |> collect
toggleDiff!.(bs0[1].center)
pars0 = markParams!(bs0, true)
bs0_0 = deepcopy(bs0)
pars0_0 = deepcopy(pars0)
bs0_1 = [bs0...]
pars0_1 = [pars0...]
arr0Ds = formatTunableParams!(pars0_1, bs0_1)
@test arr0Ds == map(x->fill(isDiffParam(x) ? x[] : x()), pars0_1)
@test mapreduce(hasEqual, *, bs0, bs0_0)
@test mapreduce(hasEqual, *, pars0, pars0_0)
testDetachedPars = function(ps0, ps1)
    bl = true
    map(ps0, ps1) do p0, p1
        if isDiffParam(p0) && isDiffParam(p1)
            bl1 = p0 === p1
            bl1 || (@show bl1)
            bl *= bl1
        else
            bl2 = p0() == p1[] == p1()
            bl2 || (@show bl2)
            bl3 = ( typeof(p1.map) == 
                    Quiqbox.DressedItself{Quiqbox.getFLevel(p0.map), typeof(p0.map)} )
            bl3 || (@show bl3)
            bl4 = outSymOf(p0) == outSymOf(p1)
            bl4 || (@show bl4)
            bl5 = isDiffParam(p0) == isDiffParam(p1)
            bl5 || (@show bl5)
            bl *= bl2*bl3*bl4*bl5
        end
    end
    bl
end
@test testDetachedPars(pars0, pars0_1)
parsAll0 = getParams(bs0)
parsAll0_1 = getParams(bs0_1)
@test testDetachedPars(parsAll0, parsAll0_1)
findDifferentiableParIdx(differ) = x->findfirst(y->compareParamBox(y, x), differ)!==nothing
parsToBeMutated0 = findall(findDifferentiableParIdx(pars0), parsAll0)
parsToBeMutated0_1 = findall(findDifferentiableParIdx(pars0_1), parsAll0_1)
@test parsToBeMutated0 == parsToBeMutated0_1

testαabsPars1 = function(ps0, ps1; sameDiff=true)
    bl = true
    map(ps0, ps1) do p0, p1
        if isOutSymEqual(p0, :α) && isOutSymEqual(p1, :α)
            bl1 = p0.data[] === p1.data[]
            bl1 || (@show bl1)
            bl2 = p0.map === p1.map.inner
            bl2 || (@show bl2)
            bl3 = p1.map isa Quiqbox.Layered{typeof(abs)}
            bl3 || (@show bl3)
            bl4 = if sameDiff
                p0.canDiff == p1.canDiff && p0.canDiff !== p1.canDiff
            else
                p1.canDiff[]
            end
            bl4 || (@show bl4)
            bl5 = p0.index == p1.index
            bl5 || (@show bl5)
            bl *= bl1*bl2*bl3*bl4*bl5
        else
            bl6 = p0 === p1
            bl6 || (@show bl6)
            bl *= bl6
        end
    end
    bl
end
pars0_2, bs0_2 = makeAbsLayerForXpnParams(pars0_1, bs0_1, tryJustFlipNegSign=false)
parsAll0_2 = getParams(bs0_2)
@test testαabsPars1(pars0_1, pars0_2)
@test testαabsPars1(parsAll0_1, parsAll0_2)
pars0_3, bs0_3 = makeAbsLayerForXpnParams(pars0_1, bs0_1, 
                                          forceDiffOn=true, tryJustFlipNegSign=false)
parsAll0_3 = getParams(bs0_3)
@test testαabsPars1(pars0_1, pars0_3, sameDiff=false)
@test testαabsPars1(parsAll0_1, parsAll0_3, sameDiff=false)


# Floating basis set
nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]
nuc = ["H", "H"]

configs = [(maxStep=100)->POconfig(;maxStep, threshold=NaN), 
           (maxStep=100)->POconfig(;maxStep, threshold=NaN, config=HFconfig((HF=:UHF,)))]

Es1Ls = Vector{Float64}[]
gradEnd = Float64[]

for config in configs, (i,j) in zip((1,2,7,8,9,10), (2,2,7,9,9,10))
    # 1->X₁, 2->X₂, 7->α₁, 8->α₂, 9->d₁, 10->d₂
    gf1 = GaussFunc(1.7, 0.8)
    gf2 = GaussFunc(0.45, 0.25)
    cens = genSpatialPoint.(nucCoords)
    bs1 = genBasisFunc.(cens, Ref((gf1, gf2)), normalizeGTO=true)
    pars1 = markParams!(bs1, true)
    c = i==10 ? config(1000) : config(100)
    Es1L, _, grads, _ = optimizeParams!(pars1[i:j], bs1, nuc, nucCoords, c, printInfo=false)
    push!(Es1Ls, Es1L)
    push!(gradEnd, norm(grads[:, end]))
end

@test all(all(Es1L[i]<=Es1L[i-1] for i in 2:lastindex(Es1L)) for Es1L in Es1Ls)
@test all(gradEnd .< 5e-5)
compr2Arrays3((Eend_1to6=last.(Es1Ls[1:6]), Eend_7toEnd=last.(Es1Ls[7:end])), 1e-5)


# Grid-based basis set
grid = GridBox(1, 3.0)
gf2 = GaussFunc(0.7, 1.0)
bs2 = genBasisFunc.(grid.point, Ref([gf2])) |> collect

pars2 = markParams!(bs2, true)[1:2]

local Es2L, ps2L, grads2L
@suppress_out begin
    Es2L, ps2L, grads2L = optimizeParams!(pars2, bs2, nuc, nucCoords, 
                                          POconfig((maxStep=50,)))
end

E_t2 = -1.6679941925321318
# L₁, α₁
par_t2  = [0.6798695445498076, 0.3635953878334846]
grad_t2 = [0.006074017206680493, 0.003461336621272404]
@test all(Es2L[i]<=Es2L[i-1] for i in 2:lastindex(Es2L))
@test isapprox(Es2L[end], E_t2, atol=errorThreshold)
@test isapprox(ps2L[:, end], par_t2, atol=errorThreshold)
@test isapprox(grads2L[:, end], grad_t2, atol=errorThreshold)


# BasisFuncMix basis set
gf2_2 = GaussFunc(0.7, 1.0)
grid2 = GridBox(1, 3.0)
bs2_2 = genBasisFunc.(grid2.point, Ref([gf2_2]))
gf3 = GaussFunc(0.5, 1.0)
bs3 = (bs2_2 .+ genBasisFunc(fill(0.0, 3), gf3)) |> collect
pars3 = markParams!(bs3, true)[1:5]
local Es3L, ps3L, grads3L

@suppress_out begin
    Es3L, ps3L, grads3L = optimizeParams!(pars3, bs3, nuc, nucCoords, getCharge(nuc), 
                                          POconfig((maxStep=50,)))
end

E_t3 = -1.7404923301470092
# L, α₁, α₂, d₁, d₂
par_t3  = [2.521964806623176, 0.2297027890247971, 0.5572444920809891, 0.7325043955992585, 
           1.2165608957203722]
grad_t3 = [0.013539198128510885, -0.009743010386111528, -0.0037981339100909353, 
           0.0006344290742911736, -0.00038199656694082275]
@test all(Es3L[i]<=Es3L[i-1] for i in 2:lastindex(Es3L))
@test isapprox(Es3L[end], E_t3, atol=errorThreshold)
@test isapprox(ps3L[:, end], par_t3, atol=errorThreshold)
@test isapprox(grads3L[:, end], grad_t3, atol=errorThreshold)

end