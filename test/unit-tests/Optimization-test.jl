using Test
using Quiqbox
using Quiqbox: formatTunableParams!, makeAbsLayerForXpnParams, compareParamBox, absMap
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

testαabsPars = function(ps0, ps1)
    bl = true
    map(ps0, ps1) do p0, p1
        if isOutSymEqual(p0, :α) && isOutSymEqual(p1, :α)
            bl1 = p0.data[] === p1.data[]
            bl1 || (@show bl1)
            bl2 = p0.map === p1.map.f
            bl2 || (@show bl2)
            bl3 = p1.map isa absMap
            bl3 || (@show bl3)
            bl4 = p0.canDiff == p1.canDiff
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
pars0_2, bs0_2 = makeAbsLayerForXpnParams(pars0_1, bs0_1)
parsAll0_2 = getParams(bs0_2)
@test testαabsPars(pars0_1, pars0_2)
@test testαabsPars(parsAll0_1, parsAll0_2)


# Floating basis set
nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]
nuc = ["H", "H"]

configs = [POconfig((maxStep=200, threshold=NaN)), 
           POconfig((maxStep=200, threshold=NaN, config=HFconfig((HF=:UHF,))))]

Eend = Float64[]
Ebegin = Float64[]

for c in configs, (i,j) in zip((1,2,7,8,9,10), (2,2,7,9,9,10))
    # 1->X₁, 2->X₂, 7->α₁, 8->α₂, 9->d₁, 10->d₂
    gf1 = GaussFunc(1.7, 0.8)
    gf2 = GaussFunc(0.45, 0.25)
    cens = genSpatialPoint.(nucCoords)
    bs1 = genBasisFunc.(cens, Ref((gf1, gf2)), normalizeGTO=true)
    pars1 = markParams!(bs1, true)

    Es1L, _, _ = optimizeParams!(pars1[i:j], bs1, nuc, nucCoords, c, printInfo=false)
    push!(Ebegin, Es1L[1])
    push!(Eend, Es1L[end])
end

@test all(Ebegin .> Eend)
compr2Arrays3((Eend_1to6=Eend[1:6], Eend_7toEnd=Eend[7:end]), 1e-5)


# Grid-based basis set
grid = GridBox(1, 3.0)
gf2 = GaussFunc(0.7, 1.0)
bs2 = genBasisFunc.(grid.point, Ref([gf2])) |> collect

pars2 = markParams!(bs2, true)[1:2]

local Es2L, ps2L, grads2L
@suppress_out begin
    Es2L, ps2L, grads2L = optimizeParams!(pars2, bs2, nuc, nucCoords, 
                                          POconfig((maxStep=200,)))
end

E_t2 = -1.16652582930629
# L₁, α₁
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
bs3 = (bs2_2 .+ genBasisFunc(fill(0.0, 3), gf3)) |> collect
pars3 = markParams!(bs3, true)[1:5]
local Es3L, ps3L, grads3L

@suppress_out begin
    Es3L, ps3L, grads3L = optimizeParams!(pars3, bs3, nuc, nucCoords, getCharge(nuc), 
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