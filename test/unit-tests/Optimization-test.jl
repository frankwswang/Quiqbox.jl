using Test
using Quiqbox
using Quiqbox: formatTunableParams!, makeAbsLayerForXpnParams, compareParamBox, Absolute, 
               initializeOFconfig!
using LinearAlgebra: norm
using Suppressor: @capture_out
using Optim

isdefined(Main, :SharedTestFunctions) || include("../../test/test-functions/Shared.jl")

@testset "Optimization.jl" begin

errorThreshold1 = 1e-8
errorThreshold2 = 1e-12
errorThreshold3 = 1e-5
maxStep = 25


# formatTunableParams!
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
    map(enumerate(ps0), ps1) do (i, p0), p1
        if isDiffParam(p0) && isDiffParam(p1)
            bl1 = p0 === p1
            bl1 || (@show i bl1)
            bl *= bl1
        else
            bl2 = p0() == p1[] == p1()
            bl2 || (@show (i, bl2) p0() p1[] p1())
            bl3 = ( typeof(p1.map) == 
                    Quiqbox.DI{typeof(p0.map)} )
            bl3 || (@show (i, bl3) typeof(p1.map) typeof(p0.map))
            bl4 = outSymOf(p0) == outSymOf(p1)
            bl4 || (@show (i, bl4))
            bl5 = isDiffParam(p0) == isDiffParam(p1)
            bl5 || (@show (i, bl5) isDiffParam(p0) isDiffParam(p1))
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


# makeAbsLayerForXpnParams
testαabsPars1 = function(ps0, ps1; sameDiff=true)
    bl = true
    map(ps0, ps1) do p0, p1
        if isOutSymEqual(p0, :α) && isOutSymEqual(p1, :α)
            bl1 = p0.data[] === p1.data[]
            bl1 || (@show bl1)
            bl2 = p0.map === p1.map.inner
            bl2 || (@show bl2)
            bl3 = p1.map isa Quiqbox.Labs
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
@test markUnique(getproperty.(pars0_1, :canDiff))[begin] == 
      markUnique(getproperty.(pars0_2, :canDiff))[begin] == 
      markUnique(getproperty.(pars0_3, :canDiff))[begin]
parsAll0_3 = getParams(bs0_3)
@test testαabsPars1(pars0_1, pars0_3, sameDiff=false)
@test testαabsPars1(parsAll0_1, parsAll0_3, sameDiff=false)
testαabsPars2 = function(ps0, ps1; onlyNegα=false)
    bl = true
    map(ps0, ps1) do p0, p1
        if isOutSymEqual(p0, :α) && isOutSymEqual(p1, :α) && (onlyNegα ? p0() < 0 : true)
            if Quiqbox.getFLevel(p0.map) == 0
                bl1 = fill(abs(p0.data[][begin][])) == p1.data[][begin]
                bl2 = p1.data[][end] == p1.data[][end]
                bl3 = p1.map === itself
            else
                bl1 = p0.data[] === p1.data[]
                bl2 = p0.map === p1.map.inner
                bl3 = p1.map isa Quiqbox.Labs
            end
            bl1 || (@show bl1)
            bl2 || (@show bl2)
            bl3 || (@show bl3)
            bl4 = p1.canDiff[]
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
pars0_4, bs0_4 = makeAbsLayerForXpnParams(pars0_1, bs0_1, tryJustFlipNegSign=true)
parsAll0_4 = getParams(bs0_4)
@test testαabsPars2(pars0_1, pars0_4)
@test testαabsPars2(parsAll0_1, parsAll0_4)
pars0_5, bs0_5 = makeAbsLayerForXpnParams(pars0_1, bs0_1, true)
parsAll0_5 = getParams(bs0_5)
@test all(pars0_5 .=== pars0_1)
@test all(parsAll0_5 .=== parsAll0_1)
pars0_1[2][] *= -1
pars0_6, bs0_6 = makeAbsLayerForXpnParams(pars0_1, bs0_1, true, tryJustFlipNegSign=true)
parsAll0_6 = getParams(bs0_6)
@test all(pars0_4 .== pars0_6)
@test all(parsAll0_4 .=== parsAll0_6)
pars0_1[2][] *= -1


nuc = ["H", "H"]
nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]
Ne = getCharge(nuc)
saveTrace = (true, false, true, false)

# Floating basis set
cfgs = [(maxStep, C0)->POconfig(;maxStep, config=HFconfig(;C0), 
                                threshold=(0.1errorThreshold1,errorThreshold3), saveTrace), 
        (maxStep, C0)->POconfig(;maxStep, config=HFconfig(;HF=:UHF, C0), 
                                threshold=(0.1errorThreshold1,errorThreshold3), saveTrace)]

Es1Ls = Vector{Float64}[]
gradEnd = Float64[]

for (m, config) in enumerate(cfgs), (i,j) in zip((1,2,7,8,9,10), (2,2,7,9,9,10))
    # 1->X₁, 2->X₂, 7->α₁, 8->α₂, 9->d₁, 10->d₂
    gf1 = GaussFunc(1.7, 0.8)
    gf2 = GaussFunc(0.45, 0.25)
    cens = genSpatialPoint.(nucCoords)
    bs1 = genBasisFunc.(cens, Ref((gf1, gf2)), normalizeGTO=true)
    siz = orbitalNumOf.(bs1) |> sum
    pars1 = markParams!(bs1, true)
    addiCfgs = (i==10 ? 2000 : 100, 
                if iseven(i)
                    (m==1 ? (zeros(siz, siz),) : (zeros(siz, siz), zeros(siz, siz)))
                else
                    :SAD
                end
                )
    c = config(addiCfgs...)
    _, Es1L, grads = optimizeParams!(pars1[i:j], bs1, nuc, nucCoords, c, printInfo=false)
    push!(Es1Ls, Es1L)
    push!(gradEnd, norm(grads[end]))
end

@test all(all(Es1L[i]<=Es1L[i-1] || isapprox(Es1L[i], Es1L[i-1], atol=errorThreshold2)
              for i in 2:lastindex(Es1L)) for Es1L in Es1Ls)
@test all(abs.(gradEnd) .< 5e-5)
@test compr2Arrays3((Eend_1to6=last.(Es1Ls[1:6]), Eend_7toEnd=last.(Es1Ls[7:end])), 
                    errorThreshold1)


# Grid-based basis set
saveTrace2 = (true, true, true, false)
## default Line-search GD optimizer
po1 = POconfig((maxStep=maxStep, target=-10.0, threshold=1e-10, saveTrace=saveTrace2))
## vanilla GD optimizer
po2 = POconfig(;maxStep=200, target=-10.0, threshold=(1e-10, 1e-10), 
               optimizer=GDconfig(itself, 0.001, stepBound=(0.0, 2.0)), 
               saveTrace=saveTrace2)
pos = (po1, po2)
E_t2s = (-1.6667086781377531, -1.1665258293062974)
## L₁, α₁
par_t2s  = ([0.8121379463625688,  0.37162490725674535], 
            [2.8465051230989435,  0.22550104532759038])
grad_t2s = ([0.0156608276525374, -0.02651677671445425], 
            [0.3752225248656483,  0.68309521346509120])

for ((i, po), E_t2, par_t2, grad_t2) in zip(enumerate(pos), E_t2s, par_t2s, grad_t2s)
    grid = GridBox(1, 3.0)
    gf2 = GaussFunc(0.7, 1.0)
    bs2 = genBasisFunc.(grid.point, Ref([gf2])) |> collect

    pars2 = markParams!(bs2, true)[1:2]

    local Es2L, ps2L, grads2L
    output = @capture_out begin
        _, Es2L, ps2L, grads2L = optimizeParams!(pars2, bs2, nuc, nucCoords, po)
    end

    Es2Ldiffs = [Es2L[i]-Es2L[i-1] for i in 2:lastindex(Es2L)]

    test2bl1 = all(i<=0 || isapprox(i, 0, atol=errorThreshold2) for i in Es2Ldiffs)
    test2bl2 = isapprox(Es2L[end], E_t2, atol=errorThreshold1)
    test2bl3 = isapprox(ps2L[end], par_t2, atol=errorThreshold1)
    test2bl4 = isapprox(grads2L[end], grad_t2, atol=errorThreshold1)

    test2bl1 || (@show test2bl1 Es2Ldiffs)
    @test test2bl1
    test2bl2 || (@show test2bl2 Es2L[end])
    @test test2bl2
    test2bl3 || (@show test2bl3 ps2L[end])
    @test test2bl3
    test2bl3 || (@show test2bl4 grads2L[end])
    @test test2bl4

    all([test2bl1, test2bl2, test2bl3, test2bl4]) || 
    println("Grid-based basis set $i process:\n", output)
end


# BasisFuncMix basis set
saveTrace3 = (true, true, true, true)
## default Line-search GD optimizer
po3 = POconfig(;maxStep, target=-10.0, saveTrace=saveTrace3)
## L-BFGS optimizer from Optim
lbfgs = function (f, gf, x0)
    # om = LBFGS(linesearch=Optim.LineSearches.BackTracking())
    om = LBFGS()
    d = Optim.OnceDifferentiable(f, x->gf(x)[begin], x0, inplace=false)
    options = Optim.Options(;allow_f_increases=false, Optim.default_options(om)...)
    state = Optim.initial_state(om, options, d, x0)
    function (x, _, _)
        state.x .= x
        Optim.update_state!(d, state, om)
        Optim.update_g!(d, state, om)
        Optim.update_h!(d, state, om)
        x .= state.x
    end
end

po4 = POconfig(;maxStep, optimizer=lbfgs, threshold=(NaN, NaN), saveTrace=saveTrace3)
pos2 = (po3, po4)
E_t3s = (-1.7380134127831015, -1.7476281333694823)
# L, α₁, α₂, d₁, d₂
par_t3s  = ([ 2.6830443950696483, 0.238645962872051,  0.5379614398083945, 
              0.7282431933544267, 1.219088046135613], 
            [-0.2741666981281971, 0.1708997973009846, 0.7225008919620662, 
              0.7726145367299866, 1.5056059151556218])
grad_t3s = ([ 0.0167374425003060550,  0.014772482378271667, 0.001854882588391090, 
             -0.0014810659839501143,  0.000884740216258058], 
            [-1.4080875367698687e-5, -0.001238455709190579, 0.000203270979579692, 
             -1.7607561187192955e-6,  9.035470436162251e-7])

for ((i, po), E_t3, par_t3, grad_t3) in zip(enumerate(pos2), E_t3s, par_t3s, grad_t3s)
    gf2_2 = GaussFunc(0.7, 1.0)
    grid2 = GridBox(1, 3.0)
    bs2_2 = genBasisFunc.(grid2.point, Ref([gf2_2]))
    gf3 = GaussFunc(0.5, 1.0)
    bs3 = (bs2_2 .+ genBasisFunc(fill(0.0, 3), gf3)) |> collect
    pars3 = markParams!(bs3, true)[1:5]

    local Es3L, ps3L, grads3L, fRess3
    output = @capture_out begin
        _, Es3L, ps3L, grads3L, fRess3 = optimizeParams!(pars3, bs3, nuc, nucCoords, Ne, po)
    end

    @test fRess3[end][begin][begin].shared.Etots[end] == Es3L[end]
    Es3Ldiffs = [Es3L[i]-Es3L[i-1] for i in 2:lastindex(Es3L)]

    test3bl1 = all(i<=0 || isapprox(i, 0, atol=errorThreshold2) for i in Es3Ldiffs)
    test3bl2 = isapprox(Es3L[end], E_t3, atol=errorThreshold1)
    test3bl3 = isapprox(ps3L[end][2:end], par_t3[2:end], atol=errorThreshold3)
    test3bl4 = isapprox(abs(ps3L[end][1]), abs(par_t3[1]), atol=errorThreshold3)
    test3bl5 = isapprox(grads3L[end], grad_t3, atol=50errorThreshold1)

    test3bl1 || (@show (i, po) test3bl1 Es3Ldiffs)
    @test test3bl1
    test3bl2 || (@show (i, po) test3bl2 Es3L[end])
    @test test3bl2
    test3bl3 || (@show (i, po) test3bl3 ps3L[end][2:end])
    @test test3bl3
    test3bl4 || (@show (i, po) test3bl4 ps3L[end][1])
    @test test3bl4
    test3bl5 || (@show (i, po) test3bl5 grads3L[end])
    @test test3bl5

    all([test3bl1, test3bl2, test3bl3, test3bl4, test3bl5]) || 
    println("BasisFuncMix basis set $i process:\n", output)
end


# Convergence test
gf4 = GaussFunc(1.0, 0.5)
bs4 = genBasisFunc.(nucCoords, Ref(gf4), ["S", "P"])
pars4 = markParams!(bs4, true)
αs = getParams(pars4, :α)
res = optimizeParams!(αs, bs4, nuc, nucCoords, printInfo=false)
@test length(res) == 2
@test res[begin]
@test αs[][] >= 0
@test isapprox(res[begin+1][end], -1.5376111420710181, atol=errorThreshold1)


# function initializeOFconfig!
nucCoords2 = [[-0.7,0.0,0.0], [0.7,0.0,0.0], [0.0, 0.0, 0.0]]
nuc2 = ["H", "H", "O"]
bs2 = genBasisFunc.(nucCoords2, "STO-3G", ["H", "H", "O"]) |> flatten
siz = orbitalNumOf.(bs2) |> sum
HFc1 = HFconfig(C0=(zeros(siz, siz),))
HFc2 = HFconfig(C0=:SAD)
res1 = runHF(bs2, nuc2, nucCoords2, 
             initializeOFconfig!(HFc1, bs2, Quiqbox.arrayToTuple(nuc2), 
                                 Quiqbox.genTupleCoords(Float64, nucCoords2)), 
             printInfo=false)
res2 = runHF(bs2, nuc2, nucCoords2, HFc2, printInfo=false)
@test hasEqual(res1, res2)

# PO method :DirectRHFenergy vs :HFenergy
points = GridBox((1,0,0), 1.4).point
coords = coordOf.(points)
bs3bf1 = genBasisFunc(points[begin], "STO-3G", "H")[]
bs3bf2 = genBasisFunc(bs3bf1, points[end])
bs3 = [bs3bf1, bs3bf2]

bs3_1 = deepcopy(bs3)
res3 = runHF(bs3, nuc, coords, printInfo=false)
bs3_2 = [sum(c.*bs3) for c in eachcol(res3.C[begin])]

pars1 = markParams!(bs3_1, true)
pcfg1 = POconfig(method=:HFenergy, optimizer=lbfgs)
poRes3_1 = optimizeParams!(pars1, bs3_1, nuc, coords, pcfg1, printInfo=false)

pcfg2 = POconfig(method=:DirectRHFenergy, optimizer=lbfgs)
pars2 = markParams!(bs3_2, true)
poRes3_2 = optimizeParams!(pars2, bs3_2, nuc, coords, pcfg2, printInfo=false)

@test poRes3_1[begin]
@test poRes3_2[begin]
@test isapprox(poRes3_1[end][end], poRes3_2[end][end], atol=0.1errorThreshold2)

end