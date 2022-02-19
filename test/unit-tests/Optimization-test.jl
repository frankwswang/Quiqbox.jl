using Quiqbox
using Test
using Quiqbox
using Suppressor: @suppress_out

@testset "Optimization.jl" begin

errorThreshold1 = 1e-10
errorThreshold2 = 1e-4

# Floating basis set
nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]
nuc = ["H", "H"]

ecMethods = [Quiqbox.defaultECmethod(:RHF, nuc, nucCoords), 
             Quiqbox.defaultECmethod(:UHF, nuc, nucCoords)]

E_t1s = [-1.7752648332225505, -1.6375013582033964]
# X₁, X₂, Y₁, Y₂, Z₁, Z₂, α₁, α₂, d₁, d₂
par_t1s = [[-0.6907112938704066, 0.6907112938704066, 0.0, 0.0, 
             0.0, 0.0, 1.3144130782048333, 0.30163535090176957, 
             0.45974281498143993, 0.6592246561546644], 
           [-0.6970757822675224, 0.6847211236284498, 0.0, 0.0, 
             0.0, 0.0, 1.2891588432667265, 0.5298715025714499, 
             0.4616267310282826, 0.65790945792686]
           ]

grad_t1s = [[-0.03813070457953324, 0.038130704579533825, 0.0, 0.0, 
              0.0, 0.0, -0.007789757633386879, 0.20037574234702205, 
             -0.11693981008446266, 0.08155374191436453], 
            [ 0.06327689285193078, -0.01711544992278955, 0.0, 0.0, 
              0.0, 0.0, -0.07993519793464789, 0.9549584774913144, 
              0.1278869287371865, -0.08973274991397628]]

for (ecMethod, E_t1, par_t1, grad_t1) in zip(ecMethods, E_t1s, par_t1s, grad_t1s)

    bfSource1 = genBasisFunc(missing, ("STO-2G", "H"))[]
    gfs1 = bfSource1.gauss |> collect
    cens = makeCenter.(nucCoords)
    bs1 = genBasisFunc.(cens, Ref(gfs1))
    pars1 = uniqueParams!(bs1, filterMapping=true)

    local Es1L, pars1L, grads1L
    @suppress_out begin
        Es1L, pars1L, grads1L = optimizeParams!(bs1, pars1, nuc, nucCoords, ecMethod, 
                                                maxSteps = 200)
    end

    @test isapprox(Es1L[end], E_t1, atol=errorThreshold1)
    @test isapprox(pars1L[end, :], par_t1, atol=errorThreshold1)
    @test isapprox(grads1L[end, :], grad_t1, atol=errorThreshold1)

end

# Grid-based basis set
grid = GridBox(1, 3.0)
gf2 = GaussFunc(0.7,1)
bs2 = genBasisFunc.(grid.box, Ref([gf2]))

pars2 = uniqueParams!(bs2, filterMapping=true)[[1,4]]

local Es2L, pars2L, grads2L
@suppress_out begin
    Es2L, pars2L, grads2L = optimizeParams!(bs2, pars2, nuc, nucCoords, maxSteps = 200)
end

E_t2 = -1.1665258292058682
# L, α
par_t2  = [2.8465051231251404, 0.22550104532878842]
grad_t2 = [0.3752225248706008, 0.6830952135402155]

@test isapprox(Es2L[end], E_t2, atol=errorThreshold2)
@test isapprox(pars2L[end, :], par_t2, atol=errorThreshold2)
@test isapprox(grads2L[end, :], grad_t2, atol=errorThreshold2)

end