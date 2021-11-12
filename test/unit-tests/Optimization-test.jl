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

E_t1s = [-1.775682556228, -1.775682514824]
# X₁, X₂, Y₁, Y₂, Z₁, Z₂, α₁, α₂, d₁, d₂
par_t1s = [[-0.686629461207,  0.686629461207,  0.0,             0.0, 
             0.0,             0.0,             1.331636672728,  0.311858635704, 
             0.454798446619,  0.662643931788], 
           [-0.686629463257,  0.686629463257,  0.0,             0.0, 
             0.0,             0.0,             1.331636672181,  0.311858633503, 
             0.454798445732,  0.662643932396]]

grad_t1s = [[-0.056475936092,  0.056475936092,  0.0,             0.0, 
              0.0,             0.0,            -0.125188275118,  0.017527948867, 
             -0.107797228786,  0.073985454102], 
            [-0.056475926135,  0.056475926135,  0.0,             0.0, 
              0.0,             0.0,            -0.125188271319,  0.017527947101, 
             -0.107797224221,  0.073985450757]]

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
    Es2L, pars2L, grads2L = optimizeParams!(bs2, pars2, nuc, nucCoords, maxSteps = 100)
end

E_t2 = -1.16666630
# L, α
par_t2  = [2.90239973, 0.17642659]
grad_t2 = [0.48936670, 0.10233850]

@test isapprox(Es2L[end], E_t2, atol=errorThreshold2)
@test isapprox(pars2L[end, :], par_t2, atol=errorThreshold2)
@test isapprox(grads2L[end, :], grad_t2, atol=errorThreshold2)

end