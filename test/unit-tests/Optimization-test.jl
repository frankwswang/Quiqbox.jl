using Test
using Quiqbox
using Suppressor: @suppress_out

@testset "Optimization.jl" begin

errorThreshold = 1e-12

# Floating basis set
nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]
mol = ["H", "H"]
bfSource1 = genBasisFunc(missing, ("STO-2G", "H"))[]
gfs1 = [bfSource1.gauss...]
cens = gridPoint.(nucCoords)
bs1 = genBasisFunc.(cens, Ref(gfs1))
pars1 = uniqueParams!(bs1, filterMapping=true)

local Es1L, pars1L, grads1L
@suppress_out begin
    Es1L, pars1L, grads1L = optimizeParams!(bs1, pars1, mol, nucCoords, maxSteps = 200)
end

E_t1 = -1.7756825544202863
par_t1 = [1.3316366727266504,  0.3118586356968696, 0.45479844661739155, 
          0.6626439317888224, -0.6866294552929423, 0.6866294671212336, 
          0.0, 0.0, 0.0, 0.0]
grad_t1 = [-0.12518827510873726,  0.017527948869488608, -0.10779722877906275, 
            0.07398545409754818, -0.056469017809637645,  0.05648285438029821, 
            0.0, 0.0, 0.0, 0.0]

@test isapprox(Es1L[end], E_t1, atol=errorThreshold)
@test isapprox(pars1L[end, :], par_t1, atol=errorThreshold)
@test isapprox(grads1L[end, :], grad_t1, atol=errorThreshold)


# Grid-based basis set
grid = GridBox(1, 1.5)
gf2 = GaussFunc(0.7,1)
bs2 = genBasisFunc.(grid.box, Ref([gf2]))

pars2 = uniqueParams!(bs2, filterMapping=true)[[1,3]]

local Es2L, pars2L, grads2L
@suppress_out begin
    Es2L, pars2L, grads2L = optimizeParams!(bs2, pars2, mol, nucCoords, maxSteps = 200)
end

E_t2 = -1.6227282934931644
par_t2  = [0.49801096561597613, 1.408314680969665]
grad_t2 = [0.4557364086913408, 0.32317362845855635]

@test isapprox(Es2L[end], E_t2, atol=errorThreshold)
@test isapprox(pars2L[end, :], par_t2, atol=errorThreshold)
@test isapprox(grads2L[end, :], grad_t2, atol=errorThreshold)

end