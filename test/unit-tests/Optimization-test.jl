using Test
using Quiqbox
using Suppressor: @suppress_out

@testset "Optimization.jl" begin

errorThreshold = 1e-12

# Floating basis set
nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]
mol = ["H", "H"]
bfSource1 = genBasisFunc(("STO-2G", "H"))[]
gfs1 = [bfSource1.gauss...]
cens = gridPoint.(nucCoords)
bs1 = genBasisFunc.(cens, Ref(gfs1))
pars1 = uniqueParams!(bs1, ignoreMapping=true)

local Es1L, pars1L, grads1L
@suppress_out begin
    Es1L, pars1L, grads1L = optimizeParams!(bs1, pars1, mol, nucCoords, maxSteps = 200)
end

E_t1 = -1.77568255622806
par_t1 = [1.3316366727278381,  0.3118586357035099, 0.45479844661867297, 
          0.6626439317879435, -0.6866294612074542, 0.6866294612074546, 
          0.0, 0.0, 0.0, 0.0]
grad_t1 = [-0.1251882751182149,   0.017527948866820964, -0.10779722878571235, 
            0.07398545410241009, -0.05647593609221346,   0.05647593609221585, 
            0.0, 0.0, 0.0, 0.0]

@test isapprox(Es1L[end], E_t1, atol=errorThreshold)
@test isapprox(pars1L[end, :], par_t1, atol=errorThreshold)
@test isapprox(grads1L[end, :], grad_t1, atol=errorThreshold)


# Grid-based basis set
grid = GridBox(1, 3.0)
gf2 = GaussFunc(0.7,1)
bs2 = genBasisFunc.(grid.box, Ref([gf2]))

pars2 = uniqueParams!(bs2, ignoreMapping=true)[[1,3]]

local Es2L, pars2L, grads2L
@suppress_out begin
    Es2L, pars2L, grads2L = optimizeParams!(bs2, pars2, mol, nucCoords, maxSteps = 200)
end

E_t2 = -1.1804184294076494
par_t2 = [0.1792288515300005, 2.8537000394661676]
grad_t2 = [-0.05218998970360911, 0.4901549373691791]

@test isapprox(Es2L[end], E_t2, atol=errorThreshold)
@test isapprox(pars2L[end, :], par_t2, atol=errorThreshold)
@test isapprox(grads2L[end, :], grad_t2, atol=errorThreshold)

end