using Test
using Quiqbox
using Suppressor: @suppress_out

@testset "Optimization.jl" begin

errorThreshold1 = 1e-10
errorThreshold2 = 1e-4

# Floating basis set
nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]
mol = ["H", "H"]
bfSource1 = genBasisFunc(missing, ("STO-2G", "H"))[]
gfs1 = bfSource1.gauss |> collect
cens = makeCenter.(nucCoords)
bs1 = genBasisFunc.(cens, Ref(gfs1))
pars1 = uniqueParams!(bs1, filterMapping=true)

local Es1L, pars1L, grads1L
@suppress_out begin
    Es1L, pars1L, grads1L = optimizeParams!(bs1, pars1, mol, nucCoords, maxSteps = 200)
end

E_t1 = -1.775682554420
par_t1 =  [ 1.331636672727,  0.311858635697,  0.454798446617, 
            0.662643931789, -0.686629455293,  0.686629467121, 
            0.0, 0.0, 0.0, 0.0]
grad_t1 = [-0.125188275109,  0.017527948869, -0.107797228779, 
            0.073985454098, -0.056469017810,  0.056482854380, 
            0.0, 0.0, 0.0, 0.0]

@test isapprox(Es1L[end], E_t1, atol=errorThreshold1)
@test isapprox(pars1L[end, :], par_t1, atol=errorThreshold1)
@test isapprox(grads1L[end, :], grad_t1, atol=errorThreshold1)


# Grid-based basis set
grid = GridBox(1, 3.0)
gf2 = GaussFunc(0.7,1)
bs2 = genBasisFunc.(grid.box, Ref([gf2]))

pars2 = uniqueParams!(bs2, filterMapping=true)[[1,3]]

local Es2L, pars2L, grads2L
@suppress_out begin
    Es2L, pars2L, grads2L = optimizeParams!(bs2, pars2, mol, nucCoords, maxSteps = 100)
end

E_t2 = -1.16666630
par_t2  = [0.17642659, 2.90239973]
grad_t2 = [0.10233850, 0.48936670]

@test isapprox(Es2L[end], E_t2, atol=errorThreshold2)
@test isapprox(pars2L[end, :], par_t2, atol=errorThreshold2)
@test isapprox(grads2L[end, :], grad_t2, atol=errorThreshold2)

end