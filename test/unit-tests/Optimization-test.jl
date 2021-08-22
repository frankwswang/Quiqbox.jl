using Test
using Quiqbox

@testset "Optimization.jl" begin

errorThreshold = 1e-12

nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]

mol = ["H", "H"]

bfSource = BasisFunc(("STO-2G", "H"))[]

gfs = [bfSource.gauss...]

cens = gridPoint.(nucCoords)

bs = BasisFunc.(cens, Ref(gfs))

pars = uniqueParams!(bs, ignoreMapping=true)

Es, pars, grads = optimizeParams!(bs, pars, mol, nucCoords, maxSteps = 500, printInfo=false)

E_t = -1.7828659178214732
par_t = [1.3683913812855784, 0.3058302439403923, 0.48622948741096983, 0.6399410170157525, -0.6716612345728881, 0.6716612345728885, 0.0, 0.0, 0.0, 0.0]
grad_t = [-0.11951740070926406, 0.02005079056112935, -0.1016591886019295, 0.07724101729724206, -0.04376145807934295, 0.04376145807934384, 0.0, 0.0, 0.0, 0.0]

@test isapprox(Es[end], E_t, atol=errorThreshold)
@test isapprox(pars[end, :], par_t, atol=errorThreshold)
@test isapprox(grads[end, :], grad_t, atol=errorThreshold)

end