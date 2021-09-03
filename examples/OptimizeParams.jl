using Quiqbox

nuc = ["H", "H"]

nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]

grid = GridBox(1, 3.0)

gf1 = GaussFunc(0.7,1)

bs = genBasisFunc.(grid.box, Ref([gf1]))

pars = uniqueParams!(bs, filterMapping=true)

parsPartial = [pars[1], pars[3]]

optimizeParams!(bs, parsPartial, nuc, nucCoords, maxSteps=20)

# # You can also use more advanced optimizers from other packages.
# using Flux # First do `Pkg.add("Flux")` if you haven't installed the package.
# using Flux.Optimise: update!
# optimizer = AMSGrad(0.001)
# GDm = (prs, grad) -> update!(optimizer, prs, grad)
# optimizeParams!(bs, parsPartial, nuc, nucCoords; GDmethod=GDm)