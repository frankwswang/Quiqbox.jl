using Quiqbox

nuc = ["H", "H"]

nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]

grid = GridBox(1, 3.0)

gf1 = GaussFunc(0.7, 1.0)

bs = genBasisFunc.(grid.point, Ref(gf1))

pars = markParams!(bs, true)

parsPartial = pars[1:2]

Es, ps, grads = optimizeParams!(parsPartial, bs, nuc, nucCoords, POconfig(maxStep=10))

# # You can also use more advanced optimizers from other packages.
# using Flux # First do `using Pkg; Pkg.add("Flux")` if you haven't installed the package.
# using Flux.Optimise: update!
# optimizer = AMSGrad(0.001)
# GDm = (prs, grad) -> update!(optimizer, prs, grad)
# optimizeParams!(parsPartial, bs, nuc, nucCoords, POconfig(GD=GDm, maxStep=20))