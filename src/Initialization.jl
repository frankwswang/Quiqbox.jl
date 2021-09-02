const QuiqboxSubModules = ["Molden"]

# Initialization function.
function __init__()
    tryIncluding.(QuiqboxSubModules)

    # Force JIT compilation
    # GTBasis(genBasisFunc.([[0,0,0], [0,0,1]], "STO-3G") |> flatten)
end