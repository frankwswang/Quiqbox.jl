# Parameter Optimization

## Selectively Optimizing Parameters

In the [Basis Sets](@ref) section, we have briefly introduced the parameters in terms of [`ParamBox`](@ref) that are embedded in containers such as [`BasisFunc`](@ref) and [`BasisFuncs`](@ref) directly used to form a basis set. This means how we construct the basis set using the parameters will determine the parameter space for the basis set optimization. For more information please refer to [Constructing basis sets based on ParamBox](@ref).

Here is an example of using [`GaussFunc`](@ref) and [`GridBox`](@ref) to quickly generate a grid-based basis set with only 3 independent parameters. One is the spacing ``L`` of the grid points that indirectly determines all the center coordinates of the basis function through a series of mapping functions; the other two are one exponent coefficient ``\alpha`` and one contraction coefficient ``d``.
```@repl 4
push!(LOAD_PATH,"../../src/") # hide
using Quiqbox # hide

nuc = ["H", "H"]

nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]

grid = GridBox(1, 3.0)

gf1 = GaussFunc(0.7,1)

bs = genBasisFunc.(grid.box, Ref([gf1]))
```

After constructing the basis set, we need to use [`markParams!`](@ref) to mark all the unique parameters that can also be optimized later:
```@repl 4
pars = markParams!(bs, true)
```

As expected, there are only 3 unique tunable independent parameters despite the basis set having 8 basis functions with a total of 40 parameters. However, if we take a step further, we can remove ``d`` since each basis function here is just one same Gaussian function. Thus, input the left parameters (along with other necessary arguments) into the [`optimizeParams!`](@ref) and we can sit and wait for the optimization iterations to complete.
```@repl 4
parsPartial = pars[1:2]

Es, ps, grads = optimizeParams!(parsPartial, bs, nuc, nucCoords, POconfig((maxStep=20,)));
```

After the optimization, you can check the basis set and we can see the parameters inside of it are also changed. This is because the `!` in the function names indicates that `optimizeParams!` is [a function that modifies its arguments](https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention).
```@repl 4
getParams(bs)
```

If you want to go through the above example by yourself, you can also find the script [here](https://github.com/frankwswang/Quiqbox.jl/blob/main/examples/OptimizeParams.jl).

## Store Customized Basis Set

You can also store the information of the basis set in [`GTBasis`](@ref) which contains not only the basis set but also the related 1-electron and 2-electron integral values. GTBasis can also be an input argument for [`runHF`](@ref) to save the time of recalculating the integrals of basis sets.
```@repl 4
GTBasis(bs)
```