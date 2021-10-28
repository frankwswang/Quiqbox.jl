# Parameter Optimization

## Selectively Optimizing Parameters

In the [Basis Sets](@ref) section, we have briefly introduced the parameters in terms of [`ParamBox`](@ref) that are embedded in containers such as [`BasisFunc`](@ref) and [`BasisFuncs`](@ref) that are directly used to form a basis set. This means how we construct the basis set using the parameters will determine the parameter space to optimize the basis set. For more information please refer to [Constructing basis sets based on ParamBox](@ref).

Here is an example of using [`GaussFunc`](@ref) and [`GridBox`](@ref) to quickly generate a grid-based basis set with only 3 actual parameters. One is the spacing ``L`` of the grid points that indirectly determines all the center coordinates of basis function through a series of mapping functions; the other two are one exponent coefficient ``\alpha`` and one contraction coefficient ``d``.
```@repl 4
push!(LOAD_PATH,"../../src/") # hide
using Quiqbox # hide

nuc = ["H", "H"]

nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]

grid = GridBox(1, 3.0)

gf1 = GaussFunc(0.7,1)

bs = genBasisFunc.(grid.box, Ref([gf1]))
```

After constructing the basis set, we need to use [`uniqueParams!`](@ref) to mark all the 
unique parameters that can also be optimized later:
```@repl 4
pars = uniqueParams!(bs, filterMapping=true)
```

As expected, there are indeed only 3 unique tunable independent parameters despite the basis set has 8 basis functions. However, if we take a step further, we can remove ``d`` since each basis function here is just one same Gaussian function. Thus, input the intent parameters (along with other necessary arguments) into the [`optimizeParams!`](@ref) and we can sit and wait for the optimization iterations to complete.
```@repl 4
parsPartial = [pars[1], pars[4]]

Es, pars, grads = optimizeParams!(bs, parsPartial, nuc, nucCoords, maxSteps=20);
```

After the optimization, you can check the basis set and we can see the parameters inside of it is also changed. This is because the `!` in the function names indicates that `optimizeParams!` is [a function that modifies its arguments](https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention).
```@repl 4
getParams(bs)
```

It you want to go through the above example by yourself, you can also find the script [here](https://github.com/frankwswang/Quiqbox.jl/blob/main/examples/OptimizeParams.jl).

## Store Customized Basis Set

Now, if you want, you can also store the information of the basis set in an container called [`GTBasis`](@ref) that not only includes the basis set, but also the related 1-electron and 2-electron integral values (nuclear attraction is not stored). `GTBasis` can also be accepted as an argument for [`runHF`](@ref) to save the time of recalculating the integrals of the basis set.
```@repl 4
GTBasis(bs)
```