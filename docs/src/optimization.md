# Parameter Optimization

## Selectively Optimizing Parameters

In the [Basis Sets](@ref) section, we have briefly introduced [`ParamBox`](@ref) as the parameters embedded in containers such as [`BasisFunc`](@ref) and [`BasisFuncs`](@ref). This means how we construct the basis set using the parameters will determine the parameter space for the basis set optimization. (For more information please refer to [Constructing basis sets based on ParamBox](@ref).) Sometimes, we can manually select parameters for optimization to achieve higher efficiency.

Here is an example of using [`GaussFunc`](@ref) and [`GridBox`](@ref) to quickly generate a grid-based basis set with only 3 independent parameters. One is the spacing ``L`` of the grid points that determines all the center coordinates of the basis function if they are put on the grid points; the other two are one exponent coefficient ``\alpha`` and one contraction coefficient ``d``.
```@repl 4
push!(LOAD_PATH,"../../src/") # hide
using Quiqbox # hide
nuc = ["H", "H"]

nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]

grid = GridBox(1, 3.0)

gf1 = GaussFunc(0.7, 1.0)

bs = genBasisFunc.(grid.point, Ref(gf1))
```

After constructing the basis set, we need to use [`markParams!`](@ref) to mark all the unique parameters that can also be optimized later:
```@repl 4
pars = markParams!(bs, true)
```

When the second argument of `markParams!` is set to `true`, the `pars` will only include 
`ParamBox`s that have unique independent variables. Thus the length of it is 3 for the aforementioned 3 independent parameter, despite the basis set having 8 basis functions with a total of 40 parameters. However, if we take a step further, we can remove the `ParamBox` 
representing ``d`` since each basis function here is just one same Gaussian function. Thus, input the rest parameters (along with other necessary arguments) into the [`optimizeParams!`](@ref) and we can sit and wait for the an efficient optimization iterations to complete.
```@repl 4
parsPartial = pars[1:2]

Es, ps, grads = optimizeParams!(parsPartial, bs, nuc, nucCoords, POconfig((maxStep=20,)));
```

After the optimization, you can check the original `bs` and find that the inside parameters are changed as well. This is because the `!` in the function names indicates that `optimizeParams!` is [a function that modifies its arguments](https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention).
```@repl 4
getParams(bs)
```

If you want to go through the above example by yourself, you can find the script [here](https://github.com/frankwswang/Quiqbox.jl/blob/main/examples/OptimizeParams.jl).

## Store Customized Basis Set

You can store the information of the basis set in [`GTBasis`](@ref) to save the time of recalculating the corresponding integral data.
```@repl 4
GTBasis(bs)
```