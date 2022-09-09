# Parameter Optimization

## Selectively optimizing parameters

In the [Basis Sets](@ref) section, we have briefly introduced [`ParamBox`](@ref) as the parameters embedded in containers such as [`BasisFunc`](@ref) and [`BasisFuncs`](@ref). This means how we construct the basis set using the parameters will determine the parameter space for the basis set optimization. (For more information please refer to [Constructing basis sets based on ParamBox](@ref).) Sometimes, we can select parameters for optimization to achieve higher efficiency.

Here is an example of using [`GaussFunc`](@ref) and [`GridBox`](@ref) to quickly generate a grid-based basis set with only 3 independent parameters. One is the spacing ``L`` of the grid points that determines all the center coordinates of the basis functions; the other two are the exponent coefficient ``\alpha`` and the contraction coefficient ``d``.
```@setup 3
    push!(LOAD_PATH,"../../src/")
    using Quiqbox
```
```@repl 3
nuc = ["H", "H"];

nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]];

grid = GridBox(1, 3.0)

gf1 = GaussFunc(0.7, 1.0);

bs = genBasisFunc.(grid.point, Ref(gf1)) |> collect;
```

After building the basis set, we need to use [`markParams!`](@ref) to mark all the unique parameters that can also be optimized later:
```@repl 3
pars = markParams!(bs, true)
```

When `markParams!`'s second argument is set to `true`, it will return only the `ParamBox`es that have unique independent variables. Thus, the length of `pars` is 3 for the aforementioned three independent parameters, despite the basis set having eight basis functions with a total of 40 parameters. However, if we take a step further, we can remove the `ParamBox` representing ``d`` since each basis function here is just one same Gaussian function. Thus, input the rest parameters (along with other necessary arguments) into [`optimizeParams!`](@ref) and we will have a more efficient optimization iteration: 
```@repl 3
parsPartial = pars[1:2];

Es, ps, grads = optimizeParams!(parsPartial, bs, nuc, nucCoords, POconfig((maxStep=10,)));
```

After the optimization, you can check the original `bs` and find that the inside parameters are changed as well. This is because the `!` in the function name indicates that `optimizeParams!` is [a function that modifies its arguments](https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention).
```@repl 3
getParams(bs)
```

If you want to go through the above example by yourself, you can find the script [here](https://github.com/frankwswang/Quiqbox.jl/blob/main/examples/OptimizeParams.jl).