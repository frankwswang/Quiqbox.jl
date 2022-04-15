# Basis Sets

The procedure to construct a basis set can fundamentally break down into several basic steps: first, choose a set of (tunable) parameters, and build the Gaussian functions around those parameters, then the basis functions around the Gaussian functions, finally the basis set.

The data structure formularized by Quiqbox in each step, namely the level of data complexity, can be summarized in the following table.

| level  | objective  |  data structure | container type  | example of type instances |
| :---: | :---:   | :---:           | :---: | :---:          |
| 4 | basis set | `Array` or `struct` as the container of basis functions | `Array`, `GTBasis` | `Vector{<:BasisFunc}`, `GTBasis{BasisFunc, Float64}`...|
| 3 | basis functions | linear combination of Gaussian functions | `CompositeGTBasisFuncs` | `BasisFunc{0, 1}`, `BasisFuncs{1, 3, 3}`, `BasisFuncMix{2, BasisFunc{0, 1}, 6}`...|
| 2 | Gaussian functions | (primitive) Gaussian functions | `AbstractGaussFunc` | `GaussFunc`|
| 1 |  pool of parameters | center coordinates, exponents of Gaussian functions | `ParamBox` | `ParamBox{Float64, :α, FLevel{1, 0}}`... |


Depending on how much control the user wants to have over each step, Quiqbox provides several [methods](https://docs.julialang.org/en/v1/manual/methods/) of related functions to give users the freedom to balance between efficiency and customizability.

Below are some examples from the simplest way to relatively more flexible ways to construct a basis set in Quiqbox. Hopefully, these use cases can also work as inspirations for more creative ways to manipulate basis sets.

## Basis Set Construction

### Constructing basis sets from existing basis sets

First, you can construct an atomic basis set at one coordinate by inputting its center coordinate and a `Tuple` of its name and corresponding atom in `String`.
```@repl 1
push!(LOAD_PATH,"../../src/") # hide
using Quiqbox # hide
bsO = Quiqbox.genBasisFunc([0,0,0], ("STO-3G", "O"))
```

Notice that in the above result there are 2 types of `struct`s in the returned `Vector`: `BasisFunc` and `BasisFuncs`. `BasisFunc` is the most basic `DataType` to hold the data of a basis function; `BasisFuncs` is very similar except it may hold multiple orbitals with only the spherical harmonics ``Y_{ml}`` being different when the orbital angular momentum ``l>0``.

!!! info "Unit System"
    Hartree atomic units are the unit system used in Quiqbox.

If you want to postpone the specification of the center, you can replace the 1st argument with `missing`, and then use the function `assignCenter!` to assign the coordinates later.
```@repl 1
bsO = genBasisFunc(missing, ("STO-3G", "O"))

assignCenter!([0,0,0], bsO[1]);

bsO

assignCenter!.(Ref([0,0,0]), bsO[2:end])
```

If you omit the atom in the arguments, "H" will be set in default. Notice that even though there's only 1 single basis function in H's STO-3G basis set, the returned value is still an `Array`.
```@repl 1
bsH_1 = genBasisFunc([-0.5, 0, 0], "STO-3G")

bsH_2 = genBasisFunc([ 0.5, 0, 0], "STO-3G")
```

Finally, you can use Quiqbox's included tool function `flatten` to merge the three atomic basis sets into one molecular basis set:
```@repl 1
bsH20 = [bsO, bsH_1, bsH_2] |> flatten
```

Not simple enough? Here's a more compact way of realizing the above steps if you are familiar with some [syntactic sugars](https://en.wikipedia.org/wiki/Syntactic_sugar) in Julia:
```@repl 1
cens = [[0,0,0], [-0.5,0,0], [0.5,0,0]]

bsH20_2 = genBasisFunc.(cens, [("STO-3G", "O"), fill("STO-3G", 2)...]) |> flatten
```

In quiqbox, the user can often deal with several multi-layer containers (mainly `struct`s), it might be easy to get lost or uncertain about whether we are creating the objects intended. Quiqbox provides another tool function `hasEqual` that lets you compare if two objects hold the same data and structure. For example, if we want to see whether `bsH20_2` created in the faster way is the same (not identical) as `bsH20`, we can verify it as follows:
```@repl 1
hasEqual(bsH20, bsH20_2)
```

If the basis set you want to use is not pre-stored in Quiqbox, you can use `genBFuncsFromText` to generate the basis set from a **Gaussian** format `String`:
```@repl 1
genBasisFunc(missing, ("6-31G", "Kr"))

# Data from https://www.basissetexchange.org
txt_Kr_631G = """
Kr     0
S    6   1.00
      0.1205524000D+06       0.1714050000D-02
      0.1810225000D+05       0.1313805000D-01
      0.4124126000D+04       0.6490006000D-01
      0.1163472000D+04       0.2265185000D+00
      0.3734612000D+03       0.4764961000D+00
      0.1280897000D+03       0.3591952000D+00
SP   6   1.00
      0.2634681000D+04       0.2225111000D-02       0.3761911000D-02
      0.6284533000D+03       0.2971122000D-01       0.2977531000D-01
      0.2047081000D+03       0.1253926000D+00       0.1311878000D+00
      0.7790827000D+02       0.1947058000D-02       0.3425019000D+00
      0.3213816000D+02      -0.5987388000D+00       0.4644938000D+00
      0.1341845000D+02      -0.4958972000D+00       0.2087284000D+00
SP   6   1.00
      0.1175107000D+03      -0.6157662000D-02      -0.6922855000D-02
      0.4152553000D+02       0.5464841000D-01      -0.3069239000D-01
      0.1765290000D+02       0.2706994000D+00       0.4480260000D-01
      0.7818313000D+01      -0.1426136000D+00       0.3636775000D+00
      0.3571775000D+01      -0.7216781000D+00       0.4952412000D+00
      0.1623750000D+01      -0.3412008000D+00       0.2086340000D+00
SP   3   1.00
      0.2374560000D+01       0.3251184000D+00      -0.3009554000D-01
      0.8691930000D+00      -0.2141533000D+00       0.3598893000D+00
      0.3474730000D+00      -0.9755083000D+00       0.7103098000D+00
SP   1   1.00
      0.1264790000D+00       0.1000000000D+01       0.1000000000D+01
D    3   1.00
      0.6853888000D+02       0.7530705000D-01
      0.1914333000D+02       0.3673551000D+00
      0.6251213000D+01       0.7120146000D+00
D    1   1.00
      0.1979236000D+01       1.0000000
""";

genBFuncsFromText(txt_Kr_631G, adjustContent=true)
```

### Constructing basis sets from `GaussFunc`

If you want to specify the parameters of each Gaussian function when constructing a basis set, you can first construct the container for Gaussian functions: `GaussFunc`, and then build the basis function upon them:
```@repl 2
using Quiqbox # hide
gf1 = GaussFunc(2.0, 1.0)

gf2 = GaussFunc(2.5, 0.75)

bf1 = genBasisFunc([1.0,0,0], [gf1, gf2])
```

Unlike `BasisFunc` there's no proprietary function for it, you simply input the exponent coefficient and the contraction coefficient as the 1st and 2nd arguments respectively to its constructor. As for the method of `genBasisFunc` in this case, the default subshell is set to be "S" as the optional 3rd argument, but you can construct a `BasisFuncs` which contains all the orbitals within one specified subshell:
```@repl 2
bf2 = genBasisFunc([1.0,0,0], [gf1, gf2], "P")
```

You can even choose one or a few orbitals to keep by indicting them using an `NTuple{3, Int}` in the Cartesian representation:
```@repl 2
bf3 = genBasisFunc([1.0,0,0], [gf1, gf2], (1,0,0))

bf4 = genBasisFunc([1.0,0,0], [gf1, gf2], [(1,0,0), (0,0,1)])
```

Again, if you want a faster solution, you can also directly define the 2 `GaussFunc` parameter(s) in a 2-element `Tuple` as the 2nd argument for `genBasisFunc`:
```@repl 2
bf5 = genBasisFunc([1.0,0,0], ([2.0, 2.5], [1.0, 0.75]), [(1,0,0), (0,0,1)])

hasEqual(bf4, bf5)
```

### Constructing basis sets based on `ParamBox`

Sometimes you may want the parameters of basis functions (or `GaussFunc`) to be under some constraints (which can be crucial for the later basis set optimization), this is when you need a deeper level of control over the parameters, through its direct container: `ParamBox`. In fact, in the above example, we have already had a glimpse of it through the printed info in the REPL:
```@repl 2
gf1
```
The 2 fields of a `GaussFunc`, `.xpn`, and `.con` are `ParamBox`, and their actual value can be accessed through syntax `[]`:
```@repl 2
gf1.xpn

gf1.con

gf1.xpn[]

gf1.con[]
```

Since the data are not directly stored as `primitive type`s but rather inside `struct` `ParamBox`, this allows the direct assignment or shallow copy of them without reconstructing new data, but bindings to the original objects:
```@repl 2
gf3 = GaussFunc(1.1, 1)

# Direct assignment
gf3_2 = gf3

gf3.xpn[] *= 2

gf3

gf3_2

# Shallow copy: `fill`
bf6 = genBasisFunc([1,0,0], fill(gf3, 2))

bf6.gauss

bf6.gauss[1].xpn[] = 1.1

gf3_2.xpn[] == gf3.xpn[] == bf6.gauss[2].xpn[] == 1.1

```

Based on such trait in Julia, you can, for instance, create a basis set that enforces all the `GaussFunc`s have the **identical** gaussian function parameters:
```@repl 2
gf4 = GaussFunc(2.5, 0.5)

bs7 = genBasisFunc.([rand(3) for _=1:2], Ref(gf4))

markParams!(bs7)
```

`markParams!` marks all the parameters of the given basis set. As you can see, even though `bs7` has 2 `GaussFunc`s as basis functions, overall it only has 1 unique coefficient exponent ``\alpha_1`` and 1 unique contraction coefficient ``d_1`` if we ignore the center coordinates.

## Dependent Variable as a parameter

Another control the user can have on the parameters in Quiqbox is to make ParamBox represent a dependent variable defined by the mapping function of another independent parameter.

Such a mapping function is stored in the `map` field of a `ParamBox` (which normally is an ``R \to R`` mapping). The mapped value can be accessed through 
syntax `()`. In default, the variable is mapped to itself:
```@repl 2
pb1 = gf4.xpn

pb1.map

pb1[] == pb1()
```

You can get a clearer view of the mapping relations in a `ParamBox` using `getVarDict`
```@repl 2
getVarDict(pb1)
```
!!! info "Parameter represented by `ParamBox`"
    The mapped variable (value) of a `ParamBox` is always used as the parameter (parameter value) it represents in the construction of any basis function component. If you want to optimize the variable that is mapped from, the `ParamBox` needs to be marked as "differentiable". For more information on parameter optimization, please see the docstring of [`ParamBox`](@ref) and section [Parameter Optimization](@ref).

## Linear combinations of basis functions

Apart from the flexible control of basis function parameters, a major feature of Quiqbox is the ability to construct a basis function from the linear combination of other basis functions. Specifically, additional methods of `+` and `*` (operator syntax for [`add`](@ref) and [`mul`](@ref)) are implemented for `CompositeGTBasisFuncs` so the user can combine basis functions as if they are `Number`:
```@repl 3
using Quiqbox # hide
bf7 = genBasisFunc([1,0,1], (1.5,3))

bf8 = genBasisFunc([1,0,1], (2,4))

bf9 = bf7*0.5 + bf8

bf9.gauss[1].con() == 3 * 0.5
```

As you can see, the type of `bf9` is still `BasisFunc` since `bf7` and `bf8` have the same center coordinates, hence all the Gaussian functions inside `bf9`also have the same center coordinates. What if the combined basis functions are multi-center?
```@repl 3
bf10 = genBasisFunc([1,1,1], (1.2,3))

bf11 = bf8 + bf10
```
The type of `bf11` is called `BasisFuncMix`, which means we can't express it as a contracted Gaussian-type orbital (CGTO), but as a "mixture" of multi-center CGTOs.

There are other cases that can result in a `BasisFuncMix` as a returned basis function. For example:
```@repl 3
bf12 = genBasisFunc([1,1,1], (1.2,3), (1,1,0))

bf10 + bf12
```

Despite the cause of generating a `BasisFuncMix`, it's still a valid basis function in Quiqbox and you can use it to call functions that accept `CompositeGTBasisFuncs` as input arguments:
```@repl 3
overlap(bf11, bf11)
```