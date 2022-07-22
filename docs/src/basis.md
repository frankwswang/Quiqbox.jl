# Basis Sets

The procedure of constructing a Gaussian-type basis set can fundamentally break down into several basic steps: first, make primitive Gaussian-type orbitals (GTO) using a set of parameters, then construct the basis functions from the linear combinations of those orbitals, finally build the basis set.

The data structures defined by Quiqbox in each step, form levels of data complexity. They can be summarized in the following table.

```@example 1
push!(LOAD_PATH,"../../src/") # hide
using Quiqbox # hide
bs1 = Quiqbox.genBasisFunc.([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], "STO-6G")|>flatten # hide
gtb = GTBasis(bs1) # hide
bf1 = bs1[1] # hide
bf2 = bf1 + genBasisFunc(fill(1.0, 3), (2.0, 1.0)) # hide
gf1 = bf1.gauss[1] # hide
pb1 = gf1.xpn # hide
```

| level  | objective  |  data structure | container type  | type examples|
| :---: | :---:   | :---:           | :---: | :---:          |
| 4 | basis set | collection of multiple basis functions | `Array`, `Tuple`, `GTBasis` | `Vector{<:BasisFunc{Float64, 3}}`, `GTBasis{Float64, 3, 2}`...|
| 3 | basis functions | linear combination of Gaussian functions | `GTBasisFuncs` | `BasisFunc{Float64, 3, 0, 6}`, `Quiqbox.BasisFuncMix{Float64, 3, 2}`...|
| 2 | Gaussian functions | (primitive) Gaussian functions | `AbstractGaussFunc` | `GaussFunc{Float64, FLevel{0}, FLevel{0}}`...|
| 1 |  pool of parameters | center coordinates, exponents of Gaussian functions | `ParamBox` | `ParamBox{Float64, :α, FLevel{0}}`... |


Depending on how much control the user wants to have over each step, Quiqbox defines several [methods](https://docs.julialang.org/en/v1/manual/methods/) of related functions to provide the freedom of balancing between efficiency and customizability.

Below are some examples from the simplest way to more flexible ways of constructing a basis set in Quiqbox. Hopefully, these use cases can also work as inspirations for more creative ways to customize basis sets.

## Basis set construction

### Constructing basis sets from existing basis sets

First, you can construct an atomic basis set at one coordinate by inputting its center coordinate, the basis set name and the corresponding atom symbol.
```@repl 1
push!(LOAD_PATH,"../../src/") # hide
using Quiqbox # hide
bsO = Quiqbox.genBasisFunc(fill(1.0, 3), "STO-3G", "O")
```

Notice that in the above result there are two types of `struct`s in the returned `Vector`: `BasisFunc` and `BasisFuncs`. `BasisFunc` is the most basic `DataType` to hold the data of a basis function; `BasisFuncs` is very similar except it may hold multiple orbitals with only the spherical harmonics ``Y_{ml}`` being different when the orbital angular momentum ``l>0``.

!!! info "Unit System"
    Hartree atomic units are the unit system used in Quiqbox.

If you want to postpone the specification of the center, you can replace the first argument with `missing`, then use the function `assignCenInVal!` to assign the coordinates later.
```@repl 1
bsO = genBasisFunc(missing, "STO-3G", "O")

[assignCenInVal!(fill(1.0, 3), b) for i in bsO]
```

If you omit the atom in the arguments, "H" will be set in default. Notice that even though there's only one single basis function in H's STO-3G basis set, the returned value is still a `Vector`.
```@repl 1
bsH_1 = genBasisFunc([-0.5, 0, 0], "STO-3G")

bsH_2 = genBasisFunc([ 0.5, 0, 0], "STO-3G")
```

Finally, you can use Quiqbox's included tool function [`flatten`](@ref) to merge the three atomic basis sets into one molecular basis set:
```@repl 1
bsH2O = [bsO, bsH_1, bsH_2] |> flatten
```

Not simple enough? Here's a more compact way of realizing the above steps if you are familiar with Julia's [vectorization syntactic sugars](https://docs.julialang.org/en/v1/manual/mathematical-operations/#man-dot-operators):
```@repl 1
cens = [fill(0.0,3), [-0.5,0,0], [0.5,0,0]]

bsH2O_2 = genBasisFunc.(cens, "STO-3G", ["O", "H", "H"]) |> flatten
```

In quiqbox, the user can often deal with several multi-layer containers (mainly `struct`s). It might be easy to get lost or unsure about whether the objects are created as intended. Quiqbox provides another tool function [`hasEqual`](@ref) that lets you verify if two objects hold the same-valued data and have the same structure. For example, if we want to see whether `bsH2O_2` created in the faster way is the same (not necessarily identical) as `bsH2O`, we can do as follows:
```@repl 1
hasEqual(bsH2O, bsH2O_2)
```

If the basis set you want to use is not pre-stored in Quiqbox, you can use `genBFuncsFromText` to generate the basis set from a **Gaussian** format `String`:
```@repl 1
genBasisFunc(missing, "6-31G", "Kr")

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

If you want to specify the parameters of each basis function when constructing a basis set, you can first construct the container for primitive GTO: `GaussFunc`, and then construct the basis function from them:
```@repl 2
using Quiqbox # hide
gf1 = GaussFunc(2.0, 1.0)

gf2 = GaussFunc(2.5, 0.75)

bf1 = genBasisFunc([1.0, 0, 0], [gf1, gf2])
```

Unlike `BasisFunc`, there's no additional constructor function for `GaussFunc`. As for the method of `genBasisFunc` in this case, the subshell is set to "S" as the default option since the third argument is omitted. You can construct a `BasisFuncs` which contains all the orbitals within one specified subshell:
```@repl 2
bf2 = genBasisFunc([1.0, 0, 0], [gf1, gf2], "P")
```

You can even select one or few orbitals to keep by specifying the corresponding orbital angular momentums in the Cartesian representation using `NTuple{3, Int}`:
```@repl 2
bf3 = genBasisFunc([1.0, 0, 0], [gf1, gf2], (1,0,0))

bf4 = genBasisFunc([1.0, 0, 0], [gf1, gf2], [(1,0,0), (0,0,1)])
```

Again, if you want a faster solution, you can directly define the exponent coefficients and the contraction coefficients separately in a 2-element `Tuple` as the second argument of `genBasisFunc`:
```@repl 2
bf5 = genBasisFunc([1.0, 0, 0], ([2.0, 2.5], [1.0, 0.75]), [(1,0,0), (0,0,1)])

hasEqual(bf4, bf5)
```

### Constructing basis sets based on `ParamBox`

Sometimes you may want the parameters of basis functions (or `GaussFunc`) to be under some constraints (which can be crucial for the later basis set optimization), this is when you need a deeper level of control over the parameters, through its direct container: [`ParamBox`](@ref). In fact, in the above example, we have already had a glimpse of it through the printed info in the REPL:
```@repl 2
gf1
```
The two fields of a `GaussFunc`, `.xpn`, and `.con` are `ParamBox`, and their input value (stored data) can be accessed through syntax `[]`:
```@repl 2
gf1.xpn

gf1.con

gf1.xpn[]

gf1.con[]
```

Since the data are not directly stored in primitive types but rather inside `ParamBox`, this allows the shallow copy of a `ParamBox` to share the same underlying data: 
```@repl 2
gf3 = GaussFunc(1.1, 1.0)

# Direct assignment
gf3_2 = gf3

gf3.xpn[] *= 2

gf3

gf3_2

# Shallow copy is performed when using `fill`
bf6 = genBasisFunc([1,0,0], fill(gf3, 2))

bf6.gauss

bf6.gauss[1].xpn[] = 1.1

gf3_2.xpn[] == gf3.xpn[] == bf6.gauss[2].xpn[] == 1.1

```

Based on such feature of `ParamBox`, the user can, for instance, create a basis set that enforces all the `GaussFunc`s to have **identical** gaussian function parameters:
```@repl 2
gf4 = GaussFunc(2.5, 0.5)

bs7 = genBasisFunc.([rand(3) for _=1:2], Ref(gf4))

markParams!(bs7)
```

`markParams!` marks all the parameters of a given basis set. Even though `bs7` has two `GaussFunc`s as basis functions, overall it only has one unique coefficient exponent ``\alpha_1`` and one unique contraction coefficient ``d_1`` besides the center coordinates.

## Dependent variable as a parameter

Another control the user have on the parameters in Quiqbox is making a `ParamBox` represent a variable equal to the returned value of a mapping function taking the stored data as the argument. In other words, the data stored in the `ParamBox` is an "input variable", while the represented variable is the "output variable".

Such a mapping function is stored in the `map` field of the `ParamBox` (which normally is an ``R \to R`` mapping). The "output value" can be accessed through syntax `()`. In default, the input variable is mapped to itself:
```@repl 2
pb1 = gf4.xpn

pb1.map

pb1[] == pb1()
```

You can get a clearer view of the variable value(s) in a `ParamBox` using `getVarDict`
```@repl 2
getVarDict(pb1)
```
!!! info "Parameter represented by `ParamBox`"
    The output variable of a `ParamBox` is always used in the construction of any basis function component. If you want to optimize the input variable when the mapping is nontrivial (i.e. not [`itself`](@ref)), the `ParamBox` needs to be marked as "differentiable". For more information on parameter optimization, please see the docstring of [`optimizeParams!`](@ref) and section [Parameter Optimization](@ref).

## Linear combinations of basis functions

Apart from the flexible control of basis function parameters, another major feature of Quiqbox is the ability to construct a basis function from the linear combination of other basis functions. Specifically, additional methods of `+` and `*` (operator syntax for [`add`](@ref) and [`mul`](@ref)) are implemented for `CompositeGTBasisFuncs`:
```@repl 3
using Quiqbox # hide
bf7 = genBasisFunc([1,0,1], (1.5,3))

bf8 = genBasisFunc([1,0,1], (2,4))

bf9 = bf7*0.5 + bf8

bf9.gauss[1].con() == 3 * 0.5
```

As we can see, the type of `bf9` is still `BasisFunc`, hence all the GTO inside `bf9` have the same center coordinates as well. This is because `bf7` and `bf8` have the same center coordinates. What if the combined basis functions are multi-center?
```@repl 3
bf10 = genBasisFunc([1,1,1], (1.2,3))

bf11 = bf8 + bf10
```
The type of `bf11` is called [`BasisFuncMix`](@ref), which means it cannot be expressed as a contracted Gaussian-type orbital (CGTO), but rather a "mixture" of multi-center GTOs (MCGTO).

There are other cases that can result in a `BasisFuncMix` as the returned object. For example:
```@repl 3
bf12 = genBasisFunc([1,1,1], (1.2,3), (1,1,0))

bf10 + bf12
```

In Quiqbox, `BasisFuncMix` is also accepted as a valid basis function and the user can use it to call functions that accept `CompositeGTBasisFuncs` as input argument(s):
```@repl 3
overlap(bf11, bf11)
```