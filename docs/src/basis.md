# Basis Sets

The procedure to construct a basis set can be fundamentally broken down into several basic steps: first, choose a set of (tunable) parameters, and build the Gaussian functions around those parameters, then the basis functions around the Gaussian functions, finally the basis set.

The data structure formularized by Quiqbox in each step, namely the level of data complexity, can be summarized in the following table.

| level  | objective  |  product examples | abstract type  | type instances |
| :---: | :---:   | :---:           | :---: | :---:          |
| 4 | basis set | Array of basis functions (with reusable integrals) | `Array`, `GTBasis` | `Array{<:BasisFunc, 1}`...|
| 3 | basis functions | single or linear combination of Gaussian functions | `FloatingGTBasisFunc` | `BasisFunc{:S, 1}`, `BasisFuncs{:P, 3, 3}`...|
| 2 | Gaussian functions | (primitive) Gaussian functions | `AbstractGaussFunc` | `GaussFunc`|
| 1 |  a pool of parameters | center coordinates, function coefficients | `ParamBox` | `ParamBox{:xpn, Float64}`... |


Depending on how much control the user wants to have over each step, Quiqbox provides several [methods](https://docs.julialang.org/en/v1/manual/methods/) of related functions to leave the user with the freedom to balance between efficiency and customizability. 

Below are some examples from the simplest way to relatively more flexible ways to construct a basis set in Quiqbox. Hopefully these use cases can also work as inspirations for more creative ways to manipulate basis sets.

## Basis Set Construction

### Constructing basis sets from existed basis sets

First, you can create a basis set at one coordinate by input the `Vector` of its center coordinate and a `Tuple` of its name and corresponding atom in `String`.
```@repl 1
push!(LOAD_PATH,"../../src/") # hide
using Quiqbox # hide

bsO = Quiqbox.genBasisFunc([0,0,0], ("STO-3G", "O"))
```

Notice that in the above result there are 2 types of `struct`s in the returned `Vector`: `BasisFunc` and `BasisFuncs`. `BasisFunc` is the most basic `type` to hold the data of a basis function; `BasisFuncs` is very similar except it may hold multiple orbitals with only the spherical harmonics ``Y_{ml}`` being different when the orbital angular momentum ``l>0``.

!!! info "Unit System"
    Hartree atomic units are the unit system used in Quiqbox.

If you want to postpone the specification of the center, you can replace the 1st argument with `missing`, and then use function `assignCenter!` to assign the coordinates later.
```@repl 1
bsO = genBasisFunc(missing, ("STO-3G", "O"))

assignCenter!([0,0,0], bsO[1]);

bsO

assignCenter!.(Ref([0,0,0]), bsO[2:end])
```

If you omit the atom in the arguments, H will be set in default. Notice that even there's only 1 single basis function in H's STO-3G basis set, the returned value is still in `Array` type.
```@repl 1
bsH_1 = genBasisFunc([-0.5, 0, 0], "STO-3G")

bsH_2 = genBasisFunc([ 0.5, 0, 0], "STO-3G")
```

Finally, you can use Quiqbox's included tool function `flatten` to merge the three atomic basis set into one molecular basis set:
```@repl 1
bsH20 = [bsO, bsH_1, bsH_2] |> flatten
```

Not simple enough? Here's a more compact way of realizing the above steps if you are familiar with some [syntactic sugars](https://en.wikipedia.org/wiki/Syntactic_sugar) in Julia:
```@repl 1
cens = [[0,0,0], [-0.5,0,0], [0.5,0,0]]

bsH20_2 = genBasisFunc.(cens, [("STO-3G", "O"), fill("STO-3G", 2)...]) |> flatten
```

In quiqbox, the user can often deal with several multi-layer containers (mainly `struct`s), it might be easy to get lost or uncertain that whether we are creating the objects intended. Quiqbox provides another tool function `hasEqual` that lets you compare if two objects hold the same data and structure. For example, if we want to see whether `bsH20_2` created in the faster way is same (not identical) as `bsH20`, we can verify it as follows:
```@repl 1
hasEqual(bsH20, bsH20_2)
```

If the basis set you want to use doesn't exist in Quiqbox's library, you can use `Function` `genBFuncsFromText` to generate the basis set from a **Gaussian** formatted `String`:
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

Unlike `BasisFunc` there's no proprietary function for it, you simply input the exponent coefficient and the contraction coefficient as the 1st and 2nd arguments respectively to its default constructor. As for the method of `genBasisFunc` in this case, the default subshell is set to be "S" as the optional 3rd argument, but you can construct a `BasisFuncs` which contains all the orbitals within a specified one:
```@repl 2
bf2 = genBasisFunc([1.0,0,0], [gf1, gf2], "P")
```

You can even choose one or a few orbitals to keep by indicting them using a 3-element `Array` of the Cartesian representation:
```@repl 2
bf3 = genBasisFunc([1.0,0,0], [gf1, gf2], [1,0,0])

bf4 = genBasisFunc([1.0,0,0], [gf1, gf2], [[1,0,0], [0,0,1]])
```

Again, if you want a faster solution, you can also directly define the 2 `GaussFunc` parameter(s) in a 2-element `Tuple` as the 2nd argument for `genBasisFunc`:
```@repl 2
bf5 = genBasisFunc([1.0,0,0], ([2.0, 2.5], [1.0, 0.75]), [[1,0,0], [0,0,1]])

hasEqual(bf4, bf5)
```

### Constructing basis sets based on `ParamBox`
Sometimes you may want the parameters of basis functions (or `GaussFunc`) to be under some constrains (which can be crucial for the later basis set optimization), this is when you need a deeper level of control over the parameters, through its direct container: `ParamBox`. In fact, in the above example we have already had an glimpse on it through the printed info in the REPL:
```@repl 2
gf1
```
the 2 fields of a `GaussFunc`, `.xpn` and `.con` are in fact `ParamBox`, and the actual value of them can be accessed through syntax `[]`:
```@repl 2
gf1.xpn 

gf1.con

gf1.xpn[] 

gf1.con[]
```

Since the data are not directly stored as `primitive type`s but rather inside `struct` `ParamBox`, this allows the direct assignment or shallow copy of them to not create new data with same values, but bindings to the original objects:
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

Based on such trait in Julia, you can, for instance, create a basis set that enforces all the `GaussFunc`s have the **identical** parameters:
```@repl 2
gf4 = GaussFunc(2.5, 0.5)

bs7 = genBasisFunc.([rand(3) for _=1:2], Ref(gf4))

uniqueParams!(bs7)
```

`uniqueParams!` marks all the parameters of the given basis set and 
return the unique parameters. As you can see, even though `bs7` has 
2 `GaussFunc`s as basis functions, but over all it only has 1 unique coefficient exponent ``\alpha_1`` and 1 unique contraction coefficient ``d_1``.


## Dependent Variable as Parameter

Another control the user can have on the parameters in Quiqbox is to not only store the each unique parameter as an independent variable, but also as a dependent variable, i.e., a math function of some more primitive independent variable:
```@repl 2
pb1 = gf4.xpn

pb1.map
```

The `map` field of a `ParamBox` stores a `RefValue{<:Function}`, referencing the `Function` that maps the actual stored value to another value through math operations (``R \to R``). The output value can be access through syntax `()`. In default the variable is mapped to itself:
```@repl 2
pb1[] == pb1()
```

Since `ParamBox` is a `mutable struct` you can redefine your own mapping `Functions` for the parameters; thus gain another layer of control over the basis set parameters:
```@repl 2
squareXpn(x) = x^2

pb1.map = Ref(squareXpn)

pb1[] = 3

pb1()
```

You can get a clearer view of the mapping relations in a `ParamBox` using `getVar`
```@repl 2
getVar(pb1, includeMapping=true)
```