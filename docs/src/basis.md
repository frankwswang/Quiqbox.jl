# Basis Sets

The procedure of constructing a Gaussian-type basis set can fundamentally break down into several basic steps: first, make primitive Gaussian-type orbitals (GTO) using a set of parameters, then construct the basis functions from the linear combinations of those orbitals, finally build the basis set.

The data structures defined by Quiqbox in each step, form levels of data complexity. They can be summarized in the following table.

| level  | objective | container type  | type examples|
| :---:  |   :---:   |      :---:      |     :---:    |
| 4 | basis set | `Array`, `Tuple`, `GTBasis` | `Vector{<:BasisFunc{Float64, 3}}`, `GTBasis{Float64, 3, 2}`...|
| 3 | basis function | `GTBasisFuncs` | `BasisFunc{Float64, 3, 0, 6}`, `Quiqbox.BasisFuncMix{Float64, 3, 2}`...|
| 2 | Gaussian-type function | `AbstractGaussFunc` | `GaussFunc{Float64, FLevel{0}, FLevel{0}}`...|
| 1 | tunable parameter | `ParamBox`, `SpatialPoint` | `ParamBox{Float64, :α, FLevel{0}}`, `SpatialPoint{Float64, 3, P3D{Float64, 0, 0, 0}}`... |

Depending on how much control the user wants to have over each step, Quiqbox defines several [methods](https://docs.julialang.org/en/v1/manual/methods/) of related functions to provide the freedom of balancing between efficiency and customizability.

Below are some examples from the simplest way to more flexible ways of constructing a basis set in Quiqbox. Hopefully, these use cases can also work as inspirations for more creative ways to customize basis sets.

## Basis set construction

### Constructing basis sets from existing basis sets

First, you can construct an atomic basis set at one coordinate by inputting its center coordinate, the basis set name and the corresponding atom symbol.
```jldoctest myLabel1; setup = :( push!(LOAD_PATH,"../../src/"); using Quiqbox )
julia> bsO = Quiqbox.genBasisFunc(fill(1.0, 3), "STO-3G", "O");

julia> bsO[begin]
BasisFunc{Float64, 3, 0, 3, P3D{Float64, 0, 0, 0}}(center, gauss, l, normalizeGTO, param)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bsO[end]
BasisFuncs{Float64, 3, 1, 3, P3D{Float64, 0, 0, 0}, 3}(center, gauss, l, normalizeGTO, param)[3/3][1.0, 1.0, 1.0]
```

Notice that in the returned `bsO` there are two types of elements: `BasisFunc` and `BasisFuncs`. `BasisFunc` is the most basic `DataType` to hold the data of a basis function; `BasisFuncs` is very similar except it may hold multiple orbitals with only the spherical harmonics ``Y_{ml}`` being different when the orbital angular momentum ``l>0``.

!!! info "Unit system"
    Hartree atomic units are the unit system used in Quiqbox.

If you want to postpone the specification of the center, you can replace the first argument with `missing`, then use the function `assignCenInVal!` to assign the coordinates later.
```jldoctest myLabel1
julia> bsO = genBasisFunc(missing, "STO-3G", "O");

julia> [assignCenInVal!(b, fill(1.0, 3)) for b in bsO]
3-element Vector{SpatialPoint{Float64, 3, Tuple{ParamBox{Float64, :X, FLevel{0}}, ParamBox{Float64, :Y, FLevel{0}}, ParamBox{Float64, :Z, FLevel{0}}}}}:
 SpatialPoint{Float64, 3, P3D{Float64, 0, 0, 0}}(param)[1.0, 1.0, 1.0][∂][∂][∂]
 SpatialPoint{Float64, 3, P3D{Float64, 0, 0, 0}}(param)[1.0, 1.0, 1.0][∂][∂][∂]
 SpatialPoint{Float64, 3, P3D{Float64, 0, 0, 0}}(param)[1.0, 1.0, 1.0][∂][∂][∂]
```

If you omit the atom in the arguments, `"H"` will be set in default. Notice that even though there's only one single basis function in H's STO-3G basis set, the returned value is still a `Vector`.
```jldoctest myLabel1
julia> bsH_1 = genBasisFunc([-0.5, 0, 0], "STO-3G")
1-element Vector{BasisFunc{Float64, 3, 0, 3, Tuple{ParamBox{Float64, :X, FLevel{0}}, ParamBox{Float64, :Y, FLevel{0}}, ParamBox{Float64, :Z, FLevel{0}}}}}:
 BasisFunc{Float64, 3, 0, 3, P3D{Float64, 0, 0, 0}}(center, gauss, l, normalizeGTO, param)[X⁰Y⁰Z⁰][-0.5, 0.0, 0.0]

julia> bsH_2 = genBasisFunc([ 0.5, 0, 0], "STO-3G")
1-element Vector{BasisFunc{Float64, 3, 0, 3, Tuple{ParamBox{Float64, :X, FLevel{0}}, ParamBox{Float64, :Y, FLevel{0}}, ParamBox{Float64, :Z, FLevel{0}}}}}:
 BasisFunc{Float64, 3, 0, 3, P3D{Float64, 0, 0, 0}}(center, gauss, l, normalizeGTO, param)[X⁰Y⁰Z⁰][0.5, 0.0, 0.0]
```

Finally, you can use Quiqbox's included tool function [`flatten`](@ref) to merge the three atomic basis sets into one molecular basis set:
```jldoctest myLabel1
julia> bsH2O = [bsO, bsH_1, bsH_2] |> flatten;
```

Not simple enough? Here's a more compact way of realizing the above steps if you are familiar with Julia's [vectorization syntactic sugars](https://docs.julialang.org/en/v1/manual/mathematical-operations/#man-dot-operators):
```jldoctest myLabel1
julia> cens = [fill(1.0,3), [-0.5,0,0], [0.5,0,0]];

julia> bsH2O_2 = genBasisFunc.(cens, "STO-3G", ["O", "H", "H"]) |> flatten;
```

In quiqbox, the user can often deal with several multi-layer containers (mainly `struct`s). It might be easy to get lost or unsure about whether the objects are created as intended. Quiqbox provides another tool function [`hasEqual`](@ref) that lets you verify if two objects hold the same-valued data and have the same structure. For example, if we want to see whether `bsH2O_2` created in the faster way is the same (not necessarily identical) as `bsH2O`, we can do as follows:
```jldoctest myLabel1
julia> hasEqual(bsH2O, bsH2O_2)
true
```

If the basis set you want to use is not pre-stored in Quiqbox, you can use `genBFuncsFromText` to generate the basis set from a **Gaussian** format `String`:
```@setup 1
    push!(LOAD_PATH,"../../src/")
    using Quiqbox
    bf8 = genBasisFunc([1.0, 0.0, 1.0], (2.0, 4.0))
```
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

genBFuncsFromText(txt_Kr_631G, adjustContent=true);
```

### Constructing basis sets from `GaussFunc`

If you want to specify the parameters of each basis function when constructing a basis set, you can first construct the container for primitive GTO: `GaussFunc`, and then construct the basis function from them:
```jldoctest myLabel2; setup = :( push!(LOAD_PATH,"../../src/"); using Quiqbox )
julia> gf1 = GaussFunc(2.0, 1.0)
GaussFunc{Float64, FLevel{0}, FLevel{0}}(xpn()=2.0, con()=1.0, param)

julia> gf2 = GaussFunc(2.5, 0.75)
GaussFunc{Float64, FLevel{0}, FLevel{0}}(xpn()=2.5, con()=0.75, param)

julia> bf1 = genBasisFunc([1.0, 0, 0], [gf1, gf2])
BasisFunc{Float64, 3, 0, 2, P3D{Float64, 0, 0, 0}}(center, gauss, l, normalizeGTO, param)[X⁰Y⁰Z⁰][1.0, 0.0, 0.0]
```

Unlike `BasisFunc`, there's no additional constructor function for `GaussFunc`. As for the method of `genBasisFunc` in this case, the subshell is set to "S" as the default option since the third argument is omitted. You can construct a `BasisFuncs` which contains all the orbitals within one specified subshell:
```jldoctest myLabel2
julia> bf2 = genBasisFunc([1.0, 0, 0], [gf1, gf2], "P")
BasisFuncs{Float64, 3, 1, 2, P3D{Float64, 0, 0, 0}, 3}(center, gauss, l, normalizeGTO, param)[3/3][1.0, 0.0, 0.0]
```

You can even select one or few orbitals to keep by specifying the corresponding orbital angular momentums in the Cartesian representation using `NTuple{3, Int}`:
```jldoctest myLabel2
julia> bf3 = genBasisFunc([1.0, 0, 0], [gf1, gf2], (1,0,0))
BasisFunc{Float64, 3, 1, 2, P3D{Float64, 0, 0, 0}}(center, gauss, l, normalizeGTO, param)[X¹Y⁰Z⁰][1.0, 0.0, 0.0]

julia> bf4 = genBasisFunc([1.0, 0, 0], [gf1, gf2], [(1,0,0), (0,0,1)])
BasisFuncs{Float64, 3, 1, 2, P3D{Float64, 0, 0, 0}, 2}(center, gauss, l, normalizeGTO, param)[2/3][1.0, 0.0, 0.0]
```

Again, if you want a faster solution, you can directly define the exponent coefficients and the contraction coefficients separately in a 2-element `Tuple` as the second argument of `genBasisFunc`:
```jldoctest myLabel2
julia> bf5 = genBasisFunc([1.0, 0, 0], ([2.0, 2.5], [1.0, 0.75]), [(1,0,0), (0,0,1)]);

julia> hasEqual(bf4, bf5)
true
```

### Constructing basis sets based on `ParamBox`

Sometimes you may want the parameters of basis functions (or `GaussFunc`) to be under some constraints (which can be crucial for the later basis set optimization), this is when you need a deeper level of control over the parameters, through its direct container: [`ParamBox`](@ref). In fact, in the above example, we have already had a glimpse of it through the printed info in the REPL:
```jldoctest myLabel2
julia> gf1
GaussFunc{Float64, FLevel{0}, FLevel{0}}(xpn()=2.0, con()=1.0, param)
```
The two fields of a `GaussFunc`, `.xpn`, and `.con` are `ParamBox`, and their input value (stored data) can be accessed through syntax `[]`:
```jldoctest myLabel2
julia> gf1.xpn
ParamBox{Float64, :α, FLevel{0}}(2.0)[∂][α]

julia> gf1.con
ParamBox{Float64, :d, FLevel{0}}(1.0)[∂][d]

julia> gf1.xpn[]
2.0

julia> gf1.con[]
1.0
```

Since the data are not directly stored in primitive types but rather inside `ParamBox`, this allows the shallow copy of a `ParamBox` to share the same underlying data: 
```jldoctest myLabel2
julia> gf3 = GaussFunc(1.1, 1.0);

julia> gf3_2 = gf3; # Direct assignment

julia> bf6 = genBasisFunc([1.0, 0, 0], fill(gf3, 2)); # Shallow copy is performed when using `fill`

julia> bf6.gauss[1].xpn[] *= 2;

julia> gf3_2.xpn[] == gf3.xpn[] == bf6.gauss[2].xpn[] == 2.2
true
```

Based on such feature of `ParamBox`, the user can, for instance, create a basis set that enforces all the `GaussFunc`s to have **identical** gaussian function parameters:
```jldoctest myLabel2
julia> gf4 = GaussFunc(2.5, 0.5);

julia> bs7 = genBasisFunc.([[0.0, 0.1, 0.0], [1.4, 0.3, 0.0]], Ref(gf4));

julia> markParams!(bs7)
10-element Vector{ParamBox{Float64, V, FLevel{0}} where V}:
 ParamBox{Float64, :X, FLevel{0}}(0.0)[∂][X₁]
 ParamBox{Float64, :Y, FLevel{0}}(0.1)[∂][Y₁]
 ParamBox{Float64, :Z, FLevel{0}}(0.0)[∂][Z₁]
 ParamBox{Float64, :α, FLevel{0}}(2.5)[∂][α₁]
 ParamBox{Float64, :d, FLevel{0}}(0.5)[∂][d₁]
 ParamBox{Float64, :X, FLevel{0}}(1.4)[∂][X₂]
 ParamBox{Float64, :Y, FLevel{0}}(0.3)[∂][Y₂]
 ParamBox{Float64, :Z, FLevel{0}}(0.0)[∂][Z₂]
 ParamBox{Float64, :α, FLevel{0}}(2.5)[∂][α₁]
 ParamBox{Float64, :d, FLevel{0}}(0.5)[∂][d₁]
```

`markParams!` marks all the parameters of a given basis set. Even though `bs7` has two `GaussFunc`s as basis functions, overall it only has one unique coefficient exponent ``\alpha_1`` and one unique contraction coefficient ``d_1`` besides the center coordinates.

## Dependent variable as a parameter

Another control the user have on the parameters in Quiqbox is making a `ParamBox` represent a variable equal to the returned value of a mapping function taking the stored data as the argument. In other words, the data stored in the `ParamBox` is an "input variable", while the represented variable is the "output variable".

Such a mapping function is stored in the `map` field of the `ParamBox` (which normally is an ``R \to R`` mapping). The "output value" can be accessed through syntax `()`. In default, the input variable is mapped to itself:
```jldoctest myLabel2
julia> pb1 = gf4.xpn
ParamBox{Float64, :α, FLevel{0}}(2.5)[∂][α₁]

julia> pb1.map
itself (generic function with 1 method)

julia> pb1[] == pb1()
true
```

You can get a clearer view of the variable value(s) in a `ParamBox` using `getVarDict`
```jldoctest myLabel2
julia> getVarDict(pb1)
Dict{Symbol, Float64} with 1 entry:
  :α₁ => 2.5
```

!!! info "Parameter represented by `ParamBox`"
    The output variable of a `ParamBox` is always used in the construction of any basis function component. If you want to optimize the input variable when the mapping is nontrivial (i.e. not [`Quiqbox.itself`](@ref)), the `ParamBox` needs to be marked as "differentiable". For more information on parameter optimization, please see the docstring of [`optimizeParams!`](@ref) and section [Parameter Optimization](@ref).

## Linear combinations of basis functions

Apart from the flexible control of basis function parameters, another major feature of Quiqbox is the ability to construct a basis function from the linear combination of other basis functions. Specifically, additional methods of `+` and `*` (operator syntax for [`add`](@ref) and [`mul`](@ref)) are implemented for `CompositeGTBasisFuncs`:
```jldoctest myLabel3; setup = :( push!(LOAD_PATH,"../../src/"); using Quiqbox )
julia> bf7 = genBasisFunc([1.0, 0.0, 1.0], (1.5, 3.0))
BasisFunc{Float64, 3, 0, 1, P3D{Float64, 0, 0, 0}}(center, gauss, l, normalizeGTO, param)[X⁰Y⁰Z⁰][1.0, 0.0, 1.0]

julia> bf8 = genBasisFunc([1.0, 0.0, 1.0], (2.0, 4.0))
BasisFunc{Float64, 3, 0, 1, P3D{Float64, 0, 0, 0}}(center, gauss, l, normalizeGTO, param)[X⁰Y⁰Z⁰][1.0, 0.0, 1.0]

julia> bf9 = bf7*0.5 + bf8
BasisFunc{Float64, 3, 0, 2, P3D{Float64, 0, 0, 0}}(center, gauss, l, normalizeGTO, param)[X⁰Y⁰Z⁰][1.0, 0.0, 1.0]

julia> bf9.gauss[1].con() == 3 * 0.5
true
```

As we can see, the type of `bf9` is still `BasisFunc`, hence all the GTO inside `bf9` have the same center coordinates as well. This is because `bf7` and `bf8` have the same center coordinates. What if the combined basis functions are multi-center?
```@repl 1
bf10 = genBasisFunc(fill(1.0, 3), (1.2, 3.0))

bf11 = bf8 + bf10
```
The type of `bf11` is called [`Quiqbox.BasisFuncMix`](@ref), which means it cannot be expressed as a contracted Gaussian-type orbital (CGTO), but rather a "mixture" of multi-center GTOs (MCGTO).

There are other cases that can result in a `BasisFuncMix` as the returned object. For example:
```@repl 1
bf12 = genBasisFunc(fill(1.0, 3), (1.2, 3.0), (1,1,0))

bf10 + bf12
```

In Quiqbox, `BasisFuncMix` is also accepted as a valid basis function and the user can use it to call functions that accept `CompositeGTBasisFuncs` as input argument(s):
```@repl 1
overlap(bf11, bf11)
```