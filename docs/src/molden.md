# Molden

Quiqbox supports outputting the information of canonical spin-orbitals to *Molden* file format.

```@autodocs
Modules = [Quiqbox.Molden]
Pages   = ["SubModule/Molden.jl"]
Order   = [:function]
```

!!! compat "Supported basis set type"
    *Molden* format only supports the basis sets that contain solely the basis`::FloatingGTBasisFuncs{<:Any, 3, <:Any, <:Any, <:Any, ON}` where `ON` is equal to its maximal value. In other words, `makeMoldenFile` only supports [`MatterByHF`](@ref) whose `.basis.basis` are full-subshell `FloatingGTBasisFuncs`. Furthermore, the field `.normalizeGTO` for every inside basis function must all be `true` to avoid potential normalization issues.

An example of `makeMoldenFile` can be found [here](https://github.com/frankwswang/Quiqbox.jl/blob/main/examples/Jmol.jl).