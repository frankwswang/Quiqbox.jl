# Molden

Quiqbox supports outputting the information of canonical spin-orbitals to [**Molden**](https://www3.cmbi.umcn.nl/molden/) file format.

```@autodocs
Modules = [Quiqbox.Molden]
Pages   = ["SubModule/Molden.jl"]
Order   = [:function]
```

!!! compat "Supported basis set type"
    Due to the limitation of Molden format, only the basis sets that contain solely `FloatingGTBasisFuncs{<:Any, D, 𝑙, <:Any, <:Any, ON} where ON` such that each `ON` is equal to its maximal value. In other words, Only the basis sets built from full-subshell `FloatingGTBasisFuncs` are supported. Furthermore, the field `.normalizeGTO` for every inside basis function must all be `true` to avoid potential normalization issue.

A concrete example of the above function can be found [here](https://github.com/frankwswang/Quiqbox.jl/tree/main/examples).