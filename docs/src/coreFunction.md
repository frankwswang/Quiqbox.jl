# Core Functions

```@docs
genBasisFunc
```

```@docs
centerOf
```

```@doc
GTBasis(basis::Vector{<:Quiqbox.AbstractFloatingGTBasisFunc})
```

```@docs
decomposeBasisFunc
```

```@docs
basisSize
```

```@docs
genBasisFuncText
```

```@docs
genBFuncsFromText
```

```@docs
assignCenter!
```

```@docs
uniqueParams!
```

```@docs
getVar
```

```@docs
getVars
```

```@docs
expressionOf
```


```@docs
GridBox(nGridPerEdge::Int, spacing::Real=10, 
        centerCoord::Array{<:Real, 1}=[0.0,0.0,0.0]; 
        canDiff::Bool=true, index::Int=0)
```

```@docs
gridPoint
```


```@docs
runHF
```

```@docs
runHFcore
```


```@docs
Molecule(basis::Array{<:Quiqbox.FloatingGTBasisFunc, 1}, nuc::Array{String, 1}, 
         nucCoords::Array{<:AbstractArray, 1}, HFfVars::Quiqbox.HFfinalVars)
```

```@docs
getMolOrbitals
```

```@docs
nnRepulsions
```


```@docs
optimizeParams!
```

```@docs
updateParams!
```

```@docs
gradDescent!
```


```@docs
overlap
```

```@docs
overlaps
```

```@docs
nucAttraction
```

```@docs
nucAttractions
```

```@docs
elecKinetic
```

```@docs
elecKinetics
```

```@docs
coreHij
```

```@docs
coreH
```

```@docs
eeInteraction
```

```@docs
eeInteractions
```

```@docs
Quiqbox.oneBodyBFTensor
```

```@docs
Quiqbox.twoBodyBFTensor
```