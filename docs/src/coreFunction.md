# Core Functions

```@docs
genBasisFunc
```

```@docs
centerOf
```

```@doc
GTBasis(basis::Vector{<:Quiqbox.AbstractGTBasisFuncs})
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
getParams
```

```@doc
dataCopy
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
GridBox(nGridPerEdge::Int, spacing::Real=10, centerCoord::Vector{<:Real}=[0.0,0.0,0.0];
        canDiff::Bool=true, index::Int=0)
```

```@docs
makeCenter
```

```@docs
gridCoords(gb::GridBox)
```


```@docs
runHF
```

```@docs
runHFcore
```


```@docs
Molecule(basis::Vector{<:Quiqbox.FloatingGTBasisFuncs}, nuc::Vector{String}, 
         nucCoords::Vector{<:AbstractArray}, HFfVars::Quiqbox.HFfinalVars)
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
Quiqbox.eeInteractionsCore(BSet::Vector{<:Quiqbox.AbstractGTBasisFuncs}; 
                           outputUniqueIndices::Bool=false)
```

```@docs
Quiqbox.oneBodyBFTensor
```

```@docs
Quiqbox.twoBodyBFTensor
```