export GaussFunc, genExponent, genContraction, SpatialPoint, genSpatialPoint, coordOf, 
       BasisFunc, BasisFuncs, genBasisFunc, lOf, subshellOf, centerOf, centerCoordOf, 
       unpackBasis, gaussCoeffOf, dimOf, GTBasis, sortBasisFuncs, sortPermBasisFuncs, 
       sortBasis, sortPermBasis, add, mergeBasisFuncsIn, mul, shift, decompose, 
       orbitalNumOf, genBasisFuncText, genBFuncsFromText, assignCenInVal!, getParams, 
       copyBasis, markParams!, hasNormFactor, getNormFactor, absorbNormFactor, 
       normalizeBasis

export P1D, P2D, P3D

using LinearAlgebra: diag
using ForwardDiff: derivative as ForwardDerivative

"""

    GaussFunc{T, FL1, FL2} <: AbstractGaussFunc{T}

A contracted primitive Gaussian-type function.

≡≡≡ Field(s) ≡≡≡

`xpn::ParamBox{T, :$(xpnSym), FL1}`：The exponent coefficient.

`con::ParamBox{T, :$(conSym), FL2}`: The contraction coefficient.

`param::Tuple{ParamBox{T, $(xpnSym)}, ParamBox{T, $(conSym)}}`: The parameter containers 
inside a `GaussFunc`.

≡≡≡ Initialization Method(s) ≡≡≡

    GaussFunc(e::Union{T, ParamBox{T}}, d::Union{T, ParamBox{T}}) where 
             {T<:AbstractFloat} -> 
    GaussFunc{T}

"""
struct GaussFunc{T, FLxpn, FLcon} <: AbstractGaussFunc{T}
    xpn::ParamBox{T, xpnSym, FLxpn}
    con::ParamBox{T, conSym, FLcon}
    param::Tuple{ParamBox{T, xpnSym, FLxpn}, ParamBox{T, conSym, FLcon}}

    GaussFunc(xpn::ParamBox{T, xpnSym, FL1}, con::ParamBox{T, conSym, FL2}) where 
             {T<:AbstractFloat, FL1, FL2} = 
    new{T, FL1, FL2}(xpn, con, (xpn, con))
end

GaussFunc(e::Union{T, ParamBox{T}}, d::Union{T, ParamBox{T}}) where {T<:AbstractFloat} = 
GaussFunc(genExponent(e), genContraction(d))


"""

    genExponent(e::T, mapFunction::Function; canDiff::Bool=true, 
                dataName::Symbol=:undef) where {T<:AbstractFloat} -> 
    ParamBox{T, :$(xpnSym)}

    genExponent(e::Array{T, 0}, mapFunction::Function; canDiff::Bool=true, 
                dataName::Symbol=:undef) where {T<:AbstractFloat} -> 
    ParamBox{T, :$(xpnSym)}

Construct a [`ParamBox`](@ref) for an exponent coefficient given a value. Keywords 
`mapFunction` and `canDiff` work the same way as in a general constructor of a `ParamBox`.
"""
genExponent(e::AbstractFloat, mapFunction::Function; 
            canDiff::Bool=true, dataName::Symbol=:undef) = 
ParamBox(Val(xpnSym), mapFunction, fill(e), genIndex(nothing), fill(canDiff), dataName)

genExponent(e::Array{<:AbstractFloat, 0}, mapFunction::Function; 
            canDiff::Bool=true, dataName::Symbol=:undef) = 
ParamBox(Val(xpnSym), mapFunction, e, genIndex(nothing), fill(canDiff), dataName)

"""

    genExponent(e::T) where {T<:AbstractFloat} -> ParamBox{T, :$(xpnSym)}

    genExponent(e::Array{T, 0}) where {T<:AbstractFloat} -> ParamBox{T, :$(xpnSym)}

"""
genExponent(e::AbstractFloat) = ParamBox(Val(xpnSym), itself, fill(e), genIndex(nothing))

genExponent(e::Array{<:AbstractFloat, 0}) = 
ParamBox(Val(xpnSym), itself, e, genIndex(nothing))

"""

    genExponent(pb::ParamBox{T}) where {T<:AbstractFloat} -> ParamBox{T, :$(xpnSym)}

Convert a [`ParamBox`](@ref) to the container of an exponent coefficient.
"""
genExponent(pb::ParamBox) = ParamBox(Val(xpnSym), pb, canDiff=fill(pb|>isDiffParam))


"""

    genContraction(c::T, mapFunction::Function; canDiff::Bool=true, 
                   dataName::Symbol=:undef) where {T<:AbstractFloat} -> 
    ParamBox{T, :$(conSym)}

    genContraction(c::Array{T, 0}, mapFunction::Function; canDiff::Bool=true, 
                   dataName::Symbol=:undef) where {T<:AbstractFloat} -> 
    ParamBox{T, :$(conSym)}

Construct a [`ParamBox`](@ref) for an contraction coefficient given a value. Keywords 
`mapFunction` and `canDiff` work the same way as in a general constructor of a `ParamBox`.
"""
genContraction(c::AbstractFloat, mapFunction::Function; 
               canDiff::Bool=true, dataName::Symbol=:undef) = 
ParamBox(Val(conSym), mapFunction, fill(c), genIndex(nothing), fill(canDiff), dataName)

genContraction(c::Array{<:AbstractFloat, 0}, mapFunction::Function; 
               canDiff::Bool=true, dataName::Symbol=:undef) = 
ParamBox(Val(conSym), mapFunction, c, genIndex(nothing), fill(canDiff), dataName)

"""

    genContraction(c::T) where {T<:AbstractFloat} -> ParamBox{T, :$(conSym)}

    genContraction(c::Array{T, 0}) where {T<:AbstractFloat} -> ParamBox{T, :$(conSym)}

"""
genContraction(c::AbstractFloat) = ParamBox(Val(conSym), itself, fill(c), genIndex(nothing))

genContraction(c::Array{<:AbstractFloat, 0}) = 
ParamBox(Val(conSym), itself, c, genIndex(nothing))

"""

    genContraction(pb::ParamBox{T}) where {T<:AbstractFloat} -> ParamBox{T, :$(conSym)}

Convert a [`ParamBox`](@ref) to an exponent coefficient parameter.
"""
genContraction(pb) = ParamBox(Val(conSym), pb, canDiff=fill(pb|>isDiffParam))


const P1D{T, Lx} = Tuple{ParamBox{T, cxSym, FLevel{Lx}}}
const P2D{T, Lx, Ly} = Tuple{ParamBox{T, cxSym, FLevel{Lx}}, 
                             ParamBox{T, cySym, FLevel{Ly}}}
const P3D{T, Lx, Ly, Lz} = Tuple{ParamBox{T, cxSym, FLevel{Lx}}, 
                                 ParamBox{T, cySym, FLevel{Ly}}, 
                                 ParamBox{T, czSym, FLevel{Lz}}}

const SPointT{T} = Union{P1D{T}, P2D{T}, P3D{T}}

"""

    SpatialPoint{T, D, PT} <: AbstractSpatialPoint{T, D}

A `D`-dimensional spatial point.

≡≡≡ Field(s) ≡≡≡

`param::PT`: A `Tuple` of [`ParamBox`](@ref)s as the components of the spatial coordinate.

≡≡≡ Initialization Method(s) ≡≡≡

    SpatialPoint(pbs::$(SPointT)) -> SpatialPoint
"""
struct SpatialPoint{T, D, PT} <: AbstractSpatialPoint{T, D}
    param::PT
    SpatialPoint(pbs::SPointT{T}) where {T} = new{T, length(pbs), typeof(pbs)}(pbs)
end

"""

    genSpatialPoint(point::Union{Tuple{Vararg{AbstractFloat}}, 
                                       AbstractVector{<:AbstractFloat}}, 
                    mapFunction::F=itself; canDiff::Bool=true, dataName::Symbol=:undef) -> 
    SpatialPoint

    genSpatialPoint(point::Union{Tuple{Vararg{Array{<:AbstractFloat, 0}}}, 
                                 AbstractVector{<:Array{<:AbstractFloat, 0}}}, 
                    mapFunction::F=itself; canDiff::Bool=true, dataName::Symbol=:undef) -> 
    SpatialPoint

The constructor of [`SpatialPoint`](@ref). Keywords `mapFunction` and `canDiff` work the 
same way as in a general constructor of a `ParamBox`.
≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> v1 = [1.0, 2.0, 3.0];

julia> genSpatialPoint(v1)
$( SpatialPoint(ParamBox.((1.0, 2.0, 3.0), SpatialParamSyms)) )

julia> v2 = [fill(1.0), 2.0, 3.0];

julia> p2 = genSpatialPoint(v2); p2[1]
ParamBox{Float64, :X, $(FI)}(1.0)[∂][X]

julia> v2[1][] = 1.2
1.2

julia> p2[1]
ParamBox{Float64, :X, $(FI)}(1.2)[∂][X]
```
"""
genSpatialPoint(v::AbstractVector, optArgs...) = genSpatialPoint(Tuple(v), optArgs...)
genSpatialPoint(v::NTuple{N, Any}, optArgs...) where {N} = 
genSpatialPoint.(v, Tuple([1:N;]), optArgs...) |> genSpatialPointCore

"""

    genSpatialPoint(comp::T, index::Int) where {T<:AbstractFloat} -> ParamBox{T}

    genSpatialPoint(comp::Array{T, 0}, index::Int) where {T<:AbstractFloat} -> ParamBox{T}

    genSpatialPoint(comp::T, index::Int, mapFunction::Function; canDiff::Bool=true, 
                    dataName::Symbol=:undef) where {T<:AbstractFloat} -> 
    ParamBox{T}

    genSpatialPoint(comp::Array{T, 0}, index::Int, mapFunction::Function; 
                    canDiff::Bool=true, dataName::Symbol=:undef) where 
                   {T<:AbstractFloat} -> 
    ParamBox{T}

    genSpatialPoint(comp::ParamBox{T}, index::Int) where {T<:AbstractFloat} -> ParamBox{T}

Return the component of a [`SpatialPoint`](@ref) given its value (or 0-D container) and 
index.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genSpatialPoint(1.2, 1)
ParamBox{Float64, :X, $(FI)}(1.2)[∂][X]

julia> pointY1 = fill(2.0);

julia> Y1 = genSpatialPoint(pointY1, 2)
ParamBox{Float64, :Y, $(FI)}(2.0)[∂][Y]

julia> pointY1[] = 1.5;

julia> Y1
ParamBox{Float64, :Y, $(FI)}(1.5)[∂][Y]
```
"""
genSpatialPoint(comp::AbstractFloat, index::Int, mapFunction::Function; 
                canDiff::Bool=true, dataName::Symbol=:undef) = 
ParamBox(Val(SpatialParamSyms[index]), mapFunction, fill(comp), 
         genIndex(nothing), fill(canDiff), dataName)

genSpatialPoint(comp::Array{<:AbstractFloat, 0}, index::Int, mapFunction::Function; 
                canDiff::Bool=true, dataName::Symbol=:undef) = 
ParamBox(Val(SpatialParamSyms[index]), mapFunction, comp, genIndex(nothing), fill(canDiff), 
         dataName)

genSpatialPoint(comp::AbstractFloat, index::Int) = 
ParamBox(Val(SpatialParamSyms[index]), itself, fill(comp), genIndex(nothing))

genSpatialPoint(comp::Array{<:AbstractFloat, 0}, index::Int) = 
ParamBox(Val(SpatialParamSyms[index]), itself, comp, genIndex(nothing))

genSpatialPoint(point::ParamBox, index::Int) = 
ParamBox(Val(SpatialParamSyms[index]), point, canDiff=fill(point|>isDiffParam))

"""

    genSpatialPoint(point::Union{Tuple{Vararg{ParamBox}}, AbstractVector{<:ParamBox}}) -> 
    SpatialPoint

Convert a collection of [`ParamBox`](@ref)s to a [`SpatialPoint`](@ref).
"""
genSpatialPoint(point::NTuple{N, ParamBox}) where {N} = 
ParamBox.(Val.(SpatialParamSyms[1:N]), point) |> genSpatialPointCore

genSpatialPointCore(point::Union{P1D, P2D, P3D}) = SpatialPoint(point)


"""

    coordOf(sp::SpatialPoint{T}) where {T} -> Vector{T}

Get the coordinate represented by the input [`SpatialPoint`](@ref).
"""
coordOf(sp::SpatialPoint{T}) where {T} = T[outValOf(i) for i in sp.param]


"""

    BasisFunc{T, D, 𝑙, GN, PT} <: FloatingGTBasisFuncs{T, D, 𝑙, GN, PT, 1}

A (floating) Gaussian-type basis function with its center assigned to a defined coordinate.

≡≡≡ Field(s) ≡≡≡

`center::SpatialPoint{T, D, PT}`: The `D`-dimensional center.

`gauss::NTuple{N, GaussFunc{T, <:Any}}`: Gaussian functions within the basis function.

`l::Tuple{LTuple{D, 𝑙}}`: Cartesian representation of the angular momentum. E.g., 
`$(LTuple(1, 0, 0))` (X¹Y⁰Z⁰) would correspond to an specific angular momentum 
configuration where the sum of all the components is `𝑙=1`.

`normalizeGTO::Bool`: Whether each `GaussFunc` inside will be normalized in calculations.

`param::NTuple{D+GN*2, ParamBox}`： All the tunable parameters stored in the basis function.

≡≡≡ Initialization Method(s) ≡≡≡

    BasisFunc(cen::SpatialPoint{T, D, PT}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
              l::Union{Tuple{LTuple{D, 𝑙}}, LTuple{D, 𝑙}}, normalizeGTO::Bool) where 
             {T, D, PT, 𝑙, GN} -> 
    BasisFunc{T, D, 𝑙, GN, PT}

    BasisFunc(cen::SpatialPoint{T, D, PT}, gs::AbstractGaussFunc{T}, 
              l::Union{Tuple{LTuple{D, 𝑙}}, LTuple{D, 𝑙}}, normalizeGTO::Bool) where 
             {T, D, PT, 𝑙, GN} -> 
    BasisFunc{T, D, 𝑙, 1, PT}
"""
struct BasisFunc{T, D, 𝑙, GN, PT} <: FGTBasisFuncs1O{T, D, 𝑙, GN, PT}
    center::SpatialPoint{T, D, PT}
    gauss::NTuple{GN, AbstractGaussFunc{T}}
    l::Tuple{LTuple{D, 𝑙}}
    normalizeGTO::Bool
    param::Tuple{Vararg{ParamBox{T}}}

    function BasisFunc(cen::SpatialPoint{T, D, PT}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
                       l::Tuple{LTuple{D, 𝑙}}, normalizeGTO::Bool) where {T, D, PT, 𝑙, GN}
        pars = joinTuple(cen.param, getproperty.(gs, :param)...)
        new{T, D, 𝑙, GN, PT}(cen, gs, l, normalizeGTO, pars)
    end
end

BasisFunc(cen, gs::NTuple{GN, AbstractGaussFunc{T}}, l::LTuple, normalizeGTO=false) where 
         {GN, T} = 
BasisFunc(cen, gs, (l,), normalizeGTO)

BasisFunc(cen, g::AbstractGaussFunc, l, normalizeGTO=false) = 
BasisFunc(cen, (g,), l, normalizeGTO)

BasisFunc(bf::BasisFunc) = itself(bf)


"""

    BasisFuncs{T, D, 𝑙, GN, PT, ON} <: FloatingGTBasisFuncs{T, D, 𝑙, GN, PT, ON}

A collection of basis functions with identical parameters except having different 
orientations within a specified subshell (i.e. same total orbital angular momentum). It has 
the same fields as `BasisFunc`. Specifically, for `l`, its size `ON` can be no less than 1 
and no larger than the size of the corresponding subshell.
"""
struct BasisFuncs{T, D, 𝑙, GN, PT, ON} <: FloatingGTBasisFuncs{T, D, 𝑙, GN, PT, ON}
    center::SpatialPoint{T, D, PT}
    gauss::NTuple{GN, AbstractGaussFunc{T}}
    l::NTuple{ON, LTuple{D, 𝑙}}
    normalizeGTO::Bool
    param::Tuple{Vararg{ParamBox{T}}}

    function BasisFuncs(cen::SpatialPoint{T, D, PT}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
                        ls::NTuple{ON, LTuple{D, 𝑙}}, normalizeGTO::Bool=false) where 
                       {T, D, PT, 𝑙, GN, ON}
        ss = SubshellXYZsizes[𝑙+1]
        @assert ON <= ss "The total number of `l` should be no more than $(ss) as " * 
                         "they are in $(subshell) subshell."
        ls = sort!(collect(ls), rev=true) |> Tuple
        pars = joinTuple(cen.param, getproperty.(gs, :param)...)
        new{T, D, 𝑙, GN, PT, ON}(cen, gs, ls, normalizeGTO, pars)
    end
end

const BFuncs1O{T, D, 𝑙, GN, PT} = BasisFuncs{T, D, 𝑙, GN, PT, 1}
const BFuncsON{ON} = BasisFuncs{<:Any, <:Any, <:Any, <:Any, <:Any, ON}

BasisFuncs(cen, g::AbstractGaussFunc, ls, normalizeGTO=false) = 
BasisFuncs(cen, (g,), ls, normalizeGTO)

BasisFuncs(bfs::BasisFuncs) = itself(bfs)

BasisFunc(bfs::BFuncs1O) = BasisFunc(bfs.center, bfs.gauss, bfs.l, bfs.normalizeGTO)


struct EmptyBasisFunc{T<:Real, D} <: CGTBasisFuncs1O{T, D, 0} end


isaFullShellBasisFuncs(::Any) = false

isaFullShellBasisFuncs(::FloatingGTBasisFuncs{<:Any, D, 𝑙, <:Any, <:Any, ON}) where 
                      {D, 𝑙, ON} = 
(ON == SubshellSizes[D][𝑙+1])

isaFullShellBasisFuncs(::FloatingGTBasisFuncs{<:Any, <:Any, 0}) = true


const Doc_genBasisFunc_eg1 = "BasisFunc{Float64, 3, 1, 1, P3D{Float64, 0, 0, 0}}(center, "*
                             "gauss, l, normalizeGTO, param)[X⁰Y¹Z⁰][0.0, 0.0, 0.0]"

const Doc_genBasisFunc_eg2 = "BasisFuncs{Float64, 3, 1, 1, P3D{Float64, 0, 0, 0}, 3}"*
                             "(center, gauss, l, normalizeGTO, param)[3/3][0.0, 0.0, 0.0]"

const Doc_genBasisFunc_eg3 = "BasisFuncs{Float64, 3, 1, 1, P3D{Float64, 0, 0, 0}, 2}"*
                             "(center, gauss, l, normalizeGTO, param)[2/3][0.0, 0.0, 0.0]"

"""

    genBasisFunc(center::Union{AbstractVector{T}, Tuple{Vararg{T}}, SpatialPoint, Missing}, 
                 args..., kws...) where {T<:Union{AbstractFloat, ParamBox}} -> 
    Union{FloatingGTBasisFuncs{T}, Vector{<:FloatingGTBasisFuncs{T}}}

The constructor of `FloatingGTBasisFuncs`, but it also returns different kinds of 
collections (`Vector`) of them based on the input arguments. The first argument `center` 
specifies the center coordinate of the generated `FloatingGTBasisFuncs`, and can be left as 
`missing` for later assignment.

≡≡≡ Method 1 ≡≡≡

    genBasisFunc(center, GsOrCoeffs, Ls; normalizeGTO=false) -> 
    FloatingGTBasisFuncs

=== Positional argument(s) ===

`GsOrCoeffs::Union{
    AbstractGaussFunc{T1}, 
    AbstractVector{<:AbstractGaussFunc{T1}}, 
    Tuple{Vararg{AbstractGaussFunc{T1}}}, 
    NTuple{2, T1}, 
    NTuple{2, AbstractVector{T1}}
} where {T1<:AbstractFloat}`: A collection of concentric `GaussFunc` that will be used to 
construct the basis function. To simplify the procedure, it can also be in the form of a 
`NTuple{2}` of the exponent coefficient(s)`::Union{AbstractFloat, 
AbstractVector{<:AbstractFloat}}` and contraction coefficients`::Union{AbstractFloat, 
AbstractVector{<:AbstractFloat}}` of the [`GaussFunc`](@ref)(s) to be input.

`Ls::Union{
    T2, 
    AbstractVector{T2}, 
    NTuple{<:Any, T2}
} where {T2<:Union{Tuple{Vararg{Int}}, LTuple}}`: A collection of angular momentum(s) 
within the same subshell, in the Cartesian coordinate representation. E.g., for p shell it 
can be set to `((1,0,0), (0,1,0))`. This will determine the number of spatial orbitals and 
their angular momentum respectively to be stored in the output `FloatingGTBasisFuncs`.

=== Keyword argument(s) ===

`normalizeGTO::Bool`: Determine whether the inside `GaussFunc`(s) will be normalized in the 
during the calculation. 

=== Example(s) ===

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genBasisFunc([0.,0.,0.], GaussFunc(2.,1.), (0,1,0))
$(Doc_genBasisFunc_eg1)
```

≡≡≡ Method 2 ≡≡≡

    genBasisFunc(center, GsOrCoeffs, subshell="S"; normalizeGTO=false) -> 
    FloatingGTBasisFuncs

    genBasisFunc(center, GsOrCoeffs, subshell, lFilter; normalizeGTO=false) -> 
    FloatingGTBasisFuncs

=== Positional argument(s) ===

`subshell::String`: The third argument of the constructor can also be the name of a 
subshell, which will make sure the output is a `BasisFuncs` that contains the spatial 
orbitals that fully occupy the subshell. 

`lFilter::Tuple{Vararg{Bool}}`: When this 4th argument is provided, it can determine the 
orbital(s) to be included based on the given `subshell`. The order of the corresponding 
orbital angular momentum(s) can be inspected using function `orbitalLin`.

=== Keyword argument(s) ===

`normalizeGTO::Bool`: Same as the one defined in method 1.

=== Example(s) ===

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genBasisFunc([0.,0.,0.], (2., 1.), "P")
$(Doc_genBasisFunc_eg2)

julia> genBasisFunc([0.,0.,0.], (2., 1.5), "P", (true, false, true))
$(Doc_genBasisFunc_eg3)
```

≡≡≡ Method 3 ≡≡≡

    genBasisFunc(center, BSkey, atm="H"; unlinkCenter=false) -> 
    Vector{<:FloatingGTBasisFuncs}

=== Positional argument(s) ===

`BSkey::String`: The name of an existed atomic basis set. The supported options are in 
`$(BasisSetNames)`.

`atm::String`: The name of the atom corresponding to the chosen basis set. The supported 
options are in `$(ElementNames)`.

=== Keyword argument(s) ===

`unlinkCenter::Bool`: Determine whether the centers of constructed `FloatingGTBasisFuncs` 
are linked to each other. If set to `true`, the center of each `FloatingGTBasisFuncs` is a 
`Base.deepcopy` of each other. Otherwise, they share the same underlying data so changing 
the value of one will affect others.

=== Example(s) ===

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genBasisFunc([0.,0.,0.], "6-31G");

julia> genBasisFunc([0.,0.,0.], "STO-3G", "Li");
```

≡≡≡ Method 4 ≡≡≡

    genBasisFunc(b::FloatingGTBasisFuncs{T, D}, newFieldVal) where {T, D} -> 
    FloatingGTBasisFuncs{T, D}

=== Positional argument(s) ===

`field::Union{
    SpatialPoint{T, D}, 
    Tuple{Vararg{AbstractGaussFunc{T}}}, 
    Tuple{Vararg{LTuple{D, 𝑙}}} where 𝑙, 
    Bool
} where {T<:AbstractFloat, D}`: Any one of the fields inside a `FloatingGTBasisFuncs` 
except `param`.

This method outputs a `FloatingGTBasisFuncs` that has identical fields as the input one 
except the field that can be replaced by `newFieldVal` (and `param` if the replaced field 
contains [`ParamBox`](@ref)).
"""
genBasisFunc(cen::SpatialPoint{T, D}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
             l::LTuple{D, 𝑙}=LTuple(fill(0, D)); normalizeGTO::Bool=false) where 
            {D, T, GN, 𝑙} = 
BasisFunc(cen, gs, l, normalizeGTO)

genBasisFunc(cen::SpatialPoint{T, D}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
             l::NTuple{D, Int}; normalizeGTO::Bool=false) where {T, D, GN} = 
BasisFunc(cen, gs, l|>LTuple, normalizeGTO)

genBasisFunc(cen::SpatialPoint{T, D}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
             ls::NTuple{ON, LTuple{D, 𝑙}}; normalizeGTO::Bool=false) where 
            {T, D, GN, ON, 𝑙} = 
BasisFuncs(cen, gs, ls, normalizeGTO)

genBasisFunc(cen::SpatialPoint{T, D}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
             ls::NTuple{ON, NTuple{D, Int}}; normalizeGTO::Bool=false) where 
            {T, D, GN, ON} = 
BasisFuncs(cen, gs, ls.|>LTuple, normalizeGTO)

genBasisFunc(cen::SpatialPoint{T, D}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
             ls::Vector{LTuple{D, 𝑙}}; normalizeGTO::Bool=false) where {T, D, GN, 𝑙} = 
genBasisFunc(cen, gs, ls|>Tuple; normalizeGTO)

genBasisFunc(cen::SpatialPoint{T, D}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
             ls::Vector{NTuple{D, Int}}; normalizeGTO::Bool=false) where {T, D, GN} = 
genBasisFunc(cen, gs, ls|>Tuple; normalizeGTO)

genBasisFunc(cen::SpatialPoint{T, D}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
             l::Tuple{LTuple{D, 𝑙}}; normalizeGTO::Bool=false) where {T, D, GN, 𝑙} = 
genBasisFunc(cen, gs, l[1]; normalizeGTO)

genBasisFunc(cen::SpatialPoint{T, D}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
             l::Tuple{NTuple{D, Int}}; normalizeGTO::Bool=false) where {T, D, GN} = 
genBasisFunc(cen, gs, l[1]; normalizeGTO)

function genBasisFunc(cen::SpatialPoint{T, D}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
                      subshell::String; normalizeGTO::Bool=false) where {T, D, GN}
    genBasisFunc(cen, gs, SubshellOrientationList[D][subshell]; normalizeGTO)
end

function genBasisFunc(cen::SpatialPoint{T, D}, xpnsANDcons::NTuple{2, AbstractVector{T}}, 
                      lOrSubshell=LTuple(fill(0, D)); normalizeGTO::Bool=false) where 
                     {T, D}
    @assert ==(length.(xpnsANDcons)...) "The length of exponent coefficients and " * 
                                        "contraction coefficients are NOT equal."
    genBasisFunc(cen, GaussFunc.(xpnsANDcons[1], xpnsANDcons[2]), lOrSubshell; normalizeGTO)
end

genBasisFunc(cen::SpatialPoint{T, D}, xpnANDcon::NTuple{2, T}, 
             lOrSubshell=LTuple(fill(0, D)); normalizeGTO::Bool=false) where {T, D} = 
genBasisFunc(cen, (GaussFunc(xpnANDcon[1], xpnANDcon[2]),), lOrSubshell; normalizeGTO)

genBasisFunc(cen::SpatialPoint{T, D}, gs::Tuple, subshell::String, 
             lFilter::Tuple{Vararg{Bool}}; normalizeGTO::Bool=false) where {T, D} = 
genBasisFunc(cen, gs, SubshellOrientationList[D][subshell][1:end .∈ Ref(findall(lFilter))]; 
             normalizeGTO)

function genBasisFunc(center::SpatialPoint{T, D}, BSkey::String, atm::String="H"; 
                      unlinkCenter::Bool=false) where {T, D}
    BSstr = BasisSetList[BSkey][AtomicNumberList[atm]]
    @assert BSstr!==nothing "Quiqbox DOES NOT have basis set "*BSkey*" for "*atm*"."
    genBFuncsFromText(BSstr; adjustContent=true, excludeLastNlines=1, center, unlinkCenter, 
                      normalizeGTO=true)
end

# A few methods for convenient arguments omissions and mutations.
genBasisFunc(cen::SpatialPoint{T}, gs::AbstractVector{<:AbstractGaussFunc{T}}, 
             args...; kws...) where {T} = 
genBasisFunc(cen, gs|>Tuple, args...; kws...)

genBasisFunc(cen::SpatialPoint{T}, g::GaussFunc{T}, args...; kws...) where {T} = 
genBasisFunc(cen, (g,), args...; kws...)

genBasisFunc(coord::Union{Tuple, AbstractVector}, args...; kws...) = 
genBasisFunc(genSpatialPoint(coord), args...; kws...)

genBasisFunc(::Missing, args...; kws...) = genBasisFunc((NaN, NaN, NaN), args...; kws...)

genBasisFunc(bf::FloatingGTBasisFuncs) = itself(bf)

genBasisFunc(bf::FloatingGTBasisFuncs{T, D}, cen::SpatialPoint{T, D}) where {T, D} = 
genBasisFunc(cen, bf.gauss, bf.l, normalizeGTO=bf.normalizeGTO)

genBasisFunc(bf::FloatingGTBasisFuncs{T, D}, gs::Tuple{Vararg{AbstractGaussFunc{T}}}) where 
            {T, D} = 
genBasisFunc(bf.center, gs, bf.l, normalizeGTO=bf.normalizeGTO)

genBasisFunc(bf::FloatingGTBasisFuncs{T, D}, ls::Tuple{Vararg{LTuple{D, 𝑙}}}) where 
            {T, D, 𝑙} = 
genBasisFunc(bf.center, bf.gauss, ls, normalizeGTO=bf.normalizeGTO)

genBasisFunc(bf::FloatingGTBasisFuncs{T, D}, normalizeGTO::Bool) where {T, D} = 
genBasisFunc(bf.center, bf.gauss, bf.l; normalizeGTO)


"""

    lOf(::FloatingGTBasisFuncs) -> Int

Return the total orbital angular momentum quantum number (in Cartesian coordinate 
representation).
"""
lOf(::FloatingGTBasisFuncs{<:Any, <:Any, 𝑙}) where {𝑙} = 𝑙


"""

    subshellOf(::FloatingGTBasisFuncs) -> String

Return the corresponding subshell of the input `FloatingGTBasisFuncs`.
"""
subshellOf(b::FloatingGTBasisFuncs) = SubshellNames[lOf(b)+1]


"""

    sortBasisFuncs(bs::AbstractArray{<:FloatingGTBasisFuncs{T, D}}, 
                   groupCenters::Bool=false; roundAtol::Real=getAtolVal(T)) where {T, D} -> 
    Vector

Sort `FloatingGTBasisFuncs`. If `groupCenters = true`, Then the function will return an 
`Vector{<:Vector{<:FloatingGTBasisFuncs}}` in which the elements are grouped basis 
functions with same center coordinates. `roundAtol` specifies the absolute approximation 
tolerance of comparing the center coordinates to determine whether they are treated as 
"equal"; when set to `NaN`, no approximation will be made during the comparison.
"""
@inline function sortBasisFuncs(bs::AbstractArray{<:FloatingGTBasisFuncs{T, D}}, 
                                groupCenters::Bool=false; 
                                roundAtol::Real=getAtolVal(T)) where {T, D}
    bfBlocks = map( groupedSort(reshape(bs, :), 
                    x->roundToMultiOfStep.(centerCoordOf(x), 
                                          nearestHalfOf(roundAtol))) ) do subbs
        # Reversed order within same subshell.
        sort!(subbs, by=x->[-getTypeParams(x)[3], x.l[1].tuple, getTypeParams(x)[4]], 
              rev=true)
    end
    groupCenters ? bfBlocks : vcat(bfBlocks...)
end

"""

    sortBasisFuncs(bs::Tuple{Vararg{FloatingGTBasisFuncs{T, D}}}, groupCenters::Bool=false; 
                   roundAtol::Real=getAtolVal(T)) where {T, D} -> 
    Tuple

"""
sortBasisFuncs(bs::Tuple{Vararg{FloatingGTBasisFuncs{T, D}}}, groupCenters::Bool=false; 
               roundAtol::Real=getAtolVal(T)) where {T, D} = 
sortBasisFuncs(FloatingGTBasisFuncs{T, D}[bs...], groupCenters; roundAtol) |> Tuple


"""

    sortPermBasisFuncs(bs::Union{AbstractArray{<:FloatingGTBasisFuncs{T, D}}, 
                                 Tuple{Vararg{FloatingGTBasisFuncs{T, D}}}}; 
                       roundAtol::Real=getAtolVal(T)) where {T, D} -> 
    Vector{Int}

Return a `Vector` of indices `I` such that `bs[I] == `[`sortBasisFuncs`](@ref)
`(bs; roundAtol)[I]`.
"""
sortPermBasisFuncs(bs::AbstractArray{<:FloatingGTBasisFuncs{T, D}}; 
                   roundAtol::Real=getAtolVal(T)) where {T, D} = 
sortperm(reshape(bs, :), 
         by=x->[-1*roundToMultiOfStep.(centerCoordOf(x), nearestHalfOf(roundAtol)), 
                -getTypeParams(x)[3], x.l[1].tuple, getTypeParams(x)[4]], rev=true)

sortPermBasisFuncs(bs::Tuple{Vararg{FloatingGTBasisFuncs{T, D}}}; 
                   roundAtol::Real=getAtolVal(T)) where {T, D} = 
sortPermBasisFuncs(FloatingGTBasisFuncs{T, D}[bs...]; roundAtol)


"""

    centerOf(bf::FloatingGTBasisFuncs{T, D}) where {T, D} -> SpatialPoint{T, D}

Return the center of the input `FloatingGTBasisFuncs`.
"""
centerOf(bf::FloatingGTBasisFuncs) = bf.center


"""

    centerCoordOf(bf::FloatingGTBasisFuncs{T}) where {T} -> Vector{T}

Return the center coordinate of the input `FloatingGTBasisFuncs`.
"""
centerCoordOf(bf::FloatingGTBasisFuncs) = coordOf(bf.center)


"""

    gaussCoeffOf(gf::GaussFunc{T}) -> Matrix{T}

Return the exponent and contraction coefficients of `gf`.
"""
gaussCoeffOf(gf::GaussFunc{T}) where {T} = hcat(outValOf.(gf.param)::NTuple{2, T}...)

"""

    gaussCoeffOf(b::FloatingGTBasisFuncs{T}) -> Matrix{T}

Return the exponent and contraction coefficients of each [`GaussFunc`](@ref) (in each row 
of the returned `Matrix`) inside `b`.
"""
function gaussCoeffOf(bf::FloatingGTBasisFuncs{T, <:Any, <:Any, GN}) where {T, GN}
    xpns = Array{T}(undef, GN)
    cons = Array{T}(undef, GN)
    for (i, g) in enumerate(bf.gauss)
        xpns[i] = outValOf(g.xpn)
        cons[i] = outValOf(g.con)
    end
    hcat(xpns, cons)
end


"""

    BasisFuncMix{T, D, BN, BFT<:BasisFunc{T, D}} <: CompositeGTBasisFuncs{T, D, BN, 1}

Sum of multiple `FloatingGTBasisFuncs{<:Any, <:Any, <:Any, <:Any, <:Any, 1}` without any 
reformulation, treated as one basis function in the calculations.

≡≡≡ Field(s) ≡≡≡

`BasisFunc::NTuple{BN, BFT}`: Basis functions used to sum up.

`param::Tuple{Vararg{ParamBox}}`: Contained parameters.

≡≡≡ Initialization Method(s) ≡≡≡

    BasisFuncMix(bfs::Union{Tuple{Vararg{T}}, AbstractArray{T}}) where 
                {T<:FloatingGTBasisFuncs{<:Any, <:Any, <:Any, <:Any, <:Any, 1}} -> 
    BasisFuncMix

"""
struct BasisFuncMix{T, D, BN, BFT<:BasisFunc{T, D}} <: CGTBasisFuncs1O{T, D, BN}
    BasisFunc::NTuple{BN, BFT}
    param::Tuple{Vararg{ParamBox{T}}}

    function BasisFuncMix(bfs::Tuple{Vararg{BasisFunc{T, D}, BN}}) where {T, D, BN}
        bs = sortBasisFuncs(bfs, roundAtol=NaN)
        new{T, D, BN, eltype(bfs)}(bs, joinTuple(getproperty.(bs, :param)...))
    end
end

BasisFuncMix(bfs::AbstractArray{<:BasisFunc}) = BasisFuncMix(bfs|>Tuple)
BasisFuncMix(bfs::AbstractArray{T}) where {T<:FGTBasisFuncsON{1}} = 
BasisFuncMix(BasisFunc.(bfs))
BasisFuncMix(bf::BasisFunc) = BasisFuncMix((bf,))
BasisFuncMix(bfm::BasisFuncMix) = itself(bfm)


getTypeParams(::FloatingGTBasisFuncs{T, D, 𝑙, GN, PT, ON}) where {T, D, 𝑙, GN, PT, ON} = 
(T, D, 𝑙, GN, PT, ON)
getTypeParams(::BasisFuncMix{T, D, BN, BFT}) where {T, D, BN, BFT} = (T, D, BN, BFT)


"""

    unpackBasis(b::GTBasisFuncs{T, D}) -> Vector{<:BasisFunc{T, D}}

Unpack `b` to return all the `BasisFunc` inside it.
"""
unpackBasis(::EmptyBasisFunc{T, D}) where {T, D} = BasisFunc{T, D}[]
unpackBasis(b::BasisFuncMix)  = collect(b.BasisFunc)
unpackBasis(b::FloatingGTBasisFuncs)  = [i for i in b]


"""

    dimOf(::DimensionalParamContainer) -> Int

Return the spatial dimension of the input parameterized container such as 
`AbstractSpatialPoint` and `QuiqboxBasis`.
"""
dimOf(::DimensionalParamContainer{<:Any, D}) where {D} = D


"""

    GTBasis{T, D, BN, BFT<:GTBasisFuncs{T, D, 1}} <: BasisSetData{T, D, BFT}

The container of basis set information.

≡≡≡ Field(s) ≡≡≡

`basis::NTuple{BN, BFT}`: Stored basis set.

`S::Matrix{T}`: Overlap matrix.

`Te::Matrix{T}`: Kinetic energy part of the electronic core Hamiltonian.

`eeI::Array{T, 4}`: Electron-electron interaction.

≡≡≡ Initialization Method(s) ≡≡≡

    GTBasis(basis::Union{Tuple{Vararg{GTBasisFuncs{T, D}}}, 
                         AbstractVector{<:GTBasisFuncs{T, D}}}) where {T, D} -> 
    GTBasis{T, D}

Construct a `GTBasis` given a basis set.
"""
struct GTBasis{T, D, BN, BFT<:GTBasisFuncs{T, D, 1}} <: BasisSetData{T, D, BFT}
    basis::NTuple{BN, BFT}
    S::Matrix{T}
    Te::Matrix{T}
    eeI::Array{T, 4}

    GTBasis(bfs::Tuple{Vararg{GTBasisFuncs{T, D, 1}, BN}}) where {T<:Real, D, BN} = 
    new{T, D, BN, eltype(bfs)}(bfs, overlaps(bfs), eKinetics(bfs), eeInteractions(bfs))

    function GTBasis(bs::Tuple{Vararg{GTBasisFuncs{T, D}}}) where {T, D}
        bfs = flatten(bs)
        S = overlaps(bs)
        Te = eKinetics(bs)
        eeI = eeInteractions(bs)
        new{T, D, length(bfs), eltype(bfs)}(bfs, S, Te, eeI)
    end
end

GTBasis(bs::AbstractVector{<:GTBasisFuncs{T, D}}) where {T, D} = GTBasis(bs |> Tuple)


"""

    sortBasis(bs::Union{AbstractArray{<:CompositeGTBasisFuncs{T, D}}, 
                        Tuple{Vararg{CompositeGTBasisFuncs{T, D}}}}; 
              roundAtol::Real=getAtolVal(T)) where {T, D} -> 
    Vector{<:CompositeGTBasisFuncs{T, D}}

Sort basis functions. `roundAtol` specifies the absolute approximation tolerance of 
comparing parameters stored in each `CompositeGTBasisFuncs` to determine whether they are 
treated as "equal"; when set to `NaN`, no approximation will be made during the comparison.
"""
function sortBasis(bs::AbstractArray{<:CompositeGTBasisFuncs{T, D}}; 
                   roundAtol::Real=getAtolVal(T)) where {T, D}
    bs = reshape(copy(bs), :)
    ids = findall(x->isa(x, FloatingGTBasisFuncs), bs)
    bfs = splice!(bs, ids)
    vcat( sortBasisFuncs(convert(AbstractVector{FloatingGTBasisFuncs{T, D}}, bfs); 
                         roundAtol), 
          sortBasis(convert(AbstractVector{BasisFuncMix{T, D}}, bs); roundAtol) )
end

sortBasis(bs::AbstractArray{<:BasisFuncMix{T, D}}; 
          roundAtol::Real=getAtolVal(T)) where {T, D} = 
bs[sortPermBasis(bs; roundAtol)]

sortBasis(bs::AbstractArray{<:FloatingGTBasisFuncs{T, D}}; 
          roundAtol::Real=getAtolVal(T)) where {T, D} = 
sortBasisFuncs(bs; roundAtol)

"""

    sortBasis(bs::Tuple{Vararg{CompositeGTBasisFuncs{T, D}}}; 
              roundAtol::Real=getAtolVal(T)) where {T, D} -> 
    Tuple{Vararg{CompositeGTBasisFuncs{T, D}}}

"""
sortBasis(bs::Tuple{Vararg{CompositeGTBasisFuncs{T, D}}}; 
          roundAtol::Real=getAtolVal(T)) where {T, D} = 
sortBasis(collect(bs); roundAtol) |> Tuple

"""

    sortBasis(b::GTBasis{T, D}; roundAtol::Real=getAtolVal(T)) where {T, D} -> 
    GTBasis{T, D}

Reconstruct a [`GTBasis`](@ref) by sorting the `GTBasisFuncs` stored in the input one.
"""
sortBasis(b::GTBasis{T}; roundAtol::Real=getAtolVal(T)) where {T} = 
          GTBasis(sortBasis(b.basis; roundAtol))


"""

    sortPermBasis(bs::AbstractArray{<:CompositeGTBasisFuncs{T, D}}; 
                  roundAtol::Real=getAtolVal(T)) where {T, D} -> 
    Vector{Int}

Return a `Vector` of indices `I` such that `bs[I] == `[`sortBasis`](@ref)
`(bs; roundAtol)[I]`.
"""
function sortPermBasis(bs::AbstractArray{<:CompositeGTBasisFuncs{T, D}}; 
                       roundAtol::Real=getAtolVal(T)) where {T, D}
    ids = objectid.(bs)
    bsN = sortBasis(bs; roundAtol)
    idsN = objectid.(bsN)
    indexin(idsN, ids)
end

sortPermBasis(bs::AbstractArray{<:BasisFuncMix{T, D}}; 
              roundAtol::Real=getAtolVal(T)) where {T, D} = 
sortPermBasisFuncs(getindex.(getproperty.(bs, :BasisFunc), 1); roundAtol)

sortPermBasis(bs::AbstractArray{<:FloatingGTBasisFuncs{T, D}}; 
              roundAtol::Real=getAtolVal(T)) where {T, D} = 
sortPermBasisFuncs(bs; roundAtol)

sortPermBasis(bs::Tuple{Vararg{CompositeGTBasisFuncs{T, D}}}; 
              roundAtol::Real=getAtolVal(T)) where {T, D} = 
sortPermBasis(collect(bs); roundAtol)


function sumOfCore(bfs::AbstractArray{<:BasisFunc{T, D}}; 
                   roundAtol::Real=getAtolVal(T)) where {T, D}
    arr1 = convert(Vector{BasisFunc{T, D}}, sortBasisFuncs(bfs; roundAtol))
    arr2 = BasisFunc{T, D}[]
    while length(arr1) > 1
        temp = add(arr1[1], arr1[2]; roundAtol)
        if temp isa BasisFunc
            arr1[1] = temp
            popat!(arr1, 2)
        else
            push!(arr2, popfirst!(arr1))
        end
    end
    if length(arr2) == 0
        arr1[]
    else
        vcat(arr1, arr2) |> BasisFuncMix
    end
end

sumOfCore(bs::Union{Tuple{Vararg{GTBasisFuncs{T, D, 1}}}, 
                    AbstractArray{<:GTBasisFuncs{T, D, 1}}}; 
          roundAtol::Real=getAtolVal(T)) where {T, D} = 
sumOfCore(BasisFunc{T, D}[vcat(unpackBasis.(bs)...)...]; roundAtol)

function sumOf(bs::Union{Tuple{Vararg{GTBasisFuncs{T, D, 1}}}, 
                         AbstractArray{<:GTBasisFuncs{T, D, 1}}}; 
               roundAtol::Real=getAtolVal(T)) where {T, D}
    length(bs) == 1 && (return bs[1])
    sumOfCore(bs; roundAtol)
end

function mergeGaussFuncs(gf1::GaussFunc{T}, gf2::GaussFunc{T}; 
                         roundAtol::Real=getAtolVal(T)) where {T}
    halfAtol = nearestHalfOf(roundAtol)
    xpn = if (xpn1 = gf1.xpn) === (xpn2 = gf2.xpn) || hasIdentical(xpn1, xpn2)
        xpn1
    elseif hasEqual(xpn1, xpn2)
        deepcopy(xpn1)
    else
        xpn1R = xpn1()
        xpn2R = xpn2()
        if isApprox(xpn1R, xpn2R, atol=roundAtol)
            genExponent( getNearestMid(xpn1R, xpn2R, halfAtol) )
        else
            return [gf1, gf2]
        end
    end

    res = if (con1 = gf1.con) === (con2 = gf2.con) || hasIdentical(con1, con2)
        mul(GaussFunc(xpn, con1), 2; roundAtol)
    elseif hasEqual(con1, con2)
        mul(GaussFunc(xpn, deepcopy(con1)), 2; roundAtol)
    else
        GaussFunc(xpn, genContraction(roundToMultiOfStep(con1() + con2(), halfAtol)))
    end

    [res]
end

mergeGaussFuncs(gf1::GaussFunc{T}, gf2::GaussFunc{T}, gf3::GaussFunc{T}, 
                gf4::GaussFunc{T}...; 
                roundAtol::Real=getAtolVal(T)) where {T} = 
mergeMultiObjs(GaussFunc{T}, mergeGaussFuncs, gf1, gf2, gf3, gf4...; roundAtol)


"""

    add(b1::CompositeGTBasisFuncs{T, D, <:Any, 1}, 
        b2::CompositeGTBasisFuncs{T, D, <:Any, 1}; 
        roundAtol::Real=getAtolVal(T)) where {T, D} -> 
    CompositeGTBasisFuncs{T, D, <:Any, 1}

Addition between two `CompositeGTBasisFuncs{T, D, <:Any, 1}` such as [`BasisFunc`](@ref) 
and [`BasisFuncMix`](@ref). `roundAtol` specifies the absolute approximation tolerance of 
comparing parameters stored in each `CompositeGTBasisFuncs` to determine whether they are 
treated as "equal"; each parameter in the returned `CompositeGTBasisFuncs` is set to the 
nearest exact multiple of `0.5atol`. When `roundAtol` is set to `NaN`, there will be no 
approximation nor rounding. This function can be called using `+` syntax with the keyword 
argument set to it default value.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> bf1 = genBasisFunc([1.,1.,1.], (2.,1.));

julia> bf2 = genBasisFunc([1.,1.,1.], (2.,2.));

julia> bf3 = bf1 + bf2;

julia> bf3.gauss[1].con() == bf1.gauss[1].con() + bf2.gauss[1].con()
true
```
"""
function add(b::BFuncs1O)
    BasisFunc(b.center, b.gauss, b.l, b.normalizeGTO)
end

add(b::BasisFunc) = itself(b)

function margeBasisFuncCenters(cen1, cen2, roundAtol)
    if cen1 === cen2 || hasIdentical(cen1, cen2)
        cen1
    elseif hasEqual(cen1, cen2)
        deepcopy(cen1)
    else
        c1 = coordOf(cen1)
        c2 = coordOf(cen2)
        if all(isApprox.(c1, c2, atol=roundAtol))
            genSpatialPoint( getNearestMid.(c1, c2, nearestHalfOf(roundAtol)) )
        else
            nothing
        end
    end
end

function add(bf1::BasisFunc{T, D, 𝑙1, GN1, PT1}, bf2::BasisFunc{T, D, 𝑙2, GN2, PT2}; 
             roundAtol::Real=getAtolVal(T)) where 
            {T, D, 𝑙1, 𝑙2, GN1, GN2, PT1, PT2}
    if 𝑙1 == 𝑙2 && bf1.l == bf2.l && bf1.normalizeGTO == bf2.normalizeGTO
        cen = margeBasisFuncCenters(bf1.center, bf2.center, roundAtol)
        cen === nothing && (return BasisFuncMix([bf1, bf2]))
        gfsN = mergeGaussFuncs(bf1.gauss..., bf2.gauss...; roundAtol) |> Tuple
        BasisFunc(cen, gfsN, bf1.l, bf1.normalizeGTO)
    else
        BasisFuncMix([bf1, bf2])
    end
end

add(bfm::BasisFuncMix{T}; roundAtol::Real=getAtolVal(T)) where {T} = 
sumOf(bfm.BasisFunc; roundAtol)

add(bf1::BasisFuncMix{T, D, 1}, bf2::BasisFunc{T, D, 𝑙}; 
    roundAtol::Real=getAtolVal(T)) where {T, D, 𝑙} = 
add(bf1.BasisFunc[1], bf2; roundAtol)

add(bf1::BasisFunc{T, D, 𝑙}, bf2::BasisFuncMix{T, D, 1}; 
    roundAtol::Real=getAtolVal(T)) where {T, D, 𝑙} = 
add(bf2, bf1; roundAtol)

add(bf::BasisFunc{T, D}, bfm::BasisFuncMix{T, D, BN}; 
    roundAtol::Real=getAtolVal(T)) where {T, D, BN} = 
sumOf((bf, bfm.BasisFunc...); roundAtol)

add(bfm::BasisFuncMix{T, D, BN}, bf::BasisFunc{T, D}; 
    roundAtol::Real=getAtolVal(T)) where {T, D, BN} = 
add(bf, bfm; roundAtol)

add(bf1::BasisFuncMix{T, D, 1}, bf2::BasisFuncMix{T, D, 1}; 
    roundAtol::Real=getAtolVal(T)) where {T, D} = 
add(bf1.BasisFunc[1], bf2.BasisFunc[1]; roundAtol)

add(bf::BasisFuncMix{T, D, 1}, bfm::BasisFuncMix{T, D, BN}; 
    roundAtol::Real=getAtolVal(T)) where {T, D, BN} = 
add(bf.BasisFunc[1], bfm; roundAtol)

add(bfm::BasisFuncMix{T, D, BN}, bf::BasisFuncMix{T, D, 1}; 
    roundAtol::Real=getAtolVal(T)) where {T, D, BN} = 
add(bf, bfm; roundAtol)

add(bfm1::BasisFuncMix{T, D, BN1}, bfm2::BasisFuncMix{T, D, BN2}; 
    roundAtol::Real=getAtolVal(T)) where {T, D, BN1, BN2} = 
sumOf((bfm1.BasisFunc..., bfm2.BasisFunc...); roundAtol)

add(::EmptyBasisFunc{<:Any, D}, b::CGTBasisFuncs1O{<:Any, D}; 
    roundAtol::Real=NaN) where {D} = 
itself(b)

add(b::CGTBasisFuncs1O{<:Any, D}, ::EmptyBasisFunc{<:Any, D}; 
    roundAtol::Real=NaN) where {D} = 
itself(b)

add(::EmptyBasisFunc{T1, D}, ::EmptyBasisFunc{T2, D}; 
    roundAtol::Real=NaN) where {D, T1, T2} = 
EmptyBasisFunc{promote_type(T1, T2), D}()


mergeBasisFuncs(bf::FloatingGTBasisFuncs{T, D}; roundAtol::Real=NaN) where {T, D} = [bf]

mergeBasisFuncs(bs::Vararg{FloatingGTBasisFuncs{T, D}, 2}; 
                roundAtol::Real=NaN) where {T, D} = 
collect(bs)

function mergeBasisFuncs(bf1::FloatingGTBasisFuncs{T, D, 𝑙, GN, PT1, ON1}, 
                         bf2::FloatingGTBasisFuncs{T, D, 𝑙, GN, PT2, ON2}; 
                         roundAtol::Real=getAtolVal(T)) where 
                        {T, D, 𝑙, GN, PT1, PT2, ON1, ON2}
    bf1.l == bf2.l && ( return [bf1, bf2] )
    ss = SubshellXYZsizes[𝑙+1]
    (ON1 == ss || ON2 == ss) && ( return [bf1, bf2] )
    if bf1.normalizeGTO == bf2.normalizeGTO
        cen = margeBasisFuncCenters(bf1.center, bf2.center, roundAtol)
        cen === nothing && (return [bf1, bf2])
        if bf1.gauss===bf2.gauss || hasIdentical(bf1.gauss, bf2.gauss)
            gfs = bf1.gauss
        elseif hasEqual(bf1.gauss, bf2.gauss)
            gfs = deepcopy(bf1.gauss)
        else
            gfPairs1 = NTuple{2, T}[outValOf.(x.param) for x in bf1.gauss]
            gfPairs2 = NTuple{2, T}[outValOf.(x.param) for x in bf2.gauss]
            ids = sortperm(gfPairs1)
            gfs1 = bf1.gauss[ids]
            gfs2 = bf2.gauss[sortperm(gfPairs2)]
            gfs = Array{GaussFunc{T}}(undef, GN)
            for (id, (i, gf1), gf2) in zip(ids, enumerate(gfs1), gfs2)
                res = if gf1 === gf2 || hasIdentical(gf1, gf2)
                    gf1
                elseif hasEqual(gf1, gf2)
                    deepcopy(gf1)
                else
                    p1 = gfPairs1[i]
                    p2 = gfPairs2[i]
                    if all(isApprox.(p1, p2, atol=roundAtol))
                        pair = getNearestMid.(p1, p2, nearestHalfOf(roundAtol))
                        GaussFunc(pair...)
                    else
                        false
                    end
                end
                (res == false) ? (return [bf1, bf2]) : (gfs[id] = res)
            end
            gfs = Tuple(gfs)
        end
        [BasisFuncs(cen, gfs, (bf1.l..., bf2.l...), bf1.normalizeGTO)]
    else
        [bf1, bf2]
    end
end

mergeBasisFuncs(bf1::FloatingGTBasisFuncs{T, D}, bf2::FloatingGTBasisFuncs{T, D}, 
                bf3::FloatingGTBasisFuncs{T, D}, bf4::FloatingGTBasisFuncs{T, D}...; 
                roundAtol::Real=getAtolVal(T)) where {T, D} = 
mergeMultiObjs(FloatingGTBasisFuncs{T, D}, mergeBasisFuncs, bf1, bf2, bf3, bf4...; 
               roundAtol)


"""

    mergeBasisFuncsIn(bs::Union{AbstractVector{<:GTBasisFuncs{T, D}}, 
                                Tuple{Vararg{GTBasisFuncs{T, D}}}}; 
                      roundAtol::Real=NaN) where {T, D} -> 
    Vector{<:GTBasisFuncs{T, D}}

Try merging multiple `FloatingGTBasisFuncs` (if there's any) in `bs` into 
`FloatingGTBasisFuncs{T, D, <:Any, <:Any, <:Any, ON}` where `ON > 1` if possible and then 
return the resulted basis collection. If no merging is performed, then the returned 
collection is same as (but not necessarily identical to) `bs`.
"""
function mergeBasisFuncsIn(bs::Union{AbstractVector{<:GTBasisFuncs{T, D}}, 
                                     Tuple{Vararg{GTBasisFuncs{T, D}}}}; 
                           roundAtol::Real=NaN) where {T, D}
    ids = findall(x->isa(x, FGTBasisFuncs1O), bs)
    if isempty(ids)
        collect(bs)
    else
        vcat(mergeBasisFuncs(bs[ids]...; roundAtol), collect(bs[1:end .∉ Ref(ids)]))
    end
end


"""

    mul(gf::GaussFunc{T}, coeff::Real; roundAtol::Real=getAtolVal(T)) where {T} -> 
    GaussFunc

    mul(coeff::Real, gf::GaussFunc{T}; roundAtol::Real=getAtolVal(T)) where {T} -> 
    GaussFunc

    mul(gf1::GaussFunc{T}, gf2::GaussFunc{T}; roundAtol::Real=getAtolVal(T)) where {T} -> 
    GaussFunc

Multiplication between a `Real` number and a [`GaussFunc`](@ref) or two `GaussFunc`s. 
`roundAtol` specifies the absolute approximation tolerance of comparing parameters stored 
in each `GaussFunc` to determine whether they are treated as "equal"; each parameter in the 
returned `GaussFunc` is set to the nearest exact multiple of `0.5atol`. When `roundAtol` is 
set to `NaN`, there will be no approximation nor rounding. This function can be called 
using `*` syntax with the keyword argument set to it default value.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> gf1 = GaussFunc(3.0, 1.0)
$( GaussFunc(3.0, 1.0) )

julia> gf1 * 2
$( GaussFunc(3.0, 2.0) )

julia> gf1 * gf1
$( GaussFunc(6.0, 1.0) )

julia> gf1 * 2 * gf1
$( GaussFunc(6.0, 2.0) )
```
"""
function mul(gf::GaussFunc{T}, coeff::Real; roundAtol::Real=getAtolVal(T)) where {T}
    if isone(coeff)
        itself(gf)
    else
        con, mapFunc, dataName = mulParamBoxCore(coeff, gf.con, nearestHalfOf(roundAtol))
        conNew = genContraction(con, mapFunc; dataName, canDiff=isDiffParam(gf.con))
        GaussFunc(gf.xpn, conNew)
    end
end

mul(coeff::Real, gf::GaussFunc{T}; roundAtol::Real=getAtolVal(T)) where {T} = 
mul(gf, coeff; roundAtol)

mul(gf1::GaussFunc{T}, gf2::GaussFunc{T}; roundAtol::Real=getAtolVal(T)) where {T} = 
GaussFunc(    genExponent(roundToMultiOfStep(gf1.xpn()+gf2.xpn(), roundAtol)), 
           genContraction(roundToMultiOfStep(gf1.con()*gf2.con(), roundAtol)) )

"""

    mul(a1::Real, a2::CompositeGTBasisFuncs{T, D, <:Any, 1}; 
        normalizeGTO::Union{Bool, Missing}=missing, 
        roundAtol::Real=getAtolVal(T)) where {T, D} -> 
    CompositeGTBasisFuncs{T, D, <:Any, 1}

    mul(a1::CompositeGTBasisFuncs{T, D, <:Any, 1}, a2::Real; 
        normalizeGTO::Union{Bool, Missing}=missing, 
        roundAtol::Real=getAtolVal(T)) where {T, D} -> 
    CompositeGTBasisFuncs{T, D, <:Any, 1}

    mul(a1::CompositeGTBasisFuncs{T, D, <:Any, 1}, 
        a2::CompositeGTBasisFuncs{T, D, <:Any, 1}; 
        normalizeGTO::Union{Bool, Missing}=missing, 
        roundAtol::Real=getAtolVal(T)) where {T, D} -> 
    CompositeGTBasisFuncs{T, D, <:Any, 1}

Multiplication between two `CompositeGTBasisFuncs{T, D, <:Any, 1}` (e.g.,  
[`BasisFunc`](@ref) and [`BasisFuncMix`](@ref)), or a `Real` number and a 
`CompositeGTBasisFuncs{T, D, <:Any, 1}`. If `normalizeGTO` is set to `missing` (in 
default), The [`GaussFunc`](@ref) inside the output containers will be normalized only if 
every input `FloatingGTBasisFuncs` (or inside the input `CompositeGTBasisFuncs`) holds 
`hasNormFactor(ai) == true`. `roundAtol` specifies the absolute approximation tolerance of 
comparing parameters stored in each `CompositeGTBasisFuncs` to determine whether they are 
treated as "equal"; each parameter in the returned `CompositeGTBasisFuncs` is set to the 
nearest exact multiple of `0.5atol`. When `roundAtol` is set to `NaN`, there will be no 
approximation nor rounding. This function can be called using `*` syntax with the keyword 
argument set to it default value.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> bf1 = genBasisFunc([1.0, 1.0, 1.0], ([2.0, 1.0], [0.1, 0.2]))
$( genBasisFunc([1.0, 1.0, 1.0], ([2.0, 1.0], [0.1, 0.2])) )

julia> bf2 = bf1 * 2
$( genBasisFunc([1.0, 1.0, 1.0], ([2.0, 1.0], [0.2, 0.4])) )

julia> getindex.(getproperty.(bf2.gauss, :con))
$( (0.2, 0.4) )

julia> bf3 = bf1 * bf2
$( genBasisFunc([1.0, 1.0, 1.0], ([4.0, 3.0, 2.0], [0.02, 0.08, 0.08])) )
```
"""
function mul(sgf1::BasisFunc{T, D, 𝑙1, 1, PT1}, sgf2::BasisFunc{T, D, 𝑙2, 1, PT2}; 
             normalizeGTO::Union{Bool, Missing}=missing, 
             roundAtol::Real=getAtolVal(T)) where {T, D, 𝑙1, 𝑙2, PT1, PT2}
    halfAtol = nearestHalfOf(roundAtol)
    α₁ = sgf1.gauss[1].xpn()
    α₂ = sgf2.gauss[1].xpn()
    d₁ = sgf1.gauss[1].con()
    d₂ = sgf2.gauss[1].con()
    n₁ = sgf1.normalizeGTO
    n₂ = sgf2.normalizeGTO
    n₁ && (d₁ *= getNormFactor(sgf1)[])
    n₂ && (d₂ *= getNormFactor(sgf2)[])
    normalizeGTO isa Missing && (normalizeGTO = n₁*n₂)

    R = if (cen1 = sgf1.center) === (cen2 = sgf2.center) || hasIdentical(cen1, cen2)
        cen1
    elseif hasEqual(cen1, cen2)
        deepcopy(cen1)
    else
        R₁ = coordOf(cen1)
        R₂ = coordOf(cen2)
        if all(isApprox.(R₁, R₂, atol=roundAtol))
            genSpatialPoint( getNearestMid.(R₁, R₂, halfAtol) )
        else
            l1 = sgf1.l[1]
            l2 = sgf2.l[1]
            xpn, con, cen = gaussProd((α₁, d₁, R₁), (α₂, d₂, R₂))
            shiftPolyFunc = @inline (n, c1, c2) -> [(c2 - c1)^k*binomial(n,k) for k=n:-1:0]
            coeffs = map(1:3) do i
                n1 = l1[i]
                n2 = l2[i]
                c1N = shiftPolyFunc(n1, R₁[i], cen[i])
                c2N = shiftPolyFunc(n2, R₂[i], cen[i])
                m = c1N * transpose(c2N |> reverse)
                [diag(m, k)|>sum for k = n2 : (-1)^(-n1 < n2) : -n1]
            end
            lCs = cat(Ref(coeffs[1] * transpose(coeffs[2])) .* coeffs[3]..., dims=3) # TC
            cen = genSpatialPoint(roundToMultiOfStep.(cen, halfAtol))
            pbα = genExponent(roundToMultiOfStep(xpn, halfAtol))
            return BasisFuncMix(
                [BasisFunc(
                    cen, 
                    GaussFunc(pbα, 
                              genContraction(roundToMultiOfStep(con*lCs[i], halfAtol))), 
                    LTuple(i.I .- 1), 
                    normalizeGTO)
                 for i in CartesianIndices(lCs)])
        end
    end

    xpn = roundToMultiOfStep(α₁ + α₂, halfAtol)
    con = roundToMultiOfStep(d₁ * d₂, halfAtol)
    BasisFunc(R, GaussFunc(genExponent(xpn), genContraction(con)), (sgf1.l .+ sgf2.l), 
              normalizeGTO)
end

function mul(sgf1::BasisFunc{T, D, 0, 1, PT1}, sgf2::BasisFunc{T, D, 0, 1, PT2}; 
             normalizeGTO::Union{Bool, Missing}=missing, 
             roundAtol::Real=getAtolVal(T)) where {T, D, PT1, PT2}
    halfAtol = nearestHalfOf(roundAtol)
    d₁ = sgf1.gauss[1].con()
    d₂ = sgf2.gauss[1].con()
    n₁ = sgf1.normalizeGTO
    n₂ = sgf2.normalizeGTO
    n₁ && (d₁ *= getNormFactor(sgf1)[])
    n₂ && (d₂ *= getNormFactor(sgf2)[])
    R₁ = centerCoordOf(sgf1)
    R₂ = centerCoordOf(sgf2)
    xpn, con, cen = gaussProd((sgf1.gauss[1].xpn(), d₁, R₁), (sgf2.gauss[1].xpn(), d₂, R₂))
    normalizeGTO isa Missing && (normalizeGTO = n₁*n₂)
    BasisFunc(genSpatialPoint( roundToMultiOfStep.(cen, halfAtol) ), 
              GaussFunc( genExponent(roundToMultiOfStep(xpn, halfAtol)), 
                         genContraction(roundToMultiOfStep(con, halfAtol)) ), 
              (LTuple(fill(0, D)),), normalizeGTO)
end

function gaussProd((α₁, d₁, R₁)::T, (α₂, d₂, R₂)::T) where 
                  {T<:Tuple{Number, Number, AbstractArray{<:Number}}}
    α = α₁ + α₂
    d = d₁ * d₂ * exp(-α₁ * α₂ / α * sum(abs2, R₁-R₂))
    R = (α₁*R₁ + α₂*R₂) / α
    (α, d, R)
end

function mulCore(bf::BasisFunc{T, D, 𝑙, GN}, coeff::Real; 
                 normalizeGTO::Union{Bool, Missing}=missing, 
                 roundAtol::Real=getAtolVal(T)) where {T, D, 𝑙, GN}
    n = bf.normalizeGTO
    normalizeGTO isa Missing && (normalizeGTO = n)
    c = (n && !normalizeGTO) ? (coeff .* getNormFactor(bf)) : coeff
    gfs = mul.(bf.gauss, c; roundAtol)
    BasisFunc(bf.center, gfs, bf.l, normalizeGTO)
end

mulCore(bfm::BasisFuncMix{T, D, BN}, coeff::Real; 
        normalizeGTO::Union{Bool, Missing}=missing, 
        roundAtol::Real=getAtolVal(T)) where {T, D, BN} = 
BasisFuncMix(mul.(bfm.BasisFunc, coeff; normalizeGTO, roundAtol))

function mul(bf1::BasisFunc{T, D, 𝑙1, GN1, PT1}, bf2::BasisFunc{T, D, 𝑙2, GN2, PT2}; 
             normalizeGTO::Union{Bool, Missing}=missing, 
             roundAtol::Real=getAtolVal(T)) where {T, D, 𝑙1, 𝑙2, GN1, GN2, PT1, PT2}
    bf1n = bf1.normalizeGTO
    bf2n = bf2.normalizeGTO
    normalizeGTO isa Missing && (normalizeGTO = bf1n * bf2n)
    bs = CGTBasisFuncs1O{T, D}[]
    for gf1 in bf1.gauss, gf2 in bf2.gauss
        push!(bs, mul(BasisFunc(bf1.center, (gf1,), bf1.l, bf1n), 
                      BasisFunc(bf2.center, (gf2,), bf2.l, bf2n); 
                      normalizeGTO, roundAtol))
    end
    sumOf(bs; roundAtol)
end

mul(bf1::BasisFuncMix{T, D, 1}, bf2::BasisFunc{T, D}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundAtol::Real=getAtolVal(T)) where 
   {T, D} = 
mul(bf1.BasisFunc[1], bf2; normalizeGTO, roundAtol)

mul(bf1::BasisFunc{T, D}, bf2::BasisFuncMix{T, D, 1}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundAtol::Real=getAtolVal(T)) where 
   {T, D} = 
mul(bf2, bf1; normalizeGTO, roundAtol)

mul(bf::BasisFunc{T, D}, bfm::BasisFuncMix{T, D, BN}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundAtol::Real=getAtolVal(T)) where 
   {T, D, BN} = 
sumOf(mul.(Ref(bf), bfm.BasisFunc; normalizeGTO, roundAtol); roundAtol)

mul(bfm::BasisFuncMix{T, D, BN}, bf::BasisFunc{T, D}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundAtol::Real=getAtolVal(T)) where 
   {T, D, BN} = 
mul(bf, bfm; normalizeGTO, roundAtol)

mul(bf1::BasisFuncMix{T, D, 1}, bf2::BasisFuncMix{T, D, 1}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundAtol::Real=getAtolVal(T)) where 
   {T, D} = 
mul(bf1.BasisFunc[1], bf2.BasisFunc[1]; normalizeGTO, roundAtol)

mul(bf::BasisFuncMix{T, D, 1}, bfm::BasisFuncMix{T, D, BN}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundAtol::Real=getAtolVal(T)) where 
   {T, D, BN} = 
mul(bf.BasisFunc[1], bfm; normalizeGTO, roundAtol)

mul(bfm::BasisFuncMix{T, D, BN}, bf::BasisFuncMix{T, D, 1}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundAtol::Real=getAtolVal(T)) where 
   {T, D, BN} = 
mul(bf, bfm; normalizeGTO, roundAtol)

function mul(bfm1::BasisFuncMix{T, D, BN1}, bfm2::BasisFuncMix{T, D, BN2}; 
             normalizeGTO::Union{Bool, Missing}=missing, 
             roundAtol::Real=getAtolVal(T)) where {T, D, BN1, BN2}
    bfms = CGTBasisFuncs1O{T, D}[]
    for bf1 in bfm1.BasisFunc, bf2 in bfm2.BasisFunc
        push!(bfms, mul(bf1, bf2; normalizeGTO, roundAtol))
    end
    sumOf(bfms; roundAtol)
end

mulCore(::EmptyBasisFunc{<:Any, D}, ::T; 
        normalizeGTO=missing, roundAtol::Real=getAtolVal(T)) where {D, T<:Real} = 
EmptyBasisFunc{T, D}()

function mul(b::CGTBasisFuncs1O{T, D}, coeff::Real; 
             normalizeGTO::Union{Bool, Missing}=missing, 
             roundAtol::Real=getAtolVal(T)) where {T, D}
    if iszero(coeff)
        EmptyBasisFunc{T, D}()
    elseif isone(coeff)
        itself(b)
    else
        mulCore(b, coeff; normalizeGTO, roundAtol)
    end
end

mul(coeff::Real, b::CGTBasisFuncs1O{T}; normalizeGTO::Union{Bool, Missing}=missing, 
    roundAtol::Real=getAtolVal(T)) where {T} = 
mul(b, coeff; normalizeGTO, roundAtol)

mul(::EmptyBasisFunc{<:Any, D}, ::CGTBasisFuncs1O{T, D}; 
    normalizeGTO=missing, roundAtol::Real=getAtolVal(T)) where {D, T} = 
EmptyBasisFunc{T, D}()

mul(::CGTBasisFuncs1O{T, D}, ::EmptyBasisFunc{<:Any, D}; 
    normalizeGTO=missing, roundAtol::Real=getAtolVal(T)) where {T, D} = 
EmptyBasisFunc{T, D}()

mul(::EmptyBasisFunc{T1, D}, ::EmptyBasisFunc{T2, D}; normalizeGTO=missing, 
    roundAtol::Int=getAtolDigits(promote_type(T1, T2))) where {T1, T2, D} = 
EmptyBasisFunc{promote_type(T1, T2), D}()

mul(bf1::BFuncs1O{T, D, 𝑙1, GN1, PT1}, bf2::BFuncs1O{T, D, 𝑙2, GN2, PT2}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundAtol::Real=getAtolVal(T)) where 
   {T, D, 𝑙1, 𝑙2, GN1, GN2, PT1, PT2} = 
[mul(add(bf1), add(bf2); normalizeGTO, roundAtol)]


"""

    shift(bf::FloatingGTBasisFuncs{T, D, 𝑙, GN, PT, 1}, 
          dl::Union{Vector{Int}, NTuple{D, Int}, LTuple{D}}, op::Function=+) where 
         {T, D, 𝑙, GN, PT} -> 
    BasisFunc{T, D, <:Any, GN, PT}

Shift (`+` as the default binary operator `op`) the angular momentum (in Cartesian 
representation) of the input `FloatingGTBasisFuncs` given `dl` that specifies the change of 
each component.
"""
shift(bf::FGTBasisFuncs1O{<:Any, D, 𝑙, GN}, dl::AbstractArray{Int}, op::F=+) where 
     {D, 𝑙, GN, F<:Function} = 
shiftCore(op, bf, LTuple(dl))

shift(bf::FGTBasisFuncs1O{<:Any, D, 𝑙, GN}, dl::NTuple{D, Int}, op::F=+) where 
     {D, 𝑙, GN, F<:Function} = 
shiftCore(op, bf, LTuple(dl))

shiftCore(::typeof(+), bf::FGTBasisFuncs1O{<:Any, D, 𝑙1}, dl::LTuple{D, 𝑙2}) where 
         {D, 𝑙1, 𝑙2} = 
BasisFunc(bf.center, bf.gauss, bf.l[1]+dl, bf.normalizeGTO)

shiftCore(::typeof(-), bf::FGTBasisFuncs1O{<:Any, D, 0}, ::LTuple{D, 0}) where {D} = 
BasisFunc(bf.center, bf.gauss, bf.l[1], bf.normalizeGTO)

shiftCore(::typeof(-), bf::FGTBasisFuncs1O{T, D, 0}, dl::LTuple{D}) where {T, D} = 
EmptyBasisFunc{T, D}()

function shiftCore(::typeof(-), bf::FGTBasisFuncs1O{T, D, 𝑙1}, dl::LTuple{D, 𝑙2}) where 
                  {T, D, 𝑙1, 𝑙2}
    xyz = bf.l[1].tuple .- dl.tuple
    for i in xyz
        i < 0 && (return EmptyBasisFunc{T, D}())
    end
    BasisFunc(bf.center, bf.gauss, LTuple(xyz), bf.normalizeGTO)
end

shift(::EmptyBasisFunc{T, D}, ::Union{LTuple{D}, NTuple{D, Int}}, 
      _::Function=+) where {T, D} = EmptyBasisFunc{T, D}()

"""

    decompose(bf::CompositeGTBasisFuncs{T, D}, splitGaussFunc::Bool=false) -> 
    Matrix{<:BasisFunc{T, D}}

Decompose a `CompositeGTBasisFuncs` into a `Matrix` of [`BasisFunc`](@ref)s. The sum of 
each column represents one orbital of the input basis function(s). If `splitGaussFunc` is 
`true`, then each column consists of the `BasisFunc`s with only 1 [`GaussFunc`](@ref).
"""
decompose(bf::CompositeGTBasisFuncs, splitGaussFunc::Bool=false) = 
decomposeCore(Val(splitGaussFunc), bf)

function decomposeCore(::Val{false}, bf::FloatingGTBasisFuncs{T, D, 𝑙, GN, PT, ON}) where 
                      {T, D, 𝑙, GN, PT, ON}
    res = Array{BasisFunc{T, D, 𝑙, GN, PT}}(undef, 1, ON)
    for i in eachindex(res)
        res[i] = BasisFunc(bf.center, bf.gauss, bf.l[i], bf.normalizeGTO)
    end
    res
end

function decomposeCore(::Val{true}, bf::FloatingGTBasisFuncs{T, D, 𝑙, GN, PT, ON}) where 
                      {T, D, 𝑙, GN, PT, ON}
    res = Array{BasisFunc{T, D, 𝑙, 1, PT}}(undef, GN, ON)
    for (c, l) in zip(eachcol(res), bf.l)
        c .= BasisFunc.(Ref(bf.center), bf.gauss, Ref(l), bf.normalizeGTO)
    end
    res
end

decomposeCore(::Val{false}, b::CGTBasisFuncsON{1}) = hcat(b)

decomposeCore(::Val{true}, b::FGTBasisFuncs1O{<:Any, <:Any, <:Any, 1}) = hcat(BasisFunc(b))

decomposeCore(::Val{false}, b::FGTBasisFuncsON{1}) = hcat(BasisFunc(b))

function decomposeCore(::Val{true}, bfm::BasisFuncMix)
    bfss = map(bfm.BasisFunc) do bf
        decomposeCore(Val(true), bf)
    end
    reshape(vcat(bfss...), :, 1)
end


"""

    orbitalNumOf(subshell::String, D::Integer=3) -> Int

Return the size (number of orbitals) of each subshell in `D` dimensional real space.
"""
@inline orbitalNumOf(subshell::String, D::Integer=3) = SubshellSizeList[D][subshell]

"""

    orbitalNumOf(b::QuiqboxBasis) -> Int

Return the numbers of orbitals of the input basis.
"""
@inline orbitalNumOf(::QuiqboxBasis{<:Any, <:Any, ON}) where {ON} = ON


# Core function to generate a customized X-Gaussian (X>1) basis function.
function genGaussFuncText(xpn::Real, con::Real; roundDigits::Int=-1)
    if roundDigits >= 0
        xpn = round(xpn, digits=roundDigits)
        con = round(con, digits=roundDigits)
    end
    "  " * alignNum(xpn; roundDigits) * (alignNum(con; roundDigits) |> rstrip) * "\n"
end

"""

    genBasisFuncText(bf::FloatingGTBasisFuncs; 
                     norm::Real=1.0, printCenter::Bool=true, roundDigits::Int=-1) -> String

Generate the text of input `FloatingGTBasisFuncs`. `norm` is the additional normalization 
factor. If `printCenter` is `true`, the center coordinate will be added to the first line 
of the output `String`. `roundDigits` specifies the rounding digits for the parameters 
inside `bf`; when set to negative, no rounding will be performed.
"""
function genBasisFuncText(bf::FloatingGTBasisFuncs{T, D}; norm::Real=1.0, 
                          printCenter::Bool=true, roundDigits::Int=-1) where {T, D}
    GFs = map(x -> genGaussFuncText(x.xpn(), x.con(); roundDigits), bf.gauss)
    cen = centerCoordOf(bf)
    firstLine = printCenter ? "X "*(alignNum.(cen; roundDigits) |> join)*"\n" : ""
    firstLine * "$(bf|>subshellOf)    $(getTypeParams(bf)[4])   $(T(norm))" * 
    "   $(bf.normalizeGTO)" * 
    ( isaFullShellBasisFuncs(bf) ? "" : "  " * 
      join( [" $i" for i in get.(Ref(AngMomIndexList[D]), bf.l, "")] |> join ) ) * "\n" * 
    (GFs|>join)
end

"""

    genBasisFuncText(bs::Union{AbstractVector{<:FloatingGTBasisFuncs{T, D}}, 
                               Tuple{Vararg{FloatingGTBasisFuncs{T, D}}}}; 
                     norm::Real=1.0, printCenter::Bool=true, 
                     groupCenters::Bool=true, roundDigits::Int=-1) where {T, D} -> 
    String

Generate the text of input basis set (consisting of `FloatingGTBasisFuncs`). `norm` is the 
additional normalization factor. `groupCenters` determines whether the function will group 
the basis functions with same center together.
"""
function genBasisFuncText(bs::Union{AbstractVector{<:FloatingGTBasisFuncs{T, D}}, 
                                    Tuple{Vararg{FloatingGTBasisFuncs{T, D}}}}; 
                          norm::Real=1.0, printCenter::Bool=true, 
                          groupCenters::Bool=true, roundDigits::Int=-1) where {T, D}
    strs = String[]
    roundAtol = roundDigits<0 ? NaN : exp10(-roundDigits)
    bfBlocks = sortBasisFuncs(bs, groupCenters; roundAtol)
    if groupCenters
        for b in bfBlocks
            push!(strs, joinConcentricBFuncStr(b; norm, printCenter, roundDigits))
        end
    else
        for b in bfBlocks
            push!(strs, genBasisFuncText(b; norm, printCenter, roundDigits))
        end
    end
    strs
end


function joinConcentricBFuncStr(bs::Union{AbstractVector{<:FloatingGTBasisFuncs{T, D}}, 
                                          Tuple{Vararg{FloatingGTBasisFuncs{T, D}}}}; 
                                norm::Real=1.0, printCenter::Bool=true, 
                                roundDigits::Int=-1) where {T, D}
    str = genBasisFuncText(bs[1]; norm, printCenter, roundDigits)
    str *= genBasisFuncText.(bs[2:end]; norm, printCenter=false, roundDigits) |> join
end


"""

    genBFuncsFromText(content::String; adjustContent::Bool=false, 
                      adjustFunction::F=sciNotReplace, 
                      excludeFirstNlines::Int=0, excludeLastNlines::Int=0, 
                      center::Union{AbstractArray, 
                                    NTuple{N, ParamBox}, 
                                    Missing}=missing, 
                      unlinkCenter::Bool=false, 
                      normalizeGTO::Union{Bool, Missing}=missing) where {N, F<:Function} -> 
    Array{<:FloatingGTBasisFuncs, 1}

Generate a basis set from `content` which is either a basis set `String` in Gaussian format 
or the output from `genBasisFuncText`. For the former, `adjustContent` needs to be set to 
`true`. `adjustFunction` is only applied when `adjustContent=true`, which in default is a 
`function` used to detect and convert the format of the scientific notation in `content`.

`excludeFirstNlines` and `excludeLastNlines` are used to exclude first or last few lines of 
`content` if intended. `center` is used to assign a center coordinate for all the basis 
functions from `content`; when it's set to `missing`, it will try to read the center 
information in `content`, and leave the center as `[NaN, NaN, Nan]` if one cannot be found 
for each corresponding `FloatingGTBasisFuncs`. If `unlinkCenter = true`, the center of each 
`FloatingGTBasisFuncs` is a `Base.deepcopy` of the input `center`. Otherwise, they share 
the same underlying data so changing the value of one will affect others. If the center 
coordinate is included in `content`, it should be right above the subshell information for 
the `FloatingGTBasisFuncs`. E.g.:
```
\"\"\"
$( genBasisFuncText(genBasisFunc([1.0, 0.0, 0.0], (2.0, 1.0))) )\"\"\"
```
Finally, `normalizeGTO` specifies the field `.normalizeGTO` of the generated 
`FloatingGTBasisFuncs`. If it's set to `missing` (in default), the normalization 
configuration of each `FloatingGTBasisFuncs` will depend on `content`, so different basis 
functions may have different normalization configurations. However, when it's set to a 
`Bool` value, `.normalizeGTO` of all the generated basis functions will be set to that 
value.
"""
function genBFuncsFromText(content::String; 
                           adjustContent::Bool=false, 
                           adjustFunction::Function=sciNotReplace, 
                           excludeFirstNlines::Int=0, excludeLastNlines::Int=0, 
                           center::Union{AbstractArray{T}, 
                                         NTuple{D, T}, 
                                         NTuple{D, ParamBox{T}}, 
                                         SpatialPoint{T, D}, 
                                         Missing}=(NaN, NaN, NaN), 
                           unlinkCenter::Bool=false, 
                           normalizeGTO::Union{Bool, Missing}=missing) where 
                          {D, T<:AbstractFloat}
    cenIsMissing = ( (all(center.|>isNaN) && (center=missing; true)) || center isa Missing )
    typ = ifelse(cenIsMissing, Float64, T)
    adjustContent && (content = adjustFunction(content))
    lines = split.(content |> IOBuffer |> readlines)
    lines = lines[1+excludeFirstNlines : end-excludeLastNlines]
    data = [advancedParse.(typ, i) for i in lines]
    index = findall( x -> (eltype(x) != typ) && (length(x) > 2) && 
                    (x[1] == "SP" || x[1] in SubshellNames), data )
    bfs = FloatingGTBasisFuncs[]
    if !cenIsMissing
        d = (center isa AbstractArray) ? length(center) : D
    end
    for i in index
        oInfo = data[i]
        gs1 = GaussFunc{typ}[]
        ng = oInfo[2] |> Int
        centerOld = center
        if center isa Missing && i != 1 && data[i-1][1] == "X"
            cenStr = data[i-1][2:end]
            center = convert(Vector{typ}, cenStr)
            d = length(cenStr)
        end
        normFactor = oInfo[3]
        (normalizeGTO isa Missing) && (normalizeGTO = parse(Bool, oInfo[4]))
        if oInfo[1] == "SP"
            gs2 = GaussFunc{typ}[]
            for j = i+1 : i+ng
                push!(gs1, GaussFunc(data[j][1], normFactor*data[j][2]))
                push!(gs2, GaussFunc(data[j][1], normFactor*data[j][3]))
            end
            append!(bfs, genBasisFunc.(Ref(unlinkCenter ? deepcopy(center) : center), 
                                       [gs1, gs2], ["S", "P"]; normalizeGTO))
        else
            for j = i+1 : i+ng
                push!(gs1, GaussFunc(data[j]...))
            end
            subshellInfo = oInfo[1] |> string
            if length(oInfo) > 4
                subshellInfo = SubshellOrientationList[d][subshellInfo][oInfo[5:end]]
            end
            push!(bfs, genBasisFunc((unlinkCenter ? deepcopy(center) : center), 
                                    gs1, subshellInfo; normalizeGTO))
        end
        center = centerOld
    end
    bfs |> flatten
end

"""

    assignCenInVal!(b::FloatingGTBasisFuncs{T, D}, center::AbstractVector{<:Real}) -> 
    SpatialPoint{T, D}

Change the input value of data stored in `b.center` (meaning the output value will also 
change according to the mapping function). Then, return the altered center.
"""
function assignCenInVal!(b::FloatingGTBasisFuncs, center::AbstractVector{<:Real})
    for (i,j) in zip(b.center, center)
        i[] = j
    end
    b.center
end


"""

    getParams(pbc::ParamBox, symbol::Union{Symbol, Missing}=missing; 
              forDifferentiation::Bool=false) -> 
    Union{ParamBox, Nothing}

    getParams(pbc::ParameterizedContainer, symbol::Union{Symbol, Missing}=missing; 
              forDifferentiation::Bool=false) -> 
    AbstractVector{<:ParamBox}

    getParams(pbc::Union{AbstractArray, Tuple}, symbol::Union{Symbol, Missing}=missing; 
              forDifferentiation::Bool=false) -> 
    AbstractVector{<:ParamBox}

Return the parameter(s) stored in the input container. If `symbol` is set to `missing`, 
then return all parameter(s); if it's set to the `Symbol` of a parameter (e.g., `:α₁` will 
match any `pb::`[`ParamBox`](@ref) such that `getVar(pb) == :α₁`; `:α` will match all the 
`pb`s that are `ParamBox{<:Any, α}`. `forDifferentiation` determines whether searching 
through the `Symbol`(s) of the independent variable(s) represented by `pbc` during the 
differentiation process. If the first argument is a collection, its entries must be 
`ParamBox` containers.
"""
getParams(pb::ParamBox, symbol::Union{Symbol, Missing}=missing; 
          forDifferentiation::Bool=false) = 
ifelse(paramFilter(pb, symbol, forDifferentiation), pb, nothing)

getParams(cs::AbstractArray{<:ParamBox}, symbol::Union{Symbol, Missing}=missing; 
          forDifferentiation::Bool=false) = 
cs[findall(x->paramFilter(x, symbol, forDifferentiation), cs)]

getParams(pbc::ParameterizedContainer, symbol::Union{Symbol, Missing}=missing; 
          forDifferentiation::Bool=false) = 
[x for x in pbc.param if paramFilter(x, symbol, forDifferentiation)]

getParams(cs::AbstractArray{<:ParameterizedContainer}, 
          symbol::Union{Symbol, Missing}=missing; forDifferentiation::Bool=false) = 
vcat(getParams.(cs, symbol; forDifferentiation)...)

function getParams(cs::AbstractArray, symbol::Union{Symbol, Missing}=missing; 
                   forDifferentiation::Bool=false)
    pbIdx = findall(x->x isa ParamBox, cs)
    vcat(getParams(convert(Vector{ParamBox}, cs[pbIdx]), symbol; forDifferentiation), 
         getParams(convert(Vector{ParameterizedContainer}, cs[1:end.∉Ref(pbIdx)]), symbol; 
                   forDifferentiation))
end

getParams(cs::Tuple, symbol=missing; forDifferentiation::Bool=false) = 
getParams(collect(cs), symbol; forDifferentiation)

paramFilter(pb::ParamBox, sym::Union{Symbol, Missing}, forDifferentiation::Bool) = 
sym isa Missing || inSymbol(sym, getVar(pb, forDifferentiation))


"""

    copyBasis(b::GaussFunc, copyOutVal::Bool=true) -> GaussFunc

    copyBasis(b::CompositeGTBasisFuncs, copyOutVal::Bool=true) -> CompositeGTBasisFuncs

Return a copy of the input basis. If `copyOutVal` is set to `true`, then only the output 
value(s) of the stored data will be copied, i.e., [`outValCopy`](@ref) is used to copy the 
[`ParamBox`](@ref)s, otherwise [`inVarCopy`](@ref) is used.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> e = genExponent(3.0, x->x^2)
$( genExponent(3.0, x->x^2) )

julia> c = genContraction(2.0)
$( genContraction(2.0) )

julia> gf1 = GaussFunc(e, c);

julia> gf2 = copyBasis(gf1)
$( GaussFunc(9.0, 2.0) )

julia> gf1.xpn() == gf2.xpn()
true

julia> (gf1.xpn[] |> gf1.xpn.map) == gf2.xpn[]
true
```
"""
function copyBasis(g::GaussFunc, copyOutVal::Bool=true)
    pbs = g.param .|> ifelse(copyOutVal, outValCopy, inVarCopy)
    GaussFunc(pbs...)
end

function copyBasis(bfs::T, copyOutVal::Bool=true) where {T<:FloatingGTBasisFuncs}
    cen = bfs.center .|> ifelse(copyOutVal, outValCopy, inVarCopy)
    gs = copyBasis.(bfs.gauss, copyOutVal)
    genBasisFunc(cen, gs, bfs.l; normalizeGTO=bfs.normalizeGTO)::T
end

function copyBasis(bfm::T, copyOutVal::Bool=true) where {T<:BasisFuncMix}
    bfs = copyBasis.(bfm.BasisFunc, copyOutVal)
    BasisFuncMix(bfs)::T
end


"""

    markParams!(b::Union{AbstractVector{T}, T, Tuple{Vararg{T}}}, 
                filterMapping::Bool=false) where {T<:ParameterizedContainer} -> 
    Vector{<:ParamBox}

Mark the parameters ([`ParamBox`](@ref)) in `b`. The parameters that will be considered 
identical in the differentiation procedure will be marked with same index. `filterMapping` 
determines whether filtering out (i.e. not return) the extra `ParamBox`s that have the same 
indices despite may having different mapping functions.
"""
markParams!(b::Union{AbstractVector{T}, T, Tuple{Vararg{T}}}, 
            filterMapping::Bool=false) where {T<:ParameterizedContainer} = 
markParams!(getParams(b), filterMapping)

function markParams!(pars::AbstractVector{<:ParamBox}, filterMapping::Bool=false)
    ids = findall(isDepParam, pars)
    if isempty(ids)
        res = markParamsCore1!(pars)
    else
        res1, d = markParamsCore2!(view(pars, ids))
        res2 = markParamsCore1!((@view pars[1:end .∉ Ref(ids)]), d)
        res = vcat(res1, res2)
    end
    filterMapping ? res : pars
end

function markParamsCore1!(pars::AbstractVector{<:ParamBox}, 
                          d::Dict{Symbol, Vector{Int}}=Dict{Symbol, Vector{Int}}())
    ids1, items = markUnique(outSymOfCore.(pars))
    uniqueParams = eltype(pars)[]
    for i=1:length(items)
        parsSameV = view(pars, findall(isequal(i), ids1))
        ids2, iPars = markUnique(parsSameV, compareFunction=compareParamBox)
        for (i, par) in enumerate(parsSameV)
            # par.index[] = i + get(d, outSymOfCore(par), 0)
            par.index[] = skipIndices(ids2, get(d, outSymOfCore(par), Int[]))[i]
        end
        append!(uniqueParams, iPars)
    end
    uniqueParams
end

function markParamsCore2!(parArray::AbstractVector{<:ParamBox})
    uniqueParams = eltype(parArray)[]
    idDict = Dict{UInt, Tuple{Symbol, Int}}()
    iVDict = Dict{Symbol, Int}()
    dVidDict = Dict{Symbol, Vector{Int}}()
    for par in parArray
        id = objectid(par.data)
        idx = get(idDict, id, (:nothing, 0))[end]
        if iszero(idx)
            iParSym = inSymOfCore(par)
            idx = get!(iVDict, iParSym, 0) + 1
            push!(idDict, id=>(iParSym, idx))
            iVDict[iParSym] += 1
            push!(uniqueParams, par)
            dParSym = outSymOfCore(par)
            if haskey(dVidDict, dParSym)
                push!(dVidDict[dParSym], idx)
            else
                dVidDict[dParSym] = [idx]
            end
        end
        par.index[] = idx
    end
    uniqueParams, dVidDict
end

markParams!(parTuple::Tuple{Vararg{ParamBox}}, filterMapping::Bool=false) = 
markParams!(collect(parTuple), filterMapping)


"""

    hasNormFactor(b::FloatingGTBasisFuncs) -> Bool

Indicate whether `b`' is be treated as having additional normalization factor(s) which its 
Gaussian-type orbital(s) will be multiplied by during any calculation.
"""
hasNormFactor(b::FloatingGTBasisFuncs) = b.normalizeGTO


getNijk(::Type{T}, i::Integer, j::Integer, k::Integer) where {T} = 
T(πvals[-0.75]) * 2^(T(1.5)*(i+j+k) + T(0.75)) * sqrt( factorial(i) * factorial(j) * 
factorial(k) / (factorial(2i) * factorial(2j) * factorial(2k)) )

getNα(i::Integer, j::Integer, k::Integer, α::T) where {T} = 
α^( T(2i + 2j + 2k + 3)*T(0.25) )

getNijkα(i::Integer, j::Integer, k::Integer, α::T) where {T} = 
getNijk(T, i, j, k) * getNα(i, j, k, α)

getNijkα(ijk, α) = getNijkα(ijk[1], ijk[2], ijk[3], α)

"""

    getNormFactor(b::FloatingGTBasisFuncs{T, 3}) where {T} -> Array{T}

Return the normalization factors of the Gaussian-type orbitals (GTO) inside the input `b`. 
Each column corresponds to one orbital.
"""
getNormFactor(b::FGTBasisFuncs1O{T, 3, 𝑙, GN})  where {T, 𝑙, GN} = 
getNijkα.(b.l[1]..., T[g.xpn() for g in b.gauss])

getNormFactor(b::FloatingGTBasisFuncs{<:Any, 3, <:Any, <:Any, <:Any, ON}) where {ON} = 
hcat(getNormFactor.(b)...)


"""

    absorbNormFactor(b::BasisFunc{T, 3, 𝑙, GN, PT}) where {T, 𝑙, GN, PT} -> 
    FloatingGTBasisFuncs{T, 3, 𝑙, GN, PT}

    absorbNormFactor(b::BasisFuncs{T, 3, 𝑙, GN, PT}) where {T, 𝑙, GN, PT} -> 
    Vector{<:FloatingGTBasisFuncs{T, 3, 𝑙, GN, PT}}

If `hasNormFactor(`b`) == true`, absorb the normalization factor of each Gaussian-type 
orbital inside `b` into the orbital's corresponding contraction coefficient and then set 
`.normalizeGTO` of `b` to `false`. Otherwise, directly return `b` when it's a `BasisFunc`, 
or `[b]` when it's a `BasisFuncs`.
"""
function absorbNormFactor(b::BasisFunc{<:Any, 3})
    if hasNormFactor(b)
        absorbNormFactorCore(b)
    else
        b
    end
end

absorbNormFactorCore(b::BasisFunc{<:Any, 3, <:Any, 1}) = 
mul(genBasisFunc(b, false), getNormFactor(b)[begin], roundAtol=NaN)

absorbNormFactorCore(b::BasisFunc{<:Any, 3}) = 
sumOf( mul.(decomposeCore(Val(true), genBasisFunc(b, false)), 
          getNormFactor(b), roundAtol=NaN), 
     roundAtol=NaN)

absorbNormFactor(b::BasisFuncs{<:Any, 3, <:Any, <:Any, <:Any, ON}) where {ON} = 
mergeBasisFuncs(absorbNormFactor.(b)...)

"""

    absorbNormFactor(bfm::BasisFuncMix{T, 3}) where {T} -> GTBasisFuncs{T, 3}

Apply `absorbNormFactor` to every one of the `BasisFunc` inside `bfm` and them sum them 
over.
"""
absorbNormFactor(bfm::BasisFuncMix{<:Any, 3}) = absorbNormFactor(bfm.BasisFunc) |> sumOf


"""

    absorbNormFactor(bs::AbstractVector{<:GTBasisFuncs{T, 3}}) where {T} -> 
    AbstractVector{<:GTBasisFuncs{T, 3}}

    absorbNormFactor(bs::Tuple{Vararg{GTBasisFuncs{T, 3}}}) where {T} -> 
    Tuple{Vararg{GTBasisFuncs{T, 3}}}

Apply `absorbNormFactor` to every element inside `bs`.
"""
absorbNormFactor(bs::AbstractVector{<:GTBasisFuncs{T, 3}}) where {T} = 
vcat(absorbNormFactor.(bs)...)

absorbNormFactor(bs::Tuple{Vararg{GTBasisFuncs{T, 3}}}) where {T} = 
absorbNormFactor(bs |> collect) |> Tuple


"""

    normalizeBasis(b::GTBasisFuncs{T, D, 1}) where {T, D} -> GTBasisFuncs{T, D, 1}

Multiply the contraction coefficient(s) inside `b` by constant coefficient(s) to normalizeBasis 
the `b`, and then return the normalized basis.
"""
function normalizeBasis(b::GTBasisFuncs{T}) where {T}
    nrm = roundToMultiOfStep(overlap(b,b), exp10(-getAtolDigits(T)-1))
    mul(b, inv(nrm|>sqrt), roundAtol=NaN)
end

"""

    normalizeBasis(b::BasisFuncs{T, D}) where {T, D} -> Vector{<:FloatingGTBasisFuncs{T, D}}

Normalize each [`BasisFunc`](@ref) inside `b` and try to merge them back to one 
[`BasisFuncs`](@ref). If the merge is performed, the returned result is a 1-element 
`Vector`.
"""
normalizeBasis(bfs::BasisFuncs) = mergeBasisFuncs(normalizeBasis.(bfs)...)