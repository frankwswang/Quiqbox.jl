export GaussFunc, genExponent, genContraction, SpatialPoint, genSpatialPoint, coordOf, 
       BasisFunc, BasisFuncs, genBasisFunc, lOf, subshellOf, centerOf, centerCoordOf, 
       gaussCoeffOf, dimOf, GTBasis, sortBasisFuncs, sortPermBasisFuncs, sortBasis, 
       sortPermBasis, add, mul, shift, decompose, orbitalNumOf, genBasisFuncText, 
       genBFuncsFromText, assignCenInVal!, getParams, copyBasis, markParams!, normOf

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
genExponent(pb::ParamBox) = ParamBox(Val(xpnSym), pb)


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
genContraction(pb) = ParamBox(Val(conSym), pb)


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
same way as in a general constructor of a `ParamBox`. If `roundDigits < 0` or `point` is a 
0-dimensional `Array`, there won't be rounding for input data.

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

genSpatialPoint(point::ParamBox, index::Int) = ParamBox(Val(SpatialParamSyms[index]), point)

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
BasisFunc{Float64, 3, 1, 1, P3D{Float64, 0, 0, 0}}(center, gauss, l, normalizeGTO, param)[X⁰Y¹Z⁰][0.0, 0.0, 0.0]
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
BasisFuncs{Float64, 3, 1, 1, P3D{Float64, 0, 0, 0}, 3}(center, gauss, l, normalizeGTO, param)[3/3][0.0, 0.0, 0.0]

julia> genBasisFunc([0.,0.,0.], (2., 1.5), "P", (true, false, true))
BasisFuncs{Float64, 3, 1, 1, P3D{Float64, 0, 0, 0}, 2}(center, gauss, l, normalizeGTO, param)[2/3][0.0, 0.0, 0.0]
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
genBasisFunc(cen, gs, SubshellOrientationList[D][subshell][1:end .∈ [findall(lFilter)]]; 
             normalizeGTO)

function genBasisFunc(center::SpatialPoint{T, D}, BSkey::String, atm::String="H"; 
                      unlinkCenter::Bool=false) where {T, D}
    BSstr = BasisSetList[BSkey][AtomicNumberList[atm]]
    @assert BSstr!==nothing "Quiqbox DOES NOT have basis set "*BSkey*" for "*atm*"."
    genBFuncsFromText(BSstr; adjustContent=true, excludeLastNlines=1, center, unlinkCenter)
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
                   groupCenters::Bool=false; roundDigits::Int=getAtolDigits(T)) where 
                  {T, D} -> 
    Vector

Sort `FloatingGTBasisFuncs`. If `groupCenters = true`, Then the function will return an 
`Vector{<:Vector{<:FloatingGTBasisFuncs}}` in which the elements are grouped basis 
functions with same center coordinates. `roundDigits` specifies the rounding digits for 
center coordinates when comparing them. When set to negative, no rounding will be performed.
"""
@inline function sortBasisFuncs(bs::AbstractArray{<:FloatingGTBasisFuncs{T, D}}, 
                                groupCenters::Bool=false; 
                                roundDigits::Int=getAtolDigits(T)) where {T, D}
    bfBlocks = map( groupedSort(reshape(bs, :), 
                    x->roundNum.(centerCoordOf(x), roundDigits)) ) do subbs
        # Reversed order within same subshell.
        sort!(subbs, by=x->[-getTypeParams(x)[3], x.l[1].tuple, getTypeParams(x)[4]], 
              rev=true)
    end
    groupCenters ? bfBlocks : vcat(bfBlocks...)
end

"""

    sortBasisFuncs(bs::Tuple{Vararg{FloatingGTBasisFuncs{T, D}}}, groupCenters::Bool=false; 
                   roundDigits::Int=getAtolDigits(T)) where {T, D} -> 
    Tuple

"""
sortBasisFuncs(bs::Tuple{Vararg{FloatingGTBasisFuncs{T, D}}}, groupCenters::Bool=false; 
               roundDigits::Int=getAtolDigits(T)) where {T, D} = 
sortBasisFuncs(FloatingGTBasisFuncs{T, D}[bs...], groupCenters; roundDigits) |> Tuple


"""

    sortPermBasisFuncs(bs::Union{AbstractArray{<:FloatingGTBasisFuncs{T, D}}, 
                                 Tuple{Vararg{FloatingGTBasisFuncs{T, D}}}}) where 
                      {T, D} -> 
    Vector{Int}

Return a `Vector` of indices `I` such that `bs[I] == `[`sortBasisFuncs`](@ref)
`(bs; roundDigits)[I]`.
"""
sortPermBasisFuncs(bs::AbstractArray{<:FloatingGTBasisFuncs{T, D}}; 
                   roundDigits::Int=getAtolDigits(T)) where {T, D} = 
sortperm(reshape(bs, :), 
         by=x->[-1*roundNum.(centerCoordOf(x), roundDigits), 
                -getTypeParams(x)[3], x.l[1].tuple, getTypeParams(x)[4]], rev=true)

sortPermBasisFuncs(bs::Tuple{Vararg{FloatingGTBasisFuncs{T, D}}}; 
                   roundDigits::Int=getAtolDigits(T)) where {T, D} = 
sortPermBasisFuncs(FloatingGTBasisFuncs{T, D}[bs...]; roundDigits)


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

    gaussCoeffOf(gf::GaussFunc{T}) -> Vector{T}

Return the exponent and contraction coefficients of `gf`.
"""
gaussCoeffOf(gf::GaussFunc) = outValOf.(gf.param) |> collect

"""

    gaussCoeffOf(b::FloatingGTBasisFuncs{T}) -> Matrix{T}

Return the exponent and contraction coefficients of each [`GaussFunc`](@ref) (in each 
column of the returned `Matrix`) inside `b`.
"""
gaussCoeffOf(bf::FloatingGTBasisFuncs) = hcat(gaussCoeffOf.(bf.gauss)...)


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
        bs = sortBasisFuncs(bfs, roundDigits=-1)
        new{T, D, BN, eltype(bfs)}(bs, joinTuple(getproperty.(bs, :param)...))
    end
end

BasisFuncMix(bfs::AbstractArray{<:BasisFunc}) = BasisFuncMix(bfs|>Tuple)
BasisFuncMix(bfs::AbstractArray{T}) where {T<:FGTBasisFuncsON{1}} = 
BasisFuncMix(BasisFunc.(bfs))
BasisFuncMix(bf::BasisFunc) = BasisFuncMix((bf,))
BasisFuncMix(bfs::BasisFuncs) = BasisFuncMix.(decompose(bfs))
BasisFuncMix(bfm::BasisFuncMix) = itself(bfm)


getTypeParams(::FloatingGTBasisFuncs{T, D, 𝑙, GN, PT, ON}) where {T, D, 𝑙, GN, PT, ON} = 
(T, D, 𝑙, GN, PT, ON)
getTypeParams(::BasisFuncMix{T, D, BN, BFT}) where {T, D, BN, BFT} = (T, D, BN, BFT)


unpackBasis(::EmptyBasisFunc) = ()
unpackBasis(b::BasisFunc)  = (b,)
unpackBasis(b::BasisFuncMix)  = b.BasisFunc
unpackBasis(b::BFuncs1O)  = (BasisFunc(b),)


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
              roundDigits::Int=getAtolDigits(T)) where {T, D} -> 
    Vector{<:CompositeGTBasisFuncs{T, D}}

Sort basis functions. `roundDigits` specifies the rounding digits for the parameters stored
in each `CompositeGTBasisFuncs` when comparing them. When set to negative, no rounding will 
be performed.
"""
function sortBasis(bs::AbstractArray{<:CompositeGTBasisFuncs{T, D}}; 
                   roundDigits::Int=getAtolDigits(T)) where {T, D}
    bs = reshape(copy(bs), :)
    ids = findall(x->isa(x, FloatingGTBasisFuncs), bs)
    bfs = splice!(bs, ids)
    vcat( sortBasisFuncs(convert(AbstractVector{FloatingGTBasisFuncs{T, D}}, bfs); 
                         roundDigits), 
          sortBasis(convert(AbstractVector{BasisFuncMix{T, D}}, bs); roundDigits) )
end

sortBasis(bs::AbstractArray{<:BasisFuncMix{T, D}}; 
          roundDigits::Int=getAtolDigits(T)) where {T, D} = 
bs[sortPermBasis(bs; roundDigits)]

sortBasis(bs::AbstractArray{<:FloatingGTBasisFuncs{T, D}}; 
          roundDigits::Int=getAtolDigits(T)) where {T, D} = 
sortBasisFuncs(bs; roundDigits)

"""

    sortBasis(bs::Tuple{Vararg{CompositeGTBasisFuncs{T, D}}}; 
              roundDigits::Int=getAtolDigits(T)) where {T, D} -> 
    Tuple{Vararg{CompositeGTBasisFuncs{T, D}}}

"""
sortBasis(bs::Tuple{Vararg{CompositeGTBasisFuncs{T, D}}}; 
          roundDigits::Int=getAtolDigits(T)) where {T, D} = 
sortBasis(collect(bs); roundDigits) |> Tuple

"""

    sortBasis(b::GTBasis{T, D}; roundDigits::Int=getAtolDigits(T)) where {T, D} -> 
    GTBasis{T, D}

Reconstruct a [`GTBasis`](@ref) by sorting the `GTBasisFuncs` stored in the input one.
"""
sortBasis(b::GTBasis; roundDigits::Int=getAtolDigits(T)) = 
          GTBasis(sortBasis(b.basis; roundDigits))


"""

    sortPermBasis(bs::AbstractArray{<:CompositeGTBasisFuncs{T, D}}; 
                  roundDigits::Int=getAtolDigits(T)) where {T, D} -> 
    Vector{Int}

Return a `Vector` of indices `I` such that `bs[I] == `[`sortBasis`](@ref)
`(bs; roundDigits)[I]`.
"""
function sortPermBasis(bs::AbstractArray{<:CompositeGTBasisFuncs{T, D}}; 
                       roundDigits::Int=getAtolDigits(T)) where {T, D}
    ids = objectid.(bs)
    bsN = sortBasis(bs; roundDigits)
    idsN = objectid.(bsN)
    indexin(idsN, ids)
end

sortPermBasis(bs::AbstractArray{<:BasisFuncMix{T, D}}; 
              roundDigits::Int=getAtolDigits(T)) where {T, D} = 
sortPermBasisFuncs(getindex.(getproperty.(bs, :BasisFunc), 1); roundDigits)

sortPermBasis(bs::AbstractArray{<:FloatingGTBasisFuncs{T, D}}; 
              roundDigits::Int=getAtolDigits(T)) where {T, D} = 
sortPermBasisFuncs(bs; roundDigits)

sortPermBasis(bs::Tuple{Vararg{CompositeGTBasisFuncs{T, D}}}; 
              roundDigits::Int=getAtolDigits(T)) where {T, D} = 
sortPermBasis(collect(bs); roundDigits)


function sumOfCore(bfs::AbstractVector{<:BasisFunc{T, D}}, 
                   roundDigits::Int=getAtolDigits(T)) where {T, D}
    arr1 = convert(Vector{BasisFunc{T, D}}, sortBasisFuncs(bfs; roundDigits))
    arr2 = BasisFunc{T, D}[]
    while length(arr1) > 1
        temp = add(arr1[1], arr1[2]; roundDigits)
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
                    AbstractVector{<:GTBasisFuncs{T, D, 1}}}, 
          roundDigits::Int=getAtolDigits(T)) where {T, D} = 
sumOfCore(BasisFunc{T, D}[joinTuple(unpackBasis.(bs)...)...], roundDigits)

function sumOf(bs::Union{Tuple{Vararg{GTBasisFuncs{T, D, 1}}}, 
                         AbstractVector{<:GTBasisFuncs{T, D, 1}}}; 
               roundDigits::Int=getAtolDigits(T)) where {T, D}
    length(bs) == 1 && (return bs[1])
    sumOfCore(bs, roundDigits)
end

function mergeGaussFuncs(gf1::GaussFunc{T}, gf2::GaussFunc{T}; 
                         roundDigits::Int=getAtolDigits(T)) where {T}
    xpn = if (xpn1 = gf1.xpn) === (xpn2 = gf2.xpn) || hasIdentical(xpn1, xpn2)
        xpn1
    elseif hasEqual(xpn1, xpn2)
        deepcopy(xpn1)
    elseif (xpn1R=roundNum(xpn1(), roundDigits)) == (xpn2R=roundNum(xpn2(), roundDigits))
        genExponent(xpn1R)
    else
        return [gf1, gf2]
    end

    res = if (con1 = gf1.con) === (con2 = gf2.con) || hasIdentical(con1, con2)
        mul(GaussFunc(xpn, con1), 2; roundDigits)
    elseif hasEqual(con1, con2)
        mul(GaussFunc(xpn, deepcopy(con1)), 2; roundDigits)
    elseif (con1R=roundNum(con1(), roundDigits)) == (con2R=roundNum(con2(), roundDigits))
        GaussFunc(xpn, genContraction(roundNum(2con1R, roundDigits)))
    else
        GaussFunc(xpn, genContraction(roundNum(con1R + con2R, roundDigits)))
    end

    [res]
end

mergeGaussFuncs(gf1::GaussFunc{T}, gf2::GaussFunc{T}, gf3::GaussFunc{T}, 
                gf4::GaussFunc{T}...; roundDigits::Int=getAtolDigits(T)) where {T} = 
mergeMultiObjs(GaussFunc{T}, mergeGaussFuncs, gf1, gf2, gf3, gf4...; roundDigits)


"""

    add(b1::CompositeGTBasisFuncs{T, D, <:Any, 1}, 
        b2::CompositeGTBasisFuncs{T, D, <:Any, 1}; roundDigits::Int=getAtolDigits(T)) where 
       {T, D} -> 
    CompositeGTBasisFuncs{T, D, <:Any, 1}

Addition between two `CompositeGTBasisFuncs{T, D, <:Any, 1}`s such as [`BasisFunc`](@ref) 
and [`BasisFuncMix`](@ref). `roundDigits` specifies the maximal number of digits after the 
radix point of the calculated values. The function can be called using `+` syntax with the 
keyword argument set to it default value.

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

function margeBasisFuncCenters(cen1, cen2, roundDigits)
    if cen1 === cen2 || hasIdentical(cen1, cen2)
        cen1
    elseif hasEqual(cen1, cen2)
        deepcopy(cen1)
    elseif (c1 = roundNum.(coordOf(cen1), roundDigits)) == 
                 roundNum.(coordOf(cen2), roundDigits)
        genSpatialPoint(c1)
    else
        nothing
    end
end

function add(bf1::BasisFunc{T, D, 𝑙1, GN1, PT1}, bf2::BasisFunc{T, D, 𝑙2, GN2, PT2}; 
             roundDigits::Int=getAtolDigits(T)) where {T, D, 𝑙1, 𝑙2, GN1, GN2, PT1, PT2}
    if 𝑙1 == 𝑙2 && bf1.l == bf2.l && bf1.normalizeGTO == bf2.normalizeGTO
        cen = margeBasisFuncCenters(bf1.center, bf2.center, roundDigits)
        cen === nothing && (return BasisFuncMix([bf1, bf2]))
        gfsN = mergeGaussFuncs(bf1.gauss..., bf2.gauss...; roundDigits) |> Tuple
        BasisFunc(cen, gfsN, bf1.l, bf1.normalizeGTO)
    else
        BasisFuncMix([bf1, bf2])
    end
end

mergeBasisFuncs(bf1::FloatingGTBasisFuncs{T, D}, bf2::FloatingGTBasisFuncs{T, D}, 
                bf3::FloatingGTBasisFuncs{T, D}, bf4::FloatingGTBasisFuncs{T, D}...; 
                roundDigits::Int=getAtolDigits(T)) where {T, D} = 
mergeMultiObjs(FloatingGTBasisFuncs{T, D}, mergeBasisFuncs, bf1, bf2, bf3, bf4...; 
               roundDigits)

mergeBasisFuncs(bs::Vararg{GTBasisFuncs{T, D}, 2}; roundDigits::Int=-1) where {T, D} = 
collect(bs)

function mergeBasisFuncs(bf1::FloatingGTBasisFuncs{T, D, 𝑙, GN, PT1, ON1}, 
                         bf2::FloatingGTBasisFuncs{T, D, 𝑙, GN, PT2, ON2}; 
                         roundDigits::Int=getAtolDigits(T)) where 
                        {T, D, 𝑙, GN, PT1, PT2, ON1, ON2}
    ss = SubshellXYZsizes[𝑙+1]
    (ON1 == ss || ON2 == ss) && ( return [bf1, bf2] )
    if bf1.normalizeGTO == bf2.normalizeGTO
        cen = margeBasisFuncCenters(bf1.center, bf2.center, roundDigits)
        cen === nothing && (return [bf1, bf2])
        if bf1.l == bf2.l
            gfs = mergeGaussFuncs(bf1.gauss..., bf2.gauss...; roundDigits) |> Tuple
            return [BasisFunc(cen, gfs, bf1.l, bf1.normalizeGTO)]
        else
            gfPairs1 = [roundNum.((x.xpn(), x.con()), roundDigits) for x in bf1.gauss]
            gfPairs2 = [roundNum.((x.xpn(), x.con()), roundDigits) for x in bf2.gauss]
            ids = sortperm(gfPairs1)
            gfs1 = bf1.gauss[ids]
            gfs2 = bf2.gauss[sortperm(gfPairs2)]
            gfs = Array{GaussFunc{T}}(undef, GN)
            for (id, (i, gf1), gf2) in zip(ids, enumerate(gfs1), gfs2)
                res = if gf1 === gf2 || hasIdentical(gf1, gf2)
                    gf1
                elseif hasEqual(gf1, gf2)
                    deepcopy(gf1)
                elseif gfPairs1[i] == gfPairs2[i]
                    GaussFunc(gfPairs1[i]...)
                else
                    false
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

add(bfm::BasisFuncMix{T}; roundDigits::Int=getAtolDigits(T)) where {T} = 
sumOf(bfm.BasisFunc; roundDigits)

add(bf1::BasisFuncMix{T, D, 1}, bf2::BasisFunc{T, D, 𝑙}; 
    roundDigits::Int=getAtolDigits(T)) where {T, D, 𝑙} = 
add(bf1.BasisFunc[1], bf2; roundDigits)

add(bf1::BasisFunc{T, D, 𝑙}, bf2::BasisFuncMix{T, D, 1}; 
    roundDigits::Int=getAtolDigits(T)) where {T, D, 𝑙} = 
add(bf2, bf1; roundDigits)

add(bf::BasisFunc{T, D}, bfm::BasisFuncMix{T, D, BN}; 
    roundDigits::Int=getAtolDigits(T)) where {T, D, BN} = 
sumOf((bf, bfm.BasisFunc...); roundDigits)

add(bfm::BasisFuncMix{T, D, BN}, bf::BasisFunc{T, D}; 
    roundDigits::Int=getAtolDigits(T)) where {T, D, BN} = 
add(bf, bfm; roundDigits)

add(bf1::BasisFuncMix{T, D, 1}, bf2::BasisFuncMix{T, D, 1}; 
    roundDigits::Int=getAtolDigits(T)) where {T, D} = 
add(bf1.BasisFunc[1], bf2.BasisFunc[1]; roundDigits)

add(bf::BasisFuncMix{T, D, 1}, bfm::BasisFuncMix{T, D, BN}; 
    roundDigits::Int=getAtolDigits(T)) where {T, D, BN} = 
add(bf.BasisFunc[1], bfm; roundDigits)

add(bfm::BasisFuncMix{T, D, BN}, bf::BasisFuncMix{T, D, 1}; 
    roundDigits::Int=getAtolDigits(T)) where {T, D, BN} = 
add(bf, bfm; roundDigits)

add(bfm1::BasisFuncMix{T, D, BN1}, bfm2::BasisFuncMix{T, D, BN2}; 
    roundDigits::Int=getAtolDigits(T)) where {T, D, BN1, BN2} = 
sumOf((bfm1.BasisFunc..., bfm2.BasisFunc...); roundDigits)

add(bf1::BFuncs1O{T, D}, bf2::BFuncs1O{T, D}; roundDigits::Int=getAtolDigits(T)) where 
   {T, D} = 
[sumOf((add(bf1), add(bf2)); roundDigits)]

add(::EmptyBasisFunc{<:Any, D}, b::CGTBasisFuncs1O{<:Any, D}) where {D} = itself(b)

add(b::CGTBasisFuncs1O{<:Any, D}, ::EmptyBasisFunc{<:Any, D}) where {D} = itself(b)

add(::EmptyBasisFunc{T1, D}, ::EmptyBasisFunc{T2, D}) where {D, T1, T2} = 
EmptyBasisFunc{promote_type(T1, T2), D}()


"""

    mul(gf::GaussFunc{T}, coeff::Real; roundDigits::Int=getAtolDigits(T)) where {T} -> 
    GaussFunc

    mul(coeff::Real, gf::GaussFunc{T}; roundDigits::Int=getAtolDigits(T)) where {T} -> 
    GaussFunc

    mul(gf1::GaussFunc{T}, gf2::GaussFunc{T}; 
        roundDigits::Int=getAtolDigits(T)) where {T} -> 
    GaussFunc

Multiplication between a `Real` number and a [`GaussFunc`](@ref) or two `GaussFunc`s. 
`roundDigits` specifies the maximal number of digits after the radix point of the 
calculated values. When set to negative, no rounding will be performed. The function can be 
called using `*` syntax with the keyword argument set to it default value.

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
function mul(gf::GaussFunc{T}, c::Real; roundDigits::Int=getAtolDigits(T)) where {T}
    con, mapFunction, dataName = mulCore(c, gf.con; roundDigits)
    conNew = genContraction(con, mapFunction; dataName, canDiff=gf.con.canDiff[])
    GaussFunc(gf.xpn, conNew)
end

function mulCore(c::T1, con::ParamBox{T2, <:Any, FI}; 
                 roundDigits::Int=getAtolDigits(T2)) where {T1, T2}
    conNew = fill(roundNum(convert(T2, con.data[]*c), roundDigits))
    mapFunction = itself
    dataName = :undef
    conNew, mapFunction, dataName
end

function mulCore(c::T1, con::ParamBox{T2}; 
                 roundDigits::Int=getAtolDigits(T2)) where {T1, T2}
    conNew = con.data
    mapFunction = Pf(convert(T2, roundNum(c, roundDigits)), con.map)
    conNew, mapFunction, con.dataName
end

mul(coeff::Real, gf::GaussFunc{T}; roundDigits::Int=getAtolDigits(T)) where {T} = 
mul(gf, coeff; roundDigits)

mul(gf1::GaussFunc{T}, gf2::GaussFunc{T}; roundDigits::Int=getAtolDigits(T)) where {T} = 
GaussFunc(    genExponent(roundNum(gf1.xpn()+gf2.xpn(), roundDigits)), 
           genContraction(roundNum(gf1.con()*gf2.con(), roundDigits)) )

"""

    mul(a1::Real, a2::CompositeGTBasisFuncs{T, D, <:Any, 1}; 
        normalizeGTO::Union{Bool, Missing}=missing, 
        roundDigits::Int=getAtolDigits(T)) where {T, D} -> 
    CompositeGTBasisFuncs{T, D, <:Any, 1}

    mul(a1::CompositeGTBasisFuncs{T, D, <:Any, 1}, a2::Real; 
        normalizeGTO::Union{Bool, Missing}=missing, 
        roundDigits::Int=getAtolDigits(T)) where {T, D} -> 
    CompositeGTBasisFuncs{T, D, <:Any, 1}

    mul(a1::CompositeGTBasisFuncs{T, D, <:Any, 1}, 
        a2::CompositeGTBasisFuncs{T, D, <:Any, 1}; 
        normalizeGTO::Union{Bool, Missing}=missing, 
        roundDigits::Int=getAtolDigits(T)) where {T, D} -> 
    CompositeGTBasisFuncs{T, D, <:Any, 1}

Multiplication between two `CompositeGTBasisFuncs{T, D, <:Any, 1}`s (e.g.,  
[`BasisFunc`](@ref) and [`BasisFuncMix`](@ref)), or a `Real` number and a 
`CompositeGTBasisFuncs{T, D, <:Any, 1}`. If `normalizeGTO` is set to `missing` (in 
default), The [`GaussFunc`](@ref) inside the output containers will be normalized only if 
all the input bases hold `.normalizeGTO == true`. `roundDigits` specifies the maximal 
number of digits after the radix point of the calculated values. When set to negative, no 
rounding will be performed. The function can be called using `*` syntax with the keyword 
arguments set to their default values.

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
             roundDigits::Int=getAtolDigits(T)) where {T, D, 𝑙1, 𝑙2, PT1, PT2}
    α₁ = sgf1.gauss[1].xpn()
    α₂ = sgf2.gauss[1].xpn()
    d₁ = sgf1.gauss[1].con()
    d₂ = sgf2.gauss[1].con()
    n₁ = sgf1.normalizeGTO
    n₂ = sgf2.normalizeGTO
    n₁ && (d₁ *= normOf(sgf1)[])
    n₂ && (d₂ *= normOf(sgf2)[])
    normalizeGTO isa Missing && (normalizeGTO = n₁*n₂)

    R = if (cen1 = sgf1.center) === (cen2 = sgf2.center) || hasIdentical(cen1, cen2)
        cen1
    elseif hasEqual(cen1, cen2)
        deepcopy(cen1)
    elseif (R₁ = roundNum.(coordOf(cen1), roundDigits)) == 
           (R₂ = roundNum.(coordOf(cen2), roundDigits))
        genSpatialPoint(R₁)
    else
        l1 = sgf1.l[1]
        l2 = sgf2.l[1]
        xpn, con, cen = gaussProd((α₁, d₁, R₁), (α₂, d₂, R₂))
        shiftPolyFunc = @inline (n, c1, c2) -> [(c2 - c1)^k*binomial(n,k) for k = n:-1:0]
        coeffs = map(1:3) do i
            n1 = l1[i]
            n2 = l2[i]
            c1N = shiftPolyFunc(n1, R₁[i], cen[i])
            c2N = shiftPolyFunc(n2, R₂[i], cen[i])
            m = c1N * transpose(c2N |> reverse)
            [diag(m, k)|>sum for k = n2 : (-1)^(-n1 < n2) : -n1]
        end
        lCs = cat(Ref(coeffs[1] * transpose(coeffs[2])) .* coeffs[3]..., dims=3) # TC
        cen = genSpatialPoint(roundNum.(cen, roundDigits))
        pbα = genExponent(roundNum(xpn, roundDigits))
        return BasisFuncMix(
            [BasisFunc(cen, 
                       GaussFunc(pbα, genContraction(roundNum(con*lCs[i], roundDigits))), 
                                 LTuple(i.I .- 1), 
                       normalizeGTO) 
             for i in CartesianIndices(lCs)])
    end

    xpn = roundNum(α₁ + α₂, roundDigits)
    con = roundNum(d₁ * d₂, roundDigits)
    BasisFunc(R, GaussFunc(genExponent(xpn), genContraction(con)), (sgf1.l .+ sgf2.l), 
              normalizeGTO)
end

function mul(sgf1::BasisFunc{T, D, 0, 1, PT1}, sgf2::BasisFunc{T, D, 0, 1, PT2}; 
             normalizeGTO::Union{Bool, Missing}=missing, 
             roundDigits::Int=getAtolDigits(T)) where {T, D, PT1, PT2}
    d₁ = sgf1.gauss[1].con()
    d₂ = sgf2.gauss[1].con()
    n₁ = sgf1.normalizeGTO
    n₂ = sgf2.normalizeGTO
    n₁ && (d₁ *= normOf(sgf1)[])
    n₂ && (d₂ *= normOf(sgf2)[])
    R₁ = centerCoordOf(sgf1)
    R₂ = centerCoordOf(sgf2)
    xpn, con, cen = gaussProd((sgf1.gauss[1].xpn(), d₁, R₁), (sgf2.gauss[1].xpn(), d₂, R₂))
    normalizeGTO isa Missing && (normalizeGTO = n₁*n₂)
    BasisFunc(genSpatialPoint( roundNum.(cen, roundDigits) ), 
              GaussFunc( genExponent(roundNum(xpn, roundDigits)), 
                         genContraction(roundNum(con, roundDigits)) ), 
              (LTuple(fill(0, D)),), normalizeGTO)
end

function gaussProd((α₁, d₁, R₁)::T, (α₂, d₂, R₂)::T) where 
                  {T<:Tuple{Number, Number, AbstractArray{<:Number}}}
    α = α₁ + α₂
    d = d₁ * d₂ * exp(-α₁ * α₂ / α * sum(abs2, R₁-R₂))
    R = (α₁*R₁ + α₂*R₂) / α
    (α, d, R)
end

function mul(bf::BasisFunc{T, D, 𝑙, GN}, coeff::Real; 
             normalizeGTO::Union{Bool, Missing}=missing, 
             roundDigits::Int=getAtolDigits(T)) where {T, D, 𝑙, GN}
    n = bf.normalizeGTO
    normalizeGTO isa Missing && (normalizeGTO = n)
    c = (n && !normalizeGTO) ? (coeff .* normOf(bf)) : coeff
    gfs = mul.(bf.gauss, c; roundDigits)
    BasisFunc(bf.center, gfs, bf.l, normalizeGTO)
end

mul(bfm::BasisFuncMix{T, D, BN}, coeff::Real; normalizeGTO::Union{Bool, Missing}=missing, 
    roundDigits::Int=getAtolDigits(T)) where {T, D, BN} = 
BasisFuncMix(mul.(bfm.BasisFunc, coeff; normalizeGTO, roundDigits))

function mul(bf1::BasisFunc{T, D, 𝑙1, GN1, PT1}, bf2::BasisFunc{T, D, 𝑙2, GN2, PT2}; 
             normalizeGTO::Union{Bool, Missing}=missing, 
             roundDigits::Int=getAtolDigits(T)) where {T, D, 𝑙1, 𝑙2, GN1, GN2, PT1, PT2}
    bf1n = bf1.normalizeGTO
    bf2n = bf2.normalizeGTO
    normalizeGTO isa Missing && (normalizeGTO = bf1n * bf2n)
    bs = CGTBasisFuncs1O{T, D}[]
    for gf1 in bf1.gauss, gf2 in bf2.gauss
        push!(bs, mul(BasisFunc(bf1.center, (gf1,), bf1.l, bf1n), 
                      BasisFunc(bf2.center, (gf2,), bf2.l, bf2n); 
                      normalizeGTO, roundDigits))
    end
    sumOf(bs; roundDigits)
end

mul(bf1::BasisFuncMix{T, D, 1}, bf2::BasisFunc{T, D}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundDigits::Int=getAtolDigits(T)) where 
   {T, D} = 
mul(bf1.BasisFunc[1], bf2; normalizeGTO, roundDigits)

mul(bf1::BasisFunc{T, D}, bf2::BasisFuncMix{T, D, 1}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundDigits::Int=getAtolDigits(T)) where 
   {T, D} = 
mul(bf2, bf1; normalizeGTO, roundDigits)

mul(bf::BasisFunc{T, D}, bfm::BasisFuncMix{T, D, BN}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundDigits::Int=getAtolDigits(T)) where 
   {T, D, BN} = 
sumOf(mul.(Ref(bf), bfm.BasisFunc; normalizeGTO, roundDigits); roundDigits)

mul(bfm::BasisFuncMix{T, D, BN}, bf::BasisFunc{T, D}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundDigits::Int=getAtolDigits(T)) where 
   {T, D, BN} = 
mul(bf, bfm; normalizeGTO, roundDigits)

mul(bf1::BasisFuncMix{T, D, 1}, bf2::BasisFuncMix{T, D, 1}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundDigits::Int=getAtolDigits(T)) where 
   {T, D} = 
mul(bf1.BasisFunc[1], bf2.BasisFunc[1]; normalizeGTO, roundDigits)

mul(bf::BasisFuncMix{T, D, 1}, bfm::BasisFuncMix{T, D, BN}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundDigits::Int=getAtolDigits(T)) where 
   {T, D, BN} = 
mul(bf.BasisFunc[1], bfm; normalizeGTO, roundDigits)

mul(bfm::BasisFuncMix{T, D, BN}, bf::BasisFuncMix{T, D, 1}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundDigits::Int=getAtolDigits(T)) where 
   {T, D, BN} = 
mul(bf, bfm; normalizeGTO, roundDigits)

function mul(bfm1::BasisFuncMix{T, D, BN1}, bfm2::BasisFuncMix{T, D, BN2}; 
             normalizeGTO::Union{Bool, Missing}=missing, 
             roundDigits::Int=getAtolDigits(T)) where {T, D, BN1, BN2}
    bfms = CGTBasisFuncs1O{T, D}[]
    for bf1 in bfm1.BasisFunc, bf2 in bfm2.BasisFunc
        push!(bfms, mul(bf1, bf2; normalizeGTO, roundDigits))
    end
    sumOf(bfms; roundDigits)
end

mul(::EmptyBasisFunc{<:Any, D}, ::T; 
    normalizeGTO=missing, roundDigits::Int=getAtolDigits(T)) where {D, T<:Real} = 
EmptyBasisFunc{T, D}()

mul(::T, ::EmptyBasisFunc{<:Any, D}; 
    normalizeGTO=missing, roundDigits::Int=getAtolDigits(T)) where {T<:Real, D} = 
EmptyBasisFunc{T, D}()

function mul(b::CGTBasisFuncs1O{T, D}, coeff::Real; 
             normalizeGTO::Union{Bool, Missing}=missing, 
             roundDigits::Int=getAtolDigits(T)) where {T, D}
    iszero(coeff) ? EmptyBasisFunc{T, D}() : mul(b, coeff; normalizeGTO, roundDigits)
end

mul(coeff::Real, b::CGTBasisFuncs1O{T}; normalizeGTO::Union{Bool, Missing}=missing, 
    roundDigits::Int=getAtolDigits(T)) where {T} = 
mul(b, coeff; normalizeGTO, roundDigits)

mul(::EmptyBasisFunc{<:Any, D}, ::CGTBasisFuncs1O{T, D}; 
    normalizeGTO=missing, roundDigits::Int=getAtolDigits(T)) where {D, T} = 
EmptyBasisFunc{T, D}()

mul(::CGTBasisFuncs1O{T, D}, ::EmptyBasisFunc{<:Any, D}; 
    normalizeGTO=missing, roundDigits::Int=getAtolDigits(T)) where {T, D} = 
EmptyBasisFunc{T, D}()

mul(::EmptyBasisFunc{T1, D}, ::EmptyBasisFunc{T2, D}; normalizeGTO=missing, 
    roundDigits::Int=getAtolDigits(promote_type(T1, T2))) where {T1, T2, D} = 
EmptyBasisFunc{promote_type(T1, T2), D}()

mul(bf1::BFuncs1O{T, D, 𝑙1, GN1, PT1}, bf2::BFuncs1O{T, D, 𝑙2, GN2, PT2}; 
    normalizeGTO::Union{Bool, Missing}=missing, roundDigits::Int=getAtolDigits(T)) where 
   {T, D, 𝑙1, 𝑙2, GN1, GN2, PT1, PT2} = 
[mul(add(bf1), add(bf2); normalizeGTO, roundDigits)]


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

shiftCore(::typeof(+), bf::FGTBasisFuncs1O{<:Any, D, 𝑙1, GN}, dl::LTuple{D, 𝑙2}) where 
         {D, 𝑙1, 𝑙2, GN} = 
BasisFunc(bf.center, bf.gauss, bf.l[1]+dl, bf.normalizeGTO)

shiftCore(::typeof(-), bf::FGTBasisFuncs1O{<:Any, D, 0, GN}, ::LTuple{D, 0}) where {D, GN} = 
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

shiftCore(::Function, ::EmptyBasisFunc{T, D}, ::LTuple{D}) where {T, D} = 
EmptyBasisFunc{T, D}()


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
inside `bf`. When set to negative, no rounding will be performed.
"""
function genBasisFuncText(bf::FloatingGTBasisFuncs{T, D}; norm::Real=1.0, 
                          printCenter::Bool=true, roundDigits::Int=-1) where {T, D}
    GFs = map(x -> genGaussFuncText(x.xpn(), x.con(); roundDigits), bf.gauss)
    cen = centerCoordOf(bf)
    firstLine = printCenter ? "X "*(alignNum.(cen; roundDigits) |> join)*"\n" : ""
    firstLine * "$(bf|>subshellOf)    $(getTypeParams(bf)[4])   $(T(norm))" * 
    ( isaFullShellBasisFuncs(bf) ? "" : " " * 
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
    bfBlocks = sortBasisFuncs(bs, groupCenters; roundDigits)
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
                      unlinkCenter::Bool=false) where {N, F<:Function} -> 
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
                           unlinkCenter::Bool=false) where {D, T<:AbstractFloat}
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
        if oInfo[1] == "SP"
            gs2 = GaussFunc{typ}[]
            for j = i+1 : i+ng
                push!(gs1, GaussFunc(data[j][1], normFactor*data[j][2]))
                push!(gs2, GaussFunc(data[j][1], normFactor*data[j][3]))
            end
            append!(bfs, genBasisFunc.(Ref(unlinkCenter ? deepcopy(center) : center), 
                                       [gs1, gs2], ["S", "P"], normalizeGTO=true))
        else
            for j = i+1 : i+ng
                push!(gs1, GaussFunc(data[j]...))
            end
            subshellInfo = oInfo[1] |> string
            if length(oInfo) > 3
                subshellInfo = SubshellOrientationList[d][subshellInfo][oInfo[2:end]]
            end
            push!(bfs, genBasisFunc((unlinkCenter ? deepcopy(center) : center), 
                                    gs1, subshellInfo, normalizeGTO=true))
        end
        center = centerOld
    end
    bfs |> flatten
end

"""

    assignCenInVal!(center::AbstractVector{<:Real}, b::FloatingGTBasisFuncs{T, D}) -> 
    SpatialPoint{T, D}

Change the input value of data stored in `b.center` (meaning the output value will also 
change according to the mapping function). Then, return the altered center.
"""
function assignCenInVal!(center::AbstractVector{<:Real}, b::FloatingGTBasisFuncs)
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
filter(x->paramFilter(x, symbol, forDifferentiation), pbc.param) |> collect

getParams(cs::AbstractArray{<:ParameterizedContainer}, 
          symbol::Union{Symbol, Missing}=missing; forDifferentiation::Bool=false) = 
vcat(getParams.(cs, symbol; forDifferentiation)...)

function getParams(cs::AbstractArray, symbol::Union{Symbol, Missing}=missing; 
                   forDifferentiation::Bool=false)
    pbIdx = findall(x->x isa ParamBox, cs)
    vcat(getParams(convert(Vector{ParamBox}, cs[pbIdx]), symbol; forDifferentiation), 
         getParams(convert(Vector{ParameterizedContainer}, cs[1:end .∉ [pbIdx]]), symbol; 
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

function markParams!(parArray::AbstractVector{<:ParamBox}, filterMapping::Bool=false)
    pars = eltype(parArray)[]
    syms = getUnique!(outSymOfCore.(parArray))
    arr = parArray |> copy
    for sym in syms
        typ = ParamBox{<:Any, sym}
        subArr = typ[]
        ids = Int[]
        for (i, val) in enumerate(arr)
            if val isa typ
                push!(subArr, val)
                push!(ids, i)
            end
        end
        deleteat!(arr, ids)
        append!(pars, markParamsCore!(subArr))
    end
    filterMapping ? unique(x->(objectid(x.data), x.index[]), pars) : pars
end

function markParamsCore!(parArray::AbstractVector{<:ParamBox{<:Any, V}}) where {V}
    res, _ = markUnique(parArray, compareFunction=compareParamBox)
    for (idx, i) in zip(parArray, res)
        idx.index[] = i
    end
    parArray
end


getNijk(::Type{T}, i::Integer, j::Integer, k::Integer) where {T} = 
    (T(2)/π)^T(0.75) * sqrt( 2^(3*(i+j+k)) * factorial(i) * factorial(j) * factorial(k) / 
    (factorial(2i) * factorial(2j) * factorial(2k)) )

getNα(i::Integer, j::Integer, k::Integer, α::T) where {T} = 
α^T(0.5*(i + j + k) + 0.75)

getNijkα(i::Integer, j::Integer, k::Integer, α::T) where {T} = 
getNijk(T, i, j, k) * getNα(i, j, k, α)

getNijkα(ijk, α) = getNijkα(ijk[1], ijk[2], ijk[3], α)

"""

    normOf(b::FloatingGTBasisFuncs{T, 3}) where {T} -> Array{T}

Return the normalization factors of the Gaussian-type orbitals (GTO) inside the input `b`. 
Each column corresponds to one orbital.
"""
normOf(b::FGTBasisFuncs1O{T, 3, 𝑙, GN})  where {T, 𝑙, GN} = 
getNijkα.(b.l[1]..., T[g.xpn() for g in b.gauss])

normOf(b::FloatingGTBasisFuncs{<:Any, 3, <:Any, <:Any, <:Any, ON}) where {ON} = 
hcat(normOf.(b)...)