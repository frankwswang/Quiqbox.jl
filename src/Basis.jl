export GaussFunc, genExponent, genContraction, SpatialPoint, genSpatialPoint, BasisFunc, 
       BasisFuncs, genBasisFunc, lOf, subshellOf, centerOf, centerCoordOf, dimOf, GTBasis, 
       sortBasisFuncs, sortPermBasisFuncs, sortBasis, sortPermBasis, add, mul, shift, 
       decompose, basisSize, genBasisFuncText, genBFuncsFromText, assignCenInVal!, 
       getParams, copyBasis, markParams!

export P1D, P2D, P3D

using LinearAlgebra: diag
using ForwardDiff: derivative as ForwardDerivative

"""

    GaussFunc{T, FL1, FL2} <: AbstractGaussFunc{T}

A single contracted gaussian function `struct` from package Quiqbox.

≡≡≡ Field(s) ≡≡≡

`xpn::ParamBox{T, :$(xpnSym), FL1}`：Exponent of the gaussian function.

`con::ParamBox{T, :$(conSym), FL2}`: Contraction coefficient of the gaussian function.

`param::NTuple{2, ParamBox}`: A Tuple that stores the `ParamBox`s of `xpn` and `con`.

≡≡≡ Initialization Method(s) ≡≡≡

    GaussFunc(xpn::Union{AbstractFloat, ParamBox}, con::Union{AbstractFloat, ParamBox}) -> 
    GaussFunc

"""
struct GaussFunc{T, FLxpn, FLcon} <: AbstractGaussFunc{T}
    xpn::ParamBox{T, xpnSym, FLxpn}
    con::ParamBox{T, conSym, FLcon}
    param::Tuple{ParamBox{T, xpnSym, FLxpn}, ParamBox{T, conSym, FLcon}}

    GaussFunc(xpn::ParamBox{T, xpnSym, FL1}, con::ParamBox{T, conSym, FL2}) where 
             {T<:AbstractFloat, FL1, FL2} = 
    new{T, FL1, FL2}(xpn, con, (xpn, con))
end

GaussFunc(e::Union{AbstractFloat, ParamBox}, d::Union{AbstractFloat, ParamBox}) = 
GaussFunc(genExponent(e), genContraction(d))


"""

    genExponent(e::T, mapFunction::Function; canDiff::Bool=true, 
                dataName::Symbol=:undef) where {T<:AbstractFloat} -> 
    ParamBox{T, :$(xpnSym)}

    genExponent(e::Array{T, 0}, mapFunction::Function; canDiff::Bool=true, 
                dataName::Symbol=:undef) where {T<:AbstractFloat} -> 
    ParamBox{T, :$(xpnSym)}

Construct a `ParamBox` for an exponent coefficient given a value. Keywords `mapFunction` 
and `canDiff` work the same way as in a general constructor of a `ParamBox`.
"""
genExponent(e::AbstractFloat, mapFunction::F; 
            canDiff::Bool=true, dataName::Symbol=:undef) where {F<:Function} = 
ParamBox(Val(xpnSym), mapFunction, fill(e), genIndex(nothing), fill(canDiff), dataName)

genExponent(e::Array{T, 0}, mapFunction::F; canDiff::Bool=true, 
            dataName::Symbol=:undef) where {T<:AbstractFloat, F<:Function} = 
ParamBox(Val(xpnSym), mapFunction, e, genIndex(nothing), fill(canDiff), dataName)

"""

    genExponent(e::T) where {T<:AbstractFloat} -> ParamBox{T, :$(xpnSym)}

    genExponent(e::Array{T, 0}) where {T<:Real} -> ParamBox{T, :$(xpnSym)}

"""
genExponent(e::AbstractFloat) = ParamBox(Val(xpnSym), itself, fill(e), genIndex(nothing))

genExponent(e::Array{T, 0}) where {T<:AbstractFloat} = 
ParamBox(Val(xpnSym), itself, e, genIndex(nothing))

"""

    genExponent(pb::ParamBox{T}) where {T<:AbstractFloat} -> ParamBox{T, :$(xpnSym)}

Convert a `$(ParamBox)` to an exponent coefficient parameter.
"""
genExponent(pb::ParamBox{T, <:Any, F}) where {T<:AbstractFloat, F} = 
ParamBox(Val(xpnSym), pb)


"""

    genContraction(c::T, mapFunction::Function; canDiff::Bool=true, 
                   dataName::Symbol=:undef) where {T<:AbstractFloat} -> 
    ParamBox{T, :$(conSym)}

    genContraction(c::Array{T, 0}, mapFunction::Function; canDiff::Bool=true, 
                   dataName::Symbol=:undef) where {T<:AbstractFloat} -> 
    ParamBox{T, :$(conSym)}

Construct a `ParamBox` for an contraction coefficient given a value. Keywords `mapFunction` 
and `canDiff` work the same way as in a general constructor of a `ParamBox`.
"""
genContraction(c::AbstractFloat, mapFunction::F; 
               canDiff::Bool=true, dataName::Symbol=:undef) where {F<:Function} = 
ParamBox(Val(conSym), mapFunction, fill(c), genIndex(nothing), fill(canDiff), dataName)

genContraction(c::Array{T, 0}, mapFunction::F; canDiff::Bool=true, 
               dataName::Symbol=:undef) where {T<:AbstractFloat, F<:Function} = 
ParamBox(Val(conSym), mapFunction, c, genIndex(nothing), fill(canDiff), dataName)

"""

    genContraction(c::T) where {T<:AbstractFloat} -> ParamBox{T, :$(conSym)}

    genContraction(c::Array{T, 0}) where {T<:AbstractFloat} -> ParamBox{T, :$(conSym)}

"""
genContraction(c::AbstractFloat) = ParamBox(Val(conSym), itself, fill(c), genIndex(nothing))

genContraction(c::Array{T, 0}) where {T<:AbstractFloat} = 
ParamBox(Val(conSym), itself, c, genIndex(nothing))

"""

    genContraction(pb::ParamBox{T}) where {T<:AbstractFloat} -> ParamBox{T, :$(conSym)}

Convert a `$(ParamBox)` to an exponent coefficient parameter.
"""
genContraction(pb::ParamBox{T, <:Any, F}) where {T<:AbstractFloat, F} = 
ParamBox(Val(conSym), pb)


const Doc_genSpatialPoint_Eg1 = "SpatialPoint{3, Float64, "*
                                "Tuple{FI, FI, FI}}"*
                                "(param)[1.0, 2.0, 3.0][∂][∂][∂]"

const P1D{T, Lx} = Tuple{ParamBox{T, cxSym, FLevel{Lx}}}
const P2D{T, Lx, Ly} = Tuple{ParamBox{T, cxSym, FLevel{Lx}}, 
                             ParamBox{T, cySym, FLevel{Ly}}}
const P3D{T, Lx, Ly, Lz} = Tuple{ParamBox{T, cxSym, FLevel{Lx}}, 
                                 ParamBox{T, cySym, FLevel{Ly}}, 
                                 ParamBox{T, czSym, FLevel{Lz}}}

const SPointTL{T} = Union{P1D{T}, P2D{T}, P3D{T}}

const SPointTU{T, D} = Tuple{Vararg{ParamBox{T, V, FL} where {V, FL<:FLevel}, D}}

struct SpatialPoint{T, D, PT} <: AbstractSpatialPoint{T, D}
    param::PT
    function SpatialPoint(pbs::SPointTU{T, D}) where {T, D}
        pbsT = typeof(pbs)
        @assert  pbsT <: SPointTL{T}
        new{T, D, pbsT}(pbs)
    end
end

"""

    genSpatialPoint(point::Union{Tuple{Vararg{AbstractFloat}}, 
                                       AbstractArray{<:AbstractFloat}}, 
                    mapFunction::F=itself; canDiff::Bool=true, dataName::Symbol=:undef) -> 
    SpatialPoint

    genSpatialPoint(point::Union{Tuple{Vararg{Array{<:AbstractFloat, 0}}}, 
                                 AbstractArray{<:Array{<:AbstractFloat, 0}}}, 
                    mapFunction::F=itself; canDiff::Bool=true, dataName::Symbol=:undef) -> 
    SpatialPoint

Return the parameter(s) of a spatial coordinate in terms of `ParamBox`. Keywords 
`mapFunction` and `canDiff` work the same way as in a general constructor of a `ParamBox`. 
If `roundDigits < 0` or `point` is a 0-d `Array`, there won't be rounding for input data.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> v1 = [1.0,2,3]
3-element Vector{Float64}:
 1.0
 2.0
 3.0

julia> genSpatialPoint(v1)
$(Doc_genSpatialPoint_Eg1)

julia> v2 = [fill(1.0), 2.0, 3.0]
3-element Vector{Any}:
  fill(1.0)
 2.0
 3.0

julia> p2 = genSpatialPoint(v2); p2[1]
ParamBox{Float64, :X, $(FI)}(1.0)[∂][X]

julia> v2[1][] = 1.2
1.2

julia> p2[1]
ParamBox{Float64, :X, $(FI)}(1.2)[∂][X]
```
"""
genSpatialPoint(v::AbstractArray, optArgs...) = genSpatialPoint(Tuple(v), optArgs...)
genSpatialPoint(v::NTuple{N, Any}, optArgs...) where {N} = 
genSpatialPoint.(v, Tuple([1:N;]), optArgs...) |> genSpatialPointCore

"""

    genSpatialPoint(comp::T, index::Int, mapFunction::F; canDiff::Bool=true, 
                    dataName::Symbol=:undef) where {T<:AbstractFloat} -> 
    ParamBox{T}

    genSpatialPoint(comp::Array{T, 0}, index::Int,mapFunction::F; canDiff::Bool=true, 
                    dataName::Symbol=:undef) where {T<:AbstractFloat} -> 
    ParamBox{T}

    genSpatialPoint(comp::T, index::Int) where {T<:AbstractFloat} -> ParamBox{T}

    genSpatialPoint(comp::Array{T, 0}, index::Int) -> ParamBox{T}

    genSpatialPoint(comp::ParamBox{T}, index::Int) -> ParamBox{T}

Return the component of a `SpatialPoint` given its value (or 0-D container) and index.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genSpatialPoint(1.2, 1)
ParamBox{Float64, :X, $(FI)}(1.2)[∂][X]

julia> pointY1 = fill(2.0)
0-dimensional Array{Float64, 0}:
2.0

julia> Y1 = genSpatialPoint(pointY1, 2)
ParamBox{Float64, :Y, $(FI)}(2.0)[∂][Y]

julia> pointY1[] = 1.5
1.5

julia> Y1
ParamBox{Float64, :Y, $(FI)}(1.5)[∂][Y]
```
"""
genSpatialPoint(comp::AbstractFloat, index::Int, mapFunction::F; 
                canDiff::Bool=true, dataName::Symbol=:undef) where {F<:Function} = 
ParamBox(Val(SpatialParamSyms[index]), mapFunction, fill(comp), 
         genIndex(nothing), fill(canDiff), dataName)

genSpatialPoint(comp::Array{T, 0}, index::Int, mapFunction::F; 
                canDiff::Bool=true, dataName::Symbol=:undef) where 
               {T<:AbstractFloat, F<:Function} = 
ParamBox(Val(SpatialParamSyms[index]), mapFunction, comp, genIndex(nothing), fill(canDiff), 
         dataName)

genSpatialPoint(comp::AbstractFloat, index::Int) = 
ParamBox(Val(SpatialParamSyms[index]), itself, fill(comp), genIndex(nothing))

genSpatialPoint(comp::Array{T, 0}, index::Int) where {T<:AbstractFloat} = 
ParamBox(Val(SpatialParamSyms[index]), itself, comp, genIndex(nothing))

genSpatialPoint(point::ParamBox, index::Int) = ParamBox(Val(SpatialParamSyms[index]), point)

"""

    genSpatialPoint(point::Union{Tuple{Vararg{ParamBox}}, AbstractArray{<:ParamBox}}) -> 
    SpatialPoint

Convert a collection of `$(ParamBox)`s to a spatial point.
"""
genSpatialPoint(point::NTuple{N, ParamBox}) where {N} = 
ParamBox.(Val.(SpatialParamSyms[1:N]), point) |> genSpatialPointCore

genSpatialPointCore(point::Union{P1D, P2D, P3D}) = SpatialPoint(point)


coordOf(sp::SpatialPoint) = [outValOf(i) for i in sp.param]


"""

    BasisFunc{T, D, 𝑙, GN, PT} <: FloatingGTBasisFuncs{T, D, 𝑙, GN, PT, 1}

A (floating) basis function with the center attached to it instead of any nucleus.

≡≡≡ Field(s) ≡≡≡

`center::SpatialPoint{T, D, PT}`: The `D`-dimensional center coordinate.

`gauss::NTuple{N, GaussFunc{T, <:Any}}`: Gaussian functions within the basis function.

`l::Tuple{$(LTuple){D, 𝑙}}`: Cartesian representation (pseudo-quantum number) of the 
angular momentum orientation. E.g., s (X⁰Y⁰Z⁰) would be `$(LTuple(0, 0, 0))`. For 
convenient syntax, `.l[]` converts it to a `NTuple{3, Int}`.

`normalizeGTO::Bool`: Whether the GTO`::GaussFunc` will be normalized in calculations.

`param::NTuple{D+GN*2, ParamBox}`： All the tunable parameters`::ParamBox` stored in the 
`BasisFunc`.

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

BasisFunc(cen, gs::NTuple{GN, GaussFunc}, l::LTuple, normalizeGTO=false) where {GN} = 
BasisFunc(cen, gs, (l,), normalizeGTO)

BasisFunc(cen, g::GaussFunc, l, normalizeGTO=false) = 
BasisFunc(cen, (g,), l, normalizeGTO)

BasisFunc(bf::BasisFunc) = itself(bf)


"""

    BasisFuncs{T, D, 𝑙, GN, PT, ON} <: FloatingGTBasisFuncs{T, D, 𝑙, GN, PT, ON}

A group of basis functions with identical parameters except they have different 
orientations in the specified subshell. It has the same fields as `BasisFunc` and 
specifically, for `l`, the size of the it (`ON`) can be no less than 1 (and no larger 
than the size of the corresponding subshell).
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
        ls = sort(collect(ls), rev=true) |> Tuple
        pars = joinTuple(cen.param, getproperty.(gs, :param)...)
        new{T, D, 𝑙, GN, PT, ON}(cen, gs, ls, normalizeGTO, pars)
    end
end

const BFuncs1O{T, D, 𝑙, GN, PT} = BasisFuncs{T, D, 𝑙, GN, PT, 1}
const BFuncsON{ON} = BasisFuncs{<:Any, <:Any, <:Any, <:Any, <:Any, ON}

BasisFuncs(cen, g::GaussFunc, ls, normalizeGTO=false) = 
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

    genBasisFunc(center::Union{AbstractArray{T}, Tuple{Vararg{T}, SpatialPoint}, Missing}, 
                 args..., kws...) where {T<:Union{Real, ParamBox}} -> 
    B where {B<:Union{FloatingGTBasisFuncs, Array{<:FloatingGTBasisFuncs}}}

Constructor of `BasisFunc` and `BasisFuncs`, but it also returns different kinds of 
collections of them based on the applied methods. The first argument `center` can be a 3-D 
coordinate (e.g. `Vector{Float64}`), a `NTuple{3}` of spatial points (e.g. generated by 
`genSpatialPoint`), or simply set to `missing` for later assignment.

≡≡≡ Method 1 ≡≡≡

    genBasisFunc(center, gs::Array{GaussFunc, 1}, 
                 lOrls::Union{T, Array{T, 1}, NTuple{<:Any, T}}; 
                 normalizeGTO::Bool=false) where {T <: NTuple{3, Int}}

`lOrls` is the Array of the pseudo-quantum number(s) to specify the angular 
momentum(s). E.g., s is (0,0,0) and p is ((1,0,0), (0,1,0), (0,0,1)).

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genBasisFunc([0,0,0], GaussFunc(2,1), (0,1,0))
BasisFunc{1, 1}(center, gauss)[X⁰Y¹Z⁰][0.0, 0.0, 0.0]
```

≡≡≡ Method 2 ≡≡≡

    genBasisFunc(center, gExpsANDgCons::NTuple{2, Array{<:Real, 1}}, subshell="S"; kw...)

Instead of directly inputting `GaussFunc`, one can also input a 2-element `Tuple` of the 
exponent(s) and contraction coefficient(s) corresponding to the same `GaussFunc`(s).

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genBasisFunc([0,0,0], (2, 1), "P")
BasisFuncs{1, 1, 3}(center, gauss)[3/3][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], ([2, 1.5], [1, 0.5]), "P")
BasisFuncs{1, 2, 3}(center, gauss)[3/3][0.0, 0.0, 0.0]
```

≡≡≡ Method 3 ≡≡≡

    genBasisFunc(center, gs::Union{GaussFunc, Array{GaussFunc, 1}}, subshell::String="S", 
                 lFilter::NTuple{N, Bool}=fill(true, SubshellSizeList[3][subshell])|>Tuple; 
                 normalizeGTO::Bool=false) where {N}

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genBasisFunc([0,0,0], GaussFunc(2,1), "S")
BasisFunc{0, 1}(center, gauss)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], GaussFunc(2,1), "P")
BasisFuncs{1, 1, 3}(center, gauss)[3/3][0.0, 0.0, 0.0]
```

≡≡≡ Method 4 ≡≡≡

    genBasisFunc(center, BSKeyANDnuc::Array{Tuple{String, String}, 1})

If the user wants to construct existed atomic basis set(s), they can use the (`Array` of) 
`(BS_name, Atom_name)` as the second input. If the atom is omitted, then basis set for H 
is used.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genBasisFunc([0,0,0], ("STO-3G", "Li"))
3-element Vector{Quiqbox.FloatingGTBasisFuncs}:
 BasisFunc{0, 3}(center, gauss)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFunc{0, 3}(center, gauss)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFuncs{1, 3, 3}(center, gauss)[3/3][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], "STO-3G")
1-element Vector{Quiqbox.FloatingGTBasisFuncs}:
 BasisFunc{0, 3}(center, gauss)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], ["STO-2G", "STO-3G"])
2-element Vector{Quiqbox.FloatingGTBasisFuncs}:
 BasisFunc{0, 2}(center, gauss)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFunc{0, 3}(center, gauss)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], [("STO-2G", "He"), ("STO-3G", "O")])
4-element Vector{Quiqbox.FloatingGTBasisFuncs}:
 BasisFunc{0, 2}(center, gauss)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFunc{0, 3}(center, gauss)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFunc{0, 3}(center, gauss)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFuncs{1, 3, 3}(center, gauss)[3/3][0.0, 0.0, 0.0]
```
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
    @compareLength xpnsANDcons[1] xpnsANDcons[2] "exponents" "contraction coefficients"
    genBasisFunc(cen, GaussFunc.(xpnsANDcons[1], xpnsANDcons[2]), lOrSubshell; 
                 normalizeGTO)
end

genBasisFunc(cen::SpatialPoint{T, D}, xpnANDcon::NTuple{2, T}, 
             lOrSubshell=LTuple(fill(0, D)); normalizeGTO::Bool=false) where {T, D} = 
genBasisFunc(cen, (GaussFunc(xpnANDcon[1], xpnANDcon[2]),), lOrSubshell; normalizeGTO)

function genBasisFunc(center::SpatialPoint{T, D}, BSKeyANDnuc::AbstractVector{NTuple{2, String}}; 
                      unlinkCenter::Bool=false) where {T, D}
    bases = FloatingGTBasisFuncs{T, D}[]
    for k in BSKeyANDnuc
        content = BasisSetList[k[1]][AtomicNumberList[k[2]]]
        @assert content!==nothing "Quiqbox DOES NOT have basis set "*k[1]*" for "*k[2]*"."
        append!(bases, genBFuncsFromText(content; adjustContent=true, 
                excludeLastNlines=1, center, unlinkCenter))
    end
    bases
end

genBasisFunc(cen::SpatialPoint{T, D}, gs::Tuple, subshell::String, lFilter::Tuple{Vararg{Bool}}; 
             normalizeGTO::Bool=false) where {T, D} = 
genBasisFunc(cen, gs, SubshellOrientationList[D][subshell][1:end .∈ [findall(lFilter)]]; 
             normalizeGTO)

genBasisFunc(cen::SpatialPoint, BSKeyANDnuc::NTuple{2, String}; 
             unlinkCenter::Bool=false) = 
genBasisFunc(cen, [BSKeyANDnuc]; unlinkCenter)

genBasisFunc(cen::SpatialPoint, BSkey::AbstractVector{String}; nucleus::String="H", 
             unlinkCenter::Bool=false) = 
genBasisFunc(cen, [(i, nucleus) for i in BSkey]; unlinkCenter)

genBasisFunc(cen::SpatialPoint, BSkey::String; nucleus::String="H", 
             unlinkCenter::Bool=false) = 
genBasisFunc(cen, [BSkey]; nucleus, unlinkCenter)

# A few methods for convenient arguments omissions and mutations.
genBasisFunc(cen::SpatialPoint{T}, gs::AbstractArray{<:AbstractGaussFunc{T}}, 
             args...; kws...) where {T} = 
genBasisFunc(cen, gs|>Tuple, args...; kws...)

genBasisFunc(cen::SpatialPoint{T}, g::GaussFunc{T}, args...; kws...) where {T} = 
genBasisFunc(cen, (g,), args...; kws...)

genBasisFunc(coord::Union{Tuple, AbstractArray}, args...; kws...) = 
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

Return the total angular momentum (in Cartesian coordinate system).
"""
lOf(::FloatingGTBasisFuncs{<:Any, <:Any, 𝑙}) where {𝑙} = 𝑙


"""

    subshellOf(::FloatingGTBasisFuncs) -> String

Return the subshell name.
"""
subshellOf(b::FloatingGTBasisFuncs) = SubshellNames[lOf(b)+1]


"""

    sortBasisFuncs(bs::AbstractArray{<:FloatingGTBasisFuncs{T, D}}, 
                   groupCenters::Bool=false; roundDigits::Int=getAtolDigits(T)) where 
                  {T, D} -> 
    Vector

Sort `FloatingGTBasisFuncs`. If `groupCenters = true`, Then the function will return an 
`Array{<:Array{<:FloatingGTBasisFuncs, 1}, 1}` in which the arrays are grouped basis 
functions with same center coordinates.
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

Return a `Vector` of indices `I` such that `bs[I] == sortBasisFuncs(bs; roundDigits)`.
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

    centerOf(bf::FloatingGTBasisFuncs) -> 
    Tuple{ParamBox{Float64, $(cxSym)}, 
          ParamBox{Float64, $(cySym)}, 
          ParamBox{Float64, $(czSym)}}

Return the center of the input `FloatingGTBasisFuncs`.
"""
centerOf(bf::FloatingGTBasisFuncs) = bf.center


"""

    centerCoordOf(bf::FloatingGTBasisFuncs) -> Vector

Return the center coordinate of the input `FloatingGTBasisFuncs`.
"""
centerCoordOf(bf::FloatingGTBasisFuncs) = coordOf(bf.center)

"""

    BasisFuncMix{T, D, BN, BT<:BasisFunc{T, D}} <: CompositeGTBasisFuncs{T, D, BN, 1}

Sum of multiple `FloatingGTBasisFuncs{<:Any, <:Any, <:Any, <:Any, <:Any, 1}` without any 
reformulation, treated as one basis function in the integral calculation.

≡≡≡ Field(s) ≡≡≡

`BasisFunc::NTuple{BN, BT}`: Inside basis functions.

`param::Tuple{Vararg{ParamBox}}`: Contained parameters.

≡≡≡ Initialization Method(s) ≡≡≡

    BasisFuncMix(bfs::Union{Tuple{Vararg{T}}, Vector{T}}) where 
                {T<:FloatingGTBasisFuncs{<:Any, <:Any, <:Any, <:Any, <:Any, 1}} ->
    BasisFuncMix

"""
struct BasisFuncMix{T, D, BN, BT<:BasisFunc{T, D}} <: CGTBasisFuncs1O{T, D, BN}
    BasisFunc::NTuple{BN, BT}
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
getTypeParams(::BasisFuncMix{T, D, BN, BT}) where {T, D, BN, BT} = (T, D, BN, BT)


unpackBasis(::EmptyBasisFunc) = ()
unpackBasis(b::BasisFunc)  = (b,)
unpackBasis(b::BasisFuncMix)  = b.BasisFunc
unpackBasis(b::BFuncs1O)  = (BasisFunc(b),)


"""

    dimOf(::AbstractSpatialPoint) -> Int

Return the spatial dimension of the input `AbstractSpatialPoint`.
"""
dimOf(::AbstractSpatialPoint{<:Any, D}) where {D} = D

"""

    dimOf(::QuiqboxBasis) -> Int

Return the spatial dimension of the input basis.
"""
dimOf(::QuiqboxBasis{<:Any, D}) where {D} = D


"""

    GTBasis{T, D, BN, BT<:GTBasisFuncs{1, T, D}} <: BasisSetData{T, D, BT}

The container to store basis set information.

≡≡≡ Field(s) ≡≡≡

`basis::NTuple{BN, BT}`: Basis set.

`S::Matrix{T}`: Overlap matrix.

`Te::Matrix{T}`: Kinetic energy part of the electronic core Hamiltonian.

`eeI::Array{T, 4}`: Electron-electron interaction.

≡≡≡ Initialization Method(s) ≡≡≡

    GTBasis(basis::Union{Tuple{Vararg{GTBasisFuncs{T, D}}}, 
                         AbstractVector{<:GTBasisFuncs{T, D}}}) where {T, D} -> 
    GTBasis

Construct a `GTBasis` given a basis set.
"""
struct GTBasis{T, D, BN, BT<:GTBasisFuncs{T, D, 1}} <: BasisSetData{T, D, BT}
    basis::NTuple{BN, BT}
    S::Matrix{T}
    Te::Matrix{T}
    eeI::Array{T, 4}

    GTBasis(bfs::Tuple{Vararg{GTBasisFuncs{T, D, 1}, BN}}) where {T<:Real, D, BN} = 
    new{T, D, BN, eltype(bfs)}(bfs, overlaps(bfs), eKinetics(bfs), eeInteractions(bfs))
end

GTBasis(bs::Tuple{Vararg{GTBasisFuncs{T, D}}}) where {T, D} = GTBasis(bs |> flatten)

GTBasis(bs::AbstractVector{<:GTBasisFuncs{T, D}}) where {T, D} = GTBasis(bs |> Tuple)


"""

    sortBasis(bs::Union{AbstractArray{<:CompositeGTBasisFuncs{T, D}}, 
                        Tuple{Vararg{CompositeGTBasisFuncs{T, D}}}}; 
              roundDigits::Int=getAtolDigits(T)) where {T, D} -> 
    Vector{<:CompositeGTBasisFuncs{T, D}}

Sort basis functions.
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

Reconstruct a `GTBasis` by sorting the `GTBasisFuncs` stored in the input one.
"""
sortBasis(b::GTBasis; roundDigits::Int=getAtolDigits(T)) = 
          GTBasis(sortBasis(b.basis; roundDigits))


"""

    sortPermBasis(bs::AbstractArray{<:CompositeGTBasisFuncs{T, D}}; 
                  roundDigits::Int=getAtolDigits(T)) where {T, D} -> 
    Vector{Int}

Return a `Vector` of indices `I` such that `bs[I] == sortBasis(bs; roundDigits)`.
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

    add(b::CompositeGTBasisFuncs{T, D, <:Any, 1}; roundDigits::Int=getAtolDigits(T)) where 
       {T, D} -> 
    CompositeGTBasisFuncs{T, D, <:Any, 1}

    add(b1::CompositeGTBasisFuncs{T, D, <:Any, 1}, 
        b2::CompositeGTBasisFuncs{T, D, <:Any, 1}; roundDigits::Int=getAtolDigits(T)) where 
       {T, D} -> 
    CompositeGTBasisFuncs{T, D, <:Any, 1}

Addition between `CompositeGTBasisFuncs{T, D, <:Any, 1}` such as `BasisFunc` and 
`Quiqbox.BasisFuncMix`. `roundDigits` specifies the maximal number of digits after the 
radix point of the calculated values. The function can be called using `+` syntax with 
the keyword argument set to it default value.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> bf1 = genBasisFunc([1,1,1], (2,1))
BasisFunc{0, 1}(center, gauss)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bf2 = genBasisFunc([1,1,1], (2,2))
BasisFunc{0, 1}(center, gauss)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bf3 = bf1 + bf2
BasisFunc{0, 1}(center, gauss)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bf3.gauss[1].con[]
3.0
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
                res = if gf1 === gf2 || hasIndentical(gf1, gf2)
                    gf1
                elseif hasEuqal(gf1, gf2)
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


const Doc_mul_Eg1 = "GaussFunc(xpn=ParamBox{Float64, :α, $(FI)}(3.0)[∂][α], " * 
                              "con=ParamBox{Float64, :d, $(FI)}(1.0)[∂][d])"

const Doc_mul_Eg2 = "GaussFunc(xpn=ParamBox{Float64, :α, $(FI)}(3.0)[∂][α], " * 
                              "con=ParamBox{Float64, :d, $(FI)}(2.0)[∂][d])"

const Doc_mul_Eg3 = "GaussFunc(xpn=ParamBox{Float64, :α, $(FI)}(6.0)[∂][α], " * 
                              "con=ParamBox{Float64, :d, $(FI)}(1.0)[∂][d])"

const Doc_mul_Eg4 = "GaussFunc(xpn=ParamBox{Float64, :α, $(FI)}(6.0)[∂][α], " * 
                              "con=ParamBox{Float64, :d, $(FI)}(2.0)[∂][d])"

"""

    mul(gf::GaussFunc{T}, coeff::Real; roundDigits::Int=getAtolDigits(T)) where {T} -> 
    GaussFunc

    mul(coeff::Real, gf::GaussFunc{T}; roundDigits::Int=getAtolDigits(T)) where {T} -> 
    GaussFunc

    mul(gf1::GaussFunc{T}, gf2::GaussFunc{T}; 
        roundDigits::Int=getAtolDigits(T)) where {T} -> 
    GaussFunc

Multiplication between `GaussFunc`s or contraction coefficient multiplication between a 
`Real` number and a `GaussFunc`. `roundDigits` specifies the maximal number of digits after 
the radix point of the calculated values. The function can be called using `*` syntax with 
the keyword argument set to it default value.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> gf1 = GaussFunc(3,1)
$(Doc_mul_Eg1)

julia> gf1 * 2
$(Doc_mul_Eg2)

julia> gf1 * gf1
$(Doc_mul_Eg3)

julia> gf1 * 2 * gf1
$(Doc_mul_Eg4)
```
"""
function mul(gf::GaussFunc{T}, coeff::Real; roundDigits::Int=getAtolDigits(T)) where {T}
    c = convert(T, coeff)
    con, mapFunction, dataName = mulCore(c, gf.con; roundDigits)
    conNew = genContraction(con, mapFunction; dataName, canDiff=gf.con.canDiff[])
    GaussFunc(gf.xpn, conNew)
end

function mulCore(c::T, con::ParamBox{T, <:Any, FI}; 
                 roundDigits::Int=getAtolDigits(T)) where {T}
    conNew = fill(roundNum(con.data[] * c, roundDigits))
    mapFunction = itself
    dataName = :undef
    conNew, mapFunction, dataName
end

function mulCore(c::T, con::ParamBox{T}; roundDigits::Int=getAtolDigits(T)) where {T}
    conNew = con.data
    mapFunction = Pf(roundNum(c, roundDigits), con.map)
    conNew, mapFunction, con.dataName
end

mulCore(c::T1, con::ParamBox{T2}; roundDigits::Int=getAtolDigits(T2)) where {T1, T2} = 
mulCore(T2(c), con; roundDigits)

mul(coeff::Real, gf::GaussFunc{T}; roundDigits::Int=getAtolDigits(T)) where {T} = 
mul(gf, coeff; roundDigits)

mul(gf1::GaussFunc{T}, gf2::GaussFunc{T}; roundDigits::Int=getAtolDigits(T)) where {T} = 
GaussFunc(    genExponent(roundNum(gf1.xpn()+gf2.xpn(), roundDigits)), 
           genContraction(roundNum(gf1.con()*gf2.con(), roundDigits)) )

"""

    mul(sgf1::BasisFunc{T, D, 𝑙1, 1}, sgf2::BasisFunc{T, D, 𝑙2, 1}; 
        normalizeGTO::Union{Bool, Missing}=missing, 
        roundDigits::Int=getAtolDigits(T)) where {T, D, 𝑙1, 𝑙2} -> 
    BasisFunc{T, D, 𝑙1+𝑙2, 1}

    mul(a1::Real, a2::CompositeGTBasisFuncs{T, D, <:Any, 1}; 
        normalizeGTO::Union{Bool, Missing}=missing, 
        roundDigits::Int=getAtolDigits(T)) where {T, D} -> 
    CompositeGTBasisFuncs{T, D, <:Any, 1}

    mul(a1::CompositeGTBasisFuncs{T, D, <:Any, 1}, 
        a2::CompositeGTBasisFuncs{T, D, <:Any, 1}; 
        normalizeGTO::Union{Bool, Missing}=missing, 
        roundDigits::Int=getAtolDigits(T)) where {T, D} -> 
    CompositeGTBasisFuncs{T, D, <:Any, 1}

Multiplication between `CompositeGTBasisFuncs{T, D, <:Any, 1}`s such as `BasisFunc` and 
`$(BasisFuncMix)`, or contraction coefficient multiplication between a `Real` number 
and a `CompositeGTBasisFuncs{T, D, <:Any, 1}`. If `normalizeGTO` is set to `missing` (in 
default), The `GaussFunc` in the output result will be normalized only if all the input 
bases have `normalizeGTO = true`. `roundDigits` specifies the maximal number of digits 
after the radix point of the calculated values. The function can be called using `*` syntax 
with the keyword arguments set to their default values.
≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> bf1 = genBasisFunc([1,1,1], ([2,1], [0.1, 0.2]))
BasisFunc{0, 2}(center, gauss)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bf2 = bf1 * 2
BasisFunc{0, 2}(center, gauss)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> getindex.(getproperty.(bf2.gauss, :con))
(0.2, 0.4)

julia> bf3 = bf1 * bf2
BasisFunc{0, 3}(center, gauss)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]
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
    n₁ && (d₁ *= getNorms(sgf1)[])
    n₂ && (d₂ *= getNorms(sgf2)[])
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
    n₁ && (d₁ *= getNorms(sgf1)[])
    n₂ && (d₂ *= getNorms(sgf2)[])
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
    c = (n && !normalizeGTO) ? (coeff .* getNorms(bf)) : coeff
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

    shift(bf::FloatingGTBasisFuncs{<:Any, D, 𝑙, GN, <:Any, 1}, 
          dl::Union{Vector{Int}, NTuple{D, Int}}, op::Function=+) where {D, 𝑙, GN} -> 
    BasisFunc

Shift (`+` as the "add" operator in default) the angular momentum (Cartesian 
representation) given the a vector that specifies the change of each pseudo-quantum number 
𝑑i, 𝑑j, 𝑑k.
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

    decompose(bf::CompositeGTBasisFuncs, splitGaussFunc::Bool=false) -> Matrix{<:BasisFunc}

Decompose a `FloatingGTBasisFuncs` into an `Array` of `BasisFunc`s. Each column represents 
one orbital of the input basis function(s). If `splitGaussFunc` is `true`, then each column 
consists of the `BasisFunc`s each with only 1 `GaussFunc`.
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

    basisSize(subshell::String, D::Integer=3) -> Int

Return the size (number of orbitals) of each subshell in `D` dimensional real space.
"""
@inline basisSize(subshell::String, D::Integer=3) = SubshellSizeList[D][subshell]

"""

    basisSize(b::CompositeGTBasisFuncs) -> Int

Return the numbers of orbitals of the input basis function(s).
"""
@inline basisSize(::CGTBasisFuncsON{ON}) where {ON} = ON
@inline basisSize(::BasisFuncMix) = 1


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
                     norm=1.0, printCenter=true, roundDigits::Int=-1) -> String

Generate a `String` of the text of the input `FloatingGTBasisFuncs`. `norm` is the 
additional normalization factor. If `printCenter` is `true`, the center coordinate 
will be added on the first line of the `String`.
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

    genBasisFuncText(bs::Union{AbstractVector{<:FloatingGTBasisFuncs}, 
                               Tuple{Vararg{FloatingGTBasisFuncs}}; 
                     norm=1.0, printCenter=true, groupCenters::Bool=true) -> 
    String

Generate a `String` of the text of the input basis set. `norm` is the additional 
normalization factor. If `printCenter` is `true`, the center coordinate will be added 
on the first line of the `String`. `groupCenters` determines whether the function will 
group the basis functions with same center together.
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


const Doc_genBFuncsFromText_strs = split(genBasisFuncText(genBasisFunc([1.0, 0.0, 0.0], 
                                                                       (2.0, 1.0))), "\n")

"""

    genBFuncsFromText(content::String; adjustContent::Bool=false, 
                      adjustFunction::F=sciNotReplace, 
                      excludeFirstNlines::Int=0, excludeLastNlines::Int=0, 
                      center::Union{AbstractArray, 
                                    NTuple{N, ParamBox}, 
                                    Missing}=missing, 
                      unlinkCenter::Bool=false) where {N, F<:Function} -> 
    Array{<:FloatingGTBasisFuncs, 1}

Generate the basis set from a `String` of basis set in Gaussian format or the String output 
from `genBasisFuncText`. For the former, `adjustContent` needs to be set to `true`. 
`adjustFunction` is only applied when `adjustContent=true`, which in default is a 
`function` used to detect and convert the format of the scientific notation in the String.

`excludeFirstNlines` and `excludeLastNlines` are used to exclude first or last few lines of 
the `String` if intent. `center` is used to assign a coordinate for all the basis functions 
from the String; it can be a `Vector`, a `Tuple` of the positional `ParamBox`s; when it's 
set to `missing`, it will try to read the center information from the input string, and 
leave the center as `[NaN, NaN, Nan]` if it can't find one for the corresponding 
`BasisFunc`. If `unlinkCenter = true`, the center of each basis function is a `deepcopy` of 
the input `center`. The coordinate information, if included, should be right above the 
subshell information for the `BasisFunc`. E.g.:
```
    \"\"\"
    $(Doc_genBFuncsFromText_strs[1])
    $(Doc_genBFuncsFromText_strs[2])
    $(Doc_genBFuncsFromText_strs[3])
    \"\"\"
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
        if oInfo[1] == "SP"
            gs2 = GaussFunc{typ}[]
            for j = i+1 : i+ng
                push!(gs1, GaussFunc(data[j][1], data[j][2]))
                push!(gs2, GaussFunc(data[j][1], data[j][3]))
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

    assignCenInVal!(center::AbstractArray, b::FloatingGTBasisFuncs) -> NTuple{3, ParamBox}

Assign a new value to the data stored in `b.center` (meaning the output value will also 
change according to the mapping function) given a `FloatingGTBasisFuncs`. Also, return 
the altered center.
"""
function assignCenInVal!(center::AbstractArray, b::FloatingGTBasisFuncs)
    for (i,j) in zip(b.center, center)
        i[] = j
    end
    b.center
end


"""

    getParams(pbc::ParamBox, symbol::Union{Symbol, Missing}=missing; 
              forDifferentiation::Bool=false) -> 
    Union{ParamBox, Nothing}

    getParams(pbc::$(ParameterizedContainer), symbol::Union{Symbol, Missing}=missing; 
              forDifferentiation::Bool=false) -> 
    AbstractVector{<:ParamBox}

    getParams(pbc::Union{AbstractArray, Tuple}, symbol::Union{Symbol, Missing}=missing; 
              forDifferentiation::Bool=false) -> 
    AbstractVector{<:ParamBox}

Return the parameter(s) stored in the input container. If the keyword argument `symbol` is 
set to `missing`, then return all parameter(s); if it's set to the `Symbol` of a parameter 
(e.g., `:α₁` will match any `pb` such that `getVar(forDifferentiation) == :α₁`; `:α` will 
match all the `pb`s that are `ParamBox{<:Any, α}`. `forDifferentiation` determines whether 
searching through the `Symbol`(s) of the independent variable(s) represented by `pbc` 
during the differentiation process. If the 1st argument is a collection, its entries must 
be `ParamBox` containers.
"""
getParams(pb::ParamBox, symbol::Union{Symbol, Missing}=missing; 
          forDifferentiation::Bool=false) = 
ifelse(paramFilter(pb, symbol, forDifferentiation), pb, nothing)

getParams(cs::AbstractArray{<:ParamBox}, symbol::Union{Symbol, Missing}=missing; 
          forDifferentiation::Bool=false) = 
cs[findall(x->paramFilter(x, symbol, forDifferentiation), cs)]

getParams(ssb::ParameterizedContainer, symbol::Union{Symbol, Missing}=missing; 
          forDifferentiation::Bool=false) = 
filter(x->paramFilter(x, symbol, forDifferentiation), ssb.param) |> collect

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


const Doc_copyBasis_Eg1 = "GaussFunc(xpn=ParamBox{Float64, :α, "*
                                    "$(FI)}(9.0)[∂][α], "*
                                    "con=ParamBox{Float64, :d, "*
                                    "$(FI)}(2.0)[∂][d])"

"""

    copyBasis(b::GaussFunc, copyOutVal::Bool=true) -> GaussFunc

    copyBasis(b::CompositeGTBasisFuncs, copyOutVal::Bool=true) -> CompositeGTBasisFuncs

Return a copy of the input basis. If `copyOutVal` is set to `true`, then only the value(s) 
of mapped data will be copied, i.e., `outValCopy` is used to copy the `ParamBox`s, 
otherwise `inVarCopy` is used.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> e = genExponent(3.0, x->x^2)
ParamBox{Float64, :α, $(FLevel(x->x^2))}(3.0)[∂][x_α]

julia> c = genContraction(2.0)
ParamBox{Float64, :d, $(FI)}(2.0)[∂][d]

julia> gf1 = GaussFunc(e, c);

julia> gf2 = copyBasis(gf1)
$(Doc_copyBasis_Eg1)

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

    markParams!(b::Union{Array{T}, T, Tuple{Vararg{T}}}, 
                filterMapping::Bool=false)  where {T<:$(ParameterizedContainer)} -> 
    Array{<:ParamBox, 1}

Mark the parameters (`ParamBox`) in input bs which can a `Vector` of `GaussFunc` or 
`FloatingGTBasisFuncs`. The identical parameters will be marked with same index. 
`filterMapping`determines whether filtering out (i.e. not return) extra `ParamBox`s that 
have same independent variables despite may having different mapping functions.
"""
markParams!(b::Union{Array{T}, T, Tuple{Vararg{T}}}, 
            filterMapping::Bool=false) where {T<:ParameterizedContainer} = 
markParams!(getParams(b), filterMapping)

function markParams!(parArray::Array{<:ParamBox}, filterMapping::Bool=false)
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

function markParamsCore!(parArray::Array{<:ParamBox{<:Any, V}}) where {V}
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

getNorms(b::FGTBasisFuncs1O{T, 3, 𝑙, GN})  where {T, 𝑙, GN} = 
getNijkα.(b.l[1]..., T[g.xpn() for g in b.gauss])