export GaussFunc, genExponent, genContraction, SpatialPoint, genSpatialPoint, BasisFunc, 
       BasisFuncs, genBasisFunc, subshellOf, centerOf, centerCoordOf, GTBasis, 
       sortBasisFuncs, add, mul, shift, decompose, basisSize, genBasisFuncText, 
       genBFuncsFromText, assignCenInVal!, getParams, copyBasis, markParams!, getVar, 
       getVarDict, expressionOf

export SP1D, SP2D, SP3D

using Symbolics
using SymbolicUtils
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

    GaussFunc(xpn::Union{Real, ParamBox}, con::Union{Real, ParamBox}) -> GaussFunc

"""
struct GaussFunc{T, FLxpn, FLcon} <: AbstractGaussFunc{T}
    xpn::ParamBox{T, xpnSym, FLxpn}
    con::ParamBox{T, conSym, FLcon}
    param::Tuple{ParamBox{T, xpnSym, FLxpn}, ParamBox{T, conSym, FLcon}}

    GaussFunc(xpn::ParamBox{T, xpnSym, FL1}, con::ParamBox{T, conSym, FL2}) where 
             {T, FL1, FL2} = 
    new{T, FL1, FL2}(xpn, con, (xpn, con))
end

GaussFunc(e::T1, d::T2) where {T1<:Union{Real, ParamBox}, T2<:Union{Real, ParamBox}} = 
GaussFunc(genExponent(e), genContraction(d))


"""

    genExponent(e::Real, mapFunction::Function; canDiff::Bool=true, 
                dataName::Symbol=:undef, roundDigits::Int=-1, 
                numberType::Type{<:Real}=Float64) -> 
    ParamBox{numberType, :$(xpnSym)}

    genExponent(e::Array{T, 0}, mapFunction::Function; canDiff::Bool=true, 
                dataName::Symbol=:undef) where {T<:Real} -> 
    ParamBox{T, :$(xpnSym)}

Construct a `ParamBox` for an exponent coefficient given a value. Keywords `mapFunction` 
and `canDiff` work the same way as in a general constructor of a `ParamBox`. If 
`roundDigits < 0`, there won't be rounding for input data.
"""
genExponent(e::Real, mapFunction::F; canDiff::Bool=true, dataName::Symbol=:undef, 
            roundDigits::Int=-1, numberType::Type{<:Real}=Float64) where 
           {F<:Function} = 
ParamBox(Val(xpnSym), mapFunction, fill(convertNumber(e, roundDigits, numberType)), 
         genIndex(nothing), fill(canDiff), dataName)

genExponent(e::Array{T, 0}, mapFunction::F; canDiff::Bool=true, 
            dataName::Symbol=:undef) where {T<:Real, F<:Function} = 
ParamBox(Val(xpnSym), mapFunction, e, genIndex(nothing), fill(canDiff), dataName)



"""

    genExponent(e::Real; roundDigits::Int=-1, numberType::Type{<:Real}=Float64) -> 
    ParamBox{numberType, :$(xpnSym)}

    genExponent(e::Array{T, 0}) where {T<:Real} -> ParamBox{T, :$(xpnSym)}

"""
genExponent(e::Real; roundDigits::Int=-1, numberType::Type{<:Real}=Float64) = 
ParamBox(Val(xpnSym), itself, fill(convertNumber(e, roundDigits, numberType)), 
         genIndex(nothing))

genExponent(e::Array{T, 0}) where {T<:Real} = 
ParamBox(Val(xpnSym), itself, e, genIndex(nothing))


"""

    genExponent(pb::ParamBox{T}) where {T<:Real} -> ParamBox{T, :$(xpnSym)}

Convert a `$(ParamBox)` to an exponent coefficient parameter.
"""
genExponent(pb::ParamBox{T, <:Any, F}) where {T<:Real, F} = ParamBox(Val(xpnSym), pb)


"""

    genContraction(c::Real, mapFunction::Function; canDiff::Bool=true, 
                   dataName::Symbol=:undef, roundDigits::Int=-1, 
                   numberType::Type{<:Real}=Float64) -> 
    ParamBox{numberType, :$(conSym)}

    genContraction(c::Array{T, 0}, mapFunction::Function; canDiff::Bool=true, 
                   dataName::Symbol=:undef) where {T<:Real} -> 
    ParamBox{T, :$(conSym)}

Construct a `ParamBox` for an contraction coefficient given a value. Keywords `mapFunction` 
and `canDiff` work the same way as in a general constructor of a `ParamBox`. If 
`roundDigits < 0`, there won't be rounding for input data.
"""
genContraction(c::Real, mapFunction::F; canDiff::Bool=true, dataName::Symbol=:undef, 
               roundDigits::Int=-1, numberType::Type{<:Real}=Float64) where 
              {F<:Function} = 
ParamBox(Val(conSym), mapFunction, fill(convertNumber(c, roundDigits, numberType)), 
         genIndex(nothing), fill(canDiff), dataName)

genContraction(c::Array{T, 0}, mapFunction::F; canDiff::Bool=true, 
               dataName::Symbol=:undef) where {T<:Real, F<:Function} = 
ParamBox(Val(conSym), mapFunction, c, genIndex(nothing), fill(canDiff), dataName)

"""

    genContraction(c::Real; roundDigits::Int=-1, numberType::Type{<:Real}=Float64) -> 
    ParamBox{numberType, :$(conSym)}

    genContraction(c::Array{T, 0}) where {T<:Real} -> ParamBox{T, :$(conSym)}

"""
genContraction(c::Real; roundDigits::Int=-1, numberType::Type{<:Real}=Float64) = 
ParamBox(Val(conSym), itself, fill(convertNumber(c, roundDigits, numberType)), 
         genIndex(nothing))

genContraction(c::Array{T, 0}) where {T<:Real} = 
ParamBox(Val(conSym), itself, c, genIndex(nothing))

"""

    genContraction(pb::ParamBox{T}) where {T<:Real} -> ParamBox{T, :$(conSym)}

Convert a `$(ParamBox)` to an exponent coefficient parameter.
"""
genContraction(pb::ParamBox{T, <:Any, F}) where {T<:Real, F} = ParamBox(Val(conSym), pb)


const Doc_genSpatialPoint_Eg1 = "SpatialPoint{3, Float64, "*
                                "Tuple{FLevel{1, 0}, FLevel{1, 0}, FLevel{1, 0}}}"*
                                "(param)[1.0, 2.0, 3.0][∂][∂][∂]"

const PT1D{FLx, T} = Tuple{ParamBox{T, cxSym, FLx}}
const PT2D{FLx, FLy, T} = Tuple{ParamBox{T, cxSym, FLx}, ParamBox{T, cySym, FLy}}
const PT3D{FLx, FLy, FLz, T} = Tuple{ParamBox{T, cxSym, FLx}, ParamBox{T, cySym, FLy}, 
                                     ParamBox{T, czSym, FLz}}

const SPointPTL = Union{Tuple{ParamBox{T, cxSym}} where T, 
                        Tuple{ParamBox{T, cxSym}, ParamBox{T, cySym}} where T, 
                        Tuple{ParamBox{T, cxSym}, ParamBox{T, cySym}, 
                              ParamBox{T, czSym}} where T}

const SPointPTU{D, T} = Tuple{Vararg{ParamBox{T, V, FL} where {V, FL<:FLevel}, D}}

struct SPoint{TT<:SPointPTL}
    param::TT
end

const SP1D{FLx, T} = SPoint{PT1D{FLx, T}}
const SP2D{FLx, FLy, T} = SPoint{PT2D{FLx, FLy, T}}
const SP3D{FLx, FLy, FLz, T} = SPoint{PT3D{FLx, FLy, FLz, T}}

struct SpatialPoint{D, T, PT} <: AbstractSpatialPoint{D, T}
    point::PT
    function SpatialPoint(pbs::SPointPTU{D, T}) where {D, T}
        sp = SPoint{typeof(pbs)}(pbs)
        new{D, T, typeof(sp)}(sp)
    end
end

"""

    genSpatialPoint(point::Union{Tuple{Vararg{Real}}, AbstractArray{<:Real}}, 
                    mapFunction::F=itself; canDiff::Bool=true, dataName::Symbol=:undef, 
                    roundDigits::Int=-1) -> 
    SpatialPoint

    genSpatialPoint(point::Union{Tuple{Vararg{Array{Float64, 0}}}, 
                                 AbstractArray{Array{Float64, 0}}}, 
                    mapFunction::F=itself; canDiff::Bool=true, dataName::Symbol=:undef) -> 
    SpatialPoint

Return the parameter(s) of a spatial coordinate in terms of `ParamBox`. Keywords 
`mapFunction` and `canDiff` work the same way as in a general constructor of a `ParamBox`. 
If `roundDigits < 0` or `point` is a 0-d `Array`, there won't be rounding for input data.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> v1 = [1,2,3]
3-element Vector{Int64}:
 1
 2
 3

julia> genSpatialPoint(v1)
$(Doc_genSpatialPoint_Eg1)

julia> v2 = [fill(1.0), 2, 3]
3-element Vector{Any}:
  fill(1.0)
 2
 3

julia> p2 = genSpatialPoint(v2); p2[1]
ParamBox{Float64, :X, $(FLi)}(1.0)[∂][X]

julia> v2[1][] = 1.2
1.2

julia> p2[1]
ParamBox{Float64, :X, $(FLi)}(1.2)[∂][X]
```
"""
genSpatialPoint(v::AbstractArray, optArgs...) = genSpatialPoint(Tuple(v), optArgs...)
genSpatialPoint(v::NTuple{N, Any}, optArgs...) where {N} = 
genSpatialPoint.(v, Tuple([1:N;]), optArgs...) |> genSpatialPointCore

"""

    genSpatialPoint(comp::Real, index::Int, mapFunction::F; canDiff::Bool=true, 
                    dataName::Symbol=:undef, roundDigits::Int=-1, 
                    numberType::Type{<:Real}=Float64) -> 
    ParamBox{numberType}

    genSpatialPoint(comp::Array{T, 0}, index::Int,mapFunction::F; canDiff::Bool=true, 
                    dataName::Symbol=:undef) -> 
    ParamBox{T}

    genSpatialPoint(comp::Real, index::Int; roundDigits::Int=-1, 
                    numberType::Type{<:Real}=Float64) -> 
    ParamBox{numberType}

    genSpatialPoint(comp::Array{T, 0}, index::Int) -> ParamBox{T}

    genSpatialPoint(comp::ParamBox{T}, index::Int) -> ParamBox{T}

Return the component of a `SpatialPoint` given its value (or 0-D container) and index.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genSpatialPoint(1.2, 1)
ParamBox{Float64, :X, $(FLi)}(1.2)[∂][X]

julia> pointY1 = fill(2.0)
0-dimensional Array{Float64, 0}:
2.0

julia> Y1 = genSpatialPoint(pointY1, 2)
ParamBox{Float64, :Y, $(FLi)}(2.0)[∂][Y]

julia> pointY1[] = 1.5
1.5

julia> Y1
ParamBox{Float64, :Y, $(FLi)}(1.5)[∂][Y]
```
"""
genSpatialPoint(comp::Real, index::Int, mapFunction::F; canDiff::Bool=true, 
                dataName::Symbol=:undef, roundDigits::Int=-1, 
                numberType::Type{<:Real}=Float64) where {F<:Function} = 
ParamBox(Val(SpatialParamSyms[index]), mapFunction, 
         fill(convertNumber(comp, roundDigits, numberType)), 
         genIndex(nothing), fill(canDiff), dataName)

genSpatialPoint(comp::Array{T, 0}, index::Int, mapFunction::F; 
                canDiff::Bool=true, dataName::Symbol=:undef) where {T<:Real, F<:Function} = 
ParamBox(Val(SpatialParamSyms[index]), mapFunction, comp, 
         genIndex(nothing), fill(canDiff), dataName)

genSpatialPoint(comp::Real, index::Int; roundDigits::Int=-1, 
                numberType::Type{<:Real}=Float64) = 
ParamBox(Val(SpatialParamSyms[index]), itself, 
         fill(convertNumber(comp, roundDigits, numberType)), genIndex(nothing))

genSpatialPoint(comp::Array{T, 0}, index::Int) where {T<:Real} = 
ParamBox(Val(SpatialParamSyms[index]), itself, comp, genIndex(nothing))

genSpatialPoint(point::ParamBox, index::Int) = ParamBox(Val(SpatialParamSyms[index]), point)

"""

    genSpatialPoint(point::Union{Tuple{Vararg{ParamBox}}, AbstractArray{<:ParamBox}}) -> 
    SpatialPoint

Convert a collection of `$(ParamBox)`s to a spatial point.
"""
genSpatialPoint(point::NTuple{N, ParamBox}) where {N} = 
ParamBox.(Val.(SpatialParamSyms[1:N]), point) |> genSpatialPointCore

genSpatialPointCore(point::PT1D{FLx, T}) where {T, FLx} = 
SpatialPoint(point)

genSpatialPointCore(point::PT2D{FLx, FLy, T}) where {T, FLx, FLy} = 
SpatialPoint(point)

genSpatialPointCore(point::PT3D{FLx, FLy, FLz, T}) where {T, FLx, FLy, FLz} = 
SpatialPoint(point)


"""

    BasisFunc{𝑙, GN, PT, D, T} <: FloatingGTBasisFuncs{𝑙, GN, 1, PT, D, T}

A (floating) basis function with the center attached to it instead of any nucleus.

≡≡≡ Field(s) ≡≡≡

`center::SpatialPoint{D, T, PT}`: The `D`-dimensional center coordinate.

`gauss::NTuple{N, GaussFunc}`: Gaussian functions within the basis function.

`ijk::Tuple{$(XYZTuple){𝑙}}`: Cartesian representation (pseudo-quantum number) of the 
angular momentum orientation. E.g., s (X⁰Y⁰Z⁰) would be `$(XYZTuple(0, 0, 0))`. For 
convenient syntax, `.ijk[]` converts it to a `NTuple{3, Int}`.

`normalizeGTO::Bool`: Whether the GTO`::GaussFunc` will be normalized in calculations.

`param::NTuple{D+GN*2, ParamBox}`： All the tunable parameters`::ParamBox` stored in the 
`BasisFunc`.

≡≡≡ Initialization Method(s) ≡≡≡

    BasisFunc(cen::SpatialPoint{D, T, PT}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
              ijk::Union{Tuple{XYZTuple{𝑙}}, XYZTuple{𝑙}}, normalizeGTO::Bool) where 
             {D, T, PT, 𝑙, GN} -> 
    BasisFunc{𝑙, GN, PT, D, T}

    BasisFunc(cen::SpatialPoint{D, T, PT}, gs::AbstractGaussFunc{T}, 
              ijk::Union{Tuple{XYZTuple{𝑙}}, XYZTuple{𝑙}}, normalizeGTO::Bool) where 
             {D, T, PT, 𝑙, GN} -> 
    BasisFunc{𝑙, 1, PT, D, T}
"""
struct BasisFunc{𝑙, GN, PT, D, T} <: FloatingGTBasisFuncs{𝑙, GN, 1, PT, D, T}
    center::SpatialPoint{D, T, PT}
    gauss::NTuple{GN, AbstractGaussFunc{T}}
    ijk::Tuple{XYZTuple{𝑙}}
    normalizeGTO::Bool
    param::Tuple{Vararg{ParamBox}}

    function BasisFunc(cen::SpatialPoint{D, T, PT}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
                       ijk::Tuple{XYZTuple{𝑙}}, normalizeGTO::Bool) where {D, T, PT, 𝑙, GN}
        pars = joinTuple(cen.point.param, getfield.(gs, :param)...)
        new{𝑙, GN, PT, D, T}(cen, gs, ijk, normalizeGTO, pars)
    end
end

BasisFunc(cen, gs::NTuple{GN, GaussFunc}, ijk::XYZTuple{𝑙}, 
          normalizeGTO=false) where {GN, 𝑙} = 
BasisFunc(cen, gs, (ijk,), normalizeGTO)

BasisFunc(cen, g::GaussFunc, ijk, normalizeGTO=false) = 
BasisFunc(cen, (g,), ijk, normalizeGTO)

BasisFunc(bf::BasisFunc) = itself(bf)

const BFunc{D, T} = BasisFunc{<:Any, <:Any, <:Any, D, T}


"""

    BasisFuncs{𝑙, GN, ON, PT, D, T} <: FloatingGTBasisFuncs{𝑙, GN, ON, PT, D, T}

A group of basis functions with identical parameters except they have different 
orientations in the specified subshell. It has the same fields as `BasisFunc` and 
specifically, for `ijk`, the size of the it (`ON`) can be no less than 1 (and no larger 
than the size of the corresponding subshell).
"""
struct BasisFuncs{𝑙, GN, ON, PT, D, T} <: FloatingGTBasisFuncs{𝑙, GN, ON, PT, D, T}
    center::SpatialPoint{D, T, PT}
    gauss::NTuple{GN, AbstractGaussFunc{Float64}}
    ijk::NTuple{ON, XYZTuple{𝑙}}
    normalizeGTO::Bool
    param::Tuple{Vararg{ParamBox}}

    function BasisFuncs(cen::SpatialPoint{D, T, PT}, gs::NTuple{GN, AbstractGaussFunc{T}}, 
                        ijks::NTuple{ON, XYZTuple{𝑙}}, normalizeGTO::Bool=false) where 
                       {D, T, PT, 𝑙, GN, ON}
        ss = SubshellXYZsizes[𝑙+1]
        @assert ON <= ss "The total number of `ijk` should be no more than $(ss) as " * 
                         "they are in $(subshell) subshell."
        ijks = sort(collect(ijks), rev=true) |> Tuple
        pars = joinTuple(cen.point.param, getfield.(gs, :param)...)
        new{𝑙, GN, ON, PT, D, T}(cen, gs, ijks, normalizeGTO, pars)
    end
end

BasisFuncs(cen, g::GaussFunc, ijks, normalizeGTO=false) = 
BasisFuncs(cen, (g,), ijks, normalizeGTO)

BasisFuncs(bfs::BasisFuncs) = itself(bfs)

BasisFunc(bfs::BasisFuncs{<:Any, <:Any, 1}) = 
BasisFunc(bfs.center, bfs.gauss, bfs.ijk, bfs.normalizeGTO)


struct EmptyBasisFunc{D, T} <: CompositeGTBasisFuncs{0, 1, D, T} end


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
                 ijkOrijks::Union{T, Array{T, 1}, NTuple{<:Any, T}}; 
                 normalizeGTO::Bool=false) where {T <: NTuple{3, Int}}

`ijkOrijks` is the Array of the pseudo-quantum number(s) to specify the angular 
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
                 ijkFilter::NTuple{N, Bool}=fill(true, SubshellSizeList[subshell])|>Tuple; 
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
genBasisFunc(cen::SpatialPoint, gs::NTuple{GN, AbstractGaussFunc{Float64}}, 
             ijk::XYZTuple{𝑙}=XYZTuple(0,0,0); normalizeGTO::Bool=false) where {GN, 𝑙} = 
BasisFunc(cen, gs, ijk, normalizeGTO)

genBasisFunc(cen::SpatialPoint, gs::NTuple{GN, AbstractGaussFunc{Float64}}, 
             ijk::NTuple{3, Int}; normalizeGTO::Bool=false) where {GN} = 
BasisFunc(cen, gs, ijk|>XYZTuple, normalizeGTO)

genBasisFunc(cen::SpatialPoint, gs::NTuple{GN, AbstractGaussFunc{Float64}}, 
             ijks::NTuple{ON, XYZTuple{𝑙}}; normalizeGTO::Bool=false) where {GN, ON, 𝑙} = 
BasisFuncs(cen, gs, ijks, normalizeGTO)

genBasisFunc(cen::SpatialPoint, gs::NTuple{GN, AbstractGaussFunc{Float64}}, 
             ijks::NTuple{ON, NTuple{3, Int}}; normalizeGTO::Bool=false) where {GN, ON} = 
BasisFuncs(cen, gs, ijks.|>XYZTuple, normalizeGTO)

genBasisFunc(cen::SpatialPoint, gs::NTuple{GN, AbstractGaussFunc{Float64}}, 
             ijks::Vector{XYZTuple{𝑙}}; normalizeGTO::Bool=false) where {GN, 𝑙} = 
genBasisFunc(cen, gs, ijks|>Tuple; normalizeGTO)

genBasisFunc(cen::SpatialPoint, gs::NTuple{GN, AbstractGaussFunc{Float64}}, 
             ijks::Vector{NTuple{3, Int}}; normalizeGTO::Bool=false) where {GN} = 
genBasisFunc(cen, gs, ijks|>Tuple; normalizeGTO)

genBasisFunc(cen::SpatialPoint, gs::NTuple{GN, AbstractGaussFunc{Float64}}, 
             ijk::Tuple{XYZTuple{𝑙}}; normalizeGTO::Bool=false) where {GN, 𝑙} = 
genBasisFunc(cen, gs, ijk[1]; normalizeGTO)

genBasisFunc(cen::SpatialPoint, gs::NTuple{GN, AbstractGaussFunc{Float64}}, 
             ijk::Tuple{NTuple{3, Int}}; normalizeGTO::Bool=false) where {GN} = 
genBasisFunc(cen, gs, ijk[1]; normalizeGTO)

function genBasisFunc(cen::SpatialPoint, gs::NTuple{GN, AbstractGaussFunc{Float64}}, 
                      subshell::String; normalizeGTO::Bool=false) where {GN}
    genBasisFunc(cen, gs, SubshellOrientationList[subshell]; normalizeGTO)
end

function genBasisFunc(cen::SpatialPoint, gs::NTuple{GN, AbstractGaussFunc{Float64}}, 
                      subshell::String, ijkFilter::NTuple{N, Bool}; 
                      normalizeGTO::Bool=false) where {GN, N}
    subshellSize = SubshellSizeList[subshell]
    @assert N == subshellSize "The length of `ijkFilter` should be $(subshellSize) "*
                              "to match the subshell's size."
    genBasisFunc(cen, gs, 
                 SubshellOrientationList[subshell][1:end.∈[findall(x->x==true,ijkFilter)]]; 
                 normalizeGTO)
end

function genBasisFunc(cen::SpatialPoint, xpnsANDcons::NTuple{2, Vector{<:Real}}, 
                      ijkOrSubshell=XYZTuple(0,0,0); normalizeGTO::Bool=false)
    @compareLength xpnsANDcons[1] xpnsANDcons[2] "exponents" "contraction coefficients"
    genBasisFunc(cen, GaussFunc.(xpnsANDcons[1], xpnsANDcons[2]), ijkOrSubshell; 
                 normalizeGTO)
end

genBasisFunc(cen::SpatialPoint, xpnANDcon::NTuple{2, Real}, 
             ijkOrSubshell=XYZTuple(0,0,0); normalizeGTO::Bool=false) = 
genBasisFunc(cen, (GaussFunc(xpnANDcon[1], xpnANDcon[2]),), ijkOrSubshell; normalizeGTO)

function genBasisFunc(center::SpatialPoint, BSKeyANDnuc::Vector{NTuple{2, String}}; 
                      unlinkCenter::Bool=false)
    bases = FloatingGTBasisFuncs[]
    for k in BSKeyANDnuc
        content = BasisSetList[k[1]][AtomicNumberList[k[2]]]
        @assert content!==nothing "Quiqbox DOES NOT have basis set "*k[1]*" for "*k[2]*"."
        append!(bases, genBFuncsFromText(content; adjustContent=true, 
                excludeLastNlines=1, center, unlinkCenter))
    end
    bases
end

genBasisFunc(cen::SpatialPoint, BSKeyANDnuc::NTuple{2, String}; 
             unlinkCenter::Bool=false) = 
genBasisFunc(cen, [BSKeyANDnuc]; unlinkCenter)

genBasisFunc(cen::SpatialPoint, BSkey::Vector{String}; nucleus::String="H", 
             unlinkCenter::Bool=false) = 
genBasisFunc(cen, [(i, nucleus) for i in BSkey]; unlinkCenter)

genBasisFunc(cen::SpatialPoint, BSkey::String; nucleus::String="H", 
             unlinkCenter::Bool=false) = 
genBasisFunc(cen, [BSkey]; nucleus, unlinkCenter)

# A few methods for convenient arguments omissions and mutations.
genBasisFunc(cen::SpatialPoint, gs::AbstractArray{<:AbstractGaussFunc{Float64}}, 
             args...; kws...) = 
genBasisFunc(cen, gs|>Tuple, args...; kws...)

genBasisFunc(cen::SpatialPoint, g::GaussFunc, args...; kws...) = 
genBasisFunc(cen, (g,), args...; kws...)

genBasisFunc(coord::Union{Tuple, AbstractArray}, args...; kws...) = 
genBasisFunc(genSpatialPoint(coord), args...; kws...)

genBasisFunc(::Missing, args...; kws...) = genBasisFunc((NaN, NaN, NaN), args...; kws...)

genBasisFunc(bf::FloatingGTBasisFuncs) = itself(bf)

genBasisFunc(bs::Vector{<:FloatingGTBasisFuncs}) = sortBasisFuncs(bs)


"""

    subshellOf(::FloatingGTBasisFuncs) -> String

Return the subshell name of the input `$(FloatingGTBasisFuncs)`.
"""
@inline subshellOf(::FloatingGTBasisFuncs{𝑙}) where {𝑙} = SubshellNames[𝑙+1]


"""

    sortBasisFuncs(bs::AbstractArray{<:FloatingGTBasisFuncs}, groupCenters::Bool=false) -> 
    Vector

    sortBasisFuncs(bs::Tuple{Vararg{FloatingGTBasisFuncs}}, groupCenters::Bool=false) -> 
    Tuple

Sort basis functions. If `groupCenters = true`, Then the function will return an 
`Array{<:Array{<:FloatingGTBasisFuncs, 1}, 1}` in which the arrays are grouped basis 
functions with same center coordinates.
"""
@inline function sortBasisFuncs(bs::AbstractVector{<:FloatingGTBasisFuncs}, 
                        groupCenters::Bool=false)
    bfBlocks = map( groupedSort(bs, centerCoordOf) ) do subbs
        # Reversed order within same subshell.
        sort!(subbs, by=x->[-getTypeParams(x)[1], x.ijk[1].tuple, getTypeParams(x)[2]], rev=true)
    end
    groupCenters ? bfBlocks : vcat(bfBlocks...)
end

sortBasisFuncs(bs::AbstractArray{<:FloatingGTBasisFuncs}, groupCenters::Bool=false) = 
sortBasisFuncs(reshape(bs, :), groupCenters)

sortBasisFuncs(bs::Tuple{Vararg{FloatingGTBasisFuncs, N}}, 
               groupCenters::Bool=false) where {N} = 
sortBasisFuncs(FloatingGTBasisFuncs[bs...], groupCenters) |> Tuple


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
centerCoordOf(bf::FloatingGTBasisFuncs) = [outValOf(i) for i in bf.center.point.param]


"""

    BasisFuncMix{BN, D, T, BT<:BFunc{D, T}} <: CompositeGTBasisFuncs{BN, 1, D, T}

Sum of multiple `FloatingGTBasisFuncs{<:Any, <:Any, 1}` without any reformulation, treated 
as one basis function in the integral calculation.

≡≡≡ Field(s) ≡≡≡

`BasisFunc::NTuple{BN, BT}`: Inside basis functions.

`param::Tuple{Vararg{ParamBox}}`: Contained parameters.

≡≡≡ Initialization Method(s) ≡≡≡

    BasisFuncMix(bfs::Union{Tuple{Vararg{T}}, Vector{T}}) where 
                {T<:FloatingGTBasisFuncs{<:Any, <:Any, 1}} ->
    BasisFuncMix

"""
struct BasisFuncMix{BN, D, T, BT<:BFunc{D, T}} <: CompositeGTBasisFuncs{BN, 1, D, T}
    BasisFunc::NTuple{BN, BT}
    param::Tuple{Vararg{ParamBox}}

    function BasisFuncMix(bfs::Tuple{Vararg{BFunc{D, T}, BN}}) where {D, T, BN}
        bs = sortBasisFuncs(bfs)
        new{BN, D, T, eltype(bfs)}(bs, joinTuple(getfield.(bs, :param)...))
    end
end

BasisFuncMix(bfs::AbstractArray{<:BasisFunc}) = BasisFuncMix(bfs|>Tuple)
BasisFuncMix(bfs::AbstractArray{T}) where {T<:FloatingGTBasisFuncs{<:Any, <:Any, 1}} = 
BasisFuncMix(BasisFunc.(bfs))
BasisFuncMix(bf::BasisFunc) = BasisFuncMix((bf,))
BasisFuncMix(bfs::BasisFuncs) = BasisFuncMix.(decompose(bfs))
BasisFuncMix(bfm::BasisFuncMix) = itself(bfm)


getTypeParams(::FloatingGTBasisFuncs{𝑙, GN, ON, PT, D, T}) where {𝑙, GN, ON, PT, D, T} = 
(𝑙, GN, ON, PT, D, T)
getTypeParams(::BasisFuncMix{BN, D, T, BT}) where {BN, D, T, BT} = (BN, D, T, BT)


unpackBasis(::EmptyBasisFunc) = ()
unpackBasis(b::BasisFunc)  = (b,)
unpackBasis(b::BasisFuncMix)  = b.BasisFunc
unpackBasis(b::BasisFuncs{<:Any, <:Any, 1})  = (BasisFunc(b),)


"""

    GTBasis{BN, D, T, BT<:GTBasisFuncs{1, D, T}} <: BasisSetData{D, T, BT}

The container to store basis set information.

≡≡≡ Field(s) ≡≡≡

`basis::NTuple{BN, BT}`: Basis set.

`S::Matrix{T}`: Overlap matrix.

`Te::Matrix{T}`: Kinetic energy part of the electronic core Hamiltonian.

`eeI::Array{T, 4}`: Electron-electron interaction.

≡≡≡ Initialization Method(s) ≡≡≡

    GTBasis(basis::Union{Tuple{Vararg{GTBasisFuncs{<:Any, D, T}}}, 
                         AbstractVector{<:GTBasisFuncs{<:Any, D, T}}}) where {D, T} -> 
    GTBasis

Construct a `GTBasis` given a basis set.
"""
struct GTBasis{BN, D, T, BT<:GTBasisFuncs{1, D, T}} <: BasisSetData{D, T, BT}
    basis::NTuple{BN, BT}
    S::Matrix{T}
    Te::Matrix{T}
    eeI::Array{T, 4}

    GTBasis(bfs::Tuple{Vararg{GTBasisFuncs{1, D, T}, BN}}) where 
           {BN, D, T<:Real} = 
    new{BN, D, T, eltype(bfs)}(bfs, overlaps(bfs), eKinetics(bfs), eeInteractions(bfs))
end

GTBasis(bs::Tuple{Vararg{GTBasisFuncs{<:Any, D, T}}}) where {D, T} = GTBasis(bs |> flatten)

GTBasis(bs::AbstractVector{<:GTBasisFuncs{<:Any, D, T}}) where {D, T} = GTBasis(bs |> Tuple)


function sumOfCore(bfs::AbstractVector{<:BasisFunc{<:Any, <:Any, <:Any, D, T}}, 
                   roundDigits::Int=-1) where {D, T}
    arr1 = convert(Vector{BasisFunc{<:Any, <:Any, <:Any, D, T}}, sortBasisFuncs(bfs))
    arr2 = BasisFunc[]
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

sumOfCore(bs::Union{Tuple{Vararg{CompositeGTBasisFuncs{<:Any, 1, D, T}}}, 
          AbstractVector{<:CompositeGTBasisFuncs{<:Any, 1, D, T}}}, 
          roundDigits::Int=15) where {D, T} = 
sumOfCore(BasisFunc{<:Any, <:Any, <:Any, D, T}[joinTuple(unpackBasis.(bs)...)...], 
          roundDigits)

function sumOf(bs::Union{Tuple, AbstractVector}; roundDigits::Int=-1)
    length(bs) == 1 && (return bs[1])
    sumOfCore(bs, roundDigits)
end

mergeGaussFuncs(gf::GaussFunc) = itself(gf)

function mergeGaussFuncs(gf1::GaussFunc, gf2::GaussFunc; roundDigits::Int=-1)
    xpn1 = gf1.xpn
    xpn2 = gf2.xpn
    con1 = gf1.con
    con2 = gf2.con
    xpn1v = xpn1()
    rBool = roundDigits < 0
    rBool || (atol = 0.1^roundDigits)
    if rBool ? xpn1v===xpn2() : isapprox(xpn1v, xpn2(); atol)
        if xpn1 === xpn2 || hasIdentical(xpn1, xpn2)
            xpn = xpn1
        elseif rBool ? hasEqual(xpn1, xpn2) : hasApprox(xpn1, xpn2; atol)
            xpn = deepcopy(xpn1)
        else
            xpn = genExponent(xpn1v; roundDigits)
        end

        if con1 === con2 || hasIdentical(con1, con2)
            res = GaussFunc(xpn, con1) * 2.0
        elseif rBool ? hasEqual(con1, con2) : hasApprox(con1, con2; atol)
            res = GaussFunc(xpn, deepcopy(con1)) * 2.0
        else
            res = GaussFunc(xpn, genContraction(con1()+con2(); roundDigits))
        end

        return [res]
    else
        return [gf1, gf2]
    end
end

function mergeGaussFuncs(gf1::GaussFunc{T}, gf2::GaussFunc{T}, 
                         gf3::GaussFunc{T}...; roundDigits::Int=-1) where {T}
    arr1 = GaussFunc[gf1, gf2, gf3...]
    arr2 = GaussFunc[]
    while length(arr1) >= 1
        i = 1
        while i < length(arr1)
            temp = mergeGaussFuncs(arr1[i], arr1[i+1]; roundDigits)
            if length(temp) == 1
                arr1[i] = temp[]
                popat!(arr1, i+1)
            else
                reverse!(arr1, i, i+1)
                i += 1
            end
        end
        push!(arr2, popat!(arr1, i))
    end
    arr2
end


"""

    add(b::CompositeGTBasisFuncs{<:Any, 1, D, T}) -> CompositeGTBasisFuncs{<:Any, 1, D, T}

    add(b1::CompositeGTBasisFuncs{<:Any, 1, D, T}, 
        b2::CompositeGTBasisFuncs{<:Any, 1, D, T}) ->
    CompositeGTBasisFuncs{<:Any, 1, D, T}

Addition between `CompositeGTBasisFuncs{<:Any, 1, D, T}` such as `BasisFunc` and 
`Quiqbox.BasisFuncMix`. It can be called using `+` syntax.

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
function add(b::BasisFuncs{𝑙, GN, 1}) where {𝑙, GN}
    BasisFunc(b.center, b.gauss, b.ijk, b.normalizeGTO)
end

add(b::BasisFunc) = itself(b)

function add(bf1::BasisFunc{𝑙1, GN1, PT1, D, T}, bf2::BasisFunc{𝑙2, GN2, PT2, D, T}; 
             roundDigits::Int=15) where {𝑙1, 𝑙2, GN1, GN2, PT1, PT2, D, T}
    if 𝑙1 == 𝑙2 && bf1.ijk == bf2.ijk && 
       bf1.normalizeGTO == bf2.normalizeGTO && 
       (c = centerCoordOf(bf1)) == centerCoordOf(bf2)

        cen1 = bf1.center
        cen2 = bf2.center
        if cen1 === cen2 || hasIdentical(cen1, cen2)
            cen = bf1.center
        elseif hasEqual(bf1, bf2)
            cen = deepcopy(bf1.center)
        else
            cen = genSpatialPoint(c)
        end
        gfsN = mergeGaussFuncs(bf1.gauss..., bf2.gauss...; roundDigits) |> Tuple
        BasisFunc(cen, gfsN, bf1.ijk, bf1.normalizeGTO)
    else
        BasisFuncMix([bf1, bf2])
    end
end

add(bfm::BasisFuncMix; roundDigits::Int=15) = sumOf(bfm.BasisFunc; roundDigits)

add(bf1::BasisFuncMix{1}, bf2::BasisFunc{𝑙}; roundDigits::Int=15) where {𝑙} = 
add(bf1.BasisFunc[1], bf2; roundDigits)

add(bf1::BasisFunc{𝑙}, bf2::BasisFuncMix{1}; roundDigits::Int=15) where {𝑙} = 
add(bf2, bf1; roundDigits)

add(bf::BasisFunc, bfm::BasisFuncMix{BN}; roundDigits::Int=15) where {BN} = 
sumOf((bf, bfm.BasisFunc...); roundDigits)

add(bfm::BasisFuncMix{BN}, bf::BasisFunc; roundDigits::Int=15) where {BN} = 
add(bf, bfm; roundDigits)

add(bf1::BasisFuncMix{1}, bf2::BasisFuncMix{1}; roundDigits::Int=15) = 
add(bf1.BasisFunc[1], bf2.BasisFunc[1]; roundDigits)

add(bf::BasisFuncMix{1}, bfm::BasisFuncMix{BN}; roundDigits::Int=15) where {BN} = 
add(bf.BasisFunc[1], bfm; roundDigits)

add(bfm::BasisFuncMix{BN}, bf::BasisFuncMix{1}; roundDigits::Int=15) where {BN} = 
add(bf, bfm; roundDigits)

add(bfm1::BasisFuncMix{BN1, D, T}, bfm2::BasisFuncMix{BN2, D, T}; 
    roundDigits::Int=15) where {BN1, BN2, D, T} = 
sumOf((bfm1.BasisFunc..., bfm2.BasisFunc...); roundDigits)

add(bf1::BasisFuncs{<:Any, <:Any, 1, <:Any, D, T}, 
    bf2::BasisFuncs{<:Any, <:Any, 1, <:Any, D, T}; roundDigits::Int=15) where {D, T} = 
[sumOf((add(bf1), add(bf2)); roundDigits)]

add(::EmptyBasisFunc{D}, b::CompositeGTBasisFuncs{<:Any, <:Any, D}) where {D} = itself(b)

add(b::CompositeGTBasisFuncs{<:Any, <:Any, D}, ::EmptyBasisFunc{D}) where {D} = itself(b)

add(::EmptyBasisFunc{D, T1}, ::EmptyBasisFunc{D, T2}) where {D, T1, T2} = 
EmptyBasisFunc{D, promote_type(T1, T2)}()


const Doc_mul_Eg1 = "GaussFunc(xpn=ParamBox{Float64, :α, $(FLi)}(3.0)[∂][α], " * 
                              "con=ParamBox{Float64, :d, $(FLi)}(1.0)[∂][d])"

const Doc_mul_Eg2 = "GaussFunc(xpn=ParamBox{Float64, :α, $(FLi)}(3.0)[∂][α], " * 
                              "con=ParamBox{Float64, :d, $(FLi)}(2.0)[∂][d])"

const Doc_mul_Eg3 = "GaussFunc(xpn=ParamBox{Float64, :α, $(FLi)}(6.0)[∂][α], " * 
                              "con=ParamBox{Float64, :d, $(FLi)}(1.0)[∂][d])"

const Doc_mul_Eg4 = "GaussFunc(xpn=ParamBox{Float64, :α, $(FLi)}(6.0)[∂][α], " * 
                              "con=ParamBox{Float64, :d, $(FLi)}(2.0)[∂][d])"

"""

    mul(gf::GaussFunc, coeff::Real) -> GaussFunc

    mul(coeff::Real, gf::GaussFunc) -> GaussFunc

    mul(gf1::GaussFunc, gf2::GaussFunc) -> GaussFunc

Multiplication between `GaussFunc`s or contraction coefficient multiplication between a 
`Real` number and a `GaussFunc`. It can be called using `*` syntax.

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
function mul(gf::GaussFunc, coeff::Real)
    c = convert(Float64, coeff)
    con, mapFunction, dataName = mulCore(c, gf.con)
    conNew = genContraction(con, mapFunction; dataName, canDiff=gf.con.canDiff[])
    GaussFunc(gf.xpn, conNew)
end

function mulCore(c::Float64, con::ParamBox{<:Any, <:Any, FLi})
    conNew = fill(con.data[] * c)
    mapFunction = itself
    dataName = :undef
    conNew, mapFunction, dataName
end

function mulCore(c::Float64, con::ParamBox{<:Any, <:Any, F}) where {F}
    conNew = con.data
    mapFunction = Pf(c, con.map)
    conNew, mapFunction, con.dataName
end

mul(coeff::Real, gf::GaussFunc) = mul(gf, coeff)

function mul(gf1::GaussFunc, gf2::GaussFunc)
    GaussFunc(genExponent(gf1.xpn()+gf2.xpn()), genContraction(gf1.con()*gf2.con()))
end

"""

    mul(sgf1::BasisFunc{𝑙1, 1}, sgf2::BasisFunc{𝑙2, 1}; 
        normalizeGTO::Union{Bool, Missing}=missing) ->
    BasisFunc{𝑙1+𝑙2, 1}

    mul(a1::Real, a2::CompositeGTBasisFuncs{<:Any, 1, D, T}; 
        normalizeGTO::Union{Bool, Missing}=missing) -> 
    CompositeGTBasisFuncs{<:Any, 1, D, T}

    mul(a1::CompositeGTBasisFuncs{<:Any, 1, D, T}, 
        a2::CompositeGTBasisFuncs{<:Any, 1, D, T}; 
        normalizeGTO::Union{Bool, Missing}=missing) -> 
    CompositeGTBasisFuncs{<:Any, 1, D, T}

Multiplication between `CompositeGTBasisFuncs{<:Any, 1, D, T}`s such as `BasisFunc` and 
`$(BasisFuncMix)`, or contraction coefficient multiplication between a `Real` number 
and a `CompositeGTBasisFuncs{<:Any, 1, D, T}`. If `normalizeGTO` is set to `missing` (in 
default), The `GaussFunc` in the output result will be normalized only if all the input 
bases have `normalizeGTO = true`. The function can be called using `*` syntax.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> bf1 = genBasisFunc([1,1,1], ([2,1], [0.1, 0.2]))
BasisFunc{0, 2}(center, gauss)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bf2 = bf1 * 2
BasisFunc{0, 2}(center, gauss)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> getindex.(getfield.(bf2.gauss, :con))
(0.2, 0.4)

julia> bf3 = bf1 * bf2
BasisFunc{0, 3}(center, gauss)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]
```
"""
function mul(sgf1::BasisFunc{𝑙1, 1, PT1, D, T}, sgf2::BasisFunc{𝑙2, 1, PT2, D, T}; 
             normalizeGTO::Union{Bool, Missing}=missing) where {𝑙1, 𝑙2, PT1, PT2, D, T}
    α₁ = sgf1.gauss[1].xpn()
    α₂ = sgf2.gauss[1].xpn()
    d₁ = sgf1.gauss[1].con()
    d₂ = sgf2.gauss[1].con()
    n₁ = sgf1.normalizeGTO
    n₂ = sgf2.normalizeGTO
    n₁ && (d₁ *= getNorms(sgf1)[])
    n₂ && (d₂ *= getNorms(sgf2)[])
    R₁ = centerCoordOf(sgf1)
    R₂ = centerCoordOf(sgf2)
    normalizeGTO isa Missing && (normalizeGTO = n₁*n₂)
    if R₁ == R₂
        xpn = α₁ + α₂
        con = d₁ * d₂
        BasisFunc(genSpatialPoint(R₁), GaussFunc(genExponent(xpn), genContraction(con)), 
                  sgf1.ijk.+sgf2.ijk, normalizeGTO)
    else
        ijk1 = sgf1.ijk[1]
        ijk2 = sgf2.ijk[1]
        xpn, con, cen = gaussProd((α₁, d₁, R₁), (α₂, d₂, R₂))
        coeffs = [Float64[] for _=1:3]
        shiftPolyFunc = @inline (n, c1, c2) -> [(c2 - c1)^k*binomial(n,k) for k = n:-1:0]
        for i = 1:3
            c1 = shiftPolyFunc(ijk1[i], R₁[i], cen[i])
            c2 = shiftPolyFunc(ijk2[i], R₂[i], cen[i])
            m = reverse(c1 * transpose(c2), dims=2)
            siz = size(m)
            s, e = siz[2]-1, 1-siz[1]
            step = (-1)^(s > e)
            coeffs[i] = [diag(m, k)|>sum for k = s:step:e]
        end
        XYZcs = cat(Ref(coeffs[1] * transpose(coeffs[2])) .* coeffs[3]..., dims=3)
        R = genSpatialPoint(cen)
        pbα = genExponent(xpn)
        BasisFuncMix([BasisFunc(R, GaussFunc(pbα, genContraction(con*XYZcs[i])), 
                                XYZTuple(i.I .- 1), normalizeGTO) 
                      for i in CartesianIndices(XYZcs)])
    end
end

function mul(sgf1::BasisFunc{0, 1, PT1, D, T}, sgf2::BasisFunc{0, 1, PT2, D, T}; 
             normalizeGTO::Union{Bool, Missing}=missing) where {PT1, PT2, D, T}
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
    BasisFunc(genSpatialPoint(cen), GaussFunc(genExponent(xpn), genContraction(con)), 
              (XYZTuple(0,0,0),), normalizeGTO)
end

function gaussProd((α₁, d₁, R₁)::T, (α₂, d₂, R₂)::T) where 
                  {T<:Tuple{Number, Number, Array{<:Number}}}
    α = α₁ + α₂
    d = d₁ * d₂ * exp(-α₁ * α₂ / α * sum(abs2, R₁-R₂))
    R = (α₁*R₁ + α₂*R₂) / α
    (α, d, R)
end

function mulCore(bf::BasisFunc{𝑙, GN}, coeff::Real; 
                 normalizeGTO::Union{Bool, Missing}=missing) where {𝑙, GN}
    n = bf.normalizeGTO
    normalizeGTO isa Missing && (normalizeGTO = n)
    c = (n && !normalizeGTO) ? (coeff .* getNorms(bf)) : coeff
    gfs = mul.(bf.gauss, c)
    BasisFunc(bf.center, gfs, bf.ijk, normalizeGTO)
end

function mulCore(bfm::T, coeff::Real; normalizeGTO::Union{Bool, Missing}=missing) where 
                {T<:BasisFuncMix}
    BasisFuncMix(mul.(bfm.BasisFunc, coeff; normalizeGTO))::T
end

function mul(bf1::BasisFunc{𝑙1, GN1, PT1, D, T}, bf2::BasisFunc{𝑙2, GN2, PT2, D, T}; 
             normalizeGTO::Union{Bool, Missing}=missing) where 
            {𝑙1, 𝑙2, GN1, GN2, PT1, PT2, D, T}
    cen1 = bf1.center
    ijk1 = bf1.ijk
    cen2 = bf2.center
    ijk2 = bf2.ijk
    bf1n = bf1.normalizeGTO
    bf2n = bf2.normalizeGTO
    normalizeGTO isa Missing && (normalizeGTO = bf1n * bf2n)
    bs = CompositeGTBasisFuncs{<:Any, 1, D, T}[]
    for gf1 in bf1.gauss, gf2 in bf2.gauss
        push!(bs, mul(BasisFunc(cen1, (gf1,), ijk1, bf1n), 
                      BasisFunc(cen2, (gf2,), ijk2, bf2n); 
                      normalizeGTO))
    end
    sumOf(bs)
end

mul(bf1::BasisFuncMix{1}, bf2::BasisFunc; normalizeGTO::Union{Bool, Missing}=missing) = 
mul(bf1.BasisFunc[1], bf2; normalizeGTO)

mul(bf1::BasisFunc, bf2::BasisFuncMix{1}; normalizeGTO::Union{Bool, Missing}=missing) = 
mul(bf2, bf1; normalizeGTO)

mul(bf::BasisFunc, bfm::BasisFuncMix{BN}; 
    normalizeGTO::Union{Bool, Missing}=missing) where {BN} = 
mul.(Ref(bf), bfm.BasisFunc; normalizeGTO) |> sumOf

mul(bfm::BasisFuncMix{BN}, bf::BasisFunc; 
    normalizeGTO::Union{Bool, Missing}=missing) where {BN} = 
mul(bf, bfm; normalizeGTO)

mul(bf1::BasisFuncMix{1}, bf2::BasisFuncMix{1}; 
    normalizeGTO::Union{Bool, Missing}=missing) = 
mul(bf1.BasisFunc[1], bf2.BasisFunc[1]; normalizeGTO)

mul(bf::BasisFuncMix{1}, bfm::BasisFuncMix{BN}; 
    normalizeGTO::Union{Bool, Missing}=missing) where {BN} = 
mul(bf.BasisFunc[1], bfm; normalizeGTO)

mul(bfm::BasisFuncMix{BN}, bf::BasisFuncMix{1}; 
    normalizeGTO::Union{Bool, Missing}=missing) where {BN} = 
mul(bf, bfm; normalizeGTO)

function mul(bfm1::BasisFuncMix{BN1, D, T}, bfm2::BasisFuncMix{BN2, D, T}; 
             normalizeGTO::Union{Bool, Missing}=missing) where {BN1, BN2, D, T}
    bfms = CompositeGTBasisFuncs{<:Any, 1, D, T}[]
    for bf1 in bfm1.BasisFunc, bf2 in bfm2.BasisFunc
        push!(bfms, mul(bf1, bf2; normalizeGTO))
    end
    sumOf(bfms)
end

mul(::EmptyBasisFunc{D, T}, ::Real; normalizeGTO=nothing) where {D, T} = 
EmptyBasisFunc{D, T}()

mul(::Real, ::EmptyBasisFunc{D, T}; normalizeGTO=nothing) where {D, T} = 
EmptyBasisFunc{D, T}()

function mul(b::CompositeGTBasisFuncs{BN, 1, D, T}, coeff::Real; 
             normalizeGTO::Union{Bool, Missing}=missing) where {BN, D, T}
    iszero(coeff) ? EmptyBasisFunc{D, T}() : mulCore(b, coeff; normalizeGTO)
end

mul(coeff::Real, b::CompositeGTBasisFuncs{<:Any, 1}; 
    normalizeGTO::Union{Bool, Missing}=missing) = 
mul(b, coeff; normalizeGTO)

mul(::EmptyBasisFunc{D}, ::CompositeGTBasisFuncs{<:Any, <:Any, D, T}; 
    normalizeGTO=nothing) where {D, T} = 
EmptyBasisFunc{D, T}()

mul(::CompositeGTBasisFuncs{<:Any, <:Any, D, T}, ::EmptyBasisFunc{D}; 
    normalizeGTO=nothing) where {D, T} = 
EmptyBasisFunc{D, T}()

mul(::EmptyBasisFunc{D, T1}, ::EmptyBasisFunc{D, T2}; 
    normalizeGTO=nothing) where {D, T1, T2} = 
EmptyBasisFunc{D, promote_type(T1, T2)}()

mul(bf1::BasisFuncs{𝑙1, GN1, 1, PT1, D, T}, bf2::BasisFuncs{𝑙2, GN2, 1, PT2, D, T}; 
    normalizeGTO::Union{Bool, Missing}=missing) where {𝑙1, 𝑙2, GN1, GN2, PT1, PT2, D, T} = 
[mul(add(bf1), add(bf2); normalizeGTO)]


"""

    shift(bf::FloatingGTBasisFuncs{𝑙, GN, 1}, 
          didjdk::Union{Vector{<:Real}, NTuple{3, Int}}, op::Function=+) where {𝑙, GN} -> 
    BasisFunc

Shift (`+` as the "add" operator in default) the angular momentum (Cartesian 
representation) given the a vector that specifies the change of each pseudo-quantum number 
𝑑i, 𝑑j, 𝑑k.
"""
shift(bf::FloatingGTBasisFuncs{𝑙, GN, 1}, didjdk::AbstractArray{<:Real}, op::F=+) where 
     {𝑙, GN, F<:Function} = 
shiftCore(op, bf, XYZTuple(didjdk.|>Int))

shift(bf::FloatingGTBasisFuncs{𝑙, GN, 1}, didjdk::NTuple{3, Int}, op::F=+) where 
     {𝑙, GN, F<:Function} = 
shiftCore(op, bf, XYZTuple(didjdk))

shiftCore(::typeof(+), bf::FloatingGTBasisFuncs{𝑙1, GN, 1}, didjdk::XYZTuple{𝑙2}) where 
         {𝑙1, 𝑙2, GN} = 
BasisFunc(bf.center, bf.gauss, bf.ijk[1]+didjdk, bf.normalizeGTO)

shiftCore(::typeof(-), bf::FloatingGTBasisFuncs{0, GN, 1}, ::XYZTuple{0}) where {GN} = 
BasisFunc(bf.center, bf.gauss, bf.ijk[1], bf.normalizeGTO)

shiftCore(::typeof(-), bf::FloatingGTBasisFuncs{0, <:Any, 1, <:Any, D, T}, 
          didjdk::XYZTuple) where {D, T} = 
EmptyBasisFunc{D, T}()

function shiftCore(::typeof(-), bf::FloatingGTBasisFuncs{𝑙1, <:Any, 1, <:Any, D, T}, 
                   didjdk::XYZTuple{𝑙2}) where {𝑙1, 𝑙2, D, T}
    xyz = bf.ijk[1].tuple .- didjdk.tuple
    for i in xyz
        i < 0 && (return EmptyBasisFunc{D, T}())
    end
    BasisFunc(bf.center, bf.gauss, XYZTuple(xyz), bf.normalizeGTO)
end

shiftCore(::Function, ::EmptyBasisFunc{D, T}, ::XYZTuple) where {D, T} = 
EmptyBasisFunc{D, T}()

"""

    decompose(bf::CompositeGTBasisFuncs, splitGaussFunc::Bool=false) -> Matrix{<:BasisFunc}

Decompose a `FloatingGTBasisFuncs` into an `Array` of `BasisFunc`s. Each column represents 
one orbital of the input basis function(s). If `splitGaussFunc` is `true`, then each column 
consists of the `BasisFunc`s each with only 1 `GaussFunc`.
"""
decompose(bf::CompositeGTBasisFuncs, splitGaussFunc::Bool=false) = 
decomposeCore(Val(splitGaussFunc), bf)

function decomposeCore(::Val{false}, bf::FloatingGTBasisFuncs{𝑙, GN, ON, PT, D, T}) where 
                      {𝑙, GN, ON, PT, D, T}
    res = Array{BasisFunc{𝑙, GN, PT, D, T}}(undef, 1, ON)
    for i in eachindex(res)
        res[i] = BasisFunc(bf.center, bf.gauss, bf.ijk[i], bf.normalizeGTO)
    end
    res
end

function decomposeCore(::Val{true}, bf::FloatingGTBasisFuncs{𝑙, GN, ON, PT, D, T}) where 
                      {𝑙, GN, ON, PT, D, T}
    res = Array{BasisFunc{𝑙, 1, PT, D, T}}(undef, GN, ON)
    for (c, ijk) in zip(eachcol(res), bf.ijk)
        c .= BasisFunc.(Ref(bf.center), bf.gauss, Ref(ijk), bf.normalizeGTO)
    end
    res
end

decomposeCore(::Val{false}, b::CompositeGTBasisFuncs{<:Any, 1}) = hcat(b)

decomposeCore(::Val{true}, b::FloatingGTBasisFuncs{<:Any, 1, 1}) = hcat(BasisFunc(b))

decomposeCore(::Val{false}, b::FloatingGTBasisFuncs{<:Any, <:Any, 1}) = hcat(BasisFunc(b))

function decomposeCore(::Val{true}, bfm::BasisFuncMix)
    bfss = map(bfm.BasisFunc) do bf
        decomposeCore(Val(true), bf)
    end
    reshape(vcat(bfss...), :, 1)
end


"""

    basisSize(subshell::String) -> Int

Return the size (number of orbitals) of each subshell.
"""
@inline basisSize(subshell::String) = SubshellSizeList[subshell]

"""

    basisSize(b::CompositeGTBasisFuncs) -> Int

Return the numbers of orbitals of the input basis function(s).
"""
@inline basisSize(::FloatingGTBasisFuncs{<:Any, <:Any, ON}) where {ON} = ON
@inline basisSize(::BasisFuncMix) = 1


# Core function to generate a customized X-Gaussian (X>1) basis function.
function genGaussFuncText(xpn::Real, con::Real, roundDigits::Int=-1)
    if roundDigits >= 0
        xpn = round(xpn, digits=roundDigits)
        con = round(con, digits=roundDigits)
    end
    "  " * alignNum(xpn) * (alignNum(con) |> rstrip) * "\n"
end

"""

    genBasisFuncText(bf::FloatingGTBasisFuncs; norm=1.0, printCenter=true) -> String

Generate a `String` of the text of the input `FloatingGTBasisFuncs`. `norm` is the 
additional normalization factor. If `printCenter` is `true`, the center coordinate 
will be added on the first line of the `String`.
"""
function genBasisFuncText(bf::FloatingGTBasisFuncs{𝑙}; 
                          norm::Float64=1.0, printCenter::Bool=true) where {𝑙}
    GFs = map(x -> genGaussFuncText(x.xpn(), x.con()), bf.gauss)
    cen = centerCoordOf(bf)
    firstLine = printCenter ? "X "*(alignNum.(cen) |> join)*"\n" : ""
    firstLine * "$(bf|>subshellOf)    $(getTypeParams(bf)[2])   $(norm)\n" * (GFs |> join)
end

"""

    genBasisFuncText(bs::Array{<:FloatingGTBasisFuncs, 1}; 
                     norm=1.0, printCenter=true, groupCenters::Bool=true) -> 
    String

Generate a `String` of the text of the input basis set. `norm` is the additional 
normalization factor. If `printCenter` is `true`, the center coordinate will be added 
on the first line of the `String`. `groupCenters` determines whether the function will 
group the basis functions with same center together.
"""
function genBasisFuncText(bs::Vector{<:FloatingGTBasisFuncs}; 
                          norm::Float64=1.0, printCenter::Bool=true, 
                          groupCenters::Bool=true)
    strs = String[]
    bfBlocks = sortBasisFuncs(bs, groupCenters)
    if groupCenters
        for b in bfBlocks
            push!(strs, joinConcentricBFuncStr(b, norm, printCenter))
        end
    else
        for b in bfBlocks
            push!(strs, genBasisFuncText(b; norm, printCenter))
        end
    end
    strs
end


function joinConcentricBFuncStr(bs::Vector{<:FloatingGTBasisFuncs},
                                norm::Float64=1.0, printFirstBFcenter::Bool=true)
    str = genBasisFuncText(bs[1]; norm, printCenter=printFirstBFcenter)
    str *= genBasisFuncText.(bs[2:end]; norm, printCenter=false) |> join
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
                           adjustFunction::F=sciNotReplace, 
                           excludeFirstNlines::Int=0, excludeLastNlines::Int=0, 
                           center::Union{AbstractArray{<:Real}, 
                                         NTuple{<:Any, Real}, 
                                         NTuple{<:Any, ParamBox}, 
                                         SpatialPoint, 
                                         Missing}=missing, 
                           unlinkCenter::Bool=false) where {F<:Function}
    adjustContent && (content = adjustFunction(content))
    lines = split.(content |> IOBuffer |> readlines)
    lines = lines[1+excludeFirstNlines : end-excludeLastNlines]
    data = [advancedParse.(i) for i in lines]
    index = findall(x -> typeof(x) != Vector{Float64} && length(x)==3, data)
    bfs = []
    for i in index
        gs1 = GaussFunc{Float64}[]
        ng = data[i][2] |> Int
        centerOld = center
        if center isa Missing && i != 1 && data[i-1][1] == "X"
            center = convert(Vector{Float64}, data[i-1][2:end])
        end
        if data[i][1] == "SP"
            gs2 = GaussFunc{Float64}[]
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
            push!(bfs, genBasisFunc((unlinkCenter ? deepcopy(center) : center), 
                                    gs1, (data[i][1] |> string), normalizeGTO=true))
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

    getParams(pbc::ParamBox, symbol::Union{Symbol, Nothing}=nothing; 
              onlyDifferentiable::Bool=false) -> 
    Union{ParamBox, Nothing}

    getParams(pbc::$(StructSpatialBasis), symbol::Union{Symbol, Nothing}=nothing; 
              onlyDifferentiable::Bool=false) -> 
    Array{<:ParamBox, 1}

    getParams(pbc::Union{Array, Tuple}, symbol::Union{Symbol, Nothing}=nothing; 
              onlyDifferentiable::Bool=false) -> 
    Array{<:ParamBox, 1}

Return the parameter(s) stored in the input container. If keyword argument `symbol` is set 
to `nothing`, then return the parameter(s); if it's set to the `Symbol` of a parameter 
(e.g. the symbol of `ParamBox{T, V}` would be `V`), return only that type of parameters 
(which might still have different indices). `onlyDifferentiable` determines whether 
ignore non-differentiable parameters. If the 1st argument is an `Array`, the entries must 
be `ParamBox` containers.
"""
function getParams(pb::ParamBox, symbol::Union{Symbol, Nothing}=nothing; 
                   onlyDifferentiable::Bool=false)
    paramFilter(pb, symbol, onlyDifferentiable) ? pb : nothing
end

function getParams(ssb::StructSpatialBasis, symbol::Union{Symbol, Nothing}=nothing; 
                   onlyDifferentiable::Bool=false)::Vector{<:ParamBox}
    filter(x->paramFilter(x, symbol, onlyDifferentiable), ssb.param) |> collect
end

getParams(cs::Array{<:ParamBox}, symbol::Union{Symbol, Nothing}=nothing; 
          onlyDifferentiable::Bool=false) = 
cs[findall(x->paramFilter(x, symbol, onlyDifferentiable), cs)]

getParams(cs::Array{<:StructSpatialBasis}, symbol::Union{Symbol, Nothing}=nothing; 
          onlyDifferentiable::Bool=false)::Vector{<:ParamBox} = 
vcat(getParams.(cs, symbol; onlyDifferentiable)...)

function getParams(cs::Array, symbol::Union{Symbol, Nothing}=nothing; 
                   onlyDifferentiable::Bool=false)::Vector{<:ParamBox}
    pbIdx = findall(x->x isa ParamBox, cs)
    vcat(getParams(convert(Vector{ParamBox}, cs[pbIdx]), symbol; onlyDifferentiable), 
         getParams(convert(Vector{StructSpatialBasis}, cs[1:end .∉ [pbIdx]]), symbol; 
                   onlyDifferentiable))
end

getParams(cs::Tuple, symbol=nothing; onlyDifferentiable=false) = 
getParams(collect(cs), symbol; onlyDifferentiable)

function paramFilter(pb::ParamBox, outSym::Union{Symbol, Nothing}=nothing, 
                     onlyDifferentiable::Bool=false)
    (outSym === nothing || outSymOfCore(pb) == outSym) && 
    (!onlyDifferentiable || pb.canDiff[])
end


const Doc_copyBasis_Eg1 = "GaussFunc(xpn=ParamBox{Float64, :α, "*
                                    "$(FLi)}(9.0)[∂][α], "*
                                    "con=ParamBox{Float64, :d, "*
                                    "$(FLi)}(2.0)[∂][d])"

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
ParamBox{Float64, :d, $(FLi)}(2.0)[∂][d]

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
    pbs = g.param .|> (copyOutVal ? outValCopy : inVarCopy)
    GaussFunc(pbs...)
end

function copyBasis(bfs::T, copyOutVal::Bool=true) where {T<:FloatingGTBasisFuncs}
    cen = bfs.center .|> (copyOutVal ? outValCopy : inVarCopy)
    gs = copyBasis.(bfs.gauss, copyOutVal)
    genBasisFunc(cen, gs, bfs.ijk; normalizeGTO=bfs.normalizeGTO)::T
end

function copyBasis(bfm::T, copyOutVal::Bool=true) where {T<:BasisFuncMix}
    bfs = copyBasis.(bfm.BasisFunc, copyOutVal)
    BasisFuncMix(bfs)::T
end


function compareParamBox(pb1::ParamBox, pb2::ParamBox)
    if pb1.canDiff[] == pb2.canDiff[]
        if pb1.canDiff[] == true
            pb1.data === pb2.data
        else
            (pb1.data === pb2.data) && (typeof(pb1.map) === typeof(pb2.map))
        end
    else
        false
    end
end

compareParamBox(pb1::ParamBox{<:Any, <:Any, FLi}, 
                pb2::ParamBox{<:Any, <:Any, FLi}) = (pb1.data === pb2.data)

function compareParamBox(pb1::ParamBox{<:Any, <:Any, FLi}, pb2::ParamBox)
    if pb2.canDiff[] == true
        pb1.data === pb2.data
    else
        false
    end
end

compareParamBox(pb1::ParamBox, pb2::ParamBox{<:Any, <:Any, FLi}) = 
compareParamBox(pb2, pb1)


"""

    markParams!(b::Union{Array{T}, T, Tuple{Vararg{StructSpatialBasis}}}, 
                filterMapping::Bool=false)  where {T<:$(StructSpatialBasis)} -> 
    Array{<:ParamBox, 1}

Mark the parameters (`ParamBox`) in input bs which can a `Vector` of `GaussFunc` or 
`FloatingGTBasisFuncs`. The identical parameters will be marked with same index. 
`filterMapping`determines weather filter out (i.e. not return) `ParamBox`s that have same 
independent variables despite they may have different mapping functions.
"""
markParams!(b::Union{Array{T}, T, Tuple{Vararg{StructSpatialBasis}}}, 
            filterMapping::Bool=false) where {T<:StructSpatialBasis} = 
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
    filterMapping ? unique(x->(x.dataName, x.index[]), pars) : pars
end

function markParamsCore!(parArray::Array{<:ParamBox{<:Any, V}}) where {V}
    res, _ = markUnique(parArray, compareFunction=compareParamBox)
    for (idx, i) in zip(parArray, res)
        idx.index[] = i
    end
    parArray
end


"""

    getVarCore(pb::ParamBox{T}, expandNonDifferentiable::Bool=false) -> 
    Vector{Pair{Symbolics.Num, T}}

Core function of `getVar`, which returns the mapping relations inside the parameter 
container. `expandNonDifferentiable` determines whether expanding the mapping relations of 
non-differentiable variable (parameters).
"""
function getVarCore(pb::ParamBox{T}, expandNonDifferentiable::Bool=false) where {T}
    if pb.canDiff[] || expandNonDifferentiable
        ivNum = inSymOf(pb)
        fNum = getFuncNum(pb.map, ivNum)
        res = Pair{Symbolics.Num, T}[fNum=>pb(), ivNum=>pb.data[]]
    else
        vNum = outSymOf(pb)
        res = Pair{Symbolics.Num, T}[vNum => pb()]
    end
end

"""

    getVar(pb::ParamBox) -> Symbolics.Num

    getVar(container::$(StructSpatialBasis)) -> Array{Symbolics.Num, 1}

Return the independent variable(s) of the input parameter container.
"""
getVar(pb::ParamBox) = getVarCore(pb, false)[end][1]

function getVar(container::CompositeGTBasisFuncs)::Vector{Symbolics.Num}
    vrs = getVarCore.(container |> getParams, false)
    getindex.(getindex.(vrs, lastindex.(vrs)), 1)
end


getVarDictCore(pb::ParamBox, expandNonDifferentiable::Bool=false) = 
getVarCore(pb, expandNonDifferentiable) |> Dict

getVarDictCore(containers, expandNonDifferentiable::Bool=false) = 
vcat(getVarCore.(containers|>getParams, expandNonDifferentiable)...) |> Dict

"""

    getVarDict(obj::Union{ParamBox, $(StructSpatialBasis), Array, Tuple}; 
               includeMapping::Bool=false) -> 
    Dict{Symbolics.Num, <:Number}

Return a `Dict` that stores the independent variable(s) of the parameter container(s) and 
its(their) corresponding value(s). If `includeMapping = true`, then the dictionary will 
also include the mapping relations between the mapped variables and the independent 
variables.
"""
getVarDict(pb::ParamBox; includeMapping::Bool=false) = 
includeMapping ? getVarDictCore(pb, true) : (inSymValOf(pb) |> Dict)

function getVarDict(containers::Union{Tuple, Array, StructSpatialBasis}; 
                    includeMapping::Bool=false)
    if includeMapping
        getVarDictCore(containers, true)
    else
        pbs = getParams(containers)
        inSymValOf.(pbs) |> Dict
    end
end


getNijk(i, j, k) = (2/π)^0.75 * 
                    sqrt( 2^(3*(i+j+k)) * factorial(i) * factorial(j) * factorial(k) / 
                          (factorial(2i) * factorial(2j) * factorial(2k)) )

getNα(i, j, k, α) = α^(0.5*(i + j + k) + 0.75)

getNijkα(i, j, k, α) = getNijk(i, j, k) * getNα(i, j, k, α)

getNijkα(ijk, α) = getNijkα(ijk[1], ijk[2], ijk[3], α)

getNorms(b::FloatingGTBasisFuncs{𝑙, GN, 1})  where {𝑙, GN} = 
getNijkα.(b.ijk[1]..., [g.xpn() for g in b.gauss])

pgf0(x, y, z, α) = exp( -α * (x^2 + y^2 + z^2) )
cgf0(x, y, z, α, d) = (d * pgf0(x, y, z, α))
cgo0(x, y, z, α, d, i, j, k, N=1.0) = (N * x^i * y^j * z^k * cgf0(x, y, z, α, d))


@inline pgf(r, α) = pgf0(r[1], r[2], r[3], α)
@inline cgf(r, α, d) = cgf0(r[1], r[2], r[3], α, d)
@inline cgo(r, α, d, l, N=getNijkα(i,j,k,α)) = 
        cgo0(r[1], r[2], r[3], α, d, l[1], l[2], l[3], N)
@inline cgo2(r, α, d, i, j, k, N=getNijkα(i,j,k,α)) = 
        cgo0(r[1], r[2], r[3], α, d, i, j, k, N)


function expressionOfCore(pb::ParamBox{<:Any, <:Any, F}, substituteValue::Bool=false) where 
                         {F}
    if substituteValue
        vrs = getVarCore(pb, false)
        recursivelyGet(vrs |> Dict, vrs[1][1])
    else
        getFuncNum(pb.map, inSymOf(pb))
    end
end

function expressionOfCore(bf::FloatingGTBasisFuncs{𝑙, GN, ON}, substituteValue::Bool=false, 
                          onlyParameter::Bool=false, splitGaussFunc::Bool=false) where 
                         {𝑙, GN, ON}
    N = bf.normalizeGTO  ?  getNijkα  :  (_...) -> 1
    R = expressionOfCore.(bf.center, substituteValue)
    α = expressionOfCore.(getfield.(bf.gauss, :xpn), substituteValue)
    d = expressionOfCore.(getfield.(bf.gauss, :con), substituteValue)
    x = onlyParameter ? (-1 .* R) : (Symbolics.variable.(:r, (1,2,3)) .- R)
    res = map(bf.ijk::NTuple{ON, XYZTuple{𝑙}}) do ijk
        i, j, k = ijk
        exprs = cgo2.(Ref(x), α, d, i, j, k, N.(i,j,k,α))
        splitGaussFunc ? collect(exprs) : sum(exprs)
    end
    ON==1 ? (vcat(res...)::Vector{Symbolics.Num}) : (hcat(res...)::Matrix{Symbolics.Num})
end

function expressionOfCore(bfm::BasisFuncMix{BN}, substituteValue::Bool=false, 
                          onlyParameter::Bool=false, splitGaussFunc::Bool=false) where {BN}
    exprs = Vector{Symbolics.Num}[expressionOfCore(bf, substituteValue, 
                                                   onlyParameter, splitGaussFunc)
                                  for bf in bfm.BasisFunc]
    splitGaussFunc ? vcat(exprs...) : sum(exprs)
end

function expressionOfCore(gf::GaussFunc, substituteValue::Bool=false)
    r = Symbolics.variable.(:r, [1:3;])
    cgf(r, expressionOfCore(gf.xpn, substituteValue), 
           expressionOfCore(gf.con, substituteValue))
end

"""

    expressionOf(bf::CompositeGTBasisFuncs, splitGaussFunc::Bool=false) -> 
    Array{<:Symbolics.Num, 2}

Return the expression(s) of a given `CompositeGTBasisFuncs` (e.g. `BasisFuncMix` or 
`FloatingGTBasisFuncs`) as a `Matrix{<:Symbolics.Num}`of which the column(s) corresponds to 
different orbitals. If `splitGaussFunc` is `true`, the column(s) will be expanded 
vertically such that the entries are `GaussFunc` inside the corresponding orbital.
"""
expressionOf(bf::CompositeGTBasisFuncs, splitGaussFunc::Bool=false) = 
expressionOfCore(bf, true, false, splitGaussFunc)

"""

    expressionOf(gf::GaussFunc) -> Symbolics.Num

Return the expression of a given `GaussFunc`.
"""
expressionOf(gf::GaussFunc) = expressionOfCore(gf, true)


function inSymbols(sym::Symbol, pool::Vector{Symbol}=ParamSyms)
    symString = sym |> string
    for i in pool
         occursin(i |> string, symString) && (return i)
    end
    return false
end

inSymbols(vr::SymbolicUtils.Sym, pool::Vector{Symbol}=ParamSyms) = 
inSymbols(Symbolics.tosymbol(vr), pool)

inSymbols(vr::SymbolicUtils.Term, pool::Vector{Symbol}=ParamSyms) = 
inSymbols(Symbolics.tosymbol(vr.f), pool)

inSymbols(::Function, args...) = false

inSymbols(vr::Num, pool::Vector{Symbol}=ParamSyms) = inSymbols(vr.val, pool)


function varVal(vr::SymbolicUtils.Sym, varDict::Dict{Num, <:Real})
    res = recursivelyGet(varDict, vr |> Num)
    if isnan(res)
        res = recursivelyGet(varDict, 
                             replaceSymbol(Symbolics.tosymbol(vr), 
                                           NoDiffMark=>"") |> Symbolics.variable)
    end
    if isnan(res)
        str = Symbolics.tosymbol(vr) |> string
        pos = findfirst(r"[₁₂₃₄₅₆₇₈₉₀]", str)[1]
        front = split(str,str[pos])[1]
        var = front*NoDiffMark*str[pos:end] |> Symbol
        recursivelyGet(varDict, var |> Symbolics.variable)
    end
    @assert !isnan(res) "Can NOT find the value of $(vr)::$(typeof(vr)) in the given "*
                        "Dict $(varDict)."
    res
end

function varVal(vr::SymbolicUtils.Add, varDict::Dict{Num, <:Real})
    r = Symbolics.@rule +(~(~xs)) => [i for i in ~(~xs)]
    vrs = r(vr)
    varVal.(vrs, Ref(varDict)) |> sum
end

function varVal(vr::SymbolicUtils.Mul, varDict::Dict{Num, <:Real})
    r = Symbolics.@rule *(~(~xs)) => [i for i in ~(~xs)]
    vrs = r(vr)
    varVal.(vrs, Ref(varDict)) |> prod
end

function varVal(vr::SymbolicUtils.Pow, varDict::Dict{Num, <:Real})
    r = Symbolics.@rule (~x)^(~k) => [~x, ~k]
    vrs = r(vr)
    varVal(vrs[1], varDict)^varVal(vrs[2], varDict)
end

function varVal(vr::SymbolicUtils.Term, varDict::Dict{Num, <:Real})
    getFsym = (t) -> t isa SymbolicUtils.Sym ? Symbolics.tosymbol(t) : Symbol(t)
    if vr.f isa Symbolics.Differential
        fv = vr.arguments[]
        if fv isa SymbolicUtils.Term
            v = fv.arguments[]
            f = getFunc(fv.f |> getFsym, 0)
            ForwardDerivative(f, varVal(v, varDict))
        else
            v = vr.f.x
            if v === fv
                1.0
            else
                varVal(Symbolics.derivative(fv, vr), varDict)
            end
        end
    elseif vr.f isa Union{SymbolicUtils.Sym, Function}
        fSymbol = vr.f |> getFsym
        f = getFunc(fSymbol, 0)
        v = varVal(vr.arguments[], varDict)
        f(v)
    else
        NaN
    end
end

varVal(vr::Num, varDict::Dict{Num, <:Real}) = varVal(vr.val, varDict)

varVal(vr::Real, args...) = itself(vr)

varVal(vr::Rational, args...) = round(vr |> Float64, digits=14)


function detectXYZ(i::SymbolicUtils.Symbolic, varDict::Dict{Num, <:Real})
    xyz = zeros(Int, 3)
    if i isa SymbolicUtils.Pow
        for j = 1:3
            if inSymbols(i.base, [ParamSyms[j]]) != false
                xyz[j] = i.exp
                return ((-1)^(i.exp), xyz) # (-X)^k -> (true, (-1)^k, [0, 0, 0])
            end
        end
    else
        for j = 1:3
            if inSymbols(i, [ParamSyms[j]]) != false
                xyz[j] = 1
                return (-1, xyz)
            end
        end
    end
    (varVal(i, varDict), xyz)
end

detectXYZ(vr::Real, _) = (vr, [0,0,0])


# res = [d_ratio, Δi, Δj, Δk]
function diffTransferCore(trm::SymbolicUtils.Symbolic, varDict::Dict{Num, <:Real})
    d = 1.0
    xyz = zeros(Int, 3)
    r = Symbolics.@rule *(~~xs) => ~~xs
    trms = SymbolicUtils.simplify(trm, rewriter=r)
    !(trms isa SubArray) && (trms = [trms])
    for vr in trms
        coeff, ijks = detectXYZ(vr, varDict)
        xyz += ijks
        d *= coeff
    end
    (d, xyz)
end

diffTransferCore(trm::Real, _...) = (Float64(trm), [0,0,0])


function diffTransfer(trm::Num, varDict::Dict{Num, <:Real})
    trms = splitTerm(trm)
    diffTransferCore.(trms, Ref(varDict))
end


function diffInfo(bf::CompositeGTBasisFuncs{BN, 1}, vr, varDict) where {BN}
    exprs = expressionOfCore(bf, false, true, true)
    relDiffs = Symbolics.derivative.(log.(exprs), vr)
    diffTransfer.(relDiffs, Ref(varDict))
end


function diffInfoToBasisFunc(bf::CompositeGTBasisFuncs{BN, 1}, 
                             info::Vector{Vector{Tuple{Float64, Vector{Int}}}}) where {BN}
    bs = decomposeCore(Val(true), bf)
    bss = map(info, bs) do gInfo, bf
        map(gInfo) do dxyz
           sgf = copyBasis(bf)
           sgf.gauss[1].con[] *= dxyz[1]
           xyz = dxyz[2]
           xyz == [0,0,0] ? sgf : shift(sgf, xyz)
        end::Vector{<:BasisFunc{<:Any, 1}}
    end
    vcat(bss...)::Vector{<:BasisFunc{<:Any, 1}} |> sumOf
end