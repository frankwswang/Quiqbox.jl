export GaussFunc, genExponent, genContraction, genSpatialPoint, BasisFunc, BasisFuncs, 
       genBasisFunc, centerOf, centerCoordOf, GTBasis, sortBasisFuncs, add, mul, shift, 
       decompose, basisSize, genBasisFuncText, genBFuncsFromText, assignCenter!, 
       makeCenter, getParams, copyBasis, uniqueParams!, getVar, getVarDict, expressionOf

using Symbolics
using SymbolicUtils
using LinearAlgebra: diag
using ForwardDiff: derivative as ForwardDerivative

"""

    GaussFunc <: AbstractGaussFunc

A single contracted gaussian function `struct` from package Quiqbox.

≡≡≡ Field(s) ≡≡≡

`xpn::ParamBox{Float64, :$(αParamSym)}`：Exponent of the gaussian function.

`con::ParamBox{Float64, :$(dParamSym)}`: Contraction coefficient of the gaussian 
function.

`param::NTuple{2, ParamBox}`: A Tuple that stores the `ParamBox`s of `xpn` and `con`.

≡≡≡ Initialization Method(s) ≡≡≡

    GaussFunc(xpn::ParamBox, con::ParamBox) -> GaussFunc

    GaussFunc(xpn::Real, con::Real) -> GaussFunc

"""
struct GaussFunc <: AbstractGaussFunc
    xpn::ParamBox{Float64, αParamSym}
    con::ParamBox{Float64, dParamSym}
    param::Tuple{ParamBox{Float64, αParamSym}, 
                 ParamBox{Float64, dParamSym}}

    GaussFunc(xpn::ParamBox{Float64, αParamSym}, 
              con::ParamBox{Float64, dParamSym}) = 
    new(xpn, con, (xpn, con))
end

function GaussFunc(e::Real, c::Real)
    xpn = ParamBox(e, αParamSym)
    con = ParamBox(c, dParamSym)
    GaussFunc(xpn, con)
end

GaussFunc(xpn::ParamBox, con::ParamBox) = GaussFunc(genExponent(xpn), genContraction(con))


"""

    genExponent(e::Real, mapFunction::Function; canDiff::Bool=true, 
                roundDigits::Int=15, dataName::Symbol=:undef) -> 
    ParamBox{Float64, :$(αParamSym)}

    genExponent(e::Array{T, 0}, mapFunction::Function; canDiff::Bool=true, 
                dataName::Symbol=:undef) where {T<:Real} -> 
    ParamBox{Float64, :$(αParamSym)}

Construct a `ParamBox` for an exponent coefficient given a value. Keywords `mapFunction` 
and `canDiff` work the same way as in a general constructor of a `ParamBox`. If 
`roundDigits < 0`, there won't be rounding for input data.
"""
genExponent(e::Real, mapFunction::F; canDiff::Bool=true, 
               roundDigits::Int=15, dataName::Symbol=:undef) where {F<:Function} = 
ParamBox{αParamSym}(mapFunction, e, genIndex(nothing), fill(canDiff), dataName; roundDigits)

genExponent(e::Array{T, 0}, mapFunction::F; canDiff::Bool=true, 
               dataName::Symbol=:undef) where {T<:Real, F<:Function} = 
ParamBox{αParamSym}(mapFunction, e, genIndex(nothing), fill(canDiff), dataName)



"""

    genExponent(e::Real; roundDigits::Int=15) -> ParamBox{Float64, :$(αParamSym)}

    genExponent(e::Array{T, 0}) where {T<:Real} -> ParamBox{Float64, :$(αParamSym)}

"""
genExponent(e::Real; roundDigits::Int=15) = 
ParamBox{αParamSym}(FunctionType{:itself}(), e, genIndex(nothing); roundDigits)

genExponent(e::Array{T, 0}) where {T<:Real} = 
ParamBox{αParamSym, :itself}(e, genIndex(nothing))


"""

    genExponent(pb::ParamBox{Float64}) -> ParamBox{Float64, :$(αParamSym)}

Convert a `$(ParamBox)` to an exponent coefficient parameter.
"""
genExponent(pb::ParamBox{Float64, V, F}) where {V, F} = ParamBox{αParamSym}(pb)


"""

    genContraction(c::Real, mapFunction::Function; canDiff::Bool=true, 
                roundDigits::Int=15, dataName::Symbol=:undef) -> 
    ParamBox{Float64, :$(dParamSym)}

    genContraction(c::Array{T, 0}, mapFunction::Function; canDiff::Bool=true, 
                dataName::Symbol=:undef) where {T<:Real} -> 
    ParamBox{Float64, :$(dParamSym)}

Construct a `ParamBox` for an contraction coefficient given a value. Keywords `mapFunction` 
and `canDiff` work the same way as in a general constructor of a `ParamBox`. If 
`roundDigits < 0`, there won't be rounding for input data.
"""
genContraction(c::Real, mapFunction::F; canDiff::Bool=true, 
               roundDigits::Int=15, dataName::Symbol=:undef) where {F<:Function} = 
ParamBox{dParamSym}(mapFunction, c, genIndex(nothing), fill(canDiff), dataName; roundDigits)

genContraction(c::Array{T, 0}, mapFunction::F; canDiff::Bool=true, 
               dataName::Symbol=:undef) where {T<:Real, F<:Function} = 
ParamBox{dParamSym}(mapFunction, c, genIndex(nothing), fill(canDiff), dataName)

"""

    genContraction(c::Real; roundDigits::Int=15) -> ParamBox{Float64, :$(dParamSym)}

    genContraction(c::Array{T, 0}) where {T<:Real} -> ParamBox{Float64, :$(dParamSym)}

"""
genContraction(c::Real; roundDigits::Int=15) = 
ParamBox{dParamSym}(FunctionType{:itself}(), c, genIndex(nothing); roundDigits)

genContraction(c::Array{T, 0}) where {T<:Real} = 
ParamBox{dParamSym, :itself}(c, genIndex(nothing))

"""

    genContraction(pb::ParamBox{Float64}) -> ParamBox{Float64, :$(dParamSym)}

Convert a `$(ParamBox)` to an exponent coefficient parameter.
"""
genContraction(pb::ParamBox{Float64, V, F}) where {V, F} = ParamBox{dParamSym}(pb)


const Doc_genSpatialPoint_Eg1 = "(ParamBox{Float64, :X, :itself}(1.0)[∂][X], " * 
                                 "ParamBox{Float64, :Y, :itself}(2.0)[∂][Y], " * 
                                 "ParamBox{Float64, :Z, :itself}(3.0)[∂][Z])"

"""

    genSpatialPoint(point::Union{Tuple, Vector}, mapFunction::F=itself; canDiff::Bool=true, 
                    roundDigits::Int=15, dataName::Symbol=:undef)

Return the parameter(s) of a spatial coordinate in terms of `ParamBox`. Keywords 
`mapFunction` and `canDiff` work the same way as in a general constructor of a `ParamBox`. 
If `roundDigits < 0` or the input `c` is a 0-d `Array`, there won't be rounding for input 
data.

≡≡≡ Method 1 ≡≡≡

    genSpatialPoint(point::Vector, mapFunction::F=itself; canDiff::Bool=true, 
                    roundDigits::Int=15, dataName::Symbol=:undef) -> 
    Tuple{ParamBox{Float64, :X}, 
          ParamBox{Float64, :Y}, 
          ParamBox{Float64, :Z}}

Return the parameters that represent a spatial point. The entry of input `Vector` can 
be either a `Real` number or a `Array{Float64, 0}`.

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
ParamBox{Float64, :X, :itself}(1.0)[∂][X]

julia> v2[1][] = 1.2
1.2

julia> p2[1]
ParamBox{Float64, :X, :itself}(1.2)[∂][X]
```

≡≡≡ Method 2 ≡≡≡

    genSpatialPoint(point::Tuple{Union{Real, Array{Float64, 0}}, Int}, 
                    mapFunction::F=itself; canDiff::Bool=true, roundDigits::Int=15, 
                    dataName::Symbol=:undef) -> 
    ParamBox{Float64}

Return the component of a spatial point given its value (or 0-D container) and index.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genSpatialPoint((1.2, 1))
ParamBox{Float64, :X, :itself}(1.2)[∂][X]

julia> pointY1 = fill(2.0)
0-dimensional Array{Float64, 0}:
2.0

julia> Y1 = genSpatialPoint((pointY1, 2))
ParamBox{Float64, :Y, :itself}(2.0)[∂][Y]

julia> pointY1[] = 1.5
1.5

julia> Y1
ParamBox{Float64, :Y, :itself}(1.5)[∂][Y]
```
"""
genSpatialPoint(point::Union{Tuple, Vector}, mapFunction::F=itself; 
                canDiff::Bool=true, roundDigits::Int=15, 
                dataName::Symbol=:undef) where {F<:Function} = 
genSpatialPointCore(point, mapFunction, canDiff, roundDigits, dataName)

function genSpatialPointCore(point::Tuple{Union{Real, Array{Float64, 0}}, Int}, 
                             mapFunction::F=itself, canDiff::Bool=true, roundDigits::Int=15, 
                             dataName::Symbol=:undef) where {F<:Function}
    dim = Symbol[:X, :Y, :Z]
    n = if point[1] isa Array
            point[1]
        elseif roundDigits < 0
            Float64(point[1])
        else
            round(point[1][], digits=roundDigits)
        end
    ParamBox(n, ParamList[dim[point[2]]], mapFunction, dataName; canDiff)
end

function genSpatialPointCore(v::Vector, mapFunction::F=itself, canDiff::Bool=true, 
                             roundDigits::Int=15, 
                             dataName::Symbol=:undef) where {F<:Function}
    genSpatialPointCore.(((v[1], 1), (v[2], 2), (v[3], 3)), 
                         mapFunction, canDiff, roundDigits, dataName)
end


"""

    BasisFunc{𝑙, GN} <: FloatingGTBasisFuncs{𝑙, GN, 1}

A (floating) basis function with the center attached to it instead of any nucleus.

≡≡≡ Field(s) ≡≡≡

`center::NTuple{3, ParamBox}`: The center coordinate in form of a 3-element `ParamBox`-type 
`Tuple`.

`gauss::NTuple{N, GaussFunc}`: Gaussian functions within the basis function.

`subshell::String`: The subshell (angular momentum symbol).

`ijk::Tuple{$(XYZTuple){𝑙}}`: Cartesian representation (pseudo-quantum number) of the 
angular momentum orientation. E.g., s (X⁰Y⁰Z⁰) would be `$(XYZTuple(0, 0, 0))`. For 
convenient syntax, `.ijk[]` converts it to a `NTuple{3, Int}`.

`normalizeGTO::Bool`: Whether the GTO`::GaussFunc` will be normalized in calculations.

`param::Tuple{Vararg{<:ParamBox}}`： All the tunable parameters`::ParamBox` stored in the 
`BasisFunc`.

≡≡≡ Initialization Method(s) ≡≡≡

    BasisFunc(center::Tuple{Vararg{<:ParamBox}}, gauss::NTuple{GN, GaussFunc}, 
              ijk::NTuple{3, Int}, normalizeGTO::Bool) where {GN} -> 
    BasisFunc{𝑙, GN} where {𝑙}

    BasisFunc(cen::Tuple{Vararg{<:ParamBox}}, gauss::GaussFunc, ijk::NTuple{3, Int}, 
              normalizeGTO::Bool) ->
    BasisFunc{𝑙, 1} where {𝑙}
"""
struct BasisFunc{𝑙, GN} <: FloatingGTBasisFuncs{𝑙, GN, 1}
    center::Tuple{ParamBox{Float64, XParamSym}, 
                  ParamBox{Float64, YParamSym}, 
                  ParamBox{Float64, ZParamSym}}
    gauss::NTuple{GN, GaussFunc}
    subshell::String
    ijk::Tuple{XYZTuple{𝑙}}
    normalizeGTO::Bool
    param::Tuple{Vararg{<:ParamBox}}

    function BasisFunc(cen::Tuple{Vararg{<:ParamBox}}, gs::NTuple{GN, GaussFunc}, 
                       ijk::Tuple{XYZTuple{𝑙}}, normalizeGTO::Bool) where {𝑙, GN}
        subshell = SubshellNames[𝑙+1]
        len = 3 + GN*2
        pars = Array{ParamBox}(undef, len)
        pars[1], pars[2], pars[3] = cen
        for (g, k) in zip(gs, 4:2:(len-1))
            pars[k], pars[k+1] = g.param
        end
        new{𝑙, GN}(cen, gs, subshell, ijk, normalizeGTO, pars|>Tuple)
    end
end

BasisFunc(cen, gs::NTuple{GN, GaussFunc}, ijk::XYZTuple{𝑙}, 
          normalizeGTO=false) where {GN, 𝑙} = 
BasisFunc(cen, gs, (ijk,), normalizeGTO)

BasisFunc(cen, g::GaussFunc, ijk, normalizeGTO=false) = 
BasisFunc(cen, (g,), ijk, normalizeGTO)


"""

    BasisFuncs{𝑙, GN, ON} <: FloatingGTBasisFuncs{𝑙, GN, ON}

A group of basis functions with identical parameters except they have different subshell 
under the specified angular momentum. It has the same fields as `BasisFunc` and 
specifically, for `ijk`, the size of the it (`ON`) can be larger than 1 (no larger than the 
size of the corresponding subshell).
"""
struct BasisFuncs{𝑙, GN, ON} <: FloatingGTBasisFuncs{𝑙, GN, ON}
    center::Tuple{ParamBox{Float64, XParamSym}, 
                  ParamBox{Float64, YParamSym}, 
                  ParamBox{Float64, ZParamSym}}
    gauss::NTuple{GN, GaussFunc}
    subshell::String
    ijk::NTuple{ON, XYZTuple{𝑙}}
    normalizeGTO::Bool
    param::Tuple{Vararg{<:ParamBox}}

    function BasisFuncs(cen::Tuple{Vararg{<:ParamBox}}, gs::NTuple{GN, GaussFunc}, 
                        ijks::NTuple{ON, XYZTuple{𝑙}}, normalizeGTO::Bool=false) where 
                       {𝑙, GN, ON}
        subshell = SubshellNames[𝑙+1]
        ss = SubshellDimList[subshell]
        @assert ON <= ss "The total number of `ijk` should be no more than $(ss) as " * 
                         "they are in $(subshell) subshell."
        ijks = sort(ijks|>collect, rev=true) |> Tuple
        pars = ParamBox[]
        len = 3 + GN*2
        pars = Array{ParamBox}(undef, len)
        pars[1], pars[2], pars[3] = cen
        for (g, k) in zip(gs, 4:2:(len-1))
            pars[k], pars[k+1] = g.param
        end
        new{𝑙, GN, ON}(cen, gs, subshell, ijks, normalizeGTO, pars|>Tuple)
    end
end

BasisFuncs(cen, g::GaussFunc, ijks, normalizeGTO=false) = 
BasisFuncs(cen, (g,), ijks, normalizeGTO)

"""

    genBasisFunc(center::Union{AbstractArray, NTuple{3, ParamBox}, Missing}, 
                 args..., kws...) -> 
    B where {B<:Union{FloatingGTBasisFuncs, Array{<:FloatingGTBasisFuncs}}}

Constructor of `BasisFunc` and `BasisFuncs`, but it also returns different kinds of 
collections of them based on the applied methods. The first argument `center` can be a 3-D 
coordinate (e.g. `Array{Float64, 1}`), a `NTuple{3}` of spatial points (e.g. generated by 
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
BasisFunc{1, 1}(gauss, subshell, center)[X⁰Y¹Z⁰][0.0, 0.0, 0.0]
```

≡≡≡ Method 2 ≡≡≡

    genBasisFunc(center, gExpsANDgCons::NTuple{2, Array{<:Real, 1}}, subshell="S"; kw...)

Instead of directly inputting `GaussFunc`, one can also input a 2-element `Tuple` of the 
exponent(s) and contraction coefficient(s) corresponding to the same `GaussFunc`(s).

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genBasisFunc([0,0,0], (2, 1), "P")
BasisFuncs{1, 1, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], ([2, 1.5], [1, 0.5]), "P")
BasisFuncs{1, 2, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]
```

≡≡≡ Method 3 ≡≡≡

    genBasisFunc(center, gs::Union{GaussFunc, Array{GaussFunc, 1}}, subshell::String="S", 
                 ijkFilter::NTuple{N, Bool}=fill(true, SubshellDimList[subshell])|>Tuple; 
                 normalizeGTO::Bool=false) where {N}

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genBasisFunc([0,0,0], GaussFunc(2,1), "S")
BasisFunc{0, 1}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], GaussFunc(2,1), "P")
BasisFuncs{1, 1, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]
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
 BasisFunc{0, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFunc{0, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFuncs{1, 3, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], "STO-3G")
1-element Vector{Quiqbox.FloatingGTBasisFuncs}:
 BasisFunc{0, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], ["STO-2G", "STO-3G"])
2-element Vector{Quiqbox.FloatingGTBasisFuncs}:
 BasisFunc{0, 2}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFunc{0, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], [("STO-2G", "He"), ("STO-3G", "O")])
4-element Vector{Quiqbox.FloatingGTBasisFuncs}:
 BasisFunc{0, 2}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFunc{0, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFunc{0, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFuncs{1, 3, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]
```
"""
genBasisFunc(cen::NTuple{3, ParamBox}, gs::NTuple{GN, GaussFunc}, 
             ijk::XYZTuple{𝑙}=XYZTuple(0,0,0); normalizeGTO::Bool=false) where {GN, 𝑙} = 
BasisFunc(cen, gs, ijk, normalizeGTO)

genBasisFunc(cen::NTuple{3, ParamBox}, gs::NTuple{GN, GaussFunc}, 
             ijk::NTuple{3, Int}; normalizeGTO::Bool=false) where {GN} = 
BasisFunc(cen, gs, ijk|>XYZTuple, normalizeGTO)

genBasisFunc(cen::NTuple{3, ParamBox}, gs::NTuple{GN, GaussFunc}, 
             ijks::NTuple{ON, XYZTuple{𝑙}}; normalizeGTO::Bool=false) where {GN, ON, 𝑙} = 
BasisFuncs(cen, gs, ijks, normalizeGTO)

genBasisFunc(cen::NTuple{3, ParamBox}, gs::NTuple{GN, GaussFunc}, 
             ijks::NTuple{ON, NTuple{3, Int}}; normalizeGTO::Bool=false) where {GN, ON} = 
BasisFuncs(cen, gs, ijks.|>XYZTuple, normalizeGTO)

genBasisFunc(cen::NTuple{3, ParamBox}, gs::NTuple{GN, GaussFunc}, 
             ijks::Vector{XYZTuple{𝑙}}; normalizeGTO::Bool=false) where {GN, 𝑙} = 
genBasisFunc(cen, gs, ijks|>Tuple; normalizeGTO)

genBasisFunc(cen::NTuple{3, ParamBox}, gs::NTuple{GN, GaussFunc}, 
             ijks::Vector{NTuple{3, Int}}; normalizeGTO::Bool=false) where {GN} = 
genBasisFunc(cen, gs, ijks|>Tuple; normalizeGTO)

genBasisFunc(cen::NTuple{3, ParamBox}, gs::NTuple{GN, GaussFunc}, 
             ijk::Tuple{XYZTuple{𝑙}}; normalizeGTO::Bool=false) where {GN, 𝑙} = 
genBasisFunc(cen, gs, ijk[1]; normalizeGTO)

genBasisFunc(cen::NTuple{3, ParamBox}, gs::NTuple{GN, GaussFunc}, 
             ijk::Tuple{NTuple{3, Int}}; normalizeGTO::Bool=false) where {GN} = 
genBasisFunc(cen, gs, ijk[1]; normalizeGTO)

function genBasisFunc(cen::NTuple{3, ParamBox}, gs::NTuple{GN, GaussFunc}, subshell::String; 
                      normalizeGTO::Bool=false) where {GN}
    genBasisFunc(cen, gs, SubshellSuborderList[subshell]; normalizeGTO)
end

function genBasisFunc(cen::NTuple{3, ParamBox}, gs::NTuple{GN, GaussFunc}, subshell::String, 
                      ijkFilter::NTuple{N, Bool}; normalizeGTO::Bool=false) where {GN, N}
    subshellSize = SubshellDimList[subshell]
    @assert N == subshellSize "The length of `ijkFilter` should be $(subshellSize) "*
                              "to match the subshell's size."
    genBasisFunc(cen, gs, 
                 SubshellSuborderList[subshell][1:end .∈ [findall(x->x==true, ijkFilter)]]; 
                 normalizeGTO)
end

function genBasisFunc(cen::NTuple{3, ParamBox}, xpnsANDcons::NTuple{2, Vector{<:Real}}, 
                      ijkOrSubshell=XYZTuple(0,0,0); normalizeGTO::Bool=false)
    @compareLength xpnsANDcons[1] xpnsANDcons[2] "exponents" "contraction coefficients"
    genBasisFunc(cen, GaussFunc.(xpnsANDcons[1], xpnsANDcons[2]), ijkOrSubshell; 
                 normalizeGTO)
end

genBasisFunc(cen::NTuple{3, ParamBox}, xpnANDcon::NTuple{2, Real}, 
             ijkOrSubshell=XYZTuple(0,0,0); normalizeGTO::Bool=false) = 
genBasisFunc(cen, (GaussFunc(xpnANDcon[1], xpnANDcon[2]),), ijkOrSubshell; normalizeGTO)

function genBasisFunc(center::NTuple{3, ParamBox}, BSKeyANDnuc::Vector{NTuple{2, String}}; 
                      unlinkCenter::Bool=false)
    bases = FloatingGTBasisFuncs[]
    for k in BSKeyANDnuc
        BFMcontent = BasisSetList[k[1]][AtomicNumberList[k[2]]]
        append!(bases, genBFuncsFromText(BFMcontent; adjustContent=true, 
                excludeLastNlines=1, center, unlinkCenter))
    end
    bases
end

genBasisFunc(cen::NTuple{3, ParamBox}, BSKeyANDnuc::NTuple{2, String}; 
             unlinkCenter::Bool=false) = 
genBasisFunc(cen, [BSKeyANDnuc]; unlinkCenter)

genBasisFunc(cen::NTuple{3, ParamBox}, BSkey::Vector{String}; nucleus::String="H", 
             unlinkCenter::Bool=false) = 
genBasisFunc(cen, [(i, nucleus) for i in BSkey]; unlinkCenter)

genBasisFunc(cen::NTuple{3, ParamBox}, BSkey::String; nucleus::String="H", 
             unlinkCenter::Bool=false) = 
genBasisFunc(cen, [BSkey]; nucleus, unlinkCenter)

# A few methods for convenient arguments omissions and mutations.
genBasisFunc(cen::NTuple{3, ParamBox}, gs::Vector{GaussFunc}, args...; kws...) = 
genBasisFunc(cen, gs|>Tuple, args...; kws...)

genBasisFunc(cen::NTuple{3, ParamBox}, g::GaussFunc, args...; kws...) = 
genBasisFunc(cen, (g,), args...; kws...)

function genBasisFunc(coord::AbstractArray, args...; kws...)
    @assert length(coord) == 3 "The dimension of the center should be 3."
    x = ParamBox(coord[1], XParamSym)
    y = ParamBox(coord[2], YParamSym)
    z = ParamBox(coord[3], ZParamSym)
    genBasisFunc((x,y,z), args...; kws...)
end

genBasisFunc(::Missing, args...; kws...) = genBasisFunc([NaN, NaN, NaN], args...; kws...)

genBasisFunc(bf::FloatingGTBasisFuncs) = itself(bf)

genBasisFunc(bs::Vector{<:FloatingGTBasisFuncs}) = sortBasisFuncs(bs)


"""

    GTBasis{N, BT} <: BasisSetData{N}

The container to store basis set information.

≡≡≡ Field(s) ≡≡≡

`basis::Array{<:AbstractGTBasisFuncs, 1}`: Basis set.

`S::Array{<:Number, 2}`: Overlap matrix.

`Te::Array{<:Number, 2}`: Kinetic energy part of the electronic core Hamiltonian.

`eeI::Array{<:Number, 4}`: Electron-electron interaction.

`getVne::Function`: A `Function` that returns the nuclear attraction Hamiltonian when 
nuclei`::Array{String, 1}` and their coordinates`::Array{<:AbstractArray, 1}` are input.

getHcore::Function: Similar as `getVne`, a `Function` that returns the core Hamiltonian 
when nuclei and their coordinates of same `DataType` are input.

≡≡≡ Initialization Method(s) ≡≡≡

    GTBasis(basis::Array{<:AbstractGTBasisFuncs, 1}, S::Matrix{<:Number}, 
            Te::Matrix{<:Number}, eeI::Array{<:Number, 4}) -> 
    GTBasis

    GTBasis(basis::Array{<:AbstractGTBasisFuncs, 1}, sortBasis::Bool=true) -> GTBasis

Directly construct a `GTBasis` given a basis set. Argument `sortBasis` determines whether 
the constructor will sort the input basis functions using `sortBasisFuncs` before build 
a `GTBasis`.
"""
struct GTBasis{N, BT} <: BasisSetData{N}
    basis::Vector{<:AbstractGTBasisFuncs}
    S::Matrix{<:Number}
    Te::Matrix{<:Number}
    eeI::Array{<:Number, 4}
    getVne::Function
    getHcore::Function

    function GTBasis(basis::Vector{<:AbstractGTBasisFuncs},
                     S::Matrix{<:Number}, Te::Matrix{<:Number}, eeI::Array{<:Number, 4})
        new{basisSize.(basis) |> sum, typeof(basis)}(basis, S, Te, eeI, 
            (mol, nucCoords) -> nucAttractions(basis, mol, nucCoords),
            (mol, nucCoords) -> nucAttractions(basis, mol, nucCoords) + Te)
    end
end

function GTBasis(basis::Vector{<:AbstractGTBasisFuncs}, sortBasis::Bool=true)
    bs = sortBasis ? sortBasisFuncs(basis) : basis
    GTBasis(bs, overlaps(bs), elecKinetics(bs), eeInteractions(bs))
end

"""

    sortBasisFuncs(bs::Array{<:FloatingGTBasisFuncs}; groupCenters::Bool=false) -> Array

Sort basis functions. If `groupCenters = true`, Then the function will return an 
`Array{<:Array{<:FloatingGTBasisFuncs, 1}, 1}` in which the arrays are grouped basis 
functions with same center coordinates.
"""
function sortBasisFuncs(bs::Array{<:FloatingGTBasisFuncs}; groupCenters::Bool=false)
    bfBlocks = Vector{<:FloatingGTBasisFuncs}[]
    sortedBasis = groupedSort(bs[:], centerCoordOf)
    for subbs in sortedBasis
        ijkn = [(i.ijk[1].tuple, typeof(i).parameters[2]) for i in subbs]

        # Reversed order within same subshell but ordinary order among different subshells.
        sortVec = sortperm(map(ijkn) do x
                               val = x[1]
                               [-sum(val); val; x[2]]
                           end, 
                           rev=true)
        push!(bfBlocks, subbs[sortVec])
    end
    groupCenters ? bfBlocks : vcat(bfBlocks...)
end


function isFull(bfs::FloatingGTBasisFuncs)
    bfs.subshell == "S" || length(bfs.ijk) == SubshellDimList[bfs.subshell]
end
isFull(::Any) = false


function ijkIndex(b::FloatingGTBasisFuncs)
    isFull(b) && (return :)
    [ijkIndexList[ijk] for ijk in b.ijk]
end


"""

    centerOf(bf::FloatingGTBasisFuncs) -> 
    Tuple{ParamBox{Float64, $(XParamSym)}, 
          ParamBox{Float64, $(YParamSym)}, 
          ParamBox{Float64, $(ZParamSym)}}

Return the center of the input `FloatingGTBasisFuncs`.
"""
centerOf(bf::FloatingGTBasisFuncs) = bf.center


"""

    centerCoordOf(bf::FloatingGTBasisFuncs) -> Vector{Float64}

Return the center coordinate of the input `FloatingGTBasisFuncs`.
"""
centerCoordOf(bf::FloatingGTBasisFuncs) = Float64[outValOf(i) for i in bf.center]


"""

    BasisFuncMix{BN, GN} <: CompositeGTBasisFuncs{BN, 1}

Sum of multiple `FloatingGTBasisFuncs` without any reformulation, treated as one basis 
Function in the integral calculation.

≡≡≡ Field(s) ≡≡≡

`BasisFunc::NTuple{BN, FloatingGTBasisFuncs{<:Any, <:Any, 1}}`: Inside basis functions

`param::Tuple{Vararg{<:ParamBox}}`: Inside parameters.

≡≡≡ Initialization Method(s) ≡≡≡

    BasisFuncMix(bfs::Array{<:FloatingGTBasisFuncs{𝑙, GN, 1} where {𝑙, GN}, 1}) ->
    BasisFuncMix{BN, GN}

"""
struct BasisFuncMix{BN, GN} <: CompositeGTBasisFuncs{BN, 1}
    BasisFunc::NTuple{BN, FloatingGTBasisFuncs{<:Any, <:Any, 1}}
    param::Tuple{Vararg{<:ParamBox}}
    function BasisFuncMix(bfs::Vector{<:FloatingGTBasisFuncs{𝑙, GN, 1} where {𝑙, GN}})
        pars = ParamBox[]
        for bf in bfs
            append!(pars, bf.param)
        end
        new{length(bfs), [length(bf.gauss) for bf in bfs]|>sum}(bfs |> Tuple, pars |> Tuple)
    end
end
BasisFuncMix(bf::BasisFunc) = BasisFuncMix([bf])
BasisFuncMix(bfs::BasisFuncs) = BasisFuncMix.(decompose(bfs))
BasisFuncMix(bfm::BasisFuncMix) = itself(bfm)


unpackBasisFuncs(bfm::BasisFuncMix{BN, GN}) where {BN, GN} = bfm.BasisFunc |> collect
unpackBasisFuncs(bf::FloatingGTBasisFuncs) = typeof(bf)[bf]
unpackBasisFuncs(::Any) = FloatingGTBasisFuncs[]

unpackBasis(bfm::BasisFuncMix{BN, GN}) where {BN, GN} = bfm.BasisFunc |> collect

function sumOf(bfs::Array{<:BasisFunc})::CompositeGTBasisFuncs{<:Any, 1}
    arr1 = convert(Vector{BasisFunc}, sortBasisFuncs(bfs[:]))
    arr2 = BasisFunc[]
    while length(arr1) > 1
        temp = add(arr1[1], arr1[2])
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
        vcat((vcat(arr2, arr1) .|> unpackBasisFuncs)...) |> BasisFuncMix
    end
end

sumOf(bfms::Array{<:CompositeGTBasisFuncs{<:Any, 1}}) = 
vcat((bfms .|> unpackBasisFuncs)...) |> sumOf


mergeGaussFuncs(gf::GaussFunc) = itself(gf)

function mergeGaussFuncs(gf1::GaussFunc, gf2::GaussFunc)::Vector{GaussFunc}
    xpn1 = gf1.xpn
    xpn2 = gf2.xpn
    con1 = gf1.con
    con2 = gf2.con
    if xpn1() === xpn2()
        if xpn1 === xpn2 || hasIdentical(xpn1, xpn2)
            xpn = xpn1
        elseif hasEqual(xpn1, xpn2)
            xpn = deepcopy(xpn1)
        else
            xpn = genExponent(xpn1())
        end

        if con1 === con2 || hasIdentical(con1, con2)
            res = GaussFunc(xpn, con1) * 2.0
        elseif hasEqual(con1, con2)
            res = GaussFunc(xpn, deepcopy(con1)) * 2.0
        else
            res = GaussFunc(xpn, genContraction(con1()+con2()))
        end

        return [res]
    else
        return [gf1, gf2]
    end
end

function mergeGaussFuncs(gf1::GaussFunc, 
                         gf2::GaussFunc, 
                         gf3::GaussFunc...)::Vector{GaussFunc}
    gfs = vcat(gf1, gf2, gf3 |> collect)
    xpns = Float64[i.xpn() for i in gfs]
    res = GaussFunc[]
    _, uList = markUnique(xpns, compareFunction=(==))
    for val in uList
        group = gfs[findall(i->i==val, xpns)]
        gxpns = getfield.(group, :xpn)
        gcons = getfield.(group, :con)
        _, uxpns = markUnique(gxpns, compareFunction=hasIdentical)
        _, ucons = markUnique(gcons, compareFunction=hasIdentical)
        if length(uxpns) == 1
            xpn = uxpns[]
        elseif (markUnique(gxpns)[2] |> length) == 1
            xpn = uxpns[1] |> deepcopy
        else
            xpn = genExponent(val())
        end
        if length(ucons) == 1
            push!( res, GaussFunc(xpn, ucons[1]) * length(group) )
        elseif (markUnique(gcons)[2] |> length) == 1
            push!( res, GaussFunc(xpn, ucons[1] |> deepcopy) * length(group) )
        else
            push!( res, GaussFunc(xpn, genContraction(Float64[i() for i in gcons] |> sum)) )
        end
    end
    res
end


"""

    add(b::CompositeGTBasisFuncs{<:Any, 1}) -> CompositeGTBasisFuncs{<:Any, 1}

    add(b1::CompositeGTBasisFuncs{<:Any, 1}, b2::CompositeGTBasisFuncs{<:Any, 1}) ->
    CompositeGTBasisFuncs{<:Any, 1}

Addition between `CompositeGTBasisFuncs{<:Any, 1}` such as `BasisFunc` and 
`Quiqbox.BasisFuncMix`. It can be called using `+` syntax.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> bf1 = genBasisFunc([1,1,1], (2,1))
BasisFunc{0, 1}(gauss, subshell, center)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bf2 = genBasisFunc([1,1,1], (2,2))
BasisFunc{0, 1}(gauss, subshell, center)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bf3 = bf1 + bf2
BasisFunc{0, 1}(gauss, subshell, center)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bf3.gauss[1].con[]
3.0
```
"""
function add(b::BasisFuncs{𝑙, GN, 1})::BasisFunc{𝑙, GN} where {𝑙, GN}
    BasisFunc(b.center, b.gauss, b.ijk, b.normalizeGTO)
end

add(b::BasisFunc) = itself(b)

function add(bf1::BasisFunc{𝑙, GN1}, 
             bf2::BasisFunc{𝑙, GN2})::CompositeGTBasisFuncs{<:Any, 1} where {𝑙, GN1, GN2}
    if bf1.ijk == bf2.ijk && 
       bf1.normalizeGTO == bf2.normalizeGTO && 
       (c = centerCoordOf(bf1)) == centerCoordOf(bf2)

        cen1 = bf1.center
        cen2 = bf2.center
        if cen1 === cen2 || hasIdentical(cen1, cen2)
            cen = bf1.center
        elseif hasEqual(bf1, bf2)
            cen = deepcopy(bf1.center)
        else
            cen = makeCenter(c)
        end
        gfs = vcat(bf1.gauss |> collect, bf2.gauss |> collect)
        gfsN = mergeGaussFuncs(gfs...) |> Tuple
        BasisFunc(cen, gfsN, bf1.ijk, bf1.normalizeGTO)
    else
        BasisFuncMix([bf1, bf2] |> sortBasisFuncs)
    end
end

function add(bf1::BasisFunc{<:Any, GN1}, 
             bf2::BasisFunc{<:Any, GN2})::BasisFuncMix{2, GN1+GN2} where {GN1, GN2}
    BasisFuncMix([bf1, bf2] |> sortBasisFuncs)
end

add(bfm::BasisFuncMix{BN, GN}) where {BN, GN} = unpackBasisFuncs(bfm) |> sumOf

add(bf1::BasisFuncMix{1, GN1}, bf2::BasisFunc{𝑙, GN2}) where {𝑙, GN1, GN2} = 
add(bf1.BasisFunc[1], bf2)

add(bf1::BasisFunc{𝑙, GN1}, bf2::BasisFuncMix{1, GN2}) where {𝑙, GN1, GN2} = 
add(bf2, bf1)

add(bf::BasisFunc, bfm::BasisFuncMix{BN}) where {BN} = 
vcat(bf, bfm.BasisFunc |> collect) |> sumOf

add(bfm::BasisFuncMix{BN}, bf::BasisFunc) where {BN} = add(bf, bfm)

add(bf1::BasisFuncMix{1, GN1}, bf2::BasisFuncMix{1, GN2}) where {GN1, GN2} = 
add(bf1.BasisFunc[1], bf2.BasisFunc[1])

add(bf::BasisFuncMix{1}, bfm::BasisFuncMix{BN}) where {BN} = add(bf.BasisFunc[1], bfm)

add(bfm::BasisFuncMix{BN}, bf::BasisFuncMix{1}) where {BN} = add(bf, bfm)

add(bfm1::BasisFuncMix{BN1}, bfm2::BasisFuncMix{BN2}) where {BN1, BN2} = 
vcat(bfm1.BasisFunc |> collect, bfm2.BasisFunc |> collect) |> sumOf

add(bf1::BasisFuncs{𝑙1, GN1, 1}, bf2::BasisFuncs{𝑙2, GN2, 1}) where {𝑙1, 𝑙2, GN1, GN2} = 
[[bf1, bf2] .|> add |> sumOf]


const Doc_mul_Eg1 = "GaussFunc(xpn=ParamBox{Float64, :α, :itself}(3.0)[∂][α], " * 
                              "con=ParamBox{Float64, :d, :itself}(1.0)[∂][d])"

const Doc_mul_Eg2 = "GaussFunc(xpn=ParamBox{Float64, :α, :itself}(3.0)[∂][α], " * 
                              "con=ParamBox{Float64, :d, :itself}(2.0)[∂][d])"

const Doc_mul_Eg3 = "GaussFunc(xpn=ParamBox{Float64, :α, :itself}(6.0)[∂][α], " * 
                              "con=ParamBox{Float64, :d, :itself}(1.0)[∂][d])"

const Doc_mul_Eg4 = "GaussFunc(xpn=ParamBox{Float64, :α, :itself}(6.0)[∂][α], " * 
                              "con=ParamBox{Float64, :d, :itself}(2.0)[∂][d])"

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
function mul(gf::GaussFunc, coeff::Real)::GaussFunc
    c = convert(Float64, coeff)::Float64
    con, mapFunction, dataName = mulCore(c, gf.con)
    conNew = genContraction(con, mapFunction; dataName, canDiff=gf.con.canDiff[])
    GaussFunc(gf.xpn, conNew)
end

function mulCore(c::Float64, con::ParamBox{<:Any, <:Any, :itself})
    conNew = fill(con.data[] * c)
    mapFunction = itself
    dataName = :undef
    conNew, mapFunction, dataName
end

function mulCore(c::Float64, con::ParamBox{<:Any, <:Any, F}) where {F}
    conNew = con.data
    mapFunction = Pf(c, Val(F))
    conNew, mapFunction, con.dataName
end

mul(coeff::Real, gf::GaussFunc) = mul(gf, coeff)

function mul(gf1::GaussFunc, gf2::GaussFunc)::GaussFunc
    GaussFunc(genExponent(gf1.xpn()+gf2.xpn()), genContraction(gf1.con()*gf2.con()))
end

"""

    mul(sgf1::BasisFunc{𝑙1, 1}, sgf2::BasisFunc{𝑙2, 1}; 
             normalizeGTO::Union{Bool, Missing}=missing)::BasisFunc{𝑙1+𝑙2, 1} where {𝑙1, 𝑙2}

    mul(a1::Real, a2::CompositeGTBasisFuncs{<:Any, 1}; 
        normalizeGTO::Union{Bool, Missing}=missing) -> 
    CompositeGTBasisFuncs{<:Any, 1}

    mul(a1::CompositeGTBasisFuncs{<:Any, 1}, a2::CompositeGTBasisFuncs{<:Any, 1}; 
        normalizeGTO::Union{Bool, Missing}=missing) -> 
    CompositeGTBasisFuncs{<:Any, 1}

Multiplication between `CompositeGTBasisFuncs{<:Any, 1}`s such as `BasisFunc` and 
`$(BasisFuncMix)`, or contraction coefficient multiplication between a `Real` number 
and a `CompositeGTBasisFuncs{<:Any, 1}`. If `normalizeGTO` is set to `missing` (in 
default), The `GaussFunc` in the output result will be normalized only if all the input 
bases have `normalizeGTO = true`. The function can be called using `*` syntax.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> bf1 = genBasisFunc([1,1,1], ([2,1], [0.1, 0.2]))
BasisFunc{0, 2}(gauss, subshell, center)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bf2 = bf1 * 2
BasisFunc{0, 2}(gauss, subshell, center)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> getindex.(getfield.(bf2.gauss, :con))
(0.2, 0.4)

julia> bf3 = bf1 * bf2
BasisFunc{0, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]
```
"""
function mul(sgf1::BasisFunc{𝑙1, 1}, sgf2::BasisFunc{𝑙2, 1}; 
             normalizeGTO::Union{Bool, Missing}=missing) where {𝑙1, 𝑙2}
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
        BasisFunc(makeCenter(R₁), GaussFunc(genExponent(xpn), genContraction(con)), 
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
        pbR = makeCenter(cen)
        pbα = genExponent(xpn)
        BasisFuncMix([BasisFunc(pbR, GaussFunc(pbα, genContraction(con*XYZcs[i])), 
                                XYZTuple(i.I .- 1), normalizeGTO) 
                      for i in CartesianIndices(XYZcs)] |> sortBasisFuncs)
    end
end

function mul(sgf1::BasisFunc{0, 1}, sgf2::BasisFunc{0, 1}; 
             normalizeGTO::Union{Bool, Missing}=missing)::BasisFunc{0, 1}
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
    BasisFunc(makeCenter(cen), GaussFunc(genExponent(xpn), genContraction(con)), 
              (XYZTuple(0,0,0),), normalizeGTO)
end

function gaussProd((α₁, d₁, R₁)::T, (α₂, d₂, R₂)::T) where 
                  {T<:Tuple{Number, Number, Array{<:Number}}}
    α = α₁ + α₂
    d = d₁ * d₂ * exp(-α₁ * α₂ / α * sum(abs2, R₁-R₂))
    R = (α₁*R₁ + α₂*R₂) / α
    (α, d, R)
end

function mul(bf::BasisFunc{𝑙, GN}, coeff::Real; 
             normalizeGTO::Union{Bool, Missing}=missing)::BasisFunc{𝑙, GN} where {𝑙, GN}
    n = bf.normalizeGTO
    normalizeGTO isa Missing && (normalizeGTO = n)
    c = (n && !normalizeGTO) ? (coeff .* getNorms(bf)) : coeff
    gfs = mul.(bf.gauss, c)
    BasisFunc(bf.center, gfs, bf.ijk, normalizeGTO)
end

function mul(coeff::Real, bf::BasisFunc{𝑙, GN}; 
             normalizeGTO::Union{Bool, Missing}=missing)::BasisFunc{𝑙, GN} where {𝑙, GN}
    mul(bf, coeff; normalizeGTO)
end

function mul(bfm::BasisFuncMix{BN, GN}, coeff::Real; 
             normalizeGTO::Union{Bool, Missing}=missing)::BasisFuncMix{BN, GN} where 
            {BN, GN}
    BasisFuncMix(mul.(bfm.BasisFunc, coeff; normalizeGTO) |> collect)
end

function mul(coeff::Real, bfm::BasisFuncMix{BN, GN}; 
             normalizeGTO::Union{Bool, Missing}=missing)::BasisFuncMix{BN, GN} where 
            {BN, GN}
    mul(bfm, coeff; normalizeGTO)
end

function mul(bf1::BasisFunc{𝑙1, GN1}, bf2::BasisFunc{𝑙2, GN2}; 
             normalizeGTO::Union{Bool, 
                                 Missing}=missing)::CompositeGTBasisFuncs{<:Any, 1} where 
            {𝑙1, 𝑙2, GN1, GN2}
    cen1 = bf1.center
    ijk1 = bf1.ijk
    cen2 = bf2.center
    ijk2 = bf2.ijk
    bf1n = bf1.normalizeGTO
    bf2n = bf2.normalizeGTO
    normalizeGTO isa Missing && (normalizeGTO = bf1n * bf2n)
    bs = CompositeGTBasisFuncs{<:Any, 1}[]
    for gf1 in bf1.gauss, gf2 in bf2.gauss
        push!(bs, mul(BasisFunc(cen1, (gf1,), ijk1, bf1n), 
                      BasisFunc(cen2, (gf2,), ijk2, bf2n); 
                      normalizeGTO))
    end
    sumOf(bs)
end

mul(bf1::BasisFuncMix{1, GN1}, bf2::BasisFunc{<:Any, GN2}; 
    normalizeGTO::Union{Bool, Missing}=missing) where {GN1, GN2} = 
mul(bf1.BasisFunc[1], bf2; normalizeGTO)

mul(bf1::BasisFunc{<:Any, GN1}, bf2::BasisFuncMix{1, GN2}; 
normalizeGTO::Union{Bool, Missing}=missing) where {GN1, GN2} = 
mul(bf2, bf1; normalizeGTO)

mul(bf::BasisFunc{<:Any, GN1}, bfm::BasisFuncMix{BN, GN2}; 
    normalizeGTO::Union{Bool, Missing}=missing) where {BN, GN1, GN2} = 
mul.(Ref(bf), bfm.BasisFunc; normalizeGTO) |> collect |> sumOf

mul(bfm::BasisFuncMix{BN, GN1}, bf::BasisFunc{<:Any, GN2}; 
    normalizeGTO::Union{Bool, Missing}=missing) where {BN, GN1, GN2} = 
mul(bf, bfm; normalizeGTO)

mul(bf1::BasisFuncMix{1, GN1}, bf2::BasisFuncMix{1, GN2}; 
    normalizeGTO::Union{Bool, Missing}=missing) where {GN1, GN2} = 
mul(bf1.BasisFunc[1], bf2.BasisFunc[1]; normalizeGTO)

mul(bf::BasisFuncMix{1, GN1}, bfm::BasisFuncMix{BN, GN2}; 
    normalizeGTO::Union{Bool, Missing}=missing) where {BN, GN1, GN2} = 
mul(bf.BasisFunc[1], bfm; normalizeGTO)

mul(bfm::BasisFuncMix{BN, GN1}, bf::BasisFuncMix{1, GN2}; 
    normalizeGTO::Union{Bool, Missing}=missing) where {BN, GN1, GN2} = 
mul(bf, bfm; normalizeGTO)

function mul(bfm1::BasisFuncMix{BN1, GN1}, bfm2::BasisFuncMix{BN2, GN2}; 
             normalizeGTO::Union{Bool, Missing}=missing) where {BN1, BN2, GN1, GN2}
    bfms = CompositeGTBasisFuncs{<:Any, 1}[]
    for bf1 in bfm1.BasisFunc, bf2 in bfm2.BasisFunc
        push!(bfms, mul(bf1, bf2; normalizeGTO))
    end
    sumOf(bfms)
end

mul(bf1::BasisFuncs{𝑙1, GN1, 1}, bf2::BasisFuncs{𝑙2, GN2, 1}; 
    normalizeGTO::Union{Bool, Missing}=missing) where {𝑙1, 𝑙2, GN1, GN2} = 
[mul(add(bf1), add(bf2); normalizeGTO)]


"""

    shift(bf::FloatingGTBasisFuncs{𝑙, GN, 1}, 
          didjdk::Union{Vector{<:Real}, NTuple{3, Int}}) where {𝑙, GN} -> 
    BasisFunc

Shift (add) the angular momentum (Cartesian representation) given the a vector that 
specifies the change of each pseudo-quantum number 𝑑i, 𝑑j, 𝑑k.
"""
shift(bf::FloatingGTBasisFuncs{𝑙, GN, 1}, didjdk::Vector{<:Real}) where {𝑙, GN} = 
shiftCore(bf, XYZTuple(didjdk.|>Int))

shift(bf::FloatingGTBasisFuncs{𝑙, GN, 1}, didjdk::NTuple{3, Int}) where {𝑙, GN} = 
shiftCore(bf, XYZTuple(didjdk))

shiftCore(bf::FloatingGTBasisFuncs{𝑙1, GN, 1}, didjdk::XYZTuple{𝑙2}) where {𝑙1, 𝑙2, GN} = 
BasisFunc(bf.center, bf.gauss, bf.ijk[1]+didjdk, bf.normalizeGTO)


"""

    decompose(bf::CompositeGTBasisFuncs; splitGaussFunc::Bool=false) -> 
    Array{<:FloatingGTBasisFuncs, 2}

Decompose a `FloatingGTBasisFuncs` into an `Array` of `BasisFunc`s. Each column represents 
one orbital of the input basis function(s). If `splitGaussFunc` is `true`, then each column 
consists of the `BasisFunc`s each with only 1 `GaussFunc`.
"""
function decompose(bf::FloatingGTBasisFuncs{𝑙, GN, ON}; 
                   splitGaussFunc::Bool=false) where {𝑙, GN, ON}
    if splitGaussFunc
        nRow = GN
        nG = 1
        gs = bf.gauss
    else
        nRow = 1
        nG = GN
        gs = Ref(bf.gauss)
    end
    res = Array{BasisFunc{𝑙, nG}, 2}(undef, nRow, ON)
    for (c, ijk) in zip(eachcol(res), bf.ijk)
        c .= BasisFunc.(Ref(bf.center), gs, Ref(ijk), bf.normalizeGTO)
    end
    res
end

function decompose(bfm::BasisFuncMix; splitGaussFunc::Bool=false)
    if splitGaussFunc
        bfs = decompose.(bfm.BasisFunc; splitGaussFunc)
        vcat(bfs...)
    else
        bfm
    end
end


"""

    basisSize(subshell::Union{String, Array{String, 1}}) -> Int

Return the size (number of orbitals) of each subshell.
"""
@inline basisSize(subshell::String) = SubshellDimList[subshell]

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
function genBasisFuncText(bf::FloatingGTBasisFuncs; 
                          norm::Float64=1.0, printCenter::Bool=true)
    gauss = bf.gauss |> collect
    GFs = map(x -> genGaussFuncText(x.xpn(), x.con()), gauss)
    cen = centerCoordOf(bf)
    firstLine = printCenter ? "X "*(alignNum.(cen) |> join)*"\n" : ""
    firstLine * "$(bf.subshell)    $(bf.gauss |> length)   $(norm)\n" * (GFs |> join)
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
    bfBlocks = sortBasisFuncs(bs; groupCenters)
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
                                    Tuple{N, ParamBox}, 
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
                           center::Union{AbstractArray, 
                                         NTuple{N, ParamBox}, 
                                         Missing}=missing, 
                           unlinkCenter::Bool=false) where {N, F<:Function}
    adjustContent && (content = adjustFunction(content))
    lines = split.(content |> IOBuffer |> readlines)
    lines = lines[1+excludeFirstNlines : end-excludeLastNlines]
    data = [advancedParse.(i) for i in lines]
    index = findall(x -> typeof(x) != Vector{Float64} && length(x)==3, data)
    bfs = []
    for i in index
        gs1 = GaussFunc[]
        ng = data[i][2] |> Int
        centerOld = center
        if center isa Missing && i != 1 && data[i-1][1] == "X"
            center = data[i-1][2:end]
        end
        if data[i][1] == "SP"
            gs2 = GaussFunc[]
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

    assignCenter!(center::AbstractArray, b::FloatingGTBasisFuncs) -> NTuple{3, ParamBox}

Assign a new coordinate to the center of the input `FloatingGTBasisFuncs`. Also return the 
altered center.
"""
function assignCenter!(center::AbstractArray, b::FloatingGTBasisFuncs)
    for (i,j) in zip(b.center, center)
        i[] = j
    end
    b.center
end


"""

    makeCenter(coord::Array{<:Real, 1}; roundDigits::Int=-1) -> NTuple{3, ParamBox}

Generate a `Tuple` of coordinate `ParamBox`s for a basis function center coordinate given a 
`Vector`. If `roundDigits < 0` then there won't be rounding for input data.
"""
function makeCenter(coord::Vector{<:Real}; roundDigits::Int=-1)
    c = roundDigits<0 ? convert(Vector{Float64}, coord) : round.(coord, digits=roundDigits)
    x = ParamBox(c[1], XParamSym)
    y = ParamBox(c[2], YParamSym)
    z = ParamBox(c[3], ZParamSym)
    (x,y,z)
end


"""

    getParams(pbc::ParamBox, symbol::Union{Symbol, Nothing}=nothing; 
              onlyDifferentiable::Bool=false) -> 
    Union{ParamBox, Nothing}

    getParams(pbc::StructSpatialBasis, symbol::Union{Symbol, Nothing}=nothing; 
              onlyDifferentiable::Bool=false) -> 
    Array{<:ParamBox, 1}

    getParams(pbc::Array, symbol::Union{Symbol, Nothing}=nothing; 
              onlyDifferentiable::Bool=false) -> 
    Array{<:ParamBox, 1}

Return the parameter(s) stored in the input container. If keyword argument `symbol` is set 
to `nothing`, then return all the different parameters; if it's set to the `Symbol` of a 
parameter (e.g. the symbol of `ParamBox{T, V}` would be `V`), return only that type of 
parameters (which might still have different indices). `onlyDifferentiable` determines 
whether ignore non-differentiable parameters. If the 1st argument is an `Array`, the 
entries must be `ParamBox` containers.
"""
function getParams(pb::ParamBox, symbol::Union{Symbol, Nothing}=nothing; 
          onlyDifferentiable::Bool=false)
    paramFilter(pb, symbol, onlyDifferentiable) ? pb : nothing
end

function getParams(ssb::StructSpatialBasis, symbol::Union{Symbol, Nothing}=nothing; 
          onlyDifferentiable::Bool=false)
    filter(x->paramFilter(x, symbol, onlyDifferentiable), ssb.param) |> collect
end

function getParams(cs::Array{<:ParamBox}, symbol::Union{Symbol, Nothing}=nothing; 
          onlyDifferentiable::Bool=false)
    idx = findall(x->paramFilter(x, symbol, onlyDifferentiable), cs)
    [cs[i] for i in idx]
end

getParams(cs::Array{<:StructSpatialBasis}, symbol::Union{Symbol, Nothing}=nothing; 
          onlyDifferentiable::Bool=false) = 
vcat(getParams.(cs, symbol; onlyDifferentiable)...)

function getParams(cs::Array, symbol::Union{Symbol, Nothing}=nothing; 
                   onlyDifferentiable::Bool=false)
    pbIdx = findall(x->x isa ParamBox, cs)
    vcat(getParams(convert(Vector{ParamBox}, cs[pbIdx]), symbol; onlyDifferentiable), 
         getParams(convert(Vector{StructSpatialBasis}, cs[1:end .∉ [pbIdx]]), symbol; 
                   onlyDifferentiable))
end

function paramFilter(pb::ParamBox, outSym::Union{Symbol, Nothing}=nothing, 
                     canDiff::Bool=false)
    (outSym === nothing || outSymOfCore(pb) == outSym) && 
    (!canDiff || pb.canDiff[])
end


const Doc_copyBasis_Eg1 = "GaussFunc(xpn=ParamBox{Float64, :α, :itself}(9.0)[∂][α], " * 
                                    "con=ParamBox{Float64, :d, :itself}(2.0)[∂][d])"

"""

    copyBasis(b::GaussFunc, copyOutVal::Bool=true) -> GaussFunc

    copyBasis(b::CompositeGTBasisFuncs, copyOutVal::Bool=true) -> CompositeGTBasisFuncs

Return a copy of the input basis. If `copyOutVal` is set to `true`, then only the value(s) 
of mapped data will be copied, i.e., `outValCopy` is used to copy the `ParamBox`s, 
otherwise `inVarCopy` is used.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> e = genExponent(3.0, x->x^2)
ParamBox{Float64, :α, :f_α₁}(3.0)[∂][x_α]

julia> c = genContraction(2.0)
ParamBox{Float64, :d, :itself}(2.0)[∂][d]

julia> gf1 = GaussFunc(e, c);

julia> gf2 = copyBasis(gf1)
$(Doc_copyBasis_Eg1)

julia> gf1.xpn() == gf2.xpn()
true

julia> (gf1.xpn[] |> gf1.xpn.map) == gf2.xpn[]
true
```
"""
function copyBasis(g::GaussFunc, copyOutVal::Bool=true)::GaussFunc
    pbs = g.param .|> (copyOutVal ? outValCopy : inVarCopy)
    GaussFunc(pbs...)
end

function copyBasis(bfs::FloatingGTBasisFuncs{𝑙, GN, ON}, 
                   copyOutVal::Bool=true)::FloatingGTBasisFuncs{𝑙, GN, ON} where {𝑙, GN, ON}
    cen = bfs.center .|> (copyOutVal ? outValCopy : inVarCopy)
    gs = copyBasis.(bfs.gauss, copyOutVal)
    genBasisFunc(cen, gs, bfs.ijk; normalizeGTO=bfs.normalizeGTO)
end

function copyBasis(bfm::BasisFuncMix{BN, GN}, 
                   copyOutVal::Bool=true)::BasisFuncMix{BN, GN} where {BN, GN}
    bfs = copyBasis.(bfm.BasisFunc, copyOutVal)
    BasisFuncMix(bfs |> collect)
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


function markParams!(parArray::Array{<:ParamBox{<:Any, V}}; 
                     filterMapping::Bool=false) where {V} # ignoreMapping
    res, _ = markUnique(parArray, compareFunction=compareParamBox)
    for (idx, i) in zip(parArray, res)
        idx.index[] = i
    end
    filterMapping ? unique(x->x.index[], parArray) : parArray
end

function markParams!(parArray::Array{<:ParamBox}; 
                     filterMapping::Bool=false)
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
        append!(pars, markParams!(subArr; filterMapping))
    end
    pars
end


"""

    uniqueParams!(bs; filterMapping::Bool=false) -> Array{<:ParamBox, 1}

Mark the parameters (`ParamBox`) in input bs which can a `Vector` of `GaussFunc` or 
`FloatingGTBasisFuncs`. The identical parameters will be marked with same index. 
`filterMapping`determines weather filter out (i.e. not return) `ParamBox`s that have same 
independent variables despite they may have same mapping functions.
"""
uniqueParams!(bs; filterMapping::Bool=false) = markParams!(getParams(bs); filterMapping)


"""

    getVarCore(pb::ParamBox, expandNonDifferentiable::Bool=false) -> Array{Pair, 1}

Core function of `getVar`, which returns the mapping relations inside the parameter 
container. `expandNonDifferentiable` determines whether expanding the mapping relations of 
non-differentiable variable (parameters).
"""
function getVarCore(pb::ParamBox, expandNonDifferentiable::Bool=false)
    vNum = outSymOf(pb)
    f = pb.map
    if pb.canDiff[] || expandNonDifferentiable
        ivNum = inSymOf(pb)
        expr = f(ivNum)
        res = Pair{Symbolics.Num, Real}[ivNum => pb.data[]]
        fNum = getFuncNum(f, ivNum)
        # fNum = Symbolics.variable(f|>nameOf, T=Symbolics.FnType{Tuple{Any}, Real})(ivNum)
        pushfirst!(res, fNum=>expr, expr=>pb())
        !(pb.canDiff[]) && pushfirst(res, vNum=>fNum)
        res |> unique!
    else
        res = Pair{Symbolics.Num, Real}[vNum => pb()]
    end
    res
end

getVarCore(pb::ParamBox{T, V, :itself}, _::Bool=false) where {T, V} = [inSymValOf(pb)]

"""

    getVar(pb::ParamBox) -> Symbolics.Num

    getVar(container::StructSpatialBasis) -> Array{Symbolics.Num, 1}

Return the independent variable(s) of the input parameter container.
"""
getVar(pb::ParamBox) = getVarCore(pb, false)[end][1]

function getVar(container::CompositeGTBasisFuncs)
    vrs = getVarCore.(container |> getParams, false)
    getindex.(getindex.(vrs, lastindex.(vrs)), 1)
end


getVarDictCore(pb::ParamBox, expandNonDifferentiable::Bool=false) = 
getVarCore(pb, expandNonDifferentiable) |> Dict

getVarDictCore(containers, expandNonDifferentiable::Bool=false) = 
vcat(getVarCore.(containers|>getParams, expandNonDifferentiable)...) |> Dict

"""

    getVarDict(obj::Union{ParamBox, StructSpatialBasis, Array}; 
               includeMapping::Bool=false) -> 
    Dict{Symbolics.Num, <:Number}

Return a `Dict` that stores the independent variable(s) of the parameter container(s) and 
its(their) corresponding value(s). If `includeMapping = true`, then the dictionary will 
also include the mapping relations between the mapped variables and the independent 
variables.
"""
getVarDict(pb::ParamBox; includeMapping::Bool=false) = 
includeMapping ? getVarDictCore(pb, true) : (inSymValOf(pb) |> Dict)

function getVarDict(containers::Union{Array, StructSpatialBasis}; 
                    includeMapping::Bool=false)
    if includeMapping
        getVarDictCore(containers, true)
    else
        pbs = getParams(containers)
        inSymValOf.(pbs) |> Dict
    end
end


#########################################################################
# Old normalization functions for libcint integral functions.
function Nlα(l, α)
    if l < 2
        ( 2^(2l+3) * factorial(l+1) * 2^(l+1.5) / 
        (factorial(2l+2) * √π) )^0.5 * α^(0.5l + 0.75)
    else
        # for higher angular momentum make the upper bound of norms be 1.
        ( 2^(3l+1.5) * factorial(l) / (factorial(2l) * π^1.5) )^0.5 * α^(0.5l + 0.75)
    end
end

Nlα(subshell::String, α) = Nlα(AngularMomentumList[subshell], α)


Nijk(i, j, k) = (2/π)^0.75 * ( 2^(3*(i+j+k)) * factorial(i) * factorial(j) * factorial(k) / 
                (factorial(2i) * factorial(2j) * factorial(2k)) )^0.5


function Nijkα(i, j, k, α)
    l = i + j + k
    if l < 2
        ( 2^(2l+3) * factorial(l+1) * 2^(l+1.5) / 
        (factorial(2l+2) * √π) )^0.5 * α^(0.5l + 0.75)
    else
        # for higher angular momentum make the upper bound of norms be 1.
        Nijk(i, j, k) * α^(0.5l + 0.75)
    end
end

normOfGTOin(b::FloatingGTBasisFuncs{𝑙, GN, 1})  where {𝑙, GN} = 
Nijkα.(b.ijk[1]..., [g.xpn() for g in b.gauss])

normOfGTOin(b::FloatingGTBasisFuncs{𝑙, GN, ON}) where {𝑙, GN, ON} = 
Nlα.(b.subshell, [g.xpn() for g in b.gauss])

#########################################################################


getNijk(i, j, k) = (2/π)^0.75 * 
                   ( 2^(3*(i+j+k)) * factorial(i) * factorial(j) * factorial(k) / 
                     (factorial(2i) * factorial(2j) * factorial(2k)) )^0.5

getNα(i, j, k, α) = α^(0.5*(i + j + k) + 0.75)

getNijkα(i, j, k, α) = getNijk(i, j, k) * getNα(i, j, k, α)

getNijkα(ijk, α) = getNijkα(ijk[1], ijk[2], ijk[3], α)

getNorms(b::FloatingGTBasisFuncs{𝑙, GN, 1})  where {𝑙, GN} = 
getNijkα.(b.ijk[1]..., [g.xpn() for g in b.gauss])

pgf0(x::T, y, z, α) where {T} = exp( -α * (x^2 + y^2 + z^2) )::T
cgf0(x::T, y, z, α, d) where {T} = (d * pgf0(x, y, z, α))::T
cgo0(x::T, y, z, α, d, i, j, k, N=1.0) where {T} = 
(N * x^i * y^j * z^k * cgf0(x, y, z, α, d))::T


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
        getFuncNum(FunctionType{F}(), inSymOf(pb))
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
    res = map(bf.ijk) do ijk
        i, j, k = ijk
        exprs = cgo2.(Ref(x), α, d, i, j, k, N.(i,j,k,α))
        splitGaussFunc ? collect(exprs) : sum(exprs)
    end
    hcat(res...)
end

function expressionOfCore(bfm::BasisFuncMix{BN}, substituteValue::Bool=false, 
                          onlyParameter::Bool=false, splitGaussFunc::Bool=false) where {BN}
    exprs = Matrix{Symbolics.Num}[expressionOfCore(bf, substituteValue, 
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

    expressionOf(bf::CompositeGTBasisFuncs; splitGaussFunc::Bool=false) -> 
    Array{<:Symbolics.Num, 2}

Return the expression(s) of a given `CompositeGTBasisFuncs` (e.g. `BasisFuncMix` or 
`FloatingGTBasisFuncs`) as a `Matrix{<:Symbolics.Num}`of which the column(s) corresponds to 
different orbitals. If `splitGaussFunc` is `true`, the column(s) will be expanded 
vertically such that the entries are `GaussFunc` inside the corresponding orbital.
"""
expressionOf(bf::CompositeGTBasisFuncs; splitGaussFunc::Bool=false) = 
expressionOfCore(bf, true, false, splitGaussFunc)

"""

    expressionOf(gf::GaussFunc) -> Symbolics.Num

Return the expression of a given `GaussFunc`.
"""
expressionOf(gf::GaussFunc) = expressionOfCore(gf, true)


function inSymbols(sym::Symbol, pool::Vector{Symbol}=ParamNames)
    symString = sym |> string
    for i in pool
         occursin(i |> string, symString) && (return i)
    end
    return false
end

inSymbols(vr::SymbolicUtils.Sym, pool::Vector{Symbol}=ParamNames) = 
inSymbols(Symbolics.tosymbol(vr), pool)

inSymbols(vr::SymbolicUtils.Term, pool::Vector{Symbol}=ParamNames) = 
inSymbols(Symbolics.tosymbol(vr.f), pool)

inSymbols(::Function, args...) = false

inSymbols(vr::Num, pool::Vector{Symbol}=ParamNames) = inSymbols(vr.val, pool)


function varVal(vr::SymbolicUtils.Sym, varDict::Dict{Num, <:Real})
    res = recursivelyGet(varDict, vr |> Num)
    if isnan(res)
        res = recursivelyGet(varDict, 
                             symbolReplace(Symbolics.tosymbol(vr), 
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


function detectXYZ(i::SymbolicUtils.Symbolic)
    vec = zeros(3)
    if i isa SymbolicUtils.Pow
        for j = 1:3
            if inSymbols(i.base, [ParamNames[j]]) != false
                vec[j] = i.exp
                sign = iseven(i.exp) ? 1 : -1
                return (true, sign, vec) # (-X)^k -> (true, (-1)^k, [0, 0, 0])
            end
        end
    else
        for j = 1:3
            if inSymbols(i, [ParamNames[j]]) != false
                vec[j] = 1
                return (true, -1, vec)
            end
        end
    end
    (false, 1, nothing)
end

detectXYZ(::Real) = (false, 1, nothing)


# res = [d_ratio, Δi, Δj, Δk]
function diffTransferCore(term::SymbolicUtils.Symbolic, varDict::Dict{Num, <:Real})
    res = Real[1,0,0,0]
    r = Symbolics.@rule *(~~xs) => ~~xs
    terms = SymbolicUtils.simplify(term, rewriter=r)
    !(terms isa SubArray) && (terms = [terms])
    for vr in terms
        isXYZ, sign, xpns = detectXYZ(vr)
        if isXYZ
            res[2:end] += xpns
            res[1] *= sign
        else
            res[1] *= varVal(vr, varDict)
        end
    end
    res
end

# diffTransferCore(term::Symbolics.Num, args...) = diffTransferCore(term.val, args...)

diffTransferCore(term::Real, _...) = Real[Float64(term), 0,0,0]


function diffTransfer(term::Num, varDict::Dict{Num, <:Real})
    terms = splitTerm(term)
    diffTransferCore.(terms, Ref(varDict))
end


function diffInfo(bf::CompositeGTBasisFuncs, vr, varDict)
# function diffInfo(bf::CompositeGTBasisFuncs{BN, 1}, vr, varDict) where {BN}
    exprs = expressionOfCore(bf, false, true, true)
    relDiffs = Symbolics.derivative.(log.(exprs), vr)
    diffTransfer.(relDiffs, Ref(varDict))
end


function diffInfoToBasisFunc(bf::FloatingGTBasisFuncs, info::Matrix{<:Any})
    bs = decompose(bf, splitGaussFunc=true)
    mat = map(bs, info) do x, y
        xs = [copyBasis(x) for _ = 1:length(y)]

        for (i,j) in zip(y, xs)
            j.gauss[1].con[] *= getindex(i, 1)
        end

        shift.(xs, getindex.(y, Ref(2:4))) |> BasisFuncMix
    end
    eachcol(mat) .|> sum
end

# function diffInfoToBasisFunc(bf::FloatingGTBasisFuncs, info::Matrix{<:Any})
#     mat = map(info) do y
#         xs = [copyBasis(x) for _ = 1:length(y)]
#         d = genContraction
#         alpha
#         genBasisFunc(bf.center, ())
#         for (i,j) in zip(y, xs)
#             j.gauss[1].con[] *= getindex(i, 1)
#         end

#         shift.(xs, getindex.(y, Ref(2:4))) |> BasisFuncMix
#     end
#     eachcol(mat) .|> sum
# end