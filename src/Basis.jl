export GaussFunc, Exponent, Contraction, BasisFunc, BasisFuncs, genBasisFunc, centerOf, 
       GTBasis, add, mul, shift, decompose, basisSize, genBasisFuncText, 
       genBFuncsFromText, assignCenter!, makeCenter, getParams, dataCopy, uniqueParams!, 
       getVar, getVars, expressionOf

using Symbolics
using SymbolicUtils

"""

    GaussFunc <: AbstractGaussFunc

A single contracted gaussian function `struct` from package Quiqbox.

≡≡≡ Field(s) ≡≡≡

`xpn::ParamBox{:𝛼, Float64}`：Exponent of the gaussian function.

`con::ParamBox{:𝑑, Float64}`: Contraction coefficient of the gaussian function.

`param::NTuple{2, ParamBox}`: A Tuple that stores the `ParamBox`s of `xpn` and `con`.

≡≡≡ Initialization Method(s) ≡≡≡

    GaussFunc(xpn::ParamBox{:$(ParamList[:xpn]), Float64}, 
              con::ParamBox{:$(ParamList[:con]), Float64}) -> 
    GaussFunc

    GaussFunc(xpn::Real, con::Real) -> GaussFunc

"""
struct GaussFunc <: AbstractGaussFunc
    xpn::ParamBox{ParamList[:xpn], Float64}
    con::ParamBox{ParamList[:con], Float64}
    param::Tuple{ParamBox{ParamList[:xpn], Float64}, ParamBox{ParamList[:con], Float64}}

    GaussFunc(xpn::ParamBox{ParamList[:xpn], Float64}, 
              con::ParamBox{ParamList[:con], Float64}) = 
    new(xpn, con, (xpn, con))
end

function GaussFunc(e::Real, c::Real)
    xpn = ParamBox(e, ParamList[:xpn])
    con = ParamBox(c, ParamList[:con])
    GaussFunc(xpn, con)
end


"""

    Exponent(e::Real; mapFunction::Function=itself, canDiff::Bool=true, 
             roundDigits::Int=15) -> 
    ParamBox{:α, Float64}

Construct a `ParamBox` for an exponent coefficient given a value. Keywords `mapFunction` 
and `canDiff` work the same way as in a general constructor of a `ParamBox`. If 
`roundDigits < 0` then there won't be rounding for input data.
"""
function Exponent(e::Real; mapFunction::F=itself, canDiff::Bool=true, 
                  roundDigits::Int=15) where {F<:Function}
    n = roundDigits < 0 ? Float64(e) : round(e, digits=roundDigits)
    ParamBox(n, ParamList[:xpn]; mapFunction, canDiff)
end


"""

    Contraction(c::Real; mapFunction::Function=itself, canDiff::Bool=true, 
                roundDigits::Int=15) -> 
    ParamBox{:d, Float64}

Construct a `ParamBox` for an contraction coefficient given a value. Keywords `mapFunction` 
and `canDiff` work the same way as in a general constructor of a `ParamBox`. If 
`roundDigits < 0` then there won't be rounding for input data.
"""
function Contraction(c::Real; mapFunction::F=itself, canDiff::Bool=true, 
                     roundDigits::Int=15) where {F<:Function}
    n = roundDigits < 0 ? Float64(c) : round(c, digits=roundDigits)
    ParamBox(n, ParamList[:con]; mapFunction, canDiff)
end


"""

    BasisFunc{S, GN} <: FloatingGTBasisFuncs{S, GN, 1}

A (floating) basis function with the center attached to it instead of any nucleus.

≡≡≡ Field(s) ≡≡≡

`center::NTuple{3, ParamBox}`: The center coordinate in form of a 3-element `ParamBox`-type 
`Tuple`.

`gauss::NTuple{N, GaussFunc}`: Gaussian functions within the basis function.

`subshell::String`: The subshell (angular momentum symbol).

`ijk::Tuple{String}`: Cartesian representation (pseudo-quantum number) of the angular 
momentum orientation. E.g., s would be ("X⁰Y⁰Z⁰")

`normalizeGTO::Bool`: Whether the GTO`::GaussFunc` will be normalized in calculations.

`param::Tuple{Vararg{<:ParamBox}}`： All the tunable parameters`::ParamBox` stored in the 
`BasisFunc`.

≡≡≡ Initialization Method(s) ≡≡≡

    BasisFunc(center::Tuple{Vararg{<:ParamBox}}, gauss::NTuple{GN, GaussFunc}, 
              ijk::Array{Int, 1}, normalizeGTO::Bool) where {GN} -> 
    BasisFunc{S, GN}

    BasisFunc(cen::Tuple{Vararg{<:ParamBox}}, gauss::GaussFunc, ijk::Vector{Int}, 
              normalizeGTO::Bool) ->
    BasisFunc{S, 1}
"""
struct BasisFunc{S, GN} <: FloatingGTBasisFuncs{S, GN, 1}
    center::Tuple{ParamBox{<:Any, Float64}, 
                  ParamBox{<:Any, Float64}, 
                  ParamBox{<:Any, Float64}}
    gauss::NTuple{GN, GaussFunc}
    subshell::String
    ijk::Tuple{String}
    normalizeGTO::Bool
    param::Tuple{Vararg{<:ParamBox}}

    function BasisFunc(cen::Tuple{Vararg{<:ParamBox}}, gs::NTuple{GN, GaussFunc}, 
                       ijk::Vector{Int}, normalizeGTO::Bool) where {GN}
        @assert length(ijk) == 3 "The length of `ijk` should be 3."
        subshell = SubshellNames[sum(ijk)+1]
        pars = ParamBox[]
        append!(pars, cen)
        for g in gs
            append!(pars, g.param)
        end
        new{Symbol(subshell), GN}(cen, gs, subshell, (ijkStringList[ijk],), 
                                          normalizeGTO, pars |> Tuple)
    end
end

BasisFunc(cen::Tuple{Vararg{<:ParamBox}}, g::GaussFunc, ijk::Vector{Int}, 
          normalizeGTO::Bool) = 
BasisFunc(cen, (g,), ijk, normalizeGTO)


"""

    BasisFuncs{S, GN, ON} <: FloatingGTBasisFuncs{S, GN, ON}

A group of basis functions with identical parameters except they have different subshell 
under the specified angular momentum. It has the same fields as `BasisFunc` and 
specifically, for `ijk`, instead of being a 1-element `Tuple`, the size of the `Tuple` is 
the size of the corresponding subshell.
"""
struct BasisFuncs{S, GN, ON} <: FloatingGTBasisFuncs{S, GN, ON}
    center::Tuple{ParamBox{<:Any, Float64}, 
                  ParamBox{<:Any, Float64}, 
                  ParamBox{<:Any, Float64}}
    gauss::NTuple{GN, GaussFunc}
    subshell::String
    ijk::Tuple{Vararg{String}}
    normalizeGTO::Bool
    param::Tuple{Vararg{<:ParamBox}}

    function BasisFuncs(cen::Tuple{Vararg{<:ParamBox}}, gs::NTuple{GN, GaussFunc}, 
                        ijks::Vector{Vector{Int}}, normalizeGTO::Bool=false) where {GN}
        @assert prod(length.(ijks) .== 3) "The length of each `ijk` should be 3."
        ls = sum.(ijks)
        @assert prod(ls .== ls[1]) "The total angular momentums (of each ijk) should be "*
                                   "the same."
        subshell = SubshellNames[ls[1]+1]
        ss = SubshellDimList[subshell]
        @assert length(ijks) <= ss "The total number of `ijk` should be no more than "*
                                   "$(ss) as they are in $(subshell) subshell."
        ijks = sort(ijks, rev=true)
        ijkStrs = [ijkStringList[i] for i in ijks] |> Tuple
        pars = ParamBox[]
        append!(pars, cen)
        for g in gs
            append!(pars, g.param)
        end
        new{Symbol(subshell), GN, length(ijks)}(cen, gs, subshell, ijkStrs, normalizeGTO, 
                                                pars|>Tuple)
    end
end

BasisFuncs(cen::Tuple{Vararg{<:ParamBox}}, g::GaussFunc, ijks::Vector{Vector{Int}}, 
          normalizeGTO::Bool=false) = 
BasisFuncs(cen, (g,), ijks, normalizeGTO)


"""

    genBasisFunc(args..., kws...) -> BasisFunc

    genBasisFunc(args..., kws...) -> BasisFuncs

    genBasisFunc(args..., kws...) -> collection

Constructor of `BasisFunc` and `BasisFuncs`, but it also returns different kinds of 
collections of them based on the applied methods.

≡≡≡ Method 1 ≡≡≡

    genBasisFunc(coord::AbstractArray, gs::Array{<:GaussFunc, 1}, 
                 ijkOrijks::Union{Array{Int, 1}, Array{Array{Int, 1}, 1}}; 
                 normalizeGTO::Bool=false)

`ijkOrijks` is the Array of the pseudo-quantum number(s) to specify the angular 
momentum(s). E.g., s is [0,0,0] and p is [[1,0,0], [0,1,0], [0,0,1]].

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genBasisFunc([0,0,0], GaussFunc(2,1), [0,1,0])
BasisFunc{:P, 1}(gauss, subshell, center)[X⁰Y¹Z⁰][0.0, 0.0, 0.0]
```

≡≡≡ Method 2 ≡≡≡

    genBasisFunc(coord::AbstractArray, gs::Array{<:GaussFunc, 1}, subshell::String="S"; 
                 ijkFilter::Array{Bool, 1}=fill(true, SubshellDimList[subshell]), 
                 normalizeGTO::Bool=false)

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genBasisFunc([0,0,0], GaussFunc(2,1), "S")
BasisFunc{:S, 1}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], GaussFunc(2,1), "P")
BasisFuncs{:P, 1, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]
```

≡≡≡ Method 3 ≡≡≡

    genBasisFunc(coord::AbstractArray, gExpsANDgCons::NTuple{2, Array{<:Real, 1}}, 
                 subshell="S"; kw...)

Instead of directly inputting `GaussFunc`, one can also input a 2-element `Tuple` of the 
exponent(s) and contraction coefficient(s) corresponding to the same `GaussFunc`(s).

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> genBasisFunc([0,0,0], (2, 1), "P")
BasisFuncs{:P, 1, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], ([2, 1.5], [1, 0.5]), "P")
BasisFuncs{:P, 2, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]
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
 BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFuncs{:P, 3, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], "STO-3G")
1-element Vector{Quiqbox.FloatingGTBasisFuncs}:
 BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], ["STO-2G", "STO-3G"])
2-element Vector{Quiqbox.FloatingGTBasisFuncs}:
 BasisFunc{:S, 2}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]

julia> genBasisFunc([0,0,0], [("STO-2G", "He"), ("STO-3G", "O")])
4-element Vector{Quiqbox.FloatingGTBasisFuncs}:
 BasisFunc{:S, 2}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
 BasisFuncs{:P, 3, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]
```
"""
genBasisFunc(cen::Tuple{Vararg{<:ParamBox}}, gs::Vector{<:GaussFunc}, 
          ijk::Vector{Int}; normalizeGTO::Bool=false) = 
BasisFunc(cen, gs|>Tuple, ijk, normalizeGTO)

genBasisFunc(cen::Tuple{Vararg{<:ParamBox}}, gs::Vector{<:GaussFunc}, 
          ijks::Vector{Vector{Int}}; normalizeGTO::Bool=false) = 
BasisFuncs(cen, gs|>Tuple, ijks, normalizeGTO)

# ijkOrijks::Union{Vector{Int}, Vector{Vector{Int}}}
function genBasisFunc(coord::AbstractArray, gs::Vector{<:GaussFunc}, ijkOrijks::Array; 
                      normalizeGTO::Bool=false)
    @assert length(coord) == 3 "The dimension of the center should be 3."
    x = ParamBox(coord[1], ParamList[:X])
    y = ParamBox(coord[2], ParamList[:Y])
    z = ParamBox(coord[3], ParamList[:Z])
    genBasisFunc((x,y,z), gs, ijkOrijks; normalizeGTO)
end

genBasisFunc(::Missing, gs::Vector{<:GaussFunc}, ijkOrijks::Array; 
             normalizeGTO::Bool=false) = 
genBasisFunc([NaN, NaN, NaN], gs, ijkOrijks; normalizeGTO)

# center::Union{AbstractArray, Tuple{Vararg{<:ParamBox}}, Missing}
function genBasisFunc(center, gs::Vector{<:GaussFunc}, subshell::String="S"; 
                      ijkFilter::Vector{Bool}=fill(true, SubshellDimList[subshell]), 
                      normalizeGTO::Bool=false)
    ijkLen = length(ijkFilter)
    subshellSize = SubshellDimList[subshell]
    @assert ijkLen == subshellSize "The length of `ijkFilter` should be $(subshellSize) "*
                                   "to match the subshell's size."
    ijks = [SubshellSuborderList[subshell][i] for i in findall(x->x==true, ijkFilter)]
    length(ijks) == 1 && (ijks = ijks[])
    genBasisFunc(center, gs, ijks; normalizeGTO)
end

# ijkOrSubshell::Union{Array, String}
function genBasisFunc(center, gExpsANDgCons::NTuple{2, Vector{<:Real}}, ijkOrSubshell="S"; 
                      kw...)
    @compareLength gExpsANDgCons[1] gExpsANDgCons[2] "exponents" "contraction coefficients"
    gs = GaussFunc.(gExpsANDgCons[1], gExpsANDgCons[2])
    genBasisFunc(center, gs, ijkOrSubshell; kw...)
end

genBasisFunc(center, gExpANDgCon::NTuple{2, Real}, ijkOrSubshell ="S"; kw...) = 
genBasisFunc(center, ([gExpANDgCon[1]], [gExpANDgCon[2]]), ijkOrSubshell; kw...)

function genBasisFunc(center, BSKeyANDnuc::Vector{Tuple{String, String}})
    bases = FloatingGTBasisFuncs[]
    for k in BSKeyANDnuc
        BFMcontent = BasisSetList[k[1]][AtomicNumberList[k[2]]]
        append!(bases, genBFuncsFromText(BFMcontent; adjustContent=true, 
                excludeLastNlines=1, center))
    end
    bases
end

genBasisFunc(center, BSKeyANDnuc::Tuple{String, String}) = 
genBasisFunc(center, [BSKeyANDnuc])

genBasisFunc(center, BSkey::Vector{String}; nucleus::String="H") = 
genBasisFunc(center, [(i, nucleus) for i in BSkey])

genBasisFunc(center, BSkey::String; nucleus::String="H") = 
genBasisFunc(center, [BSkey]; nucleus)

# A few methods for convenient arguments omissions and mutations.
genBasisFunc(arg1, g::GaussFunc, args...; kws...) = genBasisFunc(arg1, [g], args...; kws...)
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

    GTBasis(basis::Vector{<:AbstractGTBasisFuncs}, S::Matrix{<:Number}, 
            Te::Matrix{<:Number}, eeI::Array{<:Number, 4}) -> 
    GTBasis

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
        new{basisSize(basis) |> sum, typeof(basis)}(basis, S, Te, eeI, 
            (mol, nucCoords) -> nucAttractions(basis, mol, nucCoords),
            (mol, nucCoords) -> nucAttractions(basis, mol, nucCoords) + Te)
    end
end

"""

    GTBasis(basis::Array{<:AbstractGTBasisFuncs}, 1) -> GTBasis

Directly construct a `GTBasis` given a basis set.

"""
GTBasis(basis::Vector{<:AbstractGTBasisFuncs}) = 
GTBasis(basis, overlaps(basis), elecKinetics(basis), eeInteractions(basis))


function sortBasisFuncs(bs::Vector{<:FloatingGTBasisFuncs}; groupCenters::Bool=false)
    bfBlocks = Vector{<:FloatingGTBasisFuncs}[]
    sortedBasis = groupedSort(bs, centerOf)
    for subbs in sortedBasis
        # SubShells = [i.subshell for i in subbs] # compare ijkOrbitalList[ijk][end]
        # sortVec = sortperm([SubshellNumberList[i] for i in SubShells])

        ijkn = [(i.ijk[1], typeof(i).parameters[2]) for i in subbs]

        # Reversed order within same subshell but ordinary order among different subshells.
        sortVec = sortperm(map(ijkn) do x
                               val = ijkOrbitalList[x[1]]
                               [-sum(val); val; x[2]]
                           end, 
                           rev=true)
        push!(bfBlocks, subbs[sortVec])
    end
    groupCenters ? bfBlocks : (bfBlocks |> flatten)
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
    centerOf(bf::FloatingGTBasisFuncs) -> Array{<:Real, 1}

Return the center coordinate of the input `FloatingGTBasisFuncs`.
"""
function centerOf(bf::FloatingGTBasisFuncs)
    [i() for i in bf.center]
end


"""

    BasisFuncMix{BN, GN} <: CompositeGTBasisFuncs{BN, 1}

Sum of multiple `FloatingGTBasisFuncs` without any reformulation, treated as one basis 
Function in the integral calculation.

≡≡≡ Field(s) ≡≡≡

`BasisFunc::NTuple{BN, FloatingGTBasisFuncs{<:Any, <:Any, 1}}`: Inside basis functions

`param::Tuple{Vararg{<:ParamBox}}`: Inside parameters.

≡≡≡ Initialization Method(s) ≡≡≡

    BasisFuncMix(bfs::Vector{<:FloatingGTBasisFuncs{S, GN, 1} where {S, GN}}) ->
    BasisFuncMix{BN, GN}

"""
struct BasisFuncMix{BN, GN} <: CompositeGTBasisFuncs{BN, 1}
    BasisFunc::NTuple{BN, FloatingGTBasisFuncs{<:Any, <:Any, 1}}
    param::Tuple{Vararg{<:ParamBox}}
    function BasisFuncMix(bfs::Vector{<:FloatingGTBasisFuncs{S, GN, 1} where {S, GN}})
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


unpackBasisFuncs(bfm::BasisFuncMix) = bfm.BasisFunc |> collect
unpackBasisFuncs(bf::FloatingGTBasisFuncs) = typeof(bf)[bf]
unpackBasisFuncs(::Any) = FloatingGTBasisFuncs[]


function sumOf(bfs::Array{<:BasisFunc, N}) where {N}
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

sumOf(bfms::Array{<:CompositeGTBasisFuncs{<:Any, 1}, N}) where {N} = 
vcat((bfms .|> unpackBasisFuncs)...) |> sumOf


function mergeGaussFuncs(gf1::GaussFunc, gf2::GaussFunc)
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
            xpn = Exponent(xpn1())
        end

        if con1 === con2 || hasIdentical(con1, con2)
            res = GaussFunc(xpn, con1) * 2.0
        elseif hasEqual(con1, con2)
            res = GaussFunc(xpn, deepcopy(con1)) * 2.0
        else
            res = GaussFunc(xpn, Contraction(con1()+con2()))
        end

        return res
    else
        return GaussFunc[gf1, gf2]
    end
end

function mergeGaussFuncs(gf1::GaussFunc, gf2::GaussFunc, gf3::GaussFunc...)
    gfs = GaussFunc[gf1, gf2, gf3...]
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
            xpn = Exponent(val())
        end
        if length(ucons) == 1
            push!( res, GaussFunc(xpn, ucons[1]) * length(group) )
        elseif (markUnique(gcons)[2] |> length) == 1
            push!( res, GaussFunc(xpn, ucons[1] |> deepcopy) * length(group) )
        else
            push!( res, GaussFunc(xpn, Contraction(Float64[i() for i in gcons] |> sum)) )
        end
    end
    res
end

mergeGaussFuncs(gf::GaussFunc) = itself(gf)


"""

    add(b::CompositeGTBasisFuncs{<:Any, 1}) -> CompositeGTBasisFuncs{<:Any, 1}

    add(b1::CompositeGTBasisFuncs{<:Any, 1}, b2::CompositeGTBasisFuncs{<:Any, 1}) ->
    CompositeGTBasisFuncs{<:Any, 1}

    +(b::CompositeGTBasisFuncs{<:Any, 1}, bs::CompositeGTBasisFuncs{<:Any, 1}...) ->
    CompositeGTBasisFuncs{<:Any, 1}

Addition between `CompositeGTBasisFuncs{<:Any, 1}` such as `BasisFunc` and 
`Quiqbox.BasisFuncMix`. It can be called using `+` syntax.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> bf1 = genBasisFunc([1,1,1], (2,1))
BasisFunc{:S, 1}(gauss, subshell, center)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bf2 = genBasisFunc([1,1,1], (2,2))
BasisFunc{:S, 1}(gauss, subshell, center)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bf3 = bf1 + bf2
BasisFunc{:S, 1}(gauss, subshell, center)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bf3.gauss[1].con[]
3.0
```
"""
add(bf::BasisFunc) = itself(bf)

add(bf::BasisFuncs{<:Any, <:Any, 1}) = 
BasisFunc(bf.center, bf.gauss, ijkOrbitalList[bf.ijk[1]], bf.normalizeGTO)

function add(bf1::BasisFunc{T}, bf2::BasisFunc{T}) where {T}
    if bf1.ijk == bf2.ijk && 
       bf1.normalizeGTO == bf2.normalizeGTO && 
       (c = centerOf(bf1)) == centerOf(bf2)

        cen1 = bf1.center
        cen2 = bf2.center
        if cen1 === cen2 || hasIdentical(cen1, cen2)
            cen = bf1.center
        elseif hasEqual(bf1, bf2)
            cen = deepcopy(bf1.center)
        else
            cen = makeCenter(c)
        end
        gfs = GaussFunc[bf1.gauss..., bf2.gauss...]
        gfsN = mergeGaussFuncs(gfs...) |> Tuple
        BasisFunc(cen, gfsN, ijkOrbitalList[bf1.ijk[1]], bf1.normalizeGTO)
    else
        BasisFuncMix([bf1, bf2] |> sortBasisFuncs)
    end
end

add(bf1::BasisFunc, bf2::BasisFunc) = BasisFuncMix([bf1, bf2] |> sortBasisFuncs)

add(bfm::BasisFuncMix) = unpackBasisFuncs(bfm) |> sumOf

add(bf1::BasisFuncMix{1}, bf2::BasisFunc) = add(bf1.BasisFunc[1], bf2)

add(bf1::BasisFunc, bf2::BasisFuncMix{1}) = add(bf2, bf1)

add(bf::BasisFunc, bfm::BasisFuncMix) = BasisFunc[bf, bfm.BasisFunc...] |> sumOf

add(bfm::BasisFuncMix, bf::BasisFunc) = add(bf, bfm)

add(bf1::BasisFuncMix{1}, bf2::BasisFuncMix{1}) = add(bf1.BasisFunc[1], bf2.BasisFunc[1])

add(bf::BasisFuncMix{1}, bfm::BasisFuncMix) = add(bf.BasisFunc[1], bfm)

add(bfm::BasisFuncMix, bf::BasisFuncMix{1}) = add(bf, bfm)

add(bfm1::BasisFuncMix, bfm2::BasisFuncMix) = 
BasisFunc[bfm1.BasisFunc..., bfm2.BasisFunc...] |> sumOf

add(bf1::BasisFuncs{<:Any, <:Any, 1}, bf2::BasisFuncs{<:Any, <:Any, 1}) = 
BasisFunc[[bf1, bf2] .|> add |> sumOf]


"""

    mul(a1::Real, a2::GaussFunc) -> GaussFunc

    mul(a1::GaussFunc, a2::Real) -> GaussFunc

    mul(a1::GaussFunc, a2::GaussFunc) -> GaussFunc

Multiplication between `GaussFunc`s or contraction coefficient multiplication between a 
`Real` number and a `GaussFunc`. It can be called using `*` syntax.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> gf1 = GaussFunc(3,1)
GaussFunc(xpn=ParamBox{:α, Float64}(3.0)[α][∂], con=ParamBox{:d, Float64}(1.0)[d][∂])

julia> gf1 * 2
GaussFunc(xpn=ParamBox{:α, Float64}(3.0)[α][∂], con=ParamBox{:d, Float64}(2.0)[d][∂])

julia> gf1 * gf1
GaussFunc(xpn=ParamBox{:α, Float64}(6.0)[α][∂], con=ParamBox{:d, Float64}(1.0)[d][∂])

julia> gf1 * 2 * gf1
GaussFunc(xpn=ParamBox{:α, Float64}(6.0)[α][∂], con=ParamBox{:d, Float64}(2.0)[d][∂])
```
"""
function mul(gf::GaussFunc, coeff::Real)
    c = convert(Float64, coeff)::Float64
    conNew = deepcopy(gf.con)
    if conNew.map[] == itself
        conNew[] *= c
    else
        conNew.map = Ref((x)->c*gf.con.map[](x))
        conNew.data = gf.con.data
    end
    GaussFunc(gf.xpn, conNew)
end

mul(coeff::Real, gf::GaussFunc) = mul(gf, coeff)

mul(gf1::GaussFunc, gf2::GaussFunc) = 
GaussFunc(Exponent(gf1.xpn()+gf2.xpn()), Contraction(gf1.con()*gf2.con()))

"""

    mul(a1::Real, a2::CompositeGTBasisFuncs{<:Any, 1}; 
        normalizeGTO::Union{Bool, Missing}=missing) -> 
    CompositeGTBasisFuncs{<:Any, 1}

    mul(a1::CompositeGTBasisFuncs{<:Any, 1}, a2::Real; 
        normalizeGTO::Union{Bool, Missing}=missing) -> 
    CompositeGTBasisFuncs{<:Any, 1}

    mul(a1::CompositeGTBasisFuncs{<:Any, 1}, a2::CompositeGTBasisFuncs{<:Any, 1}; 
        normalizeGTO::Union{Bool, Missing}=missing) -> 
    CompositeGTBasisFuncs{<:Any, 1}

Multiplication between `CompositeGTBasisFuncs{<:Any, 1}`s such as `BasisFunc` and 
`Quiqbox.BasisFuncMix`, or contraction coefficient multiplication between a `Real` number 
and a `CompositeGTBasisFuncs{<:Any, 1}`. If `normalizeGTO` is set to `missing` (in 
default), The `GaussFunc` in the output result will be normalized only if all the input 
bases has `normalizeGTO = true`. The function can be called using `*` syntax.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> bf1 = genBasisFunc([1,1,1], ([2,1], [0.1, 0.2]))
BasisFunc{:S, 2}(gauss, subshell, center)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> bf2 = bf1 * 2
BasisFunc{:S, 2}(gauss, subshell, center)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]

julia> getindex.(getfield.(bf2.gauss, :con))
(0.2, 0.4)

julia> bf3 = bf1 * bf2
BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][1.0, 1.0, 1.0]
```
"""
function mul(sgf1::BasisFunc{<:Any, 1}, sgf2::BasisFunc{<:Any, 1}; 
             normalizeGTO::Union{Bool, Missing}=missing)
    ijk = ijkOrbitalList[sgf1.ijk[1]] + ijkOrbitalList[sgf2.ijk[1]]
    α₁ = sgf1.gauss[1].xpn()
    α₂ = sgf2.gauss[1].xpn()
    d₁ = sgf1.gauss[1].con()
    d₂ = sgf2.gauss[1].con()
    n₁ = sgf1.normalizeGTO
    n₂ = sgf2.normalizeGTO
    n₁ && (d₁ *= normOfGTOin(sgf1)[1])
    n₂ && (d₂ *= normOfGTOin(sgf2)[1])
    R₁ = centerOf(sgf1)
    R₂ = centerOf(sgf2)
    cen = (α₁*R₁ + α₂*R₂) / (α₁ + α₂)
    xpn = α₁ + α₂
    con = d₁ * d₂ * exp(-α₁ * α₂ / xpn * sum(abs2, R₁-R₂))
    normalizeGTO isa Missing && (normalizeGTO = n₁*n₂)
    BasisFunc(makeCenter(cen), GaussFunc(Exponent(xpn), Contraction(con)), 
              ijk, normalizeGTO)
end

function mul(bf::BasisFunc, coeff::Real; normalizeGTO::Union{Bool, Missing}=missing)
    n = bf.normalizeGTO
    normalizeGTO isa Missing && (normalizeGTO = n)
    c = (n && !normalizeGTO) ? (coeff .* (normOfGTOin(bf) |> Tuple)) : coeff
    gfs = mul.(bf.gauss, c)
    BasisFunc(bf.center, gfs, ijkOrbitalList[bf.ijk[1]], normalizeGTO)
end

mul(coeff::Real, bf::BasisFunc; normalizeGTO::Union{Bool, Missing}=missing) = 
mul(bf, coeff; normalizeGTO)

function mul(bfm::BasisFuncMix, coeff::Real; normalizeGTO::Union{Bool, Missing}=missing)
    bfs = mul.(bfm.BasisFunc, coeff; normalizeGTO) |> collect
    BasisFuncMix(bfs)
end

mul(coeff::Real, bfm::BasisFuncMix; normalizeGTO::Union{Bool, Missing}=missing) = 
mul(bfm, coeff; normalizeGTO)

function mul(bf1::BasisFunc, bf2::BasisFunc; normalizeGTO::Union{Bool, Missing}=missing)
    cen1 = bf1.center
    ijk1 = ijkOrbitalList[bf1.ijk[1]]
    cen2 = bf2.center
    ijk2 = ijkOrbitalList[bf2.ijk[1]]
    normalizeGTO isa Missing && (normalizeGTO = bf1.normalizeGTO * bf2.normalizeGTO)
    bfs = BasisFunc[]
    for gf1 in bf1.gauss, gf2 in bf2.gauss
        push!(bfs, mul(BasisFunc(cen1, gf1, ijk1, normalizeGTO), 
                       BasisFunc(cen2, gf2, ijk2, normalizeGTO)))
    end
    sumOf(bfs)
end

mul(bf1::BasisFuncMix{1}, bf2::BasisFunc; normalizeGTO::Union{Bool, Missing}=missing) = 
mul(bf1.BasisFunc[1], bf2; normalizeGTO)

mul(bf1::BasisFunc, bf2::BasisFuncMix{1}; normalizeGTO::Union{Bool, Missing}=missing) = 
mul(bf2, bf1; normalizeGTO)

mul(bf::BasisFunc, bfm::BasisFuncMix; normalizeGTO::Union{Bool, Missing}=missing) = 
mul.(Ref(bf), bfm.BasisFunc; normalizeGTO) |> collect |> sumOf

mul(bfm::BasisFuncMix, bf::BasisFunc; normalizeGTO::Union{Bool, Missing}=missing) = 
mul(bf, bfm; normalizeGTO)

mul(bf1::BasisFuncMix{1}, bf2::BasisFuncMix{1}; 
    normalizeGTO::Union{Bool, Missing}=missing) = 
mul(bf1.BasisFunc[1], bf2.BasisFunc[1]; normalizeGTO)

mul(bf::BasisFuncMix{1}, bfm::BasisFuncMix; normalizeGTO::Union{Bool, Missing}=missing) = 
mul(bf.BasisFunc[1], bfm; normalizeGTO)

mul(bfm::BasisFuncMix, bf::BasisFuncMix{1}; normalizeGTO::Union{Bool, Missing}=missing) = 
mul(bf, bfm; normalizeGTO)

function mul(bfm1::BasisFuncMix, bfm2::BasisFuncMix; 
             normalizeGTO::Union{Bool, Missing}=missing)
    bfms = CompositeGTBasisFuncs{<:Any, 1}[]
    for bf1 in bfm1.BasisFunc, bf2 in bfm2.BasisFunc
        push!(bfms, mul(bf1, bf2; normalizeGTO))
    end
    sumOf(bfms)
end

mul(bf1::BasisFuncs{<:Any, <:Any, 1}, bf2::BasisFuncs{<:Any, <:Any, 1}; 
    normalizeGTO::Union{Bool, Missing}=missing) = 
[mul(add(bf1), add(bf2); normalizeGTO)]


"""

    shift(bf::FloatingGTBasisFuncs{S, GN, 1}, didjdk::Vector{Int}) where {S, GN} -> 
    BasisFunc

Shift (add) the angular momentum (Cartesian representation) given the a vector that 
specifies the change of each pseudo-quantum number 𝑑i, 𝑑j, 𝑑k.
"""
function shift(bf::FloatingGTBasisFuncs{S, GN, 1}, didjdk::Vector{Int}) where {S, GN}
    @assert didjdk |> length == 3 "The length of `didjdk` should be 3."
    shiftCore(bf, didjdk)
end

shiftCore(bf::FloatingGTBasisFuncs{S, GN, 1}, didjdk::Vector{<:Real}) where {S, GN} = 
BasisFunc(bf.center, bf.gauss, 
          ijkOrbitalList[bf.ijk[1]] + convert(Vector{Int}, didjdk), bf.normalizeGTO)


"""

    decompose(bf::CompositeGTBasisFuncs; splitGaussFunc::Bool=false) -> 
    Array{<:FloatingGTBasisFuncs, 2}

Decompose a `FloatingGTBasisFuncs` into an `Array` of `BasisFunc`s. Each column represents 
one orbital of the input basis function(s). If `splitGaussFunc` is `true`, then each column 
consists of the `BasisFunc`s each with only 1 `GaussFunc`.
"""
function decompose(bf::FloatingGTBasisFuncs{S, GN, ON}; 
                   splitGaussFunc::Bool=false) where {S, GN, ON}
    if splitGaussFunc
        nRow = GN
        nG = 1
        gs = bf.gauss
    else
        nRow = 1
        nG = GN
        gs = Ref(bf.gauss)
    end
    res = Array{BasisFunc{S, nG}, 2}(undef, nRow, ON)
    for (c, ijk) in zip(eachcol(res), bf.ijk)
        c .= BasisFunc.(Ref(bf.center), gs, Ref(ijkOrbitalList[ijk]), bf.normalizeGTO)
    end
    res
end

# decompose(bf::BasisFunc{S, 1}; splitGaussFunc::Bool=false) where {S} = 
# reshape(BasisFunc[bf], 1, 1)

function decompose(bfm::BasisFuncMix; splitGaussFunc::Bool=false)
    if splitGaussFunc
        bfs = decompose.(bfm.BasisFunc; splitGaussFunc)
        vcat(bfs...)
    else
        bfm
    end
end


"""

    basisSize(subshell::Union{String, Array{String, 1}}) -> Tuple 

Return the size (number of orbitals) of each subshell.
"""
basisSize(subshell::String) = (SubshellDimList[subshell],)
basisSize(subshells::Vector{String}) = basisSize.(subshells) |> flatten |> Tuple

"""

    basisSize(basisFunctions) -> Tuple

Return the numbers of orbitals of the input basis function(s).
"""
basisSize(basis::FloatingGTBasisFuncs) = (basis.ijk |> length,)
basisSize(::BasisFuncMix) = (1,)
basisSize(basisSet::Vector{<:Any}) = basisSize.(basisSet) |> flatten |> Tuple


# Core function to generate a customized X-Gaussian (X>1) basis function.
function genGaussFuncText(xpn::Real, con::Real)
    """
         $(join(map(x -> rpad(round(x, sigdigits=10)|>alignSignedNum, 20), 
                [xpn, con])) |> rstrip)
    """
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
    GFs = map(x -> genGaussFuncText(x.xpn[], x.con[]), gauss)
    cen = centerOf(bf)
    firstLine = printCenter ? "X   "*rpad(cen[1]|>alignSignedNum, 20)*
                                     rpad(cen[2]|>alignSignedNum, 20)*
                                     rpad(cen[3]|>alignSignedNum, 20)*"\n" : ""
    firstLine*"$(bf.subshell)    $(bf.gauss |> length)   $(norm)\n" * (GFs |> join)
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
            str = genBasisFuncText(b[1]; norm, printCenter)
            str *= genBasisFuncText.(b[2:end]; norm, printCenter=false) |> join
            push!(strs, str)
        end
    else
        for b in bfBlocks
            push!(strs, genBasisFuncText(b; norm, printCenter))
        end
    end
    strs
end


"""

    genBFuncsFromText(content::String; adjustContent::Bool=false, 
                      adjustFunction::F=sciNotReplace, 
                      excludeFirstNlines::Int=0, excludeLastNlines::Int=0, 
                      center::Union{AbstractArray, 
                                    Tuple{N, ParamBox}, 
                                    Missing}=missing) where {N, F<:Function} -> 
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
`BasisFunc`. The coordinate information, if included, should be right above the subshell 
information for the `BasisFunc`. E.g.:
```
    "X    1.0                 0.0                 0.0                \nS    1   1.0"*
    "\n      2.0                 1.0\n"
```
"""
function genBFuncsFromText(content::String;
                           adjustContent::Bool=false,
                           adjustFunction::F=sciNotReplace, 
                           excludeFirstNlines::Int=0, excludeLastNlines::Int=0, 
                           center::Union{AbstractArray, 
                                         NTuple{N, ParamBox}, 
                                         Missing}=missing) where {N, F<:Function}
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
            append!(bfs, genBasisFunc.(Ref(center), [gs1, gs2], ["S", "P"], 
                                       normalizeGTO=true))
        else
            for j = i+1 : i+ng
                push!(gs1, GaussFunc(data[j]...))
            end
            push!(bfs, genBasisFunc(center, gs1, (data[i][1] |> string), normalizeGTO=true))
        end
        center = centerOld
    end
    bfs |> flatten
end

"""

    assignCenter!(center::AbstractArray, b::FloatingGTBasisFuncs) -> NTuple{3, ParamBox}

Assign a new coordinate to the center of the input `FloatingGTBasisFuncs`. 
Also return the altered center.
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
    x = ParamBox(c[1], ParamList[:X])
    y = ParamBox(c[2], ParamList[:Y])
    z = ParamBox(c[3], ParamList[:Z])
    (x,y,z)
end


"""

    getParams(pbc::Union{ParamBox, GaussFunc, FloatingGTBasisFuncs}, 
              symbol::Union{Symbol, Nothing}=nothing; onlyDifferentiable::Bool=false) -> 
    Union{ParamBox, Array{<:ParamBox, 1}}

Return the parameter(s)`::ParamBox` stored in the input container. If keyword argument 
`symbol` is `nothing`, then return all the different parameters; if it's set to the 
`Symbol` type of a parameter (e.g. the symbol of `ParamBox{T}` would be `T`), the function 
will only search for that type of parameters (which might have different indices). 
`onlyDifferentiable` determines whether ignore non-differentiable parameters.
"""
function getParams(pb::ParamBox, symbol::Union{Symbol, Nothing}=nothing; 
                   onlyDifferentiable::Bool=false)
    !(onlyDifferentiable ? pb.canDiff[] : true) && (return nothing)
    !(symbol === nothing ? true : (symbol == typeof(pb).parameters[1])) && (return nothing)
    pb
end

function getParams(pbs::Vector{<:ParamBox}, symbol::Union{Symbol, Nothing}=nothing; 
                   onlyDifferentiable::Bool=false)
    res = symbol === nothing ? ParamBox[] : ParamBox{symbol}[]
    for i in pbs
     j = getParams(i, symbol; onlyDifferentiable)
     j !== nothing && push!(res, j)
    end
    res
end

getParams(gf::GaussFunc, symbol::Union{Symbol, Nothing}=nothing; 
          onlyDifferentiable::Bool=false) = 
getParams([gf.xpn, gf.con], symbol; onlyDifferentiable)

getParams(bf::FloatingGTBasisFuncs, symbol::Union{Symbol, Nothing}=nothing; 
          onlyDifferentiable::Bool=false) = 
vcat( getParams.(bf.gauss, symbol; onlyDifferentiable)..., 
      getParams(bf.center |> collect, symbol; onlyDifferentiable) )

getParams(bfm::BasisFuncMix, symbol::Union{Symbol, Nothing}=nothing; 
          onlyDifferentiable::Bool=false) = 
vcat( getParams.((collect ∘ flatten)( getfield.(bfm.BasisFunc, :gauss) ), symbol; 
                 onlyDifferentiable)..., 
      getParams((collect ∘ flatten)( getfield.(bfm.BasisFunc, :center) ), symbol; 
                 onlyDifferentiable) )

getParams(cs, symbols::Vector{Symbol}; onlyDifferentiable::Bool=false) = 
getParams.(Ref(cs), symbols; onlyDifferentiable) |> flatten

"""

    getParams(cs::Array, symbol::Union{Symbol, Nothing}=nothing; 
              onlyDifferentiable::Bool=false) -> 
    Array{<:ParamBox, 1}

Method of `getParams` when the 1st argument is an `Array` of `ParamBox`, `GaussFunc`, 
`FloatingGTBasisFuncs` or any of them.
"""
getParams(cs::Array, symbol::Union{Symbol, Nothing}=nothing; 
          onlyDifferentiable::Bool=false) = 
getParams.(cs, symbol; onlyDifferentiable) |> flatten


"""

    dataCopy(b::Union{GaussFunc, CompositeGTBasisFuncs}) -> typeof(b)

A deep copy of the input basis function(s) except the mapping relation(s) of stored 
parameter(s) won't be kept.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> e = Exponent(3.0, mapFunction=x->x^2)
ParamBox{:α, Float64}(3.0)[α -> #1(α)][∂]

julia> c = Contraction(2.0)
ParamBox{:d, Float64}(2.0)[d][∂]

julia> gf1 = GaussFunc(e, c);

julia> gf2 = dataCopy(gf1)
GaussFunc(xpn=ParamBox{:α, Float64}(9.0)[α][∂], con=ParamBox{:d, Float64}(2.0)[d][∂])

julia> gf1.xpn() == gf2.xpn()
true

julia> (gf1.xpn[] |> gf1.xpn.map[]) == gf2.xpn[]
true
```
"""
function dataCopy(b::StructSpatialBasis)
    bL = deepcopy(b)
    pbs = getParams(bL)
    for pb in pbs
        pb[] = pb()
        pb.map = Ref(itself)
        pb.index = nothing
    end
    bL
end


function markParams!(parArray::Vector{<:ParamBox{V}}; 
                     ignoreContainer::Bool=false, filter::Bool=true, 
                     filterMapping::Bool=false) where {V}
    res, _ = markUnique(parArray, compareFunction=hasIdentical, ignoreFunction=true; 
                        ignoreContainer)
    for i=1:length(parArray)
        parArray[i].index = res[i]
    end
    if filter
        _, cmprList = markUnique(parArray, compareFunction=hasIdentical, 
                                 ignoreFunction=filterMapping; ignoreContainer)
        return cmprList
    end
    parArray
end

function markParams!(parArray::Vector{<:ParamBox}; 
                     ignoreContainer::Bool=false, filter::Bool=true, 
                     filterMapping::Bool=false)
    pars = ParamBox[]
    syms = getUnique!([typeof(i).parameters[1] for i in parArray])
    arr = parArray |> copy
    for sym in syms
        typ = ParamBox{sym}
        subArr = typ[]
        ids = Int[]
        for i = 1:length(arr)
            if arr[i] isa typ
                push!(subArr, arr[i])
                push!(ids, i)
            end
        end
        deleteat!(arr, ids)
        append!(pars, markParams!(subArr; ignoreContainer, filter, filterMapping))
    end
    pars
end

"""

    uniqueParams!(bs; ignoreContainer::Bool=false, filter::Bool=true, 
                  filterMapping::Bool=false) -> 
    Array{<:ParamBox, 1}

Mark the parameters (`ParamBox`) in input bs which can a `Vector` of `GaussFunc` or 
`FloatingGTBasisFuncs`. The identical parameters will be marked with same index.

=== Keyword argument(s) ===

`ignoreContainer`: If set to `true`, then only the field `data` of the `ParamBox`s will be 
compared to determine whether each `ParamBox` are unique. 

`filter`: Determine whether filter out the identical `ParamBox`s and only return the unique 
ones.

`filterMapping`: Determine wether return the `ParamBox`s with identical fields except the 
`map` field. When `filter=false`, this argument is automatically overwritten to be `false`.
"""
uniqueParams!(bs; ignoreContainer::Bool=false, filter::Bool=true, 
              filterMapping::Bool=false) = 
markParams!(getParams(bs); ignoreContainer, filter, filterMapping)


"""

    getVar(pb::ParamBox; includeMapping::Bool=false) -> 
    Array{<:Pair{Symbolics.Num, <:Number}, 1}

Return a 1-element `Vector` of `Pair` to show the `Symbol::Symbolics.Num` of the stored 
variable and the corresponding values. `includeMapping` determines whether mappings from 
the variable to the dependent variable if there is one.
"""
function getVar(pb::ParamBox; includeMapping::Bool=false)
    varName = typeof(pb).parameters[1]
    superscript = pb.canDiff[] ? "" : NoDiffMark
    varSymbol = Symbol((varName |> string) * superscript)
    vr = (pb.index isa Int) ? Symbolics.variable(varSymbol, pb.index) : 
                              Symbolics.variable(varSymbol)
    res = Pair[vr => pb[]]
    if includeMapping && !(pb.map[] isa typeof(itself))
        mapName = pb.map[] |> nameof
        dvr = Symbolics.variable(mapName, T=Symbolics.FnType{Tuple{Any}, Real})(vr)
        expr = pb.map[](vr)
        (pushfirst!(res, dvr=>expr, expr=>pb()) |> unique!)
    end
    res
end

getVar(pbType::Type{<:ParamBox}) = Symbolics.variable(pbType.parameters[1])

"""

    getVars(obj::Union{GaussFunc, BasisFunc}; includeMapping::Bool=false) -> 
    Array{<:Pair, 1}

    getVars(collection::Array{<:Union{GaussFunc, BasisFunc, ParamBox}, 1}; 
            includeMapping::Bool=false) -> 
    Array{<:Pair, 1}

Return a `Vector` of `Pair` to of the mapping relations between the variables stored in the 
`ParamBox`s and the corresponding values. `includeMapping` determines whether mappings from 
the variable(s) to the dependent variable(s) if exists.
"""
getVars(gf::GaussFunc; includeMapping::Bool=false) = 
getVar.(gf.param; includeMapping) |> flatten |> Dict

getVars(bf::BasisFunc; includeMapping::Bool=false) = 
getVar.(bf.param; includeMapping) |> flatten |> Dict

getVars(pbs::Vector{<:ParamBox}; includeMapping::Bool=false) = 
getVar.(pbs; includeMapping) |> flatten |> Dict

getVars(fs::Vector{<:Union{GaussFunc, BasisFunc}}; includeMapping::Bool=false) = 
merge(getVars.(fs; includeMapping)...)


function Nlα(l, α)
    if l < 2
        ( ( 2^(2l+3) * factorial(l+1) * 2^(l+1.5) / 
        (factorial(2l+2) * √π) ) |> sqrt) * α^(0.5l + 0.75)
    else
        # for higher angular momentum make the upper bound of norms be 1.
        ( ( 2^(3l+1.5) * factorial(l) / 
        (factorial(2l) * π^1.5) ) |> sqrt ) * α^(0.5l + 0.75)
    end
end

Nlα(subshell::String, α) = Nlα(SubshellNumberList[subshell], α)


Nijk(i, j, k) = (2/π)^0.75 * ( 2^(3*(i+j+k)) * factorial(i) * factorial(j) * factorial(k) / 
                (factorial(2i) * factorial(2j) * factorial(2k)) )^0.5


function Nijkα(i, j, k, α)
    l = i + j + k
    if l < 2
        ( ( 2^(2l+3) * factorial(l+1) * 2^(l+1.5) / 
        (factorial(2l+2) * √π) ) |> sqrt ) * α^(0.5l + 0.75)
    else
        # for higher angular momentum make the upper bound of norms be 1.
        Nijk(i, j, k) * α^(0.5l + 0.75)
    end
end


pgf0(x, y, z, α) = exp( -α * (x^2 + y^2 + z^2) )
cgf0(x, y, z, α, d) = d * pgf0(x, y, z, α)
cgo0(x, y, z, α, d, i, j, k, N=1.0) = N * x^i * y^j * z^k * cgf0(x, y, z, α, d)
# fgo0(x, y, z, Rx, Ry, Rz, α, d, i, j, k, N=1.0) = cgo0(x-Rx, y-Ry, z-Rz, α, d, i, j, k, N)


pgf(r, α) = pgf0(r[1], r[2], r[3], α)
cgf(r, α, d) = cgf0(r[1], r[2], r[3], α, d)
cgo(r, α, d, l, N=Nijkα(i,j,k,α)) = cgo0(r[1], r[2], r[3], α, d, l[1], l[2], l[3], N)
# fgo(r, R, α, d, l, N=Nijkα(i,j,k,α)) = fgo0(r[1], r[2], r[3], R[1], R[2], R[3], 
#                                             α, d, l[1], l[2], l[3], N)
cgo2(r, α, d, i, j, k, N=Nijkα(i,j,k,α)) = cgo0(r[1], r[2], r[3], α, d, i, j, k, N)
# fgo2(r, R, α, d, i, j, k, N=Nijkα(i,j,k,α)) = fgo0(r[1], r[2], r[3], R[1], R[2], R[3], 
#                                                    α, d, i, j, k, N)


normOfGTOin(b::FloatingGTBasisFuncs{S, GN, 1})  where {S, GN} = 
Nijkα.(ijkOrbitalList[b.ijk[1]]..., [g.xpn() for g in b.gauss])

normOfGTOin(b::FloatingGTBasisFuncs{S, GN, ON}) where {S, GN, ON} = 
Nlα.(b.subshell, [g.xpn() for g in b.gauss])


function expressionOfCore(bf::FloatingGTBasisFuncs{S, GN, ON}, substituteValue::Bool=false, 
                          onlyParameter::Bool=false, splitGaussFunc::Bool=false) where 
                         {S, GN, ON}
    includeMapping = true
    index = substituteValue ? 2 : 1
    N = bf.normalizeGTO  ?  (ON == 1 ? Nijkα : (i,j,k,α)->Nlα(i+j+k, α))  :  (_...) -> 1
    R = getindex.(getindex.(getVar.(bf.center |> collect; includeMapping), 1), index)
    pars = getfield.(bf.gauss, :param)
    α = getindex.(getindex.(getVar.(getindex.(pars, 1); includeMapping), 1), index)
    d = getindex.(getindex.(getVar.(getindex.(pars, 2); includeMapping), 1), index)
    x = onlyParameter ? -R : Symbolics.variable.(:r, [1:3;]) - R
    res = map(bf.ijk) do ijk
        i, j, k = ijkOrbitalList[ijk]
        exprs = cgo2.(Ref(x), α, d, i, j, k, N.(i,j,k,α)) |> collect
        splitGaussFunc ? exprs : sum(exprs)
    end
    hcat(res...)
end

function expressionOfCore(bfm::BasisFuncMix, substituteValue::Bool=false, 
                          onlyParameter::Bool=false, splitGaussFunc::Bool=false)
    exprs = Matrix{Symbolics.Num}[expressionOfCore(bf, substituteValue, 
                                                   onlyParameter, splitGaussFunc)
                                  for bf in bfm.BasisFunc]
    splitGaussFunc ? vcat(exprs...) : sum(exprs)
end

function expressionOfCore(gf::GaussFunc, substituteValue::Bool=false)
    r = Symbolics.variable.(:r, [1:3;])
    includeMapping = true
    index = substituteValue ? 2 : 1
    cgf(r, getVar(gf.xpn; includeMapping)[1][index], 
           getVar(gf.con; includeMapping)[1][index])
end

"""

    expressionOf(bf::CompositeGTBasisFuncs; splitGaussFunc::Bool=false) -> Array{<:Symbolics.Num, 2}

Return the expression(s) of a given `CompositeGTBasisFuncs` (e.g. `BasisFuncMix` or 
`FloatingGTBasisFuncs`) as a `Matrix{<:Symbolics.Num}`of which the column(s) corresponds to 
different orbitals. If `splitGaussFunc` is `true`, the column(s) will be expanded such that its 
entries are `GaussFunc` inside the corresponding orbital.
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
    if res === nothing
        res = recursivelyGet(varDict, 
                             symbolReplace(Symbolics.tosymbol(vr), 
                                           NoDiffMark=>"") |> Symbolics.variable)
    end
    if res === nothing
        str = Symbolics.tosymbol(vr) |> string
        # pos = findfirst(r"\p{No}", str)[1]
        pos = findfirst(r"[₁₂₃₄₅₆₇₈₉₀]", str)[1]
        front = split(str,str[pos])[1]
        var = front*NoDiffMark*str[pos:end] |> Symbol
        recursivelyGet(varDict, var |> Symbolics.variable)
    end
    @assert res !== nothing "Can NOT find the value of $(vr)::$(typeof(vr)) in the given "*
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
    getFunc = (fSym) -> try getfield(Quiqbox, fSym) catch; getfield(Main, fSym) end
    getFsym = (t) -> t isa SymbolicUtils.Sym ? Symbolics.tosymbol(t) : Symbol(t)
    if vr.f isa Symbolics.Differential
        dvr = vr.arguments[]
        vr = vr.f.x
        if dvr isa SymbolicUtils.Term
            f = getFunc(dvr.f |> getFsym)
            expr = f(dvr.arguments[]) # assuming AD propagates only through 1 var: f(g(x)).
        else
            expr = dvr
        end
        return varVal(Symbolics.derivative(expr, vr), varDict)
    else
        if vr.f isa Union{SymbolicUtils.Sym, Function}
            fSymbol = vr.f |> getFsym
        else return NaN end
        f = getFunc(fSymbol)
        v = varVal(vr.arguments[], varDict)
        return f(v)
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

diffTransferCore(term::Symbolics.Num, args...) = diffTransferCore(term.val, args...)

diffTransferCore(term::Real, _...) = Real[Float64(term), 0,0,0]


function diffTransfer(term::Num, varDict::Dict{Num, <:Real})
    terms = splitTerm(term)
    diffTransferCore.(terms, Ref(varDict))
end


function diffInfo(bf::CompositeGTBasisFuncs, vr, varDict)
    exprs = expressionOfCore(bf, false, true, true)
    relDiffs = Symbolics.simplify.(Symbolics.derivative.(log.(exprs), vr), expand=true)
    diffTransfer.(relDiffs, Ref(varDict))
end


function diffInfoToBasisFunc(bf::FloatingGTBasisFuncs, info::Matrix{<:Any})
    bs = decompose(bf, splitGaussFunc=true)
    mat = map(function (x, y)
                  shiftCore.(x .* getindex.(y, 1), getindex.(y, Ref(2:4))) |> BasisFuncMix
              end, bs, info)
    eachcol(mat) .|> sum
end