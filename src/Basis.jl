export GaussFunc, BasisFunc, BasisFuncs, genBasisFunc, centerOf, GTBasis, add, mul, shift, 
       decomposeBasisFunc, basisSize, genBasisFuncText, genBFuncsFromText, assignCenter!, 
       getParams, uniqueParams!, getVar, getVars, expressionOf

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

"""
struct GaussFunc <: AbstractGaussFunc
    xpn::ParamBox{ParamList[:xpn], Float64}
    con::ParamBox{ParamList[:con], Float64}
    param::Tuple{ParamBox{ParamList[:xpn], Float64}, ParamBox{ParamList[:con], Float64}}

    GaussFunc(xpn::ParamBox{ParamList[:xpn], Float64}, 
              con::ParamBox{ParamList[:con], Float64}) = 
    new(xpn, con, (xpn, con))
end

"""

    GaussFunc(xpn::Real, con::Real) -> GaussFunc

Generate a `GaussFunc` given the specified exponent coefficient `xpn` and contraction 
coefficient `con`.
"""
function GaussFunc(e::Real, c::Real)
    xpn = ParamBox(e, ParamList[:xpn])
    con = ParamBox(c, ParamList[:con])
    GaussFunc(xpn, con)
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

    julia> genBasisFunc([0,0,0], GaussFunc(2,1), [0,1,0])
    BasisFunc{:P, 1}(gauss, subshell, center)[X⁰Y¹Z⁰][0.0, 0.0, 0.0]

≡≡≡ Method 2 ≡≡≡

    genBasisFunc(coord::AbstractArray, gs::Array{<:GaussFunc, 1}, subshell::String="S"; 
                 ijkFilter::Array{Bool, 1}=fill(true, SubshellDimList[subshell]), 
                 normalizeGTO::Bool=false)

≡≡≡ Example(s) ≡≡≡

    julia> genBasisFunc([0,0,0], GaussFunc(2,1), "S")
    BasisFunc{:S, 1}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]

    julia> genBasisFunc([0,0,0], GaussFunc(2,1), "P")
    BasisFuncs{:P, 1, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]

≡≡≡ Method 3 ≡≡≡

    genBasisFunc(coord::AbstractArray, gExpsANDgCons::NTuple{2, Array{<:Real, 1}}, 
                 subshell="S"; kw...)

Instead of directly inputting `GaussFunc`, one can also input a 2-element `Tuple` of the 
exponent(s) and contraction coefficient(s) corresponding to the same `GaussFunc`(s).

≡≡≡ Example(s) ≡≡≡

    julia> genBasisFunc([0,0,0], (2, 1), "P")
    BasisFuncs{:P, 1, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]

    julia> genBasisFunc([0,0,0], ([2, 1.5], [1, 0.5]), "P")
    BasisFuncs{:P, 2, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]

≡≡≡ Method 4 ≡≡≡

    genBasisFunc(center, BSKeyANDnuc::Array{Tuple{String, String}, 1})

If the user wants to construct existed atomic basis set(s), they can use the (`Array` of) 
`(BS_name, Atom_name)` as the second input. If the atom is omitted, then basis set for H 
is used.

≡≡≡ Example(s) ≡≡≡

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
    cens = centerOf.(bs)
    bfBlocks = Vector{<:FloatingGTBasisFuncs}[]
    mark, _ = markUnique(cens, compareFunction=isequal)
    m = max(mark...)
    for i=1:m
        idx = findall(x->x==i, mark)
        subbs = bs[idx]
        # SubShells = [i.subshell for i in subbs] # compare ijkOrbitalList[ijk][end]
        # sortVec = sortperm([SubshellNumberList[i] for i in SubShells])

        ijks = [i.ijk[1] for i in subbs]
        # Reversed order within same subshell but ordinary order among different subshells.
        sortVec = sortperm(map(ijks) do x
                               val = ijkOrbitalList[x]
                               [-sum(val); val]
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
Sum of multiple `FloatingGTBasisFuncs`, treated as one basis Function.
"""
struct BasisFuncMix{BN, GN} <: CompositeGTBasisFuncs{BN, 1}
    BasisFunc::NTuple{BN, FloatingGTBasisFuncs}
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
BasisFuncMix(bfs::BasisFuncs) = BasisFuncMix.(decomposeBasisFunc(bfs))
BasisFuncMix(bfm::BasisFuncMix) = itself(bfm)


getBasisFuncs(bfm::BasisFuncMix) = bfm.BasisFunc
getBasisFuncs(bf::FloatingGTBasisFuncs) = (bf,)
getBasisFuncs(::Any) = ()


function sumOf(bfs::Array{<:BasisFunc, N}) where {N}
    bfs = sortBasisFuncs(bfs[:])
    head = bfs[1]
    body = @view bfs[2:end]
    for bf in body
        head = (head isa BasisFuncMix) ? 
               BasisFuncMix([head.BasisFunc..., bf]) : add(head, bf)
    end
    head
end


add(bf::BasisFunc) = itself(bf)

add(bf::BasisFuncs{<:Any, <:Any, 1}) = 
BasisFunc(bf.center, bf.gauss, ijkOrbitalList[bf.ijk[1]], bf.normalizeGTO)

function add(bf1::BasisFunc{T}, bf2::BasisFunc{T}) where {T}
    if bf1.ijk == bf2.ijk && 
       bf1.normalizeGTO == bf2.normalizeGTO && 
       centerOf(bf1) == centerOf(bf2)

        BasisFunc(bf1.center, (bf1.gauss..., bf2.gauss...,), 
                  ijkOrbitalList[bf1.ijk[1]], bf1.normalizeGTO)
    else
        BasisFuncMix([bf1, bf2])
    end
end

add(bf1::BasisFunc, bf2::BasisFunc) = BasisFuncMix([bf1, bf2])

add(bfm::BasisFuncMix) = getBasisFuncs(bfm) |> sumOf

add(bf1::BasisFuncMix{1}, bf2::BasisFunc) = add(bf1.BasisFunc[1], bf2)

add(bf1::BasisFunc, bf2::BasisFuncMix{1}) = add(bf2, bf1)

add(bf::BasisFunc, bfm::BasisFuncMix) = [bf, bfm.BasisFunc...] |> sumOf

add(bfm::BasisFuncMix, bf::BasisFunc) = add(bf, bfm)

add(bf1::BasisFuncMix{1}, bf2::BasisFuncMix{1}) = add(bf1.BasisFunc[1], bf2.BasisFunc[1])

add(bf::BasisFuncMix{1}, bfm::BasisFuncMix) = add(bf.BasisFunc[1], bfm)

add(bfm::BasisFuncMix, bf::BasisFuncMix{1}) = add(bf, bfm)

add(bfm1::BasisFuncMix, bfm2::BasisFuncMix) = 
[bfm1.BasisFunc..., bfm2.BasisFunc...] |> sumOf

add(bf1::BasisFuncs{<:Any, <:Any, 1}, bf2::BasisFuncs{<:Any, <:Any, 1}) = 
[[bf1, bf2] .|> add |> sumOf]


function mul(gf::GaussFunc, coeff::Real)
    c = convert(Float64, coeff)::Float64
    conNew = deepcopy(gf.con)
    if conNew.map[] == itself
        conNew[] *= c
    else
        conNew.map = Ref((x)->c*gf.con.map(x))
        conNew.data = gf.con.data
    end
    GaussFunc(gf.xpn, conNew)
end

mul(coeff::Real, gf::GaussFunc) = mul(gf, coeff)

mul(gf1::GaussFunc, gf2::GaussFunc) = GaussFunc(gf1.xpn()+gf2.xpn(), gf1.con()*gf2.con())

function mul(sgf1::BasisFunc{<:Any, 1}, sgf2::BasisFunc{<:Any, 1}; 
             normalizeGTO::Union{Bool, Missing}=missing)
    ijk = ijkOrbitalList[sgf1.ijk[1]] + ijkOrbitalList[sgf2.ijk[1]]
    α₁ = sgf1.gauss[1].xpn()
    α₂ = sgf2.gauss[1].xpn()
    d₁ = sgf1.gauss[1].con()
    d₂ = sgf2.gauss[1].con()
    R₁ = centerOf(sgf1)
    R₂ = centerOf(sgf2)
    cen = (α₁*R₁ + α₂*R₂) / (α₁ + α₂)
    xpn = α₁ + α₂
    con = d₁ * d₂ * exp(-α₁ * α₂ / xpn * sum(abs2, R₁-R₂))
    normalizeGTO isa Missing && (normalizeGTO = sgf1.normalizeGTO*sgf2.normalizeGTO)
    genBasisFunc(cen, (xpn, con), ijk; normalizeGTO)
end

function mul(bf::BasisFunc, coeff::Real; normalizeGTO::Union{Bool, Missing}=missing)
    gfs = mul.(bf.gauss, coeff)
    normalizeGTO isa Missing && (normalizeGTO = bf.normalizeGTO)
    BasisFunc(bf.center, gfs, ijkOrbitalList[bf.ijk[1]], normalizeGTO)
end

mul(coeff::Real, bf::BasisFunc) = mul(bf, coeff)

function mul(bf1::BasisFunc, bf2::BasisFunc; normalizeGTO::Union{Bool, Missing}=missing)
    cen1 = bf1.center
    ijk1 = ijkOrbitalList[bf1.ijk[1]]
    cen2 = bf2.center
    ijk2 = ijkOrbitalList[bf2.ijk[1]]
    normalizeGTO isa Missing && (normalizeGTO = bf.normalizeGTO)
    bfs = BasisFunc[]
    for gf1 in bf1.gauss, gf2 in bf2.gauss
        push!(bfs, mul(BasisFunc(cen1, (gf1,), ijk1, normalizeGTO), 
                       BasisFunc(cen2, (gf2,), ijk2, normalizeGTO)))
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

function mul(bfm1::BasisFuncMix, bfm2::BasisFuncMix)
    bfs = BasisFunc[]
    for bf1 in bfm1.BasisFunc, bf2 in bfm2.BasisFunc
        push!(bfs, mul(bf1, bf2))
    end
    sumOf(bfs)
end

mul(bf1::BasisFuncs{<:Any, <:Any, 1}, bf2::BasisFuncs{<:Any, <:Any, 1}) = 
[mul(add(bf1), add(bf2))]

"""

    decomposeBasisFunc(bf::FloatingGTBasisFuncs; splitGaussFunc::Bool=false) -> 
    Array{<:FloatingGTBasisFuncs, 2}

Decompose a `FloatingGTBasisFuncs` into an `Array` of `BasisFunc`s. Each column represents 
one orbital of the input basis function(s). If `splitGaussFunc` is `true`, then each column 
consists of the `BasisFunc`s each with only 1 `GaussFunc`.
"""
function decomposeBasisFunc(bf::FloatingGTBasisFuncs; splitGaussFunc::Bool=false)
    cen = bf.center
    res = BasisFunc[]
    nRow = 1
    if splitGaussFunc
        for ijk in bf.ijk, g in bf.gauss
            push!(res, BasisFunc(cen, (g,), ijkOrbitalList[ijk], bf.normalizeGTO))
        end
        nRow = bf.gauss |> length
    else
        for ijk in bf.ijk
            push!(res, BasisFunc(cen, bf.gauss, ijkOrbitalList[ijk], 
                  bf.normalizeGTO))
        end
    end
    reshape(res, (nRow, bf.ijk |> length))
end

"""

    decomposeBasisFunc(bf::BasisFuncMix; splitGaussFunc::Bool=false) -> 
    Array{<:FloatingGTBasisFuncs, 3}

Return a 3-dimensional `Array` of `BasisFunc`s of which the pages (3rd dim) are the 
decomposed terms of the `BasisFunc`s stored in the input `BasisFuncMix`.
"""
function decomposeBasisFunc(bfm::BasisFuncMix; splitGaussFunc::Bool=false)
    bfs = decomposeBasisFunc.(getBasisFuncs(bfm) |> collect; splitGaussFunc)
    cat(bfs..., dims=3)
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
function genGaussFuncText(exponent::Real, contraction::Real)
    """
         $(join(map(x -> rpad(round(x, sigdigits=10)|>alignSignedNum, 20), 
                [exponent, contraction])) |> rstrip)
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
    cen = round.(centerOf(bf), sigdigits=15)
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
the `String` if intent. `genBFuncsFromText` can't directly read center coordinate 
information from the String even if it's included, so argument `center` is used to assign a 
coordinate for all the basis functions from the String; it can be a `Vector`, a `Tuple` of 
the positional `ParamBox`s, or simply (in default) set to `missing` for later assignment.
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
    mapName = pb.map[] |> nameof
    dvr = Symbolics.variable(mapName, T=Symbolics.FnType{Tuple{Any}, Real})(vr)
    expr = pb.map[](vr)
    res = Pair[vr => pb[]]
    includeMapping && !(pb.map[] isa typeof(itself)) && 
        (pushfirst!(res, dvr=>expr, expr=>pb()) |> unique!)
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
fgo0(x, y, z, Rx, Ry, Rz, α, d, i, j, k, N=1.0) = cgo0(x-Rx, y-Ry, z-Rz, α, d, i, j, k, N)


pgf(r, α) = pgf0(r[1], r[2], r[3], α)
cgf(r, α, d) = cgf0(r[1], r[2], r[3], α, d)
cgo(r, α, d, l, N=Nijkα(i,j,k,α)) = cgo0(r[1], r[2], r[3], α, d, l[1], l[2], l[3], N)
fgo(r, R, α, d, l, N=Nijkα(i,j,k,α)) = fgo0(r[1], r[2], r[3], R[1], R[2], R[3], 
                                            α, d, l[1], l[2], l[3], N)
cgo2(r, α, d, i, j, k, N=Nijkα(i,j,k,α)) = cgo0(r[1], r[2], r[3], α, d, i, j, k, N)
fgo2(r, R, α, d, i, j, k, N=Nijkα(i,j,k,α)) = fgo0(r[1], r[2], r[3], R[1], R[2], R[3], 
                                                   α, d, i, j, k, N)


normOfGTOin(b::FloatingGTBasisFuncs{S, GN, 1})  where {S, GN} = 
Nijkα.(ijkOrbitalList[b.ijk[1]]..., [g.xpn() for g in b.gauss])

normOfGTOin(b::FloatingGTBasisFuncs{S, GN, ON}) where {S, GN, ON} = 
Nlα.(b.subshell, [g.xpn() for g in b.gauss])


function expressionOfCore(bf::FloatingGTBasisFuncs; substituteValue::Bool=false, 
                          onlyParameter::Bool=false, expand::Bool=false)
    if bf.normalizeGTO
        N = (bf isa BasisFunc) ? Nijkα : (i,j,k,α) -> Nlα(i+j+k, α)
    else
        N = (_...) -> 1
    end
    nOrbital = bf.ijk |> length
    nGaussFunc = bf.gauss |> length
    res = Num[]
    if expand
        f1 = (x, y) -> append!(x, y)
    else
        f1 = (x, y) -> push!(x, sum(y))
    end
    includeMapping = true
    index = substituteValue ? 2 : 1
    R = [getVar(bf.center[1]; includeMapping)[1][index], 
         getVar(bf.center[2]; includeMapping)[1][index], 
         getVar(bf.center[3]; includeMapping)[1][index]]
    r = Symbolics.variable.(:r, [1:3;])
    f2 = onlyParameter ? (α, d, i, j, k)->cgo2(-R, α, d, i, j, k, N(i,j,k,α)) : 
                         (α, d, i, j, k)->fgo2(r, R, α, d, i, j, k, N(i,j,k,α))
    for ijk in bf.ijk
        i, j, k = ijkOrbitalList[ijk]
        gfs = Num[]
        for g in bf.gauss
            α = getVar(g.xpn; includeMapping)[1][index]
            d = getVar(g.con; includeMapping)[1][index]
            push!(gfs, f2(α, d, i, j, k))
        end
        f1(res, gfs)
    end
    expand ? reshape(res, (nGaussFunc, nOrbital)) : (res |> transpose |> Array)
end

function expressionOfCore(bfm::BasisFuncMix; substituteValue::Bool=false, 
                          onlyParameter::Bool=false, expand::Bool=false)
    exprs = [expressionOfCore(bf; substituteValue, onlyParameter, expand)
             for bf in bfm.BasisFunc]
    expand ? vcat(exprs...) : sum(exprs)
end


"""

    expressionOf(bf::CompositeGTBasisFuncs; 
                 substituteValue::Bool=false, expand::Bool=false) -> 
    Array{<:Symbolics.Num, 2}

Return the expression(s) of a given `CompositeGTBasisFuncs` (e.g. `BasisFuncMix` or 
`FloatingGTBasisFuncs`) as a `Matrix{<:Symbolics.Num}`of which the column(s) corresponds to 
different orbitals. If `substituteValue` is `true`, the variables inside each expression 
will be substituted by their values. If `expand` is `true`, the column(s) will be expanded 
such that its entries are `GaussFunc` inside the corresponding orbital.
"""
expressionOf(bf::CompositeGTBasisFuncs; substituteValue::Bool=false, expand::Bool=false) = 
expressionOfCore(bf; substituteValue, expand, onlyParameter=false)

"""

    expressionOf(gf::GaussFunc; substituteValue::Bool=false) -> 
    Symbolics.Num

Return the expression of a given `GaussFunc`. If `substituteValue` is `true`, the variables 
inside the expression will be substituted by their values.
"""
function expressionOf(gf::GaussFunc; substituteValue::Bool=false)
    r = Symbolics.variable.(:r, [1:3;])
    includeMapping = true
    index = substituteValue ? 2 : 1
    cgf(r, getVar(gf.xpn; includeMapping)[1][index], 
           getVar(gf.con; includeMapping)[1][index])
end


#! Optimize
function shift(bf::FloatingGTBasisFuncs{S, GN, 1}; ijkShift::Vector{Int}, 
                      conRatio::Vector{<:Real}, fixNorm::Bool=false) where {S, GN}
    @assert ijkShift |> length == 3 "The length of `ijkShift` should be 3."
    gfs = bf.gauss |> deepcopy
    @assert length(conRatio) == length(gfs)
    normalizeGTO = bf.normalizeGTO
    if fixNorm && normalizeGTO
        normalizeGTO = false
        conRatio .*= normOfGTOin(bf)
        for i in gfs
            i.con.map[] = itself
        end
    end
    for (i,j) in zip(gfs, conRatio)
        i.con[] *= j
    end
    BasisFunc(bf.center, gfs, ijkOrbitalList[bf.ijk[1]] + ijkShift, normalizeGTO)
end

shift(bf::FloatingGTBasisFuncs{S, 1, 1}, shiftInfo::Vector{<:Real}; 
             fixNorm::Bool=false) where {S} =
shift(bf, ijkShift=shiftInfo[2:end].|>Int, conRatio=[shiftInfo[1]], fixNorm=fixNorm)


function diffInfoToBasisFunc(bf::FloatingGTBasisFuncs, info::Matrix{<:Any})
    bs = decomposeBasisFunc(bf, splitGaussFunc=true)
    bfs = [shift.(Ref(bf), shiftInfo, fixNorm=true) for (shiftInfo, bf) in zip(info, bs)]
    [i |> collect |> flatten for i in eachcol(bfs)] .|> BasisFuncMix
end


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
                sign = iseven(i.exp) ? 1 : -1 # (-X)^k -> [k, 0, 0, 0]
                return (true, sign, vec)
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


function diffInfo(exprs::Array{Num, N}, vr, varDict) where {N}
    relDiffs = Symbolics.simplify.(Symbolics.derivative.(log.(exprs), vr), expand=true)
    diffTransfer.(relDiffs, Ref(varDict))
end