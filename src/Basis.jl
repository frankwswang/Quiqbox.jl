export GaussFunc, BasisFuncMix, mergeBasisFuncs, genBFuncsFromText, genBasisFuncText, 
       BasisFunc, BasisFuncs, genBasisFunc, BasisFuncMix, GTBasis, changeCenter, 
       basisSize, getParams, uniqueParams!, getVar, getVars, expressionOf, assignCenter!

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

    GaussFunc(xpn::Real, con::Real) -> GaussFunc

Generate a `GaussFunc` given the specified exponent and contraction coefficient.

≡≡≡ Example(s) ≡≡≡

```
julia> GaussFunc(5.0, 1.0)
GaussFunc(xpn=ParamBox{:α, Float64}(5.0)[α][∂], con=ParamBox{:d, Float64}(1.0)[d][∂])
```
"""
struct GaussFunc <: AbstractGaussFunc
    xpn::ParamBox{ParamList[:xpn], Float64}
    con::ParamBox{ParamList[:con], Float64}
    param::Tuple{ParamBox{ParamList[:xpn], Float64}, ParamBox{ParamList[:con], Float64}}
    function GaussFunc(e::Real, c::Real)
        xpn = ParamBox(e, ParamList[:xpn]) 
        con = ParamBox(c, ParamList[:con])
        new(xpn, con, (xpn, con))
        # new(xpn, con) 
    end
end


"""

    BasisFunc{S, GN} <: FloatingGTBasisFunc{S, GN, 1}

A (floating) basis function with the center attached to it instead of any nucleus.

≡≡≡ Field(s) ≡≡≡

`center::NTuple{3, ParamBox}`: The center coordinate in form of a 3-element `ParamBox`-type `Tuple`.

`gauss::NTuple{N, GaussFunc}`: Gaussian functions within the basis function.

`subshell::String`: The subshell (angular momentum symbol).

`ijk::Tuple{String}`: Cartesian representation (pseudo-quantum number) of the angular momentum orientation. 
E.g., s would be ("X⁰Y⁰Z⁰")

`normalizeGTO::Bool`: Whether the GTO`::GaussFunc` will be normailzed in calculations.

`param::Tuple{Vararg{<:ParamBox}}`： All the tunable parameters`::ParamBox` stored in the `BasisFunc`.
"""
struct BasisFunc{S, GN} <: FloatingGTBasisFunc{S, GN, 1}
    center::Tuple{ParamBox{<:Any, Float64}, ParamBox{<:Any, Float64}, ParamBox{<:Any, Float64}}
    gauss::NTuple{GN, GaussFunc}
    subshell::String
    ijk::Tuple{String}
    normalizeGTO::Bool
    param::Tuple{Vararg{<:ParamBox}}

    function BasisFunc(cen::Tuple{Vararg{<:ParamBox}}, gs::Vector{<:GaussFunc}, 
                       ijk::Vector{Int}, normalizeGTO::Bool)
        @assert prod(length(ijk) == 3) "The length of `ijk` should be 3."
        subshell = SubshellNames[sum(ijk)+1]
        pars = ParamBox[]
        append!(pars, cen)
        for g in gs
            append!(pars, g.param)
        end
        new{Symbol(subshell), length(gs)}(cen, gs|>Tuple, subshell, (ijkStringList[ijk],), normalizeGTO, pars |> Tuple)
    end
end


"""

    BasisFuncs{S, GN, ON} <: FloatingGTBasisFunc{S, GN, ON}

A group of basis functions with identical parameters except they have different subshell under the specified 
agular momentum. It has the same fields as `BasisFunc` and specifically, for `ijk`, instead of being a 
1-element `Tuple`, the size of the `Tuple` is the size of the corresponding subshell.

"""
struct BasisFuncs{S, GN, ON} <: FloatingGTBasisFunc{S, GN, ON}
    center::Tuple{ParamBox{<:Any, Float64}, ParamBox{<:Any, Float64}, ParamBox{<:Any, Float64}}
    gauss::NTuple{GN, GaussFunc}
    subshell::String
    ijk::Tuple{Vararg{String}}
    normalizeGTO::Bool
    param::Tuple{Vararg{<:ParamBox}}

    function BasisFuncs(cen::Tuple{Vararg{<:ParamBox}}, gs::Vector{<:GaussFunc}, 
                        ijks::Vector{Vector{Int}}, normalizeGTO::Bool=false)
        @assert prod(length.(ijks) .== 3) "The length of each `ijk` should be 3."
        ls = sum.(ijks)
        @assert prod(ls .== ls[1]) "The total angular momentums (of each ijk) should be the same."
        subshell = SubshellNames[ls[1]+1]
        ss = SubshellDimList[subshell]
        @assert length(ijks) <= ss  "The total number of `ijk` should be no more than $(ss) as they are in $(subshell) subshell."
        ijks = sort(ijks, rev=true)
        ijkStrs = [ijkStringList[i] for i in ijks] |> Tuple
        pars = ParamBox[]
        append!(pars, cen)
        for g in gs
            append!(pars, g.param)
        end
        new{Symbol(subshell), length(gs), length(ijks)}(cen, gs |> Tuple, subshell, ijkStrs, normalizeGTO, pars |> Tuple)
    end
end


"""

    genBasisFunc(args..., kws...) -> BasisFunc
    genBasisFunc(args..., kws...) -> BasisFuncs
    genBasisFunc(args..., kws...) -> collection

Constructoer of `BasisFunc` and `BasisFuncs`, but it also returns different kinds of collections of them based on 
the applied methods.

≡≡≡ Method 1 ≡≡≡

    genBasisFunc(coord::AbstractArray, gs::Array{<:GaussFunc, 1}, ijkOrijks::Union{Array{Int, 1}, Array{Array{Int, 1}, 1}}; normalizeGTO::Bool=false)

`ijkOrijks` is the Array of the pseudo-quantum number(s) to specify the angular momentum(s). E.g., s is [0,0,0] and p is [[1,0,0], [0,1,0], [0,0,1]].

≡≡≡ Example(s) ≡≡≡

    julia> genBasisFunc([0,0,0], GaussFunc(2,1), [0,1,0])
    BasisFunc{:P, 1}(gauss, subshell, center)[X⁰Y¹Z⁰][0.0, 0.0, 0.0]

≡≡≡ Method 2 ≡≡≡
    
    genBasisFunc(coord::AbstractArray, gs::Array{<:GaussFunc, 1}, subshell::String="S"; ijkFilter::Array{Bool, 1}=fill(true, SubshellDimList[subshell]), normalizeGTO::Bool=false)

≡≡≡ Example(s) ≡≡≡

    julia> genBasisFunc([0,0,0], GaussFunc(2,1), "S")
    BasisFunc{:S, 1}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]

    julia> genBasisFunc([0,0,0], GaussFunc(2,1), "P")
    BasisFuncs{:P, 1, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]

≡≡≡ Method 3 ≡≡≡

    genBasisFunc(coord::AbstractArray, gExpsANDgCons::NTuple{2, Array{<:Real, 1}}, subshell="S"; kw...)

Instead of directly inputting `GaussFunc`, one can also input a 2-element `Tuple` of the exponent(s) and 
contraction coefficient(s) corresponding to the same `GaussFunc`(s).

≡≡≡ Example(s) ≡≡≡

    julia> genBasisFunc([0,0,0], (2, 1), "P")
    BasisFuncs{:P, 1, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]

    julia> genBasisFunc([0,0,0], ([2, 1.5], [1, 0.5]), "P")
    BasisFuncs{:P, 2, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]

≡≡≡ Method 4 ≡≡≡

    genBasisFunc(center, BSKeyANDnuc::Array{Tuple{String, String}, 1})

If the user wants to construct existed atomic basis set(s), they can use the (`Array` of) `(BS_name, Atom_name)` 
as the second input. If the atom is omitted, then basis set for H is used.

≡≡≡ Example(s) ≡≡≡

    julia> genBasisFunc([0,0,0], ("STO-3G", "Li"))
    3-element Vector{Quiqbox.FloatingGTBasisFunc}:
    BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
    BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
    BasisFuncs{:P, 3, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]

    julia> genBasisFunc([0,0,0], "STO-3G")
    1-element Vector{Quiqbox.FloatingGTBasisFunc}:
    BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]

    julia> genBasisFunc([0,0,0], ["STO-2G", "STO-3G"])
    2-element Vector{Quiqbox.FloatingGTBasisFunc}:
    BasisFunc{:S, 2}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
    BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]

    julia> genBasisFunc([0,0,0], [("STO-2G", "He"), ("STO-3G", "O")])
    4-element Vector{Quiqbox.FloatingGTBasisFunc}:
    BasisFunc{:S, 2}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
    BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
    BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰][0.0, 0.0, 0.0]
    BasisFuncs{:P, 3, 3}(gauss, subshell, center)[3/3][0.0, 0.0, 0.0]
"""
genBasisFunc(cen::Tuple{Vararg{<:ParamBox}}, gs::Vector{<:GaussFunc}, 
          ijk::Vector{Int}; normalizeGTO::Bool=false) = 
BasisFunc(cen, gs, ijk, normalizeGTO)

genBasisFunc(cen::Tuple{Vararg{<:ParamBox}}, gs::Vector{<:GaussFunc}, 
          ijks::Vector{Vector{Int}}; normalizeGTO::Bool=false) = 
BasisFuncs(cen, gs, ijks, normalizeGTO)

# ijkOrijks::Union{Vector{Int}, Vector{Vector{Int}}}
function genBasisFunc(coord::AbstractArray, gs::Vector{<:GaussFunc}, ijkOrijks::Array; normalizeGTO::Bool=false)
    @assert length(coord) == 3 "The dimension of the center should be 3."
    x = ParamBox(coord[1], ParamList[:X])
    y = ParamBox(coord[2], ParamList[:Y])
    z = ParamBox(coord[3], ParamList[:Z])
    genBasisFunc((x,y,z), gs, ijkOrijks; normalizeGTO)
end

genBasisFunc(::Missing, gs::Vector{<:GaussFunc}, ijkOrijks::Array; normalizeGTO::Bool=false) = 
genBasisFunc([NaN, NaN, NaN], gs, ijkOrijks; normalizeGTO)

# center::Union{AbstractArray, Tuple{Vararg{<:ParamBox}}, Missing}
function genBasisFunc(center, gs::Vector{<:GaussFunc}, subshell::String="S"; 
                   ijkFilter::Vector{Bool}=fill(true, SubshellDimList[subshell]), normalizeGTO::Bool=false)
    ijkLen = length(ijkFilter)
    subshellSize = SubshellDimList[subshell]
    @assert ijkLen == subshellSize "The length of `ijkFilter` should be $(subshellSize) to match the subshell's size."
    ijks = [SubshellSuborderList[subshell][i] for i in findall(x->x==true, ijkFilter)]
    length(ijks) == 1 && (ijks = ijks[])
    genBasisFunc(center, gs, ijks; normalizeGTO)
end

# ijkOrSubshell::Union{Array, String}
function genBasisFunc(center, gExpsANDgCons::NTuple{2, Vector{<:Real}}, ijkOrSubshell="S"; kw...)
    @compareLength gExpsANDgCons[1] gExpsANDgCons[2] "exponents" "contraction coefficients"
    gs = GaussFunc.(gExpsANDgCons[1], gExpsANDgCons[2])
    genBasisFunc(center, gs, ijkOrSubshell; kw...)
end

genBasisFunc(center, gExpANDgCon::NTuple{2, Real}, ijkOrSubshell ="S"; kw...) = 
genBasisFunc(center, ([gExpANDgCon[1]], [gExpANDgCon[2]]), ijkOrSubshell; kw...)

function genBasisFunc(center, BSKeyANDnuc::Vector{Tuple{String, String}})
    bases = FloatingGTBasisFunc[]
    for k in BSKeyANDnuc
        BFMcontent = BasisSetList[k[1]][AtomicNumberList[k[2]]]
        append!(bases, genBFuncsFromText(BFMcontent; adjustContent=true, excludeLastNlines=1, center))
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
genBasisFunc(bf::FloatingGTBasisFunc) = itself(bf)
genBasisFunc(bs::Vector{<:FloatingGTBasisFunc}) = sortBasisFuncs(bs)


struct GTBasis{N, BT} <: BasisSetData{N}
    basis::Vector{<:AbstractFloatingGTBasisFunc}
    S::Matrix{<:Number}
    Te::Matrix{<:Number}
    eeI::Array{<:Number, 4}
    getVne::Function
    getHcore::Function

    function GTBasis(basis::Vector{<:AbstractFloatingGTBasisFunc},
                     S::Matrix{<:Number}, Te::Matrix{<:Number}, eeI::Array{<:Number, 4})
        new{basisSize(basis) |> sum, typeof(basis)}(basis, S, Te, eeI, 
            (mol, nucCoords)->dropdims(nucAttractions(basis, mol, nucCoords), dims=3),
            (mol, nucCoords)->dropdims(nucAttractions(basis, mol, nucCoords), dims=3) + Te)
    end
end

GTBasis(basis::Vector{<:AbstractFloatingGTBasisFunc}) = 
GTBasis(basis, dropdims(overlaps(basis), dims=3), 
               dropdims(elecKinetics(basis), dims=3), 
               dropdims(eeInteractions(basis), dims=5))


function sortBasisFuncs(bs::Vector{<:FloatingGTBasisFunc}; groupCenters::Bool=false)
    cens = centerOf.(bs)
    bfBlocks = Vector{<:FloatingGTBasisFunc}[]
    mark, _ = markUnique(cens, compareFunction=isequal)
    m = max(mark...)
    for i=1:m
        idx = findall(x->x==i, mark)
        subbs = bs[idx]
        SubShells = [i.subshell for i in subbs]
        sortVec = sortperm([SubshellNumberList[i] for i in SubShells])
        push!(bfBlocks, subbs[sortVec])
    end
    groupCenters ? bfBlocks : (bfBlocks |> flatten)
end


function isFull(bfs::FloatingGTBasisFunc)
    bfs.subshell == "S" || length(bfs.ijk) == SubshellDimList[bfs.subshell]
end
isFull(::Any) = false


function ijkIndex(b::FloatingGTBasisFunc)
    isFull(b) && (return :)
    [ijkIndexList[ijk] for ijk in b.ijk]
end


function centerOf(bf::FloatingGTBasisFunc)
    [i() for i in bf.center]
end


# Treated as one basis Function
struct BasisFuncMix{ON} <: AbstractFloatingGTBasisFunc
    BasisFunc::NTuple{ON, FloatingGTBasisFunc}
    param::Tuple{Vararg{<:ParamBox}}
    function BasisFuncMix(bfs::Vector{<:FloatingGTBasisFunc{S, GN, 1} where {S, GN}})
        pars = ParamBox[]
        for bf in bfs
            append!(pars, bf.param)
        end
        new{length(bfs)}(bfs |> Tuple, pars |> Tuple)
    end
end
BasisFuncMix(bf::BasisFunc) = BasisFuncMix([bf])
BasisFuncMix(bfs::BasisFuncs) = BasisFuncMix.(decomposeBasisFunc(bfs))
BasisFuncMix(bfm::BasisFuncMix) = itself(bfm)


getBasisFuncs(bfm::BasisFuncMix) = bfm.BasisFunc
getBasisFuncs(bf::FloatingGTBasisFunc) = (bf,)
getBasisFuncs(::Any) = ()


function decomposeBasisFunc(bf::FloatingGTBasisFunc; splitGaussFunc::Bool=false)
    cen = bf.center
    res = BasisFunc[]
    nRow = 1
    if splitGaussFunc
        for ijk in bf.ijk, g in bf.gauss
            push!(res, BasisFunc(cen, [g], ijkOrbitalList[ijk], bf.normalizeGTO))
        end
        nRow = bf.gauss |> length
    else
        for ijk in bf.ijk
            push!(res, BasisFunc(cen, bf.gauss|>collect, ijkOrbitalList[ijk], bf.normalizeGTO))
        end
    end
    reshape(res, (nRow, bf.ijk |> length))
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
basisSize(basis::FloatingGTBasisFunc) = (basis.ijk |> length,)
basisSize(::BasisFuncMix) = (1,)
basisSize(basisSet::Vector{<:Any}) = basisSize.(basisSet) |> flatten |> Tuple


# Core function to generate a customized X-Gaussian (X>1) basis function.
function genGaussFuncText(exponent::Real, contraction::Real)
    """
         $(join(map(x -> rpad(round(x, sigdigits=10)|>alignSignedNum, 20), [exponent, contraction])) |> rstrip)
    """
end


function genBasisFuncText(bf::FloatingGTBasisFunc; norm=1.0, printCenter=true)
    gauss = bf.gauss |> collect
    GFs = map(x -> genGaussFuncText(x.xpn[], x.con[]), gauss)
    cen = round.(centerOf(bf), sigdigits=15)
    firstLine = printCenter ? "X   "*rpad(cen[1]|>alignSignedNum, 20)*
                                     rpad(cen[2]|>alignSignedNum, 20)*
                                     rpad(cen[3]|>alignSignedNum, 20)*"\n" : ""
    firstLine*"$(bf.subshell)    $(bf.gauss |> length)   $(norm)\n" * (GFs |> join)
end

function genBasisFuncText(bs::Vector{<:FloatingGTBasisFunc}; norm=1.0, printCenter=true, groupCenters=true)
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


function genBFuncsFromText(content::String;
                           adjustContent::Bool=false,
                           adjustFunction::F=sciNotReplace, 
                           excludeFirstNlines=0, excludeLastNlines=0, 
                           center::Union{AbstractArray, 
                                         Tuple{Vararg{<:ParamBox}}, 
                                         Missing}=missing) where {F<:Function}
    adjustContent && (content = adjustFunction(content))
    lines = split.(content |> IOBuffer |> readlines)[1+excludeFirstNlines : end-excludeLastNlines]
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
            append!(bfs, genBasisFunc.(Ref(center), [gs1, gs2], ["S", "P"], normalizeGTO=true))
        else
            for j = i+1 : i+ng
                push!(gs1, GaussFunc(data[j]...))
            end
            push!(bfs, genBasisFunc(center, gs1, (data[i][1] |> string), normalizeGTO=true))
        end
    end
    bfs |> flatten
end


function assignCenter!(center::AbstractArray, b::FloatingGTBasisFunc)
    for (i,j) in zip(b.center, center)
        i[] = j
    end
    b.center
end


function getParams(pb::ParamBox, symbol::Union{Symbol, Nothing}=nothing; onlyDifferentiable::Bool=false)
    !(onlyDifferentiable ? pb.canDiff[] : true) && (return nothing)
    !(symbol === nothing ? true : (symbol == typeof(pb).parameters[1])) && (return nothing)
    pb
end

function getParams(pbs::Vector{<:ParamBox}, symbol::Union{Symbol, Nothing}=nothing; onlyDifferentiable::Bool=false)
    res = symbol === nothing ? ParamBox[] : ParamBox{symbol}[]
    for i in pbs
     j = getParams(i, symbol; onlyDifferentiable)
     j !== nothing && push!(res, j)
    end
    res
end

getParams(gf::GaussFunc, symbol::Union{Symbol, Nothing}=nothing; onlyDifferentiable::Bool=false) = getParams([gf.xpn, gf.con], symbol; onlyDifferentiable)
getParams(bf::FloatingGTBasisFunc, symbol::Union{Symbol, Nothing}=nothing; onlyDifferentiable::Bool=false) = vcat( getParams.(bf.gauss, symbol; onlyDifferentiable)..., getParams(bf.center |> collect, symbol; onlyDifferentiable) )
getParams(ds, symbols::Vector{Symbol}; onlyDifferentiable::Bool=false) = getParams.(Ref(ds), symbols; onlyDifferentiable) |> flatten
getParams(ds::Array, symbol::Union{Symbol, Nothing}=nothing; onlyDifferentiable::Bool=false) = getParams.(ds, symbol; onlyDifferentiable) |> flatten


function markParams!(parArray::Vector{<:ParamBox{V}}; 
                     ignoreContainerType::Bool=false, filter::Bool=true, ignoreMapping::Bool=false) where {V}
    res, _ = markUnique(parArray, compareFunction=hasIdentical, ignoreFunction=true; ignoreContainerType)
    for i=1:length(parArray)
        parArray[i].index = res[i]
    end
    if filter
        _, cmprList = markUnique(parArray, compareFunction=hasIdentical, ignoreFunction=ignoreMapping; ignoreContainerType)
        return cmprList
    end
    parArray
end

function markParams!(parArray::Vector{<:ParamBox}; 
                     ignoreContainerType::Bool=false, filter::Bool=true, ignoreMapping::Bool=false)
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
        append!(pars, markParams!(subArr; ignoreContainerType, filter, ignoreMapping))
    end
    pars
end


function uniqueParams!(bs; onlyDifferentiable::Bool=false, ignoreContainerType::Bool=false, filter::Bool=true, ignoreMapping::Bool=false)
    markParams!(getParams(bs; onlyDifferentiable); ignoreContainerType, filter, ignoreMapping)
end


function getVar(pb::ParamBox; markUndifferentiable::Bool=false, includeMapping::Bool=false)
    varName = typeof(pb).parameters[1]
    superscript = (pb.canDiff[] == true || !markUndifferentiable) ? "" : NoDiffMark
    varSymbol = Symbol((varName |> string) * superscript)
    vr = (pb.index isa Int) ? Symbolics.variable(varSymbol, pb.index) : Symbolics.variable(varName)
    mapName = pb.map[] |> nameof
    dvr = Symbolics.variable(mapName, T=Symbolics.FnType{Tuple{Any}, Real})(vr)
    expr = pb.map[](vr)
    res = Pair[vr => pb[]]
    includeMapping && !(pb.map[] isa typeof(itself)) && (pushfirst!(res, dvr=>expr, expr=>pb()) |> unique!)
    res
end

getVar(pbType::Type{<:ParamBox}) = Symbolics.variable(pbType.parameters[1])


getVars(gf::GaussFunc; markUndifferentiable::Bool=false, includeMapping::Bool=false) = 
getVar.(gf.param; markUndifferentiable, includeMapping) |> flatten |> Dict

getVars(bf::BasisFunc; markUndifferentiable::Bool=false, includeMapping::Bool=false) = 
getVar.(bf.param; markUndifferentiable, includeMapping) |> flatten |> Dict

getVars(pbs::Vector{<:ParamBox}; markUndifferentiable::Bool=false, includeMapping::Bool=false) = 
getVar.(pbs; markUndifferentiable, includeMapping) |> flatten |> Dict

getVars(fs::Vector{<:Union{GaussFunc, BasisFunc}}; markUndifferentiable::Bool=false, includeMapping::Bool=false) = 
merge(getVars.(fs; markUndifferentiable, includeMapping)...)


function Nlα(l, α)
    if l < 2
        ( ( 2^(2l+3) * factorial(l+1) * 2^(l+1.5) / (factorial(2l+2) * √π) ) |> sqrt) * α^(0.5l + 0.75)
    else
        # for higher angular momentum make the upper bound of norms be 1.
        ( ( 2^(3l+1.5) * factorial(l) / (factorial(2l) * π^1.5) ) |> sqrt ) * α^(0.5l + 0.75)
    end
end

Nlα(subshell::String, α) = Nlα(SubshellNumberList[subshell], α)


Nijk(i, j, k) = (2/π)^0.75 * ( 2^(3*(i+j+k)) * factorial(i) * factorial(j) * factorial(k) / (factorial(2i) * factorial(2j) * factorial(2k)) )^0.5


function Nijkα(i, j, k, α)
    l = i + j + k
    if l < 2
        ( ( 2^(2l+3) * factorial(l+1) * 2^(l+1.5) / (factorial(2l+2) * √π) ) |> sqrt ) * α^(0.5l + 0.75)
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
fgo(r, R, α, d, l, N=Nijkα(i,j,k,α)) = fgo0(r[1], r[2], r[3], R[1], R[2], R[3], α, d, l[1], l[2], l[3], N)
cgo2(r, α, d, i, j, k, N=Nijkα(i,j,k,α)) = cgo0(r[1], r[2], r[3], α, d, i, j, k, N)
fgo2(r, R, α, d, i, j, k, N=Nijkα(i,j,k,α)) = fgo0(r[1], r[2], r[3], R[1], R[2], R[3], α, d, i, j, k, N)


normOfGTOin(b::FloatingGTBasisFunc{S, GN, 1})  where {S, GN} = Nijkα.(ijkOrbitalList[b.ijk[1]]..., [g.xpn() for g in b.gauss])
normOfGTOin(b::FloatingGTBasisFunc{S, GN, ON}) where {S, GN, ON} = Nlα.(b.subshell, [g.xpn() for g in b.gauss])


function expressionOf(gf::GaussFunc; markUndifferentiable::Bool=false, substituteValue::Bool=false)
    r = Symbolics.variable.(:r, [1:3;])
    includeMapping = true
    index = substituteValue ? 2 : 1
    cgf(r, getVar(gf.xpn; markUndifferentiable, includeMapping)[1][index], 
           getVar(gf.con; markUndifferentiable, includeMapping)[1][index])
end

function expressionOf(bf::FloatingGTBasisFunc; markUndifferentiable::Bool=false, substituteValue::Bool=false, 
                      onlyParameter::Bool=false, expand::Bool=false)
    if bf.normalizeGTO
        N = (bf isa BasisFunc) ? Nijkα : (i,j,k,α) -> Nlα(i+j+k, α)
    else
        N = (_...)->1
    end
    lSize = bf.ijk |> length
    gSize = bf.gauss |> length
    res = Num[]
    if expand
        f1 = (x, y) -> append!(x, y)
    else    
        f1 = (x, y) -> push!(x, sum(y))
    end
    includeMapping = true
    index = substituteValue ? 2 : 1
    R = [getVar(bf.center[1]; markUndifferentiable, includeMapping)[1][index], 
         getVar(bf.center[2]; markUndifferentiable, includeMapping)[1][index], 
         getVar(bf.center[3]; markUndifferentiable, includeMapping)[1][index]]
    r = Symbolics.variable.(:r, [1:3;])
    f2 = onlyParameter ? (α, d, i, j, k)->cgo2(-R, α, d, i, j, k, N(i,j,k,α)) : (α, d, i, j, k)->fgo2(r, R, α, d, i, j, k, N(i,j,k,α))
    for ijk in bf.ijk
        i, j, k = ijkOrbitalList[ijk]
        gfs = Num[]
        for g in bf.gauss
            α = getVar(g.xpn; markUndifferentiable, includeMapping)[1][index]
            d = getVar(g.con; markUndifferentiable, includeMapping)[1][index]
            push!(gfs, f2(α, d, i, j, k))
        end
        f1(res, gfs)
    end
    expand ? reshape(res, (gSize, lSize))  :  res |> transpose |> Array
end

function expressionOf(bfm::BasisFuncMix; markUndifferentiable::Bool=false, substituteValue::Bool=false, 
                      onlyParameter::Bool=false, expand::Bool=false)
    [expressionOf(bf; markUndifferentiable, substituteValue, onlyParameter, expand) for bf in bfm.BasisFunc] |> sum
end


function orbitalShift(bf::FloatingGTBasisFunc{S, GN, 1}; ijkShift::Vector{Int}, conRatio::Vector{<:Real}, fixNorm::Bool=false) where {S, GN}
    @assert ijkShift |> length == 3 "The length of `ijkShift` should be 3."
    gfs = bf.gauss |> collect |> deepcopy
    @assert length(conRatio) == length(gfs)
    normalizeGTO = bf.normalizeGTO
    if fixNorm && normalizeGTO
        normalizeGTO = false
        conRatio .*= normOfGTOin(bf)
        for i in gfs
            i.con.map[] = itself   
        end
    end
    for (i,j) in zip(gfs,conRatio)
        i.con[] *= j
    end
    BasisFunc(bf.center, gfs, ijkOrbitalList[bf.ijk[1]] + ijkShift, normalizeGTO)
end

orbitalShift(bf::FloatingGTBasisFunc{S, 1, 1}, shiftInfo::Vector{<:Real}; fixNorm::Bool=false) where {S} =
orbitalShift(bf, ijkShift=shiftInfo[2:end].|>Int, conRatio=[shiftInfo[1]], fixNorm=fixNorm)


function diffInfoToBasisFunc(bf::FloatingGTBasisFunc, info::Matrix{<:Any})
    bs = decomposeBasisFunc(bf, splitGaussFunc=true)
    bfs = [orbitalShift.(Ref(bf), shift, fixNorm=true) for (shift, bf) in zip(info, bs)]
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
        res = recursivelyGet(varDict, symbolReplace(Symbolics.tosymbol(vr), NoDiffMark=>"") |> Symbolics.variable)
    end
    if res === nothing
        str = Symbolics.tosymbol(vr) |> string
        # pos = findfirst(r"\p{No}", str)[1]
        pos = findfirst(r"[₁₂₃₄₅₆₇₈₉₀]", str)[1]
        front = split(str,str[pos])[1]
        var = front*NoDiffMark*str[pos:end] |> Symbol
        recursivelyGet(varDict, var |> Symbolics.variable)
    end
    @assert res !== nothing "Can NOT find the value of $(vr)::$(typeof(vr)) in the given Dict $(varDict)."
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
            expr = f(dvr.arguments[]) # assumming AD propagates only through 1 var: f(g(x)).
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