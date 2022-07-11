export gridBoxCoords, GridBox, gridCoords

export GP1D, GP2D, GP3D

# const SPSpatialPoint{T, 1, SPoint{Tuple{ParamBox{T, :X, FLevel{2}}}}}

# SpatialPoint{T, 2, SPoint{Tuple{ParamBox{T, :X, FLevel{2}}, 
#                                 ParamBox{T, :Y, FLevel{2}}}}}

# SpatialPoint{T, 3, SPoint{Tuple{ParamBox{T, :X, FLevel{2}}, 
#                                 ParamBox{T, :Y, FLevel{2}}, 
#                                 ParamBox{T, :Z, FLevel{2}}}}}

const GP1D{T, L} = SP1D{T, FLevel{L}}
const GP2D{T, L} = SP2D{T, FLevel{L}, FLevel{L}}
const GP3D{T, L} = SP3D{T, FLevel{L}, FLevel{L}, FLevel{L}}

getGPT(::Type{T}, ::Val{1}, ::Val{L}) where {T, L} = GP1D{T, L}
getGPT(::Type{T}, ::Val{2}, ::Val{L}) where {T, L} = GP2D{T, L}
getGPT(::Type{T}, ::Val{3}, ::Val{L}) where {T, L} = GP3D{T, L}

# const SPoints{T, D, FL} = Tuple{Vararg{ParamBox{T, V, FL} where {V}, D}}

function makeGridFuncsCore(nG::Int, prefix::String)
    res = Array{Function}(undef, nG+1)
    if nG == 0
        res[] = itself
    else
        for i = 0:nG
            funcName = prefix * "_$(nG)" * numToSubs(i)
            funcSym = Symbol(funcName)
            res[i+1] = if isdefined(Quiqbox, funcSym)
                getfield(Quiqbox, funcSym)
            else
                renameFunc(funcName, L -> (i - 0.5nG)*L)
            end
        end
    end
    res
end

makeGridFuncs(c, f::F) where {F<:Function} = ifelse(c == 0, f, Sf(c, f))
makeGridFuncs(_, f::itselfT) = itself

"""

    GridBox{T, D, NP}

A `struct` that stores coordinates of grid points in terms of both `AbstractVector`s and 
`ParamBox`s.

≡≡≡ Field(s) ≡≡≡

`num::Int`: Total number of the grid points.

`spacing::Float64`: The length between adjacent grid points.

`box::NTuple{NP, NTuple{3, ParamBox}}`: The coordinates of grid points.

`coord::Array{AbstractVector{Float64}, 1}`: The coordinates of grid points in terms of `AbstractVector`s.

≡≡≡ Initialization Method(s) ≡≡≡

    GridBox(nGrids::NTuple{3, Int}, spacing::Real=10, 
            center::Array{<:Real, 1}=[0.0,0.0,0.0];
            canDiff::Bool=true, index::Int=0) -> GridBox

Construct a general `GridBox` that doesn't have to shape as a cube. `nGrid` is a 3-element 
`Tuple` that specifies the number of grids (number of grid points - 1) along 3 dimensions. 
`spacing` specifies the length between adjacent grid points. `center` specifies the 
geometry center coordinate of the box. `canDiff` determines whether the `ParamBox` should 
be marked as differentiable. `index` defines the index number for the actual parameter: 
spacing `L`, with the default value 0 it would be `L₀`.

    GridBox(nGridPerEdge::Int, spacing::Real=10, 
            center::Array{<:Real, 1}=[0.0,0.0,0.0]; 
            canDiff::Bool=true, index::Int=0) -> GridBox

Method of generating a cubic `GridBox`. `nGridPerEdge` specifies the number of grids 
(number of grid points - 1) along each dimension.`spacing` specifies the length between 
adjacent grid points. `center` specifies the geometry center coordinate of the box. 
`canDiff` determines whether the `ParamBox` should be marked as differentiable. `index` 
defines the index number for the actual parameter: spacing `L`, with the default value 0 
it would be `L₀`.
"""
struct GridBox{T, D, NP, GPT} <: SpatialStructure{T, D}
    nPoint::Int
    spacing::T
    box::NTuple{NP, SpatialPoint{T, D, GPT}}
    param::Tuple{Vararg{ParamBox{T}}}

    function GridBox(nGrids::NTuple{D, Int}, spacing::Union{T, Array{T, 0}}, 
                     center::AbstractVector{T}=fill(T(0), D); 
                     canDiff::Bool=true, index::Int=0) where {T<:AbstractFloat, D}
        @assert all(nGrids.>=0) "The number of gird of each edge must be no less than 0."
        @assert length(center)==D "The dimension of center coordinate must be equal to $D."
        NP = prod(nGrids .+ 1)
        sym = ParamList[:spacing]
        data = ifelse(spacing isa AbstractFloat, fill(spacing), spacing)
        GPT = getGPT(T, Val(D), Val(2))
        box = Array{SpatialPoint{T, D}}(undef, NP)
        param = Array{ParamBox{T}}(undef, 3NP)
        nGx, nGy, nGz = nGrids
        # prefix = "G"*numToSups(nGx)*superscriptSym['-']*numToSups(nGy)*superscriptSym['-']*numToSups(nGz)*"_"
        prefix = "G" * "_" * "$(nGx)" * "_" * "$(nGy)" * "_" * "$(nGz)" * "_"
        funcs = makeGridFuncsCore.(nGrids, prefix)
        n = 0
        for i=0:nGx, j=0:nGy, k=0:nGz
            n += 1
            fX = makeGridFuncs(center[1], funcs[1][i+1])
            fY = makeGridFuncs(center[2], funcs[2][j+1])
            fZ = makeGridFuncs(center[3], funcs[3][k+1])
            X = ParamBox(data, cxSym, fX, sym; canDiff, index)
            Y = ParamBox(data, cySym, fY, sym; canDiff, index)
            Z = ParamBox(data, czSym, fZ, sym; canDiff, index)
            p = (X, Y, Z)
            box[n] = SpatialPoint(p)
            param[3n-2:3n] .= p
        end
        new{T, D, NP, GPT}(NP, spacing, Tuple(box), Tuple(param))
    end
end

GridBox(nGridPerEdge::Int, spacing::T=T(1), center::AbstractVector{T}= T[0,0,0];
        canDiff::Bool=true, index::Int=0) where {T} = 
GridBox(fill(nGridPerEdge, 3) |> Tuple, spacing, center; canDiff, index)


"""

    gridCoords(gb::GridBox) -> Array{AbstractVector{Float64}, 1}

Return the grid-point coordinates in `AbstractVector`s given the `GriBox`.
"""
gridCoords(gb::GridBox) = [outValOf.(i) |> collect for i in gb.box]