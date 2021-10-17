export gridBoxCoords, GridBox, gridCoords

"""

    GridBox{NX, NY, NZ} <: SemiMutableParameter{GridBox, Float64}

A `struct` that stores coordinates of grid points in terms of both `Vector`s and 
`ParamBox`s.

≡≡≡ Field(s) ≡≡≡

`num::Int`: Total number of the grid points.

`spacing::Float64`: The length between adjacent grid points.

`box::Vector{NTuple{3, ParamBox}}`: The coordinates of grid points in terms of `ParamBox`s.

`coord::Array{Array{Float64, 1}, 1}`: The coordinates of grid points in terms of `Vector`s.

≡≡≡ Initialization Method(s) ≡≡≡

    GridBox(nGrids::NTuple{3, Int}, spacing::Real=10, 
            centerCoord::Array{<:Real, 1}=[0.0,0.0,0.0];
            canDiff::Bool=true, index::Int=0) -> GridBox

Constructor of a general `GridBox` that doesn't have to shape as a cube. `nGrid` is a 
3-element `Tuple` that specifies the number of grids (number of grid points - 1) along 
3 dimensions. `spacing` specifies the length between adjacent grid points. 
`centerCoord` specifies the geometry center coordinate of the box. `canDiff` determines 
whether the `ParamBox` should be marked as differentiable. `index` defines the index 
number for the actual parameter: spacing `L`, with the default value 0 it would be `L₀`.
"""
struct GridBox{NX, NY, NZ} <: SemiMutableParameter{GridBox, Float64}
    num::Int
    spacing::Float64
    box::Vector{NTuple{3, ParamBox}}

    function GridBox(nGrids::NTuple{3, Int}, spacing::Real=10, 
                     centerCoord::Vector{<:Real}=[0.0,0.0,0.0];
                     canDiff::Bool=true, index::Int=0)
        @assert prod(nGrids .> 0) "The number of gird of each edge should be larger than 0."
        sym = ParamList[:spacing]
        spc = spacing |> Float64
        pbRef = ParamBox(spc; canDiff, index)
        boxes = NTuple{3, ParamBox{Float64}}[]
        n = 0
        supIndex = "ᴳ"*numToSups(nGrids[1])*superscriptSym['-']*numToSups(nGrids[2])*
                   superscriptSym['-']*numToSups(nGrids[3])
        for i=0:nGrids[1], j=0:nGrids[2], k=0:nGrids[3]
            n += 1
            fX0 = L -> centerCoord[1] + (i - 0.5*nGrids[1]) * L
            fY0 = L -> centerCoord[2] + (j - 0.5*nGrids[2]) * L
            fZ0 = L -> centerCoord[3] + (k - 0.5*nGrids[3]) * L
            fXname = (ParamList[:X] |> string) * supIndex * numToSubs(n)
            fYname = (ParamList[:Y] |> string) * supIndex * numToSubs(n)
            fZname = (ParamList[:Z] |> string) * supIndex * numToSubs(n)
            fX = renameFunc(fXname, fX0)
            fY = renameFunc(fYname, fY0)
            fZ = renameFunc(fZname, fZ0)
            X = ParamBox(pbRef.data, ParamList[:X], fX, sym; canDiff, index)
            Y = ParamBox(pbRef.data, ParamList[:Y], fY, sym; canDiff, index)
            Z = ParamBox(pbRef.data, ParamList[:Z], fZ, sym; canDiff, index)
            push!(boxes, (X, Y, Z))
        end
        new{nGrids[1], nGrids[2], nGrids[3]}(prod(nGrids .+ 1), spc, boxes)
    end
end

"""

    GridBox(nGridPerEdge::Int, spacing::Real=10, 
            centerCoord::Array{<:Real, 1}=[0.0,0.0,0.0]; 
            canDiff::Bool=true, index::Int=0) -> GridBox
    
Method of generating a cubic `GridBox`. `nGridPerEdge` specifies the number of grids 
(number of grid points - 1) along each dimension.`spacing` specifies the length between 
adjacent grid points. `centerCoord` specifies the geometry center coordinate of the box. 
`canDiff` determines whether the `ParamBox` should be marked as differentiable. `index` 
defines the index number for the actual parameter: spacing `L`, with the default value 0 
it would be `L₀`.
"""
GridBox(nGridPerEdge::Int, spacing::Real=10, centerCoord::Vector{<:Real}=[0.0,0.0,0.0];
        canDiff::Bool=true, index::Int=0) = 
GridBox(fill(nGridPerEdge, 3) |> Tuple, spacing, centerCoord; canDiff, index)


"""

    gridCoords(gb::GridBox) -> Array{Array{Float64, 1}, 1}

Return the grid-point coordinates in `Vector`s given the `GriBox`.
"""
function gridCoords(gb::GridBox)
    coords = Vector{Float64}[]
    for point in gb.box
        push!(coords, [point[1](), point[2](), point[3]()])
    end
    coords
end