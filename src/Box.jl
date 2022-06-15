export gridBoxCoords, GridBox, gridCoords

"""

    GridBox{NP} <: SemiMutableParameter{GridBox, Float64}

A `struct` that stores coordinates of grid points in terms of both `Vector`s and 
`ParamBox`s.

≡≡≡ Field(s) ≡≡≡

`num::Int`: Total number of the grid points.

`spacing::Float64`: The length between adjacent grid points.

`box::NTuple{NP, NTuple{3, ParamBox}}`: The coordinates of grid points.

`coord::Array{Vector{Float64}, 1}`: The coordinates of grid points in terms of `Vector`s.

≡≡≡ Initialization Method(s) ≡≡≡

    GridBox(nGrids::NTuple{3, Int}, spacing::Real=10, 
            centerCoord::Array{<:Real, 1}=[0.0,0.0,0.0];
            canDiff::Bool=true, index::Int=0) -> GridBox

Construct a general `GridBox` that doesn't have to shape as a cube. `nGrid` is a 3-element 
`Tuple` that specifies the number of grids (number of grid points - 1) along 3 dimensions. 
`spacing` specifies the length between adjacent grid points. `centerCoord` specifies the 
geometry center coordinate of the box. `canDiff` determines whether the `ParamBox` should 
be marked as differentiable. `index` defines the index number for the actual parameter: 
spacing `L`, with the default value 0 it would be `L₀`.

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
struct GridBox{NP} <: SemiMutableParameter{GridBox, Float64}
    num::Int
    spacing::Float64
    box::NTuple{NP, 
                Tuple{ParamBox{Float64, :X, FLevel{2, 0}}, 
                      ParamBox{Float64, :Y, FLevel{2, 0}}, 
                      ParamBox{Float64, :Z, FLevel{2, 0}}}}

    function GridBox((nGx, nGy, nGz)::NTuple{3, Int}, spacing::Real=10, 
                     centerCoord::Vector{<:Real}=[0.0,0.0,0.0];
                     canDiff::Bool=true, index::Int=0)
        nGrids = (nGx, nGy, nGz)
        @assert all(nGrids.>=0) "The number of gird of each edge should be no less than 0."
        sym = ParamList[:spacing]
        spc = spacing |> Float64
        pbRef = ParamBox(spc; index)
        boxes = NTuple{3, ParamBox{Float64}}[]
        n = 0
        prefix = "G" * "_" * "$(nGx)" * "_" * "$(nGy)" * "_" * "$(nGz)" * "_"
        for i=0:nGx, j=0:nGy, k=0:nGz
            n += 1
            fX0 = nGx==0  ?  itself  :  L -> centerCoord[1] + (i - 0.5*nGx) * L
            fY0 = nGy==0  ?  itself  :  L -> centerCoord[2] + (j - 0.5*nGy) * L
            fZ0 = nGz==0  ?  itself  :  L -> centerCoord[3] + (k - 0.5*nGz) * L
            fXname = prefix * (cxSym |> string) * numToSubs(n)
            fYname = prefix * (cySym |> string) * numToSubs(n)
            fZname = prefix * (czSym |> string) * numToSubs(n)
            fX = renameFunc(fXname, fX0)
            fY = renameFunc(fYname, fY0)
            fZ = renameFunc(fZname, fZ0)
            X = ParamBox(pbRef.data, cxSym, fX, sym; canDiff, index)
            Y = ParamBox(pbRef.data, cySym, fY, sym; canDiff, index)
            Z = ParamBox(pbRef.data, czSym, fZ, sym; canDiff, index)
            push!(boxes, (X, Y, Z))
        end
        new{(nGx+1)*(nGy+1)*(nGz+1)}(prod(nGrids .+ 1), spc, Tuple(boxes))
    end
end

GridBox(nGridPerEdge::Int, spacing::Real=10, centerCoord::Vector{<:Real}=[0.0,0.0,0.0];
        canDiff::Bool=true, index::Int=0) = 
GridBox(fill(nGridPerEdge, 3) |> Tuple, spacing, centerCoord; canDiff, index)


"""

    gridCoords(gb::GridBox) -> Array{Vector{Float64}, 1}

Return the grid-point coordinates in `Vector`s given the `GriBox`.
"""
gridCoords(gb::GridBox) = [outValOf.(i) |> collect for i in gb.box]