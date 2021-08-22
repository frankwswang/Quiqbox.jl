export gridBoxCoords, GridBox, gridPoint

#===== Grid-based coordinates =====#
struct GridBox{NX, NY, NZ}
    num::Int
    len::Real
    box::Array{NTuple{3, ParamBox}, 1}
    coord::Array{Array{Float64, 1}, 1}

    function GridBox(nGrids::NTuple{3, Int}, edgeLength::Real=10, centerCoord::Array{<:Real,1}=[0.0,0.0,0.0];
                     canDiff::Bool=true, index::Int=0)
        @assert prod(nGrids .> 0) "The number of gird of each edge should be larger than 0."
        sym = ParamList[:len]
        pbRef = ParamBox(edgeLength, sym; canDiff, index)
        boxes = NTuple{3, ParamBox{sym, Float64}}[]
        coords = Array{Float64, 1}[]
        n = 0
        supIndex = "ᴳ"*numToSups(nGrids[1])*superscriptSym['-']*numToSups(nGrids[2])*superscriptSym['-']*numToSups(nGrids[3])
        for i=0:nGrids[1], j=0:nGrids[2], k=0:nGrids[3]
            n += 1
            fX0 = L -> centerCoord[1] + (i / nGrids[1] - 0.5) * L
            fY0 = L -> centerCoord[2] + (j / nGrids[2] - 0.5) * L
            fZ0 = L -> centerCoord[3] + (k / nGrids[3] - 0.5) * L
            fXname = (ParamList[:X] |> string) * supIndex * numToSubs(n)
            fYname = (ParamList[:Y] |> string) * supIndex * numToSubs(n)
            fZname = (ParamList[:Z] |> string) * supIndex * numToSubs(n)
            fX = renameFunc(fXname, fX0)
            fY = renameFunc(fYname, fY0)
            fZ = renameFunc(fZname, fZ0)
            X = ParamBox(pbRef.data, sym, mapFunction = fX)
            Y = ParamBox(pbRef.data, sym, mapFunction = fY)
            Z = ParamBox(pbRef.data, sym, mapFunction = fZ)
            X.canDiff = Y.canDiff = Z.canDiff = pbRef.canDiff
            X.index = Y.index = Z.index = pbRef.index
            push!(boxes, (X, Y, Z))
            push!(coords, [X(), Y(), Z()])
        end
        new{nGrids[1], nGrids[2], nGrids[3]}(prod(nGrids .+ 1), edgeLength, boxes, coords)
    end
end

GridBox(nGridPerEdge::Int, edgeLength::Real=10, centerCoord::Array{<:Real,1}=[0.0,0.0,0.0];
        canDiff::Bool=true, index::Int=0) = 
GridBox(fill(nGridPerEdge, 3) |> Tuple, edgeLength, centerCoord; canDiff, index)


function gridPoint(coord::Array{<:Real,1})
    @assert length(coord) == 3
    x = ParamBox(coord[1], ParamList[:X])
    y = ParamBox(coord[2], ParamList[:Y])
    z = ParamBox(coord[3], ParamList[:Z])
    (x,y,z)
end