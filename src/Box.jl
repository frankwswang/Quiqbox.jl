export gridBoxCoords, GridBox, gridCoordOf

function makeGridFuncsCore(nG::Int)
    res = Array{Function}(undef, nG+1)
    if nG == 0
        res[] = itself
    else
        for i = 0:nG
            funcName = "G$(nG)" * numToSups(i)
            funcSym = Symbol(funcName)
            res[i+1] = if isdefined(Quiqbox, funcSym)
                getproperty(Quiqbox, funcSym)
            else
                renameFunc(funcName, L -> (i - 0.5nG)*L)
            end
        end
    end
    res
end

makeGridFuncs(c, f::F) where {F<:Function} = Sf(c, f)
makeGridFuncs(_, ::itselfT) = itself

makeGridPBoxData(cenCompData::Array{T, 0}, spacingData::Array{T, 0}, nG::Int) where {T} = 
ifelse(nG>0, spacingData, cenCompData)

"""

    GridBox{T, D, NP, GPT<:SpatialPoint{T, D}} <: SpatialStructure{T, D}

A container of multiple `D`-dimensional grid points.

≡≡≡ Field(s) ≡≡≡

`spacing::NTuple{D, T}`: The distance between adjacent grid points along each dimension.

`nPoint::Int`: Total number of the grid points.

`point::NTuple{NP, GPT}`: The grid points represented by [`SpatialPoint`](@ref).

`param::Tuple{Vararg{ParamBox{T}}}`: All the parameters in the `GridBox`.

≡≡≡ Initialization Method(s) ≡≡≡

    GridBox(nGrids::NTuple{D, Int}, spacing::NTuple{D, Union{Array{T, 0}, T}}, 
            center::Union{AbstractVector{T}, NTuple{D, T}}=ntuple(_->T(0),Val(D)); 
            canDiff::NTuple{D, Bool}=ntuple(_->true, Val(D)), 
            index::NTuple{D, Int}=ntuple(_->0, Val(D))) where {T<:AbstractFloat, D} -> 
    GridBox{T, D}

    GridBox(nGrids::NTuple{D, Int}, spacingForAllDim::Union{T, Array{T, 0}}, 
            center::Union{AbstractVector{T}, NTuple{D, T}}=ntuple(_->T(0), Val(D)); 
            canDiff::Bool=true, index::Int=0) where {T<:AbstractFloat, D} -> 
    GridBox{T, D}

Construct a general `D`-dimensional `GridBox`.

=== Positional argument(s) ===

`nGrids::NTuple{D, Int}`: The numbers of grids along each dimension.

`spacing::NTuple{D, Union{Array{T, 0}, T}}`: The spacing between grid points along each 
dimension.

`spacingForAllDim::NTuple{D, Union{Array{T, 0}, T}}`: A single spacing applied for all 
dimensions.

`center::Union{AbstractVector{T}, NTuple{D, T}}`: The coordinate of the geometric center of 
the grid box.

=== Keyword argument(s) ===

`canDiff`::NTuple{D, Bool}: Whether the `ParamBox`es of each dimension stored in the 
constructed `GridBox` will be marked as differentiable.

`index`::NTuple{D, Int}: The Index(s) that will be assigned to the shared input variable(s) 
`$(ParamList[:spacing])` of the stored `ParamBox`es.
"""
struct GridBox{T, D, NP, GPT<:SpatialPoint{T, D}} <: SpatialStructure{T, D}
    spacing::NTuple{D, T}
    nPoint::Int
    point::NTuple{NP, GPT}
    param::Tuple{Vararg{ParamBox{T}}}

    function GridBox(nGrids::NTuple{D, Int}, spacing::NTuple{D, Union{Array{T, 0}, T}}, 
                     center::Union{AbstractVector{T}, NTuple{D, T}}=ntuple(_->T(0),Val(D)); 
                     canDiff::NTuple{D, Bool}=ntuple(_->true, Val(D)), 
                     index::NTuple{D, Int}=ntuple(_->0, Val(D))) where {T<:AbstractFloat, D}
        @assert all(nGrids.>=0) "The number of gird of each edge must be no less than 0."
        @assert length(center)==D "The dimension of center coordinate must be equal to $D."
        NP = prod(nGrids .+ 1)
        iVsym = ParamList[:spacing]
        oVsym = SpatialParamSyms[1:D]
        point = Array{SpatialPoint{T, D}}(undef, NP)
        param = Array{ParamBox{T}}(undef, D*NP)
        funcs = makeGridFuncsCore.(nGrids)
        spacing = fillObj.(spacing)
        data = makeGridPBoxData.(fill.(center), spacing, nGrids)
        for (n, i) in enumerate( CartesianIndices(nGrids .+ 1) )
            fs = makeGridFuncs.(center, [funcs[j][k] for (j, k) in enumerate(i|>Tuple)])
            p = broadcast((a, b, c, d, canDiff, index) -> 
                          ParamBox(a, b, c, d; canDiff, index), 
                          data, oVsym, fs, iVsym, canDiff, index)|>Tuple
            point[n] = SpatialPoint(p)
            param[D*(n-1)+1 : D*n] .= p
        end
        point = Tuple(point)
        new{T, D, NP, eltype(point)}(getindex.(spacing), NP, point, Tuple(param))
    end
end

GridBox(nGrids::NTuple{D, Int}, spacingForAllDim::Union{T, Array{T, 0}}, 
        center::Union{AbstractVector{T}, NTuple{D, T}}=ntuple(_->T(0), Val(D)); 
        canDiff::Bool=true, index::Int=0) where {T<:AbstractFloat, D} = 
GridBox(nGrids, Tuple(fill(fillObj(spacingForAllDim), D)), center; 
        canDiff=Tuple(fill(canDiff, D)), index=Tuple(fill(index, D)))

"""

    GridBox(nGridPerEdge::Int, spacing, center=ntuple(_->eltype(spacing[begin])(0), 3); 
            canDiff::Bool=true, index::Int=0) -> 
    GridBox{T, D}

The method of generating a cubic `GridBox`. Aside from the common arguments, `nGridPerEdge` 
specifies the number of grids for every dimension. The dimension of the grid box is 
determined by the dimension of `center`.
"""
GridBox(nGridPerEdge::Int, spacing, center=ntuple(_->eltype(spacing[begin])(0), 3); 
        canDiff::Bool=true, index::Int=0) = 
GridBox(ntuple(_->nGridPerEdge, length(center)), spacing, center; canDiff, index)


"""

    gridCoordOf(gb::GridBox{T}) where {T} -> Tuple{Vararg{Vector{T}}}

Return the coordinates of the grid points stored in `gb`.
"""
gridCoordOf(gb::GridBox) = coordOf.(gb.point)