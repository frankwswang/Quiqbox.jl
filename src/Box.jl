export gridBoxCoords, GridBox, gridCoords

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
makeGridFuncs(_, f::itselfT) = itself

"""

    GridBox{T, D, NP, GPT<:SpatialPoint{T, D}} <: SpatialStructure{T, D}

A container of multiple `D`-dimensional grid points.

≡≡≡ Field(s) ≡≡≡

`spacing::T`: The distance between adjacent grid points, a.k.a the edge length of each grid.

`nPoint::Int`: Total number of the grid points.

`point::NTuple{NP, GPT}`: The grid points represented by [`SpatialPoint`](@ref).

`param::Tuple{Vararg{ParamBox{T}}}`: All the parameters in the `GridBox`.

≡≡≡ Initialization Method(s) ≡≡≡

    GridBox(nGrids::NTuple{D, Int}, spacing::Union{T, Array{T, 0}}, 
            center::Union{AbstractVector{T}, NTuple{D, T}}=ntuple(_->T(0),Val(D)); 
            canDiff::Bool=true, index::Int=0) where {T<:AbstractFloat, D} -> 
    GridBox{T, D}

Construct a general `D`-dimensional `GridBox`.

=== Positional argument(s) ===

`nGrids::NTuple{D, Int}`: The numbers of grids along each dimension.

`spacing::Union{T, Array{T, 0}}`: The edge length of each grid.

`center::Union{AbstractVector{T}, NTuple{D, T}}`: The coordinate of the geometric center of 
the grid box.

=== Keyword argument(s) ===

`canDiff`: If all the `ParamBox`es stored in the constructed `GridBox` will be marked as 
differentiable.

`index`: The Index that will be assigned to the shared input variable 
`$(ParamList[:spacing])` of all the stored `ParamBox`es.
"""
struct GridBox{T, D, NP, GPT<:SpatialPoint{T, D}} <: SpatialStructure{T, D}
    spacing::T
    nPoint::Int
    point::NTuple{NP, GPT}
    param::Tuple{Vararg{ParamBox{T}}}

    function GridBox(nGrids::NTuple{D, Int}, spacing::Union{T, Array{T, 0}}, 
                     center::Union{AbstractVector{T}, NTuple{D, T}}=ntuple(_->T(0),Val(D)); 
                     canDiff::Bool=true, index::Int=0) where {T<:AbstractFloat, D}
        @assert all(nGrids.>=0) "The number of gird of each edge must be no less than 0."
        @assert length(center)==D "The dimension of center coordinate must be equal to $D."
        NP = prod(nGrids .+ 1)
        iVsym = ParamList[:spacing]
        oVsym = SpatialParamSyms[1:D]
        data = ifelse(spacing isa AbstractFloat, fill(spacing), spacing)
        point = Array{SpatialPoint{T, D}}(undef, NP)
        param = Array{ParamBox{T}}(undef, D*NP)
        funcs = makeGridFuncsCore.(nGrids)
        for (n, i) in enumerate( CartesianIndices(nGrids .+ 1) )
            fs = makeGridFuncs.(center, [funcs[j][k] for (j, k) in enumerate(i|>Tuple)])
            p = ParamBox.(Ref(data), oVsym, fs, iVsym; canDiff, index) |> Tuple
            point[n] = SpatialPoint(p)
            param[D*(n-1)+1 : D*n] .= p
        end
        point = Tuple(point)
        new{T, D, NP, eltype(point)}(spacing, NP, point, Tuple(param))
    end
end

"""

    GridBox(nGridPerEdge::Int, spacing::T, 
            center::Union{AbstractVector{T}, NTuple{D, T}}=ntuple(_->T(0), 3); 
            canDiff::Bool=true, index::Int=0) where {T<:AbstractFloat, D} -> 
    GridBox{T, D}

The method of generating a cubic `GridBox`. Aside from the common arguments, `nGridPerEdge` 
specifies the number of grids for every dimension. The dimension of the grid box is 
determined by the dimension of `center`.
"""
GridBox(nGridPerEdge::Int, spacing::T, 
        center::Union{AbstractVector{T}, NTuple{D, T}}=ntuple(_->T(0), 3); 
        canDiff::Bool=true, index::Int=0) where {T<:AbstractFloat, D} = 
GridBox(ntuple(_->nGridPerEdge, length(center)), spacing, center; canDiff, index)


"""

    gridCoords(gb::GridBox{T}) where {T} -> Tuple{Vararg{Vector{T}}}

Return the coordinates of the grid points stored in `gb`.
"""
gridCoords(gb::GridBox) = coordOf.(gb.point)