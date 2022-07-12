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
                getfield(Quiqbox, funcSym)
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
struct GridBox{T, D, NP, GPT<:SpatialPoint{T, D}} <: SpatialStructure{T, D}
    nPoint::Int
    spacing::T
    box::NTuple{NP, GPT}
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
        box = Array{SpatialPoint{T, D}}(undef, NP)
        param = Array{ParamBox{T}}(undef, D*NP)
        funcs = makeGridFuncsCore.(nGrids)
        for (n, i) in enumerate( CartesianIndices(nGrids .+ 1) )
            fs = makeGridFuncs.(center, [funcs[j][k] for (j, k) in enumerate(i|>Tuple)])
            p = ParamBox.(Ref(data), oVsym, fs, iVsym; canDiff, index) |> Tuple
            box[n] = SpatialPoint(p)
            param[D*(n-1)+1 : D*n] .= p
        end
        box = Tuple(box)
        new{T, D, NP, eltype(box)}(NP, spacing, box, Tuple(param))
    end
end

GridBox(nGridPerEdge::Int, spacing::T=T(1), 
        center::Union{AbstractVector{T}, NTuple{D, T}}=ntuple(_->T(0), 3); 
        canDiff::Bool=true, index::Int=0) where {T<:AbstractFloat, D} = 
GridBox(ntuple(_->nGridPerEdge, length(center)), spacing, center; canDiff, index)


"""

    gridCoords(gb::GridBox) -> Array{AbstractVector{Float64}, 1}

Return the grid-point coordinates in `AbstractVector`s given the `GriBox`.
"""
gridCoords(gb::GridBox) = [outValOf.(i) |> collect for i in gb.box]