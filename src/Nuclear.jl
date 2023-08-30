struct Nuc{T, D, NN, CPT}
    sym::NTuple{NN, Symbol}
    pos::Tuple{Vararg{SpatialPoint{T, D, <:CPT}, NN}}

    function Nuc{CPT}(@nospecialize(sps::Tuple{SpatialPoint{T, D, <:CPT}, 
                                               Vararg{SpatialPoint{T, D, <:CPT}, 
                                                      NN}})) where {CPT, T, D, NN}
        sym = map(sps) do sp
            marker = sp.marker
            haskey(AtomicNumberList, marker) || (throw∘KeyError)(marker)
            marker
        end
        new{T, D, NN+1, CPT}(sym, sps)
    end
end

Nuc(sps::Tuple{SpatialPoint{T, D, CPT}, Vararg{SpatialPoint{T, D, CPT}}}) where 
   {T, D, CPT} = 
Nuc{CPT}(sps)

Nuc(@nospecialize(sps::Tuple{SpatialPoint{T, D, <:NTuple{D, ParamBox{T}}}, 
                             Vararg{SpatialPoint{T, D, <:NTuple{D, ParamBox{T}}}}})) where 
   {T, D} = 
Nuc{NTuple{D, ParamBox{T}}}(sps)

Nuc(sps::AbstractVector{<:SpatialPoint{T, D}}) where {T, D} = (Nuc∘Tuple)(sps)