export eeInteraction, eeInteractions

"""

    eeInteraction(bf1::AbstractGTBasisFuncs{T, D, 1}, 
                  bf2::AbstractGTBasisFuncs{T, D, 1}, 
                  bf3::AbstractGTBasisFuncs{T, D, 1}, 
                  bf4::AbstractGTBasisFuncs{T, D, 1}) where {T, D} -> 
    T

Return an electron-electron interaction tensor element given four basis functions (ordered 
in the chemists' notation).
"""
eeInteraction(bf1::AbstractGTBasisFuncs{T, D, 1}, 
              bf2::AbstractGTBasisFuncs{T, D, 1}, 
              bf3::AbstractGTBasisFuncs{T, D, 1}, 
              bf4::AbstractGTBasisFuncs{T, D, 1}) where {T, D} = 
getCompositeInt(∫eeInteractionCore, (), 
                (bf4===bf3, bf4===bf2, bf3===bf2, ifelse(bf4===bf2, bf3, bf2)===bf1), 
                bf1, bf2, bf3, bf4)


"""

    eeInteractions(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                             AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) -> 
    Array{T, 4}

Return the tensor of electron-electron interactions (in the chemists' notation) given a 
basis set.
"""
eeInteractions(bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}) where {T, D} = 
getTwoBodyInts(∫eeInteractionCore, (), collectTuple(bs))