export eeInteraction, eeInteractions

"""

    eeInteraction(bf1::AbstractGTBasisFuncs{T, D, 1}, 
                  bf2::AbstractGTBasisFuncs{T, D, 1}, 
                  bf3::AbstractGTBasisFuncs{T, D, 1}, 
                  bf4::AbstractGTBasisFuncs{T, D, 1}) where {T, D} -> 
    T

Return a tensor element of the electron-electron interaction given four basis functions.
"""
eeInteraction(bf1::AbstractGTBasisFuncs{T, D, 1}, 
              bf2::AbstractGTBasisFuncs{T, D, 1}, 
              bf3::AbstractGTBasisFuncs{T, D, 1}, 
              bf4::AbstractGTBasisFuncs{T, D, 1}) where {T, D} = 
getCompositeInt(∫eeInteractionCore, (false, false, false, false), (bf1, bf2, bf3, bf4))
# getCompositeInt(∫eeInteractionCore, (bf4===bf3, bf4===bf2, bf3===bf2, false), 
#                 (bf1, bf2, bf3, bf4))


"""

    eeInteractions(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                             AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) -> 
    Array{T, 4}

Return the electron-electron interaction given a basis set.
"""
eeInteractions(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                         AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} = 
getTwoBodyInts(∫eeInteractionCore, bs|>arrayToTuple)