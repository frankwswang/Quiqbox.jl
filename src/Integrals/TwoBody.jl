export eeInteraction, eeInteractions

"""

    eeInteraction(bf1::AbstractGTBasisFuncs{T, D, 1}, 
                  bf2::AbstractGTBasisFuncs{T, D, 1}, 
                  bf3::AbstractGTBasisFuncs{T, D, 1}, 
                  bf4::AbstractGTBasisFuncs{T, D, 1}) -> 
    T

Return the electron-electron interaction tensor (an N×N×N×N Tensor where N is the number of 
spatial orbitals) given 4 basis functions.
"""
eeInteraction(bf1::AbstractGTBasisFuncs{T, D, 1}, 
              bf2::AbstractGTBasisFuncs{T, D, 1}, 
              bf3::AbstractGTBasisFuncs{T, D, 1}, 
              bf4::AbstractGTBasisFuncs{T, D, 1}) where {T, D} = 
getEleEleInteraction(bf1, bf2, bf3, bf4)


"""

    eeInteractions(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                             AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) -> 
    Array{T, 4}

Return the electron-electron interaction tensor (an N×N×N×N Tensor where N is the number 
of spatial orbitals) given a basis set in the form of an `AbstractVector`.
"""
eeInteractions(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                         AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} = 
getEleEleInteraction(bs |> arrayToTuple)