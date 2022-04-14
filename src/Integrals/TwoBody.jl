export eeInteraction, eeInteractions

"""

    eeInteraction(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs, 
                  bf3::AbstractGTBasisFuncs, bf4::AbstractGTBasisFuncs) -> 
    Array{Float64, 4}

Return the electron-electron interaction tensor (an N×N×N×N Tensor where N is the number of 
spatial orbitals) given 4 basis functions.
"""
eeInteraction(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs, 
              bf3::AbstractGTBasisFuncs, bf4::AbstractGTBasisFuncs) = 
cat(get2eInteraction(bf1, bf2, bf3, bf4), dims=4)


"""

    eeInteractions(bs::Union{Tuple{Vararg{<:AbstractGTBasisFuncs}}, 
                             Vector{<:AbstractGTBasisFuncs}}) -> 
    Array{Float64, 4}

Return the electron-electron interaction tensor (an N×N×N×N Tensor where N is the number 
of spatial orbitals) given a basis set in the form of an `Vector`.
"""
eeInteractions(bs::Union{Tuple{Vararg{<:AbstractGTBasisFuncs}}, 
                         Vector{<:AbstractGTBasisFuncs}}) = 
get2eInteractions(bs |> arrayToTuple)