export eeInteraction, eeInteractions

"""

    eeInteraction(bf1::AbstractGTBasisFuncs, 
                  bf2::AbstractGTBasisFuncs, 
                  bf3::AbstractGTBasisFuncs, 
                  bf4::AbstractGTBasisFuncs) -> 
    Array{Float64, 4}

Return the electron-electron interaction tensor (an N×N×N×N Tensor where N is the number of 
spatial orbitals) given 4 basis functions.
"""
eeInteraction(bf1::AbstractGTBasisFuncs, 
              bf2::AbstractGTBasisFuncs, 
              bf3::AbstractGTBasisFuncs, 
              bf4::AbstractGTBasisFuncs) = cat(get2eInteraction(bf1, bf2, bf3, bf4), dims=4)


"""

    eeInteractionsCore(BSet::Array{<:AbstractGTBasisFuncs, 1}; 
                       outputUniqueIndices::Bool=false) -> 
    Array{Float64, 5}, [Array{<:Array{Int, 1}, 1}]

Return the electron-electron interaction tensor (an N×N×N×N×1 Tensor where N is the number 
of spatial orbitals) given a basis set in the form of an `Array`.

If `outputUniqueIndices=true`, additionally return the indices for all the unique integrals.
"""
@inline eeInteractionsCore(BSet::Vector{<:AbstractGTBasisFuncs}; 
                           outputUniqueIndices::Bool=false) = 
        twoBodyBSTensor(BSet, eeInteractionCore; outputUniqueIndices)

"""

    eeInteractions(BSet::Array{<:AbstractGTBasisFuncs, 1}) -> Array{Float64, 4}

Return the electron-electron interaction tensor (an N×N×N×N Tensor where N is the number 
of spatial orbitals) given a basis set in the form of an `Array`. 
"""
eeInteractions(BSet::Vector{<:AbstractGTBasisFuncs}) = get2eInteractions(BSet)