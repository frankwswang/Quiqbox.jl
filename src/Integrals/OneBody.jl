export overlap, overlaps, nucAttraction, nucAttractions, elecKinetic, elecKinetics, 
       coreH, coreHij

"""

    overlap(fb1::AbstractGTBasisFuncs, fb2::AbstractGTBasisFuncs) -> 
    Array{Float64, 2}

Return the orbital overlap matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions.
"""
overlap(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs) = 
cat(getOverlap(bf1, bf2), dims=2)


"""

    overlaps(BSet::Array{<:AbstractGTBasisFuncs, 1}) -> Array{Float64, 2}

Return the orbital overlap matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Array`.
"""
overlaps(BSet::Vector{<:AbstractGTBasisFuncs}) = getOverlaps(BSet)


"""

    nucAttraction(fb1::AbstractGTBasisFuncs, fb2::AbstractGTBasisFuncs, 
                  nuc::Array{String, 1}, nucCoords::Array{<:AbstractArray, 1}) -> 
    Array{Float64, 2}

Return the nuclear attraction matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions, and the nuclei with their coordinates (in atomic unit).
"""
nucAttraction(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs, 
              nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
cat(getNucAttraction(bf1, bf2, nuc, nucCoords), dims=2)


"""

    nucAttractions(BSet::Array{<:AbstractGTBasisFuncs, 1}, nuc::Array{String, 1}, 
                   nucCoords::Array{<:AbstractArray, 1}) -> 
    Array{Float64, 2}

Return the nuclear attraction matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Array`, and the nuclei with their 
coordinates (in atomic unit).
"""
nucAttractions(BSet::Vector{<:AbstractGTBasisFuncs}, 
               nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
getNucAttractions(BSet, nuc, nucCoords)


"""

    elecKinetic(fb1::AbstractGTBasisFuncs, fb2::AbstractGTBasisFuncs) -> 
    Array{Float64, 2}

Return the electron kinetic energy matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions.
"""
elecKinetic(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs) = 
cat(getElecKinetic(bf1, bf2), dims=2)


"""

    elecKinetics(BSet::Array{<:AbstractGTBasisFuncs, 1}) -> Array{Float64, 2}

Return the electron kinetic energy matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Array`.
"""
elecKinetics(BSet::Vector{<:AbstractGTBasisFuncs}) = getElecKinetics(BSet)


@inline coreHijCore(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs, 
                    nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
        elecKineticCore(bf1, bf2) + nucAttractionCore(bf1, bf2, nuc, nucCoords)

"""

    coreHij(fb1::AbstractGTBasisFuncs, fb2::AbstractGTBasisFuncs, nuc::Array{String, 1}, 
            nucCoords::Array{<:AbstractArray, 1}) -> 
    Array{Float64, 2}

Return a matrix element or block of the core Hamiltonian (an N×N `Matrix` where N is the 
number of spatial orbitals) given 2 basis functions.
"""
coreHij(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs, 
        nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
cat(getCoreHij(bf1, bf2, nuc, nucCoords), dims=2)


@inline coreHCore(BSet::Vector{<:AbstractGTBasisFuncs}, 
                  nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
        elecKineticsCore(BSet) + nucAttractionsCore(BSet, nuc, nucCoords)

"""

    coreH(BSet::Array{<:AbstractGTBasisFuncs, 1}, nuc::Array{String, 1}, 
          nucCoords::Array{<:AbstractArray, 1}) -> Array{Float64, 2}

Return the core Hamiltonian matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Array`.
"""
coreH(BSet::Vector{<:AbstractGTBasisFuncs}, 
      nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
getCoreH(BSet, nuc, nucCoords)