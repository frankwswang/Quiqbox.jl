export overlap, overlaps, nucAttraction, nucAttractions, elecKinetic, elecKinetics, 
       coreH, coreHij

"""

    overlap(fb1::AbstractGTBasisFuncs, fb2::AbstractGTBasisFuncs) -> Matrix{Float64}

Return the orbital overlap matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions.
"""
overlap(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs) = 
cat(getOverlap(bf1, bf2), dims=2)


"""

    overlaps(BSet::Vector{<:AbstractGTBasisFuncs}) -> Matrix{Float64}

Return the orbital overlap matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Vector`.
"""
overlaps(BSet::Vector{<:AbstractGTBasisFuncs}) = getOverlaps(BSet)


"""

    elecKinetic(fb1::AbstractGTBasisFuncs, fb2::AbstractGTBasisFuncs) -> Matrix{Float64}

Return the electron kinetic energy matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions.
"""
elecKinetic(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs) = 
cat(getElecKinetic(bf1, bf2), dims=2)


"""

    elecKinetics(BSet::Vector{<:AbstractGTBasisFuncs}) -> Matrix{Float64}

Return the electron kinetic energy matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Vector`.
"""
elecKinetics(BSet::Vector{<:AbstractGTBasisFuncs}) = getElecKinetics(BSet)


"""

    nucAttraction(fb1::AbstractGTBasisFuncs, fb2::AbstractGTBasisFuncs, 
                  nuc::Vector{String}, nucCoords::Vector{<:AbstractArray{<:Real}}) -> 
    Matrix{Float64}

Return the nuclear attraction matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions, and the nuclei with their coordinates (in atomic unit).
"""
nucAttraction(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs, 
              nuc::Vector{String}, nucCoords::Vector{<:AbstractArray{<:Real}}) = 
nucAttraction(bf1, bf2, Tuple(nuc), genTupleCoords(nucCoords))

"""

    nucAttraction(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs, 
                  nuc::NTuple{NN, String}, 
                  nucCoords::NTuple{NN, NTuple{3,Float64}}) where {NN}-> 
    Matrix{Float64}

"""
nucAttraction(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs, 
              nuc::NTuple{NN, String}, 
              nucCoords::NTuple{NN, NTuple{3,Float64}}) where {NN} = 
cat(getNucAttraction(bf1, bf2, nuc, nucCoords), dims=2)


"""

    nucAttractions(BSet::Vector{<:AbstractGTBasisFuncs}, 
                   nuc::Vector{String}, nucCoords::Vector{<:AbstractArray{<:Real}}) -> 
    Matrix{Float64}

Return the nuclear attraction matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Vector`, and the nuclei with their 
coordinates (in atomic unit).
"""
nucAttractions(BSet::Vector{<:AbstractGTBasisFuncs}, 
               nuc::Vector{String}, nucCoords::Vector{<:AbstractArray{<:Real}}) = 
nucAttractions(BSet, Tuple(nuc), genTupleCoords(nucCoords))

"""

    nucAttractions(BSet::Vector{<:AbstractGTBasisFuncs}, 
                   nuc::NTuple{NN, String}, 
                   nucCoords::NTuple{NN, NTuple{3,Float64}}) where {NN} -> 
    Matrix{Float64}
"""
nucAttractions(BSet::Vector{<:AbstractGTBasisFuncs}, 
               nuc::NTuple{NN, String}, 
               nucCoords::NTuple{NN, NTuple{3,Float64}}) where {NN} = 
getNucAttractions(BSet, nuc, nucCoords)


"""

    coreHij(fb1::AbstractGTBasisFuncs, fb2::AbstractGTBasisFuncs, 
            nuc::Vector{String}, nucCoords::Vector{<:AbstractArray{<:Real}}) -> 
    Matrix{Float64}

Return a block M×N matrix of the core Hamiltonian given 2 basis functions, where M and N 
are the numbers of spatial orbitals within in the basis functions.
"""
coreHij(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs, 
        nuc::Vector{String}, nucCoords::Vector{<:AbstractArray{<:Real}}) = 
coreHij(bf1, bf2, Tuple(nuc), genTupleCoords(nucCoords))


"""

    coreHij(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs, 
            nuc::NTuple{NN, String}, 
            nucCoords::NTuple{NN, NTuple{3,Float64}}) where {NN} -> 
    Matrix{Float64}
"""
coreHij(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs, 
        nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{3,Float64}}) where {NN} = 
cat(getCoreHij(bf1, bf2, nuc, nucCoords), dims=2)


"""

    coreH(BSet::Vector{<:AbstractGTBasisFuncs}, 
          nuc::Vector{String}, nucCoords::Vector{<:AbstractArray{<:Real}}) -> 
    Matrix{Float64}

Return the core Hamiltonian matrix (an N×N `Matrix` where N is the total number of spatial 
orbitals) given a basis set in the form of an `Vector`.
"""
coreH(BSet::Vector{<:AbstractGTBasisFuncs}, 
      nuc::Vector{String}, nucCoords::Vector{<:AbstractArray{<:Real}}) = 
coreH(BSet, Tuple(nuc), genTupleCoords(nucCoords))

"""

    coreH(BSet::Vector{<:AbstractGTBasisFuncs}, 
          nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{3,Float64}}) where {NN} -> 
    Matrix{Float64}

"""
coreH(BSet::Vector{<:AbstractGTBasisFuncs}, 
      nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{3,Float64}}) where {NN} = 
getCoreH(BSet, nuc, nucCoords)