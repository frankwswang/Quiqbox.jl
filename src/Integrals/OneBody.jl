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

    overlaps(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs}}, 
                       Vector{<:AbstractGTBasisFuncs}}) -> 
    Matrix{Float64}

Return the orbital overlap matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Vector`.
"""
overlaps(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs}}, Vector{<:AbstractGTBasisFuncs}}) = 
getOverlaps(bs |> arrayToTuple)


"""

    elecKinetic(fb1::AbstractGTBasisFuncs, fb2::AbstractGTBasisFuncs) -> Matrix{Float64}

Return the electron kinetic energy matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions.
"""
elecKinetic(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs) = 
cat(getElecKinetic(bf1, bf2), dims=2)


"""

    elecKinetics(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs}}, 
                           Vector{<:AbstractGTBasisFuncs}}) -> 
    Matrix{Float64}

Return the electron kinetic energy matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Vector`.
"""
elecKinetics(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs}}, 
                       Vector{<:AbstractGTBasisFuncs}}) = 
getElecKinetics(bs |> arrayToTuple)


"""

    nucAttraction(fb1::AbstractGTBasisFuncs, fb2::AbstractGTBasisFuncs, 
                  nuc::Union{NTuple{NN, String}, Vector{String}}, 
                  nucCoords::Union{NTuple{NN, NTuple{3, Float64}}, 
                                   Vector{<:AbstractArray{<:Real}}}) where {NN} -> 
    Matrix{Float64}

Return the nuclear attraction matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions, and the nuclei with their coordinates (in atomic unit).
"""
nucAttraction(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs, 
              nuc::Union{NTuple{NN, String}, Vector{String}}, 
              nucCoords::Union{NTuple{NN, NTuple{3, Float64}}, 
                               Vector{<:AbstractArray{<:Real}}}) where {NN} = 
cat(getNucAttraction(bf1, bf2, arrayToTuple(nuc), genTupleCoords(nucCoords)), dims=2)


"""

    nucAttractions(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs}}, 
                             Vector{<:AbstractGTBasisFuncs}}, 
                   nuc::Union{NTuple{NN, String}, Vector{String}}, 
                   nucCoords::Union{NTuple{NN, NTuple{3, Float64}}, 
                                    Vector{<:AbstractArray{<:Real}}}) where {NN} -> 
    Matrix{Float64}

Return the nuclear attraction matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Vector`, and the nuclei with their 
coordinates (in atomic unit).
"""
nucAttractions(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs}}, 
                         Vector{<:AbstractGTBasisFuncs}}, 
               nuc::Union{NTuple{NN, String}, Vector{String}}, 
               nucCoords::Union{NTuple{NN, NTuple{3, Float64}}, 
                                Vector{<:AbstractArray{<:Real}}}) where {NN} = 
getNucAttractions(bs|>arrayToTuple, arrayToTuple(nuc), genTupleCoords(nucCoords))


"""

    coreHij(fb1::AbstractGTBasisFuncs, fb2::AbstractGTBasisFuncs, 
            nuc::Union{NTuple{NN, String}, Vector{String}}, 
            nucCoords::Union{NTuple{NN, NTuple{3, Float64}}, 
                             Vector{<:AbstractArray{<:Real}}}) where {NN} -> 
    Matrix{Float64}

Return a block M×N matrix of the core Hamiltonian given 2 basis functions, where M and N 
are the numbers of spatial orbitals within in the basis functions.
"""
coreHij(bf1::AbstractGTBasisFuncs, bf2::AbstractGTBasisFuncs, 
        nuc::Union{NTuple{NN, String}, Vector{String}}, 
        nucCoords::Union{NTuple{NN, NTuple{3, Float64}}, 
                         Vector{<:AbstractArray{<:Real}}}) where {NN} = 
cat(getCoreHij(bf1, bf2, arrayToTuple(nuc), genTupleCoords(nucCoords)), dims=2)


"""

    coreH(bs::Union{GTBasis, Tuple{Vararg{AbstractGTBasisFuncs}}, 
                    Vector{<:AbstractGTBasisFuncs}}, 
          nuc::Union{NTuple{NN, String}, Vector{String}}, 
          nucCoords::Union{NTuple{NN, NTuple{3, Float64}}, 
                           Vector{<:AbstractArray{<:Real}}}) where {NN} -> 
    Matrix{Float64}

Return the core Hamiltonian matrix (an N×N `Matrix` where N is the total number of spatial 
orbitals) given a basis set in the form of an `Vector`.
"""
coreH(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs}}, Vector{<:AbstractGTBasisFuncs}}, 
      nuc::Union{NTuple{NN, String}, Vector{String}}, 
      nucCoords::Union{NTuple{NN, NTuple{3, Float64}}, 
                       Vector{<:AbstractArray{<:Real}}}) where {NN} = 
getCoreH(bs|>arrayToTuple, arrayToTuple(nuc), genTupleCoords(nucCoords))

coreH(b::GTBasis, nuc::Union{NTuple{NN, String}, Vector{String}}, 
      nucCoords::Union{NTuple{NN, NTuple{3, Float64}}, 
                       Vector{<:AbstractArray{<:Real}}}) where {NN} = 
nucAttractions(b.basis, nuc, nucCoords) + b.Te