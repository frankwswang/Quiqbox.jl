export overlap, overlaps, neAttraction, neAttractions, eKinetic, eKinetics, 
       coreH, coreHij

"""

    overlap(fb1::AbstractGTBasisFuncs{T, D, 1}, fb2::AbstractGTBasisFuncs{T, D, 1}) where 
           {T, D, 1} -> 
    T

Return the orbital overlap matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions.
"""
overlap(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}) where 
       {T, D} = 
getOverlap(bf1, bf2)


"""

    overlaps(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                       AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} -> 
    Matrix{T}

Return the orbital overlap matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `AbstractVector`.
"""
overlaps(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                   AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} = 
getOverlap(bs |> arrayToTuple)


"""

    eKinetic(fb1::AbstractGTBasisFuncs{T, D, 1}, fb2::AbstractGTBasisFuncs{T, D, 1}) where 
            {T, D} -> 
    T

Return the electron kinetic energy matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions.
"""
eKinetic(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}) where 
        {D, T} = 
getEleKinetic(bf1, bf2)


"""

    eKinetics(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                        AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} -> 
    Matrix{T}

Return the electron kinetic energy matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `AbstractVector`.
"""
eKinetics(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                    AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} = 
getEleKinetic(bs |> arrayToTuple)


"""

    neAttraction(fb1::AbstractGTBasisFuncs{T, D, 1}, fb2::AbstractGTBasisFuncs{T, D, 1}, 
                 nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                 nucCoords::Union{NTuple{NN, NTuple{D, T}}, 
                                  AbstractVector{<:AbstractVector{T}}}) where {T, D, NN} -> 
    T

Return the nuclear attraction matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions, and the nuclei with their coordinates (in atomic unit).
"""
neAttraction(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
              nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
              nucCoords::Union{NTuple{NN, NTuple{D, T}}, 
                               AbstractVector{<:AbstractVector{T}}}) where {T, D, NN} = 
getNucEleAttraction(bf1, bf2, arrayToTuple(nuc), genTupleCoords(T, nucCoords))


"""

    neAttractions(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                            AbstractVector{<:AbstractGTBasisFuncs{T, D}}}, 
                  nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                  nucCoords::Union{NTuple{NN, NTuple{D, T}}, 
                                   AbstractVector{<:AbstractVector{T}}}) where 
                 {T, D, NN} -> 
    Matrix{T}

Return the nuclear attraction matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `AbstractVector`, and the nuclei with their 
coordinates (in atomic unit).
"""
neAttractions(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                        AbstractVector{<:AbstractGTBasisFuncs{T, D}}}, 
              nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
              nucCoords::Union{NTuple{NN, NTuple{D, T}}, 
                               AbstractVector{<:AbstractVector{T}}}) where {T, D, NN} = 
getNucEleAttraction(bs|>arrayToTuple, arrayToTuple(nuc), genTupleCoords(T, nucCoords))


"""

    coreHij(fb1::AbstractGTBasisFuncs{T, D, 1}, fb2::AbstractGTBasisFuncs{T, D, 1}, 
            nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
            nucCoords::Union{NTuple{NN, NTuple{D, T}}, 
                             AbstractVector{<:AbstractVector{T}}}) where {T, D, NN} -> 
    T

Return a block M×N matrix of the core Hamiltonian given 2 basis functions, where M and N 
are the numbers of spatial orbitals within in the basis functions.
"""
coreHij(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
        nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
        nucCoords::Union{NTuple{NN, NTuple{D, T}}, 
                         AbstractVector{<:AbstractVector{T}}}) where {T, D, NN} = 
getCoreH(bf1, bf2, arrayToTuple(nuc), genTupleCoords(T, nucCoords))


"""

    coreH(bs::Union{GTBasis, Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                    AbstractVector{<:AbstractGTBasisFuncs{T, D}}}, 
          nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
          nucCoords::Union{NTuple{NN, NTuple{D, T}}, 
                           AbstractVector{<:AbstractVector{T}}}) where {T, D, NN} -> 
    Matrix{T}

Return the core Hamiltonian matrix (an N×N `Matrix` where N is the total number of spatial 
orbitals) given a basis set in the form of an `AbstractVector`.
"""
coreH(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                AbstractVector{<:AbstractGTBasisFuncs{T, D}}}, 
      nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
      nucCoords::Union{NTuple{NN, NTuple{D, T}}, 
                       AbstractVector{<:AbstractVector{T}}}) where {T, D, NN} = 
getCoreH(bs|>arrayToTuple, arrayToTuple(nuc), genTupleCoords(T, nucCoords))

coreH(b::GTBasis{T, D}, nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
      nucCoords::Union{NTuple{NN, NTuple{D, T}}, 
                       AbstractVector{<:AbstractVector{T}}}) where {T, D, NN} = 
neAttractions(b.basis, nuc, nucCoords) + b.Te