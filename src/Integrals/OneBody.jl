export overlap, overlaps, neAttraction, neAttractions, eKinetic, eKinetics, 
       coreH, coreHij

"""

    overlap(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}) where 
           {T, D, 1} -> 
    T

Return the orbital overlap between two basis functions.
"""
overlap(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}) where 
       {T, D} = 
getOverlap(bf1, bf2)


"""

    overlaps(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                       AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} -> 
    Matrix{T}

Return the orbital overlap matrix given a basis set.
"""
overlaps(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                   AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} = 
getOverlap(bs |> arrayToTuple)


"""

    eKinetic(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}) where 
            {T, D} -> 
    T

Return the electron kinetic energy between two basis functions.
"""
eKinetic(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}) where 
        {D, T} = 
getEleKinetic(bf1, bf2)


"""

    eKinetics(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                        AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} -> 
    Matrix{T}

Return the electron kinetic energy matrix given a basis set.
"""
eKinetics(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                    AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} = 
getEleKinetic(bs |> arrayToTuple)


"""

    neAttraction(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
                 nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                 nucCoords::Union{NTuple{NN, NTuple{D, T}}, 
                                  AbstractVector{<:AbstractVector{T}}}) where {T, D, NN} -> 
    T

Return the nuclear attraction between two basis functions, provided with the nuclei and 
their coordinates (in the atomic units).
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

Return the nuclear attraction matrix given a basis set and the corresponding nuclei with 
their coordinates (in atomic units).
"""
neAttractions(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                        AbstractVector{<:AbstractGTBasisFuncs{T, D}}}, 
              nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
              nucCoords::Union{NTuple{NN, NTuple{D, T}}, 
                               AbstractVector{<:AbstractVector{T}}}) where {T, D, NN} = 
getNucEleAttraction(bs|>arrayToTuple, arrayToTuple(nuc), genTupleCoords(T, nucCoords))


"""

    coreHij(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
            nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
            nucCoords::Union{NTuple{NN, NTuple{D, T}}, 
                             AbstractVector{<:AbstractVector{T}}}) where {T, D, NN} -> 
    T

Return a matrix element of the core Hamiltonian given two basis functions.
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

Return the core Hamiltonian given a basis set and the corresponding nuclei with their 
coordinates (in atomic units).
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