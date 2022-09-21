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
getCompositeInt(∫overlapCore, (bf2===bf1,), (bf1, bf2))


"""

    overlaps(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                       AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} -> 
    Matrix{T}

Return the orbital overlap matrix given a basis set.
"""
overlaps(bs::AbstractVector{<:AbstractGTBasisFuncs{T, D}}) where {T, D} = 
getOneBodyInts(∫overlapCore, bs)

overlaps(bs::Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}) where {T, D} = overlaps(bs|>collect)


"""

    eKinetic(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}) where 
            {T, D} -> 
    T

Return the electron kinetic energy between two basis functions.
"""
eKinetic(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}) where 
        {D, T} = 
getCompositeInt(∫elecKineticCore, (bf2===bf1,), (bf1, bf2))


"""

    eKinetics(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                        AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} -> 
    Matrix{T}

Return the electron kinetic energy matrix given a basis set.
"""
eKinetics(bs::AbstractVector{<:AbstractGTBasisFuncs{T, D}}) where {T, D} = 
getOneBodyInts(∫elecKineticCore, bs)

eKinetics(bs::Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}) where {T, D} = 
eKinetics(bs|>collect)


"""

    neAttraction(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
                 nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                 nucCoords::$(SpatialCoordType|>typeStrNotUnionAll)) where {T, D, NN} -> 
    T

Return the nuclear attraction between two basis functions, provided with the nuclei and 
their coordinates (in the atomic units).
"""
neAttraction(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
             nuc::VectorOrNTuple{String, NN}, 
             nucCoords::SpatialCoordType{T, D, NN}) where {T, D, NN} = 
getCompositeInt(∫nucAttractionCore, (bf2===bf1,), (bf1, bf2), 
                arrayToTuple(nuc), genTupleCoords(T, nucCoords))


"""

    neAttractions(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                            AbstractVector{<:AbstractGTBasisFuncs{T, D}}}, 
                  nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                  nucCoords::$(SpatialCoordType|>typeStrNotUnionAll)) where {T, D, NN} -> 
    Matrix{T}

Return the nuclear attraction matrix given a basis set and the corresponding nuclei with 
their coordinates (in atomic units).
"""
neAttractions(bs::AbstractVector{<:AbstractGTBasisFuncs{T, D}}, 
              nuc::VectorOrNTuple{String, NN}, 
              nucCoords::SpatialCoordType{T, D, NN}) where {T, D, NN} = 
getOneBodyInts(∫nucAttractionCore, bs, 
               arrayToTuple(nuc), genTupleCoords(T, nucCoords))

neAttractions(bs::Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
              nuc::VectorOrNTuple{String, NN}, 
              nucCoords::SpatialCoordType{T, D, NN}) where {T, D, NN} = 
neAttractions(collect(bs), nuc, nucCoords)


"""

    coreHij(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
            nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
            nucCoords::$(SpatialCoordType|>typeStrNotUnionAll)) where {T, D, NN} -> 
    T

Return a matrix element of the core Hamiltonian given two basis functions.
"""
coreHij(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
        nuc::VectorOrNTuple{String, NN}, 
        nucCoords::SpatialCoordType{T, D, NN}) where {T, D, NN} = 
eKinetic(bf1, bf2) + neAttraction(bf1, bf2, nuc, nucCoords)


"""

    coreH(bs::Union{GTBasis, Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                    AbstractVector{<:AbstractGTBasisFuncs{T, D}}}, 
          nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
          nucCoords::$(SpatialCoordType|>typeStrNotUnionAll)) where {T, D, NN} -> 
    Matrix{T}

Return the core Hamiltonian given a basis set and the corresponding nuclei with their 
coordinates (in atomic units).
"""
coreH(bs::VectorOrNTuple{AbstractGTBasisFuncs{T, D}}, 
      nuc::VectorOrNTuple{String, NN}, 
      nucCoords::SpatialCoordType{T, D, NN}) where {T, D, NN} = 
eKinetics(bs) + neAttractions(bs, nuc, nucCoords)

coreH(b::GTBasis{T, D}, 
      nuc::VectorOrNTuple{String, NN}, 
      nucCoords::SpatialCoordType{T, D, NN}) where {T, D, NN} = 
neAttractions(b.basis, nuc, nucCoords) + b.Te