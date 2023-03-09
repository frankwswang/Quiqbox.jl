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
getCompositeInt(∫overlapCore, (), (bf2===bf1,), bf1, bf2)


"""

    overlaps(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                       AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} -> 
    Matrix{T}

Return the orbital overlap matrix given a basis set.
"""
overlaps(bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}) where {T, D} = 
getOneBodyInts(∫overlapCore, (), collectTuple(bs))


"""

    eKinetic(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}) where 
            {T, D} -> 
    T

Return the electron kinetic energy between two basis functions.
"""
eKinetic(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}) where 
        {D, T} = 
getCompositeInt(∫elecKineticCore, (), (bf2===bf1,), bf1, bf2)


"""

    eKinetics(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                        AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} -> 
    Matrix{T}

Return the electron kinetic energy matrix given a basis set.
"""
eKinetics(bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}) where {T, D} = 
getOneBodyInts(∫elecKineticCore, (), collectTuple(bs))


"""

    neAttraction(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
                 nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                 nucCoords::$(SpatialCoordType|>typeStrNotUnionAll)) where {T, D, NN} -> 
    T

Return the nuclear attraction between two basis functions, provided with the nuclei and 
their coordinates (in the atomic units).
"""
neAttraction(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
             nuc::AVectorOrNTuple{String, NN}, 
             nucCoords::SpatialCoordType{T, D, NN}) where {T, D, NN} = 
getCompositeInt(∫nucAttractionCore, (arrayToTuple(nuc), genTupleCoords(T, nucCoords)), 
                (bf2===bf1,), bf1, bf2)


"""

    neAttractions(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                            AbstractVector{<:AbstractGTBasisFuncs{T, D}}}, 
                  nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                  nucCoords::$(SpatialCoordType|>typeStrNotUnionAll)) where {T, D, NN} -> 
    Matrix{T}

Return the nuclear attraction matrix given a basis set and the corresponding nuclei with 
their coordinates (in atomic units).
"""
neAttractions(bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}, 
              nuc::AVectorOrNTuple{String, NN}, 
              nucCoords::SpatialCoordType{T, D, NN}) where {T, D, NN} = 
getOneBodyInts(∫nucAttractionCore, (arrayToTuple(nuc), genTupleCoords(T, nucCoords)), 
               collectTuple(bs))


"""

    coreHij(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
            nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
            nucCoords::$(SpatialCoordType|>typeStrNotUnionAll)) where {T, D, NN} -> 
    T

Return a matrix element of the core Hamiltonian given two basis functions.
"""
coreHij(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
        nuc::AVectorOrNTuple{String, NN}, 
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
coreH(bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}, 
      nuc::AVectorOrNTuple{String, NN}, 
      nucCoords::SpatialCoordType{T, D, NN}) where {T, D, NN} = 
eKinetics(bs) + neAttractions(bs, nuc, nucCoords)

coreH(b::GTBasis{T, D}, 
      nuc::AVectorOrNTuple{String, NN}, 
      nucCoords::SpatialCoordType{T, D, NN}) where {T, D, NN} = 
neAttractions(b.basis, nuc, nucCoords) + b.Te