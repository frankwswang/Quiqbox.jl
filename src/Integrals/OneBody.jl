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
get1BCompInt(T, Val(D), ∫overlapCore, (), (bf2===bf1,), (1, 1), bf1, bf2)


"""

    overlaps(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                       AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} -> 
    Matrix{T}

Return the orbital overlap matrix given a basis set.
"""
overlaps(bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}) where {T, D} = 
getOneBodyInts(∫overlapCore, (), lazyCollect(bs))


"""

    eKinetic(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}) where 
            {T, D} -> 
    T

Return the electron kinetic energy between two basis functions.
"""
eKinetic(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}) where 
        {D, T} = 
get1BCompInt(T, Val(D), ∫elecKineticCore, (), (bf2===bf1,), (1, 1), bf1, bf2)


"""

    eKinetics(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                        AbstractVector{<:AbstractGTBasisFuncs{T, D}}}) where {T, D} -> 
    Matrix{T}

Return the electron kinetic energy matrix given a basis set.
"""
eKinetics(bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}) where {T, D} = 
getOneBodyInts(∫elecKineticCore, (), lazyCollect(bs))


"""

    neAttraction(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
                 nuc::Union{Tuple{String, Vararg{String, NNMO}}, AbstractVector{String}}, 
                 nucCoords::$(SpatialCoordType|>typeStrNotUnionAll)) where {T, D, NNMO} -> 
    T

Return the nuclear attraction between two basis functions, provided with the nuclei and 
their coordinates (in the atomic units).
"""
neAttraction(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
             nuc::AVectorOrNTuple{String, NNMO}, 
             nucCoords::SpatialCoordType{T, D, NNMO}) where {T, D, NNMO} = 
get1BCompInt(T, Val(D), ∫nucAttractionCore, 
             (arrayToTuple(nuc), genTupleCoords(T, nucCoords)), (bf2===bf1,), 
             (1, 1), bf1, bf2)


"""

    neAttractions(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                            AbstractVector{<:AbstractGTBasisFuncs{T, D}}}, 
                  nuc::Union{Tuple{String, Vararg{String, NNMO}}, AbstractVector{String}}, 
                  nucCoords::$(SpatialCoordType|>typeStrNotUnionAll)) where {T, D, NNMO} -> 
    Matrix{T}

Return the nuclear attraction matrix given a basis set and the corresponding nuclei with 
their coordinates (in atomic units).
"""
neAttractions(bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}, 
              nuc::AVectorOrNTuple{String, NNMO}, 
              nucCoords::SpatialCoordType{T, D, NNMO}) where {T, D, NNMO} = 
getOneBodyInts(∫nucAttractionCore, (arrayToTuple(nuc), genTupleCoords(T, nucCoords)), 
               lazyCollect(bs))


"""

    coreHij(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
            nuc::Union{Tuple{String, Vararg{String, NNMO}}, AbstractVector{String}}, 
            nucCoords::$(SpatialCoordType|>typeStrNotUnionAll)) where {T, D, NNMO} -> 
    T

Return a matrix element of the core Hamiltonian given two basis functions.
"""
coreHij(bf1::AbstractGTBasisFuncs{T, D, 1}, bf2::AbstractGTBasisFuncs{T, D, 1}, 
        nuc::AVectorOrNTuple{String, NNMO}, 
        nucCoords::SpatialCoordType{T, D, NNMO}) where {T, D, NNMO} = 
eKinetic(bf1, bf2) + neAttraction(bf1, bf2, nuc, nucCoords)


"""

    coreH(bs::Union{GTBasis, Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                    AbstractVector{<:AbstractGTBasisFuncs{T, D}}}, 
          nuc::Union{Tuple{String, Vararg{String, NNMO}}, AbstractVector{String}}, 
          nucCoords::$(SpatialCoordType|>typeStrNotUnionAll)) where {T, D, NNMO} -> 
    Matrix{T}

Return the core Hamiltonian given a basis set and the corresponding nuclei with their 
coordinates (in atomic units).
"""
coreH(bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}, 
      nuc::AVectorOrNTuple{String, NNMO}, 
      nucCoords::SpatialCoordType{T, D, NNMO}) where {T, D, NNMO} = 
eKinetics(bs) + neAttractions(bs, nuc, nucCoords)

coreH(b::GTBasis{T, D}, 
      nuc::AVectorOrNTuple{String, NNMO}, 
      nucCoords::SpatialCoordType{T, D, NNMO}) where {T, D, NNMO} = 
neAttractions(b.basis, nuc, nucCoords) + b.Te