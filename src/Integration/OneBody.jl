export overlap, overlaps, nucAttraction, nucAttractions, elecKinetic, elecKinetics, 
       coreH, coreHij


@inline function oneBodyBFTensorCore(libcinFunc::Symbol, 
                                     bf1::FloatingGTBasisFunc, bf2::FloatingGTBasisFunc, 
                                     nuclei::Vector{String}, 
                                     nucleiCoords::Vector{<:AbstractArray};
                                     isGradient::Bool=false)
    env = Float64[]
    atm = Int32[]
    bas = Int32[]
    #= 
    Add actual nuclei before bases can allow 
        natm = length(nuclei) 
    to save calculating overhead.
    Otherwise set 
        natm = length(nuclei) + length(BasisSet) 
    to include ghost atoms.
    =#
    addToDataChain!(env, atm, nuclei, nucleiCoords) 
    addToDataChain!(env, atm, bas, bf1)
    addToDataChain!(env, atm, bas, bf2)

    cintFunc!(Val(libcinFunc), 
              zeros(basisSize([bf1.subshell, bf2.subshell])..., 1+isGradient*2), 
              [0,1], atm, length(nuclei), bas, 2, env)
end


"""

    oneBodyBFTensor(libcinFunc::Symbol, b1::AbstractFloatingGTBasisFunc, 
                    b2::AbstractFloatingGTBasisFunc, nuclei::Array{String, 1}=String[], 
                    nucleiCoords::Array{<:AbstractArray, 1}=Array[]; 
                    isGradient::Bool=false) -> 
    Array{Float64, 2}

Core function for one-electron integrals.

`libcinFunc::Symbol` specifies the backend libcint (https://github.com/sunqm/libcint) 
function name, e.g. "int1e_nuc_cart" should be converted to `:int1e_nuc_cart`as the input 
argument. If the integral does not need the information of nuclei and their coordinates, 
those 2 arguments can be omitted. If the integral is a spacial gradient, `isGradient` 
should be set to `true`.
"""
@inline function oneBodyBFTensor(libcinFunc::Symbol, 
                                 b1::AbstractFloatingGTBasisFunc, 
                                 b2::AbstractFloatingGTBasisFunc, 
                                 nuclei::Vector{String}=String[], 
                                 nucleiCoords::Vector{<:AbstractArray}=Array[];
                                 isGradient::Bool=false)
    f = @inline function (i,j)
        ints = oneBodyBFTensorCore(libcinFunc, i, j, nuclei, nucleiCoords; isGradient)
        ints[ijkIndex(i), ijkIndex(j), :]
    end
    sum([f(i,j) for i in getBasisFuncs(b1), j in getBasisFuncs(b2)])
end


@inline function oneBodyBSTensor(BasisSet::Vector{<:AbstractFloatingGTBasisFunc}, 
                                 intFunc::F) where {F<:Function}
    subSize = basisSize(BasisSet) |> collect
    accuSize = vcat(0, accumulate(+, subSize))
    len = subSize |> sum
    nPage = (intFunc(BasisSet[1], BasisSet[1]) |> size)[3]
    tensor = cat(fill(-2*ones(len, len), nPage)..., dims=3)
    for i = 1:length(BasisSet), j = 1:i
        intTensor =  intFunc(BasisSet[i], BasisSet[j])
        for page = 1:nPage
            rowRange = accuSize[i]+1 : accuSize[i+1] 
            colRange = accuSize[j]+1 : accuSize[j+1]
            int = intTensor[:, :, page]
            tensor[rowRange, colRange, page] = int
            tensor[colRange, rowRange, page] = int |> transpose
        end
    end
    tensor
end


@inline overlapCore(bf1::AbstractFloatingGTBasisFunc, bf2::AbstractFloatingGTBasisFunc) = 
        oneBodyBFTensor(:cint1e_ovlp_cart, bf1, bf2)

"""

    overlap(fb1::AbstractFloatingGTBasisFunc, fb2::AbstractFloatingGTBasisFunc) -> 
    Array{Float64, 2}

Return the orbital overlap matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions.
"""
overlap(bf1::AbstractFloatingGTBasisFunc, bf2::AbstractFloatingGTBasisFunc) = 
dropdims(overlapCore(bf1, bf2), dims=3)


@inline overlapsCore(BSet::Vector{<:AbstractFloatingGTBasisFunc}) = 
        oneBodyBSTensor(BSet, overlapCore)

"""

    overlaps(BSet::Array{<:AbstractFloatingGTBasisFunc, 1}) -> Array{Float64, 2}

Return the orbital overlap matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Array`.
"""
overlaps(BSet::Vector{<:AbstractFloatingGTBasisFunc}) = dropdims(overlapsCore(BSet), dims=3)


@inline nucAttractionCore(bf1::AbstractFloatingGTBasisFunc, 
                          bf2::AbstractFloatingGTBasisFunc, 
                          nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
        oneBodyBFTensor(:cint1e_nuc_cart, bf1, bf2, nuc, nucCoords)

"""

    nucAttraction(fb1::AbstractFloatingGTBasisFunc, fb2::AbstractFloatingGTBasisFunc, 
                  nuc::Array{String, 1}, nucCoords::Array{<:AbstractArray, 1}) -> 
    Array{Float64, 2}

Return the nuclear attraction matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions, and the nuclei with their coordinates (in atomic unit).
"""
nucAttraction(bf1::AbstractFloatingGTBasisFunc, bf2::AbstractFloatingGTBasisFunc, 
              nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
dropdims(nucAttractionCore(bf1, bf2, nuc, nucCoords), dims=3)


@inline nucAttractionsCore(BSet::Vector{<:AbstractFloatingGTBasisFunc}, 
                           nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
        oneBodyBSTensor(BSet, @inline (bf1, bf2)->oneBodyBFTensor(:cint1e_nuc_cart, 
                                                                  bf1, bf2, nuc, nucCoords))

"""

    nucAttractions(BSet::Array{<:AbstractFloatingGTBasisFunc, 1}, nuc::Array{String, 1}, 
                   nucCoords::Array{<:AbstractArray, 1}) -> 
    Array{Float64, 2}

Return the nuclear attraction matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Array`, and the nuclei with their 
coordinates (in atomic unit).
"""
nucAttractions(BSet::Vector{<:AbstractFloatingGTBasisFunc}, 
               nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
dropdims(nucAttractionsCore(BSet, nuc, nucCoords), dims=3)


@inline elecKineticCore(bf1::AbstractFloatingGTBasisFunc, 
                        bf2::AbstractFloatingGTBasisFunc) = 
        oneBodyBFTensor(:cint1e_kin_cart, bf1, bf2)

"""

    elecKinetic(fb1::AbstractFloatingGTBasisFunc, fb2::AbstractFloatingGTBasisFunc) -> 
    Array{Float64, 2}

Return the electron kinetic energy matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given 2 basis functions.
"""
elecKinetic(bf1::AbstractFloatingGTBasisFunc, bf2::AbstractFloatingGTBasisFunc) = 
dropdims(elecKineticCore(bf1, bf2), dims=3)


@inline elecKineticsCore(BSet::Vector{<:AbstractFloatingGTBasisFunc}) = 
                     oneBodyBSTensor(BSet, elecKineticCore)

"""

    elecKinetics(BSet::Array{<:AbstractFloatingGTBasisFunc, 1}) -> Array{Float64, 2}

Return the electron kinetic energy matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Array`.
"""
elecKinetics(BSet::Vector{<:AbstractFloatingGTBasisFunc}) = 
dropdims(elecKineticsCore(BSet), dims=3)


@inline coreHijCore(bf1::AbstractFloatingGTBasisFunc, bf2::AbstractFloatingGTBasisFunc, 
                    nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
        elecKineticCore(bf1, bf2) + nucAttractionCore(bf1, bf2, nuc, nucCoords)

"""

    coreHij(fb1::AbstractFloatingGTBasisFunc, fb2::AbstractFloatingGTBasisFunc) -> 
    Array{Float64, 2}

Return a matrix element or block of the core Hamiltonian (an N×N `Matrix` where N is the 
number of spatial orbitals) given 2 basis functions.
"""
coreHij(bf1::AbstractFloatingGTBasisFunc, bf2::AbstractFloatingGTBasisFunc, 
        nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
dropdims(coreHijCore(bf1, bf2, nuc, nucCoords), dims=3)


@inline coreHCore(BSet::Vector{<:AbstractFloatingGTBasisFunc}, 
                  nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
        elecKineticsCore(BSet) + nucAttractionsCore(BSet, nuc, nucCoords)

"""

    coreH(BSet::Array{<:AbstractFloatingGTBasisFunc, 1}) -> Array{Float64, 2}

Return the core Hamiltonian matrix (an N×N `Matrix` where N is the number of spatial 
orbitals) given a basis set in the form of an `Array`.
"""
coreH(BSet::Vector{<:AbstractFloatingGTBasisFunc}, 
      nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
dropdims(coreHCore(BSet, nuc, nucCoords), dims=3)