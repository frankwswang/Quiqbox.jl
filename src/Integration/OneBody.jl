export overlap, overlaps, nucAttraction, nucAttractions, elecKinetic, elecKinetics, coreH, coreHij


function oneBodyBFTensorCore(libcinFunc::Val, bf1::FloatingGTBasisFunc, bf2::FloatingGTBasisFunc, 
                             nuclei::Vector{String}, nucleiCoords::Vector{<:AbstractArray};
                             isGradient::Bool=false)
    env = Float64[]
    atm = Int32[]
    bas = Int32[]
    #= 
    Add actuall nuclei before bases can allow 
        natm = length(nuclei) 
    to save calculating overhead.
    Otherwise set 
        natm = length(nuclei) + length(BasisSet) 
    to include ghost atoms.
    =#
    addToDataChain!(env, atm, nuclei, nucleiCoords) 
    addToDataChain!(env, atm, bas, bf1)
    addToDataChain!(env, atm, bas, bf2)

    cintFunc!(libcinFunc, zeros(basisSize([bf1.subshell, bf2.subshell])..., 1+isGradient*2), 
              [0,1], atm, length(nuclei), bas, 2, env)
end


function oneBodyBFTensor(libcinFunc::Val, b1::AbstractFloatingGTBasisFunc, b2::AbstractFloatingGTBasisFunc, 
                         nuclei::Vector{String}, nucleiCoords::Vector{<:AbstractArray};
                         isGradient::Bool=false)
    f = @inline function (i,j)
        ints = oneBodyBFTensorCore(libcinFunc, i, j, nuclei, nucleiCoords; isGradient)
        ints[ijkIndex(i), ijkIndex(j), :]
    end
    sum([f(i,j) for i in getBasisFuncs(b1), j in getBasisFuncs(b2)])
end


function oneBodyBSTensor(BasisSet::Vector{<:AbstractFloatingGTBasisFunc}, 
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


"""

    overlap(fb1::AbstractFloatingGTBasisFunc, fb2::AbstractFloatingGTBasisFunc) -> Array{Float64, 3}

Return the orbital overlap matrix (an N×N×1 Tensor where N is the number of spatial orbitals) given 2 basis functions.
"""
overlap(bf1::AbstractFloatingGTBasisFunc, bf2::AbstractFloatingGTBasisFunc) = 
oneBodyBFTensor(Val(:cint1e_ovlp_cart), bf1, bf2, String[], Array[])


"""

    overlaps(BSet::Array{<:AbstractFloatingGTBasisFunc, 1}) -> Array{Float64, 3}

Return the orbital overlap matrix (an N×N×1 Tensor where N is the number of spatial orbitals) given a basis set in the form of an `Array`.
"""
overlaps(BSet::Vector{<:AbstractFloatingGTBasisFunc}) = 
oneBodyBSTensor(BSet, overlap)


"""

    nucAttraction(fb1::AbstractFloatingGTBasisFunc, fb2::AbstractFloatingGTBasisFunc, nuc::Array{String, 1}, nucCoords::Array{<:AbstractArray, 1}) -> Array{Float64, 3}

Return the nuclear attraction matrix (an N×N×1 Tensor where N is the number of spatial orbitals) given 2 basis functions, 
and the nuclei with their coordinates (in atomic unit).
"""
nucAttraction(bf1::AbstractFloatingGTBasisFunc, bf2::AbstractFloatingGTBasisFunc, nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
oneBodyBFTensor(Val(:cint1e_nuc_cart), bf1, bf2, nuc, nucCoords)


"""

    nucAttractions(BSet::Array{<:AbstractFloatingGTBasisFunc, 1}, nuc::Array{String, 1}, nucCoords::Array{<:AbstractArray, 1}) -> Array{Float64, 3}

Return the nuclear attraction matrix (an N×N×1 Tensor where N is the number of spatial orbitals) given a basis set in the form of an `Array`, 
and the nuclei with their coordinates (in atomic unit).
"""
nucAttractions(BSet::Vector{<:AbstractFloatingGTBasisFunc}, nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
oneBodyBSTensor(BSet, (bf1, bf2)->oneBodyBFTensor(Val(:cint1e_nuc_cart), bf1, bf2, nuc, nucCoords))


"""

    elecKinetic(fb1::AbstractFloatingGTBasisFunc, fb2::AbstractFloatingGTBasisFunc) -> Array{Float64, 3}

Return the electron kinetic energy matrix (an N×N×1 Tensor where N is the number of spatial orbitals) given 2 basis functions.
"""
elecKinetic(bf1::AbstractFloatingGTBasisFunc, bf2::AbstractFloatingGTBasisFunc) = 
oneBodyBFTensor(Val(:cint1e_kin_cart), bf1, bf2, String[], Array[])


"""

    elecKinetics(BSet::Array{<:AbstractFloatingGTBasisFunc, 1}) -> Array{Float64, 3}

Return the electron kinetic energy matrix (an N×N×1 Tensor where N is the number of spatial orbitals) given a basis set in the form of an `Array`.
"""
elecKinetics(BSet::Vector{<:AbstractFloatingGTBasisFunc}) = 
oneBodyBSTensor(BSet, elecKinetic)


"""

    coreHij(fb1::AbstractFloatingGTBasisFunc, fb2::AbstractFloatingGTBasisFunc) -> Array{Float64, 3}

Return a matrix element or block of the core Hamiltonian (an N×N×1 Tensor where N is the number of spatial orbitals) given 2 basis functions.
"""
coreHij(bf1::AbstractFloatingGTBasisFunc, bf2::AbstractFloatingGTBasisFunc, nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
elecKinetic(bf1, bf2) + nucAttraction(bf1, bf2, nuc, nucCoords)


"""

    coreH(BSet::Array{<:AbstractFloatingGTBasisFunc, 1}) -> Array{Float64, 3}

Return the core Hamiltonian matrix (an N×N×1 Tensor where N is the number of spatial orbitals) given a basis set in the form of an `Array`.
"""
coreH(BSet::Vector{<:AbstractFloatingGTBasisFunc}, nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
elecKinetics(BSet) + nucAttractions(BSet, nuc, nucCoords)