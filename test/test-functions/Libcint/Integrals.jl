# One-body functions

@inline function oneBodyBFTensorCore(libcinFunc::Symbol, 
                                     bf1::Quiqbox.FGTBasisFuncsON{ON1}, 
                                     bf2::Quiqbox.FGTBasisFuncsON{ON2}, 
                                     nuclei::Vector{String}, 
                                     nucleiCoords::Vector{<:AbstractArray{<:Real}};
                                     isGradient::Bool=false) where {ON1, ON2}
    env = Float64[]
    atm = Int32[]
    bas = Int32[]
    #= 
    Add actual nuclei before bases can allow 
        natm = length(nuclei) 
    to save calculating overhead.
    Otherwise set 
        natm = length(nuclei) + length(BSet) 
    to include ghost atoms.
    =#
    addToDataChain!(env, atm, nuclei, nucleiCoords)
    addToDataChain!(env, atm, bas, bf1)
    addToDataChain!(env, atm, bas, bf2)

    cintFunc!(Val(libcinFunc), 
              zeros(Quiqbox.basisSize(bf1), Quiqbox.basisSize(bf2), 1+isGradient*2), 
              [0,1], atm, length(nuclei), bas, 2, env)
end


unpack(bfm::Quiqbox.BasisFuncMix) = collect(bfm.BasisFunc)
unpack(bf::Quiqbox.FloatingGTBasisFuncs) = [bf]
unpack(::Any) = Quiqbox.FloatingGTBasisFuncs[]

"""

    oneBodyBFTensor(libcinFunc::Symbol, b1::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                    b2::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                    nuclei::Vector{String}=String[], 
                    nucleiCoords::Vector{<:AbstractArray{<:Real}}=Vector{Float64}[]; 
                    isGradient::Bool=false) -> 
    Array{Float64, 3}

Core function for one-electron integrals.

`libcinFunc::Symbol` specifies the backend [libcint](https://github.com/sunqm/libcint) 
function name, e.g. `"int1e_nuc_cart"` should be converted to `:int1e_nuc_cart` as the 
input argument. If the integral does not need the information of nuclei and their 
coordinates, those 2 arguments can be omitted. If the integral is a spacial gradient, 
`isGradient` should be set to `true`.
"""
@inline function oneBodyBFTensor(libcinFunc::Symbol, 
                                 b1::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                                 b2::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                                 nuclei::Vector{String}=String[], 
                                 nucleiCoords::Vector{<:AbstractArray{<:Real}}=
                                             Vector{Float64}[];
                                 isGradient::Bool=false)
    f = @inline function (i,j)
        ints = oneBodyBFTensorCore(libcinFunc, i, j, nuclei, nucleiCoords; isGradient)
        ints[ijkIndex(i), ijkIndex(j), :]
    end
    sum([f(i,j) for i in unpack(b1), j in unpack(b2)])
end


@inline function oneBodyBSTensor(BSet::Vector{<:Quiqbox.AbstractGTBasisFuncs{Float64, 3}}, 
                                 intFunc::F) where {F<:Function}
    subSize = Quiqbox.basisSize.(BSet)
    accuSize = vcat(0, accumulate(+, subSize))
    len = subSize |> sum
    nPage = (intFunc(BSet[1], BSet[1]) |> size)[3]
    tensor = cat(fill(-2*ones(len, len), nPage)..., dims=3)
    for i = 1:length(BSet), j = 1:i
        intTensor =  intFunc(BSet[i], BSet[j])
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


@inline overlapCoreLibcint(bf1::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                           bf2::Quiqbox.AbstractGTBasisFuncs{Float64, 3}) = 
        oneBodyBFTensor(:cint1e_ovlp_cart, bf1, bf2)

overlapLibcint(bf1::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
               bf2::Quiqbox.AbstractGTBasisFuncs{Float64, 3}) = 
dropdims(overlapCoreLibcint(bf1, bf2), dims=3)


@inline overlapsCoreLibcint(BSet::Vector{<:Quiqbox.AbstractGTBasisFuncs{Float64, 3}}) = 
        oneBodyBSTensor(BSet, overlapCoreLibcint)

overlapsLibcint(BSet::Vector{<:Quiqbox.AbstractGTBasisFuncs{Float64, 3}}) = 
dropdims(overlapsCoreLibcint(BSet), dims=3)


@inline nucAttractionCoreLibcint(bf1::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                                 bf2::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                                 nuc::Vector{String}, 
                                 nucCoords::Vector{<:AbstractArray{<:Real}}) = 
        oneBodyBFTensor(:cint1e_nuc_cart, bf1, bf2, nuc, nucCoords)

nucAttractionLibcint(bf1::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                     bf2::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                     nuc::Vector{String}, nucCoords::Vector{<:AbstractArray{<:Real}}) = 
dropdims(nucAttractionCoreLibcint(bf1, bf2, nuc, nucCoords), dims=3)


@inline nucAttractionsCoreLibcint(BSet::Vector{<:Quiqbox.AbstractGTBasisFuncs{Float64, 3}}, 
                                  nuc::Vector{String}, 
                                  nucCoords::Vector{<:AbstractArray{<:Real}}) = 
        oneBodyBSTensor(BSet, @inline (bf1, bf2)->oneBodyBFTensor(:cint1e_nuc_cart, 
                                                                  bf1, bf2, nuc, nucCoords))

nucAttractionsLibcint(BSet::Vector{<:Quiqbox.AbstractGTBasisFuncs{Float64, 3}}, 
                      nuc::Vector{String}, nucCoords::Vector{<:AbstractArray{<:Real}}) = 
dropdims(nucAttractionsCoreLibcint(BSet, nuc, nucCoords), dims=3)


@inline elecKineticCoreLibcint(bf1::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                               bf2::Quiqbox.AbstractGTBasisFuncs{Float64, 3}) = 
        oneBodyBFTensor(:cint1e_kin_cart, bf1, bf2)

elecKineticLibcint(bf1::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                   bf2::Quiqbox.AbstractGTBasisFuncs{Float64, 3}) = 
dropdims(elecKineticCoreLibcint(bf1, bf2), dims=3)


@inline elecKineticsCoreLibcint(BSet::Vector{<:Quiqbox.AbstractGTBasisFuncs{Float64, 3}}) = 
        oneBodyBSTensor(BSet, elecKineticCoreLibcint)

elecKineticsLibcint(BSet::Vector{<:Quiqbox.AbstractGTBasisFuncs{Float64, 3}}) = 
dropdims(elecKineticsCoreLibcint(BSet), dims=3)


# Two-body functions

@inline function twoBodyBFTensorCore(libcinFunc::Symbol, 
                                     bf1::Quiqbox.FGTBasisFuncsON{ON1}, 
                                     bf2::Quiqbox.FGTBasisFuncsON{ON2}, 
                                     bf3::Quiqbox.FGTBasisFuncsON{ON3}, 
                                     bf4::Quiqbox.FGTBasisFuncsON{ON4}; 
                                     isGradient::Bool=false) where {ON1, ON2, ON3, ON4}
    env = Float64[]
    atm = Int32[]
    bas = Int32[]
    subSize = Quiqbox.basisSize.((bf1, bf2, bf3, bf4))

    id, uniqueBFs = Quiqbox.markUnique([bf1, bf2, bf3, bf4])

    for bf in uniqueBFs
        addToDataChain!(env, atm, bas, bf)
    end

    cintFunc!(Val(libcinFunc), (subSize..., 1+isGradient*2)|>zeros, 
              id .- 1, atm, 0, bas, length(uniqueBFs), env)
end


"""

    twoBodyBFTensor(libcinFunc::Symbol, 
                    b1::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                    b2::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                    b3::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                    b4::Quiqbox.AbstractGTBasisFuncs{Float64, 3}; 
                    isGradient::Bool=false) -> 
    Array{Float64, 5}

Core function for one-electron integrals.

`libcinFunc::Symbol` specifies the backend [libcint](https://github.com/sunqm/libcint) 
function name, e.g. `"cint2e_cart"` should be converted to `:cint2e_cart` as the input 
argument.
"""
@inline function twoBodyBFTensor(libcinFunc::Symbol, 
                                 b1::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                                 b2::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                                 b3::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                                 b4::Quiqbox.AbstractGTBasisFuncs{Float64, 3}; 
                                 isGradient::Bool=false)
    f = @inline function (i,j,k,l)
        ints = twoBodyBFTensorCore(libcinFunc, i, j, k, l; isGradient)
        ints[ijkIndex(i), ijkIndex(j), ijkIndex(k), ijkIndex(l),:]
    end
    sum([f(i,j,k,l) for i in unpack(b1), j in unpack(b2), 
                        k in unpack(b3), l in unpack(b4)])
end


@inline function twoBodyBSTensor(BSet::Vector{<:Quiqbox.AbstractGTBasisFuncs{Float64, 3}}, 
                                 intFunc::F; outputUniqueIndices::Bool=false) where 
                                {F<:Function}
    subSize = Quiqbox.basisSize.(BSet) |> collect
    accuSize = vcat(0, accumulate(+, subSize))
    totalSize = subSize |> sum
    nPage = (intFunc(BSet[1], BSet[1], BSet[1], BSet[1]) |> size)[5]
    buf = ones(totalSize, totalSize, totalSize, totalSize, nPage)
    for i = 1:length(BSet), j = 1:i, k = 1:i, l = 1:ifelse(k==i, j, k)
        I = accuSize[i]+1 : accuSize[i+1]
        J = accuSize[j]+1 : accuSize[j+1]
        K = accuSize[k]+1 : accuSize[k+1]
        L = accuSize[l]+1 : accuSize[l+1]
        subBuf = intFunc(BSet[i], BSet[j], BSet[k], BSet[l])
        for page = 1:nPage
            buf[I,J,K,L,page:page] .= subBuf
            buf[J,I,K,L,page:page] .= PermutedDimsArray(subBuf, [2,1,3,4,5])
            buf[J,I,L,K,page:page] .= PermutedDimsArray(subBuf, [2,1,4,3,5])
            buf[I,J,L,K,page:page] .= PermutedDimsArray(subBuf, [1,2,4,3,5])
            buf[L,K,I,J,page:page] .= PermutedDimsArray(subBuf, [4,3,1,2,5])
            buf[K,L,I,J,page:page] .= PermutedDimsArray(subBuf, [3,4,1,2,5])
            buf[K,L,J,I,page:page] .= PermutedDimsArray(subBuf, [3,4,2,1,5])
            buf[L,K,J,I,page:page] .= PermutedDimsArray(subBuf, [4,3,2,1,5])
        end
    end
    if outputUniqueIndices
        s = sum(subSize)
        uniqueInts = fill(Int[0,0,0,0,0], 
                          (3*binomial(s, 4)+6*binomial(s, 3)+4*binomial(s, 2)+s)*nPage)
        index = 1
        for i = 1:s, j = 1:i, k = 1:i, l = 1:ifelse(k==i, j, k), p=1:nPage
            uniqueInts[index] = [i, j, k, l, p]
            index += 1
        end
    end
    outputUniqueIndices ? (return buf, uniqueInts) : (return buf)
end


@inline eeInteractionCoreLibcint(bf1::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                                 bf2::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                                 bf3::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                                 bf4::Quiqbox.AbstractGTBasisFuncs{Float64, 3}) = 
        twoBodyBFTensor(:cint2e_cart, bf1, bf2, bf3, bf4)

eeInteractionLibcint(bf1::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                     bf2::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                     bf3::Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 
                     bf4::Quiqbox.AbstractGTBasisFuncs{Float64, 3}) = 
dropdims(eeInteractionCoreLibcint(bf1, bf2, bf3, bf4), dims=5)


"""

    eeInteractionsCoreLibcint(BSet::Array{<:Quiqbox.AbstractGTBasisFuncs{Float64, 3}, 1}; 
                       outputUniqueIndices::Bool=false) -> 
    Array{Float64, 5}, [Array{<:Array{Int, 1}, 1}]

Return the electron-electron interaction tensor (an N×N×N×N×1 Tensor where N is the number 
of spatial orbitals) given a basis set in the form of an `Array`.

If `outputUniqueIndices=true`, additionally return the indices for all the unique integrals.
"""
@inline eeInteractionsCoreLibcint(BSet::Vector{<:Quiqbox.AbstractGTBasisFuncs{Float64, 3}}; 
                                  outputUniqueIndices::Bool=false) = 
        twoBodyBSTensor(BSet, eeInteractionCoreLibcint; outputUniqueIndices)

eeInteractionsLibcint(BSet::Vector{<:Quiqbox.AbstractGTBasisFuncs{Float64, 3}}) = 
dropdims(eeInteractionsCoreLibcint(BSet), dims=5)