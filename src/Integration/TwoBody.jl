export eeInteraction, eeInteractions


@inline function twoBodyBFTensorCore(libcinFunc::Symbol, 
                                     bf1::FloatingGTBasisFuncs{<:Any, <:Any, ON1}, 
                                     bf2::FloatingGTBasisFuncs{<:Any, <:Any, ON2}, 
                                     bf3::FloatingGTBasisFuncs{<:Any, <:Any, ON3}, 
                                     bf4::FloatingGTBasisFuncs{<:Any, <:Any, ON4}; 
                                     isGradient::Bool=false) where {ON1, ON2, ON3, ON4}
    env = Float64[]
    atm = Int32[]
    bas = Int32[]
    subSize = basisSize.((bf1, bf2, bf3, bf4))

    id, uniqueBFs = markUnique([bf1, bf2, bf3, bf4])

    for bf in uniqueBFs
        addToDataChain!(env, atm, bas, bf)
    end

    cintFunc!(Val(libcinFunc), (subSize..., 1+isGradient*2)|>zeros, 
              id .- 1, atm, 0, bas, length(uniqueBFs), env)
end


"""

    twoBodyBFTensor(libcinFunc::Symbol, 
                    b1::AbstractGTBasisFuncs, b2::AbstractGTBasisFuncs, 
                    b3::AbstractGTBasisFuncs, b4::AbstractGTBasisFuncs; 
                    isGradient::Bool=false) -> 
    Array{Float64, 5}

Core function for one-electron integrals.

`libcinFunc::Symbol` specifies the backend [libcint](https://github.com/sunqm/libcint) 
function name, e.g. `"cint2e_cart"` should be converted to `:cint2e_cart` as the input 
argument.

**WARNING: This function might be completely removed in the future release.**
"""
@inline function twoBodyBFTensor(libcinFunc::Symbol, 
                                 b1::AbstractGTBasisFuncs, 
                                 b2::AbstractGTBasisFuncs, 
                                 b3::AbstractGTBasisFuncs, 
                                 b4::AbstractGTBasisFuncs; 
                                 isGradient::Bool=false)
    f = @inline function (i,j,k,l)
        ints = twoBodyBFTensorCore(libcinFunc, i, j, k, l; isGradient)
        ints[ijkIndex(i), ijkIndex(j), ijkIndex(k), ijkIndex(l),:]
    end
    sum([f(i,j,k,l) for i in unpackBasisFuncs(b1), j in unpackBasisFuncs(b2), 
                        k in unpackBasisFuncs(b3), l in unpackBasisFuncs(b4)])
end


@inline function twoBodyBSTensor(BasisSet::Vector{<:AbstractGTBasisFuncs}, 
                                 intFunc::F; outputUniqueIndices::Bool=false) where 
                                {F<:Function}
    subSize = basisSize.(BasisSet) |> collect
    accuSize = vcat(0, accumulate(+, subSize))
    totalSize = subSize |> sum
    nPage = (intFunc(BasisSet[1], BasisSet[1], BasisSet[1], BasisSet[1]) |> size)[5]
    buf = ones(totalSize, totalSize, totalSize, totalSize, nPage)
    for i = 1:length(BasisSet), j = 1:i, k = 1:i, l = 1:(k==i ? j : k)
        I = accuSize[i]+1 : accuSize[i+1]
        J = accuSize[j]+1 : accuSize[j+1]
        K = accuSize[k]+1 : accuSize[k+1]
        L = accuSize[l]+1 : accuSize[l+1]
        subBuf = intFunc(BasisSet[i], BasisSet[j], BasisSet[k], BasisSet[l])
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
        for i = 1:s, j = 1:i, k = 1:i, l = 1:(k==i ? j : k), p=1:nPage
            uniqueInts[index] = [i, j, k, l, p]
            index += 1
        end
    end
    outputUniqueIndices ? (return buf, uniqueInts) : (return buf)
end


@inline eeInteractionCore(bf1::AbstractGTBasisFuncs, 
                          bf2::AbstractGTBasisFuncs, 
                          bf3::AbstractGTBasisFuncs, 
                          bf4::AbstractGTBasisFuncs) = 
        twoBodyBFTensor(:cint2e_cart, bf1, bf2, bf3, bf4)

"""

    eeInteraction(bf1::AbstractGTBasisFuncs, 
                  bf2::AbstractGTBasisFuncs, 
                  bf3::AbstractGTBasisFuncs, 
                  bf4::AbstractGTBasisFuncs) -> 
    Array{Float64, 4}

Return the electron-electron interaction tensor (an N×N×N×N Tensor where N is the number of 
spatial orbitals) given 4 basis functions.
"""
eeInteraction(bf1::AbstractGTBasisFuncs, 
              bf2::AbstractGTBasisFuncs, 
              bf3::AbstractGTBasisFuncs, 
              bf4::AbstractGTBasisFuncs) = 
cat(get2eInteraction(bf1, bf2, bf3, bf4), dims=4)
"""

    eeInteractionsCore(BSet::Array{<:AbstractGTBasisFuncs, 1}; 
                       outputUniqueIndices::Bool=false) -> 
    Array{Float64, 5}, [Array{<:Array{Int, 1}, 1}]

Return the electron-electron interaction tensor (an N×N×N×N×1 Tensor where N is the number 
of spatial orbitals) given a basis set in the form of an `Array`.

If `outputUniqueIndices=true`, additionally return the indices for all the unique integrals.
"""
@inline eeInteractionsCore(BSet::Vector{<:AbstractGTBasisFuncs}; 
                          outputUniqueIndices::Bool=false) = 
        twoBodyBSTensor(BSet, eeInteractionCore; outputUniqueIndices)

"""

    eeInteractions(BSet::Array{<:AbstractGTBasisFuncs, 1}) -> Array{Float64, 4}

Return the electron-electron interaction tensor (an N×N×N×N Tensor where N is the number 
of spatial orbitals) given a basis set in the form of an `Array`. 
"""
eeInteractions(BSet::Vector{<:AbstractGTBasisFuncs}) = 
get2eInteractions(BSet)