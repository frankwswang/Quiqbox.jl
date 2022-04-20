export gradHFenergy

using LinearAlgebra: eigen, Symmetric

function oneBodyDerivativeCore(::Val{false}, 
                               ∂bfs::Union{NTuple{BN,BT1},NTuple{BN,AbstractGTBasisFuncs}}, 
                               bfs::Union{NTuple{BN,BT2}, NTuple{BN,AbstractGTBasisFuncs}}, 
                               X::Matrix{Float64}, ∂X::Matrix{Float64}, 
                               tf::TypedFunction{F}) where 
                              {BN, BT1<:CompositeGTBasisFuncs{<:Any, 1}, 
                                   BT2<:CompositeGTBasisFuncs{<:Any, 1}, F}
    ʃ = getFunc(tf)
    ∂ʃ = ones(BN, BN)
    ʃab = ones(BN, BN)
    ∂ʃab = ones(BN, BN)
    for i = 1:BN, j = 1:i
       ʃab[i,j] = ʃab[j,i] = ʃ(bfs[i], bfs[j])
    end
    for i = 1:BN, j = 1:i
        ∂ʃab[i,j] = ∂ʃab[j,i] = ʃ(∂bfs[i], bfs[j]) + ʃ(bfs[i], ∂bfs[j])
    end
    @views begin
        for i=1:BN, j=1:i
            # X[i,j] == X[j,i]
            ∂ʃ[i,j] = ∂ʃ[j,i] = 
            transpose( X[:,i]) * ∂ʃab[:,:] *  X[:,j] +
            transpose(∂X[:,i]) *  ʃab[:,:] *  X[:,j] +
            transpose( X[:,i]) *  ʃab[:,:] * ∂X[:,j]
        end
    end
    ∂ʃ
end


function twoBodyDerivativeCore(::Val{false}, 
                               ∂bfs::Union{NTuple{BN,BT1},NTuple{BN,AbstractGTBasisFuncs}}, 
                               bfs::Union{NTuple{BN,BT2}, NTuple{BN,AbstractGTBasisFuncs}},
                               X::Matrix{Float64}, ∂X::Matrix{Float64}, 
                               tf::TypedFunction{F}) where 
                              {BN, BT1<:CompositeGTBasisFuncs{<:Any, 1}, 
                                   BT2<:CompositeGTBasisFuncs{<:Any, 1}, F}
    ʃ = getFunc(tf)
    bsSize = ∂bfs |> length
    ∂ʃ = ones(bsSize, bsSize, bsSize, bsSize)
    ʃabcd = ones(bsSize, bsSize, bsSize, bsSize)
    ʃ∂abcd = ones(bsSize, bsSize, bsSize, bsSize)
    for i = 1:bsSize, j = 1:i, k = 1:i, l = 1:(k==i ? j : k)
        ʃabcd[i,j,k,l] = ʃabcd[j,i,k,l] = ʃabcd[j,i,l,k] = ʃabcd[i,j,l,k] = 
        ʃabcd[l,k,i,j] = ʃabcd[k,l,i,j] = ʃabcd[k,l,j,i] = ʃabcd[l,k,j,i] = 
        ʃ(bfs[i],  bfs[j],  bfs[k],  bfs[l])
    end
    for l = 1:bsSize, k=1:l, j=1:bsSize, i=1:bsSize
        ʃ∂abcd[i,j,k,l] = ʃ∂abcd[i,j,l,k] = ʃ(∂bfs[i], bfs[j],  bfs[k],  bfs[l])
    end
    # [∂ʃ4[i,j,k,l] == ∂ʃ4[j,i,l,k] == ∂ʃ4[j,i,k,l] != ∂ʃ4[l,j,k,i]
    for i = 1:bsSize, j = 1:i, k = 1:i, l = 1:(k==i ? j : k)
        val = 0
        # ʃ∂abcd[i,j,k,l] == ʃ∂abcd[i,j,l,k] == ʃab∂cd[l,k,i,j] == ʃab∂cd[k,l,i,j]
        for a = 1:bsSize, b = 1:bsSize, c = 1:bsSize, d = 1:bsSize
            val += (  X[a,i]*X[b,j]*X[c,k]*X[d,l] + X[a,j]*X[b,i]*X[c,k]*X[d,l] + 
                      X[c,i]*X[d,j]*X[a,k]*X[b,l] + X[c,i]*X[d,j]*X[a,l]*X[b,k]  ) * 
                   ʃ∂abcd[a,b,c,d] + 
                   ( ∂X[a,i]*X[b,j]* X[c,k]*X[d,l] + X[a,i]*∂X[b,j]*X[c,k]* X[d,l] + 
                      X[a,i]*X[b,j]*∂X[c,k]*X[d,l] + X[a,i]* X[b,j]*X[c,k]*∂X[d,l] ) * 
                   ʃabcd[a,b,c,d]
        end
        ∂ʃ[i,j,k,l] = ∂ʃ[j,i,k,l] = ∂ʃ[j,i,l,k] = ∂ʃ[i,j,l,k] = 
        ∂ʃ[l,k,i,j] = ∂ʃ[k,l,i,j] = ∂ʃ[k,l,j,i] = ∂ʃ[l,k,j,i] = val
    end
    ∂ʃ
end


function deriveBasisFunc(bf::CompositeGTBasisFuncs{BN, 1}, par::ParamBox) where {BN}
    varDict = getVarDict(bf.param)
    vr = getVar(par)
    info = diffInfo(bf, vr, varDict)
    diffInfoToBasisFunc(bf, info)
end


function derivativeCore(FoutputIsVector::Val{B}, 
                        bs::Union{NTuple{BN, BT}, NTuple{BN, AbstractGTBasisFuncs}}, 
                        par::ParamBox, S::Matrix{Float64}, 
                        oneBodyF::TypedFunction{F1}, twoBodyF::TypedFunction{F2}) where 
                       {B, BN, BT<:AbstractGTBasisFuncs, F1, F2}
    # ijkl in chemists' notation of spatial bases (ij|kl).
    bfs = Tuple(hcat(decomposeCore.(Val(false), bs)...))
    ∂bfs = deriveBasisFunc.(bfs, par)
    bsSize = basisSize.(bs) |> sum
    ∂S = ones(bsSize, bsSize)
    ∂X = ones(bsSize, bsSize) # ∂X corresponds to the derivative of X = S^(-0.5)
    ∂X₀ = ones(bsSize, bsSize) # ∂X in its eigen basis
    for i=1:bsSize, j=1:i
        ∂S[i,j] = ∂S[j,i] = getOverlap(∂bfs[i], bfs[j]) + getOverlap(bfs[i], ∂bfs[j])
    end
    X = getXcore1(S)
    λ, 𝑣 = eigen(S|>Symmetric)
    ∂S2 = transpose(𝑣)*∂S*𝑣
    for i=1:bsSize, j=1:i
        ∂X₀[i,j] = ∂X₀[j,i] = (- ∂S2[i,j] * inv(sqrt(λ[i])) * inv(sqrt(λ[j])) * 
                               inv(sqrt(λ[i]) + sqrt(λ[j])))
    end
    for i=1:bsSize, j=1:bsSize
        ∂X[j,i] = [𝑣[j,k]*∂X₀[k,l]*𝑣[i,l] for k=1:bsSize, l=1:bsSize] |> sum
    end
    ∂ʃ2 = oneBodyDerivativeCore(FoutputIsVector, ∂bfs, bfs, X, ∂X, oneBodyF)
    ∂ʃ4 = twoBodyDerivativeCore(FoutputIsVector, ∂bfs, bfs, X, ∂X, twoBodyF)
    ∂ʃ2, ∂ʃ4
end


function ∂HFenergy(bs::Union{NTuple{BN, BT}, NTuple{BN, AbstractGTBasisFuncs}}, 
                   par::ParamBox, C::NTuple{HFTS, Matrix{Float64}}, 
                   S::Matrix{Float64}, nuc::NTuple{NN, String}, 
                   nucCoords::NTuple{NN, NTuple{3,Float64}}, 
                   nElectron::NTuple{HFTS, Int}) where 
                  {BN, BT<:AbstractGTBasisFuncs, HFTS, NN}
    Xinv = sqrt(S)::Matrix{Float64}
    cH = (i, j)->getCoreHij(i, j, nuc, nucCoords)
    ∂hij, ∂hijkl = derivativeCore(Val(false), bs, par, S, 
                                  TypedFunction(cH), TypedFunction(get2eInteraction))
    getEᵀ(∂hij, ∂hijkl, Ref(Xinv).*C, nElectron)
end


function gradHFenergy(bs::Union{NTuple{BN, BT}, NTuple{BN, AbstractGTBasisFuncs}, 
                                Vector{<:AbstractGTBasisFuncs}}, 
                      par::Vector{<:ParamBox}, 
                      C::NTuple{HFTS, Matrix{Float64}}, 
                      S::Matrix{Float64}, 
                      nuc::Union{NTuple{NN, String}, Vector{String}}, 
                      nucCoords::Union{NTuple{NN, NTuple{3,Float64}}, 
                                       Vector{<:AbstractArray{<:Real}}}, 
                      nElectron::Union{Int, NTuple{2, Int}}=getCharge(nuc)) where 
                     {BN, BT<:AbstractGTBasisFuncs, HFTS, NN}
    bs = arrayToTuple(bs)
    nuc = arrayToTuple(nuc)
    nucCoords = genTupleCoords(nucCoords)
    Ns = splitSpins(Val(HFTS), nElectron)
    ∂HFenergy.(Ref(bs), par, Ref(C), Ref(S), Ref(nuc), Ref(nucCoords), Ref(Ns))
end