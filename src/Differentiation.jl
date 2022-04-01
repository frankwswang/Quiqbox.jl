export gradHFenergy

using LinearAlgebra: eigen, Symmetric

function oneBodyDerivativeCore(::Val{false}, ∂bfs::Vector{<:CompositeGTBasisFuncs}, 
                               bfs::Vector{<:CompositeGTBasisFuncs}, 
                               X::Matrix{Float64}, ∂X::Matrix{Float64}, 
                               ft::FunctionType{F}) where {F}
    ʃ = getFunc(ft.f)
    bsSize = ∂bfs |> length
    ∂ʃ = ones(bsSize, bsSize)
    ʃab = ones(bsSize, bsSize)
    ∂ʃab = ones(bsSize, bsSize)
    for i = 1:bsSize, j = 1:i
       ʃab[i,j] = ʃab[j,i] = ʃ(bfs[i], bfs[j])
    end
    for i = 1:bsSize, j = 1:i
        ∂ʃab[i,j] = ∂ʃab[j,i] = ʃ(∂bfs[i], bfs[j]) + ʃ(bfs[i], ∂bfs[j])
    end
    @views begin
        for i=1:bsSize, j=1:i
            # X[i,j] == X[j,i]
            ∂ʃ[i,j] = ∂ʃ[j,i] = 
            transpose( X[:,i]) * ∂ʃab[:,:] *  X[:,j] +
            transpose(∂X[:,i]) *  ʃab[:,:] *  X[:,j] +
            transpose( X[:,i]) *  ʃab[:,:] * ∂X[:,j]
        end
    end
    ∂ʃ
end


function twoBodyDerivativeCore(::Val{false}, ∂bfs::Vector{<:CompositeGTBasisFuncs}, 
                               bfs::Vector{<:CompositeGTBasisFuncs}, 
                               X::Matrix{Float64}, ∂X::Matrix{Float64}, 
                               ft::FunctionType{F}) where {F}
    ʃ = getFunc(ft.f)
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
    varDict = getVarDict(bf)
    vr = getVar(par)
    info = diffInfo(bf, vr, varDict)
    diffInfoToBasisFunc(bf, info)
end


function derivativeCore(FoutputIsVector::Val{B}, 
                        bs::Vector{<:CompositeGTBasisFuncs}, par::ParamBox, 
                        S::Matrix{Float64}, 
                        oneBodyF::FunctionType{F1}, twoBodyF::FunctionType{F2}) where 
                       {B, F1, F2}
    # ijkl in chemists' notation of spatial bases (ij|kl).
    bfs = reshape(hcat(decompose.(bs)...), :)
    ∂bfs = deriveBasisFunc.(bfs, par)
    bsSize = basisSize.(bs) |> sum
    ∂S = ones(bsSize, bsSize)
    ∂X = ones(bsSize, bsSize) # ∂X corresponds to the derivative of X = S^(-0.5)
    ∂X₀ = ones(bsSize, bsSize) # ∂X in its eigen basis
    for i=1:bsSize, j=1:i
        ∂S[i,j] = ∂S[j,i] = getOverlap(∂bfs[i], bfs[j]) + getOverlap(bfs[i], ∂bfs[j])
    end
    X = (S^(-0.5))::Symmetric{Float64, Matrix{Float64}} |> Array
    λ, 𝑣 = eigen(S|>Symmetric)
    ∂S2 = transpose(𝑣)*∂S*𝑣
    for i=1:bsSize, j=1:i
        ∂X₀[i,j] = ∂X₀[j,i] = (- ∂S2[i,j] * λ[i]^(-0.5) * λ[j]^(-0.5) * 
                               (λ[i]^0.5 + λ[j]^0.5)^(-1))
    end
    for i=1:bsSize, j=1:bsSize
        ∂X[j,i] = [𝑣[j,k]*∂X₀[k,l]*𝑣[i,l] for k=1:bsSize, l=1:bsSize] |> sum
    end
    ∂ʃ2 = oneBodyDerivativeCore(FoutputIsVector, ∂bfs, bfs, X, ∂X, oneBodyF)
    ∂ʃ4 = twoBodyDerivativeCore(FoutputIsVector, ∂bfs, bfs, X, ∂X, twoBodyF)
    ∂ʃ2, ∂ʃ4
end


function ∂HFenergy(bs::Vector{<:CompositeGTBasisFuncs}, par::ParamBox, 
                   C::Union{Matrix{Float64}, NTuple{2, Matrix{Float64}}}, 
                   S::Matrix{Float64}, nuc::Vector{String}, 
                   nucCoords::Vector{<:AbstractArray}, 
                   nElectron::Union{Int, NTuple{2, Int}})
    Xinv = sqrt(S)::Matrix{Float64}
    Cₓ = (C isa Tuple) ? (Ref(Xinv) .* C) : (Xinv * C)
    cH = (i, j)->getCoreHij(i, j, nuc, nucCoords)
    ∂hij, ∂hijkl = derivativeCore(Val(false), bs, par, S, 
                                  FunctionType(cH), FunctionType{:get2eInteraction}())
    getEᵀ(∂hij, ∂hijkl, Cₓ, nElectron)
end


function gradHFenergy(bs::Vector{<:CompositeGTBasisFuncs}, par::Vector{<:ParamBox}, 
                      C::Union{Matrix{Float64}, NTuple{2, Matrix{Float64}}}, 
                      S::Matrix{Float64}, nuc::Vector{String}, 
                      nucCoords::Vector{<:AbstractArray}, 
                      nElectron::Union{Int, NTuple{2, Int}}=getCharge(nuc))
    if length(C) == 2 && nElectron isa Int
        nElectron = (nElectron÷2, nElectron-nElectron÷2)
    end
    ∂HFenergy.(Ref(bs), par, Ref(C), Ref(S), Ref(nuc), Ref(nucCoords), Ref(nElectron))
end

gradHFenergy(bs::Vector{<:CompositeGTBasisFuncs}, par::ParamBox, 
            C::Union{Matrix{Float64}, NTuple{2, Matrix{Float64}}}, S::Matrix{Float64}, 
            nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}, 
            nElectron::Union{Int, NTuple{2, Int}}=getCharge(nuc)) = 
gradHFenergy(bs, [par], C, S, nuc, nucCoords, nElectron)