export gradOfHFenergy

using LinearAlgebra: eigen, Symmetric, Hermitian
using ForwardDiff: derivative as ForwardDerivative
using Tullio: @tullio

function oneBodyDerivativeCore(::Val{false}, 
                               ∂bfs::AbstractVector{<:GTBasisFuncs{T, D, 1}}, 
                               bfs::AbstractVector{<:GTBasisFuncs{T, D, 1}}, 
                               X::AbstractMatrix{T}, ∂X::AbstractMatrix{T}, 
                               ʃ::F) where {T, D, F<:Function}
    BN = length(bfs)
    ∂ʃ = Array{T}(undef, BN, BN)
    ʃab = Array{T}(undef, BN, BN)
    ∂ʃab = Array{T}(undef, BN, BN)
    @sync for i = 1:BN, j = 1:i
        Threads.@spawn ʃab[i,j] = ʃab[j,i] = ʃ(bfs[i], bfs[j])
    end
    @sync for i = 1:BN, j = 1:i
        Threads.@spawn ∂ʃab[i,j] = ∂ʃab[j,i] = ʃ(∂bfs[i], bfs[j]) + ʃ(bfs[i], ∂bfs[j])
    end
    @views begin
        @sync for i=1:BN
            Threads.@spawn for j=1:i # Spawn here is faster than spawn inside the loop.
                # X[i,j] == X[j,i]
                @inbounds ∂ʃ[i,j] = ∂ʃ[j,i] = X[:,i]' * ∂ʃab *  X[:,j] + 
                                             ∂X[:,i]' *  ʃab *  X[:,j] + 
                                              X[:,i]' *  ʃab * ∂X[:,j]
            end
        end
    end
    ∂ʃ
end


function twoBodyDerivativeCore(::Val{false}, 
                               ∂bfs::AbstractVector{<:GTBasisFuncs{T, D, 1}}, 
                               bfs::AbstractVector{<:GTBasisFuncs{T, D, 1}}, 
                               X::AbstractMatrix{T}, ∂X::AbstractMatrix{T}, 
                               ʃ::F) where {T, D, F<:Function}
    BN = length(bfs)
    ∂ʃ = Array{T}(undef, BN, BN, BN, BN)
    ʃabcd = Array{T}(undef, BN, BN, BN, BN)
    ʃ∂abcd = Array{T}(undef, BN, BN, BN, BN)

    # ijkl in the chemists' notation of spatial bases (ij|kl).
    @sync for i = 1:BN, j = 1:i, k = 1:i, l = 1:ifelse(k==i, j, k)
        Threads.@spawn begin
            ʃabcd[i,j,k,l] = ʃabcd[j,i,k,l] = ʃabcd[j,i,l,k] = ʃabcd[i,j,l,k] = 
            ʃabcd[l,k,i,j] = ʃabcd[k,l,i,j] = ʃabcd[k,l,j,i] = ʃabcd[l,k,j,i] = 
            ʃ(bfs[i],  bfs[j],  bfs[k],  bfs[l])
        end
    end
    @sync for l = 1:BN, k=1:l, j=1:BN, i=1:BN
        Threads.@spawn begin
            ʃ∂abcd[i,j,l,k] = ʃ∂abcd[i,j,k,l] = ʃ(∂bfs[i], bfs[j],  bfs[k],  bfs[l])
        end
    end
    # [∂ʃ4[i,j,k,l] == ∂ʃ4[j,i,l,k] == ∂ʃ4[j,i,k,l] != ∂ʃ4[l,j,k,i]
    for i = 1:BN, j = 1:i, k = 1:i, l = 1:ifelse(k==i, j, k)
        # ʃ∂abcd[i,j,k,l] == ʃ∂abcd[i,j,l,k] == ʃab∂cd[l,k,i,j] == ʃab∂cd[k,l,i,j]
        @tullio val := begin
            @inbounds ( X[a,$i]* X[b,$j]* X[c,$k]* X[d,$l] + 
                        X[a,$j]* X[b,$i]* X[c,$k]* X[d,$l] + 
                        X[c,$i]* X[d,$j]* X[a,$k]* X[b,$l] + 
                        X[c,$i]* X[d,$j]* X[a,$l]* X[b,$k]  ) *ʃ∂abcd[a,b,c,d] + 
                      (∂X[a,$i]* X[b,$j]* X[c,$k]* X[d,$l] + 
                        X[a,$i]*∂X[b,$j]* X[c,$k]* X[d,$l] + 
                        X[a,$i]* X[b,$j]*∂X[c,$k]* X[d,$l] + 
                        X[a,$i]* X[b,$j]* X[c,$k]*∂X[d,$l]  ) * ʃabcd[a,b,c,d]
        end

        ∂ʃ[i,j,k,l] = ∂ʃ[j,i,k,l] = ∂ʃ[j,i,l,k] = ∂ʃ[i,j,l,k] = 
        ∂ʃ[l,k,i,j] = ∂ʃ[k,l,i,j] = ∂ʃ[k,l,j,i] = ∂ʃ[l,k,j,i] = val
    end
    ∂ʃ
end


function derivativeCore(FoutputIsVector::Val{B}, 
                        bfs::AbstractVector{<:GTBasisFuncs{T, D, 1}}, 
                        par::ParamBox, S::AbstractMatrix{T}, 
                        ʃ2::F1, ʃ4::F2) where {B, T, D, F1<:Function, F2<:Function}
    BN = length(bfs)
    ∂bfs = ∂Basis.(par, bfs)
    ∂S = Array{T}(undef, BN, BN)
    ∂X = Array{T}(undef, BN, BN) # ∂X corresponds to the derivative of X = S^(-0.5)
    ∂X₀ = Array{T}(undef, BN, BN) # ∂X in its eigen basis
    @sync for i=1:BN, j=1:i
        Threads.@spawn begin
            ∂S[i,j] = ∂S[j,i] = overlap(∂bfs[i], bfs[j]) + overlap(bfs[i], ∂bfs[j])
        end
    end
    X = getXcore1(S)
    λ, 𝑣 = eigen(S|>Hermitian)
    ∂S2 = 𝑣'*∂S*𝑣
    for i=1:BN, j=1:i # Faster without multi-threading
        @inbounds ∂X₀[i,j] = ∂X₀[j,i] = ( -∂S2[i,j] / ( sqrt(λ[i]) * sqrt(λ[j]) * 
                                          (sqrt(λ[i]) + sqrt(λ[j])) ) )
    end
    ∂X = 𝑣*∂X₀*𝑣'
    ∂ʃ2 = oneBodyDerivativeCore(FoutputIsVector, ∂bfs, bfs, X, ∂X, ʃ2)
    ∂ʃ4 = twoBodyDerivativeCore(FoutputIsVector, ∂bfs, bfs, X, ∂X, ʃ4)
    ∂ʃ2, ∂ʃ4
end


function ∂HFenergy(par::ParamBox{T}, 
                   bs::AbstractVector{<:GTBasisFuncs{T, D, 1}}, 
                   S::AbstractMatrix{T}, 
                   C::NTuple{HFTS, AbstractMatrix{T}}, 
                   nuc::NTuple{NN, String}, 
                   nucCoords::NTuple{NN, NTuple{D, T}}, 
                   N::NTuple{HFTS, Int}) where {T, D, HFTS, NN}
    Xinv = sqrt(S)::Matrix{T} # necessary assertion for type stability
    cH = (i, j)->coreHij(i, j, nuc, nucCoords)
    ∂hij, ∂hijkl = derivativeCore(Val(false), bs, par, S, cH, eeInteraction)
    getEᵗ(∂hij, ∂hijkl, Ref(Xinv).*C, N)
end


"""

    gradOfHFenergy(par::AbstractVector{<:ParamBox{T}}, HFres::HFfinalVars{T}) where {T} -> 
    AbstractVector{T}

Given a Hartree-Fock approximation result `HFres`, return the gradient of the Hartree-Fock 
energy with respect to a collection of parameters `par`. Specifically, for any 
[`ParamBox`](@ref) in `par`, unlike other cases where the it is always the output variable 
that is represented by the `ParamBox`, here the corresponding independent variable is 
represented by the `ParamBox`, so when the `ParamBox` is marked as differentiable (i.e., 
[`isDiffParam`](@ref) returns `true`), the variable it represents switches to its input 
variable.
"""

gradOfHFenergy(par::AbstractVector{<:ParamBox{T}}, HFres::HFfinalVars{T}) where {T} = 
gradOfHFenergy(par, HFres.basis, HFres.C, HFres.nuc, HFres.nucCoord, HFres.Ns)


"""

    gradOfHFenergy(par, basis, C, nuc, nucCoords, N=getCharge(nuc)) ->
    AbstractVector

    gradOfHFenergy(par, bs, S, C, nuc, nucCoords, N=getCharge(nuc)) ->
    AbstractVector

Two methods of `gradOfHFenergy`.

≡≡≡ Positional argument(s) ≡≡≡

`par::AbstractVector{<:ParamBox}`: The parameters for differentiation.

`basis::`[`GTBasis`](@ref)`{T, D} where {T, D}`: Basis set information.

`C::NTuple{<:Any, AbstractMatrix{T}} where T`: The coefficient matrix(s) of the canonical 
orbitals with respect to the selected basis set.

`nuc::Union{
    NTuple{NN, String} where NN, 
    AbstractVector{String}
}`: The nuclei in the studied system.

`nucCoords::$(SpatialCoordType)`: The coordinates of corresponding nuclei.

`N::Union{Int, Tuple{Int}, NTuple{2, Int}}`: Total number of electrons, or the number(s) of 
electrons with same spin configurations(s).

`bs::Union{
    NTuple{BN, GTBasisFuncs{T, D, 1}}, 
    AbstractVector{<:GTBasisFuncs{T, D, 1}}
} where {T, D}`: A collection of basis functions.

`S::AbstractMatrix{T} where T`: The overlap lap of the basis set when `bs` is provided as 
the second argument.

**NOTE 1:** If any of these two methods is applied, the user needs to make sure the row 
orders as well as the colum orders of `C` and (or) `S` are consistent with the element 
order of `bs` (`basis.basis`).
``
"""
gradOfHFenergy(par::AbstractVector{<:ParamBox}, b::GTBasis{T, D}, 
               C::NTuple{HFTS, AbstractMatrix{T}}, 
               nuc::VectorOrNTuple{String, NN}, 
               nucCoords::SpatialCoordType{T, D, NN}, 
               N::Union{Int, Tuple{Int}, NTuple{2, Int}}=getCharge(nuc)) where 
              {T, D, HFTS, NN} = 
gradOfHFenergy(par, b.basis, b.S, C, nuc, nucCoords, N)

function gradOfHFenergy(par::AbstractVector{<:ParamBox{T}}, 
                        bs::VectorOrNTuple{GTBasisFuncs{T, D, 1}}, 
                        S::AbstractMatrix{T}, 
                        C::NTuple{HFTS, AbstractMatrix{T}}, 
                        nuc::VectorOrNTuple{String, NN}, 
                        nucCoords::SpatialCoordType{T, D, NN}, 
                        N::Union{Int, Tuple{Int}, NTuple{2, Int}}=getCharge(nuc)) where 
                       {T, D, HFTS, NN}
    nuc = arrayToTuple(nuc)
    nucCoords = genTupleCoords(T, nucCoords)
    Ns = splitSpins(Val(HFTS), N)
    ∂HFenergy.(par, Ref(bs|>collect), Ref(S), Ref(C), Ref(nuc), Ref(nucCoords), Ref(Ns))
end


𝑑f(::Type{FL}, f::F, x::T) where {FL<:FLevel, F<:Function, T} = ForwardDerivative(f, x)

𝑑f(::Type{FI}, ::Function, ::T) where {T} = T(1.0)

∂SGFcore(::Val{xpnSym}, sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T=T(1)) where {T, 𝑙} = 
hasNormFactor(sgf) ? ∂SGF∂xpn2(sgf, c) : ∂SGF∂xpn1(sgf, c)

function ∂SGF∂xpn1(sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T) where {T, 𝑙}
    ( shiftCore(+, sgf, LTuple(2,0,0)) + shiftCore(+, sgf, LTuple(0,2,0)) + 
      shiftCore(+, sgf, LTuple(0,0,2)) ) * (-c)
end

function ∂SGF∂xpn2(sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T) where {T, 𝑙}
    α = sgf.gauss[begin].xpn()
    ugf = genBasisFunc(sgf, false)
    ∂SGF∂xpn1(ugf, c) * getNijkα(sgf.l[1].tuple, α) + sgf * ( c * (𝑙/T(2) + T(0.75)) / α )
end

function ∂SGFcore(::Val{conSym}, sgf::FGTBasisFuncs1O{T, D, 𝑙, 1}, c::T=T(1)) where {T, D, 𝑙}
    BasisFunc(sgf.center, GaussFunc(sgf.gauss[begin].xpn, c), sgf.l, sgf.normalizeGTO)
end

function ∂SGFcore(::Val{cxSym}, sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T=T(1)) where {T, 𝑙}
    sgf = hasNormFactor(sgf) ? absorbNormFactor(sgf)[begin] : sgf
    shiftCore(-, sgf, LTuple(1,0,0)) * (-c*sgf.l[begin][1]) + 
    shiftCore(+, sgf, LTuple(1,0,0)) * (2c*sgf.gauss[begin].xpn())
end

function ∂SGFcore(::Val{cySym}, sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T=T(1)) where {T, 𝑙}
    sgf = hasNormFactor(sgf) ? absorbNormFactor(sgf)[begin] : sgf
    shiftCore(-, sgf, LTuple(0,1,0)) * (-c*sgf.l[begin][2]) + 
    shiftCore(+, sgf, LTuple(0,1,0)) * (2c*sgf.gauss[begin].xpn())
end

function ∂SGFcore(::Val{czSym}, sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T=T(1)) where {T, 𝑙}
    sgf = hasNormFactor(sgf) ? absorbNormFactor(sgf)[begin] : sgf
    shiftCore(-, sgf, LTuple(0,0,1)) * (-c*sgf.l[begin][3]) + 
    shiftCore(+, sgf, LTuple(0,0,1)) * (2c*sgf.gauss[begin].xpn())
end

const sgfSample = genBasisFunc([0.0, 0.0, 0.0], (2.0, 1.0))

const cxIndex = findfirst(x->getTypeParams(x)[2]==cxSym, sgfSample.param)
const cyIndex = findfirst(x->getTypeParams(x)[2]==cySym, sgfSample.param)
const czIndex = findfirst(x->getTypeParams(x)[2]==czSym, sgfSample.param)
const xpnIndex = findfirst(x->getTypeParams(x)[2]==xpnSym, sgfSample.param)
const conIndex = findfirst(x->getTypeParams(x)[2]==conSym, sgfSample.param)

getVpar(sgf::FGTBasisFuncs1O{<:Any, <:Any, <:Any, 1}, ::Val{cxSym}) = sgf.param[cxIndex]
getVpar(sgf::FGTBasisFuncs1O{<:Any, <:Any, <:Any, 1}, ::Val{cySym}) = sgf.param[cyIndex]
getVpar(sgf::FGTBasisFuncs1O{<:Any, <:Any, <:Any, 1}, ::Val{czSym}) = sgf.param[czIndex]
getVpar(sgf::FGTBasisFuncs1O{<:Any, D, <:Any, 1}, ::Val{xpnSym}) where {D} = 
sgf.param[xpnIndex-3+D]
getVpar(sgf::FGTBasisFuncs1O{<:Any, D, <:Any, 1}, ::Val{conSym}) where {D} = 
sgf.param[conIndex-3+D]
getVpar(::FGTBasisFuncs1O{T, D, <:Any, 1}, ::Val) where {T, D} = ParamBox(0.0)

function ∂BasisCore1(par::ParamBox{T, V, FL}, sgf::FGTBasisFuncs1O{T, D, <:Any, 1}) where 
                    {T, FL, V, D}
    mapreduce(+, sgf.param) do fPar
        c = if isDiffParam(fPar) && compareParamBoxCore1(fPar, par)
            _, V2, FL2 = getTypeParams(fPar)
            𝑑f(FL2, fPar.map, fPar[])
        else
            0
        end
        iszero(c) ? EmptyBasisFunc{T, D}() : ∂SGFcore(Val(V2), sgf, c)
    end
end

function ∂BasisCore2(par::ParamBox{T, V, FL}, sgf::FGTBasisFuncs1O{T, D, <:Any, 1}) where 
                    {T, V, FL, D}
    dividend = getVpar(sgf, Val(V))
    if !isDiffParam(dividend) && compareParamBoxCore2(par, dividend)
        ∂SGFcore(Val(V), sgf)
    else
        EmptyBasisFunc{T, D}()
    end
end

∂Basis(par::ParamBox{T, V, FL}, sgf::FGTBasisFuncs1O{T, D, <:Any, 1}) where {T, V, FL, D} = 
isDiffParam(par) ? ∂BasisCore1(par, sgf) : ∂BasisCore2(par, sgf)

∂Basis(par::ParamBox{T, V, FL}, b::FGTBasisFuncs1O{T}) where {T, V, FL} = 
∂Basis.(par, reshape(decomposeCore(Val(true), b), :)) |> sum

∂Basis(par::ParamBox{T}, b::BasisFuncMix{T}) where {T} = 
∂Basis.(par, b.BasisFunc) |> sum

∂Basis(par::ParamBox{T}, b::EmptyBasisFunc{T, D}) where {T, D} = EmptyBasisFunc{T, D}()