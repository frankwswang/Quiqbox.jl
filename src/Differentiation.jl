export gradOfHFenergy

using LinearAlgebra: eigen, Symmetric, Hermitian
using ForwardDiff: derivative as ForwardDerivative
using Tullio: @tullio

function oneBodyDerivativeCore(::Val{false}, 
                               ∂bfs::NTuple{BN, GTBasisFuncs{T, D, 1}}, 
                               bfs::NTuple{BN, GTBasisFuncs{T, D, 1}}, 
                               X::AbstractMatrix{T}, ∂X::AbstractMatrix{T}, 
                               tf::TypedFunction{F}) where {BN, T, D, F}
    ʃ = getFunc(tf)
    ∂ʃ = Array{T}(undef, BN, BN)
    ʃab = Array{T}(undef, BN, BN)
    ∂ʃab = Array{T}(undef, BN, BN)
    for i = 1:BN, j = 1:i
       ʃab[i,j] = ʃab[j,i] = ʃ(bfs[i], bfs[j])
    end
    for i = 1:BN, j = 1:i
        ∂ʃab[i,j] = ∂ʃab[j,i] = ʃ(∂bfs[i], bfs[j]) + ʃ(bfs[i], ∂bfs[j])
    end
    @views begin
        @inbounds for i=1:BN, j=1:i
            # X[i,j] == X[j,i]
            ∂ʃ[i,j] = ∂ʃ[j,i] = X[:,i]' * ∂ʃab *  X[:,j] +
                               ∂X[:,i]' *  ʃab *  X[:,j] +
                                X[:,i]' *  ʃab * ∂X[:,j]
        end
    end
    # for i=1:BN, j=1:i
    #     # X[i,j] == X[j,i]
    #     @tullio temp1 := X[a,$i] * ∂ʃab[a,b] *  X[b,$j]
    #     @tullio temp2 := ∂X[a,$i] *ʃab[a,b] *  X[b,$j]
    #     @tullio temp3 := X[a,$i] *  ʃab[a,b] * ∂X[b,$j]
    #     ∂ʃ[i,j] = ∂ʃ[j,i] = temp1 + temp2 + temp3
    # end
    ∂ʃ
end


function twoBodyDerivativeCore(::Val{false}, 
                               ∂bfs::NTuple{BN, GTBasisFuncs{T, D, 1}}, 
                               bfs::NTuple{BN, GTBasisFuncs{T, D, 1}}, 
                               X::AbstractMatrix{T}, ∂X::AbstractMatrix{T}, 
                               tf::TypedFunction{F}) where {BN, T, D, F}
    ʃ = getFunc(tf)
    ∂ʃ = Array{T}(undef, BN, BN, BN, BN)
    ʃabcd = Array{T}(undef, BN, BN, BN, BN)
    ʃ∂abcd = Array{T}(undef, BN, BN, BN, BN)

    # ijkl in chemists' notation of spatial bases (ij|kl).
    for i = 1:BN, j = 1:i, k = 1:i, l = 1:ifelse(k==i, j, k)
        ʃabcd[i,j,k,l] = ʃabcd[j,i,k,l] = ʃabcd[j,i,l,k] = ʃabcd[i,j,l,k] = 
        ʃabcd[l,k,i,j] = ʃabcd[k,l,i,j] = ʃabcd[k,l,j,i] = ʃabcd[l,k,j,i] = 
        ʃ(bfs[i],  bfs[j],  bfs[k],  bfs[l])
    end
    for l = 1:BN, k=1:l, j=1:BN, i=1:BN
        ʃ∂abcd[i,j,k,l] = ʃ∂abcd[i,j,l,k] = ʃ(∂bfs[i], bfs[j],  bfs[k],  bfs[l])
    end
    # [∂ʃ4[i,j,k,l] == ∂ʃ4[j,i,l,k] == ∂ʃ4[j,i,k,l] != ∂ʃ4[l,j,k,i]
    for i = 1:BN, j = 1:i, k = 1:i, l = 1:ifelse(k==i, j, k)
        # val = T(0.0)
        # # ʃ∂abcd[i,j,k,l] == ʃ∂abcd[i,j,l,k] == ʃab∂cd[l,k,i,j] == ʃab∂cd[k,l,i,j]
        # @inbounds for a = 1:BN, b = 1:BN, c = 1:BN, d = 1:BN
        #     val += (  X[a,i]*X[b,j]*X[c,k]*X[d,l] + X[a,j]*X[b,i]*X[c,k]*X[d,l] + 
        #               X[c,i]*X[d,j]*X[a,k]*X[b,l] + X[c,i]*X[d,j]*X[a,l]*X[b,k]  ) * 
        #            ʃ∂abcd[a,b,c,d] + 
        #            ( ∂X[a,i]*X[b,j]* X[c,k]*X[d,l] + X[a,i]*∂X[b,j]*X[c,k]* X[d,l] + 
        #               X[a,i]*X[b,j]*∂X[c,k]*X[d,l] + X[a,i]* X[b,j]*X[c,k]*∂X[d,l] ) * 
        #            ʃabcd[a,b,c,d]
        # end

        # ʃ∂abcd[i,j,k,l] == ʃ∂abcd[i,j,l,k] == ʃab∂cd[l,k,i,j] == ʃab∂cd[k,l,i,j]
        @tullio val := ( X[a,$i]* X[b,$j]* X[c,$k]* X[d,$l] + 
                         X[a,$j]* X[b,$i]* X[c,$k]* X[d,$l] + 
                         X[c,$i]* X[d,$j]* X[a,$k]* X[b,$l] + 
                         X[c,$i]* X[d,$j]* X[a,$l]* X[b,$k]  ) *ʃ∂abcd[a,b,c,d] + 
                       (∂X[a,$i]* X[b,$j]* X[c,$k]* X[d,$l] + 
                         X[a,$i]*∂X[b,$j]* X[c,$k]* X[d,$l] + 
                         X[a,$i]* X[b,$j]*∂X[c,$k]* X[d,$l] + 
                         X[a,$i]* X[b,$j]* X[c,$k]*∂X[d,$l]  ) * ʃabcd[a,b,c,d]

        ∂ʃ[i,j,k,l] = ∂ʃ[j,i,k,l] = ∂ʃ[j,i,l,k] = ∂ʃ[i,j,l,k] = 
        ∂ʃ[l,k,i,j] = ∂ʃ[k,l,i,j] = ∂ʃ[k,l,j,i] = ∂ʃ[l,k,j,i] = val
    end
    ∂ʃ
end


function derivativeCore(FoutputIsVector::Val{B}, 
                        bfs::NTuple{BN, GTBasisFuncs{T, D, 1}}, 
                        par::ParamBox, S::AbstractMatrix{T}, 
                        oneBodyF::TypedFunction{F1}, twoBodyF::TypedFunction{F2}) where 
                       {B, BN, T, D, F1, F2}
    ∂bfs = ∂Basis.(par, bfs)
    ∂S = Array{T}(undef, BN, BN)
    ∂X = Array{T}(undef, BN, BN) # ∂X corresponds to the derivative of X = S^(-0.5)
    ∂X₀ = Array{T}(undef, BN, BN) # ∂X in its eigen basis
    for i=1:BN, j=1:i
        ∂S[i,j] = ∂S[j,i] = overlap(∂bfs[i], bfs[j]) + overlap(bfs[i], ∂bfs[j])
    end
    X = getXcore1(S)
    λ, 𝑣 = eigen(S|>Hermitian)
    ∂S2 = 𝑣'*∂S*𝑣
    @inbounds for i=1:BN, j=1:i
        ∂X₀[i,j] = ∂X₀[j,i] = (- ∂S2[i,j] * inv(sqrt(λ[i])) * inv(sqrt(λ[j])) * 
                               inv(sqrt(λ[i]) + sqrt(λ[j])))
    end
    # @tullio ∂X₀[i,j] := ( -∂S2[i,j] * inv(sqrt(λ[i])) * inv(sqrt(λ[j])) * 
    #                       inv(sqrt(λ[i]) + sqrt(λ[j])) )
    ∂X = 𝑣*∂X₀*𝑣'
    # @tullio ∂X[j,i] := 𝑣[j,k]*∂X₀[k,l]*𝑣[i,l]
    ∂ʃ2 = oneBodyDerivativeCore(FoutputIsVector, ∂bfs, bfs, X, ∂X, oneBodyF)
    ∂ʃ4 = twoBodyDerivativeCore(FoutputIsVector, ∂bfs, bfs, X, ∂X, twoBodyF)
    ∂ʃ2, ∂ʃ4
end


function ∂HFenergy(par::ParamBox{T}, 
                   bs::NTuple{BN, GTBasisFuncs{T, D, 1}}, 
                   S::AbstractMatrix{T}, 
                   C::NTuple{HFTS, AbstractMatrix{T}}, 
                   nuc::NTuple{NN, String}, 
                   nucCoords::NTuple{NN, NTuple{D, T}}, 
                   N::NTuple{HFTS, Int}) where {BN, T, D, HFTS, NN}
    Xinv = sqrt(S)::Matrix{T} # necessary assertion for type stability
    cH = (i, j)->coreHij(i, j, nuc, nucCoords)
    ∂hij, ∂hijkl = derivativeCore(Val(false), bs, par, S, 
                                  TypedFunction(cH), TypedFunction(eeInteraction))
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
               nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
               nucCoords::SpatialCoordType{T, D, NN}, 
               N::Union{Int, Tuple{Int}, NTuple{2, Int}}=getCharge(nuc)) where 
              {T, D, HFTS, NN} = 
gradOfHFenergy(par, b.basis, b.S, C, nuc, nucCoords, N)

function gradOfHFenergy(par::AbstractVector{<:ParamBox{T}}, 
                        bs::Union{NTuple{BN, GTBasisFuncs{T, D, 1}}, 
                                  AbstractVector{<:GTBasisFuncs{T, D, 1}}}, 
                        S::AbstractMatrix{T}, 
                        C::NTuple{HFTS, AbstractMatrix{T}}, 
                        nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                        nucCoords::SpatialCoordType{T, D, NN}, 
                        N::Union{Int, Tuple{Int}, NTuple{2, Int}}=getCharge(nuc)) where 
                       {BN, T, D, HFTS, NN}
    bs = arrayToTuple(bs)
    nuc = arrayToTuple(nuc)
    nucCoords = genTupleCoords(T, nucCoords)
    Ns = splitSpins(Val(HFTS), N)
    ∂HFenergy.(par, Ref(bs), Ref(S), Ref(C), Ref(nuc), Ref(nucCoords), Ref(Ns))
end


𝑑f(::Type{FL}, f::F, x::T) where {FL<:FLevel, F<:Function, T} = ForwardDerivative(f, x)

𝑑f(::Type{FI}, f::Function, x::T) where {T} = T(1.0)

function ∂SGFcore(::Val{xpnSym}, sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T=T(1)) where {T, 𝑙}
    res = ( shiftCore(+, sgf, LTuple(2,0,0)) + shiftCore(+, sgf, LTuple(0,2,0)) + 
            shiftCore(+, sgf, LTuple(0,0,2)) ) * (-c)
    if sgf.normalizeGTO
        res += sgf * ((𝑙/T(2) + T(3)/4) / sgf.gauss[1].xpn() * c)
    end
    res
end

function ∂SGFcore(::Val{conSym}, sgf::FGTBasisFuncs1O{T, D, 𝑙, 1}, c::T=T(1)) where {T, D, 𝑙}
    BasisFunc(sgf.center, GaussFunc(sgf.gauss[1].xpn, c), sgf.l, sgf.normalizeGTO)
end

function ∂SGFcore(::Val{cxSym}, sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T=T(1)) where {T, 𝑙}
    shiftCore(-, sgf, LTuple(1,0,0)) * (-c*sgf.l[1][1]) + 
    shiftCore(+, sgf, LTuple(1,0,0)) * (2c*sgf.gauss[1].xpn())
end

function ∂SGFcore(::Val{cySym}, sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T=T(1)) where {T, 𝑙}
    shiftCore(-, sgf, LTuple(0,1,0)) * (-c*sgf.l[1][2]) + 
    shiftCore(+, sgf, LTuple(0,1,0)) * (2c*sgf.gauss[1].xpn())
end

function ∂SGFcore(::Val{czSym}, sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T=T(1)) where {T, 𝑙}
    shiftCore(-, sgf, LTuple(0,0,1)) * (-c*sgf.l[1][3]) + 
    shiftCore(+, sgf, LTuple(0,0,1)) * (2c*sgf.gauss[1].xpn())
end

const sgfSample = genBasisFunc([0.0, 0.0, 0.0], (2.0, 1.0))

const cxIndex = findfirst(x->getTypeParams(x)[2]==cxSym, sgfSample.param)
const cyIndex = findfirst(x->getTypeParams(x)[2]==cySym, sgfSample.param)
const czIndex = findfirst(x->getTypeParams(x)[2]==czSym, sgfSample.param)
const xpnIndex = findfirst(x->getTypeParams(x)[2]==xpnSym, sgfSample.param)
const conIndex = findfirst(x->getTypeParams(x)[2]==conSym, sgfSample.param)

paramIndex(::Val{cxSym}, ::Val) = cxIndex
paramIndex(::Val{cySym}, ::Val) = cyIndex
paramIndex(::Val{czSym}, ::Val) = czIndex
paramIndex(::Val{xpnSym}, ::Val{D}) where {D} = xpnIndex - 3 + D
paramIndex(::Val{conSym}, ::Val{D}) where {D} = conIndex - 3 + D

function ∂BasisCore1(par::ParamBox{T, V, FL}, sgf::FGTBasisFuncs1O{T, D, <:Any, 1}) where 
                    {T, FL, V, D}
    params = sgf.param
    is = findall(x->compareParamBoxCore1(x, par), params)
    if length(is) > 0
        map(is) do i
            fPar = params[i]
            _, V2, FL2 = getTypeParams(fPar)
            c = 𝑑f(FL2, fPar.map, fPar[])
            if c == 0.0
                EmptyBasisFunc{T, D}()
            else
                ∂SGFcore(Val(V2), sgf, c)
            end
        end |> sumOf
    else
        EmptyBasisFunc{T, D}()
    end
end

function ∂BasisCore2(par::ParamBox{T, V, FL}, sgf::FGTBasisFuncs1O{T, D, <:Any, 1}) where 
                    {T, V, FL, D}
    dividend = sgf.param[paramIndex(Val(V), Val(D))]
    if !(divident.canDiff[]) && compareParamBoxCore2(par, dividend)
        ∂SGFcore(Val(V), sgf)
    else
        EmptyBasisFunc{T, D}()
    end
end

∂Basis(par::ParamBox{T, V, FI}, sgf::FGTBasisFuncs1O{T, D, <:Any, 1}) where {T, V, D} = 
∂BasisCore1(par, sgf)

∂Basis(par::ParamBox{T, V, FL}, sgf::FGTBasisFuncs1O{T, D, <:Any, 1}) where {T, V, FL, D} = 
par.canDiff[] ? ∂BasisCore1(par, sgf) : ∂BasisCore2(par, sgf)

∂Basis(par::ParamBox{T, V, FL}, b::FGTBasisFuncs1O{T}) where {T, V, FL} = 
∂Basis.(par, reshape(decomposeCore(Val(true), b), :)) |> sum

∂Basis(par::ParamBox{T}, b::BasisFuncMix{T}) where {T} = 
∂Basis.(par, b.BasisFunc) |> sum

∂Basis(par::ParamBox{T}, b::EmptyBasisFunc{T, D}) where {T, D} = EmptyBasisFunc{T, D}()