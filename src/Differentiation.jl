export gradOfHFenergy

using LinearAlgebra: eigen, Symmetric, Hermitian
using ForwardDiff: derivative as ForwardDerivative
using TensorOperations: @tensor as @TOtensor
using DoubleFloats: Double64
using Base: OneTo

# Reference(s):
## [DOI] 10.1063/1.445528

function ∂1BodyCore(bfs::AbstractVector{<:GTBasisFuncs{T1, D1, 1}}, 
                    ∂bfs::AbstractVector{<:GTBasisFuncs{T1, D1, 1}}, 
                    X::Matrix{T2}, ∂X::Matrix{T2}, 
                    ʃab::Array{T1}, ʃ::F) where {T1, D1, T2, F<:Function}
    BN = length(bfs)
    ∂ʃab = similar(ʃab)
    shift1 = firstindex( bfs) - 1
    shift2 = firstindex(∂bfs) - 1
    Threads.@threads for k in (OneTo∘triMatEleNum)(BN)
        i, j = convert1DidxTo2D(BN, k)
        @inbounds ∂ʃab[i,j] = ∂ʃab[j,i] = 
                  ʃ(∂bfs[i+shift2], bfs[j+shift1]) + ʃ(bfs[i+shift1], ∂bfs[j+shift2])
    end
    X' * ∂ʃab * X + ∂X' * ʃab * X + X' * ʃab * ∂X
end

function ∂2BodyCore(bfs::AbstractVector{<:GTBasisFuncs{T1, D1, 1}}, 
                    ∂bfs::AbstractVector{<:GTBasisFuncs{T1, D1, 1}}, 
                    Xcols::MatrixCol{T2}, ∂Xcols::MatrixCol{T2}, 
                    ʃabcd::Array{T1, D2}, ʃ::F) where {T1, D1, T2, D2, F<:Function}
    BN = length(bfs)
    ʃ∂abcd = similar(ʃabcd)
    ∂ʃ = Array{promote_type(T1, T2)}(undef, size(ʃabcd)...)
    shift1 = firstindex( bfs) - 1
    shift2 = firstindex(∂bfs) - 1
    # ijkl in the chemists' notation of spatial bases (ij|kl).
    @sync for lk in (OneTo∘triMatEleNum)(BN), j=OneTo(BN), i=OneTo(BN)
        Threads.@spawn begin
            k, l = convert1DidxTo2D(BN, lk)
            @inbounds ʃ∂abcd[i,j,l,k] = ʃ∂abcd[i,j,k,l] = 
                      ʃ(∂bfs[i+shift2], bfs[j+shift1],  bfs[k+shift1],  bfs[l+shift1])
        end
    end
    # [∂ʃ4[i,j,k,l] == ∂ʃ4[j,i,l,k] == ∂ʃ4[j,i,k,l] != ∂ʃ4[l,j,k,i]
    Threads.@threads for m in (OneTo∘triMatEleNum∘triMatEleNum)(BN)
        # ʃ∂abcd[i,j,k,l] == ʃ∂abcd[i,j,l,k] == ʃab∂cd[l,k,i,j] == ʃab∂cd[k,l,i,j]
        i, j, k, l = convert1DidxTo4D(BN, m)
        @inbounds begin
             Xvi =  Xcols[i]
             Xvj =  Xcols[j]
             Xvk =  Xcols[k]
             Xvl =  Xcols[l]
            ∂Xvi = ∂Xcols[i]
            ∂Xvj = ∂Xcols[j]
            ∂Xvk = ∂Xcols[k]
            ∂Xvl = ∂Xcols[l]

            @TOtensor val = 
                (Xvi[a] * Xvj[b] * Xvk[c] * Xvl[d] + Xvj[a] * Xvi[b] * Xvk[c] * Xvl[d] + 
                 Xvi[c] * Xvj[d] * Xvk[a] * Xvl[b] + Xvi[c] * Xvj[d] * Xvl[a] * Xvk[b]) * 
                ʃ∂abcd[a,b,c,d] + 
                (∂Xvi[a] *  Xvj[b] *  Xvk[c] *  Xvl[d] + 
                  Xvi[a] * ∂Xvj[b] *  Xvk[c] *  Xvl[d] + 
                  Xvi[a] *  Xvj[b] * ∂Xvk[c] *  Xvl[d] + 
                  Xvi[a] *  Xvj[b] *  Xvk[c] * ∂Xvl[d] ) * 
                ʃabcd[a,b,c,d]

            ∂ʃ[i,j,k,l] = ∂ʃ[j,i,k,l] = ∂ʃ[j,i,l,k] = ∂ʃ[i,j,l,k] = 
            ∂ʃ[l,k,i,j] = ∂ʃ[k,l,i,j] = ∂ʃ[k,l,j,i] = ∂ʃ[l,k,j,i] = val
        end
    end
    ∂ʃ
end


function ∂NBodyInts(bfs::AbstractVector{<:GTBasisFuncs{T1, D, 1}}, par::ParamBox, 
                    (λ, 𝑣)::Tuple{Vector{T2}, Matrix{T2}}, X::Hermitian{T2, Matrix{T2}}, 
                    ʃab::Array{T1}, ʃabcd::Array{T1}, 
                    ʃ2::F1, ʃ4::F2) where {T1, T2, D, F1<:Function, F2<:Function}
    BN = length(bfs)
    ∂bfs = ∂Basis.(par, bfs)
    ∂S = Array{T2}(undef, BN, BN)
    ∂X = Array{T2}(undef, BN, BN) # ∂X corresponds to the derivative of X = S^(-0.5)
    ∂X₀ = Array{T2}(undef, BN, BN) # ∂X in its eigen basis
    shift1 = firstindex( bfs) - 1
    shift2 = firstindex(∂bfs) - 1
    rng = (OneTo∘triMatEleNum)(BN)
    Threads.@threads for k in rng
        i, j = convert1DidxTo2D(BN, k)
        @inbounds ∂S[i,j] = ∂S[j,i] = overlap(∂bfs[i+shift2],  bfs[j+shift1]) + 
                                      overlap( bfs[i+shift1], ∂bfs[j+shift2])
    end
    ∂S2 = 𝑣' * ∂S * 𝑣
    Threads.@threads for k in rng
        i, j = convert1DidxTo2D(BN, k)
        @inbounds ∂X₀[i,j] = ∂X₀[j,i] = ( -∂S2[i,j] / ( sqrt(λ[i]) * sqrt(λ[j]) * 
                                          (sqrt(λ[i]) + sqrt(λ[j])) ) )
    end
    ∂X = 𝑣 * ∂X₀ * 𝑣'
    nX = norm(X)
    n∂X = norm(∂X)
    T = ifelse( (0.317 < nX < 1.778) && # ⁴√0.01 < nX < ⁴√10
                (0.01    < n∂X < 10) && (0.01 < nX*n∂X < 10) && (0.01 < nX^3*n∂X < 10), 
        T1, T2)
    X = convert(Matrix{T}, X)
    ∂X = convert(Matrix{T}, ∂X)
    Xcols = (collect∘eachcol)(X)
    ∂Xcols = (collect∘eachcol)(∂X)
    ∂ʃ2 = ∂1BodyCore(bfs, ∂bfs, X,     ∂X,     ʃab,   ʃ2)
    ∂ʃ4 = ∂2BodyCore(bfs, ∂bfs, Xcols, ∂Xcols, ʃabcd, ʃ4)
    ∂ʃ2, ∂ʃ4
end


function ∇Ehf(pars::AbstractVector{<:ParamBox}, 
              b::GTBasis{T, D}, 
              C::NTuple{HFTS, AbstractMatrix{T}}, 
              nuc::Tuple{String, Vararg{String, NNMO}}, 
              nucCoords::Tuple{NTuple{D, T}, Vararg{NTuple{D, T}, NNMO}}, 
              N::NTuple{HFTS, Int}) where {T, D, HFTS, NNMO}
    bfs = collect(b.basis)
    S = b.S
    numEps(T) > eps(Double64) && (S = Double64.(S))
    (λ, 𝑣) = eigen(S|>Hermitian)
    X = getXcore1(S)
    Hcore = coreH(b, nuc, nucCoords)
    eeI = b.eeI
    cH = (i, j)->coreHij(i, j, nuc, nucCoords)
    map(pars) do par
        ∂hij, ∂hijkl = ∂NBodyInts(bfs, par, (λ, 𝑣), X, Hcore, eeI, cH, eeInteraction)
        # ∂hij and ∂hijkl are on an orthonormal basis.
        Cₓs = convert.(Matrix{eltype(∂hij)}, (Ref∘inv)(X).*C)
        convert(T, getEhf(∂hij, ∂hijkl, Cₓs, N))
    end
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

    gradOfHFenergy(par, basis, C, nuc, nucCoords, N=getCharge(nuc)) -> AbstractVector

    gradOfHFenergy(par, bs, C, nuc, nucCoords, N=getCharge(nuc)) -> AbstractVector

Two methods of `gradOfHFenergy`.

≡≡≡ Positional argument(s) ≡≡≡

`par::AbstractVector{<:ParamBox}`: The parameters for differentiation.

`basis::`[`GTBasis`](@ref)`{T, D} where {T, D}`: Basis set information.

`C::NTuple{<:Any, AbstractMatrix{T}} where T`: The coefficient matrix(s) of the canonical 
orbitals with respect to the selected basis set.

`nuc::Union{
    Tuple{String, Vararg{String, NNMO}} where NNMO, 
    AbstractVector{String}
}`: The nuclei in the studied system.

`nucCoords::$(SpatialCoordType)`: The coordinates of corresponding nuclei.

`N::Union{Int, Tuple{Int}, NTuple{2, Int}}`: Total number of electrons, or the number(s) of 
electrons with same spin configurations(s).

`bs::Union{
    NTuple{BN, GTBasisFuncs{T, D, 1}}, 
    AbstractVector{<:GTBasisFuncs{T, D, 1}}
} where {T, D}`: A collection of basis functions.

**NOTE:** If any of these two methods is applied, the user needs to make sure the row 
orders as well as the colum orders of `C` are consistent with the element order of `bs` 
(`basis.basis`).
``
"""
gradOfHFenergy(pars::AbstractVector{<:ParamBox}, b::GTBasis{T, D}, 
               C::NTuple{HFTS, AbstractMatrix{T}}, 
               nuc::AVectorOrNTuple{String, NNMO}, 
               nucCoords::SpatialCoordType{T, D, NNMO}, 
               N::Union{Int, Tuple{Int}, NTuple{2, Int}}=getCharge(nuc)) where 
              {T, D, HFTS, NNMO} = 
∇Ehf(pars, b, C, arrayToTuple(nuc), genTupleCoords(T, nucCoords), splitSpins(Val(HFTS), N))

gradOfHFenergy(pars::AbstractVector{<:ParamBox{T}}, 
               bs::AVectorOrNTuple{GTBasisFuncs{T, D, 1}}, 
               C::NTuple{HFTS, AbstractMatrix{T}}, 
               nuc::AVectorOrNTuple{String, NNMO}, 
               nucCoords::SpatialCoordType{T, D, NNMO}, 
               N::Union{Int, Tuple{Int}, NTuple{2, Int}}=getCharge(nuc)) where 
              {T, D, HFTS, NNMO} = 
gradOfHFenergy(pars, GTBasis(bs), C, nuc, nucCoords, N)


𝑑f(f::Function, x) = ForwardDerivative(f, x)

𝑑f(::DI, ::T) where {T} = T(1.0)

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

const cxIndex  = findfirst(x -> outSymOf(x) ==  cxSym, sgfSample.param)
const cyIndex  = findfirst(x -> outSymOf(x) ==  cySym, sgfSample.param)
const czIndex  = findfirst(x -> outSymOf(x) ==  czSym, sgfSample.param)
const xpnIndex = findfirst(x -> outSymOf(x) == xpnSym, sgfSample.param)
const conIndex = findfirst(x -> outSymOf(x) == conSym, sgfSample.param)

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
            𝑑f(fPar.map, fPar[])
        else
            0
        end
        iszero(c) ? EmptyBasisFunc{T, D}() : ∂SGFcore(Val(outSymOf(fPar)), sgf, c)
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