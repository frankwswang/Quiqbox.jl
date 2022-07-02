export gradHFenergy

using LinearAlgebra: eigen, Symmetric
using ForwardDiff: derivative as ForwardDerivative

function oneBodyDerivativeCore(::Val{false}, 
                               ∂bfs::Union{NTuple{BN, GTBasisFuncs{T, D, 1}}}, 
                               bfs::Union{NTuple{BN, GTBasisFuncs{T, D, 1}}}, 
                               X::Matrix{T}, ∂X::Matrix{T}, 
                               tf::TypedFunction{F}) where {BN, T, D, F}
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
                               ∂bfs::Union{NTuple{BN, GTBasisFuncs{T, D, 1}}}, 
                               bfs::Union{NTuple{BN, GTBasisFuncs{T, D, 1}}}, 
                               X::Matrix{T}, ∂X::Matrix{T}, 
                               tf::TypedFunction{F}) where {BN, T, D, F}
    ʃ = getFunc(tf)
    bsSize = ∂bfs |> length
    ∂ʃ = ones(bsSize, bsSize, bsSize, bsSize)
    ʃabcd = ones(bsSize, bsSize, bsSize, bsSize)
    ʃ∂abcd = ones(bsSize, bsSize, bsSize, bsSize)
    for i = 1:bsSize, j = 1:i, k = 1:i, l = 1:ifelse(k==i, j, k)
        ʃabcd[i,j,k,l] = ʃabcd[j,i,k,l] = ʃabcd[j,i,l,k] = ʃabcd[i,j,l,k] = 
        ʃabcd[l,k,i,j] = ʃabcd[k,l,i,j] = ʃabcd[k,l,j,i] = ʃabcd[l,k,j,i] = 
        ʃ(bfs[i],  bfs[j],  bfs[k],  bfs[l])
    end
    for l = 1:bsSize, k=1:l, j=1:bsSize, i=1:bsSize
        ʃ∂abcd[i,j,k,l] = ʃ∂abcd[i,j,l,k] = ʃ(∂bfs[i], bfs[j],  bfs[k],  bfs[l])
    end
    # [∂ʃ4[i,j,k,l] == ∂ʃ4[j,i,l,k] == ∂ʃ4[j,i,k,l] != ∂ʃ4[l,j,k,i]
    for i = 1:bsSize, j = 1:i, k = 1:i, l = 1:ifelse(k==i, j, k)
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


function derivativeCore(FoutputIsVector::Val{B}, 
                        bs::NTuple{BN, GTBasisFuncs{T, D, 1}}, 
                        par::ParamBox, S::Matrix{T}, 
                        oneBodyF::TypedFunction{F1}, twoBodyF::TypedFunction{F2}) where 
                       {B, BN, T, D, F1, F2}
    # ijkl in chemists' notation of spatial bases (ij|kl).
    bfs = Tuple(hcat(decomposeCore.(Val(false), bs)...))
    # ∂bfs = deriveBasisFunc.(bfs, par)
    ∂bfs = ∂Basis.(par, bfs)
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


function ∂HFenergy(bs::NTuple{BN, GTBasisFuncs{T, D, 1}}, 
                   par::ParamBox, C::NTuple{HFTS, Matrix{T}}, 
                   S::Matrix{T}, nuc::NTuple{NN, String}, 
                   nucCoords::NTuple{NN, NTuple{3, T}}, 
                   nElectron::NTuple{HFTS, Int}) where {BN, T, D, HFTS, NN}
    Xinv = sqrt(S)::Matrix{T}
    cH = (i, j)->getCoreH(i, j, nuc, nucCoords)
    ∂hij, ∂hijkl = derivativeCore(Val(false), bs, par, S, 
                                  TypedFunction(cH), TypedFunction(getEleEleInteraction))
    getEᵀ(∂hij, ∂hijkl, Ref(Xinv).*C, nElectron)
end


function gradHFenergy(bs::Union{NTuple{BN, GTBasisFuncs{T, D, 1}}, 
                                AbstractVector{<:GTBasisFuncs{T, D, 1}}}, 
                      par::Vector{<:ParamBox}, 
                      C::NTuple{HFTS, Matrix{T}}, 
                      S::Matrix{T}, 
                      nuc::Union{NTuple{NN, String}, Vector{String}}, 
                      nucCoords::Union{NTuple{NN, NTuple{3, T}}, 
                                       Vector{<:AbstractArray{<:Real}}}, 
                      nElectron::Union{Int, NTuple{2, Int}}=getCharge(nuc)) where 
                     {BN, T, D, HFTS, NN}
    bs = arrayToTuple(bs)
    nuc = arrayToTuple(nuc)
    nucCoords = genTupleCoords(nucCoords)
    Ns = splitSpins(Val(HFTS), nElectron)
    ∂HFenergy.(Ref(bs), par, Ref(C), Ref(S), Ref(nuc), Ref(nucCoords), Ref(Ns))
end


𝑑f(::Type{FL}, f::F, x::T) where {FL<:FLevel, F<:Function, T} = ForwardDerivative(f, x)

𝑑f(::Type{FLi}, f::Function, x::T) where {T} = 1.0

function ∂SGFcore(::Val{xpnSym}, sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T=T(1)) where {T, D, 𝑙}
    res = ( shiftCore(+, sgf, LTuple(2,0,0)) + shiftCore(+, sgf, LTuple(0,2,0)) + 
            shiftCore(+, sgf, LTuple(0,0,2)) ) * (-c)
    if sgf.normalizeGTO
        res += sgf * (T(0.5𝑙 + 0.75) / sgf.gauss[1].xpn() * c)
    end
    res
end

function ∂SGFcore(::Val{conSym}, sgf::FGTBasisFuncs1O{T, D, 𝑙, 1}, c::T=T(1)) where {T, D, 𝑙}
    BasisFunc(sgf.center, GaussFunc(sgf.gauss[1].xpn, c), sgf.l, sgf.normalizeGTO)
end

function ∂SGFcore(::Val{cxSym}, sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T=T(1)) where {T, D, 𝑙}
    shiftCore(-, sgf, LTuple(1,0,0)) * (-c*sgf.l[1][1]) + 
    shiftCore(+, sgf, LTuple(1,0,0)) * (2c*sgf.gauss[1].xpn())
end

function ∂SGFcore(::Val{cySym}, sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T=T(1)) where {T, D, 𝑙}
    shiftCore(-, sgf, LTuple(0,1,0)) * (-c*sgf.l[1][2]) + 
    shiftCore(+, sgf, LTuple(0,1,0)) * (2c*sgf.gauss[1].xpn())
end

function ∂SGFcore(::Val{czSym}, sgf::FGTBasisFuncs1O{T, 3, 𝑙, 1}, c::T=T(1)) where {T, D, 𝑙}
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

function ∂Basis(par::ParamBox{T, V, FL}, sgf::FGTBasisFuncs1O{T, D, <:Any, 1}) where 
               {T, V, FL, D}
    params = sgf.param
    if FL == FLi || !par.canDiff[]
        if par.index == params[paramIndex(Val(V), Val(D))].index
            ∂SGFcore(Val(V), sgf)
        else
            EmptyBasisFunc{T, D}()
        end
    else
        is = findall(x->x.dataName==par.dataName && x.index==par.index, params)
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
end

∂Basis(par::ParamBox{T, V, FL}, b::FGTBasisFuncs1O{T}) where {T, V, FL} = 
∂Basis.(par, reshape(decomposeCore(Val(true), b), :)) |> sum

∂Basis(par::ParamBox{T}, b::BasisFuncMix{T}) where {T} = 
∂Basis.(par, b.BasisFunc) |> sum

∂Basis(par::ParamBox{T}, b::EmptyBasisFunc{T, D}) where {T, D} = EmptyBasisFunc{T, D}()