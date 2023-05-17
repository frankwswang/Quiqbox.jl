export SCFconfig, HFconfig, runHF, runHFcore

using LinearAlgebra: dot, Hermitian, \, det, I, ishermitian, diag, norm
using Base: OneTo
using Combinatorics: powerset
using LineSearches
using SPGBox: spgbox!
using LBFGSB: lbfgsb

const defaultDS = 0.5
const defaultDIISsize = 10
const defaultDIISsolver = :LBFGS
const SADHFmaxStep = 50
const defaultHFinfoL = 2
const defaultHFmaxStep = 150
const defaultHFsaveTrace = (false, false, false, true) # C, D, F, E
const DEtotIndices = [2, 4]

const HFminItr = 10
const HFinterEstoreSize = 15
const HFinterValStoreSizes = (2,3,2, HFinterEstoreSize) # C(>1), D(>2), F(>1), E(>1)
const defaultHFCStr = "HFconfig()"
const defaultSCFconfigArgs = ( (:ADIIS, :DIIS), (1e-3, 1e-10) )
const defaultSecConvRatio = 50
const defaultOscThreshold = 1e-6

# Reference(s):
## [ISBN-13] 978-0486691862
## [DOI] 10.1016/0009-2614(80)80396-4
## [DOI] 10.1002/jcc.540030413
## [DOI] 10.1063/1.1470195
## [DOI] 10.1063/1.3304922

getXcore1(S::AbstractMatrix{T}) where {T} = Hermitian(S)^(-T(0.5))

precompile(getXcore1, (Matrix{Float64},))

const getXmethods = (m1=getXcore1,)

getX(S::AbstractMatrix{T}, method::Symbol=:m1) where {T} = 
getproperty(getXmethods, method)(S)


function getCϵ(X::AbstractMatrix{T}, Fˢ::AbstractMatrix{T}, 
               stabilizeSign::Bool=true) where {T}
    ϵ, Cₓ = eigen(X'*Fˢ*X |> Hermitian)
    outC = X*Cₓ
    # Stabilize the sign factor of each column.
    stabilizeSign && for j in axes(outC, 2)
       outC[:, j] *= ifelse(outC[begin, j] < 0, -1, 1)
    end
    outC, ϵ
end

@inline getC(X::AbstractMatrix{T}, Fˢ::AbstractMatrix{T}, 
             stabilizeSign::Bool=true) where {T} = 
        getCϵ(X, Fˢ, stabilizeSign)[begin]


splitSpins(::Val{1}, N::Int) = (N÷2,)

splitSpins(::Val{2}, N::Int) = (N÷2, N-N÷2)

splitSpins(::Val{N}, Ns::NTuple{N, Int}) where {N} = itself(Ns)

splitSpins(::Val{2}, (Nˢ,)::Tuple{Int}) = (Nˢ, Nˢ)

splitSpins(::Val{:RHF}, Ns::NTuple{2, Int}) = 
error("For restricted closed-shell Hartree–Fock (RHF), the input spin configuration $(Ns)"*
      " is not supported.")

splitSpins(::Val{:RHF}, N) = splitSpins(Val(HFtypeSizeList[:RHF]), N)

splitSpins(::Val{:UHF}, N) = splitSpins(Val(HFtypeSizeList[:UHF]), N)

function breakSymOfC(::Val{:UHF}, C::AbstractMatrix{T}) where {T}
    C2 = copy(C)
    l = min(size(C2, 1), 2)
    C2[begin:(begin+l-1), begin:(begin+l-1)] .= 0 # Breaking spin symmetry.
    # C2[l, :] .= 0 # Another way.
    (C, C2)
end

breakSymOfC(::Val{:RHF}, C::AbstractMatrix{T}) where {T} = (C,)

breakSymOfC(::Val{:RHF}, Hcore, HeeI, X, Dᵅ, Dᵝ) = 
getC.( Ref(X), getF(Hcore, HeeI, ((Dᵅ + Dᵝ)./2,)) )

breakSymOfC(::Val{:UHF}, Hcore, HeeI, X, Dᵅ, Dᵝ) = 
getC.( Ref(X), getF(Hcore, HeeI, (Dᵅ, Dᵝ)) )


function getCfromGWH(::Val{HFT}, S::AbstractMatrix{T}, Hcore::AbstractMatrix{T}, 
                     X::AbstractMatrix{T}) where {HFT, T}
    H = similar(Hcore)
    Δi1 = firstindex(S, 1) - 1
    Δi2 = firstindex(H, 1) - 1
    len = size(H, 1)
    Threads.@threads for k in (OneTo∘triMatEleNum)(len)
        i, j = convert1DidxTo2D(len, k)
        H[j+Δi2, i+Δi2] = H[i+Δi2, j+Δi2] = 
        3 * S[i+Δi1, j+Δi1] * (Hcore[i+Δi2, i+Δi2] + Hcore[j+Δi2, j+Δi2]) / 8
    end
    Cˢ = getC(X, H)
    breakSymOfC(Val(HFT), Cˢ)
end


function getCfromHcore(::Val{HFT}, X::AbstractMatrix{T}, Hcore::AbstractMatrix{T}) where 
                      {HFT, T}
    Cˢ = getC(X, Hcore)
    breakSymOfC(Val(HFT), Cˢ)
end


function getCfromSAD(::Val{HFT}, S::AbstractMatrix{T}, 
                     Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
                     bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}, 
                     nuc::Tuple{String, Vararg{String, NNMO}}, 
                     nucCoords::Tuple{NTuple{D, T}, Vararg{NTuple{D, T}, NNMO}}, 
                     X::AbstractMatrix{T}, 
                     config=SCFconfig((:ADIIS,), (max(1e-2, 10getAtolVal(T)),))) where 
                    {HFT, T, D, NNMO}
    N₁tot = 0
    N₂tot = 0
    atmNs = fill((0,0), NNMO+1)
    order = sortperm(collect(nuc), by=x->AtomicNumberList[x])
    orderedNuc = nuc[order]
    for (i, N) in enumerate(orderedNuc .|> getCharge)
        N₁, N₂ = splitSpins(Val(:UHF), N)
        if N₂ > N₁ && N₂tot > N₁tot
            N₁, N₂ = N₂, N₁
        end
        N₁tot += N₁
        N₂tot += N₂
        atmNs[i] = (N₁, N₂)
    end

    len1, len2 = size(Hcore)
    Dᵅ = zeros(T, len1, len2)
    Dᵝ = zeros(T, len1, len2)
    for (atm, atmN, coord) in zip(orderedNuc, atmNs, nucCoords[order])
        h1 = coreH(bs, (atm,), (coord,))
        r, _ = runHFcore(Val(:UHF), 
                         config, atmN, h1, HeeI, S, X, getCfromHcore(Val(:UHF), X, h1), 
                         SADHFmaxStep, true, (false, false, false, false))
        Dᵅ += r[1].Ds[end]
        Dᵝ += r[2].Ds[end]
    end

    breakSymOfC(Val(HFT), Hcore, HeeI, X, Dᵅ, Dᵝ)
end


const C0methods = (GWH=getCfromGWH, Hcore=getCfromHcore, SAD=getCfromSAD)

getC0symbol(::iT) = :InputC
getC0symbol(::typeof(getCfromGWH)) = :GWH
getC0symbol(::typeof(getCfromSAD)) = :SAD
getC0symbol(::typeof(getCfromHcore)) = :Hcore
getC0symbol(::F) where {F<:Function} = Symbol(F)


function getD(Cˢ::AbstractMatrix{T}, Nˢ::Int) where {T}
    iBegin = firstindex(Cˢ, 1)
    @views (Cˢ[:, iBegin:(iBegin+Nˢ-1)]*Cˢ[:, iBegin:(iBegin+Nˢ-1)]')
end
# Nˢ: number of electrons with the same spin.

@inline getD(X::AbstractMatrix{T}, Fˢ::AbstractMatrix{T}, Nˢ::Int) where {T} = 
        getD(getC(X, Fˢ), Nˢ)


function getGcore(HeeI::AbstractArray{T1, 4}, DJ::T2, DK::T2) where 
                 {T1, T2<:AbstractMatrix{T1}}
    G = similar(DJ)
    Δi1 = firstindex(HeeI, 1) - 1
    Δi2 = firstindex(DJ, 1) - 1
    len = size(G, 1)
    Threads.@threads for k in (OneTo∘triMatEleNum)(len)
        μ, ν = convert1DidxTo2D(len, k)
        G[ν+Δi2, μ+Δi2] = G[μ+Δi2, ν+Δi2] = 
        dot(transpose(DJ), @view HeeI[μ+Δi1,ν+Δi1,:,:]) - 
        dot(          DK,  @view HeeI[μ+Δi1,:,:,ν+Δi1])
    end
    G
end

# RHF
@inline getG(HeeI::AbstractArray{T, 4}, (Dˢ,)::Tuple{AbstractMatrix{T}}) where {T} = 
        ( getGcore(HeeI, 2Dˢ, Dˢ), )

# UHF
@inline getG(HeeI::AbstractArray{T1, 4}, (Dᵅ, Dᵝ)::NTuple{2, T2}) where 
            {T1, T2<:AbstractMatrix{T1}} = 
        ( getGcore(HeeI, Dᵅ+Dᵝ, Dᵅ), getGcore(HeeI, Dᵅ+Dᵝ, Dᵝ) )


@inline getF(Hcore::AbstractMatrix{T1}, G::NTuple{HFTS, T2}) where 
            {T1, HFTS, T2<:AbstractMatrix{T1}} = 
        Ref(Hcore) .+ G

@inline getF(Hcore::AbstractMatrix{T1}, HeeI::AbstractArray{T1, 4}, 
             D::NTuple{HFTS, T2}) where {T1, HFTS, T2<:AbstractMatrix{T1}} = 
        getF(Hcore, getG(HeeI, D))


# RHF or UHF
@inline getE(Hcore::AbstractMatrix{T}, 
             Fˢ::AbstractMatrix{T}, Dˢ::AbstractMatrix{T}) where {T} = dot(Dˢ, Hcore+Fˢ) / 2

get2SpinQuantity(O::NTuple{HFTS, T}) where {HFTS, T} = abs(3-HFTS) * sum(O)
get2SpinQuantities(O, nRepeat::Int) = ntuple(_->get2SpinQuantity(O), nRepeat)

# RHF or UHF
getEhfCore(Hcore::AbstractMatrix{T1}, Fˢ::NTuple{HFTS, T2}, Dˢ::NTuple{HFTS, T2}) where 
          {T1, HFTS, T2<:AbstractMatrix{T1}} = 
get2SpinQuantity(getE.(Ref(Hcore), Fˢ, Dˢ))

# RHF or UHF
function getEhf(Hcore::AbstractMatrix{T1}, HeeI::AbstractArray{T1, 4}, 
                C::NTuple{HFTS, T2}, Ns::NTuple{HFTS, Int}) where 
               {T1, HFTS, T2<:AbstractMatrix{T1}}
    D = getD.(C, Ns)
    F = getF(Hcore, HeeI, D)
    getEhfCore(Hcore, F, D)
end

# RHF for MO
function getEhf((HcoreMO,)::Tuple{AbstractMatrix{T}}, 
                (HeeIMO,)::Tuple{AbstractArray{T, 4}}, (Nˢ,)::Tuple{Int}) where {T}
    shift1 = firstindex(HcoreMO, 1) - 1
    shift2 = firstindex( HeeIMO, 1) - 1
    term1 = 2 * (sum∘view)(diag(HcoreMO), OneTo(Nˢ).+shift1)
    term2 = T(0)
    rng = OneTo(Nˢ) .+ shift2
    for i in rng, j in rng
        term2 += 2 * HeeIMO[i,i,j,j] - HeeIMO[i,j,j,i]
    end
    term1 + term2
end

#  RHF for MO in GTBasis
function getEhf(gtb::GTBasis{T, D, BN}, 
                nuc::AVectorOrNTuple{String, NNMO}, 
                nucCoords::SpatialCoordType{T, D, NNMO}, 
                N::Union{Int, Tuple{Int}, NTuple{2, Int}}; 
                errorThreshold::Real=10getAtolVal(T)) where {T, D, BN, NNMO}
    Hcore = coreH(gtb, nuc, nucCoords)
    HeeI = gtb.eeI
    S = gtb.S
    if !isapprox(S, I, atol=errorThreshold)
        X = (Array∘getXcore1)(S)
        Hcore = changeHbasis(Hcore, X)
        HeeI = changeHbasis(HeeI, X)
    end
    getEhf((Hcore,), (HeeI,), splitSpins(Val(1), N))
end

# UHF for MO
function getEhf(HcoreMOs::NTuple{2, <:AbstractMatrix{T}}, 
                HeeIMOs::NTuple{2, <:AbstractArray{T, 4}}, 
                Jᵅᵝ::AbstractMatrix{T}, 
                Ns::NTuple{2, Int}) where {T}
    shift1 = firstindex(HcoreMOs[begin], 1) - 1
    shift2 = firstindex( HeeIMOs[begin], 1) - 1
    shift3 = firstindex(Jᵅᵝ, 1) - 1
    res = mapreduce(+, HcoreMOs, HeeIMOs, Ns) do HcoreMO, HeeIMO, Nˢ
        (sum∘view)(diag(HcoreMO), OneTo(Nˢ).+shift1) + 
        sum((HeeIMO[i,i,j,j] - HeeIMO[i,j,j,i]) for j in (OneTo(Nˢ ).+shift2)
                                                for i in (OneTo(j-1).+shift2))
    end
    res + sum(Jᵅᵝ[i,j] for i=OneTo(Ns[begin]).+shift3, j=OneTo(Ns[end]).+shift3)
end


function getCDFE(Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, X::AbstractMatrix{T}, 
                 Ns::NTuple{HFTS, Int}, F::NTuple{HFTS, AbstractMatrix{T}}) where {T, HFTS}
    Cnew = getC.(Ref(X), F)
    Dnew = getD.(Cnew, Ns)
    Fnew = getF(Hcore, HeeI, Dnew)
    Enew = getE.(Ref(Hcore), Fnew, Dnew)
    Dᵗnew = get2SpinQuantities(Dnew, HFTS)
    Eᵗnew = get2SpinQuantities(Enew, HFTS)
    map(themselves, Cnew, Dnew, Fnew, Enew, Dᵗnew, Eᵗnew)
end


mutable struct HFinterrelatedVars{T} <: HartreeFockintermediateData{T}
    Dtots::Vector{Matrix{T}}
    Etots::Vector{T}

    HFinterrelatedVars{T}() where {T} = new{T}()
    HFinterrelatedVars(Dts::AbstractVector{<:AbstractMatrix{T}}, 
                       Ets::AbstractVector{T}) where {T} = 
    new{T}(Dts, Ets)
end

const HFIVfields = (:Dtots, :Etots)

getSpinOccupations(::Val{:RHF}, (Nˢ,)::Tuple{Int}, BN) = 
((fill(spinOccupations[4], Nˢ)..., fill(spinOccupations[begin], BN-Nˢ)...),)

getSpinOccupations(::Val{:UHF}, (Nᵅ, Nᵝ)::NTuple{2, Int}, BN) = 
( (fill(spinOccupations[2], Nᵅ)..., fill(spinOccupations[begin], BN-Nᵅ)...), 
  (fill(spinOccupations[3], Nᵝ)..., fill(spinOccupations[begin], BN-Nᵝ)...) )

"""
    HFtempVars{T, HFT} <: HartreeFockintermediateData{T}

The container to store the intermediate values (only of the one spin configuration) for 
each iteration during the Hartree–Fock SCF procedure.

≡≡≡ Field(s) ≡≡≡

`N::Int`: The number of electrons with the one spin function.

`Cs::Vector{Matrix{T}}`: Orbital coefficient matrices.

`Ds::Vector{Matrix{T}}`: Density matrices corresponding to only one spin configuration.

`Fs::Vector{Matrix{T}}`: Fock matrices.

`Es::Vector{T}`: Part of the Hartree–Fock energy corresponding to one spin configuration.

`shared.Dtots::Vector{Matrix{T}}`: The total density matrices.

`shared.Etots::Vector{T}`: The total Hartree–Fock energy.

**NOTE:** For unrestricted Hartree–Fock, there are 2 `HFtempVars` being updated during the 
iterations, and changing the field `shared.Dtots` or `shared.Etots` of one `HFtempVars` 
will affect the other one's.
"""
struct HFtempVars{T, HFT} <: HartreeFockintermediateData{T}
    N::Int
    Cs::Vector{Matrix{T}}
    Ds::Vector{Matrix{T}}
    Fs::Vector{Matrix{T}}
    Es::Vector{T}
    shared::HFinterrelatedVars{T}
end

const HFTVVfields = (:Cs, :Ds, :Fs, :Es)

HFtempVars(::Val{HFT}, Nˢ::Int, 
           C::AbstractMatrix{T}, D::AbstractMatrix{T}, F::AbstractMatrix{T}, E::T) where 
          {HFT, T} = 
HFtempVars{T, HFT}(Nˢ, [C], [D], [F], [E], HFinterrelatedVars{T}())

HFtempVars(::Val{HFT}, Nˢ::Int, 
           Cs::AbstractVector{<:AbstractMatrix{T}}, 
           Ds::AbstractVector{<:AbstractMatrix{T}}, 
           Fs::AbstractVector{<:AbstractMatrix{T}}, 
           Es::AbstractVector{T}, 
           Dtots::AbstractVector{<:AbstractMatrix{T}}, Etots::AbstractVector{T}) where 
          {HFT, T} = 
HFtempVars{T, HFT}(Nˢ, Cs, Ds, Fs, Es, HFinterrelatedVars(Dtots, Etots))

function getHFTVforUpdate1(tVars::HFtempVars)
    getproperty.(Ref(tVars), HFTVVfields)
end

function getHFTVforUpdate2(tVars::HFtempVars)
    getproperty.(Ref(tVars.shared), HFIVfields)
end

function updateHFTVcore!(varMaxLen::Int, var::Vector{T}, res::T) where {T}
    length(var) < varMaxLen || popfirst!(var)
    push!(var, res)
end

function updateHFtempVars!(maxLens::NTuple{4, Int}, 
                           αβVars::NTuple{HFTS, HFtempVars{T, HFT}}, 
                           ress::NTuple{HFTS, 
                                        Tuple{AbstractMatrix{T}, AbstractMatrix{T}, 
                                              AbstractMatrix{T}, T, 
                                              AbstractMatrix{T}, T}}) where {HFTS, T, HFT}
    for (tVars, res) in zip(αβVars, ress)
        fs = getHFTVforUpdate1(tVars)
        for (s, f, r) in zip(maxLens, fs, res)
            updateHFTVcore!(s, f, r)
        end
    end
    for (s, f, r) in zip(maxLens[DEtotIndices], 
                          getHFTVforUpdate2(αβVars[begin]), ress[begin][end-1:end])
        updateHFTVcore!(s, f, r)
    end
end

function popHFtempVars!(αβVars::NTuple{HFTS, T}, counts::Int=1) where {HFTS, T<:HFtempVars}
    for tVars in αβVars
        fs = getHFTVforUpdate1(tVars)
        for fEach in fs
            for _ in OneTo(counts)
                pop!(fEach)
            end
        end
    end
    for fTot in getHFTVforUpdate2(αβVars[begin])
        for _ in OneTo(counts)
            pop!(fTot)
        end
    end
end

function clearHFtempVars!(saveTrace::NTuple{4, Bool}, αβVars::NTuple{HFTS, T}) where 
                         {HFTS, T<:HFtempVars}
    for tVars in αβVars
        fs = getHFTVforUpdate1(tVars)
        for (bl, fEach) in zip(saveTrace, fs)
            bl || deleteat!(fEach, firstindex(fEach):(lastindex(fEach)-1))
        end
    end
    for (bl, fTot) in zip(saveTrace[DEtotIndices], getHFTVforUpdate2(αβVars[begin]))
        bl || deleteat!(fTot, firstindex(fTot):(lastindex(fTot)-1))
    end
end


function initializeSCFcore(::Val{HFT}, Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
                           C::NTuple{HFTS, AbstractMatrix{T}}, Ns::NTuple{HFTS, Int}) where 
                          {HFT, T, HFTS}
    D = getD.(C, Ns)
    F = getF(Hcore, HeeI, D)
    E = getE.(Ref(Hcore), F, D)
    res = HFtempVars.(Val(HFT), Ns, C, D, F, E)
    sharedFields = getproperty.(res, :shared)
    for (field, val) in zip(HFIVfields, fill.(get2SpinQuantity.((D, E)), 1))
        setproperty!.(sharedFields, field, Ref(val))
    end
    res::NTuple{HFTS, HFtempVars{T, HFT}} # A somehow necessary assertion for type stability
end

# Additional wrapper to correlate `HTF` and `HFTS` for type stability.
initializeSCF(::Val{:RHF}, Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
              C::Tuple{AbstractMatrix{T}}, Ns::Tuple{Int}) where {T} = 
initializeSCFcore(Val(:RHF), Hcore, HeeI, C, Ns)

initializeSCF(::Val{:UHF}, Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
              C::NTuple{2, AbstractMatrix{T}}, Ns::NTuple{2, Int}) where {T} = 
initializeSCFcore(Val(:UHF), Hcore, HeeI, C, Ns)


const Doc_SCFconfig_OneRowTable = "|`:DIIS`, `:EDIIS`, `:ADIIS`|subspace size; "*
                                  "DIIS-Method solver; reset threshold¹|"*
                                  "`DIISsize`; `solver`; `resetThreshold`"*
                                  "|`1`,`2`...; `:LBFGS`...; `1e-14`... |"*
                                  "`$(defaultDIISsize)`; `:$(defaultDIISsolver)`;"*
                                  " N/A|"

const Doc_SCFconfig_DIIS = "[Direct inversion in the iterative subspace]"*
                           "(https://onlinelibrary.wiley.com/doi/10.1002/jcc.540030413)."
const Doc_SCFconfig_ADIIS = "[DIIS based on the augmented Roothaan–Hall (ARH) energy "*
                            "function](https://aip.scitation.org/doi/10.1063/1.3304922)."
const Doc_SCFconfig_LBFGSB = "[Limited-memory BFGS with box constraints]"*
                             "(https://github.com/Gnimuc/LBFGSB.jl)."

const Doc_SCFconfig_SPGB = "[Spectral Projected Gradient Method with box constraints]"*
                           "(https://github.com/m3g/SPGBox.jl)."

const Doc_SCFconfig_eg1 = "SCFconfig{Float64, 2, Tuple{Val{:ADIIS}, Val{:DIIS}}}(method, "*
                          "interval=(0.001, 1.0e-8), methodConfig, secondaryConvRatio, "*
                          "oscillateThreshold)"

"""

    SCFconfig{T, L, MS<:NTuple{L, Val}} <: ImmutableParameter{T, SCFconfig}

The `struct` for self-consistent field (SCF) iteration configurations.

≡≡≡ Field(s) ≡≡≡

`method::MS`: The applied convergence methods. They can be specified as the elements inside 
an `NTuple{L, Symbol}`, which is then input to the constructor of `SCFconfig` as the 
positional argument `methods`. The available configuration(s) corresponding to each method 
in terms of keyword arguments are:

| Convergence Method(s) | Configuration(s) | Keyword(s) | Range(s)/Option(s) | Default(s) |
| :----                 | :---:            | :---:      | :---:              |      ----: |
| `:DD`                 | damping strength |`dampStrength`|    [`0`, `1`]  |`$(defaultDS)`|
$(Doc_SCFconfig_OneRowTable)

¹ The reset threshold (`resetThreshold::Real`) determines when to clear the memory of the 
DIIS-based method's subspace and reset the second-to-latest residual vector as the first 
reference. The reset is executed when the latest computed energy increases an amount above 
the threshold compared to the second-to-latest computed energy. In default, the threshold 
is always slightly larger than the machine epsilon of the numerical data type for the SCF 
computation.

### Convergence Methods
* `:DD`: Direct diagonalization of the Fock matrix.
* `:DIIS`: $(Doc_SCFconfig_DIIS)
* `:EDIIS`: [Energy-DIIS](https://aip.scitation.org/doi/abs/10.1063/1.1470195).
* `:ADIIS`: $(Doc_SCFconfig_ADIIS)

### DIIS-Method Solvers
* `:LBFGS`: $(Doc_SCFconfig_LBFGSB)
* `:LCM`: Lagrange multiplier solver.
* `:SPGB`: $(Doc_SCFconfig_SPGB)

`interval::NTuple{L, T}`: The stopping (or skipping) thresholds for required methods. The 
last threshold will be the convergence threshold for the SCF procedure. When the last 
threshold is set to `NaN`, there will be no convergence detection.

`methodConfig::NTuple{L, Vector{<:Pair}}`: The additional keywords arguments for each 
method stored as `Tuple`s of `Pair`s.

`secondaryConvRatio::T`: The ratio of all the secondary convergence criteria (e.g., the 
convergence of density matrix, the error array based on the commutation relationship 
between the Fock matrix and the density matrix) to the primary convergence indicator, i.e., 
the convergence of the energy.

`oscillateThreshold::T`: The threshold for oscillatory convergence.

≡≡≡ Initialization Method(s) ≡≡≡

    SCFconfig(methods::NTuple{L, Symbol}, intervals::NTuple{L, T}, 
              config::Dict{Int, <:AbstractVector{<:Pair}}=Dict(1=>Pair[]); 
              secondaryConvRatio::Real=$(defaultSecConvRatio), 
              oscillateThreshold::Real=$(defaultOscThreshold)) where {L, T} -> 
    SCFconfig{T, L}

`methods` and `intervals` are the convergence methods to be applied and their stopping 
(or skipping) thresholds respectively. `config` specifies additional keyword argument(s) 
for each methods by a `Pair` of which the key `i::Int` is for `i`th method and the pointed 
`AbstractVector{<:Pair}` is the pairs of keyword arguments and their values respectively.

    SCFconfig(;threshold::AbstractFloat=$(defaultSCFconfigArgs[2]), 
               secondaryConvRatio::Real=$(defaultSecConvRatio), 
               oscillateThreshold::Real=defaultOscThreshold) -> 
    SCFconfig{$(defaultSCFconfigArgs[2] |> eltype), $(defaultSCFconfigArgs[1] |> length)}

`threshold` will update the stopping threshold of the default SCF configuration used in 
$(defaultHFCStr) with a new value. In other words, it updates the stopping threshold of 
`:$(defaultSCFconfigArgs[2][end])`.

≡≡≡ Example(s) ≡≡≡
```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> SCFconfig((:DD, :ADIIS, :DIIS), (1e-4, 1e-12, 1e-13), Dict(2=>[:solver=>:LCM]));

julia> SCFconfig(threshold=1e-8, oscillateThreshold=1e-5)
$(Doc_SCFconfig_eg1)
```
"""
struct SCFconfig{T, L, MS<:NTuple{L, Val}} <: ImmutableParameter{T, SCFconfig}
    method::MS
    interval::NTuple{L, T}
    methodConfig::NTuple{L, Vector{<:Pair}}
    secondaryConvRatio::T
    oscillateThreshold::T

    function SCFconfig(methods::NTuple{L, Symbol}, intervals::NTuple{L, T}, 
                       config::Dict{Int, <:AbstractVector{<:Pair}}=Dict(1=>Pair[]); 
                       secondaryConvRatio::Real=defaultSecConvRatio, 
                       oscillateThreshold::Real=defaultOscThreshold) where {L, T}
        any(i < 0 for i in intervals) && throw(DomainError(intervals, "Thresholds in "*
                                               "`intervals` must all be non-negative."))
        kwPairs = [Pair[] for _ in OneTo(L)]
        for i in keys(config)
            kwPairs[i] = config[i]
        end
        methods = Val.(methods)
        new{T, L, typeof(methods)}(methods, intervals, Tuple(kwPairs), 
                                   secondaryConvRatio, oscillateThreshold)
    end
end

const defaultSCFconfig = SCFconfig(defaultSCFconfigArgs...)

SCFconfig(;threshold::AbstractFloat=defaultSCFconfigArgs[2][end], 
          secondaryConvRatio::Real=defaultSecConvRatio, 
          oscillateThreshold::Real=defaultOscThreshold) = 
SCFconfig( defaultSCFconfigArgs[1], 
          (defaultSCFconfigArgs[2][begin:end-1]..., Float64(threshold)); 
           secondaryConvRatio, oscillateThreshold )

function getMaxSCFsizes(scfConfig::SCFconfig)
    maxSize = 0
    for (m, c) in zip(scfConfig.method, scfConfig.methodConfig)
        newSize = if m == :DD
            1
        else # xDIIS
            idx = findfirst(isequal(:DIISsize), getindex.(c, 1))
            if idx === nothing
                defaultDIISsize
            else
                c[idx][end]
            end
        end
        maxSize = max(maxSize, newSize)
    end
    (1, maxSize, maxSize, maxSize) # Cs, Ds, Fs, Es
end


"""

    HFfinalVars{T, D, HFT, NN, BN, HFTS} <: HartreeFockFinalValue{T, HFT}

The container of the final values after a Hartree–Fock SCF procedure.

≡≡≡ Field(s) ≡≡≡

`Ehf::T`: Hartree–Fock energy of the electronic Hamiltonian.

`Enn::T`: The nuclear repulsion energy.

`Ns::NTuple{HFTS, Int}`: The number(s) of electrons with same spin configurations(s). For 
restricted closed-shell Hartree–Fock (RHF), the single element in `.Ns` represents both 
spin-up electrons and spin-down electrons.

`nuc::NTuple{NN, String}`: The nuclei in the studied system.

`nucCoords::NTuple{NN, NTuple{D, T}}`: The coordinates of corresponding nuclei.

`C::NTuple{HFTS, Matrix{T}}`: Orbital coefficient matrix(s) for one spin configuration.

`D::NTuple{HFTS, Matrix{T}}`: Density matrix(s) for one spin configuration.

`F::NTuple{HFTS, Matrix{T}}`: Fock matrix(s) for one spin configuration.

`Eo::NTuple{HFTS, Vector{T}}`: Energies of canonical orbitals.

`occu::NTuple{HFTS, NTuple{BN, Int}}`: Occupations of canonical orbitals.

`temp::NTuple{HFTS, [HFtempVars](@ref){T, HFT}}`: the intermediate values stored during 
the Hartree–Fock interactions.

`isConverged::Union{Bool, Missing}`: Whether the SCF iteration is converged in the end. 
When convergence detection is off (see [SCFconfig](@ref)), it is set to `missing`.

`basis::GTBasis{T, D, BN}`: The basis set used for the Hartree–Fock approximation.
"""
struct HFfinalVars{T, D, HFT, NN, BN, HFTS} <: HartreeFockFinalValue{T, HFT}
    Ehf::T
    Enn::T
    Ns::NTuple{HFTS, Int}
    nuc::NTuple{NN, String}
    nucCoord::NTuple{NN, NTuple{D, T}}
    C::NTuple{HFTS, Matrix{T}}
    D::NTuple{HFTS, Matrix{T}}
    F::NTuple{HFTS, Matrix{T}}
    Eo::NTuple{HFTS, Vector{T}}
    occu::NTuple{HFTS, NTuple{BN, String}}
    temp::NTuple{HFTS, HFtempVars{T, HFT}}
    isConverged::Union{Bool, Missing}
    basis::GTBasis{T, D, BN}

    function HFfinalVars(basis::GTBasis{T, 𝐷, BN}, 
                         nuc::AVectorOrNTuple{String, NNMO}, 
                         nucCoords::SpatialCoordType{T, 𝐷, NNMO}, 
                         X::AbstractMatrix{T}, 
                         vars::NTuple{HFTS, HFtempVars{T, HFT}}, 
                         isConverged::Union{Bool, Missing}) where 
                        {T, 𝐷, BN, NNMO, HFTS, HFT}
        (NNval = length(nuc)) == length(nucCoords) || 
        throw(AssertionError("The length of `nuc` and `nucCoords` should be the same."))
        any(length(i)!=𝐷 for i in nucCoords) && 
        throw(DomainError(nucCoords, "The lengths of the elements in `nucCoords` should "*
               "all be length $𝐷."))
        Ehf = vars[begin].shared.Etots[end]
        nuc = arrayToTuple(nuc)
        nucCoords = genTupleCoords(T, nucCoords)
        Enn = nnRepulsions(nuc, nucCoords)
        Ns = getproperty.(vars, :N)
        C = last.(getproperty.(vars, :Cs))
        D = last.(getproperty.(vars, :Ds))
        F = last.(getproperty.(vars, :Fs))
        Eo = getindex.(getCϵ.(Ref(X), F), 2)
        occu = getSpinOccupations(Val(HFT), Ns, BN)
        new{T, 𝐷, HFT, NNval, BN, HFTS}(Ehf, Enn, Ns, nuc, nucCoords, 
                                        C, D, F, Eo, occu, vars, isConverged, basis)
    end
end

struct InitialC{T<:Number, HFT, F<:Function}
    mat::NTuple{<:Any, Matrix{T}}
    f::F

    InitialC(::Val{HFT}, f::F, ::Type{T}) where {HFT, F, T} = new{T, HFT, F}((), f)

    InitialC(::Val{:RHF}, C0::NTuple{1, AbstractMatrix{T}}) where {T} = 
    new{T, :RHF, iT}(C0, itself)

    InitialC(::Val{:UHF}, C0::NTuple{2, AbstractMatrix{T}}) where {T} = 
    new{T, :UHF, iT}(C0, itself)
end

const defaultHFconfigPars = [:RHF, :SAD, defaultSCFconfig, defaultHFmaxStep, true, 
                             defaultHFsaveTrace]

"""

    HFconfig{T1, HFT, F, T2, L, MS} <: ConfigBox{T1, HFconfig, HFT}

The container of Hartree–Fock method configuration.

≡≡≡ Field(s) ≡≡≡

`HF::Val{HFT}`: Hartree–Fock method type. Available values of `HFT` are 
$(string(HFtypes)[2:end-1]).

`C0::InitialC{T1, HFT, F}`: Initial guess of the orbital coefficient matrix(s) C of the 
canonical orbitals. When `C0` is as an argument of `HFconfig`'s constructor, it can be set 
to `sym::Symbol` where available values of `sym` are 
`$((C0methods|>typeof|>fieldnames|>string)[2:end-1])`; it can also be a `Tuple` of 
prepared orbital coefficient matrix(s) for the corresponding Hartree–Fock method type.

`SCF::SCFconfig{T2, L, MS}`: SCF iteration configuration. For more information please refer 
to [`SCFconfig`](@ref).

`maxStep::Int`: Maximum iteration steps allowed regardless if the iteration converges.

`earlyStop::Bool`: Whether automatically terminate (or skip) a convergence method early 
when its performance becomes unstable or poor.

`saveTrace::NTuple{4, Bool}`: Determine whether saving (by pushing) the intermediate 
information from all the iterations steps to the field `.temp` of the output 
[`HFfinalVars`](@ref) of `runHF`. The types of relevant information are:

| Sequence | Information | Corresponding field in [`HFtempVars`](@ref) |
|  :---:   |    :---:    |                   :---:                     |
| 1 | orbital coefficient matrix(s)      | `.Cs`                       |
| 2 | density matrix(s)                  | `.Ds`, `.shared.Dtots`      |
| 3 | Fock matrix(s)                     | `.Fs`                       |
| 4 | unconverged Hartree–Fock energy(s) | `.Es`, `.shared.Etots`      |

≡≡≡ Initialization Method(s) ≡≡≡

    HFconfig(;HF::Union{Symbol, Val}=:$(defaultHFconfigPars[1]), 
              C0::Union{Tuple{AbstractMatrix}, NTuple{2, AbstractMatrix}, 
                        Symbol, Val}=:$(defaultHFconfigPars[2]), 
              SCF::SCFconfig=$(defaultHFconfigPars[3]), 
              maxStep::Int=$(defaultHFconfigPars[4]), 
              earlyStop::Bool=$(defaultHFconfigPars[5]), 
              saveTrace::NTuple{4, Bool}=$(defaultHFconfigPars[6])) -> 
    HFconfig

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> HFconfig();

julia> HFconfig(HF=:UHF);
```
"""
mutable struct HFconfig{T1, HFT, F, T2, L, MS} <: ConfigBox{T1, HFconfig, HFT}
    HF::Val{HFT}
    C0::InitialC{T1, HFT, F}
    SCF::SCFconfig{T2, L, MS}
    maxStep::Int
    earlyStop::Bool
    saveTrace::NTuple{4, Bool}

    HFconfig(::Val{:UHF}, 
             a2::NTuple{2, AbstractMatrix{T1}}, a3::SCFconfig{T2, L, MS}, a4, a5, a6) where 
            {T1, T2, L, MS} = 
    new{T1, :UHF, iT, T2, L, MS}(Val(:UHF), InitialC(Val(:UHF), a2), a3, a4, a5, a6)

    HFconfig(::Val{:RHF}, 
             a2::Tuple{AbstractMatrix{T1}}, a3::SCFconfig{T2, L, MS}, a4, a5, a6) where 
            {T1, T2, L, MS} = 
    new{T1, :RHF, iT, T2, L, MS}(Val(:RHF), InitialC(Val(:RHF), a2), a3, a4, a5, a6)

    function HFconfig(::Val{HFT}, a2::Val{CF}, a3::SCFconfig{T, L, MS}, a4, a5, a6) where 
                     {T, HFT, CF, L, MS}
        f = getproperty(C0methods, CF)
        new{T, HFT, typeof(f), T, L, MS}(Val(HFT), InitialC(Val(HFT), f, T), a3, a4, a5, a6)
    end
end

HFconfig(a1::Symbol, a2, args...) = HFconfig(Val(a1), a2, args...)

HFconfig(a1, a2::Symbol, args...) = HFconfig(a1, Val(a2), args...)

HFconfig(a1::Symbol, a2::Symbol, args...) = HFconfig(Val(a1), Val(a2), args...)

HFconfig(t::NamedTuple) = genNamedTupleC(:HFconfig, defaultHFconfigPars)(t)

HFconfig(;kws...) = 
length(kws) == 0 ? HFconfig(defaultHFconfigPars...) : HFconfig(kws|>NamedTuple)

const defaultHFC = Meta.parse(defaultHFCStr) |> eval


const C0methodArgOrders = (itself=(1,), 
                           getCfromGWH=(2,3,5,4), 
                           getCfromHcore=(2,4,5), 
                           getCfromSAD=(2,3,5,6,7,8,9,4))


"""
    runHF(bs, nuc, nucCoords, config=$(defaultHFCStr), N=getCharge(nuc); 
          printInfo=true, infoLevel=$(defaultHFinfoL)) -> 
    HFfinalVars

    runHF(bs, nuc, nucCoords, N=getCharge(nuc), config=$(defaultHFCStr); 
          printInfo=true, infoLevel=$(defaultHFinfoL)) -> 
    HFfinalVars

Main function to run a Hartree–Fock method in Quiqbox. The returned result and relevant 
information is stored in a [`HFfinalVars`](@ref).

    runHFcore(args...; printInfo=false, infoLevel=$(defaultHFinfoL)) -> 
    Tuple{Tuple{Vararg{HFtempVars}}, Bool}

The core function of `runHF` that accept the same positional arguments as `runHF`, except 
it returns the data (`HFtempVars`) collected during the iteration and the boolean result of 
whether the SCF procedure is converged.

≡≡≡ Positional argument(s) ≡≡≡

`bs::Union{
    BasisSetData{T, D}, 
    AbstractVector{<:AbstractGTBasisFuncs{T, D}}, 
    Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}
} where {T, D}`: The basis set used for the Hartree–Fock approximation.

`nuc::Union{
    Tuple{String, Vararg{String, NNMO}} where NNMO, 
    AbstractVector{String}
}`: The nuclei in the studied system.

`nucCoords::$(SpatialCoordType)`: The coordinates of corresponding nuclei.

`config::HFconfig`: The Configuration of selected Hartree–Fock method. For more information 
please refer to [`HFconfig`](@ref).

`N::Union{Int, Tuple{Int}, NTuple{2, Int}}`: Total number of electrons, or the number(s) of 
electrons with same spin configurations(s). **NOTE:** `N::NTuple{2, Int}` is only supported 
by unrestricted Hartree–Fock (UHF).

≡≡≡ Keyword argument(s) ≡≡≡

`printInfo::Bool`: Whether print out the information of iteration steps and result.

`infoLevel::Int`: Printed info's level of details when `printInfo=true`. The higher 
(the absolute value of) it is, more intermediate steps will be printed. Once `infoLevel` 
achieve `5`, every step will be printed.
"""
function runHF(bs::GTBasis{T}, args...; 
               printInfo::Bool=true, infoLevel::Int=defaultHFinfoL) where {T}
    nuc = arrayToTuple(args[begin])
    nucCoords = genTupleCoords(T, args[begin+1])
    vars, isConverged = runHFcore(bs, nuc, nucCoords, args[begin+2:end]...; 
                                  printInfo, infoLevel)
    res = HFfinalVars(bs, nuc, nucCoords, getX(bs.S), vars, isConverged)
    roundDigits = min(DefaultDigits, getAtolDigits(T))
    if printInfo
        EhfStr  = alignNum(res.Ehf, 0; roundDigits)
        EnnStr  = alignNum(res.Enn, 0; roundDigits)
        EtotStr = alignNum(res.Ehf+res.Enn, 0; roundDigits)
        len = max(19, length(EhfStr)+3, length(EnnStr)+3, length(EtotStr)+3)
        println(rpad("Hartree–Fock Energy", len), " ¦ ", rpad("Nuclear Repulsion", len), 
                " ¦ Total Energy")
        println(rpad(EhfStr*" Ha", len), "   ", rpad(EnnStr*" Ha", len), "   ", 
                EtotStr, " Ha\n")
    end
    res
end

runHF(bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}, args...; 
      printInfo::Bool=true, infoLevel::Int=defaultHFinfoL) where {T, D} = 
runHF(GTBasis(bs), args...; printInfo, infoLevel)

function runHFcore(bs::GTBasis{T, D, BN, BFT}, 
                   nuc::AVectorOrNTuple{String, NNMO}, 
                   nucCoords::SpatialCoordType{T, D, NNMO}, 
                   N::Union{Int, Tuple{Int}, NTuple{2, Int}}=getCharge(nuc), 
                   config::HFconfig{<:Any, HFT}=defaultHFC; 
                   printInfo::Bool=false, 
                   infoLevel::Int=defaultHFinfoL) where {T, D, BN, BFT, NNMO, HFT}
    timerBool = printInfo && infoLevel > 2
    timerBool && (tBegin = time_ns())
    Nlow = Int(HFT==:RHF)
    Ntot = (N isa Int) ? N : (N[begin] + N[end])
    Ntot > Nlow || throw(DomainError(N, "$(HFT) requires more than $(Nlow) electrons."))
    Ns = splitSpins(Val(HFT), N)
    leastNb = max(Ns...)
    BN < leastNb &&  throw(DomainError(BN, "The number of basis functions should be no "*
                           "less than $(leastNb)."))
    nuc = arrayToTuple(nuc)
    nucCoords = genTupleCoords(T, nucCoords)
    Hcore = coreH(bs, nuc, nucCoords)
    X = getX(bs.S)
    getC0f = config.C0.f
    C0mats = config.C0.mat
    getC0f isa iT && 
    ( all(all(size(C) .== BN) for C in C0mats) || 
      throw(DimensionMismatch("The input initial orbital coefficient matrix(s)'s size "*
            "($(size.(C0mats))) does not match the size of the input basis set ($BN).")) )
    C0 = uniCallFunc(getC0f, getproperty(C0methodArgOrders, nameOf(getC0f)), C0mats, 
                     Val(HFT), bs.S, X, Hcore, bs.eeI, bs.basis, nuc, nucCoords)
    timerBool && (tEnd = time_ns())
    if printInfo && infoLevel > 1
        roundDigits = min(DefaultDigits, getAtolDigits(T))
        println(ifelse(Ntot>1, "Many", "Single"), "-Electron System Information: ")
        println("•Electron Number: ", Ntot)
        println("•Nuclear Coordinate: ")
        for (atm, coord) in zip(nuc, nucCoords)
            println(" ∘", atm, ": ", "[", alignNumSign(coord[1]; roundDigits), ", ", 
                                         alignNumSign(coord[2]; roundDigits), ", ", 
                                         alignNumSign(coord[3]; roundDigits), "]")
        end
        println()
        print("Hartree–Fock (HF) Initialization")
        timerBool && print(" (Finished in ", genTimeStr(tEnd - tBegin), ")")
        println(":")
        println("•HF Type: ", HFT)
        println("•Basis Set Size: ", BN)
        println("•Initial Guess Method: ", getC0symbol(getC0f))
    end
    runHFcore(Val(HFT), config.SCF, Ns, Hcore, bs.eeI, bs.S, X, 
              C0, config.maxStep, config.earlyStop, config.saveTrace, printInfo, infoLevel)
end

runHFcore(bs::BasisSetData, nuc, nucCoords, config::HFconfig, N=getCharge(nuc); 
          printInfo::Bool=false, infoLevel::Int=defaultHFinfoL) = 
runHFcore(bs::BasisSetData, nuc, nucCoords, N, config; printInfo, infoLevel)

runHFcore(bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}, args...; 
          printInfo::Bool=false, infoLevel::Int=defaultHFinfoL) where {T, D} = 
runHFcore(GTBasis(bs), args...; printInfo, infoLevel)

"""

    runHFcore(HTtype, scfConfig, Ns, Hcore, HeeI, S, X, C0, maxStep, earlyStop, saveTrace, 
              printInfo=false, infoLevel=$(defaultHFinfoL)) -> 
    Tuple{Tuple{Vararg{HFtempVars}}, Bool}

Another method of `runHFcore` that has the same return value, but takes more underlying 
data as arguments.

=== Positional argument(s) ===

`HTtype::Val{HFT} where HFT`: Hartree–Fock method type. Available values of `HFT` are 
$(string(HFtypes)[2:end-1]).

`scfConfig::SCFconfig`: The SCF iteration configuration.

`Ns::NTuple{HFTS, Int} where HFTS`: The numbers of electrons with same spin configurations. 

`Hcore::AbstractMatrix{T} where T`: The core Hamiltonian of the electronic Hamiltonian.

`HeeI::AbstractArray{T, 4} where T`: The electron-electron interaction tensor (in the 
chemists' notation) which includes both the Coulomb interactions and the Exchange 
Correlations.

`S::AbstractMatrix{T} where T`: The overlap matrix of the used basis set.

`X::AbstractMatrix{T} where T`: The transformation matrix of `S`.

`C0::NTuple{HFTS, AbstractMatrix{T}} where {HFTS, T}`: Initial guess of the coefficient 
matrix(s) of the canonical spin-orbitals.

`maxStep::Int`: Maximum iteration steps allowed regardless if the iteration converges.

`earlyStop::Bool`: Whether automatically terminate (or skip) a convergence method early 
when its performance becomes unstable or poor.

`saveTrace::NTuple{4, Bool}`: Determine whether saving (by pushing) the intermediate 
information from all the iterations steps to the output [`HFtempVars`](@ref) of 
`runHFcore`. Its definition is the same as the field `.saveTrace` inside a 
[`HFconfig`](@ref).

`printInfo::Bool`: Whether print out the information of iteration steps and result.

`infoLevel::Int`: Printed info's level of details when `printInfo=true`. The higher 
(the absolute value of) it is, more intermediate steps and other information will be 
printed. Once `infoLevel` achieve `5`, every step and all available information will be 
printed.
"""
function runHFcore(::Val{HFT}, 
                   scfConfig::SCFconfig{T1, L, MS}, 
                   Ns::NTuple{HFTS, Int}, 
                   Hcore::AbstractMatrix{T2}, 
                   HeeI::AbstractArray{T2, 4}, 
                   S::AbstractMatrix{T2}, 
                   X::AbstractMatrix{T2}, 
                   C0::NTuple{HFTS, AbstractMatrix{T2}}, 
                   maxStep::Int, 
                   earlyStop::Bool, 
                   saveTrace::NTuple{4, Bool}, 
                   printInfo::Bool=false, 
                   infoLevel::Int=defaultHFinfoL) where {HFT, T1, L, MS, HFTS, T2}
    timerBool = printInfo && infoLevel > 2
    timerBool && (tBegin = time_ns())

    vars = initializeSCF(Val(HFT), Hcore, HeeI, C0, Ns)
    secondaryConvRatio = scfConfig.secondaryConvRatio
    varsShared = vars[begin].shared
    Etots = varsShared.Etots
    ΔEs = zeros(T2, 1)
    ΔDrms = zeros(T2, 1)
    δFrms = T2[getErrorNrms(vars, S)]
    endThreshold = scfConfig.interval[end]
    detectConvergence = !isnan(endThreshold)
    isConverged::Union{Bool, Missing, Int} = true
    rollbackRange = 0 : (HFminItr÷3)
    rollbackCount = length(rollbackRange)
    i = 0

    if printInfo
        roundDigits = setNumDigits(T2, endThreshold)
        titles = ("Step", "E (Ha)", "ΔE (Ha)", "RMS(FDS-SDF)", "RMS(ΔD)")
        titleRange = 1 : (3 + 2*(infoLevel > 1))
        colSpaces = (
            max(ndigits(maxStep), (length∘string)(HFT), length(titles[begin])), 
            roundDigits + (ndigits∘floor)(Int, Etots[]) + 2, 
            roundDigits + 3
        )
        titleStr = ""
        colSpaces = map(titles[titleRange], (1, 2, 3, 3, 3)[titleRange]) do title, idx
            printSpace = max(length(title), colSpaces[idx])
            titleStr *= "| " * rpad(title, printSpace) * " "
            printSpace
        end

        if infoLevel > 0
            println("•Initial HF energy E: ", alignNum(Etots[], 0; roundDigits), " Ha")
            println("•Initial RMS(FDS-SDF): ", 
                      alignNum(δFrms[], 0; roundDigits))
            println("•Convergence Threshold: ", endThreshold, " a.u.")
            if infoLevel > 2
                println("•Secondary Convergence Threshold: ", 
                        secondaryConvRatio*endThreshold, " a.u.")
                println("•Oscillatory Convergence Threshold: ", 
                        scfConfig.oscillateThreshold, " a.u.")
            end
            println()
            println("Self-Consistent Field (SCF) Iteration:")
            (println∘repeat)('=', length(titleStr))
            println(titleStr)
            (println∘replace)(titleStr, r"[^|]"=>'=')
        end
    end

    adaptStepBl = genAdaptStepBl(infoLevel, maxStep)
    maxLens = map(saveTrace, 
                  getMaxSCFsizes(scfConfig), HFinterValStoreSizes) do bl, scfSize, storeSize
        max((ifelse(bl, maxStep+1, 1)), scfSize, storeSize)
    end

    for (MVal, kws, breakPoint, l) in 
        zip(fieldtypes(MS), scfConfig.methodConfig, scfConfig.interval, 1:L)
        HFcore, keyArgs = genHFcore(MVal, vars, S, X, Ns, Hcore, HeeI; kws...)
        oscThreshold = max(breakPoint, scfConfig.oscillateThreshold)
        flucThreshold = max(10breakPoint, 1.5e-3) # ≈3.8kJ/mol (0.95 chemical accuracy)
        m = getValParm(MVal)
        isConverged = false
        endM = l==L
        n = 0

        if printInfo && infoLevel > 1
            print('|', repeat('–', colSpaces[1]+1), "<$l>–", ("[:$m"))
            if infoLevel > 2
                kaStr = mapreduce(*, keyArgs) do ka
                    key = ka[begin]
                    val = ka[end]
                    string(key) * "=" * ifelse(val isa Symbol, ":", "") * string(val) * ", "
                end
                print(", (", kaStr[begin:end-2], ")")
            end
            println("]")
        end

        while true
            i < maxStep || break
            i += 1
            n += 1

            updateHFtempVars!(maxLens, vars, HFcore())

            push!(ΔEs, Etots[end] - Etots[end-1])
            if endM || printInfo
                push!(ΔDrms, rmsOf(varsShared.Dtots[end] - varsShared.Dtots[end-1]))
                push!(δFrms, getErrorNrms(vars, S))
            end
            ΔEᵢ = ΔEs[end]
            ΔDrmsᵢ = ΔDrms[end]
            δFrmsᵢ = δFrms[end]
            ΔEᵢabs = abs(ΔEᵢ)

            if printInfo && infoLevel > 0 && (adaptStepBl(i) || i == maxStep)
                print( "| ", rpad("$i", colSpaces[1]), 
                      " | ", cropStrR(alignNumSign(Etots[end]; roundDigits), colSpaces[2]), 
                      " | ", cropStrR(alignNumSign(ΔEᵢ; roundDigits), colSpaces[3]) )
                if infoLevel > 1
                    print( " | ", cropStrR(alignNum(δFrmsᵢ, 0; roundDigits), colSpaces[4]), 
                           " | ", cropStrR(alignNum(ΔDrmsᵢ, 0; roundDigits), colSpaces[5]) )
                end
                println()
            end

            convThresholds = ifelse(δFrmsᵢ <= secondaryConvRatio*breakPoint, 
                                    (1, secondaryConvRatio), (0, 0)) .* breakPoint
            ΔEᵢabs <= convThresholds[begin] && ΔDrmsᵢ <= convThresholds[end] && 
            (isConverged = true; break)

            # oscillating convergence & early termination of non-convergence.
            if n > 1 && i > HFminItr && ΔEᵢ > flucThreshold
                isOsc, _ = isOscillateConverged(Etots, 10oscThreshold, 
                                                minLen=HFminItr, 
                                                maxRemains=HFinterEstoreSize)
                if isOsc
                    if ΔEᵢabs <= oscThreshold && 
                       (endM ? (δFrmsᵢ <= secondaryConvRatio*oscThreshold && 
                                ΔDrmsᵢ <= secondaryConvRatio*oscThreshold) : true)
                        isConverged = 1
                        break
                    end
                else
                    if earlyStop
                        isRaising = all(rollbackRange) do j
                            ΔEs[end-j] > 10flucThreshold
                        end
                        if isRaising
                            rbCount = min(rollbackCount, length(Etots))
                            terminateSCF!(vars, rbCount, m, printInfo)
                            i -= rbCount
                            break
                        end
                    end
                end
            end
        end
    end

    timerBool && (tEnd = time_ns())

    if printInfo
        tStr = timerBool ? " after "*genTimeStr(tEnd - tBegin) : ""
        negStr = if detectConvergence
            ifelse(isConverged===1, (isConverged=true; "converged to an oscillation"), 
                   ifelse(isConverged, "converged", "stopped but not converged"))
        else
            "stopped"
        end
        println("\nThe SCF iteration of $HFT has ", negStr, " at step $i", tStr, ":\n", 
                "|ΔE| → ", alignNum(abs(ΔEs[end]), 0; roundDigits), " Ha, ", 
                "RMS(FDS-SDF) → ", alignNum(δFrms[end], 0; roundDigits), ", ", 
                "RMS(ΔD) → ", alignNum(ΔDrms[end], 0; roundDigits), ".\n")
    end
    clearHFtempVars!(saveTrace, vars)
    detectConvergence || (isConverged = missing)
    vars, isConverged
end

function getErrorNrms(vars::NTuple{HFTS, HFtempVars{T, HFT}}, 
                      S::AbstractMatrix{T}) where {HFTS, T, HFT}
    mapreduce(+, vars) do tVar
        D = tVar.Ds[end]
        F = tVar.Fs[end]
        (rmsOf∘getEresidual)(F, D, S)
    end / length(vars)
end

function terminateSCF!(vars, counts, method, printInfo)
    popHFtempVars!(vars, counts)
    printInfo && println("Early termination of ", method, " due to its poor performance.")
end


function DDcore(Nˢ::Int, X::AbstractMatrix{T}, F::AbstractMatrix{T}, D::AbstractMatrix{T}, 
                dampStrength::T) where {T}
    0 <= dampStrength <= 1 || throw(DomainError(dampStrength, "The value of `dampStrength`"*
                                    " should be between 0 and 1."))
    Dnew = getD(X, F, Nˢ)
    (1 - dampStrength)*Dnew + dampStrength*D
end

function genDD(αβVars::NTuple{HFTS, HFtempVars{T, HFT}}, X::AbstractMatrix{T}, 
               Ns::NTuple{HFTS, Int}, Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}; 
               dampStrength::T=T(defaultDS)) where {HFTS, T, HFT}
    f = function ()
        Fs = last.(getproperty.(αβVars, :Fs))
        Ds = last.(getproperty.(αβVars, :Ds))
        Dnew = DDcore.(Ns, Ref(X), Fs, Ds, dampStrength)
        getCDFE(Hcore, HeeI, X, Ns, getF(Hcore, HeeI, Dnew))
    end
    keyArgs = (:dampStrength=>dampStrength,)
    f, keyArgs
end


function EDIIScore(Ds::Vector{<:AbstractMatrix{T}}, ∇s::Vector{<:AbstractMatrix{T}}, 
                   Es::Vector{T}) where {T}
    len = length(Ds)
    B = similar(∇s[begin], len, len)
    Δi = firstindex(B, 1) - 1
    Threads.@threads for k in (OneTo∘triMatEleNum)(len)
        i, j = convert1DidxTo2D(len, k)
        @inbounds B[i+Δi, j+Δi] = B[j+Δi, i+Δi] = -dot(Ds[i]-Ds[j], ∇s[i]-∇s[j])
    end
    Es, B
end

function ADIIScore(Ds::Vector{<:AbstractMatrix{T}}, ∇s::Vector{<:AbstractMatrix{T}}) where 
                  {T}
    v = dot.(Ds .- Ref(Ds[end]), Ref(∇s[end]))
    DsL = Ds[end]
    ∇sL = ∇s[end]
    B = map(Iterators.product(eachindex(Ds), eachindex(∇s))) do (i,j)
        @inbounds dot(Ds[i]-DsL, ∇s[j]-∇sL)
    end
    v, B
end

getEresidual(F::AbstractMatrix{T}, D::AbstractMatrix{T}, S::AbstractMatrix{T}) where {T} = 
F*D*S - S*D*F

function DIIScore(Ds::Vector{<:AbstractMatrix{T}}, ∇s::Vector{<:AbstractMatrix{T}}, 
                  S::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T}
    len = length(Ds)
    B = similar(∇s[begin], len, len)
    v = zeros(T, len)
    Δi = firstindex(B, 1) - 1
    Threads.@threads for k in (OneTo∘triMatEleNum)(len)
        i, j = convert1DidxTo2D(len, k)
        @inbounds B[i+Δi, j+Δi] = B[j+Δi, i+Δi] = dot( X'*getEresidual(∇s[i], Ds[i], S)*X, 
                                                       X'*getEresidual(∇s[j], Ds[j], S)*X )
    end
    v, B
end

#                     convex constraint|unified function signature
const DIISconfigs = ( DIIS=(Val(false), (Ds, ∇s, Es, S, X)-> DIIScore(Ds, ∇s, S, X)), 
                      EDIIS=(Val(true), (Ds, ∇s, Es, S, X)->EDIIScore(Ds, ∇s, Es)), 
                      ADIIS=(Val(true), (Ds, ∇s, Es, S, X)->ADIIScore(Ds, ∇s)) )

function xDIIScore!(mDIIS::F, c::Vector{T}, S::AbstractMatrix{T}, X::AbstractMatrix{T}, 
                    Ds::Vector{<:AbstractMatrix{T}}, 
                    Fs::Vector{<:AbstractMatrix{T}}, 
                    Es::Vector{T}, 
                    cvxConstraint::Val{CCB}, 
                    solver::Symbol) where {F<:Function, T, CCB}
    v, B = mDIIS(Ds, Fs, Es, S, X)
    constraintSolver!(cvxConstraint, c, v, B, solver)
    sum(c.*Fs) # Fnew
end

function genxDIIS(::Type{Val{M}}, αβVars::NTuple{HFTS, HFtempVars{T, HFT}}, 
                  S::AbstractMatrix{T}, X::AbstractMatrix{T}, Ns::NTuple{HFTS, Int}, 
                  Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}; 
                  resetThreshold::Real=10getAtolVal(T), 
                  DIISsize::Int=defaultDIISsize, 
                  solver::Symbol=defaultDIISsolver) where {M, HFTS, T, HFT}
    DIISsize < 2 && (throw∘DomainError)(intervals, "$M space need to be at least 2.")
    DFEsyms = HFTVVfields[begin+1:end]
    DFElens = map(DFEsyms) do fieldSym
        getproperty(αβVars[begin], fieldSym) |> length
    end
    initialSize = min(DIISsize, DFElens...)
    cs = Tuple(collect(T, OneTo(initialSize)) for _ in OneTo(HFTS))
    Dss, Fss, Ess = map(DFEsyms) do fieldSym
        fs = getproperty.(αβVars, fieldSym)
        iEnd = lastindex(fs[begin])
        getindex.(fs, Ref(iEnd-initialSize+1:iEnd))
    end
    cvxConstraint, mDIIS = getproperty(DIISconfigs, M)

    f = function ()
        Fn = xDIIScore!.(mDIIS, cs, Ref(S), Ref(X), Dss, Fss, Ess, cvxConstraint, solver)
        res = getCDFE(Hcore, HeeI, X, Ns, Fn)
        push!.(cs, 1)
        push!.(Dss, getindex.(res, 2))
        push!.(Fss, getindex.(res, 3))
        push!.(Ess, getindex.(res, 4))
        map(cs, Dss, Fss, Ess) do c, Ds, Fs, Es
            if length(Es) > 2 && # Let the new (not first) DIIS space have 2+ samples
               Es[end] - Es[end-1] > resetThreshold
                keepIndex = lastindex(Es) - 1
                keepOnly!(c,   keepIndex)
                keepOnly!(Ds,  keepIndex)
                keepOnly!(Fs,  keepIndex)
                keepOnly!(Es,  keepIndex)
            else
                if length(c) > DIISsize
                    popIndex = argmax(Es)
                    popat!(c,   popIndex)
                    popat!(Ds,  popIndex)
                    popat!(Fs,  popIndex)
                    popat!(Es,  popIndex)
                end
            end
        end
        res
    end
    keyArgs = (:resetThreshold=>resetThreshold, :DIISsize=>DIISsize, :solver=>solver)
    f, keyArgs
end


# Unified input arguments: m, vars, S, X, Ns, Hcore, HeeI; kws...
genHFcore(::Type{VT}, vars::NTuple{HFTS, HFtempVars{T, HFT}}, args...; kws...) where 
         {VT<:Union{Val{:DIIS}, Val{:ADIIS}, Val{:EDIIS}}, HFTS, T, HFT} = 
genxDIIS(VT, vars, args...; kws...)

genHFcore(::Type{Val{:DD}}, vars::NTuple{HFTS, HFtempVars{T, HFT}}, ::AbstractMatrix{T}, 
          args...; kws...) where {HFTS, T, HFT} = 
genDD(vars, args...; kws...)


# Included normalization condition, but not non-negative condition.
@inline function genxDIISf(v, B, shift)
    function (c)
        s, _ = shiftLastEle!(c, shift)
        res = dot(v, c) / s + dot(c, B, c) / (2s^2)
        res
    end
end

@inline function genxDIIS∇f(v, B, shift)
    function (g, c)
        s, _ = shiftLastEle!(c, shift)
        g.= v./s + (B + transpose(B))*c ./ (2s^2) .- (dot(v, c)/s^2 + dot(c, B, c)/s^3)
        g
    end
end


# Default method
function LBFGSBsolver!(::Val{CCB}, c::AbstractVector{T}, 
                       v::AbstractVector{T}, B::AbstractMatrix{T}) where {CCB, T}
    shift = getAtolVal(T)
    f = genxDIISf(v, B, shift)
    g! = genxDIIS∇f(v, B, shift)
    lb = ifelse(CCB, T(0), T(-Inf))
    oldstd = stdout
    redirect_stdout(devnull)
    c .= lbfgsb(f, g!, c; lb, m=min(getAtolDigits(T), 50), 
                factr=1e5, pgtol=exp10(-getAtolDigits(T)), 
                iprint=-1, maxfun=10000, maxiter=10000)[end]
    redirect_stdout(oldstd)
    s, _ = shiftLastEle!(c, shift)
    c ./= s
end

function SPGBsolver!(::Val{CCB}, c::AbstractVector{T}, 
                     v::AbstractVector{T}, B::AbstractMatrix{T}) where {CCB, T}
    shift = getAtolVal(T)
    f = genxDIISf(v, B, shift)
    g! = genxDIIS∇f(v, B, shift)
    lb = ifelse(CCB, T(0), T(-Inf))
    vL = length(v)
    spgbox!(f, g!, c, lower=fill(lb, vL), 
            eps=exp10(-getAtolDigits(T)), nitmax=10000, nfevalmax=10000, m=20)
    s, _ = shiftLastEle!(c, shift)
    c ./= s
end

function CMsolver!(::Val{CCB}, c::AbstractVector{T}, 
                   v::AbstractVector{T}, B::AbstractMatrix{T}, 
                   ϵ::T=T(1000getAtolVal(T))) where {CCB, T}
    len = length(v)
    getA = M->[M  ones(T, len); ones(T, 1, len) T(0)]
    b = vcat(-v, 1)
    while true
        A = getA(B)
        while det(A) == 0
            B += ϵ*I
            A = getA(B)
        end
        c .= @view (A \ b)[begin:end-1]
        (CCB && findfirst(x->x<0, c) !== nothing) || (return c)
        idx = powerset(sortperm(abs.(c)), 1)

        for is in idx
            Atemp = @view A[begin:end .∉ Ref(is), begin:end .∉ Ref(is)]
            det(Atemp) == 0 && continue
            btemp = @view b[begin:end .∉ Ref(is)]
            cL = Atemp \ btemp
            popat!(cL, lastindex(cL))
            for i in sort(is)
                insert!(cL, i, 0.0)
            end
            c .= cL
            (findfirst(x->x<0, c) !== nothing) || (return c)
        end

        B += ϵ*I
    end
    c
end


const ConstraintSolvers = (LCM=CMsolver!, LBFGS=LBFGSBsolver!, SPGB=SPGBsolver!)

constraintSolver!(::Val{CCB}, 
                  c::AbstractVector{T}, v::AbstractVector{T}, B::AbstractMatrix{T}, 
                  solver::Symbol) where {T, CCB} = 
getproperty(ConstraintSolvers, solver)(Val(CCB), c, v, B)