export SCFconfig, HFconfig, runHartreeFock, RCHartreeFock, UOHartreeFock
public HFfinalInfo
#!! Need to standardize the format for global values, types (aliases), and type parameters
using LinearAlgebra: dot, Hermitian, \, det, I, ishermitian, diag, norm
using Base: OneTo
using Combinatorics: powerset
using SPGBox: spgbox!
using LBFGSB: lbfgsb

const CONSTVAR_defaultHFinfoDigits = 10
const FullHFStepLevel = 5
const defaultDS = 0.75
const defaultDIISsize = 15
const defaultDIISsolver = :LBFGS
const SADHFmaxStep = 50
const defaultHFinfoLevel = 2
const defaultHFmaxStep = 200
const defaultHFsaveTrace = (false, false, false, true) # C, D, F, E
const DEtotIndices = [2, 4]

const HFminItr = 10
const HFinterEsOscConvSize = 15
const HFinterMatStoreSizes = (2, 3, 2) # C(>1), D(>2), F(>1)
const defaultSCFconfigArgs = ((:DD, :ADIIS, :DIIS), (5e-3, 1e-4, 1e-9))
const defaultSecConvRatio = (1000, 1000)
const defaultOscThreshold = 5e-6

#> Reference(s):
#>> [ISBN-13] 978-0486691862
#>> [DOI] 10.1016/0009-2614(80)80396-4
#>> [DOI] 10.1002/jcc.540030413
#>> [DOI] 10.1063/1.1470195
#>> [DOI] 10.1063/1.3304922


function getOrthonormalization(::Val{:Symmetric}, matOverlap::AbstractMatrix{T}) where 
                              {R<:Real, T<:RealOrComplex{R}}
    Hermitian(matOverlap)^(-R(1//2))
end

precompile(getOrthonormalization, (Matrix{Float64},))

function solveFockMatrix(X::AbstractMatrix{T}, Fˢ::AbstractMatrix{T}, 
                         stabilizeSign::Bool=true) where {R, T<:RealOrComplex{R}}
    eigenVals, matCx = eigen((X' * Fˢ * X) |> Hermitian)
    matC = X * matCx

    stabilizeSign && for j in axes(matC, 2) #> Stabilize the sign factor of each column
       matC[:, j] *= ifelse(matC[begin, j] < 0, -1, 1)
    end

    matC, eigenVals::AbstractVector{T}
end

getC(X::AbstractMatrix{T}, Fˢ::AbstractMatrix{T}, stabilizeSign::Bool=true) where {T} = 
(first∘solveFockMatrix)(X, Fˢ, stabilizeSign)


struct RCHartreeFock <: SymbolType end #> Restricted closed-shell Hartree–Fock
struct UOHartreeFock <: SymbolType end #> Unrestricted open-shell Hartree–Fock
const CONSTVAR_HartreeFockTypes = (RCHartreeFock, UOHartreeFock)
const HartreeFockType = Union{CONSTVAR_HartreeFockTypes...}

getHartreeFockName(::RCHartreeFock) = "Restricted closed-shell (RHF)"
getHartreeFockName(::UOHartreeFock) = "Unrestricted open-shell (UHF)"


function breakCoeffSymmetry(::UOHartreeFock, spinSec1OrbCoeff::AbstractMatrix{T}) where {T}
    spinSec2OrbCoeff = copy(spinSec1OrbCoeff)
    l = min(size(spinSec2OrbCoeff, 1), 2)
    spinSec2OrbCoeff[begin:begin+l-1, begin:begin+l-1] .= 0 #> Breaking spin symmetry
    #> spinSec2OrbCoeff[l, :] .= 0 # Alternative solution
    (spinSec1OrbCoeff, spinSec2OrbCoeff)
end

breakCoeffSymmetry(::RCHartreeFock, spinSec1OrbCoeff::AbstractMatrix{T}) where {T} = 
(spinSec1OrbCoeff,)

function breakCoeffSymmetry(::RCHartreeFock, matOrth::AbstractMatrix{T}, 
                            matOneBody::AbstractMatrix{T}, matTwoBody::AbstractArray{T, 4}, 
                            densityPair::NTuple{2, AbstractMatrix{T}}) where 
                           {T<:RealOrComplex}
    spinSec1Density = sum(densityPair) ./ 2
    getC.( Ref(matOrth), getF(matOneBody, matTwoBody, (spinSec1Density,)) )
end

function breakCoeffSymmetry(::UOHartreeFock, matOrth::AbstractMatrix{T}, 
                            matOneBody::AbstractMatrix{T}, matTwoBody::AbstractArray{T, 4}, 
                            densityPair::NTuple{2, AbstractMatrix{T}}) where 
                           {T<:RealOrComplex}
    getC.( Ref(matOrth), getF(matOneBody, matTwoBody, densityPair) )
end


struct ElecHamiltonianConfig{T<:RealOrComplex, A1<:AbstractMatrix{T}, 
                             A2<:AbstractMatrix{T}, A3<:AbstractMatrix{T}, 
                             A4<:AbstractArray{T, 4}}
    spin::OccupationState{2}
    orthonormalization::A1
    overlap::A2
    oneBody::A3
    twoBody::A4
end


const CONSTVAR_OrbCoeffInitializationMethods::NTuple{4, Val} = 
      (Val(:Direct), Val(:CoreH), Val(:GWH), Val(:SAD))

struct OrbCoeffInitialConfig{T<:RealOrComplex, HFT<:HartreeFockType, M, 
                             P<:N12Tuple{ AbstractMatrix{T} }} <: ConfigBox
    method::Val{M}
    data::P

    function OrbCoeffInitialConfig(::HFT, ::Type{T}, ::Val{M}=Val(:SAD)) where 
                                  {HFT<:HartreeFockType, M, T<:RealOrComplex}

        new{T, HFT, M, Tuple{ MatrixMemory{T} }}(Val(M), (ShapedMemory{T}(undef, (0, 0)),))
    end

    OrbCoeffInitialConfig(::RCHartreeFock, data::NTuple{1, AbstractMatrix{T}}) where {T} = 
    new{T, RCHartreeFock, :Direct, typeof(data)}(Val(:Direct), data)

    OrbCoeffInitialConfig(::UOHartreeFock, data::NTuple{2, AbstractMatrix{T}}) where {T} = 
    new{T, UOHartreeFock, :Direct, typeof(data)}(Val(:Direct), data)
end

const ElectronicSysConfig{R<:Real, D} = 
      @NamedTuple{spin::OccupationState{2}, geometry::NuclearCluster{R, D}}

function initializeHartreeFock(coeffConfig::OrbCoeffInitialConfig{T, HFT, M}, 
                               basisData::MultiOrbitalData{R, D, T}, 
                               systemInfo::ElectronicSysConfig{R, D}) where 
                              {R<:Real, T<:RealOrComplex{R}, HFT<:HartreeFockType, M, D}
    spinInfo, nucInfo = systemInfo
    style1B = OneBodyIntegral{D, T}()
    style2B = TwoBodyIntegral{D, T}()
    normInfo, ovlp = computeOrbDataIntegral(style1B, genOverlapSampler(), basisData)
    coreH = evalOrbIntegralInfo!(genCoreHamiltonianSampler(nucInfo), normInfo)
    eriOp = genCoulombInteractionSampler(T, Count(D))
    eriBasis = basisData #!> `normInfo.basis` will crash Julia as of 1.11.6
    _, eriH = computeOrbDataIntegral(style2B, eriOp, eriBasis)
    matOrth = getOrthonormalization(Val(:Symmetric), ovlp)
    sysConfig = ElecHamiltonianConfig(spinInfo, matOrth, ovlp, coreH, eriH)
    matInitCoeff = if coeffConfig.method isa Val{:Direct}
        coeffData = coeffConfig.data
        cSizeData = size.(coeffData)
        nBasis = length(basisData.format)
        if !all(all(cSize .== nBasis) for cSize in cSizeData)
            throw(DimensionMismatch("The size/sizes of the input initial orbital"* 
                                    "coefficient matrix/matrices's size ($cSizeData)"*
                                    " does not match the basis-set size ($nBasis)."))
        end
        coeffData
    else
        initializeOrbCoeffData(Val(M), HFT(), (nucInfo, sysConfig, normInfo))
    end
    sysConfig, matInitCoeff
end

const FullElecHamilInfo{R<:Real, D, T<:RealOrComplex{R}} = 
      Tuple{NuclearCluster{R, D}, ElecHamiltonianConfig{T}, OrbitalOverlapInfo{R, D, T}}

function initializeOrbCoeffData(::Val{:CoreH}, ::HFT, info::FullElecHamilInfo) where 
                               {HFT<:HartreeFockType}
    _, part2, _ = info
    initializeOrbCoeffData(Val(:CoreH), HFT(), part2.orthonormalization, part2.oneBody)
end

function initializeOrbCoeffData(::Val{:CoreH}, ::HFT, matOrth::AbstractMatrix{T}, 
                                matOneBody::AbstractMatrix{T}) where 
                               {HFT<:HartreeFockType, T<:RealOrComplex}
    spinSec1OrbCoeff = getC(matOrth, matOneBody)
    breakCoeffSymmetry(HFT(), spinSec1OrbCoeff)
end

function initializeOrbCoeffData(::Val{:GWH}, ::HFT, info::FullElecHamilInfo) where 
                               {HFT<:HartreeFockType}
    _, part2, _ = info
    initializeOrbCoeffData(Val(:GWH), HFT(), part2.orthonormalization, part2.overlap, 
                           part2.oneBody)
end

function initializeOrbCoeffData(::Val{:GWH}, ::HFT, matOrth::AbstractMatrix{T}, 
                                matOverlap::AbstractMatrix{T}, 
                                matOneBody::AbstractMatrix{T}) where 
                               {HFT<:HartreeFockType, T}
    nBasis = size(matOneBody, 1)
    newOneBody = similar(matOneBody)
    offset1 = firstindex(matOverlap, 1) - 1
    offset2 = firstindex(newOneBody, 1) - 1

    Threads.@threads for k in (OneTo∘symmetric2DArrEleNum)(nBasis)
        i, j = convertIndex1DtoTri2D(k)
        res = 3 * matOverlap[i+offset1, j+offset1] * 
                 (matOneBody[i+offset2, i+offset2] + matOneBody[j+offset2, j+offset2]) / 8
        newOneBody[i+offset2, j+offset2] = res
        newOneBody[j+offset2, i+offset2] = conj(res)
    end

    spinSec1OrbCoeff = getC(matOrth, newOneBody)
    breakCoeffSymmetry(HFT(), spinSec1OrbCoeff)
end

function initializeOrbCoeffData(::Val{:SAD}, ::HFT, info::FullElecHamilInfo) where 
                               {HFT<:HartreeFockType}
    part1, part2, part3 = info
    initializeOrbCoeffData(Val(:SAD), HFT(), part1, part2, part3)
end

function getDefaultSCFconfigForSAD(::Type{T}) where {T<:Real}
    SCFconfig( (:ADIIS,), (max(1e-2, 10getAtolVal(T)),) )
end

function initializeOrbCoeffData(::Val{:SAD}, ::HFT, nucInfo::NuclearCluster{R, D}, 
                                sysConfig::ElecHamiltonianConfig{T}, 
                                normInfo::OrbitalOverlapInfo{R, D, T}, 
                                atmSCFconfig=getDefaultSCFconfigForSAD(R)) where 
                               {HFT<:HartreeFockType, R<:Real, T<:RealOrComplex{R}, D}
    nAtm = length(nucInfo)
    matOrth = sysConfig.orthonormalization
    matOverlap = sysConfig.overlap
    sysOneBody = sysConfig.oneBody
    sysTwoBody = sysConfig.twoBody
    spinSec1Den, spinSec2Den = oneBodyDensityPair = (zero(matOverlap), zero(matOverlap))
    atmHFtype = UOHartreeFock()

    for (nuc, coord) in nucInfo
        op = genCoreHamiltonianSampler([nuc], [coord])
        atmOneBody = evalOrbIntegralInfo!(op, normInfo)
        atmSysConfig = ElecHamiltonianConfig(sysConfig.spin, matOrth, matOverlap, 
                                             atmOneBody, sysTwoBody)
        orbCoeff = initializeOrbCoeffData(Val(:CoreH), atmHFtype, matOverlap, atmOneBody)
        atmRes, _ = runHartreeFockCore(atmHFtype=>atmSCFconfig, atmSysConfig, orbCoeff, 
                                       SADHFmaxStep, true, (false, false, false, false))
        spinSec1Res, spinSec2Res = atmRes
        spinSec1Den .+= spinSec1Res.Ds[end]
        spinSec2Den .+= spinSec2Res.Ds[end]
    end

    breakCoeffSymmetry(HFT(), matOrth, sysOneBody, sysTwoBody, oneBodyDensityPair./nAtm)
end


function getD(Cˢ::AbstractMatrix{T}, Nˢ::Int) where {T}
    iBegin = firstindex(Cˢ, 1)
    @views (Cˢ[:, iBegin:(iBegin+Nˢ-1)]*Cˢ[:, iBegin:(iBegin+Nˢ-1)]')
end #> Nˢ: number of electrons with the same spin

getD(X::AbstractMatrix{T}, Fˢ::AbstractMatrix{T}, Nˢ::Int) where {T} = 
getD(getC(X, Fˢ), Nˢ)


function getGcore(HeeI::AbstractArray{T, 4}, DJ::AbstractMatrix{T}, 
                  DK::AbstractMatrix{T}) where {T}
    G = similar(DJ)
    Δi1 = firstindex(HeeI, 1) - 1
    Δi2 = firstindex(DJ, 1) - 1
    len = size(G, 1)
    Threads.@threads for k in (OneTo∘symmetric2DArrEleNum)(len)
        μ, ν = convertIndex1DtoTri2D(k)
        res = dot(transpose(DJ), @view HeeI[μ+Δi1,ν+Δi1,:,:]) - 
              dot(          DK,  @view HeeI[μ+Δi1,:,:,ν+Δi1])
        G[μ+Δi2, ν+Δi2] = res
        G[ν+Δi2, μ+Δi2] = conj(res)
    end
    G
end

# RHF
getG(HeeI::AbstractArray{T, 4}, (Dˢ,)::Tuple{AbstractMatrix{T}}) where {T} = 
( getGcore(HeeI, 2Dˢ, Dˢ), )

# UHF
getG(HeeI::AbstractArray{T, 4}, (Dᵅ, Dᵝ)::NTuple{2, AbstractMatrix{T}}) where {T} = 
( getGcore(HeeI, Dᵅ+Dᵝ, Dᵅ), getGcore(HeeI, Dᵅ+Dᵝ, Dᵝ) )


getF(Hcore::AbstractMatrix{T}, G::N12Tuple{AbstractMatrix{T}}) where {T} = 
Ref(Hcore) .+ G

getF(Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
     D::N12Tuple{AbstractMatrix{T}}) where {T} = 
getF(Hcore, getG(HeeI, D))


# RHF or UHF
function getE(Hcore::AbstractMatrix{T}, Fˢ::AbstractMatrix{T}, 
              Dˢ::AbstractMatrix{T}) where {T}
    real(dot(Dˢ, Hcore+Fˢ)) / 2
end

get2SpinQuantity(O::NonEmptyTuple{T, N}) where {T, N} = abs(2-N) * sum(O)
get2SpinQuantities(O, nRepeat::Int) = ntuple(_->get2SpinQuantity(O), nRepeat)

# RHF or UHF
getEhfCore(Hcore::AbstractMatrix{T}, Fˢ::NonEmptyTuple{AbstractMatrix{T}, N}, 
           Dˢ::NonEmptyTuple{AbstractMatrix{T}, N}) where {T, N} = 
get2SpinQuantity(getE.(Ref(Hcore), Fˢ, Dˢ))

# RHF or UHF
function getEhf(Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
                C::NonEmptyTuple{AbstractMatrix{T}, N}, 
                Ns::NonEmptyTuple{Int, N}) where {T, N}
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


# UHF for MO
function getEhf(HcoreMOs::NTuple{2, AbstractMatrix{T}}, 
                HeeIMOs::NTuple{2, AbstractArray{T, 4}}, 
                Jᵅᵝ::AbstractMatrix{T}, Ns::NTuple{2, Int}) where {T}
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
                 Ns::NonEmptyTuple{Int, N}, F::NonEmptyTuple{AbstractMatrix{T}, N}) where 
                {T, N}
    Cnew = getC.(Ref(X), F)
    Dnew = getD.(Cnew, Ns)
    Fnew = getF(Hcore, HeeI, Dnew)
    Enew = getE.(Ref(Hcore), Fnew, Dnew)
    HFTS = N+1
    Dᵗnew = get2SpinQuantities(Dnew, HFTS)
    Eᵗnew = get2SpinQuantities(Enew, HFTS)
    map(themselves, Cnew, Dnew, Fnew, Enew, Dᵗnew, Eᵗnew)
end


mutable struct HFinterrelatedVars{R<:Real, T<:RealOrComplex{R}} <: StateBox{T}
    Dtots::Vector{Matrix{T}}
    Etots::Vector{R}

    function HFinterrelatedVars(Dts::Vector{Matrix{T}}, Ets::Vector{R}) where 
                               {R<:Real, T<:RealOrComplex{R}}
        new{R, T}(Dts, Ets)
    end

    HFinterrelatedVars{R, T}() where {R, T<:RealOrComplex{R}} = new{R, T}()
end


struct HFtempInfo{R<:Real, T<:RealOrComplex{R}, HFT<:HartreeFockType} <: StateBox{T}
    N::Int
    Cs::Vector{Matrix{T}} #! Change container from `Vector` to `LRU`
    Ds::Vector{Matrix{T}}
    Fs::Vector{Matrix{T}}
    Es::Vector{R}
    shared::HFinterrelatedVars{R, T}

    HFtempInfo(::HFT, Nˢ::Int, C::Matrix{T}, D::Matrix{T}, F::Matrix{T}, E::R) where 
              {HFT<:HartreeFockType, R<:Real, T<:RealOrComplex{R}} = 
    new{R, T, HFT}(Nˢ, [C], [D], [F], [E], HFinterrelatedVars{R, T}())

    HFtempInfo(::HFT, Nˢ::Int, Cs::Vector{Matrix{T}}, Ds::Vector{Matrix{T}}, 
               Fs::Vector{Matrix{T}}, Es::Vector{R}, Dtots::Vector{Matrix{T}}, 
               Etots::Vector{R}) where {HFT<:HartreeFockType, R, T<:RealOrComplex{R}} = 
    new{R, T, HFT}(Nˢ, Cs, Ds, Fs, Es, HFinterrelatedVars(Dtots, Etots))
end

const HFTVVfields = (:Cs, :Ds, :Fs, :Es)

function getHFTVforUpdate1(tVars::HFtempInfo)
    getproperty.(Ref(tVars), HFTVVfields)
end

const HFIVfields = (:Dtots, :Etots)

function getHFTVforUpdate2(tVars::HFtempInfo)
    getproperty.(Ref(tVars.shared), HFIVfields)
end

function updateHFTVcore!(varMaxLen::Int, var::Vector{T}, res::T, 
                         truncateMemory::Bool) where {T}
    while truncateMemory && length(var) >= varMaxLen
        popfirst!(var)
    end
    push!(var, res)
end

const HFtempInfoCore{R, T<:RealOrComplex{R}} = Tuple{
    AbstractMatrix{T}, AbstractMatrix{T}, AbstractMatrix{T}, R, #> .Cs, .Ds, .Fs, .Es
    AbstractMatrix{T}, R                                        #> :Dtots, :Etots
}

function updateHFtempInfo!(maxLen::Int, αβVars::NonEmptyTuple{HFtempInfo{R, T, HFT}, N}, 
                           ress::NonEmptyTuple{HFtempInfoCore{R, T}, N}) where 
                          {R, T<:RealOrComplex{R}, HFT, N}
    for (tVars, res) in zip(αβVars, ress)
        fs = getHFTVforUpdate1(tVars)
        for (f, r, truncateBl) in zip(fs, res, 
                                      (true, true, true, false)) #> (:Cs, :Ds, :Fs, :Es)
            updateHFTVcore!(maxLen, f, r, truncateBl)
        end
    end
    for (f, r, truncateBl) in zip(getHFTVforUpdate2(αβVars[begin]), ress[begin][end-1:end], 
                                  (true, false)) #> (Dtots, Etots)
        updateHFTVcore!(maxLen, f, r, truncateBl)
    end
end

function popHFtempInfo!(αβVars::N12Tuple{HFtempInfo{R, T, HFT}}, counts::Int=1) where 
                       {R<:Real, T<:RealOrComplex{R}, HFT<:HartreeFockType}
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

function clearHFtempInfo!(saveTrace::NTuple{4, Bool}, 
                          αβVars::N12Tuple{HFtempInfo{R, T, HFT}}
                          ) where {R<:Real, T<:RealOrComplex{R}, HFT<:HartreeFockType}
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


function formatSpinConfiguration(::RCHartreeFock, state::OccupationState{2})
    spinSec1Num, spinSec2Num = state
    if spinSec1Num != spinSec2Num
        throw(AssertionError("Close-shell system must have the same number of spins for "*
                             "both spin orientations."))
    end
    (spinSec1Num,)
end

function formatSpinConfiguration(::UOHartreeFock, state::OccupationState{2})
    spinSec1Num, spinSec2Num = state
    (spinSec1Num, spinSec2Num)
end


function initializeSCF(::HFT, config::ElecHamiltonianConfig{T}, 
                       C::N12Tuple{AbstractMatrix{T}}) where 
                      {HFT<:HartreeFockType, R<:Real, T<:RealOrComplex{R}}
    nSector = length(C)
    Hcore = config.oneBody
    HeeI = config.twoBody
    spinConfig = formatSpinConfiguration(HFT(), config.spin)::NTuple{nSector, Int}
    D = getD.(C, spinConfig)
    F = getF(Hcore, HeeI, D)
    E = getE.(Ref(Hcore), F, D)
    res = HFtempInfo.(HFT(), spinConfig, C, D, F, E)
    sharedFields = getproperty.(res, :shared)
    for (field, val) in zip(HFIVfields, fill.(get2SpinQuantity.((D, E)), 1))
        setproperty!.(sharedFields, field, Ref(val))
    end
    res::NTuple{nSector, HFtempInfo{R, T, HFT}}
end


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

const Doc_SCFconfig_SecConv1 = "root mean square of the error matrix defined in DIIS"

const Doc_SCFconfig_SecConv2 = "root mean square of the change of the density matrix"

const Doc_SCFconfig_eg1 = "SCFconfig{Float64, 2, Tuple{Val{:ADIIS}, Val{:DIIS}}}(method, "*
                          "interval=(0.001, 1.0e-8), methodConfig, secondaryConvRatio, "*
                          "oscillateThreshold)"

const SCFKeywordArgDict = AbstractDict{Int, <:AbstractVector{ <:Pair{Symbol} }}

"""

    SCFconfig{T<:Real, L, MS<:NTuple{L, Val}} <: $ConfigBox

The conatiner to configure the self-consistent field (SCF) iteration strategy for 
Hartree–Fock computation.

≡≡≡ Property/Properties ≡≡≡

`method::MS`: The applied convergence methods. The parameters `S::Symbol` in each `Val{S}` 
specifies a SCF ietration method. The available methods their corresponding supported 
keyword arguments are:

| Convergence Method(s) | Configuration(s) | Keyword(s) | Range(s)/Option(s) | Default(s) |
| :----                 | :---:            | :---:      | :---:              |      ----: |
| `:DD`                 | damping strength |`dampStrength`|    [`0`, `1`]  |`$(defaultDS)`|
$(Doc_SCFconfig_OneRowTable)

¹ The reset threshold (`resetThreshold::Real`) determines when to clear the memory of the 
DIIS-based method's subspace and reset the second-to-latest residual vector as the first 
reference. The reset is executed when the latest computed energy increases an amount above 
the threshold compared to the second-to-latest computed energy. In default, the threshold 
is slightly larger than the machine epsilon of the numerical data type applied to the SCF 
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

`methodConfig::NTuple{L, Memory{ <:Pair{Symbol} }}`: The additional keywords arguments for 
each method stored as `Tuple`s of `Pair`s.

`secondaryConvRatio::NTuple{2, T}`: The ratios of the secondary convergence criteria to the 
primary convergence indicator, i.e., the change of the energy (ΔE):

| Order |Symbols       | Meaning                   | Default                   |
| :---- | :---:        | :---:                     | :---:                     |
| 1     | RMS(FDS-SDF) | $(Doc_SCFconfig_SecConv1) | $(defaultSecConvRatio[1]) |
| 2     | RMS(ΔD)      | $(Doc_SCFconfig_SecConv2) | $(defaultSecConvRatio[2]) |

`oscillateThreshold::T`: The threshold for oscillatory convergence.

≡≡≡ Initialization Method(s) ≡≡≡

    SCFconfig(methods::NTuple{L, Symbol}, intervals::NTuple{L, T}, 
              config::$SCFKeywordArgDict=$EmptyDict{Int, Vector{ Pair{Symbol} }}(); 
              secondaryConvRatio::Union{Real, NTuple{2, Real}}=
              T.($defaultSecConvRatio), oscillateThreshold::Real=
              T($defaultOscThreshold)) where {L, T<:AbstractFloat} -> 
    SCFconfig{T, L}

`methods` and `intervals` are the convergence methods to be applied and their stopping 
(or skipping) thresholds respectively. `config` specifies additional keyword argument(s) 
for each methods by a `Pair{Symbol}` of which the key `i::Int` is for `i`th method and the 
pointed `AbstractVector{<:Pair}` is the pairs of keyword arguments and their values 
respectively. If `secondaryConvRatio` is `AbstractFloat`, it will be assigned as the value 
for all the secondary convergence ratios.

    SCFconfig(;threshold::AbstractFloat=$(defaultSCFconfigArgs[end][end]), 
              secondaryConvRatio::Union{Real, NTuple{2, Real}}=$defaultSecConvRatio, 
              oscillateThreshold::Real=$defaultOscThreshold) -> 
    SCFconfig{$(defaultSCFconfigArgs[end]|>eltype), $(defaultSCFconfigArgs[begin]|>length)}

`threshold` will replace the stopping threshold of the default SCF configuration with a new 
value.

≡≡≡ Example(s) ≡≡≡
```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> SCFconfig((:DD, :ADIIS, :DIIS), (1e-4, 1e-12, 1e-13), Dict(2=>[:solver=>:LCM]));

julia> SCFconfig(threshold=1e-8, oscillateThreshold=1e-5)
$(Doc_SCFconfig_eg1)
```
"""
struct SCFconfig{T<:Real, L, MS<:NTuple{L, Val}} <: ConfigBox
    method::MS
    interval::NTuple{L, T}
    methodConfig::NTuple{L, Memory{ <:Pair{Symbol} }}
    secondaryConvRatio::NTuple{2, T}
    oscillateThreshold::T

    function SCFconfig(methods::NonEmptyTuple{Symbol, L}, intervals::NonEmptyTuple{T, L}, 
                       config::SCFKeywordArgDict=EmptyDict{Int, Vector{ Pair{Symbol} }}(); 
                       secondaryConvRatio::Union{Real, NTuple{2, Real}}=
                       T.(defaultSecConvRatio), oscillateThreshold::Real=
                       T(defaultOscThreshold)) where {L, T<:AbstractFloat}
        for interval in intervals; checkPositivity(interval, true) end
        kwPairs = [Memory{Pair{Symbol}}(undef, 0) for _ in OneTo(L+1)]
        for i in keys(config)
            kwPairs[i] = genMemory(config[i])
        end
        methods = Val.(methods)
        secondaryConvRatio = (T(secondaryConvRatio[begin]), T(secondaryConvRatio[end]))
        new{T, L+1, typeof(methods)}(methods, intervals, Tuple(kwPairs), 
                                     secondaryConvRatio, T(oscillateThreshold))
    end
end

SCFconfig(;threshold::AbstractFloat=defaultSCFconfigArgs[end][end], 
          secondaryConvRatio::Union{Real, NTuple{2, Real}}=defaultSecConvRatio, 
          oscillateThreshold::Real=defaultOscThreshold) = 
SCFconfig( defaultSCFconfigArgs[begin], 
          (defaultSCFconfigArgs[end][begin:end-1]..., Float64(threshold)); 
           secondaryConvRatio, oscillateThreshold )

function getSCFcacheSizes(scfConfig::SCFconfig)
    map(scfConfig.method, scfConfig.methodConfig) do method, config
        if method == Val(:DD)
            1
        else
            idx = findfirst(x->isequal(:DIISsize, x.first), config)
            if idx === nothing
                defaultDIISsize
            else
                cacheSize = c[idx][end]
                checkPositivity(cacheSize)
                cacheSize
            end
        end
    end
end


function getOrbitalOccupations(::RCHartreeFock, X::AbstractMatrix{T}, (spinSec1N,)::Tuple{Int}, 
                               (spinSec1F,)::Tuple{AbstractMatrix{T}}) where 
                              {R<:Real, T<:RealOrComplex{R}}
    nMode = size(X, 1)
    energyHolder = Memory{R}(undef, nMode)
    spinHolder = Memory{NTuple{2, Bool}}(undef, nMode)
    orbEs = solveFockMatrix(X, spinSec1F) |> last
    counter = 0
    elecNum = 2spinSec1N

    for (i, j, energy) in zip(eachindex(energyHolder), eachindex(spinHolder), orbEs)
        counter += 2
        energyHolder[i] = energy
        spinHolder[j] = ifelse(counter > elecNum, (false, false), (true, true))
    end

    (MemoryPair(energyHolder, spinHolder),)
end

function getOrbitalOccupations(::UOHartreeFock, X::AbstractMatrix{T}, 
                               spinPairN::NTuple{2, Int}, 
                               spinPairF::NTuple{2, AbstractMatrix{T}}) where 
                              {R<:Real, T<:RealOrComplex{R}}
    nMode = size(X, 1)
    n = 0
    map(spinPairN, spinPairF) do spinSec1N, spinSec1F
        n += 1
        marker = ifelse(n==1, (true, false), (false, true))
        energyHolder = Memory{R}(undef, nMode)
        spinHolder = Memory{NTuple{2, Bool}}(undef, nMode)
        orbEs = solveFockMatrix(X, spinSec1F) |> last
        counter = 0

        for (i, j, energy) in zip(eachindex(energyHolder), eachindex(spinHolder), orbEs)
            counter += 1
            energyHolder[i] = energy
            spinHolder[j] = ifelse(counter > spinSec1N, (false, false), marker)
        end

        MemoryPair(energyHolder, spinHolder)
    end
end


"""

    HFfinalInfo{R<:Real, D, T<:$RealOrComplex{R}, HFT<:$HartreeFockType, HFTS, 
                B<:$MultiOrbitalData{R, D, T}} <: StateBox{T}

The container of the final values after a Hartree–Fock SCF procedure. `HFTS` specifies the 
number of distinct data sectors corresponding to the specified spin configurations. For 
restricted closed-shell Hartree–Fock (RHF), `HFTS` is `1`.

≡≡≡ Property/Properties ≡≡≡
`@NamedTuple{spin::$OccupationState{2}, geometry::$NuclearCluster{R, T}}`: The spin and 
nuclear-geometry configurations of the target system. `.spin` specifies numbers of 
electrons in two orthonormal spin configurations (e.g., spin-up vs. spin-down). For any 
property data (`::P`) enclosed in `NTuple{HFTS, P}` and `HFTS==2`, each element correspond 
to one spin configuration.

`energy::NTuple{2, R}`: The electronic and nuclear (repulsion potential) parts of the 
target system's ground-state energy under the Hartree–Fock and the Born–Oppenheimer 
approximation.

`coeff::NTuple{HFTS, $MatrixMemory{T}}`: Distinct orbital coefficient matrix/matrices.

`density::NTuple{HFTS, $MatrixMemory{T}}`: Distinct one-electron density matrix/matrices.

`fock::NTuple{HFTS, $MatrixMemory{T}}`: Distinct Fock matrix/matrices.

`occu::NTuple{HFTS, $MemoryPair{ R, NTuple{2, Bool} }}`: The spin occupations of distinct 
canonical (spatial) orbitals and their corresponding orbital energies.

`memory::NTuple{HFTS, $HFtempInfo{R, T, HFT}}`: the intermediate data stored during the 
Hartree–Fock SCF (self-consistent field) interactions. (**NOTE:** The interface of 
`$HFtempInfo` is internal and subject to **BREAKING CHANGES**.)

`converged::Union{Bool, Missing}`: Whether the SCF iteration is converged in the end. 
When convergence detection is off (see [`SCFconfig`](@ref)), it is set to `missing`.

`basis::B`: The orbital basis-set data used for the Hartree–Fock SCF computation.
"""
struct HFfinalInfo{R<:Real, D, T<:RealOrComplex{R}, HFT<:HartreeFockType, HFTS, 
                   B<:MultiOrbitalData{R, D, T}} <: StateBox{T}
    system::ElectronicSysConfig{R, D}
    energy::NTuple{2, R}
    coeff::NTuple{HFTS, MatrixMemory{T}}
    density::NTuple{HFTS, MatrixMemory{T}}
    fock::NTuple{HFTS, MatrixMemory{T}}
    occu::NTuple{HFTS, MemoryPair{ R, NTuple{2, Bool} }}
    memory::NTuple{HFTS, HFtempInfo{R, T, HFT}}
    converged::Union{Bool, Missing}
    basis::B

    function HFfinalInfo(vars::N12Tuple{HFtempInfo{R, T, HFT}}, 
                         systemInfo::ElectronicSysConfig, 
                         basisData::MultiOrbitalData{R, D, T}, 
                         matOrth::AbstractMatrix{T}, 
                         converged::Union{Bool, Missing}) where 
                        {R<:Real, T<:RealOrComplex{R}, HFT<:HartreeFockType, D}
        enE = vars[begin].shared.Etots[end]
        nnE = nucRepulsion(systemInfo.geometry)
        sectorNum = getproperty.(vars, :N)
        matC = (ShapedMemory∘last).(getproperty.(vars, :Cs))
        matD = (ShapedMemory∘last).(getproperty.(vars, :Ds))
        matF = (ShapedMemory∘last).(getproperty.(vars, :Fs))
        occu = getOrbitalOccupations(HFT(), matOrth, sectorNum, matF)
        new{R, D, T, HFT, length(vars), typeof(basisData)}(
            systemInfo, (enE, nnE), matC, matD, matF, occu, vars, converged, basisData)
    end
end


"""

    HFconfig{R<:Real, T<:$RealOrComplex{R}, HFT<:$HartreeFockType, 
             CM<:$OrbCoeffInitialConfig{T, HFT}, SCFM<:$SCFconfig, S} <: $ConfigBox

The container of a Hartree–Fock method's configuration.

≡≡≡ Property/Properties ≡≡≡

`type::HFT`: Hartree–Fock method type. Available types are $CONSTVAR_HartreeFockTypes 
$(getHartreeFockName.( map(x->x(), CONSTVAR_HartreeFockTypes) )).

`initial::CM`: Initial guess of the orbital coefficient matrix/matrices of the canonical 
orbitals. When `initial` is as an argument of `HFconfig`'s constructor, it can be set 
to `sym::Symbol` where available values of `sym` are 
`$(CONSTVAR_OrbCoeffInitializationMethods.|>evalTypedData)`.

`strategy::SCFM`: SCF iteration strategy. For more information please refer to 
[`SCFconfig`](@ref).

`maxStep::Int`: Maximum iteration steps allowed regardless if the iteration converges.

`earlyStop::Bool`: Whether automatically terminate (or skip) a convergence method early 
when its performance becomes unstable or poor.

`saveTrace::NTuple{4, Bool}`: Determine whether saving (by pushing) the intermediate 
information from all the iterations steps to the field `.temp` of the output 
[`HFfinalInfo`](@ref) of `runHartreeFock`. The types of relevant information are:

| Sequence | Information | Corresponding field in `HFtempInfo` (subject to **CHANGES**) |
|  :---:   |    :---:    |                   :---:                     |
| 1 | orbital coefficient matrix/matrices      | `.Cs`                       |
| 2 | density matrix/matrices                  | `.Ds`, `.shared.Dtots`      |
| 3 | Fock matrix/matrices                     | `.Fs`                       |
| 4 | unconverged Hartree–Fock energy(s) | `.Es`, `.shared.Etots`      |

≡≡≡ Initialization Method(s) ≡≡≡

    HFconfig(::Type{T}, type::HFT=$RCHartreeFock(); 
             initial::Union{$OrbCoeffInitialConfig{T}, Symbol}=:SAD, 
             strategy::$SCFconfig=SCFconfig(), 
             maxStep::Int=$defaultHFmaxStep, earlyStop::Bool=true, 
             saveTrace::NTuple{4, Bool}=$defaultHFsaveTrace) where 
            {R, T<:RealOrComplex{R}, HFT<:$HartreeFockType} -> 
    HFconfig{R, T, HFT}

≡≡≡ Initialization Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> HFconfig(Float64, UOHartreeFock());
```
"""
mutable struct HFconfig{R<:Real, T<:RealOrComplex{R}, HFT<:HartreeFockType, 
                        CM<:OrbCoeffInitialConfig{T, HFT}, SCFM<:SCFconfig, S
                        } <: ConfigBox
    type::HFT
    initial::CM
    strategy::SCFM
    orthonormalization::Val{S} #! Need to be able to use it in the future
    maxStep::Int
    earlyStop::Bool
    saveTrace::NTuple{4, Bool}

    function HFconfig(::HFT, initial::CM, strategy::SCFM, ::Val{S}, 
                      maxStep::Int, earlyStop::Bool, saveTrace::NTuple{4, Bool}) where 
                     {R<:Real, T<:RealOrComplex{R}, HFT<:HartreeFockType, 
                      CM<:OrbCoeffInitialConfig{T, HFT}, SCFM<:SCFconfig, S}
        checkPositivity(maxStep)
        new{R, T, HFT, CM, SCFM, S}(HFT(), initial, strategy, Val(S), maxStep, earlyStop, 
                                    saveTrace)
    end
end

function HFconfig(::Type{T}, type::HFT=RCHartreeFock(); 
                  initial::Union{OrbCoeffInitialConfig{T}, Symbol}=:SAD, 
                  strategy::SCFconfig=SCFconfig(), orthonormalization::Symbol=:Symmetric, 
                  maxStep::Int=defaultHFmaxStep, earlyStop::Bool=true, 
                  saveTrace::NTuple{4, Bool}=defaultHFsaveTrace) where 
                 {T<:RealOrComplex, HFT<:HartreeFockType}
    if initial isa Symbol; initial = OrbCoeffInitialConfig(type, T, Val(initial)) end
    HFconfig(type, initial, strategy, Val(orthonormalization), maxStep, earlyStop, 
             saveTrace)
end


"""

    runHartreeFock(nucInfo::$NuclearCluster{R, D}, basis::$OrbBasisData{R, D}, 
                   config::MissingOr{$HFconfig{R}}=missing; 
                   printInfo::Bool=true, infoLevel::Int=$defaultHFinfoLevel) where 
                  {R<:Real, D} -> 
    HFfinalInfo{R, D}

    runHartreeFock(systemInfo::Pair{<:$ElectronSpinConfig, $NuclearCluster{R, D}}, 
                   basis::$OrbBasisData{R, D}, config::MissingOr{$HFconfig{R}}=missing; 
                   printInfo::Bool=true, infoLevel::Int=$defaultHFinfoLevel) where 
                  {R<:Real, D} -> 
    HFfinalInfo{R, D}

Main function to run a Hartree–Fock method in Quiqbox. The returned result and relevant 
information is stored in a [`Quiqbox.HFfinalVars`](@ref).

≡≡≡ Positional argument(s) ≡≡≡
`nucInfo::NuclearCluster{R, D}`: the nuclear geometry of the system. When `nucInfo` is the 
first argument the spin configuration is automatically set such that the target system is 
charge neutral and maximizing pairing electron spins.

`systemInfoPair{<:$ElectronSpinConfig, $NuclearCluster{R, D}}`: A `Pair` information used 
to specify both the spin and nuclear-geometry configuration of the target system.

`basis::$OrbBasisData{R, D}`: The orbital basis-set configuration.

`config::HFconfig`: The Configuration of the Hartree–Fock method. For more information 
please refer to [`HFconfig`](@ref).

≡≡≡ Keyword argument(s) ≡≡≡

`printInfo::Bool`: Whether print out the information of iteration steps and result.

`infoLevel::Int`: Printed info's level of details when `printInfo=true`. The higher 
(the absolute value of) `infoLevel` is, more intermediate steps will be printed. Once 
`infoLevel` achieve `$FullHFStepLevel`, every step will be printed.
"""
function runHartreeFock(nucInfo::NuclearCluster{R, D}, basis::OrbBasisData{R, D}, 
                        config::MissingOr{HFconfig{R}}=missing; 
                        printInfo::Bool=true, infoLevel::Int=defaultHFinfoLevel) where 
                       {R<:Real, D}
    spinInfo = prepareSpinConfiguration(nucInfo)
    runHartreeFock(spinInfo=>nucInfo, basis, config; printInfo, infoLevel)
end

const ElectronSpinConfig = Union{NTuple{2, Int}, OccupationState{2}}

function runHartreeFock(systemInfo::Pair{<:ElectronSpinConfig, NuclearCluster{R, D}}, 
                        basis::OrbBasisData{R, D}, config::MissingOr{HFconfig{R}}=missing; 
                        printInfo::Bool=true, infoLevel::Int=defaultHFinfoLevel) where 
                       {R<:Real, D}
    spinInfo, nucInfo = systemInfo
    spinInfo isa Tuple && (spinInfo = OccupationState(spinInfo))
    systemInfo = (spin=spinInfo, geometry=nucInfo)
    runHartreeFock(systemInfo, basis, config; printInfo, infoLevel)
end


function runHartreeFock(systemInfo::ElectronicSysConfig{R, D}, basis::OrbBasisData{R, D}, 
                        config::MissingOr{HFconfig{R}}=missing; 
                        printInfo::Bool=false, infoLevel::Int=defaultHFinfoLevel) where 
                       {R<:Real, D}
    basisData = basis isa AbstractVector ? MultiOrbitalData(basis, True()) : basis
    outType = getOutputType(basisData)
    spinInfo, nucInfo = systemInfo
    totalElecNum = getTotalOccupation(spinInfo)
    if ismissing(config)
        type = ifelse(iseven(totalElecNum), RCHartreeFock(), UOHartreeFock())
        config = HFconfig(outType, type)
    end
    HFtype = config.type
    timerBool = printInfo && infoLevel > 2
    timerBool && (tBegin = time_ns())
    minElecNum = Int(HFtype isa RCHartreeFock)
    if totalElecNum <= minElecNum
        throw(DomainError(N, "$(HFtype) requires more than $(minElecNum) electrons."))
    end
    leastNb = (Int∘maximum)(spinInfo.layout)
    nBasis = length(basisData.format)
    if nBasis < leastNb
        throw(DomainError(nBasis, "The basis-set size should be no less than $(leastNb)."))
    end
    initCoeffCOnfig = config.initial
    hamilConfig, initCoeff = initializeHartreeFock(initCoeffCOnfig, basisData, systemInfo)

    timerBool && (tEnd = time_ns())
    if printInfo && infoLevel > 3
        roundDigits = min(CONSTVAR_defaultHFinfoDigits, getAtolDigits(R))
        nucNum = length(nucInfo)
        println("System Information: ")
        println("•Number of Electrons: ", totalElecNum)
        println("•Number of Nuclei: ", nucNum)
        println("•Nuclear Coordinate: ")
        nucNumStrLen = ndigits(nucNum) + 3
        for (i, (atm, coord)) in zip(OneTo(nucNum), nucInfo)
            println(rpad(" $i)", nucNumStrLen), atm, ": ", "[", 
                    alignNumSign(coord[1]; roundDigits), ", ", 
                    alignNumSign(coord[2]; roundDigits), ", ", 
                    alignNumSign(coord[3]; roundDigits), "]")
        end
        println()
    end
    if printInfo && infoLevel > 1
        print("Hartree–Fock (HF) Initialization")
        timerBool && print(" (Finished in ", genTimeStr(tEnd - tBegin), ")")
        println(":")
        println("•HF Type: ", getHartreeFockName(HFtype))
        println("•Basis Set Size: ", nBasis)
        println("•Initial Guess Method: ", evalTypedData(initCoeffCOnfig.method))
    end

    configCore = Pair(HFtype, config.strategy)
    vars, converged = runHartreeFockCore(configCore, hamilConfig, initCoeff, 
                                         config.maxStep, config.earlyStop, 
                                         config.saveTrace, printInfo, infoLevel)
    matOrth = hamilConfig.orthonormalization
    resHolder = HFfinalInfo(vars, systemInfo, basisData, matOrth, converged)

    roundDigits = min(CONSTVAR_defaultHFinfoDigits, getAtolDigits(R))
    if printInfo
        eleE, nucE = resHolder.energy
        EhfStr  = alignNum(eleE, 0; roundDigits)
        EnnStr  = alignNum(nucE, 0; roundDigits)
        EtotStr = alignNum(sum(resHolder.energy), 0; roundDigits)
        len = max(19, length(EhfStr)+3, length(EnnStr)+3, length(EtotStr)+3)
        println(rpad("Hartree–Fock Energy", len), " ¦ ", rpad("Nuclear Repulsion", len), 
                " ¦ Total Energy")
        println(rpad(EhfStr*" Ha", len), "   ", rpad(EnnStr*" Ha", len), "   ", 
                EtotStr, " Ha\n")
    end

    resHolder
end


function runHartreeFockCore(configCore::Pair{HFT, <:SCFconfig{<:Real, L, MS}}, 
                            sysConfig::ElecHamiltonianConfig{T}, 
                            coeffData::N12Tuple{AbstractMatrix{T}}, 
                            maxStep::Int, 
                            earlyStop::Bool, 
                            saveTrace::NTuple{4, Bool}, 
                            printInfo::Bool=false, 
                            infoLevel::Int=defaultHFinfoLevel) where 
                           {HFT<:HartreeFockType, R<:Real, T<:RealOrComplex{R}, L, MS}
    scfConfig = configCore.second
    timerBool = printInfo && infoLevel > 2
    timerBool && (tBegin = time_ns())
    matOverlap = sysConfig.overlap

    vars = initializeSCF(HFT(), sysConfig, coeffData)
    secondaryConvRatio = scfConfig.secondaryConvRatio
    varsShared = vars[begin].shared
    Etots = varsShared.Etots
    ΔEs = zeros(R, 1)
    ΔDrms = zeros(T, 1)
    δFrms = R[getErrorNrms(vars, matOverlap)]
    ΔEendThreshold = scfConfig.interval[end]
    δFbreakPoints = secondaryConvRatio[begin] .* scfConfig.interval
    ΔDbreakPoints = secondaryConvRatio[begin+1] .* scfConfig.interval
    ΔEoscThresholds = max.(scfConfig.interval, scfConfig.oscillateThreshold)
    δFoscThresholds = secondaryConvRatio[begin] .* ΔEoscThresholds
    ΔDoscThresholds = secondaryConvRatio[begin+1] .* ΔEoscThresholds
    detectConvergence = !isnan(ΔEendThreshold)
    converged::Union{Bool, Missing, Int} = true
    rollbackRange = 0 : (HFminItr÷3)
    rollbackCount = length(rollbackRange)
    maxMemoryLens = map(getSCFcacheSizes(scfConfig)) do cacheSize
        max(HFinterMatStoreSizes..., cacheSize)
    end
    i = 0

    if printInfo
        roundDigits = setNumDigits(R, ΔEendThreshold)
        titles = ("Step", "E (Ha)", "ΔE (Ha)", "RMS(FDS-SDF)", "RMS(ΔD)")
        colPFs = (  lpad, cropStrR,  cropStrR,       cropStrR,  cropStrR)
        colSps = (max(ndigits(maxStep), (length∘string)(HFT), length(titles[begin])), 
                  roundDigits + (ndigits∘floor)(Int, Etots[]) + 2, 
                  roundDigits + 3)
        colSls = (1, 2, 3, 3, 3)
        titleStr = ""
        titleRng = 1 : (3 + 2*(infoLevel > 1))
        colSps = map(titles[titleRng], colPFs[titleRng], colSls[titleRng]) do title, f, idx
            printSpace = max(length(title), colSps[idx])
            titleStr *= "| " * f(title, printSpace) * " "
            printSpace
        end

        if infoLevel > 0
            adaptStepBl = genAdaptStepBl(infoLevel, maxStep)
            println("•Initial HF energy E: ", alignNum(Etots[], 0; roundDigits), " Ha")
            println("•Initial RMS(FDS-SDF): ", alignNum(δFrms[], 0; roundDigits))
            println("•Convergence Threshold of E: ", ΔEendThreshold, " Ha")
            println("•Convergence Threshold Ratios of (FDS-SDF, D) to E: ", 
                    secondaryConvRatio)
            if infoLevel > 2
                println("•Oscillatory Convergence Threshold: ", 
                        scfConfig.oscillateThreshold)
                println("•Maximum Number of Iterations Allowed: ", maxStep)
            end
            println()
            println("Self-Consistent Field (SCF) Iteration:")
            (println∘repeat)('=', length(titleStr))
            println(titleStr)
            (println∘replace)(titleStr, r"[^|]"=>'=')
        end
    end

    for (MType, kws, breakPoint, l, len) in zip(fieldtypes(MS), scfConfig.methodConfig, 
                                                scfConfig.interval, OneTo(L), maxMemoryLens)
        HFcore, keyArgs = initializeHFcore(MType(), vars, sysConfig; kws...)
        flucThreshold = max(10breakPoint, 1.5e-3) # ≈3.8kJ/mol (0.95 chemical accuracy)
        symHFcore = evalTypedData(MType)
        converged = false
        endM = l==L
        n = 0

        if printInfo && infoLevel > 1
            print('|', repeat('–', first(colSps)+1), "<$l>–", ("[:$symHFcore"))
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

            updateHFtempInfo!(len, vars, HFcore())

            push!(ΔEs, Etots[end] - Etots[end-1])
            if endM || printInfo
                push!(ΔDrms, rmsOf(varsShared.Dtots[end] - varsShared.Dtots[end-1]))
                push!(δFrms, getErrorNrms(vars, matOverlap))
            end
            ΔEᵢ = ΔEs[end]
            ΔDrmsᵢ = ΔDrms[end]
            δFrmsᵢ = δFrms[end]
            ΔEᵢabs = abs(ΔEᵢ)

            if printInfo && infoLevel > 0 && (adaptStepBl(i) || i == maxStep)
                print( "| ", colPFs[1]("$i", colSps[1]), 
                      " | ", colPFs[2](alignNumSign(Etots[end]; roundDigits), colSps[2]), 
                      " | ", colPFs[3](alignNumSign(ΔEᵢ; roundDigits), colSps[3]) )
                if infoLevel > 1
                    print( " | ", colPFs[4](alignNum(δFrmsᵢ, 0; roundDigits), colSps[4]), 
                           " | ", colPFs[5](alignNum(ΔDrmsᵢ, 0; roundDigits), colSps[5]) )
                end
                println()
            end

            convThresholds = (breakPoint, ΔDbreakPoints[l])
            δFrmsᵢ > δFbreakPoints[l] && (convThresholds = convThresholds .* 0)
            ΔEᵢabs <= convThresholds[begin] && ΔDrmsᵢ <= convThresholds[end] && 
            (converged = true; break)

            # oscillating convergence & early termination of non-convergence.
            if n > 1 && i > HFminItr && ΔEᵢ > flucThreshold
                isOsc = false
                if scfConfig.oscillateThreshold > 0
                    isOsc, _ = isOscillateConverged(Etots, 10ΔEoscThresholds[l], 
                                                    minLen=HFminItr, 
                                                    maxRemains=HFinterEsOscConvSize)
                    if isOsc && ΔEᵢabs <= ΔEoscThresholds[l] && 
                       (endM ? (δFrmsᵢ <= δFoscThresholds[l] && 
                                ΔDrmsᵢ <= ΔDoscThresholds[l]) : true)
                        converged = 1
                        break
                    end
                end
                if earlyStop && !isOsc
                    isRaising = all(rollbackRange) do j
                        ΔEs[end-j] > 10flucThreshold
                    end
                    if isRaising
                        memoryLen = length(first(vars).Cs)
                        rbCount = min(rollbackCount, max(memoryLen-1, 0))
                        terminateSCF!(vars, rbCount, symHFcore, printInfo)
                        i -= rbCount
                        break
                    end
                end
            end
        end
    end

    timerBool && (tEnd = time_ns())

    if printInfo
        tStr = timerBool ? " after "*genTimeStr(tEnd - tBegin) : ""
        negStr = if detectConvergence
            if converged===1
                converged=true
                "converged to an oscillation"
            else
                ifelse(converged, "converged", "stopped but not converged")
            end
        else
            "stopped"
        end
        println("\nThe SCF iteration of $HFT has ", negStr, " at step $i", tStr, ":\n", 
                "|ΔE| → ", alignNum(abs(ΔEs[end]), 0; roundDigits), " Ha, ", 
                "RMS(FDS-SDF) → ", alignNum(δFrms[end], 0; roundDigits), ", ", 
                "RMS(ΔD) → ", alignNum(ΔDrms[end], 0; roundDigits), ".\n")
    end
    clearHFtempInfo!(saveTrace, vars)
    detectConvergence || (converged = missing)
    vars, converged
end

function getErrorNrms(vars::N12Tuple{HFtempInfo{R, T, HFT}}, 
                      S::AbstractMatrix{T}) where {R<:Real, T<:RealOrComplex{R}, HFT}
    mapreduce(+, vars) do tVar
        D = tVar.Ds[end]
        F = tVar.Fs[end]
        (rmsOf∘getEresidual)(F, D, S)
    end / length(vars)
end

function terminateSCF!(vars, counts, method::Symbol, printInfo)
    popHFtempInfo!(vars, counts)
    printInfo && println("Early termination of ", method, " due to its poor performance.")
end


function directDiag(Nˢ::Int, X::AbstractMatrix{T}, F::AbstractMatrix{T}, 
                    D::AbstractMatrix{T}, dampStrength::T) where {T}
    0 <= dampStrength <= 1 || throw(DomainError(dampStrength, "The value of `dampStrength`"*
                                    " should be between 0 and 1."))
    Dnew = getD(X, F, Nˢ)
    (1 - dampStrength)*Dnew + dampStrength*D
end

function initializeHFcore(::Val{:DD}, αβVars::N12Tuple{HFtempInfo{R, T, HFT}}, 
                          config::ElecHamiltonianConfig{T}; 
                          dampStrength::R=R(defaultDS)) where 
                         {R<:Real, T<:RealOrComplex{R}, HFT<:HartreeFockType}
    spinInfoCore = formatSpinConfiguration(HFT(), config.spin)

    f = let Ns=spinInfoCore, X=config.orthonormalization, Hcore=config.oneBody, 
            HeeI=config.twoBody
        function DDcore()
            Fs = last.(getproperty.(αβVars, :Fs))
            Ds = last.(getproperty.(αβVars, :Ds))
            Dnew = directDiag.(Ns, Ref(X), Fs, Ds, dampStrength)
            getCDFE(Hcore, HeeI, X, Ns, getF(Hcore, HeeI, Dnew))
        end
    end
    keyArgs = (:dampStrength=>dampStrength,)
    f, keyArgs
end


function EDIIScore(Ds::Vector{<:AbstractMatrix{T}}, ∇s::Vector{<:AbstractMatrix{T}}, 
                   Es::Vector{T}) where {T}
    len = length(Ds)
    B = similar(∇s[begin], len, len)
    Δi = firstindex(B, 1) - 1
    Threads.@threads for k in (OneTo∘symmetric2DArrEleNum)(len)
        i, j = convertIndex1DtoTri2D(k)
        res = -dot(Ds[i]-Ds[j], ∇s[i]-∇s[j])
        @inbounds B[i+Δi, j+Δi] = res
        @inbounds B[j+Δi, i+Δi] = conj(res)
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
    Threads.@threads for k in (OneTo∘symmetric2DArrEleNum)(len)
        i, j = convertIndex1DtoTri2D(k)
        res = dot( X'*getEresidual(∇s[i], Ds[i], S)*X, 
                   X'*getEresidual(∇s[j], Ds[j], S)*X )
        @inbounds B[i+Δi, j+Δi] = res
        @inbounds B[j+Δi, i+Δi] = conj(res)
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

const DIIStype = Union{Val{:DIIS}, Val{:ADIIS}, Val{:EDIIS}}

function initializeHFcore(::M, αβVars::N12Tuple{HFtempInfo{R, T, HFT}}, 
                          config::ElecHamiltonianConfig{T}; 
                          resetThreshold::Real=1000getAtolVal(T), 
                          DIISsize::Int=defaultDIISsize, 
                          solver::Symbol=defaultDIISsolver) where 
                         {M<:DIIStype, R<:Real, T<:RealOrComplex{R}, HFT<:HartreeFockType}
    symDIIS = evalTypedData(M)
    DIISsize < 2 && (throw∘DomainError)(intervals, "$symDIIS space need to be at least 2.")
    DFEsyms = HFTVVfields[begin+1:end]
    DFElens = map(DFEsyms) do fieldSym
        getproperty(αβVars[begin], fieldSym) |> length
    end
    initialSize = min(DIISsize, DFElens...)
    cs = Tuple(collect(T, OneTo(initialSize)) for _ in OneTo(αβVars|>length))
    Dss, Fss, Ess = map(DFEsyms) do fieldSym
        fs = getproperty.(αβVars, fieldSym)
        iEnd = lastindex(fs[begin])
        getindex.(fs, Ref(iEnd-initialSize+1:iEnd))
    end
    cvxConstraint, mDIIS = getproperty(DIISconfigs, symDIIS)

    spinInfoCore = formatSpinConfiguration(HFT(), config.spin)

    f = let Ns=spinInfoCore, X=config.orthonormalization, S=config.overlap, 
            Hcore=config.oneBody, HeeI=config.twoBody

        function xDIIScore()
            Fn = xDIIScore!.(mDIIS, cs, Ref(S), Ref(X), Dss, Fss, Ess, cvxConstraint, 
                             solver)
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

    end

    keyArgs = (:resetThreshold=>resetThreshold, :DIISsize=>DIISsize, :solver=>solver)
    f, keyArgs
end


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
                   perturbation::T=T(1000getAtolVal(T))) where {CCB, T}
    len = length(v)
    getA = M->[M  ones(T, len); ones(T, 1, len) T(0)]
    b = vcat(-v, 1)
    while true
        A = getA(B)
        while det(A) == 0
            B += perturbation*I
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

        B += perturbation*I
    end
    c
end


const ConstraintSolvers = (LCM=CMsolver!, LBFGS=LBFGSBsolver!, SPGB=SPGBsolver!)

constraintSolver!(::Val{CCB}, 
                  c::AbstractVector{T}, v::AbstractVector{T}, B::AbstractMatrix{T}, 
                  solver::Symbol) where {T, CCB} = 
getproperty(ConstraintSolvers, solver)(Val(CCB), c, v, B)