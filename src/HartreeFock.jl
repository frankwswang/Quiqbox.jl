export SCFconfig, HFconfig, runHF, runHFcore

using LinearAlgebra: dot, Hermitian, \, det, I, ishermitian, diag
using Combinatorics: powerset
using LineSearches
using Optim: LBFGS, Fminbox, optimize as OptimOptimize, minimizer as OptimMinimizer, 
             Options as OptimOptions
using SPGBox: spgbox!

const defaultDS = 0.5
const defaultDIISconfig = (12, :LBFGS)
const SADHFmaxStep = 50
const defaultHFinfoL = 3

const HFOminCycle = 10
const defaultHFCStr = "HFconfig()"
const defaultSCFconfigArgs = ( (:ADIIS, :DIIS), (1e-3, 1e-12) )
const defultOscThreshold = 1e-6

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
    stabilizeSign && for j = 1:size(outC, 2)
       outC[:, j] *= ifelse(outC[1,j] < 0, -1, 1)
    end
    outC, ϵ
end

@inline getC(X::AbstractMatrix{T}, Fˢ::AbstractMatrix{T}, 
             stabilizeSign::Bool=true) where {T} = 
        getCϵ(X, Fˢ, stabilizeSign)[1]


splitSpins(::Val{1}, N::Int) = (N÷2,)

splitSpins(::Val{2}, N::Int) = (N÷2, N-N÷2)

splitSpins(::Val{N}, Ns::NTuple{N, Int}) where {N} = itself(Ns)

splitSpins(::Val{2}, (Nˢ,)::Tuple{Int}) = (Nˢ, Nˢ)

splitSpins(::Val{:RHF}, Ns::NTuple{2, Int}) = 
error("For restricted closed-shell Hartree-Fock (RHF), the input spin configuration $(Ns)"*
      " is not supported.")

splitSpins(::Val{:RHF}, N) = splitSpins(Val(HFtypeSizeList[:RHF]), N)

splitSpins(::Val{:UHF}, N) = splitSpins(Val(HFtypeSizeList[:UHF]), N)

function breakSymOfC(::Val{:UHF}, C::AbstractMatrix{T}) where {T}
    C2 = copy(C)
    l = min(size(C2, 1), 2)
    C2[1:l, 1:l] .= 0 # Breaking spin symmetry.
    # C2[l, :] .= 0 # Another way.
    (copy(C), C2)
end

breakSymOfC(::Val{:RHF}, C::AbstractMatrix{T}) where {T} = (C,)

breakSymOfC(::Val{:RHF}, Hcore, HeeI, X, Dᵅ, Dᵝ) = 
getC.( Ref(X), getF(Hcore, HeeI, ((Dᵅ + Dᵝ)./2,)) )

breakSymOfC(::Val{:UHF}, Hcore, HeeI, X, Dᵅ, Dᵝ) = 
getC.( Ref(X), getF(Hcore, HeeI, (Dᵅ, Dᵝ)) )


function getCfromGWH(::Val{HFT}, S::AbstractMatrix{T}, Hcore::AbstractMatrix{T}, 
                     X::AbstractMatrix{T}) where {HFT, T}
    H = similar(Hcore)
    Threads.@threads for j in 1:size(H, 1)
        for i in 1:j
            H[j,i] = H[i,j] = 3 * S[i,j] * (Hcore[i,i] + Hcore[j,j]) / 8
        end
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
                     nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{D, T}}, 
                     X::AbstractMatrix{T}, 
                     config=SCFconfig((:ADIIS,), (max(1e-2, 10getAtolVal(T)),))) where 
                    {HFT, T, D, NN}
    N₁tot = 0
    N₂tot = 0
    atmNs = fill((0,0), NN)
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

    nThreads = Threads.nthreads()
    len1, len2 = size(Hcore)
    Dᵅs = [zeros(T, len1, len2) for _=1:nThreads]
    Dᵝs = [zeros(T, len1, len2) for _=1:nThreads]
    @sync for (atm, atmN, coord) in zip(orderedNuc, atmNs, nucCoords[order])
        Threads.@spawn begin
            h1 = coreH(bs, (atm,), (coord,))
            r, _ = runHFcore(Val(:UHF), 
                             config, atmN, h1, HeeI, S, X, getCfromHcore(Val(:UHF), X, h1), 
                             SADHFmaxStep, true)
            Dᵅs[Threads.threadid()] += r[1].Ds[end]
            Dᵝs[Threads.threadid()] += r[2].Ds[end]
        end
    end

    breakSymOfC(Val(HFT), Hcore, HeeI, X, sum(Dᵅs), sum(Dᵝs))
end


const guessCmethods = (GWH=getCfromGWH, Hcore=getCfromHcore, SAD=getCfromSAD)


getD(Cˢ::AbstractMatrix{T}, Nˢ::Int) where {T} = @views (Cˢ[:,1:Nˢ]*Cˢ[:,1:Nˢ]')
# Nˢ: number of electrons with the same spin.

@inline getD(X::AbstractMatrix{T}, Fˢ::AbstractMatrix{T}, Nˢ::Int) where {T} = 
        getD(getC(X, Fˢ), Nˢ)


function getGcore(HeeI::AbstractArray{T, 4}, 
                  DJ::AbstractMatrix{T}, DK::AbstractMatrix{T}) where {T}
    G = similar(DJ)
    Threads.@threads for ν = 1:size(G, 1)
        for μ = 1:ν
            G[ν, μ] = G[μ, ν] = 
            dot(transpose(DJ), @view HeeI[μ,ν,:,:]) - dot(DK, @view HeeI[μ,:,:,ν])
        end
    end
    G
end

# RHF
@inline getG(HeeI::AbstractArray{T, 4}, (Dˢ,)::Tuple{AbstractMatrix{T}}) where {T} = 
        ( getGcore(HeeI, 2Dˢ, Dˢ), )

# UHF
@inline getG(HeeI::AbstractArray{T, 4}, (Dᵅ, Dᵝ)::NTuple{2, AbstractMatrix{T}}) where {T} = 
        ( getGcore(HeeI, Dᵅ+Dᵝ, Dᵅ), getGcore(HeeI, Dᵅ+Dᵝ, Dᵝ) )


@inline getF(Hcore::AbstractMatrix{T}, G::NTuple{HFTS, AbstractMatrix{T}}) where 
            {T, HFTS} = 
        Ref(Hcore) .+ G

@inline getF(Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
             D::NTuple{HFTS, AbstractMatrix{T}}) where {T, HFTS} = 
        getF(Hcore, getG(HeeI, D))


# RHF or UHF
@inline getE(Hcore::AbstractMatrix{T}, 
             Fˢ::AbstractMatrix{T}, Dˢ::AbstractMatrix{T}) where {T} = dot(Dˢ, Hcore+Fˢ) / 2

get2SpinQuantity(O::NTuple{HFTS, T}) where {HFTS, T} = abs(3-HFTS) * sum(O)
get2SpinQuantities(O, nRepeat::Int) = ntuple(_->get2SpinQuantity(O), nRepeat)

# RHF or UHF
getEhfCore(Hcore::AbstractMatrix{T}, 
           Fˢ::NTuple{HFTS, AbstractMatrix{T}}, Dˢ::NTuple{HFTS, AbstractMatrix{T}}) where 
          {T, HFTS} = 
get2SpinQuantity(getE.(Ref(Hcore), Fˢ, Dˢ))

# RHF or UHF
function getEhf(Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
                C::NTuple{HFTS, AbstractMatrix{T}}, Ns::NTuple{HFTS, Int}) where {T, HFTS}
    D = getD.(C, Ns)
    F = getF(Hcore, HeeI, D)
    getEhfCore(Hcore, F, D)
end

# RHF for MO
function getEhf((HcoreMO,)::Tuple{<:AbstractMatrix{T}}, 
                (HeeIMO,)::Tuple{<:AbstractArray{T, 4}}, (Nˢ,)::Tuple{Int}) where {T}
    term1 = 2 * (sum∘view)(diag(HcoreMO), 1:Nˢ)
    term2 = T(0)
    for i in 1:Nˢ, j in 1:Nˢ
        term2 += 2 * HeeIMO[i,i,j,j] - HeeIMO[i,j,j,i]
    end
    term1 + term2
end

#  RHF for MO in GTBasis
function getEhf(gtb::GTBasis{T, D, BN}, 
                nuc::AVectorOrNTuple{String, NN}, 
                nucCoords::SpatialCoordType{T, D, NN}, 
                N::Union{Int, Tuple{Int}, NTuple{2, Int}}; 
                errorThreshold::Real=10getAtolVal(T)) where {T, D, BN, NN}
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
    res = mapreduce(+, HcoreMOs, HeeIMOs, Ns) do HcoreMO, HeeIMO, Nˢ
        (sum∘view)(diag(HcoreMO), 1:Nˢ) + 
        sum((HeeIMO[i,i,j,j] - HeeIMO[i,j,j,i]) for j in 1:Nˢ for i in 1:(j-1))
    end
    res + sum(Jᵅᵝ[i,j] for i=1:Ns[begin], j=1:Ns[end])
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


function initializeSCFcore(::Val{HFT}, Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
                           C::NTuple{HFTS, AbstractMatrix{T}}, Ns::NTuple{HFTS, Int}) where 
                          {HFT, T, HFTS}
    D = getD.(C, Ns)
    F = getF(Hcore, HeeI, D)
    E = getE.(Ref(Hcore), F, D)
    res = HFtempVars.(Val(HFT), Ns, C, D, F, E)
    sharedFields = getproperty.(res, :shared)
    fields = (:Dtots, :Etots)
    for (field, val) in zip(fields, fill.(get2SpinQuantity.((D, E)), 1))
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


const slvArgN = :solver

const Doc_SCFconfig_OneRowTable = "|`:DIIS`, `:EDIIS`, `:ADIIS`|subspace size; "*
                                  "DIIS-Method solver|`DIISsize`; `$(slvArgN)`"*
                                  "|`1`,`2`...; `:LBFGS`... |`$(defaultDIISconfig[1])"*
                                  "`; `:$(defaultDIISconfig[2])`|"

const Doc_SCFconfig_DIIS = "[Direct inversion in the iterative subspace]"*
                           "(https://onlinelibrary.wiley.com/doi/10.1002/jcc.540030413)."
const Doc_SCFconfig_ADIIS = "[DIIS based on the augmented Roothaan–Hall (ARH) energy "*
                            "function](https://aip.scitation.org/doi/10.1063/1.3304922)."
const Doc_SCFconfig_LBFGSB = "[Limited-memory BFGS with box constraints]"*
                             "(https://github.com/JuliaNLSolvers/Optim.jl)."

const Doc_SCFconfig_SPGB = "[Spectral Projected Gradient Method with box constraints]"*
                           "(https://github.com/m3g/SPGBox.jl)."

const Doc_SCFconfig_eg1 = "SCFconfig{Float64, 2, Tuple{Val{:ADIIS}, Val{:DIIS}}}(method, "*
                          "interval=(0.001, 1.0e-8), methodConfig, oscillateThreshold)"

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

### Convergence Methods
* `:DD`: Direct diagonalization of the Fock matrix.
* `:DIIS`: $(Doc_SCFconfig_DIIS)
* `:EDIIS`: [Energy-DIIS](https://aip.scitation.org/doi/abs/10.1063/1.1470195).
* `:ADIIS`: $(Doc_SCFconfig_ADIIS)

### DIIS-Method Solvers
* `:LBFGS`: $(Doc_SCFconfig_LBFGSB)
* `:LCM`: Lagrange multiplier solver.
* `:SPGB`: $(Doc_SCFconfig_SPGB)

`interval::NTuple{L, T}`: The stopping (or skipping) thresholds for required methods.

`methodConfig::NTuple{L, Vector{<:Pair}}`: The additional keywords arguments for each 
method stored as `Tuple`s of `Pair`s.

`oscillateThreshold::T`: The threshold for oscillating convergence.

≡≡≡ Initialization Method(s) ≡≡≡

    SCFconfig(methods::NTuple{L, Symbol}, intervals::NTuple{L, T}, 
              config::Dict{Int, <:AbstractVector{<:Pair}}=Dict(1=>Pair[]);
              oscillateThreshold::Real=$(defultOscThreshold)) where {L, T} -> 
    SCFconfig{T, L}

`methods` and `intervals` are the convergence methods to be applied and their stopping 
(or skipping) thresholds respectively. `config` specifies additional keyword argument(s) 
for each methods by a `Pair` of which the key `i::Int` is for `i`th method and the pointed 
`AbstractVector{<:Pair}` is the pairs of keyword arguments and their values respectively.

    SCFconfig(;threshold::AbstractFloat=$(defaultSCFconfigArgs[2][1:end]), 
               oscillateThreshold::Real=defultOscThreshold) -> 
    SCFconfig{$(defaultSCFconfigArgs[2] |> eltype), $(defaultSCFconfigArgs[1] |> length)}

`threshold` will update the stopping threshold of the default SCF configuration used in 
$(defaultHFCStr) with a new value. In other words, it updates the stopping threshold of 
`:$(defaultSCFconfigArgs[2][end])`.

≡≡≡ Example(s) ≡≡≡
```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> SCFconfig((:DD, :ADIIS, :DIIS), (1e-4, 1e-12, 1e-13), Dict(2=>[:$(slvArgN)=>:LCM]));

julia> SCFconfig(threshold=1e-8, oscillateThreshold=1e-5)
$(Doc_SCFconfig_eg1)
```
"""
struct SCFconfig{T, L, MS<:NTuple{L, Val}} <: ImmutableParameter{T, SCFconfig}
    method::MS
    interval::NTuple{L, T}
    methodConfig::NTuple{L, Vector{<:Pair}}
    oscillateThreshold::T

    function SCFconfig(methods::NTuple{L, Symbol}, intervals::NTuple{L, T}, 
                       config::Dict{Int, <:AbstractVector{<:Pair}}=Dict(1=>Pair[]);
                       oscillateThreshold::Real=defultOscThreshold) where {L, T}
        any(i < 0 for i in intervals) && throw(DomainError(intervals, "Thresholds in "*
                                               "`intervals` must all be non-negative."))
        kwPairs = [Pair[] for _=1:L]
        for i in keys(config)
            kwPairs[i] = config[i]
        end
        methods = Val.(methods)
        new{T, L, typeof(methods)}(methods, intervals, Tuple(kwPairs), oscillateThreshold)
    end
end

const defaultSCFconfig = SCFconfig(defaultSCFconfigArgs...)

SCFconfig(;threshold::AbstractFloat=defaultSCFconfigArgs[2][end], 
          oscillateThreshold::Real=defultOscThreshold) = 
SCFconfig( defaultSCFconfigArgs[1], 
          (defaultSCFconfigArgs[2][1:end-1]..., Float64(threshold)); 
           oscillateThreshold )


mutable struct HFinterrelatedVars{T} <: HartreeFockintermediateData{T}
    Dtots::Vector{Matrix{T}}
    Etots::Vector{T}

    HFinterrelatedVars{T}() where {T} = new{T}()
    HFinterrelatedVars(Dts::AbstractVector{<:AbstractMatrix{T}}, 
                       Ets::AbstractVector{T}) where {T} = 
    new{T}(Dts, Ets)
end

getSpinOccupations(::Val{:RHF}, (Nˢ,)::Tuple{Int}, BN) = 
((fill(spinOccupations[4], Nˢ)..., fill(spinOccupations[1], BN-Nˢ)...),)

getSpinOccupations(::Val{:UHF}, (Nᵅ, Nᵝ)::NTuple{2, Int}, BN) = 
( (fill(spinOccupations[2], Nᵅ)..., fill(spinOccupations[1], BN-Nᵅ)...), 
  (fill(spinOccupations[3], Nᵝ)..., fill(spinOccupations[1], BN-Nᵝ)...) )

"""
    HFtempVars{T, HFT} <: HartreeFockintermediateData{T}

The container to store the intermediate values (only of the one spin configuration) for 
each iteration during the Hartree-Fock SCF procedure.

≡≡≡ Field(s) ≡≡≡

`N::Int`: The number of electrons with the one spin function.

`Cs::Vector{Matrix{T}}`: Coefficient matrices.

`Ds::Vector{Matrix{T}}`: Density matrices corresponding to only spin configuration.

`Fs::Vector{Matrix{T}}`: Fock matrices.

`Es::Vector{T}`: Part of the Hartree-Fock energy corresponding to one spin configuration.

`shared.Dtots::Vector{Matrix{T}}`: The total density matrices.

`shared.Etots::Vector{T}`: The total Hartree-Fock energy.

**NOTE:** For unrestricted Hartree-Fock, there are 2 `HFtempVars` being updated during the 
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


"""

    HFfinalVars{T, D, HFT, NN, BN, HFTS} <: HartreeFockFinalValue{T, HFT}

The container of the final values after a Hartree-Fock SCF procedure.

≡≡≡ Field(s) ≡≡≡

`Ehf::T`: Hartree-Fock energy of the electronic Hamiltonian.

`Enn::T`: The nuclear repulsion energy.

`Ns::NTuple{HFTS, Int}`: The number(s) of electrons with same spin configurations(s). For 
restricted closed-shell Hartree-Fock (RHF), the single element in `.Ns` represents both 
spin-up electrons and spin-down electrons.

`nuc::NTuple{NN, String}`: The nuclei in the studied system.

`nucCoords::NTuple{NN, NTuple{D, T}}`: The coordinates of corresponding nuclei.

`C::NTuple{HFTS, Matrix{T}}`: Coefficient matrix(s) for one spin configuration.

`D::NTuple{HFTS, Matrix{T}}`: Density matrix(s) for one spin configuration.

`F::NTuple{HFTS, Matrix{T}}`: Fock matrix(s) for one spin configuration.

`Eo::NTuple{HFTS, Vector{T}}`: Energies of canonical orbitals.

`occu::NTuple{HFTS, NTuple{BN, Int}}`: Occupations of canonical orbitals.

`temp::NTuple{HFTS, HFtempVars{T, HFT}}`: the intermediate values.

`isConverged::Bool`: Whether the SCF procedure is converged in the end.

`basis::GTBasis{T, D, BN}`: The basis set used for the Hartree-Fock approximation.
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
    isConverged::Bool
    basis::GTBasis{T, D, BN}

    @inline function HFfinalVars(basis::GTBasis{T, 𝐷, BN}, 
                                 nuc::AVectorOrNTuple{String, NN}, 
                                 nucCoords::SpatialCoordType{T, 𝐷, NN}, 
                                 X::AbstractMatrix{T}, 
                                 vars::NTuple{HFTS, HFtempVars{T, HFT}}, 
                                 isConverged::Bool) where {T, 𝐷, BN, NN, HFTS, HFT}
        (NNval = length(nuc)) == length(nucCoords) || 
        throw(AssertionError("The length of `nuc` and `nucCoords` should be the same."))
        any(length(i)!=𝐷 for i in nucCoords) && 
        throw(DomainError(nucCoords, "The lengths of the elements in `nucCoords` should "*
               "all be length $𝐷."))
        Ehf = vars[1].shared.Etots[end]
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

const defaultHFmaxStep = 150
const defaultHFconfigPars = [:RHF, :SAD, defaultSCFconfig, defaultHFmaxStep, true]

"""

    HFconfig{T1, HFT, F, T2, L, MS} <: ConfigBox{T1, HFconfig, HFT}

The container of Hartree-Fock method configuration.

≡≡≡ Field(s) ≡≡≡

`HF::Val{HFT}`: Hartree-Fock method type. Available values of `HFT` are 
$(string(HFtypes)[2:end-1]).

`C0::InitialC{T1, HFT, F}`: Initial guess of the coefficient matrix(s) C of the canonical 
orbitals. When `C0` is as an argument of `HFconfig`'s constructor, it can be set to 
`sym::Symbol` where available values of `sym` are 
`$((guessCmethods|>typeof|>fieldnames|>string)[2:end-1])`; it can also be a `Tuple` of 
prepared coefficient matrix(s) for the corresponding Hartree-Fock method type.

`SCF::SCFconfig{T2, L, MS}`: SCF iteration configuration. For more information please refer 
to [`SCFconfig`](@ref).

`maxStep::Int`: Maximum iteration steps allowed regardless if the iteration converges.

`earlyStop::Bool`: Whether automatically terminate (or skip) a convergence method early 
when its performance becomes unstable or poor.

≡≡≡ Initialization Method(s) ≡≡≡

    HFconfig(;HF::Union{Symbol, Val}=:$(defaultHFconfigPars[1]), 
              C0::Union{Tuple{AbstractMatrix}, NTuple{2, AbstractMatrix}, 
                        Symbol, Val}=:$(defaultHFconfigPars[2]), 
              SCF::SCFconfig=$(defaultHFconfigPars[3]), 
              maxStep::Int=$(defaultHFconfigPars[4]), 
              earlyStop::Bool=$(defaultHFconfigPars[5])) -> 
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

    HFconfig(::Val{:UHF}, 
             a2::NTuple{2, AbstractMatrix{T1}}, a3::SCFconfig{T2, L, MS}, a4, a5) where 
            {T1, T2, L, MS} = 
    new{T1, :UHF, iT, T2, L, MS}(Val(:UHF), InitialC(Val(:UHF), a2), a3, a4, a5)

    HFconfig(::Val{:RHF}, 
             a2::Tuple{AbstractMatrix{T1}}, a3::SCFconfig{T2, L, MS}, a4, a5) where 
            {T1, T2, L, MS} = 
    new{T1, :RHF, iT, T2, L, MS}(Val(:RHF), InitialC(Val(:RHF), a2), a3, a4, a5)

    function HFconfig(::Val{HFT}, a2::Val{CF}, a3::SCFconfig{T, L, MS}, a4, a5) where 
                     {T, HFT, CF, L, MS}
        f = getproperty(guessCmethods, CF)
        new{T, HFT, typeof(f), T, L, MS}(Val(HFT), InitialC(Val(HFT), f, T), a3, a4, a5)
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

Main function to run a Hartree-Fock method in Quiqbox. The returned result and relevant 
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
} where {T, D}`: The basis set used for the Hartree-Fock approximation.

`nuc::Union{
    NTuple{NN, String} where NN, 
    AbstractVector{String}
}`: The nuclei in the studied system.

`nucCoords::$(SpatialCoordType)`: The coordinates of corresponding nuclei.

`config::HFconfig`: The Configuration of selected Hartree-Fock method. For more information 
please refer to [`HFconfig`](@ref).

`N::Union{Int, Tuple{Int}, NTuple{2, Int}}`: Total number of electrons, or the number(s) of 
electrons with same spin configurations(s). **NOTE:** `N::NTuple{2, Int}` is only supported 
by unrestricted Hartree-Fock (UHF).

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
    if printInfo
        Etot = round(res.Ehf + res.Enn, digits=nDigitShown)
        Ehf = round(res.Ehf, digits=nDigitShown)
        Enn = round(res.Enn, digits=nDigitShown)
        println(rpad("Hartree-Fock Energy", 20), "| ", rpad("Nuclear Repulsion", 20), 
                "| Total Energy")
        println(rpad(string(Ehf)* " Ha", 22), rpad(string(Enn)* " Ha", 22), Etot, " Ha\n")
    end
    res
end

runHF(bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}, args...; 
      printInfo::Bool=true, infoLevel::Int=defaultHFinfoL) where {T, D} = 
runHF(GTBasis(bs), args...; printInfo, infoLevel)

@inline function runHFcore(bs::GTBasis{T, D, BN, BFT}, 
                           nuc::AVectorOrNTuple{String, NN}, 
                           nucCoords::SpatialCoordType{T, D, NN}, 
                           N::Union{Int, Tuple{Int}, NTuple{2, Int}}=getCharge(nuc), 
                           config::HFconfig{<:Any, HFT}=defaultHFC; 
                           printInfo::Bool=false, 
                           infoLevel::Int=defaultHFinfoL) where {T, D, BN, BFT, NN, HFT}
    Nlow = Int(HFT==:RHF)
    totN = (N isa Int) ? N : (N[begin] + N[end])
    totN > Nlow || throw(DomainError(N, "$(HFT) requires more than $(Nlow) electrons."))
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
      throw(DimensionMismatch("The size of the input initial coefficient matrix(s) C "*
            "($(size.(C0mats))) does not match the size of the input basis set ($BN).")) )
    C0 = uniCallFunc(getC0f, getproperty(C0methodArgOrders, nameOf(getC0f)), C0mats, 
                     Val(HFT), bs.S, X, Hcore, bs.eeI, bs.basis, nuc, nucCoords)
    runHFcore(Val(HFT), config.SCF, Ns, Hcore, bs.eeI, bs.S, X, 
              C0, config.maxStep, config.earlyStop, printInfo, infoLevel)
end

runHFcore(bs::BasisSetData, nuc, nucCoords, config::HFconfig, N=getCharge(nuc); 
          printInfo::Bool=false, infoLevel::Int=defaultHFinfoL) = 
runHFcore(bs::BasisSetData, nuc, nucCoords, N, config; printInfo, infoLevel)

runHFcore(bs::AVectorOrNTuple{AbstractGTBasisFuncs{T, D}}, args...; 
          printInfo::Bool=false, infoLevel::Int=defaultHFinfoL) where {T, D} = 
runHFcore(GTBasis(bs), args...; printInfo, infoLevel)

"""

    runHFcore(HTtype, scfConfig, Ns, Hcore, HeeI, S, X, C0, maxStep, earlyStop, 
              printInfo=false, infoLevel=$(defaultHFinfoL)) -> 
    Tuple{Tuple{Vararg{HFtempVars}}, Bool}

Another method of `runHFcore` that has the same return value, but takes more underlying 
data as arguments.

=== Positional argument(s) ===

`HTtype::Val{HFT} where HFT`: Hartree-Fock method type. Available values of `HFT` are 
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

`printInfo::Bool`: Whether print out the information of iteration steps and result.

`infoLevel::Int`: Printed info's level of details when `printInfo=true`. The higher 
(the absolute value of) it is, more intermediate steps will be printed. Once `infoLevel` 
achieve `5`, every step will be printed.
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
                   printInfo::Bool=false, 
                   infoLevel::Int=defaultHFinfoL) where {HFT, T1, L, MS, HFTS, T2}
    vars = initializeSCF(Val(HFT), Hcore, HeeI, C0, Ns)
    Etots = vars[1].shared.Etots
    oscThreshold = scfConfig.oscillateThreshold
    printInfo && println(rpad(HFT, 9)*rpad("| Initial Gauss", 16), 
                         "| E = ", alignNumSign(Etots[end], roundDigits=getAtolDigits(T2)))
    isConverged = true
    i = 0
    ΔE = 0.0
    ΔDrms = 0.0
    HFcores = [genHFcore(T2, m; kws...) for (m, kws) in 
               zip(fieldtypes(MS), scfConfig.methodConfig)]
    adaptStepBl = genAdaptStepBl(infoLevel, maxStep)

    for ((HFcore, mSym), breakPoint, l) in zip(HFcores, scfConfig.interval, 1:L)
        isConverged = true
        n = 0

        while true
            i < maxStep || (isConverged = false) || break
            i += 1
            n += 1

            res = HFcore(Ns, Hcore, HeeI, S, X, vars)
            pushHFtempVars!(vars, res)

            ΔE = Etots[end] - Etots[end-1]
            relDiff = ΔE / abs(Etots[end-1])
            sqrtBreakPoint = sqrt(breakPoint)

            if l==L
                ΔD = vars[1].shared.Dtots[end] - vars[1].shared.Dtots[end-1]
                ΔDrms = sqrt( sum(ΔD .^ 2) ./ length(ΔD) )
            end

            if n > 1 && (!isConverged || (bl = relDiff > 1e-3))
                flag, Std = isOscillateConverged(Etots, 10breakPoint, minCycles=HFOminCycle)
                if flag
                    isConverged = ifelse(
                        begin
                            bl2 = Std > max(breakPoint, oscThreshold)
                            ifelse(l==L, bl2 || (ΔDrms > sqrtBreakPoint), bl2)
                        end, false, true)
                else
                    earlyStop && bl && (i > HFOminCycle) && 
                    (i = terminateSCF(i, vars, mSym, printInfo); isConverged=false; break)
                end
            end

            printInfo && (adaptStepBl(i) || i == maxStep) && 
            println(rpad("Step $i", 9), rpad("| #$l ($(mSym))", 16), 
                    "| E = ", alignNumSign(Etots[end], roundDigits=getAtolDigits(T2)))

            isConverged && abs(ΔE) <= breakPoint && break
        end
    end
    negStr = ifelse(isConverged, "converged", "stopped but not converged")
    if printInfo
        println("\nThe SCF iteration is ", negStr, " at step $i:\n", 
                "|ΔE| → ", round(abs(ΔE), digits=nDigitShown), " Ha, ", 
                "RMS(ΔD) → ", round(ΔDrms, digits=nDigitShown), ".\n")
    end
    vars, isConverged
end

function terminateSCF(i, vars, method, printInfo)
    popHFtempVars!(vars)
    printInfo && println("Early termination of ", method, " due to its poor performance.")
    i-1
end


function DDcore(Nˢ::Int, X::AbstractMatrix{T}, F::AbstractMatrix{T}, D::AbstractMatrix{T}, 
                dampStrength::T=T(defaultDS)) where {T}
    0 <= dampStrength <= 1 || throw(DomainError(dampStrength, "The value of `dampStrength`"*
                                    " should be between 0 and 1."))
    Dnew = getD(X, F, Nˢ)
    (1 - dampStrength)*Dnew + dampStrength*D
end

function DD(::Type{T}; kw...) where {T}
    dampStrength = get(kw, :dampStrength, T(defaultDS))
    function (Ns::NTuple{HFTS, Int}, Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
              _S, X::AbstractMatrix{T}, 
              vars::NTuple{HFTS, HFtempVars{T, HFT}}) where {HFTS, HFT}
        Fs = last.(getproperty.(vars, :Fs))
        Ds = last.(getproperty.(vars, :Ds))
        Dnew = DDcore.(Ns, Ref(X), Fs, Ds, dampStrength)
        getF(Hcore, HeeI, Dnew)
    end
end


function EDIIScore(∇s::AbstractVector{<:AbstractMatrix{T}}, 
                   Ds::AbstractVector{<:AbstractMatrix{T}}, Es::AbstractVector{T}) where {T}
    len = length(Ds)
    B = similar(∇s[begin], len, len)
    Threads.@threads for j in eachindex(Ds)
        for i = 1:j
            B[i,j] = B[j,i] = -dot(Ds[i]-Ds[j], ∇s[i]-∇s[j])
        end
    end
    Es, B
end

function ADIIScore(∇s::AbstractVector{<:AbstractMatrix{T}}, 
                   Ds::AbstractVector{<:AbstractMatrix{T}}) where {T}
    v = dot.(Ds .- Ref(Ds[end]), Ref(∇s[end]))
    B = map(Iterators.product(eachindex(Ds), eachindex(∇s))) do idx
        i, j = idx
        dot(Ds[i]-Ds[end], ∇s[j]-∇s[end])
    end
    v, B
end

function DIIScore(∇s::AbstractVector{<:AbstractMatrix{T}}, 
                  Ds::AbstractVector{<:AbstractMatrix{T}}, S::AbstractMatrix{T}) where {T}
    len = length(Ds)
    B = similar(∇s[begin], len, len)
    v = zeros(T, len)
    Threads.@threads for j in eachindex(Ds)
        for i = 1:j
            B[i,j] = B[j,i] = dot( ∇s[i]*Ds[i]*S - S*Ds[i]*∇s[i], 
                                   ∇s[j]*Ds[j]*S - S*Ds[j]*∇s[j] )
        end
    end
    v, B
end

function xDIIS(::Type{T}, ::Val{M}; 
               DIISsize::Int=defaultDIISconfig[begin], 
               solver::Symbol=defaultDIISconfig[end]) where {T, M}
    c = collect(T, 1:DIISsize)
    cvxConstraint, mDIIS = getproperty(DIISconfigs, M)
    @inline function (_Ns, _Hcore, _HeeI, S::AbstractMatrix{T}, 
                      _X, vars::NTuple{<:Any, HFtempVars{T, HFT}}) where {HFT}
        Fs = getproperty.(vars, :Fs)
        Ds = getproperty.(vars, :Ds)
        Es = getproperty.(vars, :Es)
        xDIIScore!.(mDIIS, Ref(c), Ref(S), Fs, Ds, Es, DIISsize, cvxConstraint, solver)
    end
end

#                     convex constraint|unified function signature
const DIISconfigs = ( DIIS=(Val(false), (∇s, Ds, Es, S)-> DIIScore(∇s, Ds, S)), 
                      EDIIS=(Val(true), (∇s, Ds, Es, S)->EDIIScore(∇s, Ds, Es)), 
                      ADIIS=(Val(true), (∇s, Ds, Es, S)->ADIIScore(∇s, Ds)) )

function xDIIScore!(mDIIS::F, c::AbstractVector{T}, S::AbstractMatrix{T}, 
                    Fs::AbstractVector{<:AbstractMatrix{T}}, 
                    Ds::AbstractVector{<:AbstractMatrix{T}}, 
                    Es::AbstractVector{T}, 
                    DIISsize::Int, 
                    cvxConstraint::Val{CCB}, 
                    solver::Symbol) where {F, T, CCB}
    if length(Fs) > DIISsize
        push!(c, 0)
        popfirst!(c)
        is = ( (1-DIISsize):0 ) .+ lastindex(Fs)
        cp = c
    else
        is = (:)
        cp = view(c, eachindex(Fs))
    end
    ∇s = view(Fs, is)
    Ds = view(Ds, is)
    Es = view(Es, is)
    v, B = mDIIS(∇s, Ds, Es, S)
    constraintSolver!(cvxConstraint, cp, v, B, solver)
    sum(cp.*∇s) # Fnew
end


getMethodForF(::Type{T}, ::Val{:DIIS}; kws...) where {T} = xDIIS(T, Val(:DIIS); kws...)
getMethodForF(::Type{T}, ::Val{:ADIIS}; kws...) where {T} = xDIIS(T, Val(:ADIIS); kws...)
getMethodForF(::Type{T}, ::Val{:EDIIS}; kws...) where {T} = xDIIS(T, Val(:EDIIS); kws...)
getMethodForF(::Type{T}, ::Val{:DD}; kws...) where {T} = DD(T; kws...)


function genHFcore(::Type{T}, ::Type{Val{M}}; kws...) where {T, M}
    methodForF = getMethodForF(T, Val(M); kws...)
    f = function (Ns::NTuple{HFTS, Int}, 
                  Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
                  S::AbstractMatrix{T}, X::AbstractMatrix{T}, 
                  vars::NTuple{HFTS, HFtempVars{T, HFT}}) where {HFTS, HFT}
        F = methodForF(Ns, Hcore, HeeI, S, X, vars)
        getCDFE(Hcore, HeeI, X, Ns, F)
    end
    f, M
end


function pushHFtempVarsCore1!(tVars::HFtempVars, 
                              res::Tuple{AbstractMatrix{T}, AbstractMatrix{T}, 
                                         AbstractMatrix{T}, T, 
                                         AbstractMatrix{T}, T}) where {T}
    push!(tVars.Cs, res[1])
    push!(tVars.Ds, res[2])
    push!(tVars.Fs, res[3])
    push!(tVars.Es, res[4])
end

function pushHFtempVarsCore2!(tVars::HFtempVars, 
                              res::Tuple{AbstractMatrix{T}, AbstractMatrix{T}, 
                                         AbstractMatrix{T}, T, 
                                         AbstractMatrix{T}, T}) where {T}
    push!(tVars.shared.Dtots, res[5])
    push!(tVars.shared.Etots, res[6])
end

function pushHFtempVars!(αβVars::NTuple{HFTS, HFtempVars{T, HFT}}, 
                         res::NTuple{HFTS, 
                                     Tuple{AbstractMatrix{T}, AbstractMatrix{T}, 
                                           AbstractMatrix{T}, T, 
                                           AbstractMatrix{T}, T}}) where {HFTS, T, HFT}
    pushHFtempVarsCore1!.(αβVars, res)
    pushHFtempVarsCore2!(αβVars[1], res[1])
end


function popHFtempVarsCore1!(tVars::HFtempVars)
    pop!(tVars.Cs)
    pop!(tVars.Ds)
    pop!(tVars.Fs)
    pop!(tVars.Es)
end

function popHFtempVarsCore2!(tVars::HFtempVars)
    pop!(tVars.shared.Dtots)
    pop!(tVars.shared.Etots)
end

function popHFtempVars!(αβVars::NTuple{HFTS, HFtempVars{T, HFT}}) where {HFTS, T, HFT}
    popHFtempVarsCore1!.(αβVars)
    popHFtempVarsCore2!(αβVars[1])
end

function shiftLastEle!(v, shiftVal)
    s = sum(v)
    signedShift = asymSign(s)*shiftVal
    s += signedShift
    v[end] += signedShift
    s, signedShift
end

# Included normalization condition, but not non-negative condition.
@inline function genxDIISf(v, B, shift)
    function (c)
        s, signedShift = shiftLastEle!(c, shift)
        res = dot(v, c) / s + dot(c, B, c) / (2s^2)
        c[end] -= signedShift
        res
    end
end

@inline function genxDIIS∇f(v, B, shift)
    function (g, c)
        s, signedShift = shiftLastEle!(c, shift)
        g.= v./s + (B + transpose(B))*c ./ (2s^2) .- (dot(v, c)/s^2 + dot(c, B, c)/s^3)
        c[end] -= signedShift
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
    vL = length(v)
    innerOptimizer = LBFGS(m=min(getAtolDigits(T), 50), 
                                 linesearch=HagerZhang(linesearchmax=100, epsilon=1e-7), 
                                 alphaguess=InitialHagerZhang())
    c .= OptimOptimize(f, g!, fill(lb, vL), fill(T(Inf), vL), collect(T, 1:vL), 
                       Fminbox(innerOptimizer), 
                       OptimOptions(g_tol=exp10(-getAtolDigits(T)), iterations=10000, 
                       allow_f_increases=false)) |> OptimMinimizer
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
    spgbox!(f, g!, c, lower=fill(lb, vL), eps=1e-6)
    s, _ = shiftLastEle!(c, shift)
    c ./= s
end

function CMsolver!(::Val{CCB}, c::AbstractVector{T}, 
                   v::AbstractVector{T}, B::AbstractMatrix{T}, ϵ::T=T(1e-6)) where {CCB, T}
    len = length(v)
    getA = M->[M  ones(T, len); ones(T, 1, len) T(0)]
    b = vcat(-v, 1)
    while true
        A = getA(B)
        while det(A) == 0
            B += ϵ*I
            A = getA(B)
        end
        c .= (A \ b)[begin:end-1]
        (CCB && findfirst(x->x<0, c) !== nothing) || (return c)
        idx = powerset(sortperm(abs.(c)), 1)

        for is in idx
            Atemp = A[begin:end .∉ Ref(is), begin:end .∉ Ref(is)]
            det(Atemp) == 0 && continue
            btemp = b[begin:end .∉ Ref(is)]
            cL = (Atemp \ btemp)[begin:end-1]
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