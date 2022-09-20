export SCFconfig, HFconfig, runHF, runHFcore

using LinearAlgebra: dot, Hermitian, \, det, I, ishermitian
using Combinatorics: powerset
using LineSearches
using Optim: LBFGS, Fminbox, optimize as OptimOptimize, minimizer as OptimMinimizer, 
             Options as OptimOptions

const defaultDS = 0.5
const defaultDIISconfig = (12, :LBFGS)

const defaultHFCStr = "HFconfig()"
const defaultSCFconfigArgs = ( (:ADIIS, :DIIS), (5e-3, 1e-12) )
const defultOscThreshold = 1e-6


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

splitSpins(::Val{1}, Ns::NTuple{2, Int}) = (sum(Ns)÷2,)

splitSpins(::Val{:RHF}, N) = splitSpins(Val(HFtypeSizeList[:RHF]), N)

splitSpins(::Val{:UHF}, N) = splitSpins(Val(HFtypeSizeList[:UHF]), N)

groupSpins(::Val{1}, (Nˢ,)::Tuple{Int}) = (Nˢ, Nˢ)

groupSpins(::Val{2}, Ns::NTuple{2, Int}) = itself(Ns)

groupSpins(::Val{:RHF}, Ns::Tuple{Vararg{Int}}) = groupSpins(Val(HFtypeSizeList[:RHF]), Ns)

groupSpins(::Val{:UHF}, Ns::Tuple{Vararg{Int}}) = groupSpins(Val(HFtypeSizeList[:UHF]), Ns)


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
    for j in 1:size(H, 1), i in 1:j
        H[j,i] = H[i,j] = 3 * S[i,j] * (Hcore[i,i] + Hcore[j,j]) / 8
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
                     bs::NTuple{BN, AbstractGTBasisFuncs{T, D}}, 
                     nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{D, T}}, 
                     X::AbstractMatrix{T}, 
                     config=SCFconfig((:ADIIS,), (max(1e-2, 10getAtolVal(T)),))) where 
                    {HFT, T, D, BN, NN}
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
                            config, atmN, h1, HeeI, S, X, getCfromHcore(Val(:UHF), X, h1))
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
    @sync for ν = 1:size(G, 1)
        Threads.@spawn for μ = 1:ν # Spawn here is faster than spawn inside the loop.
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
             Fˢ::AbstractMatrix{T}, Dˢ::AbstractMatrix{T}) where {T} = 
        dot(transpose(Dˢ), Hcore+Fˢ) / 2

get2SpinQuantity(O::NTuple{HFTS, T}) where {HFTS, T} = abs(3-HFTS) * sum(O)
get2SpinQuantities(O, nRepeat::Int) = ntuple(_->get2SpinQuantity(O), nRepeat)

# RHF or UHF
getEᵗcore(Hcore::AbstractMatrix{T}, 
          Fˢ::NTuple{HFTS, AbstractMatrix{T}}, Dˢ::NTuple{HFTS, AbstractMatrix{T}}) where 
         {T, HFTS} = 
get2SpinQuantity(getE.(Ref(Hcore), Fˢ, Dˢ))

# RHF or UHF
function getEᵗ(Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
               C::NTuple{HFTS, AbstractMatrix{T}}, N::NTuple{HFTS, Int}) where {T, HFTS}
    D = getD.(C, N)
    F = getF(Hcore, HeeI, D)
    getEᵗcore(Hcore, F, D)
end


function getCDFE(Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, X::AbstractMatrix{T}, 
                 N::NTuple{HFTS, Int}, F::NTuple{HFTS, AbstractMatrix{T}}) where {T, HFTS}
    Cnew = getC.(Ref(X), F)
    Dnew = getD.(Cnew, N)
    Fnew = getF(Hcore, HeeI, Dnew)
    Enew = getE.(Ref(Hcore), Fnew, Dnew)
    Dᵗnew = get2SpinQuantities(Dnew, HFTS)
    Eᵗnew = get2SpinQuantities(Enew, HFTS)
    map(themselves, Cnew, Dnew, Fnew, Enew, Dᵗnew, Eᵗnew)
end


function initializeSCFcore(::Val{HFT}, Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
                       C::NTuple{HFTS, AbstractMatrix{T}}, N::NTuple{HFTS, Int}) where 
                      {HFT, T, HFTS}
    D = getD.(C, N)
    F = getF(Hcore, HeeI, D)
    E = getE.(Ref(Hcore), F, D)
    res = HFtempVars.(Val(HFT), N, C, D, F, E)
    sharedFields = getproperty.(res, :shared)
    fields = (:Dtots, :Etots)
    for (field, val) in zip(fields, fill.(get2SpinQuantity.((D, E)), 1))
        setproperty!.(sharedFields, field, Ref(val))
    end
    res::NTuple{HFTS, HFtempVars{T, HFT}} # A somehow necessary assertion for type stability
end

# Additional wrapper to correlate `HTF` and `HFTS` for type stability.
initializeSCF(::Val{:RHF}, Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
              C::Tuple{AbstractMatrix{T}}, N::Tuple{Int}) where {T} = 
initializeSCFcore(Val(:RHF), Hcore, HeeI, C, N)

initializeSCF(::Val{:UHF}, Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
              C::NTuple{2, AbstractMatrix{T}}, N::NTuple{2, Int}) where {T} = 
initializeSCFcore(Val(:UHF), Hcore, HeeI, C, N)


const Doc_SCFconfig_OneRowTable = "|`:DIIS`, `:EDIIS`, `:ADIIS`|subspace size; "*
                                  "coefficient solver|`DIISsize`; `solver`|`1`,`2`...; "*
                                  "`:LCM`, `:LBFGS`|`$(defaultDIISconfig[1])`; "*
                                  "`:$(defaultDIISconfig[2])`|"

const Doc_SCFconfig_DIIS = "[Direct inversion in the iterative subspace]"*
                           "(https://onlinelibrary.wiley.com/doi/10.1002/jcc.540030413)."
const Doc_SCFconfig_ADIIS = "[DIIS based on the augmented Roothaan–Hall (ARH) energy "*
                            "function](https://aip.scitation.org/doi/10.1063/1.3304922)."
const Doc_SCFconfig_LBFGSB = "[Limited-memory BFGS with box constraints]"*
                             "(https://github.com/JuliaNLSolvers/Optim.jl)."

const Doc_SCFconfig_eg1 = "SCFconfig{Float64, 2}(method=(:ADIIS, :DIIS), "*
                          "interval=(0.005, 1.0e-8), methodConfig, oscillateThreshold)"

"""

    SCFconfig{T, L} <: ImmutableParameter{T, SCFconfig}

The `struct` for self-consistent field (SCF) iteration configurations.

≡≡≡ Field(s) ≡≡≡

`method::NTuple{L, Symbol}`: The applied convergence methods. The available methods and 
their configurations (in terms of keyword arguments):

| Convergence Method(s) | Configuration(s) | Keyword(s) | Range(s)/Option(s) | Default(s) |
| :----                 | :---:            | :---:      | :---:              |      ----: |
| `:DD`                 | damping strength |`dampStrength`|    [`0`, `1`]  |`$(defaultDS)`|
$(Doc_SCFconfig_OneRowTable)

### Convergence Methods
* DD: Direct diagonalization of the Fock matrix.
* DIIS: $(Doc_SCFconfig_DIIS)
* EDIIS: [Energy-DIIS](https://aip.scitation.org/doi/abs/10.1063/1.1470195).
* ADIIS: $(Doc_SCFconfig_ADIIS)

### DIIS-type Method Solvers
* LCM: Lagrange multiplier solver.
* LBFGS: $(Doc_SCFconfig_LBFGSB)

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
julia> SCFconfig((:DD, :ADIIS, :DIIS), (1e-4, 1e-12, 1e-13), Dict(2=>[:solver=>:LCM]));

julia> SCFconfig(threshold=1e-8, oscillateThreshold=1e-5)
$(Doc_SCFconfig_eg1)
```
"""
struct SCFconfig{T, L} <: ImmutableParameter{T, SCFconfig}
    method::NTuple{L, Symbol}
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
        new{T, L}(methods, intervals, Tuple(kwPairs), oscillateThreshold)
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

`Ns::NTuple{2, Int}`: The numbers of two different spins respectively.

`nuc::NTuple{NN, String}`: The nuclei in the studied system.

`nucCoords::NTuple{NN, NTuple{D, T}}`: The coordinates of corresponding nuclei.

`C::NTuple{HFTS, Matrix{T}}`: Coefficient matrix(s) for one spin configuration.

`D::NTuple{HFTS, Matrix{T}}`: Density matrix(s) for one spin configuration.

`F::NTuple{HFTS, Matrix{T}}`: Fock matrix(s) for one spin configuration.

`Eo::NTuple{HFTS, Vector{T}}`: Energies of canonical orbitals.

`occu::NTuple{HFTS, NTuple{BN, String}}`: Occupations of canonical orbitals.

`temp::NTuple{HFTS, HFtempVars{T, HFT}}`: the intermediate values.

`isConverged::Bool`: Whether the SCF procedure is converged in the end.

`basis::GTBasis{T, D, BN}`: The basis set used for the Hartree-Fock approximation.
"""
struct HFfinalVars{T, D, HFT, NN, BN, HFTS} <: HartreeFockFinalValue{T, HFT}
    Ehf::T
    Enn::T
    Ns::NTuple{2, Int}
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
                                 nuc::VectorOrNTuple{String, NN}, 
                                 nucCoords::SpatialCoordType{T, 𝐷, NN}, 
                                 X::AbstractMatrix{T}, 
                                 vars::NTuple{HFTS, HFtempVars{T, HFT}}, 
                                 isConverged::Bool) where {T, 𝐷, BN, NN, HFTS, HFT}
        (NNval = length(nuc)) == length(nucCoords) || 
        throw(AssertionError("The length of `nuc` and `nucCoords` should be the same."))
        any(length(i)!=𝐷 for i in nucCoords) && 
        throw(DomainError(nucCoords, "The lengths of the elements in `nucCoords` should "*
               "all be length $D."))
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
        new{T, 𝐷, HFT, NNval, BN, HFTS}(Ehf, Enn, groupSpins(Val(HFT), Ns), nuc, nucCoords, 
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

const defaultHFconfigPars = [:RHF, :SAD, defaultSCFconfig, 100, true]

"""

    HFconfig{T1, HFT, F, T2, L} <: ConfigBox{T1, HFconfig, HFT}

The container of Hartree-Fock method configuration.

≡≡≡ Field(s) ≡≡≡

`HF::Val{HFT}`: Hartree-Fock method type. Available values of `HFT` are 
$(string(HFtypes)[2:end-1]).

`C0::InitialC{T1, HFT, F}`: Initial guess of the coefficient matrix(s) C of the canonical 
orbitals. When `C0` is a `Val{T}`, the available values of `T1` are 
`$((guessCmethods|>typeof|>fieldnames|>string)[2:end-1])`.

`SCF::SCFconfig{T2, L}`: SCF iteration configuration. For more information please refer to 
[`SCFconfig`](@ref).

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
mutable struct HFconfig{T1, HFT, F, T2, L} <: ConfigBox{T1, HFconfig, HFT}
    HF::Val{HFT}
    C0::InitialC{T1, HFT, F}
    SCF::SCFconfig{T2, L}
    maxStep::Int
    earlyStop::Bool

    HFconfig(::Val{:UHF}, 
             a2::NTuple{2, AbstractMatrix{T1}}, a3::SCFconfig{T2, L}, a4, a5) where 
            {T1, T2, L} = 
    new{T1, :UHF, iT, T2, L}(Val(:UHF), InitialC(Val(:UHF), a2), a3, a4, a5)

    HFconfig(::Val{:RHF}, 
             a2::Tuple{AbstractMatrix{T1}}, a3::SCFconfig{T2, L}, a4, a5) where 
            {T1, T2, L} = 
    new{T1, :RHF, iT, T2, L}(Val(:RHF), InitialC(Val(:RHF), a2), a3, a4, a5)

    function HFconfig(::Val{HFT}, a2::Val{CF}, a3::SCFconfig{T, L}, a4, a5) where 
                     {T, HFT, CF, L}
        f = getproperty(guessCmethods, CF)
        new{T, HFT, typeof(f), T, L}(Val(HFT), InitialC(Val(HFT), f, T), a3, a4, a5)
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
    runHF(bs, nuc, nucCoords, config=$(defaultHFCStr), N=getCharge(nuc); printInfo=true) -> 
    HFfinalVars

    runHF(bs, nuc, nucCoords, N=getCharge(nuc), config=$(defaultHFCStr); printInfo=true) -> 
    HFfinalVars

Main function to run a Hartree-Fock method in Quiqbox. The returned result and relevant 
information is stored in a [`HFfinalVars`](@ref).

    runHFcore(args...; printInfo=false) -> Tuple{Tuple{Vararg{HFtempVars}}, Bool}

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
electrons with same spin configurations(s).

≡≡≡ Keyword argument(s) ≡≡≡

`printInfo::Bool`: Whether print out the information of iteration steps and result.
"""
function runHF(bs::GTBasis{T}, args...; printInfo::Bool=true) where {T}
    nuc = arrayToTuple(args[begin])
    nucCoords = genTupleCoords(T, args[begin+1])
    vars, isConverged = runHFcore(bs, nuc, nucCoords, args[begin+2:end]...; printInfo)
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

runHF(bs::VectorOrNTuple{AbstractGTBasisFuncs{T, D}}, args...; 
      printInfo::Bool=true) where {T, D} = 
runHF(GTBasis(bs), args...; printInfo)

@inline function runHFcore(bs::GTBasis{T1, D, BN, BFT}, 
                           nuc::VectorOrNTuple{String, NN}, 
                           nucCoords::SpatialCoordType{T1, D, NN}, 
                           N::Union{Int, Tuple{Int}, NTuple{2, Int}}=getCharge(nuc), 
                           config::HFconfig{T2, HFT}=defaultHFC; 
                           printInfo::Bool=false) where {T1, D, BN, BFT, NN, HFT, T2}
    Nlow = Int(HFT==:RHF)
    N > Nlow || throw(DomainError(N, "$(HFT) requires more than $(Nlow) electrons."))
    Ns = splitSpins(Val(HFT), N)
    leastNb = max(Ns...)
    BN < leastNb &&  throw(DomainError(BN, "The number of basis functions should be no "*
                           "less than $(leastNb)."))
    nuc = arrayToTuple(nuc)
    nucCoords = genTupleCoords(T1, nucCoords)
    Hcore = coreH(bs, nuc, nucCoords)
    X = getX(bs.S)
    getC0f = config.C0.f
    C0 = uniCallFunc(getC0f, getproperty(C0methodArgOrders, nameOf(getC0f)), config.C0.mat, 
                     Val(HFT), bs.S, X, Hcore, bs.eeI, bs.basis, nuc, nucCoords)
    runHFcore(Val(HFT), config.SCF, Ns, Hcore, bs.eeI, bs.S, X, 
              C0, printInfo, config.maxStep, config.earlyStop)
end

runHFcore(bs::BasisSetData, nuc, nucCoords, config::HFconfig, N::Int=getCharge(nuc); 
          printInfo::Bool=false) = 
runHFcore(bs::BasisSetData, nuc, nucCoords, N, config; printInfo)

runHFcore(bs::VectorOrNTuple{AbstractGTBasisFuncs{T, D}}, args...; 
          printInfo::Bool=false) where {T, D} = 
runHFcore(GTBasis(bs), args...; printInfo)

"""

    runHFcore(HTtype, scfConfig, N, Hcore, HeeI, S, X, C0, 
              printInfo=false, maxStep=1000, earlyStop=true) -> 
    Tuple{Tuple{Vararg{HFtempVars}}, Bool}

Another method of `runHFcore` that has the same return value, but takes more underlying 
data as arguments.

=== Positional argument(s) ===

`HTtype::Val{HFT} where HFT`: Hartree-Fock method type. Available values of `HFT` are 
$(string(HFtypes)[2:end-1]).

`scfConfig::SCFconfig`: The SCF iteration configuration.

`N::NTuple{HFTS, Int} where HFTS`: The numbers of electrons with same spin configurations. 

`Hcore::AbstractMatrix{T} where T`: The core Hamiltonian of the electronic Hamiltonian.

`HeeI::AbstractArray{T, 4} where T`: The electron-electron interaction tensor (in the 
chemists' notation) which includes both the Coulomb interactions and the Exchange 
Correlations.

`S::AbstractMatrix{T} where T`: The overlap matrix of the used basis set.

`X::AbstractMatrix{T} where T`: The transformation matrix of `S`.

`C0::NTuple{HFTS, AbstractMatrix{T}} where {HFTS, T}`: Initial guess of the coefficient 
matrix(s) of the canonical spin-orbitals.

`printInfo::Bool`: Whether print out the information of iteration steps and result.

`maxStep::Int`: Maximum iteration steps allowed regardless if the iteration converges.

`earlyStop::Bool`: Whether automatically terminate (or skip) a convergence method early 
when its performance becomes unstable or poor.
"""
function runHFcore(::Val{HFT}, 
                   scfConfig::SCFconfig{T1, L}, 
                   N::NTuple{HFTS, Int}, 
                   Hcore::AbstractMatrix{T2}, 
                   HeeI::AbstractArray{T2, 4}, 
                   S::AbstractMatrix{T2}, 
                   X::AbstractMatrix{T2}, 
                   C0::NTuple{HFTS, AbstractMatrix{T2}}, 
                   printInfo::Bool=false, 
                   maxStep::Int=1000, 
                   earlyStop::Bool=true) where {HFT, T1, L, HFTS, T2}
    vars = initializeSCF(Val(HFT), Hcore, HeeI, C0, N)
    Etots = vars[1].shared.Etots
    oscThreshold = scfConfig.oscillateThreshold
    printInfo && println(rpad(HFT, 9)*rpad("| Initial Gauss", 16), 
                         "| E = ", alignNumSign(Etots[end], roundDigits=getAtolDigits(T2)))
    isConverged = true
    i = 0
    ΔE = 0.0
    ΔDrms = 0.0
    for (m, kws, breakPoint, l) in 
        zip(scfConfig.method, scfConfig.methodConfig, scfConfig.interval, 1:L)
        isConverged = true
        n = 0

        while true
            i += 1
            n += 1
            i <= maxStep || (isConverged = false) || break

            res = HFcore(m, N, Hcore, HeeI, S, X, vars; kws...)
            pushHFtempVars!(vars, res)

            ΔE = Etots[end] - Etots[end-1]
            relDiff = ΔE / abs(Etots[end-1])
            sqrtBreakPoint = sqrt(breakPoint)

            if l==L
                ΔD = vars[1].shared.Dtots[end] - vars[1].shared.Dtots[end-1]
                ΔDrms = sqrt( sum(ΔD .^ 2) ./ length(ΔD) )
            end

            if n > 1 && (!isConverged || (bl = relDiff > max(sqrtBreakPoint, 1e-5)))
                flag, Std = isOscillateConverged(Etots, 10breakPoint)
                if flag
                    isConverged = ifelse(
                        begin
                            bl2 = Std > max(breakPoint, oscThreshold)
                            ifelse(l==L, bl2 || (ΔDrms > sqrtBreakPoint), bl2)
                        end, false, true)
                else
                    earlyStop && bl && 
                    (i = terminateSCF(i, vars, m, printInfo); isConverged=false; break)
                end
            end

            printInfo && (i % floor(log(4, i) + 1) == 0 || i == maxStep) && 
            println(rpad("Step $i", 9), rpad("| #$l ($(m))", 16), 
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


function EDIIScore(∇s::AbstractVector{<:AbstractMatrix{T}}, 
                   Ds::AbstractVector{<:AbstractMatrix{T}}, Es::AbstractVector{T}) where {T}
    len = length(Ds)
    B = similar(∇s[begin], len, len)
    @sync for j=1:len, i=1:j
        Threads.@spawn B[i,j] = B[j,i] = -dot(Ds[i]-Ds[j], ∇s[i]-∇s[j])
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
    v = zeros(len)
    @sync for j in 1:len, i=1:j
        Threads.@spawn B[i,j] = B[j,i] = dot( ∇s[i]*Ds[i]*S - S*Ds[i]*∇s[i], 
                                              ∇s[j]*Ds[j]*S - S*Ds[j]*∇s[j] )
    end
    v, B
end


function DD(Nˢ::NTuple{HFTS, Int}, Hcore, HeeI, _S, X, 
            tVars::NTuple{HFTS, HFtempVars{T, HFT}}; kws...) where {HFTS, T, HFT}
    Fs = last.(getproperty.(tVars, :Fs))
    Ds = last.(getproperty.(tVars, :Ds))
    Dnew = DDcore.( Nˢ, Ref(X), Fs, Ds, get(kws, :dampStrength, T(defaultDS)) )
    getF(Hcore, HeeI, Dnew)
end


function xDIIS(::Val{M}) where {M}
    @inline function (_Nˢ, _Hcore, _HeeI, S, _X, tVars; kws...)
        Fs = getproperty.(tVars, :Fs)
        Ds = getproperty.(tVars, :Ds)
        Es = getproperty.(tVars, :Es)
        oArg1, oArg2 = get.(Ref(kws), (:DIISsize, :solver), defaultDIISconfig)
        xDIIScore.(Val(M), Ref(S), Fs, Ds, Es, oArg1, oArg2)
    end
end

const DIIScoreMethods = (DIIS=DIIScore, EDIIS=EDIIScore, ADIIS=ADIIScore)

const DIISmethodArgOrders = (DIIScore=(1,2,4), EDIIScore=(1,2,3), ADIIScore=(1,2))

const DIISadditionalConfigs = ( DIIS=(Val(false), true), 
                               EDIIS=(Val(true), false), 
                               ADIIS=(Val(true), false) )

function xDIIScore(::Val{M}, S::AbstractMatrix{T}, 
                   Fs::AbstractVector{<:AbstractMatrix{T}}, 
                   Ds::AbstractVector{<:AbstractMatrix{T}}, 
                   Es::AbstractVector{T}, 
                   DIISsize::Int=defaultDIISconfig[1], 
                   solver::Symbol=defaultDIISconfig[2]) where {M, T}
    cvxConstraint, permData = getproperty(DIISadditionalConfigs, M)
    is = length(Es)>DIISsize ? (permData ? sortperm(Es)[begin:DIISsize] : 1:DIISsize) : (:)
    ∇s = view(Fs, is)
    Ds = view(Ds, is)
    Es = view(Es, is)
    DIIS = getproperty(DIIScoreMethods, M)
    v, B = uniCallFunc(DIIS, getproperty(DIISmethodArgOrders, nameOf(DIIS)), ∇s, Ds, Es, S)
    c = constraintSolver(v, B, cvxConstraint, solver)
    sum(c.*∇s) # Fnew
end


const SCFmethodSelector = 
      (DD=DD, DIIS=xDIIS(Val(:DIIS)), ADIIS=xDIIS(Val(:ADIIS)), EDIIS=xDIIS(Val(:EDIIS)))


function HFcore(m::Symbol, N::NTuple{HFTS, Int}, 
                Hcore::AbstractMatrix{T}, HeeI::AbstractArray{T, 4}, 
                S::AbstractMatrix{T}, X::AbstractMatrix{T}, 
                rVars::NTuple{HFTS, HFtempVars{T, HFT}}; 
                kws...) where {HFTS, T, HFT}
    F = getproperty(SCFmethodSelector, m)(N, Hcore, HeeI, S, X, rVars; kws...)
    getCDFE(Hcore, HeeI, X, N, F)
end


function pushHFtempVarsCore1!(rVars::HFtempVars, 
                              res::Tuple{AbstractMatrix{T}, AbstractMatrix{T}, 
                                         AbstractMatrix{T}, T, 
                                         AbstractMatrix{T}, T}) where {T}
    push!(rVars.Cs, res[1])
    push!(rVars.Ds, res[2])
    push!(rVars.Fs, res[3])
    push!(rVars.Es, res[4])
end

function pushHFtempVarsCore2!(rVars::HFtempVars, 
                              res::Tuple{AbstractMatrix{T}, AbstractMatrix{T}, 
                                         AbstractMatrix{T}, T, 
                                         AbstractMatrix{T}, T}) where {T}
    push!(rVars.shared.Dtots, res[5])
    push!(rVars.shared.Etots, res[6])
end

function pushHFtempVars!(αβVars::NTuple{HFTS, HFtempVars{T, HFT}}, 
                         res::NTuple{HFTS, 
                                     Tuple{AbstractMatrix{T}, AbstractMatrix{T}, 
                                           AbstractMatrix{T}, T, 
                                           AbstractMatrix{T}, T}}) where {HFTS, T, HFT}
    pushHFtempVarsCore1!.(αβVars, res)
    pushHFtempVarsCore2!(αβVars[1], res[1])
end


function popHFtempVarsCore1!(rVars::HFtempVars)
    pop!(rVars.Cs)
    pop!(rVars.Ds)
    pop!(rVars.Fs)
    pop!(rVars.Es)
end

function popHFtempVarsCore2!(rVars::HFtempVars)
    pop!(rVars.shared.Dtots)
    pop!(rVars.shared.Etots)
end

function popHFtempVars!(αβVars::NTuple{HFTS, HFtempVars{T, HFT}}) where {HFTS, T, HFT}
    popHFtempVarsCore1!.(αβVars)
    popHFtempVarsCore2!(αβVars[1])
end


# Included normalization condition, but not non-negative condition.
@inline function genxDIISf(v, B)
    function (c)
        s = sum(c)
        dot(v, c) / s + transpose(c) * B * c / (2s^2)
    end
end

@inline function genxDIIS∇f(v, B)
    function (g, c)
        s = sum(c)
        g.= v./c + (B + transpose(B))*c ./ (2s^2) .- (dot(v, c)/s^2 + transpose(c)*B*c/s^3)
    end
end


# Default method
function LBFGSBsolver(::Val{CCB}, v::AbstractVector{T}, B::AbstractMatrix{T}) where {CCB, T}
    f = genxDIISf(v, B)
    g! = genxDIIS∇f(v, B)
    lb = ifelse(CCB, T(0), T(-Inf))
    vL = length(v)
    c0 = fill(T(1)/vL, vL)
    innerOptimizer = LBFGS(m=min(getAtolDigits(T), 50), 
                                 linesearch=HagerZhang(linesearchmax=100, epsilon=1e-7), 
                                 alphaguess=InitialHagerZhang())
    res = OptimOptimize(f, g!, fill(lb, vL), fill(T(Inf), vL), c0, Fminbox(innerOptimizer), 
                        OptimOptions(g_tol=exp10(-getAtolDigits(T)), iterations=10000, 
                        allow_f_increases=false))
    c = OptimMinimizer(res)
    c ./ sum(c)
end

function CMsolver(::Val{CCB}, v::AbstractVector{T}, B::AbstractMatrix{T}, 
                  ϵ::T=T(1e-6)) where {CCB, T}
    len = length(v)
    getA = (B)->[B  ones(len); ones(1, len) 0]
    b = vcat(-v, 1)
    local c
    while true
        A = getA(B)
        while det(A) == 0
            B += ϵ*I
            A = getA(B)
        end
        x = A \ b
        c = x[1:end-1]
        (CCB && findfirst(x->x<0, c) !== nothing) || (return c)
        idx = (sortperm(abs.(c)) |> powerset |> collect)
        popfirst!(idx)

        for is in idx
            Atemp = A[1:end .∉ Ref(is), 1:end .∉ Ref(is)]
            btemp = b[begin:end .∉ Ref(is)]
            det(Atemp) == 0 && continue
            xtemp = Atemp \ btemp
            c = xtemp[1:end-1]
            for i in sort(is)
                insert!(c, i, 0.0)
            end
            (findfirst(x->x<0, c) !== nothing) || (return c)
        end

        B += ϵ*I
    end
    c
end


const ConstraintSolvers = (LCM=CMsolver, LBFGS=LBFGSBsolver)

constraintSolver(v::AbstractVector{T}, B::AbstractMatrix{T}, 
                 ::Val{CCB}, # cvxConstraint
                 solver::Symbol) where {T, CCB} = 
getproperty(ConstraintSolvers, solver)(Val(CCB), v, B)