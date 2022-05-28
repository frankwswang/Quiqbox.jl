export SCFconfig, HFconfig, runHF, runHFcore

using LinearAlgebra: dot, Hermitian, \, det, I
using PiecewiseQuadratics: indicator
using Combinatorics: powerset
using LBFGSB: lbfgsb

getXcore1(S::Matrix{Float64}) = Hermitian(S)^(-0.5) |> Array

precompile(getXcore1, (Matrix{Float64},))

const getXmethods = (m1=getXcore1,)

getX(S::Matrix{Float64}, method::Symbol=:m1) = getfield(getXmethods, method)(S)


function getCϵ(X::Matrix{Float64}, F::Matrix{Float64}, stabilizeSign::Bool=true)
    ϵ, Cₓ = eigen(X'*F*X |> Hermitian)
    outC = X*Cₓ
    # Stabilize the sign factor of each column.
    stabilizeSign && for j = 1:size(outC, 2)
       outC[:, j] *= (outC[1,j] < 0 ? -1 : 1)
    end
    outC, ϵ
end

@inline getC(X::Matrix{Float64}, F::Matrix{Float64}, stabilizeSign::Bool=true) = 
        getCϵ(X, F, stabilizeSign)[1]


splitSpins(::Val{1}, N::Int) = (N,)

splitSpins(::Val{2}, N::Int) = (N÷2, N-N÷2)

splitSpins(::Val{:RHF}, N::Int) = splitSpins(Val(1), N)

splitSpins(::Val{:UHF}, N::Int) = splitSpins(Val(2), N)

splitSpins(::Val, Ns::Tuple) = itself(Ns)


function breakSymOfC(::Val{:UHF}, C::Matrix{Float64})
    C2 = copy(C)
    l = min(size(C2)[1], 2)
    C2[1:l, 1:l] .= 0 # Breaking spin symmetry.
    # C2[l, :] .= 0 # Another way.
    (copy(C), C2)
end

breakSymOfC(::Val{:RHF}, C::Matrix{Float64}) = (C,)

function breakSymOfC(::Val{:RHF}, X, D₁, D₂, Hcore, HeeI)
    Dᵀ = D₁ + D₂
    (getC(X, getF(Hcore, HeeI, Dᵀ.*0.5, Dᵀ)),)
end

function breakSymOfC(::Val{:UHF}, X, D₁, D₂, Hcore, HeeI)
    Dᵀ = D₁ + D₂
    getC.(Ref(X), getF.(Ref(Hcore), Ref(HeeI), (D₁, D₂), Ref(Dᵀ)))
end


function getCfromGWH(::Val{HFT}, S::Matrix{Float64}, Hcore::Matrix{Float64}, 
                     X::Matrix{Float64}) where {HFT}
    l = size(Hcore)[1]
    H = zero(Hcore)
    for j in 1:l, i in 1:l
        H[i,j] = 1.75 * S[i,j] * (Hcore[i,i] + Hcore[j,j]) * 0.5
    end
    C = getC(X, H)
    breakSymOfC(Val(HFT), C)
end


function getCfromHcore(::Val{HFT}, X::Matrix{Float64}, Hcore::Matrix{Float64}) where {HFT}
    C = getC(X, Hcore)
    breakSymOfC(Val(HFT), C)
end


function getCfromSAD(::Val{HFT}, S::Matrix{Float64}, 
                     Hcore::Matrix{Float64}, HeeI::Array{Float64, 4},
                     bs::Union{NTuple{BN, BT}, NTuple{BN, AbstractGTBasisFuncs}}, 
                     nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{3,Float64}}, 
                     X::Matrix{Float64}, 
                     config=SCFconfig((:ADIIS,), (1e-10,))) where 
                    {HFT, BN, BT<:AbstractGTBasisFuncs, NN}
    D₁ = zero(Hcore)
    D₂ = zero(Hcore)
    N₁tot = 0
    N₂tot = 0
    order = sortperm(collect(nuc), by=x->AtomicNumberList[x])
    for (atm, coord) in zip(nuc[order], nucCoords[order])
        N = getCharge(atm)
        N₁, N₂ = splitSpins(Val(:UHF), N)
        if N₂ > N₁ && N₂tot > N₁tot
            N₁, N₂ = N₂, N₁
        end
        h1 = coreH(bs, (atm,), (coord,))
        r, _ = runHFcore(Val(:UHF), 
                         config, (N₁, N₂), h1, HeeI, S, X, getCfromHcore(Val(:UHF), X, h1))
        D₁ += r[1].Ds[end]
        D₂ += r[2].Ds[end]
        N₁tot += N₁
        N₂tot += N₂
    end
    breakSymOfC(Val(HFT), X, D₁, D₂, Hcore, HeeI)
end


const guessCmethods = 
    (  GWH = (HFT, S, X, Hcore, _...)->getCfromGWH(HFT, S, Hcore, X), 
     Hcore = (HFT, S, X, Hcore, _...)->getCfromHcore(HFT, X, Hcore), 
       SAD = (HFT, S, X, Hcore, HeeI, bs, nuc, nucCoords)->
             getCfromSAD(HFT, S, Hcore, HeeI, bs, nuc, nucCoords, X))


@inline guessC(::Type{Val{M}}, ::Val{HFT}, S, X, Hcore, HeeI, bs, nuc, nucCoords) where 
       {M, HFT} = 
        getfield(guessCmethods, M)(Val(HFT), S, X, Hcore, HeeI, bs, nuc, nucCoords)

@inline guessC(Cs::Tuple{Vararg{Matrix}}, _...) = itself(Cs)


getD(C::Matrix{Float64}, Nˢ::Int) = @views (C[:,1:Nˢ]*C[:,1:Nˢ]')
# Nˢ: number of electrons with the same spin.

@inline getD(X::Matrix{Float64}, F::Matrix{Float64}, Nˢ::Int) = getD(getC(X, F), Nˢ)


function getGcore(HeeI::Array{Float64, 4}, DJ::Matrix{Float64}, DK::Matrix{Float64})
    G = zero(DJ)
    l = size(G)[1]
    for ν = 1:l, μ = 1:l # fastest
        G[μ, ν] = dot(transpose(DJ), @view HeeI[μ,ν,:,:]) - dot(DK, @view HeeI[μ,:,:,ν]) 
    end
    G |> Hermitian |> Array
end

# RHF
@inline getG(HeeI::Array{Float64, 4}, D::Matrix{Float64}) = getGcore(HeeI, 2D, D)

# UHF
@inline getG(HeeI::Array{Float64, 4}, D::Matrix{Float64}, Dᵀ::Matrix{Float64}) = 
        getGcore(HeeI, Dᵀ, D)


@inline getF(Hcore::Matrix{Float64}, G::Matrix{Float64}) = Hcore + G

# RHF
@inline getF(Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, D::Matrix{Float64}) = 
        getF(Hcore, getG(HeeI, D))

# UHF
@inline getF(Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
             D::Matrix{Float64}, Dᵀ::Matrix{Float64}) = 
        getF(Hcore, getG(HeeI, D, Dᵀ))

# RHF or UHF
@inline getF(Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
             Ds::NTuple{N, Matrix{Float64}}) where {N} = 
        getF(Hcore, getG(HeeI, Ds...))

@inline getE(Hcore::Matrix{Float64}, F::Matrix{Float64}, D::Matrix{Float64}) = 
        dot(transpose(D), 0.5*(Hcore + F))


# RHF
@inline getEᵀcore(Hcore::Matrix{Float64}, F::Matrix{Float64}, D::Matrix{Float64}) = 
        2*getE(Hcore, F, D)

# UHF
@inline getEᵀcore(Hcore::Matrix{Float64}, Fᵅ::Matrix{Float64},   Dᵅ::Matrix{Float64}, 
                  Fᵝ::Matrix{Float64},    Dᵝ::Matrix{Float64}) = 
        getE(Hcore, Fᵅ, Dᵅ) + getE(Hcore, Fᵝ, Dᵝ)

# RHF
function getEᵀ(Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
              (C,)::Tuple{Matrix{Float64}}, (N,)::Tuple{Int})
    D = getD(C, N÷2)
    F = getF(Hcore, HeeI, D)
    getEᵀcore(Hcore, F, D)
end

# UHF
function getEᵀ(Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
               (Cᵅ,Cᵝ)::NTuple{2, Matrix{Float64}}, (Nᵅ,Nᵝ)::NTuple{2, Int})
    Dᵅ = getD(Cᵅ, Nᵅ)
    Dᵝ = getD(Cᵝ, Nᵝ)
    Dᵀ = Dᵅ + Dᵝ
    Fᵅ = getF(Hcore, HeeI, Dᵅ, Dᵀ)
    Fᵝ = getF(Hcore, HeeI, Dᵝ, Dᵀ)
    getEᵀcore(Hcore, Fᵅ, Dᵅ, Fᵝ, Dᵝ)
end


@inline function getCFDE(Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
                         X::Matrix{Float64}, Ds::Vararg{Matrix{Float64}, N}) where {N}
    Fnew = getF(Hcore, HeeI, Ds)
    Enew = getE(Hcore, Fnew, Ds[1])
    Cnew = getC(X, Fnew)
    (Cnew, Fnew, Ds[1], Enew) # Fnew is based on latest variables.
end


# RHF
function initializeSCF(::Val{:RHF}, Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
                       (C,)::Tuple{Matrix{Float64}}, (N,)::Tuple{Int})
    Nˢ = N÷2
    D = getD(C, Nˢ)
    F = getF(Hcore, HeeI, D)
    E = getE(Hcore, F, D)
    (HFtempVars(Val(:RHF), Nˢ, C, F, D, E, 2D, 2E),)
end

# UHF
function initializeSCF(::Val{:UHF}, Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
                       Cs::NTuple{2, Matrix{Float64}}, Ns::NTuple{2, Int})
    Ds = getD.(Cs, Ns)
    Dᵀs = [Ds |> sum]
    Fs = getF.(Ref(Hcore), Ref(HeeI), Ds, Ref(Dᵀs[]))
    Es = getE.(Ref(Hcore), Fs, Ds)
    Eᵀs = [Es |> sum]
    res = HFtempVars.(Val(:UHF), Ns, Cs, Fs, Ds, Es)
    res[1].shared.Dtots = res[2].shared.Dtots = Dᵀs
    res[1].shared.Etots = res[2].shared.Etots = Eᵀs
    res
end


const Doc_SCFconfig_OneRowTable = "|`:DIIS`, `:EDIIS`, `:ADIIS`|subspace size; "*
                                  "coefficient solver|`DIISsize`; `solver`|`1`,`2`...; "*
                                  "`:LCM`, `:BFGS`|`15`; `:BFGS`|"

const Doc_SCFconfig_DIIS = "[Direct inversion in the iterative subspace]"*
                           "(https://onlinelibrary.wiley.com/doi/10.1002/jcc.540030413)."
const Doc_SCFconfig_ADIIS = "[DIIS based on the augmented Roothaan–Hall (ARH) energy "*
                            "function](https://aip.scitation.org/doi/10.1063/1.3304922)."
const Doc_SCFconfig_LBFGSB = "[Limited-memory BFGS with box constraints]"*
                             "(https://github.com/Gnimuc/LBFGSB.jl)."

const Doc_SCFconfig_Eg1 = "SCFconfig{3}(interval=(0.0001, 1.0e-12, 1.0e-13), "*
                          "oscillateThreshold=1.0e-5, method, methodConfig)"*
                          "[:DD, :ADIIS, :DIIS]"

"""

    SCFconfig{N} <: ImmutableParameter{SCFconfig, Any}

The `struct` for SCF iteration configurations.

≡≡≡ Field(s) ≡≡≡

`method::NTuple{N, Symbol}`: The applied methods. The available methods and their 
configurations (in terms of keyword arguments):

| Convergence Method(s) | Configuration(s) | Keyword(s) | Range(s)/Option(s) | Default(s) |
| :----                 | :---:            | :---:      | :---:              |      ----: |
| `:DD`                 | damping strength |`dampStrength`|    [`0`, `1`]    |      `0.0` |
$(Doc_SCFconfig_OneRowTable)

### Convergence Methods
* DD: Direct diagonalization of the Fock matrix.
* DIIS: $(Doc_SCFconfig_DIIS)
* EDIIS: [Energy-DIIS](https://aip.scitation.org/doi/abs/10.1063/1.1470195).
* ADIIS: $(Doc_SCFconfig_ADIIS)

### DIIS-type Method Solvers
* LCM: Lagrange multiplier solver.
* BFGS: $(Doc_SCFconfig_LBFGSB)

`interval::NTuple{N, Float64}`: The stopping (skipping) thresholds for required methods.

`methodConfig::NTuple{N, Vector{<:Pair}}`: The additional keywords arguments for each 
method stored as `Tuple`s of `Pair`s.

`oscillateThreshold::Float64`: The threshold for oscillating convergence.

≡≡≡ Initialization Method(s) ≡≡≡

    SCFconfig(methods::NTuple{N, Symbol}, intervals::NTuple{N, Float64}, 
              configs::Dict{Int, <:Vector{<:Pair}}=Dict(1=>Pair[]);
              oscillateThreshold::Float64=1e-5) where {N} -> 
    SCFconfig{N}

`methods` and `intervals` are the methods to be applied and their stopping (skipping) 
thresholds respectively; the length of those two `Vector`s should be the same. `configs` 
specifies the additional keyword arguments for each methods by a `Pair` of which the `Int` 
key `i` is for `i`th method and the pointed `Vector{<:Pair}` is the pairs of keyword 
arguments and their values respectively.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> SCFconfig((:DD, :ADIIS, :DIIS), (1e-4, 1e-12, 1e-13), Dict(2=>[:solver=>:LCM]))
$(Doc_SCFconfig_Eg1)
```
"""
struct SCFconfig{N} <: ImmutableParameter{SCFconfig, Any}
    method::NTuple{N, Symbol}
    interval::NTuple{N, Float64}
    methodConfig::NTuple{N, Vector{<:Pair}}
    oscillateThreshold::Float64

    function SCFconfig(methods::NTuple{N, Symbol}, intervals::NTuple{N, Float64}, 
                       configs::Dict{Int, <:Vector{<:Pair}}=Dict(1=>Pair[]);
                       oscillateThreshold::Float64=1e-5) where {N}
        kwPairs = [Pair[] for _=1:N]
        for i in keys(configs)
            kwPairs[i] = configs[i]
        end
        new{N}(methods, intervals, Tuple(kwPairs), oscillateThreshold)
    end
end


const defaultSCFconfig = SCFconfig((:ADIIS, :DIIS), (5e-3, 1e-16))


mutable struct HFinterrelatedVars <: HartreeFockintermediateData
    Dtots::Vector{Matrix{Float64}}
    Etots::Vector{Float64}

    HFinterrelatedVars() = new()
    HFinterrelatedVars(Dtots, Etots) = new(Dtots, Etots)
end


"""
    HFtempVars{HFT} <: HartreeFockintermediateData

The container to store the intermediate values (only of the same spin configuration) for 
each iteration during the Hartree-Fock SCF procedure. 

≡≡≡ Field(s) ≡≡≡

`N::Int`: The number of electrons with the same spin function.

`Cs::Array{Array{Float64, 2}, 1}`: Coefficient matrices.

`Fs::Array{Array{Float64, 2}, 1}`: Fock matrices

`Ds::Array{Array{Float64, 2}, 1}`: Density matrices corresponding 
to only spin configuration. For RHF each elements means (unconverged) 0.5*Dᵀ.

`Es::Vector{Float64}`: Part of Hartree-Fock energy corresponding to only one spin 
configuration. For RHF each element means (unconverged) 0.5*Ehf.

`shared.Dtots::Array{Array{Float64, 2}, 1}`: The total density 
matrices.

`shared.Etots::Vector{Float64}`: The total Hartree-Fock energy.

**NOTE: For UHF, there are 2 `HFtempVars` being updated during the SCF iterations, and 
change the field `shared.Dtots` or `shared.Etots` of one container will affect the other 
one's.**
"""
struct HFtempVars{HFT} <: HartreeFockintermediateData
    N::Int
    Cs::Vector{Matrix{Float64}}
    Fs::Vector{Matrix{Float64}}
    Ds::Vector{Matrix{Float64}}
    Es::Vector{Float64}
    shared::HFinterrelatedVars
end

HFtempVars(::Val{HFT}, 
           N::Int, C::Matrix{Float64}, F::Matrix{Float64}, 
           D::Matrix{Float64}, E::Real) where {HFT} = 
HFtempVars{HFT}(N, [C], [F], [D], [E], HFinterrelatedVars())

HFtempVars(::Val{HFT}, 
           N::Int, C::Matrix{Float64}, F::Matrix{Float64}, 
           D::Matrix{Float64}, E::Real, 
           Dtot::Matrix{Float64}, Etot::Real) where {HFT} = 
HFtempVars{HFT}(N, [C], [F], [D], [E], HFinterrelatedVars([Dtot], [Etot]))

HFtempVars(::Val{HFT}, 
           Nˢ::Int, Cs::Vector{Matrix{Float64}}, Fs::Vector{Matrix{Float64}}, 
           Ds::Vector{Matrix{Float64}}, Es::Vector{<:Real}, 
           Dtots::Vector{Matrix{Float64}}, Etots::Vector{<:Real}) where {HFT} = 
HFtempVars{HFT}(Nˢ, Cs, Fs, Ds, Es, HFinterrelatedVars(Dtots, Etots))


"""

    HFfinalVars{HFT, NN, HFTS} <: HartreeFockFinalValue{HFT}

The container of the final values after a Hartree-Fock SCF procedure.

≡≡≡ Field(s) ≡≡≡

`Ehf::Float64`: Hartree-Fock energy of the electronic Hamiltonian.

`Enn::Float64`: The nuclear repulsion energy.

`N::Int`: The total number of electrons.

`nuc::Tuple{NTuple{NN, String}}`: Nuclei of the system.

`nucCoords::Tuple{NTuple{NN, NTuple{3,Float64}}}`: Nuclei coordinates.

`C::NTuple{HFTS, Matrix{Float64}}`: Coefficient matrix(s) for one spin configuration.

`F::NTuple{HFTS, Matrix{Float64}}`: Fock matrix(s) for one spin configuration.

`D::NTuple{HFTS, Matrix{Float64}}`: Density matrix(s) for one spin configuration.

`Emo::NTuple{HFTS, Vector{Float64}}`: Energies of molecular orbitals.

`occu::NTuple{HFTS, Vector{Int}}`: occupation numbers of molecular orbitals.

`temp::NTuple{HFTS, HFtempVars{HFT}}`: the intermediate values.

`isConverged::Bool`: Whether the SCF procedure is converged in the end.
"""
struct HFfinalVars{HFT, NN, HFTS} <: HartreeFockFinalValue{HFT}
    Ehf::Float64
    Enn::Float64
    N::Int
    nuc::NTuple{NN, String}
    nucCoords::NTuple{NN, NTuple{3,Float64}}
    C::NTuple{HFTS, Matrix{Float64}}
    F::NTuple{HFTS, Matrix{Float64}}
    D::NTuple{HFTS, Matrix{Float64}}
    Emo::NTuple{HFTS, Vector{Float64}}
    occu::NTuple{HFTS, Vector{Int}}
    temp::NTuple{HFTS, HFtempVars{HFT}}
    isConverged::Bool

    function HFfinalVars(nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{3,Float64}}, 
                         X::Matrix{Float64}, (vars,)::Tuple{HFtempVars{:RHF}}, 
                         isConverged::Bool) where {NN}
        C = (vars.Cs[end],)
        F = vars.Fs[end]
        D = (vars.Ds[end],)
        Ehf = vars.shared.Etots[end]
        Emo = (getCϵ(X, F)[2],)
        N = vars.N
        occu = (vcat(2*ones(Int, N), zeros(Int, size(X, 1) - N)),)
        Enn = nnRepulsions(nuc, nucCoords)
        new{:RHF, NN, 1}(Ehf, Enn, 2N, nuc, nucCoords, C, (F,), D, Emo, occu, (vars,), 
                         isConverged)
    end

    function HFfinalVars(nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{3,Float64}}, 
                         X::Matrix{Float64}, αβVars::NTuple{2, HFtempVars{:UHF}}, 
                         isConverged::Bool) where {NN}
        C = last.(getfield.(αβVars, :Cs))
        F = last.(getfield.(αβVars, :Fs))
        D = last.(getfield.(αβVars, :Ds))
        Ehf = αβVars[1].shared.Etots[end]
        Emo = getindex.(getCϵ.(Ref(X), F), 2)
        Ns = getfield.(αβVars, :N)
        occu = vcat.(ones.(Int, Ns), zeros.(Int, size(X, 1) .- Ns))
        Enn = nnRepulsions(nuc, nucCoords)
        N = sum(Ns)
        new{:UHF, NN, 2}(Ehf, Enn, N, nuc, nucCoords, C, F, D, Emo, occu, αβVars, 
                         isConverged)
    end
end


const Doc_HFconfig_Eg1 = "HFconfig{:RHF, Val{:SAD}, 3}(Val{:RHF}(), Val{:SAD}(), "*
                         "SCFconfig{3}(interval=(0.0001, 1.0e-6, 1.0e-15), "*
                         "oscillateThreshold=1.0e-5, method, methodConfig)"*
                         "[:ADIIS, :DIIS, :ADIIS], 1000, true)"

const Doc_HFconfig_Eg2 = Doc_HFconfig_Eg1[1:10] * "U" * Doc_HFconfig_Eg1[12:34] * "U" * 
                         Doc_HFconfig_Eg1[36:end]

const HFtypes = (:RHF, :UHF)

"""

    HFconfig{HFT, CT, L} <: ConfigBox{HFconfig, HFT}

The container of Hartree-Fock method configuration.

≡≡≡ Field(s) ≡≡≡

`HF::Val{HFT}`: Hartree-Fock method type. Available values of `HFT` are 
$(string(HFtypes)[2:end-1]).

`C0::CT`: Initial guess of the coefficient matrix(s) C of the molecular orbitals. When `C0` 
is a `Val{T}`, the available values of `T` are 
`$((guessCmethods|>typeof|>fieldnames|>string)[2:end-1])`.

`SCF::SCFconfig`: SCF iteration configuration. For more information please refer to 
`SCFconfig`.

`earlyStop::Bool`: Whether automatically terminate (skip) a convergence method early when 
its performance becomes unstable or poor.

`maxStep::Int`: Maximum allowed iteration steps regardless of whether the SCF converges.

≡≡≡ Initialization Method(s) ≡≡≡

    HFconfig(;kws...) -> HFconfig

    HFconfig(t::NamedTuple) -> HFconfig

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> HFconfig()
$(Doc_HFconfig_Eg1)

julia> HFconfig(HF=:UHF)
$(Doc_HFconfig_Eg2)
```
"""
mutable struct HFconfig{HFT, CT, L} <: ConfigBox{HFconfig, HFT}
    HF::Val{HFT}
    C0::CT
    SCF::SCFconfig{L}
    maxStep::Int
    earlyStop::Bool

    HFconfig(::Val{:UHF}, a2::NTuple{2, Matrix{Float64}}, a3::SCFconfig{L}, a4, a5) where 
            {L} = 
    new{:UHF, NTuple{2, Matrix{Float64}}, L}(Val(:UHF), a2, a3, a4, a5)

    HFconfig(::Val{:RHF}, a2::Matrix{Float64}, a3::SCFconfig{L}, a4, a5) where {L} = 
    new{:RHF, Tuple{Matrix{Float64}}, L}(Val(:RHF), (a2,), a3, a4, a5)

    HFconfig(::Val{HFT}, a2::Val{CT}, a3::SCFconfig{L}, a4, a5) where {HFT, CT, L} = 
    new{HFT, Val{CT}, L}(Val(HFT), a2, a3, a4, a5)
end

HFconfig(a1::Symbol, a2, args...) = HFconfig(Val(a1), a2, args...)

HFconfig(a1, a2::Symbol, args...) = HFconfig(a1, Val(a2), args...)

HFconfig(a1::Symbol, a2::Symbol, args...) = HFconfig(Val(a1), Val(a2), args...)

const defaultHFconfigPars = Any[Val(:RHF), Val(:SAD), defaultSCFconfig, 1000, true]

HFconfig(t::NamedTuple) = genNamedTupleC(:HFconfig, defaultHFconfigPars)(t)

HFconfig(;kws...) = 
length(kws) == 0 ? HFconfig(defaultHFconfigPars...) : HFconfig(kws|>NamedTuple)

const defaultHFconfig = HFconfig()

const defaultHFconfigStr = "HFconfig()"


"""
    runHF(bs::Union{BasisSetData, Vector{<:AbstractGTBasisFuncs}, 
                    Tuple{Vararg{AbstractGTBasisFuncs}}}, 
          nuc::Union{NTuple{NN, String}, Vector{String}}, 
          nucCoords::Union{NTuple{NN, NTuple{3,Float64}}, 
                           Vector{<:AbstractArray{<:Real}}}, 
          config::HFconfig{HFT, CT, L}=$(defaultHFconfigStr), 
          N::Int=getCharge(nuc); 
          printInfo::Bool=true) where {HFT} -> 
    HFfinalVars{HFT}

Main function to run Hartree-Fock in Quiqbox.

=== Positional argument(s) ===

`bs::Union{BasisSetData, Vector{<:AbstractGTBasisFuncs}, 
           Tuple{Vararg{AbstractGTBasisFuncs}}}`: Basis set.

`nuc::Union{NTuple{NN, String}, Vector{String}}`: The element symbols of the nuclei for the 
studied system.

`nucCoords::Union{NTuple{NN, NTuple{3,Float64}}, Vector{<:AbstractArray{<:Real}}}`: Nuclei 
coordinates.

`config::HFconfig`: The Configuration of selected Hartree-Fock method. For more information 
please refer to `HFconfig`.

`N::Int`: Total number of electrons.

=== Keyword argument(s) ===

`printInfo::Bool`: Whether print out the information of iteration steps.
"""
function runHF(bs::GTBasis{BN, BT}, 
               nuc::Union{NTuple{NN, String}, Vector{String}}, 
               nucCoords::Union{NTuple{NN, NTuple{3,Float64}}, 
                                Vector{<:AbstractArray{<:Real}}}, 
               config::HFconfig{HFT, CT, L}=defaultHFconfig, 
               N::Int=getCharge(nuc); 
               printInfo::Bool=true) where {BN, BT, NN, HFT, CT, L}
    nuc = arrayToTuple(nuc)
    nucCoords = genTupleCoords(nucCoords)
    leastNb = ceil(N/2)
    @assert BN >= leastNb "The number of basis functions should be no less than $(leastNb)."
    @assert N > (HFT==:RHF) "$(HFT) requires more than $(HFT==:RHF) electrons."
    Ns = splitSpins(Val(HFT), N)
    Hcore = bs.getHcore(nuc, nucCoords)
    X = getX(bs.S)
    C0s = guessC(CT, Val(HFT), bs.S, X, Hcore, bs.eeI, bs.basis, nuc, nucCoords)
    vars, isConverged = runHFcore(Val(HFT), config.SCF, Ns, Hcore, bs.eeI, bs.S, X, 
                                  C0s, printInfo, config.maxStep, config.earlyStop)
    res = HFfinalVars(nuc, nucCoords, X, vars, isConverged)
    if printInfo
        Etot = round(res.Ehf + res.Enn, digits=10)
        Ehf = round(res.Ehf, digits=10)
        Enn = round(res.Enn, digits=10)
        println(rpad("Hartree-Fock Energy", 20), "| ", rpad("Nuclear Repulsion", 20), 
                "| Total Energy")
        println(rpad(string(Ehf)* " Ha", 22), rpad(string(Enn)* " Ha", 22), Etot, " Ha\n")
    end
    res
end

"""

    runHF(bs::Union{BasisSetData, Vector{<:AbstractGTBasisFuncs}, 
                    Tuple{Vararg{AbstractGTBasisFuncs}}}, 
          nuc::Union{NTuple{NN, String}, Vector{String}}, 
          nucCoords::Union{NTuple{NN, NTuple{3,Float64}}, 
                           Vector{<:AbstractArray{<:Real}}},
          N::Int=getCharge(nuc), 
          config::HFconfig{HFT, CT, L}=$(defaultHFconfigStr); 
          printInfo::Bool=true) where {HFT} -> 
    HFfinalVars{HFT}
"""
runHF(bs::BasisSetData, nuc, nucCoords, N::Int, config=defaultHFconfig; printInfo=true) = 
runHF(bs::BasisSetData, nuc, nucCoords, config, N; printInfo)

runHF(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs}}, Vector{<:AbstractGTBasisFuncs}}, 
      args...; printInfo=true) = 
runHF(GTBasis(bs), args...; printInfo)


"""

    runHFcore(HTtype::Val{HFT}, 
              scfConfig::SCFconfig, 
              N::NTuple{HFTS, Int}, 
              Hcore::Array{Float64, 2}, 
              HeeI::Array{Float64, 4}, 
              S::Array{Float64, 2}, 
              X::Array{Float64, 2}, 
              C0::NTuple{HFTS, Matrix{Float64}}, 
              printInfo::Bool=false, 
              maxStep::Int=1000, 
              earlyStop::Bool=true) where {HFT, HFTS} -> 
    Tuple{Vararg{HFtempVars}}, Bool

The core function of `runHF` which returns the data collected during the iteration and the 
result of whether the SCF procedure is converged.

=== Positional argument(s) ===

`HTtype::Val{HFT}`: Hartree-Fock method type. Available values of `HFT` are 
$(string(HFtypes)[2:end-1]).

`scfConfig::SCFconfig`: SCF iteration configuration.

`N::Union{NTuple{2, Int}, Int}`: The total number of electrons or the numbers of electrons 
with different spins respectively. When the latter is input, an UHF is performed.

`Hcore::Array{Float64, 2}`: Core Hamiltonian of electronic Hamiltonian.

`HeeI::Array{Float64, 4}`: The electron-electron interaction Hamiltonian which includes both the 
Coulomb interactions and the Exchange Correlations.

`S::Array{Float64, 2}`: Overlap matrix of the corresponding basis set.

`X::Array{Float64, 2}`: Orthogonal transformation matrix of S. Default value is S^(-0.5).

`C0::Union{Array{Float64, 2}, NTuple{2, Array{Float64, 2}}}`: Initial guess of the 
coefficient matrix(s) C of the molecular orbitals.

`printInfo::Bool`: Whether print out the information of iteration steps.

`maxStep::Int`: Maximum allowed iteration steps regardless of whether the SCF converges.

`earlyStop::Bool`: Whether automatically early terminate (skip) a convergence method 
when its performance becomes unstable or poor.
"""
function runHFcore(::Val{HFT}, 
                   scfConfig::SCFconfig{L}, 
                   N::NTuple{HFTS, Int}, 
                   Hcore::Matrix{Float64}, 
                   HeeI::Array{Float64, 4}, 
                   S::Matrix{Float64}, 
                   X::Matrix{Float64}, 
                   C0::NTuple{HFTS, Matrix{Float64}}, 
                   printInfo::Bool=false, 
                   maxStep::Int=1000, 
                   earlyStop::Bool=true) where {HFT, L, HFTS}
    vars = initializeSCF(Val(HFT), Hcore, HeeI, C0, N)
    Etots = vars[1].shared.Etots
    printInfo && println(rpad(HFT, 4)*rpad(" | Initial Gauss", 18), "E = $(Etots[end])")
    isConverged = true
    EtotMin = Etots[]
    i = 0
    for (m, kws, breakPoint::Float64, l) in 
        zip(scfConfig.method, scfConfig.methodConfig, scfConfig.interval, 1:L)

        while true
            i += 1
            i <= maxStep || (isConverged = false) || break

            res = HFcore(m, N, Hcore, HeeI, S, X, vars; kws...)
            pushHFtempVars!(vars, res)

            printInfo && (i % floor(log(4, i) + 1) == 0 || i == maxStep) && 
            println(rpad("Step $i", 10), rpad("#$l ($(m))", 12), "E = $(Etots[end])")

            abs(Etots[end]-Etots[end-1]) > breakPoint || (isConverged = true) && break

            flag, Std = isOscillateConverged(Etots, 10^(log(10, breakPoint)÷2))

            flag && (isConverged = Std > scfConfig.oscillateThreshold ? false : true; break)

            if earlyStop && (Etots[end] - EtotMin) / abs(EtotMin) > 0.2
                printInfo && println("Early termination of ", m, 
                                     " due to the poor performance.")
                isConverged = false
                break
            end
        end

    end
    negStr = isConverged ? "is " : "has not "
    printInfo && println("The SCF procedure ", negStr, "converged.\n")
    vars, isConverged
end


function DDcore(Nˢ::Int, Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
                X::Matrix{Float64}, F::Matrix{Float64}, D::Matrix{Float64}; 
                dampStrength::Float64=0.0, _kws...)
    @assert 0 <= dampStrength <= 1 "The range of `dampStrength`::Float64 is [0,1]."
    Dnew = getD(X, F, Nˢ)
    (1-dampStrength)*Dnew + dampStrength*D
end


function xDIIScore(::Val{M}, Nˢ::Int, Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
                   S::Matrix{Float64}, X::Matrix{Float64}, Fs::Vector{Matrix{Float64}}, 
                   Ds::Vector{Matrix{Float64}},Es::Vector{Float64}; 
                   DIISsize::Int=10, solver::Symbol=:BFGS, 
                   _kws...) where {M}
    DIISmethod, convexConstraint, permuteData = getfield(DIISmethods, M)
    is = permuteData ? sortperm(Es) : (:)
    ∇s = (@view Fs[is])[1:end .> end-DIISsize]
    Ds = (@view Ds[is])[1:end .> end-DIISsize]
    Es = (@view Es[is])[1:end .> end-DIISsize]
    v, B = DIISmethod(∇s, Ds, Es, S)
    c = constraintSolver(v, B, convexConstraint, solver)
    grad = c.*∇s |> sum
    getD(X, grad |> Hermitian |> Array, Nˢ) # grad == F.
end

const DIISmethods = ( DIIS = ((∇s, Ds, _ , S)-> DIIScore(∇s, Ds, S ), false, true ),
                     EDIIS = ((∇s, Ds, Es, _)->EDIIScore(∇s, Ds, Es), true , false),
                     ADIIS = ((∇s, Ds, _ , _)->ADIIScore(∇s, Ds    ), true , false))


function EDIIScore(∇s::Vector{Matrix{Float64}}, Ds::Vector{Matrix{Float64}}, 
                   Es::Vector{Float64})
    len = length(Ds)
    B = ones(len, len)
    for j=1:len, i=1:len
        B[i,j] = -dot(Ds[i]-Ds[j], ∇s[i]-∇s[j])
    end
    Es, B
end


function ADIIScore(∇s::Vector{Matrix{Float64}}, Ds::Vector{Matrix{Float64}})
    len = length(Ds)
    B = ones(len, len)
    v = [dot(D - Ds[end], ∇s[end]) for D in Ds]
    for j=1:len, i=1:len
        B[i,j] = dot(Ds[i]-Ds[len], ∇s[j]-∇s[len])
    end
    v, B
end


function DIIScore(∇s::Vector{Matrix{Float64}}, Ds::Vector{Matrix{Float64}}, 
                  S::Matrix{Float64})
    len = length(Ds)
    B = ones(len, len)
    v = zeros(len)
    for j=1:len, i=1:len
        B[i,j] = dot(∇s[i]*Ds[i]*S - S*Ds[i]*∇s[i], ∇s[j]*Ds[j]*S - S*Ds[j]*∇s[j])
    end
    v, B
end


@inline DD(Nˢ, Hcore, HeeI, _dm::Any, X, tVars; kws...) = 
        DDcore(Nˢ, Hcore, HeeI, X, tVars.Fs[end], tVars.Ds[end]; kws...)

@inline function xDIIS(::Val{M}) where {M}
    @inline (Nˢ, Hcore, HeeI, S, X, tVars; kws...) ->
            xDIIScore(Val(M), Nˢ, Hcore, HeeI, S, X, tVars.Fs, tVars.Ds, tVars.Es; kws...)
end

const SCFmethodSelector = 
      (DD=DD, DIIS=xDIIS(Val(:DIIS)), ADIIS=xDIIS(Val(:ADIIS)), EDIIS=xDIIS(Val(:EDIIS)))


# RHF
@inline function HFcore(m::Symbol, (N,)::Tuple{Int}, 
                        Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
                        S::Matrix{Float64}, X::Matrix{Float64}, 
                        (rVars,)::Tuple{HFtempVars{:RHF}}; kws...)
    D = getfield(SCFmethodSelector, m)(N÷2, Hcore, HeeI, S, X, rVars; kws...)
    partRes = getCFDE(Hcore, HeeI, X, D)
    partRes..., 2D, 2partRes[end]
end

@inline function pushHFtempVars!((rVars,)::Tuple{HFtempVars}, 
                                 res::Tuple{Matrix{Float64}, Matrix{Float64}, 
                                            Matrix{Float64}, Float64, 
                                            Matrix{Float64}, Float64})
    push!(rVars.Cs, res[1])
    push!(rVars.Fs, res[2])
    push!(rVars.Ds, res[3])
    push!(rVars.Es, res[4])
    push!(rVars.shared.Dtots, res[5])
    push!(rVars.shared.Etots, res[6])
end

# UHF
@inline function HFcore(m::Symbol, Ns::NTuple{2, Int}, 
                        Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
                        S::Matrix{Float64}, X::Matrix{Float64}, 
                        uVars::NTuple{2, HFtempVars{:UHF}}; kws...)
    Ds = getfield(SCFmethodSelector, m).(Ns, Ref(Hcore), Ref(HeeI), Ref(S), Ref(X), uVars; 
                                         kws...)
    Dᵀnew = Ds |> sum
    partRes = getCFDE.(Ref(Hcore), Ref(HeeI), Ref(X), Ds, Ref(Dᵀnew))
    Eᵀnew = partRes[1][end] + partRes[2][end]
    (partRes[1]..., Dᵀnew, Eᵀnew), (partRes[2]..., Dᵀnew, Eᵀnew)
end

@inline function pushHFtempVars!((αVars, βVars)::NTuple{2, HFtempVars{:UHF}}, 
                                 res::NTuple{2, Tuple{Matrix{Float64}, Matrix{Float64}, 
                                                      Matrix{Float64}, Float64, 
                                                      Matrix{Float64}, Float64}})
    pushHFtempVars!((αVars,), res[1])
    pushHFtempVars!((βVars,), res[2])
    pop!(αVars.shared.Dtots)
    pop!(αVars.shared.Etots)
end


@inline function popHFtempVars!(rVars::HFtempVars)
    pop!(rVars.Cs)
    pop!(rVars.Fs)
    pop!(rVars.Ds)
    pop!(rVars.Es)
    pop!(rVars.shared.Dtots)
    pop!(rVars.shared.Etots)
end

@inline popHFtempVars!((rVars,)::Tuple{HFtempVars}) = popHFtempVars!(rVars)

@inline function popHFtempVars!(uVars::NTuple{2, HFtempVars{:UHF}})
    for field in [:Cs, :Fs, :Ds, :Es] pop!.(getfield.(uVars, field)) end
    pop!(uVars[1].shared.Dtots)
    pop!(uVars[1].shared.Etots)
end


# Included normalization condition, but not non-negative condition.
@inline function genxDIISf(v, B)
    function (c)
        s = sum(c)
        dot(v, c) / s + 0.5 * transpose(c) * B * c / s^2
    end
end

@inline function genxDIIS∇f(v, B)
    function (g, c)
        s = sum(c)
        g.= v./c + (B + transpose(B))*c ./ (2*s^2) .- (dot(v, c)/s^2 + transpose(c)*B*c/s^3)
    end
end


# Default
function LBFGSBsolver(v::Vector{Float64}, B::Matrix{Float64}, convexConstraint::Bool)
    f = genxDIISf(v, B)
    g! = genxDIIS∇f(v, B)
    len = length(v)
    lb = convexConstraint ? 0.0 : -Inf
    _, c = lbfgsb(f, g!, fill(1e-2, len); lb, 
                  m=15, factr=1, pgtol=1e-8, maxfun=20000, maxiter=20000)
    c ./ sum(c)
end


function CMsolver(v::Vector, B::Matrix, convexConstraint::Bool, ϵ::Float64=1e-5)
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
        (findfirst(x->x<0, c) !== nothing && convexConstraint) || (return c)
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


const ConstraintSolvers = (LCM=CMsolver, BFGS=LBFGSBsolver)

constraintSolver(v::Vector{Float64}, B::Matrix{Float64}, convexConstraint::Bool, 
                 solver::Symbol) = 
getfield(ConstraintSolvers, solver)(v, B, convexConstraint)