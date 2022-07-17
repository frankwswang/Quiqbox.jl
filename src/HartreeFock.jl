export SCFconfig, HFconfig, runHF, runHFcore

using LinearAlgebra: dot, Hermitian, \, det, I, ishermitian
using PiecewiseQuadratics: indicator
using Combinatorics: powerset
using LineSearches
using Optim: LBFGS, Fminbox, optimize as OptimOptimize, minimizer as OptimMinimizer, 
             Options as OptimOptions

getXcore1(S::Matrix{T}) where {T<:Real} = Hermitian(S)^(-T(0.5)) |> Array

precompile(getXcore1, (Matrix{Float64},))

const getXmethods = (m1=getXcore1,)

getX(S::Matrix{T}, method::Symbol=:m1) where {T<:Real} = getproperty(getXmethods, method)(S)


function getCϵ(X::Matrix{T}, Fˢ::Matrix{T}, stabilizeSign::Bool=true) where {T<:Real}
    ϵ, Cₓ = eigen(X'*Fˢ*X |> Hermitian)
    outC = X*Cₓ
    # Stabilize the sign factor of each column.
    stabilizeSign && for j = 1:size(outC, 2)
       outC[:, j] *= ifelse(outC[1,j] < 0, -1, 1)
    end
    outC, ϵ
end

@inline getC(X::Matrix{T}, Fˢ::Matrix{T}, stabilizeSign::Bool=true) where {T<:Real} = 
        getCϵ(X, Fˢ, stabilizeSign)[1]


splitSpins(::Val{1}, N::Int) = (N÷2,)

splitSpins(::Val{2}, N::Int) = (N÷2, N-N÷2)

splitSpins(::Val{:RHF}, N::Int) = splitSpins(Val(1), N)

splitSpins(::Val{:UHF}, N::Int) = splitSpins(Val(2), N)

splitSpins(::Val, Ns::Tuple) = itself(Ns)


function breakSymOfC(::Val{:UHF}, C::Matrix{T}) where {T<:Real}
    C2 = copy(C)
    l = min(size(C2)[1], 2)
    C2[1:l, 1:l] .= 0 # Breaking spin symmetry.
    # C2[l, :] .= 0 # Another way.
    (copy(C), C2)
end

breakSymOfC(::Val{:RHF}, C::Matrix{T}) where {T<:Real} = (C,)

breakSymOfC(::Val{:RHF}, Hcore, HeeI, X, Dᵅ, Dᵝ, Nᵅ, Nᵝ) = 
getC.( Ref(X), getF(Hcore, HeeI, ((Nᵅ*Dᵅ + Nᵝ*Dᵝ)./(Nᵅ+Nᵝ),)) )

breakSymOfC(::Val{:UHF}, Hcore, HeeI, X, Dᵅ, Dᵝ, _, _) =
getC.( Ref(X), getF(Hcore, HeeI, (Dᵅ, Dᵝ)) )


function getCfromGWH(::Val{HFT}, S::Matrix{T}, Hcore::Matrix{T}, X::Matrix{T}) where 
                    {HFT, T<:Real}
    l = size(Hcore)[1]
    H = zero(Hcore)
    for j in 1:l, i in 1:l
        H[i,j] = 3 * S[i,j] * (Hcore[i,i] + Hcore[j,j]) / 8
    end
    Cˢ = getC(X, H)
    breakSymOfC(Val(HFT), Cˢ)
end


function getCfromHcore(::Val{HFT}, X::Matrix{T}, Hcore::Matrix{T}) where {HFT, T}
    Cˢ = getC(X, Hcore)
    breakSymOfC(Val(HFT), Cˢ)
end


function getCfromSAD(::Val{HFT}, S::Matrix{T}, 
                     Hcore::Matrix{T}, HeeI::Array{T, 4},
                     bs::NTuple{BN, AbstractGTBasisFuncs{T, D}}, 
                     nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{D, T}}, 
                     X::Matrix{T}, 
                     config=SCFconfig((:ADIIS,), (1e4*getAtolDigits(T),))) where 
                    {HFT, T, D, BN, NN}
    Dᵅ = zero(Hcore)
    Dᵝ = zero(Hcore)
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
        Dᵅ += r[1].Ds[end]
        Dᵝ += r[2].Ds[end]
        N₁tot += N₁
        N₂tot += N₂
    end
    breakSymOfC(Val(HFT), Hcore, HeeI, X, Dᵅ, Dᵝ, N₁tot, N₂tot)
end


const guessCmethods = (GWH=getCfromGWH, Hcore=getCfromHcore, SAD=getCfromSAD)


getD(Cˢ::Matrix{T}, Nˢ::Int) where {T} = @views (Cˢ[:,1:Nˢ]*Cˢ[:,1:Nˢ]')
# Nˢ: number of electrons with the same spin.

@inline getD(X::Matrix{T}, Fˢ::Matrix{T}, Nˢ::Int) where {T<:Real} = getD(getC(X, Fˢ), Nˢ)


function getGcore(HeeI::Array{T, 4}, DJ::Matrix{T}, DK::Matrix{T}) where {T<:Real}
    G = zero(DJ)
    l = size(G)[1]
    for ν = 1:l, μ = 1:l # fastest
        G[μ, ν] = dot(transpose(DJ), @view HeeI[μ,ν,:,:]) - dot(DK, @view HeeI[μ,:,:,ν]) 
    end
    G |> Hermitian |> Array
end

# RHF
@inline getG(HeeI::Array{T, 4}, (Dˢ,)::Tuple{Matrix{T}}) where {T<:Real} = 
        ( getGcore(HeeI, 2Dˢ, Dˢ), )

# UHF
@inline getG(HeeI::Array{T, 4}, (Dᵅ, Dᵝ)::NTuple{2, Matrix{T}}) where {T<:Real} = 
        ( getGcore(HeeI, Dᵅ+Dᵝ, Dᵅ), getGcore(HeeI, Dᵅ+Dᵝ, Dᵝ) )


@inline getF(Hcore::Matrix{T}, G::NTuple{HFTS, Matrix{T}}) where {T<:Real, HFTS} = 
        Ref(Hcore) .+ G

@inline getF(Hcore::Matrix{T}, HeeI::Array{T, 4}, D::NTuple{HFTS, Matrix{T}}) where 
            {T<:Real, HFTS} = 
        getF(Hcore, getG(HeeI, D))


# RHF or UHF
@inline getE(Hcore::Matrix{T}, Fˢ::Matrix{T}, Dˢ::Matrix{T}) where {T<:Real} = 
        dot(transpose(Dˢ), Hcore+Fˢ) / 2

get2SpinQuantity(O::NTuple{HFTS, T}) where {HFTS, T} = abs(3-HFTS) * sum(O)
get2SpinQuantities(O, nRepeat::Int) = fill(get2SpinQuantity(O), nRepeat) |> Tuple

# RHF or UHF
getEᵀcore(Hcore::Matrix{T}, Fˢ::NTuple{HFTS, Matrix{T}}, Dˢ::NTuple{HFTS, Matrix{T}}) where 
         {T<:Real, HFTS} = 
get2SpinQuantity(getE.(Ref(Hcore), Fˢ, Dˢ))

# RHF or UHF
function getEᵀ(Hcore::Matrix{T}, HeeI::Array{T, 4}, 
               C::NTuple{HFTS, Matrix{T}}, N::NTuple{HFTS, Int}) where {T<:Real, HFTS}
    D = getD.(C, N)
    F = getF(Hcore, HeeI, D)
    getEᵀcore(Hcore, F, D)
end


function getCFDE(Hcore::Matrix{T}, HeeI::Array{T, 4}, X::Matrix{T}, 
                 N::NTuple{HFTS, Int}, F::NTuple{HFTS, Matrix{T}}) where {T, HFTS}
    Cnew = getC.(Ref(X), F)
    Dnew = getD.(Cnew, N)
    Fnew = getF(Hcore, HeeI, Dnew)
    Enew = getE.(Ref(Hcore), Fnew, Dnew)
    Dᵀnew = get2SpinQuantities(Dnew, HFTS)
    Eᵀnew = get2SpinQuantities(Enew, HFTS)
    map(themselves, Cnew, Fnew, Dnew, Enew, Dᵀnew, Eᵀnew)
end


function initializeSCF(::Val{HFT}, Hcore::Matrix{T}, HeeI::Array{T, 4}, 
                       C::NTuple{HFTS, Matrix{T}}, N::NTuple{HFTS, Int}) where 
                      {HFT, T<:Real, HFTS}
    D = getD.(C, N)
    F = getF(Hcore, HeeI, D)
    E = getE.(Ref(Hcore), F, D)
    res = HFtempVars.(Val(HFT), N, C, F, D, E)
    sharedFields = getproperty.(res, :shared)
    for (field, val) in zip( (:Dtots, :Etots), fill.(get2SpinQuantity.((D, E)), 1)  )
        setproperty!.(sharedFields, field, Ref(val))
    end
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

const Doc_SCFconfig_Eg1 = "SCFconfig{Float64, 3}(method=(:DD, :ADIIS, :DIIS), "*
                          "interval=(0.0001, 1.0e-12, 1.0e-13), methodConfig, "*
                          "oscillateThreshold)"

"""

    SCFconfig{T, L} <: ImmutableParameter{T, SCFconfig}

The `struct` for SCF iteration configurations.

≡≡≡ Field(s) ≡≡≡

`method::NTuple{L, Symbol}`: The applied methods. The available methods and their 
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

`interval::NTuple{L, T}`: The stopping (skipping) thresholds for required methods.

`methodConfig::NTuple{L, Vector{<:Pair}}`: The additional keywords arguments for each 
method stored as `Tuple`s of `Pair`s.

`oscillateThreshold::T`: The threshold for oscillating convergence.

≡≡≡ Initialization Method(s) ≡≡≡

    SCFconfig(methods::NTuple{L, Symbol}, intervals::NTuple{L, T}, 
              configs::Dict{Int, <:AbstractVector{<:Pair}}=Dict(1=>Pair[]);
              oscillateThreshold::Real=1e-5) where {L, T} -> 
    SCFconfig{T, L}

`methods` and `intervals` are the methods to be applied and their stopping (skipping) 
thresholds respectively; the length of those two `AbstractVector`s should be the same. 
`configs` specifies the additional keyword arguments for each methods by a `Pair` of which 
the `Int` key `i` is for `i`th method and the pointed `AbstractVector{<:Pair}` is the pairs 
of keyword arguments and their values respectively.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> SCFconfig((:DD, :ADIIS, :DIIS), (1e-4, 1e-12, 1e-13), Dict(2=>[:solver=>:LCM]))
$(Doc_SCFconfig_Eg1)
```
"""
struct SCFconfig{T, L} <: ImmutableParameter{T, SCFconfig}
    method::NTuple{L, Symbol}
    interval::NTuple{L, T}
    methodConfig::NTuple{L, Vector{<:Pair}}
    oscillateThreshold::T

    function SCFconfig(methods::NTuple{L, Symbol}, intervals::NTuple{L, T}, 
                       configs::Dict{Int, <:AbstractVector{<:Pair}}=Dict(1=>Pair[]);
                       oscillateThreshold::Real=1e-5) where {L, T}
        kwPairs = [Pair[] for _=1:L]
        for i in keys(configs)
            kwPairs[i] = configs[i]
        end
        new{T, L}(methods, intervals, Tuple(kwPairs), oscillateThreshold)
    end
end


const defaultSCFconfig = SCFconfig((:ADIIS, :DIIS), (5e-3, 2e-16))


mutable struct HFinterrelatedVars{T} <: HartreeFockintermediateData{T}
    Dtots::Vector{Matrix{T}}
    Etots::Vector{T}

    HFinterrelatedVars{T}() where {T} = new{T}()
    HFinterrelatedVars(Dts::Vector{Matrix{T}}, Ets::Vector{T}) where {T} = new{T}(Dts, Ets)
end


"""
    HFtempVars{T, HFT} <: HartreeFockintermediateData{T}

The container to store the intermediate values (only of the same spin configuration) for 
each iteration during the Hartree-Fock SCF procedure. 

≡≡≡ Field(s) ≡≡≡

`N::Int`: The number of electrons with the same spin function.

`Cs::Vector{Matrix{T}}`: Coefficient matrices.

`Fs::Vector{Matrix{T}}`: Fock matrices

`Ds::Vector{Matrix{T}}`: Density matrices corresponding 
to only spin configuration. For RHF each elements means (unconverged) 0.5*Dᵀ.

`Es::Vector{T}`: Part of Hartree-Fock energy corresponding to only one spin 
configuration. For RHF each element means (unconverged) 0.5*Ehf.

`shared.Dtots::Vector{Matrix{T}}`: The total density 
matrices.

`shared.Etots::Vector{T}`: The total Hartree-Fock energy.

**NOTE: For UHF, there are 2 `HFtempVars` being updated during the SCF iterations, and 
change the field `shared.Dtots` or `shared.Etots` of one container will affect the other 
one's.**
"""
struct HFtempVars{T, HFT} <: HartreeFockintermediateData{T}
    N::Int
    Cs::Vector{Matrix{T}}
    Fs::Vector{Matrix{T}}
    Ds::Vector{Matrix{T}}
    Es::Vector{T}
    shared::HFinterrelatedVars{T}
end

HFtempVars(::Val{HFT}, N::Int, C::Matrix{T}, F::Matrix{T}, D::Matrix{T}, E::T) where 
          {HFT, T} = 
HFtempVars{T, HFT}(N, [C], [F], [D], [E], HFinterrelatedVars{T}())

HFtempVars(::Val{HFT}, N::Int, C::Matrix{T}, F::Matrix{T}, D::Matrix{T}, E::T, 
           Dtot::Matrix{T}, Etot::T) where {HFT, T} = 
HFtempVars{T, HFT}(N, [C], [F], [D], [E], HFinterrelatedVars([Dtot], [Etot]))

HFtempVars(::Val{HFT}, Nˢ::Int, Cs::Vector{Matrix{T}}, Fs::Vector{Matrix{T}}, 
           Ds::Vector{Matrix{T}}, Es::Vector{T}, Dtots::Vector{Matrix{T}}, 
           Etots::Vector{T}) where {HFT, T} = 
HFtempVars{T, HFT}(Nˢ, Cs, Fs, Ds, Es, HFinterrelatedVars(Dtots, Etots))


"""

    HFfinalVars{T, D, HFT, NN, BN, HFTS} <: HartreeFockFinalValue{T, HFT}

The container of the final values after a Hartree-Fock SCF procedure.

≡≡≡ Field(s) ≡≡≡

`Ehf::T`: Hartree-Fock energy of the electronic Hamiltonian.

`Enn::T`: The nuclear repulsion energy.

`N::NTuple{2, Int}`: The numbers of two different spins respectively.

`nuc::Tuple{NTuple{NN, String}}`: Nuclei of the system.

`nucCoords::Tuple{NTuple{NN, NTuple{D, T}}}`: Nuclei coordinates.

`C::NTuple{HFTS, Matrix{T}}`: Coefficient matrix(s) for one spin configuration.

`F::NTuple{HFTS, Matrix{T}}`: Fock matrix(s) for one spin configuration.

`D::NTuple{HFTS, Matrix{T}}`: Density matrix(s) for one spin configuration.

`Eo::NTuple{HFTS, Vector{T}}`: Energies of canonical orbitals.

`occu::NTuple{HFTS, NTuple{BN, String}}`: Spin occupations of canonical orbitals.

`temp::NTuple{HFTS, HFtempVars{T, HFT}}`: the intermediate values.

`isConverged::Bool`: Whether the SCF procedure is converged in the end.
"""
struct HFfinalVars{T, D, HFT, NN, BN, HFTS} <: HartreeFockFinalValue{T, HFT}
    Ehf::T
    Enn::T
    N::NTuple{2, Int}
    nuc::NTuple{NN, String}
    nucCoord::NTuple{NN, NTuple{D, T}}
    C::NTuple{HFTS, Matrix{T}}
    F::NTuple{HFTS, Matrix{T}}
    D::NTuple{HFTS, Matrix{T}}
    Eo::NTuple{HFTS, Vector{T}}
    occu::NTuple{HFTS, NTuple{BN, String}}
    temp::NTuple{HFTS, HFtempVars{T, HFT}}
    isConverged::Bool
    basis::GTBasis{T, D, BN}

    function HFfinalVars(basis::GTBasis{T, 𝐷, BN}, 
                         nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{𝐷, T}}, 
                         X::Matrix{T}, (vars,)::Tuple{HFtempVars{T, :RHF}}, 
                         isConverged::Bool) where {T, 𝐷, BN, NN}
        C = (vars.Cs[end],)
        F = vars.Fs[end]
        D = (vars.Ds[end],) 
        Ehf = vars.shared.Etots[end]
        Eo = (getCϵ(X, F)[2],)
        N = vars.N
        occu = ((fill(spinOccupations[4], N)..., fill(spinOccupations[1], BN-N)...),)
        Enn = nnRepulsions(nuc, nucCoords)
        new{T, 𝐷, :RHF, NN, BN, 1}(Ehf, Enn, (N, N), nuc, nucCoords, C, (F,), D, Eo, occu, 
                                   (vars,), isConverged, basis)
    end

    function HFfinalVars(basis::GTBasis{T, 𝐷, BN}, 
                         nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{𝐷, T}}, 
                         X::Matrix{T}, αβVars::NTuple{2, HFtempVars{T, :UHF}}, 
                         isConverged::Bool) where {T, 𝐷, BN, NN}
        C = last.(getproperty.(αβVars, :Cs))
        F = last.(getproperty.(αβVars, :Fs))
        D = last.(getproperty.(αβVars, :Ds))
        Ehf = αβVars[1].shared.Etots[end]
        Eo = getindex.(getCϵ.(Ref(X), F), 2)
        Nα, Nβ = Ns = getproperty.(αβVars, :N)
        occu = ( (fill(spinOccupations[2], Nα)..., fill(spinOccupations[1], BN-Nα)...), 
                 (fill(spinOccupations[3], Nβ)..., fill(spinOccupations[1], BN-Nβ)...) )
        Enn = nnRepulsions(nuc, nucCoords)
        new{T, 𝐷, :UHF, NN, BN, 2}(Ehf, Enn, Ns, nuc, nucCoords, C, F, D, Eo, occu, 
                                   αβVars, isConverged, basis)
    end
end

struct InitialC{T<:Number, HFT, F<:Function}
    mat::NTuple{<:Any, Matrix{T}}
    f::F

    InitialC(::Val{HFT}, f::F, ::Type{T}) where {HFT, F, T} = new{T, HFT, F}((), f)

    InitialC(::Val{:RHF}, C0::NTuple{1, Matrix{T}}) where {T} = 
    new{T, :RHF, itselfT}(C0, itself)

    InitialC(::Val{:UHF}, C0::NTuple{2, Matrix{T}}) where {T} = 
    new{T, :UHF, itselfT}(C0, itself)
end

const Doc_HFconfig_Eg1 = "HFconfig{:RHF, Val{:SAD}, 3}(Val{:RHF}(), Val{:SAD}(), "*
                         "SCFconfig{Float64, 3}(interval=(0.0001, 1.0e-6, 1.0e-15), "*
                         "oscillateThreshold=1.0e-5, method, methodConfig)"*
                         "[:ADIIS, :DIIS, :ADIIS], 1000, true)"

const Doc_HFconfig_Eg2 = Doc_HFconfig_Eg1[1:10] * "U" * Doc_HFconfig_Eg1[12:34] * "U" * 
                         Doc_HFconfig_Eg1[36:end]

const HFtypes = (:RHF, :UHF)

"""

    HFconfig{T1, HFT, F, T2, L} <: ConfigBox{T1, HFconfig, HFT}

The container of Hartree-Fock method configuration.

≡≡≡ Field(s) ≡≡≡

`HF::Val{HFT}`: Hartree-Fock method type. Available values of `HFT` are 
$(string(HFtypes)[2:end-1]).

`C0::ICT`: Initial guess of the coefficient matrix(s) C of the canonical orbitals. When `C0` 
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
mutable struct HFconfig{T1, HFT, F, T2, L} <: ConfigBox{T1, HFconfig, HFT}
    HF::Val{HFT}
    C0::InitialC{T1, HFT, F}
    SCF::SCFconfig{T2, L}
    maxStep::Int
    earlyStop::Bool

    HFconfig(::Val{:UHF}, a2::NTuple{2, Matrix{T1}}, a3::SCFconfig{T2, L}, a4, a5) where 
            {T1, T2, L} = 
    new{T1, :UHF, itselfT, T2, L}(Val(:UHF), InitialC(Val(:UHF), a2), a3, a4, a5)

    HFconfig(::Val{:RHF}, a2::Matrix{T1}, a3::SCFconfig{T2, L}, a4, a5) where {T1, T2, L} = 
    new{T1, :RHF, itselfT, T2, L}(Val(:RHF), InitialC(Val(:RHF), (a2,)), a3, a4, a5)

    function HFconfig(::Val{HFT}, a2::Val{CF}, a3::SCFconfig{T, L}, a4, a5) where 
                     {T, HFT, CF, L}
        f = getproperty(guessCmethods, CF)
        new{T, HFT, typeof(f), T, L}(Val(HFT), InitialC(Val(HFT), f, T), a3, a4, a5)
    end
end

HFconfig(a1::Symbol, a2, args...) = HFconfig(Val(a1), a2, args...)

HFconfig(a1, a2::Symbol, args...) = HFconfig(a1, Val(a2), args...)

HFconfig(a1::Symbol, a2::Symbol, args...) = HFconfig(Val(a1), Val(a2), args...)

const defaultHFconfigPars = Any[Val(:RHF), Val(:SAD), defaultSCFconfig, 100, true]

HFconfig(t::NamedTuple) = genNamedTupleC(:HFconfig, defaultHFconfigPars)(t)

HFconfig(;kws...) = 
length(kws) == 0 ? HFconfig(defaultHFconfigPars...) : HFconfig(kws|>NamedTuple)

const defaultHFC = HFconfig()

const defaultHFCStr = "HFconfig()"


const C0methodArgOrders = (itself=(1,), 
                           getCfromGWH=(2,3,5,4), 
                           getCfromHcore=(2,4,5), 
                           getCfromSAD=(2,3,5,6,7,8,9,4))

"""
    runHF(bs::Union{BasisSetData, AbstractVector{<:AbstractGTBasisFuncs{T1, D}}, 
                    Tuple{Vararg{AbstractGTBasisFuncs{T1, D}}}}, 
          nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
          nucCoords::Union{NTuple{NN, NTuple{D, T1}}, 
                           AbstractVector{<:AbstractArray{<:Real}}}, 
          config::Union{HFconfig{T1, HFT, itselfT}, HFconfig{T2, HFT}}=$(defaultHFCStr), 
          N::Int=getCharge(nuc); 
          printInfo::Bool=true) where {T1, D, NN, HFT, T2} -> 
    HFfinalVars{T1, D, HFT, NN}

Main function to run Hartree-Fock in Quiqbox.

=== Positional argument(s) ===

`bs::Union{BasisSetData, AbstractVector{<:AbstractGTBasisFuncs{T1, D}}, 
           Tuple{Vararg{AbstractGTBasisFuncs{T1, D}}}}`: Basis set.

`nuc::Union{NTuple{NN, String}, AbstractVector{String}}`: The element symbols of the nuclei 
for the studied system.

`nucCoords::Union{NTuple{NN, NTuple{D, T1}}, 
AbstractVector{<:AbstractVector{<:Real}}}`: Nuclei coordinates.

`config::HFconfig`: The Configuration of selected Hartree-Fock method. For more information 
please refer to `HFconfig`.

`N::Int`: Total number of electrons.

=== Keyword argument(s) ===

`printInfo::Bool`: Whether print out the information of iteration steps.
"""
function runHF(bs::GTBasis{T1, D, BN, BT}, 
               nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
               nucCoords::Union{NTuple{NN, NTuple{D, T1}}, 
                                AbstractVector{<:AbstractVector{T1}}}, 
               config::Union{HFconfig{T1, HFT, itselfT}, HFconfig{T2, HFT}}=defaultHFC, 
               N::Int=getCharge(nuc); 
               printInfo::Bool=true) where {T1, D, BN, BT, NN, HFT, T2}
    nuc = arrayToTuple(nuc)
    nucCoords = genTupleCoords(T1, nucCoords)
    leastNb = ceil(N/2) |> Int
    @assert BN >= leastNb "The number of basis functions should be no less than $(leastNb)."
    @assert N > (HFT==:RHF) "$(HFT) requires more than $(HFT==:RHF) electrons."
    Ns = splitSpins(Val(HFT), N)
    Hcore = coreH(bs, nuc, nucCoords)
    X = getX(bs.S)
    getC0f = config.C0.f
    C0 = uniCallFunc(getC0f, getproperty(C0methodArgOrders, nameOf(getC0f)), config.C0.mat, 
                     Val(HFT), bs.S, X, Hcore, bs.eeI, bs.basis, nuc, nucCoords)
    vars, isConverged = runHFcore(Val(HFT), config.SCF, Ns, Hcore, bs.eeI, bs.S, X, 
                                  C0, printInfo, config.maxStep, config.earlyStop)
    res = HFfinalVars(bs, nuc, nucCoords, X, vars, isConverged)
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

    runHF(bs::Union{BasisSetData{T1, D}, AbstractVector{<:AbstractGTBasisFuncs{T1, D}}, 
                    Tuple{Vararg{AbstractGTBasisFuncs{T1, D}}}}, 
          nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
          nucCoords::Union{NTuple{NN, NTuple{D, T1}}, 
                           AbstractVector{<:AbstractArray{<:Real}}},
          N::Int=getCharge(nuc), 
          config::Union{HFconfig{T1, HFT, itselfT}, HFconfig{T2, HFT}}=$(defaultHFCStr); 
          printInfo::Bool=true) where {T1, D, NN, HFT, T2} -> 
    HFfinalVars{T1, D, HFT, NN}
"""
runHF(bs::BasisSetData, nuc, nucCoords, N::Int, config=defaultHFC; printInfo=true) = 
runHF(bs::BasisSetData, nuc, nucCoords, config, N; printInfo)

runHF(bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                AbstractVector{<:AbstractGTBasisFuncs{T, D}}}, args...; 
     printInfo=true) where {T, D} = 
runHF(GTBasis(bs), args...; printInfo)


"""

    runHFcore(::Val{HFT}, 
              scfConfig::SCFconfig{T1, L}, 
              N::NTuple{HFTS, Int}, 
              Hcore::Matrix{T2}, 
              HeeI::Array{T2, 4}, 
              S::Matrix{T2}, 
              X::Matrix{T2}, 
              C0::NTuple{HFTS, Matrix{T2}}, 
              printInfo::Bool=false, 
              maxStep::Int=1000, 
              earlyStop::Bool=true) where {HFT, T1, L, HFTS, T2} -> 
    Tuple{Vararg{HFtempVars}}, Bool

The core function of `runHF` which returns the data collected during the iteration and the 
result of whether the SCF procedure is converged.

=== Positional argument(s) ===

`HTtype::Val{HFT}`: Hartree-Fock method type. Available values of `HFT` are 
$(string(HFtypes)[2:end-1]).

`scfConfig::SCFconfig`: SCF iteration configuration.

`N::Union{NTuple{2, Int}, Int}`: The total number of electrons or the numbers of electrons 
with different spins respectively. When the latter is input, an UHF is performed.

`Hcore::Matrix{T}`: Core Hamiltonian of electronic Hamiltonian.

`HeeI::Array{T, 4}`: The electron-electron interaction Hamiltonian which includes both the 
Coulomb interactions and the Exchange Correlations.

`S::Matrix{T}`: Overlap matrix of the corresponding basis set.

`X::Matrix{T}`: Orthogonal transformation matrix of S. Default value is S^(-0.5).

`C0::Union{Matrix{T}, NTuple{2, Matrix{T}}}`: Initial guess of the 
coefficient matrix(s) C of the canonical orbitals.

`printInfo::Bool`: Whether print out the information of iteration steps.

`maxStep::Int`: Maximum allowed iteration steps regardless of whether the SCF converges.

`earlyStop::Bool`: Whether automatically early terminate (skip) a convergence method 
when its performance becomes unstable or poor.
"""
function runHFcore(::Val{HFT}, 
                   scfConfig::SCFconfig{T1, L}, 
                   N::NTuple{HFTS, Int}, 
                   Hcore::Matrix{T2}, 
                   HeeI::Array{T2, 4}, 
                   S::Matrix{T2}, 
                   X::Matrix{T2}, 
                   C0::NTuple{HFTS, Matrix{T2}}, 
                   printInfo::Bool=false, 
                   maxStep::Int=1000, 
                   earlyStop::Bool=true) where {HFT, T1, L, HFTS, T2}
    vars = initializeSCF(Val(HFT), Hcore, HeeI, C0, N)
    Etots = vars[1].shared.Etots
    printInfo && println(rpad(HFT, 4)*rpad(" | Initial Gauss", 18), "E = $(Etots[end])")
    isConverged = true
    i = 0
    for (m, kws, breakPoint, l) in 
        zip(scfConfig.method, scfConfig.methodConfig, scfConfig.interval, 1:L)

        while true
            i += 1
            i <= maxStep || (isConverged = false) || break

            res = HFcore(m, N, Hcore, HeeI, S, X, vars; kws...)
            pushHFtempVars!(vars, res)

            if earlyStop && i > 1 && (Etots[end] - Etots[end-1]) / abs(Etots[end-1]) > 0.05
                isConverged = false
                i = terminateSCF(i, vars, m, printInfo)
                break
            end

            flag, Std = isOscillateConverged(Etots, sqrt(breakPoint))

            if flag 
                isConverged = ifelse(Std > scfConfig.oscillateThreshold, false, true)
                if !isConverged
                    i = terminateSCF(i, vars, m, printInfo)
                end
                break
            end

            printInfo && (i % floor(log(4, i) + 1) == 0 || i == maxStep) && 
            println(rpad("Step $i", 10), rpad("#$l ($(m))", 12), "E = $(Etots[end])")

            abs(Etots[end]-Etots[end-1]) > breakPoint || (isConverged = true) && break
        end

    end
    negStr = ifelse(isConverged, "is ", "has not ")
    printInfo && println("The SCF procedure ", negStr, "converged.\n")
    vars, isConverged
end

function terminateSCF(i, vars, method, printInfo)
    popHFtempVars!(vars)
    printInfo && println("Early termination of ", method, " due to the poor performance.")
    i-1
end

const defaultDampStrength = 0.5

function DDcore(Nˢ::Int, X::AbstractMatrix{T}, F::AbstractMatrix{T}, D::AbstractMatrix{T}, 
                dampStrength::T=T(defaultDampStrength)) where {T}
    @assert 0 <= dampStrength <= 1 "The range of `dampStrength`::$(T) is [0,1]."
    Dnew = getD(X, F, Nˢ)
    (1 - dampStrength)*Dnew + dampStrength*D
end


function EDIIScore(∇s::AbstractVector{<:AbstractMatrix{T}}, 
                   Ds::AbstractVector{<:AbstractMatrix{T}}, Es::AbstractVector{T}) where {T}
    len = length(Ds)
    B = ones(len, len)
    for j=1:len, i=1:len
        B[i,j] = -dot(Ds[i]-Ds[j], ∇s[i]-∇s[j])
    end
    Es, B
end


function ADIIScore(∇s::AbstractVector{<:AbstractMatrix{T}}, 
                   Ds::AbstractVector{<:AbstractMatrix{T}}) where {T}
    len = length(Ds)
    B = ones(len, len)
    v = [dot(D - Ds[end], ∇s[end]) for D in Ds]
    for j=1:len, i=1:len
        B[i,j] = dot(Ds[i]-Ds[len], ∇s[j]-∇s[len])
    end
    v, B
end


function DIIScore(∇s::AbstractVector{<:AbstractMatrix{T}}, 
                  Ds::AbstractVector{<:AbstractMatrix{T}}, S::AbstractMatrix{T}) where {T}
    len = length(Ds)
    B = ones(len, len)
    v = zeros(len)
    for j=1:len, i=1:len
        B[i,j] = dot(∇s[i]*Ds[i]*S - S*Ds[i]*∇s[i], ∇s[j]*Ds[j]*S - S*Ds[j]*∇s[j])
    end
    v, B
end


function DD(Nˢ::NTuple{HFTS, Int}, Hcore, HeeI, _S, X, 
            tVars::NTuple{HFTS, HFtempVars{T, HFT}}; kws...) where {HFTS, T, HFT}
    Fs = last.(getproperty.(tVars, :Fs))
    Ds = last.(getproperty.(tVars, :Ds))
    Dnew = DDcore.( Nˢ, Ref(X), Fs, Ds, get(kws, :dampStrength, T(defaultDampStrength)) )
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

const defaultDIISconfig = (12, :BFGS)

const DIIScoreMethods = (DIIS=DIIScore, EDIIS=EDIIScore, ADIIS=ADIIScore)

const DIISmethodArgOrders = (DIIScore=(1,2,4), EDIIScore=(1,2,3), ADIIScore=(1,2))

const DIISadditionalConfigs = (DIIS=(false, true), EDIIS=(true, false), ADIIS=(true, false))

function xDIIScore(::Val{M}, S::Matrix{T}, 
                   Fs::Vector{Matrix{T}}, Ds::Vector{Matrix{T}}, Es::Vector{T}, 
                   DIISsize::Int=defaultDIISconfig[1], 
                   solver::Symbol=defaultDIISconfig[2]) where {M, T}
    cvxConstraint, permuteData = getproperty(DIISadditionalConfigs, M)
    is = permuteData ? sortperm(Es, rev=true) : (:)
    ∇s = @view Fs[is][1:end .> end-DIISsize]
    Ds = @view Ds[is][1:end .> end-DIISsize]
    Es = @view Es[is][1:end .> end-DIISsize]
    DIIS = getproperty(DIIScoreMethods, M)
    v, B = uniCallFunc(DIIS, getproperty(DIISmethodArgOrders, nameOf(DIIS)), ∇s, Ds, Es, S)
    c = constraintSolver(v, B, cvxConstraint, solver)
    sum(c.*∇s) # Fnew
end


const SCFmethodSelector = 
      (DD=DD, DIIS=xDIIS(Val(:DIIS)), ADIIS=xDIIS(Val(:ADIIS)), EDIIS=xDIIS(Val(:EDIIS)))


function HFcore(m::Symbol, N::NTuple{HFTS, Int}, Hcore::Matrix{T}, HeeI::Array{T, 4}, 
                S::Matrix{T}, X::Matrix{T}, rVars::NTuple{HFTS, HFtempVars{T, HFT}}; 
                kws...) where {HFTS, T, HFT}
    F = getproperty(SCFmethodSelector, m)(N, Hcore, HeeI, S, X, rVars; kws...)
    getCFDE(Hcore, HeeI, X, N, F)
end


function pushHFtempVarsCore1!(rVars::HFtempVars, 
                              res::Tuple{Matrix{T}, Matrix{T}, 
                                         Matrix{T}, T, Matrix{T}, T}) where {T}
    push!(rVars.Cs, res[1])
    push!(rVars.Fs, res[2])
    push!(rVars.Ds, res[3])
    push!(rVars.Es, res[4])
end

function pushHFtempVarsCore2!(rVars::HFtempVars, 
                              res::Tuple{Matrix{T}, Matrix{T}, 
                                         Matrix{T}, T, Matrix{T}, T}) where {T}
    push!(rVars.shared.Dtots, res[5])
    push!(rVars.shared.Etots, res[6])
end

function pushHFtempVars!(αβVars::NTuple{HFTS, HFtempVars{T, HFT}}, 
                         res::NTuple{HFTS, Tuple{Matrix{T}, Matrix{T}, Matrix{T}, T, 
                                                 Matrix{T}, T}}) where {HFTS, T, HFT}
    pushHFtempVarsCore1!.(αβVars, res)
    pushHFtempVarsCore2!(αβVars[1], res[1])
end


function popHFtempVarsCore1!(rVars::HFtempVars)
    pop!(rVars.Cs)
    pop!(rVars.Fs)
    pop!(rVars.Ds)
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
function LBFGSBsolver(v::AbstractVector{T}, B::AbstractMatrix{T}, cvxConstraint::Bool) where {T<:Real}
    f = genxDIISf(v, B)
    g! = genxDIIS∇f(v, B)
    lb = ifelse(cvxConstraint, T(0), T(-Inf))
    vL = length(v)
    c0 = fill(T(1)/vL, vL)
    innerOptimizer = LBFGS(m=min(getAtolDigits(T), 50), 
                                 linesearch=HagerZhang(linesearchmax=100, epsilon=1e-7), 
                                 alphaguess=InitialHagerZhang())
    res = OptimOptimize(f, g!, fill(lb, vL), fill(T(Inf), vL), c0, Fminbox(innerOptimizer), 
                        OptimOptions(g_tol=getAtolVal(T), iterations=20000))
    c = OptimMinimizer(res)
    c ./ sum(c)
end

function CMsolver(v::AbstractVector{T}, B::AbstractMatrix{T}, cvxConstraint::Bool, ϵ::T=T(1e-5)) where {T}
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
        (findfirst(x->x<0, c) !== nothing && cvxConstraint) || (return c)
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

constraintSolver(v::AbstractVector{T}, B::AbstractMatrix{T}, 
                 cvxConstraint::Bool, solver::Symbol) where {T} = 
getproperty(ConstraintSolvers, solver)(v, B, cvxConstraint)