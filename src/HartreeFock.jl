export SCFconfig, runHF, runHFcore

using LinearAlgebra: dot, Hermitian, \, det, I
using PiecewiseQuadratics: indicator
using SeparableOptimization
using Combinatorics: powerset

const TelLB = Float64 # Union{} to loose constraint
const TelUB = Float64 # Any to loose constraint

getXcore1(S::Matrix{T}) where {TelLB<:T<:TelUB} = S^(-0.5) |> Array

const getXmethods = Dict{Int, Function}(1=>getXcore1)

getX(S::Matrix{T}; method::Int=1) where {TelLB<:T<:TelUB} = getXmethods[method](S)


function getC(X::Matrix{T1}, F::Matrix{T2}; 
              outputCx::Bool=false, outputEmo::Bool=false, stabilizeSign::Bool=true) where 
             {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB}
    ϵ, Cₓ = eigen(X'*F*X |> Hermitian, sortby=x->x)
    outC = outputCx ? Cₓ : X*Cₓ
    # Stabilize the sign factor of each column.
    stabilizeSign && for j = 1:size(outC, 2)
       outC[:, j] *= (outC[1,j] < 0 ? -1 : 1)
    end
    outputEmo ? (outC, ϵ) : outC
end


function breakSymOfC(C::Matrix{T}) where {TelLB<:T<:TelUB}
    C2 = copy(C)
    l = min(size(C2)[1], 2)
    C2[1:l, 1:l] .= 0 # Breaking spin symmetry.
    # C2[l, :] .= 0 # Another way.
    (copy(C), C2)
end


function getCfromGWH(S::Matrix{T1}, Hcore::Matrix{T2}, K::Float64=1.75; X=getX(S),
                     forUHF::Bool=false) where 
                    {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB}
    l = size(Hcore)[1]
    H = zero(Hcore)
    for i in 1:l, j in 1:l
        H[i,j] = K * S[i,j] * (Hcore[i,i] + Hcore[j,j]) * 0.5
    end
    C = getC(X, H)
    forUHF ? breakSymOfC(C) : C
end


function getCfromHcore(X::Matrix{T1}, Hcore::Matrix{T2}; forUHF::Bool=false) where 
                      {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB}
    C = getC(X, Hcore)
    forUHF ? breakSymOfC(C) : C
end


function getCfromSAD(S::Matrix{T1}, Hcore::Matrix{T2}, HeeI::Array{T3, 4},
                     bs::Vector{<:AbstractGTBasisFuncs}, 
                     nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}; 
                     X=getX(S), forUHF::Bool=false, 
                     scfConfig=SCFconfig([:ADIIS], [1e-10])) where 
                    {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB}
    D₁ = zero(Hcore)
    D₂ = zero(Hcore)
    N₁tot = 0
    N₂tot = 0
    order = sortperm(nuc, by=x->AtomicNumberList[x])
    for (atm, coord) in zip(nuc[order], nucCoords[order])
        N = getCharge(atm)
        N₁ = N ÷ 2
        N₂ = N - N₁
        if N₂ > N₁ && N₂tot > N₁tot
            temp = N₁
            N₁ = N₂
            N₂ = temp
        end
        h1 = coreH(bs, [atm], [coord])
        res = runHFcore(scfConfig, 
                        (N₁, N₂), h1, HeeI, S, X, getCfromHcore(X, h1, forUHF=true))
        D₁ += res.D[1]
        D₂ += res.D[2]
        N₁tot += N₁
        N₂tot += N₂
    end
    Dᵀ = D₁ + D₂
    if forUHF
        getC.(Ref(X), getF.(Ref(Hcore), Ref(HeeI), (D₁, D₂), Ref(Dᵀ)))
    else
        getC(X, getF(Hcore, HeeI, Dᵀ.*0.5, Dᵀ))
    end
end


const guessCmethods = 
    Dict(  :GWH => (forUHF, S, X, Hcore, _...) -> getCfromGWH(S, Hcore; X, forUHF),
         :Hcore => (forUHF, S, X, Hcore, _...) -> getCfromHcore(X, Hcore; forUHF), 
           :SAD => (forUHF, S, X, Hcore, HeeI, bs, nuc, nucCoords) -> 
                   getCfromSAD(S, Hcore, HeeI, bs, nuc, nucCoords; X, forUHF))


guessC(method::Symbol, forUHF::Bool, S::Matrix{T1}, X::Matrix{T1}, Hcore::Matrix{T2}, 
       HeeI::Array{T3, 4}, bs::Vector{<:AbstractGTBasisFuncs}, 
       nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) where 
      {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB} = 
guessCmethods[method](forUHF, S, X, Hcore, HeeI, bs, nuc, nucCoords)


getD(C::Matrix{T}, Nˢ::Int) where {TelLB<:T<:TelUB} = 
@views (C[:,1:Nˢ]*C[:,1:Nˢ]') |> Hermitian |> Array
# Nˢ: number of electrons with the same spin.

getD(X::Matrix{T1}, F::Matrix{T2}, Nˢ::Int) where {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB} = 
getD(getC(X, F), Nˢ)


function getGcore(HeeI::Array{T1, 4}, DJ::Matrix{T2}, DK::Matrix{T3}) where 
    {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB}
    G = zero(DJ)
    l = size(G)[1]
    for μ = 1:l, ν = 1:l # fastest
        G[μ, ν] = dot(transpose(DJ), @view HeeI[μ,ν,:,:]) - dot(DK, @view HeeI[μ,:,:,ν]) 
    end
    G |> Hermitian |> Array
end

# RHF
getG(HeeI::Array{T1, 4}, D::Matrix{T2}) where 
    {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB} = 
getGcore(HeeI, 2D, D)

# UHF
getG(HeeI::Array{T1, 4}, D::Matrix{T2}, Dᵀ::Matrix{T3}) where 
    {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB} = 
getGcore(HeeI, Dᵀ, D)


getF(Hcore::Matrix{T1}, G::Matrix{T2}) where 
            {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB} = 
        Hcore + G

# RHF
getF(Hcore::Matrix{T1}, HeeI::Array{T2, 4}, D::Matrix{T3}) where 
    {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB} = 
getF(Hcore, getG(HeeI, D))

# UHF
getF(Hcore::Matrix{T1}, HeeI::Array{T2, 4}, D::Matrix{T3}, Dᵀ::Matrix{T4}) where 
    {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB, TelLB<:T4<:TelUB} = 
getF(Hcore, getG(HeeI, D, Dᵀ))

# RHF or UHF
getF(Hcore::Matrix{T1}, HeeI::Array{T2, 4}, Ds::NTuple{N, Matrix{T3}}) where 
    {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB, N} = 
getF(Hcore, getG(HeeI, Ds...))

getE(Hcore::Matrix{T1}, F::Matrix{T2}, D::Matrix{T3}) where 
    {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB} = 
dot(transpose(D), 0.5*(Hcore + F))


# RHF
getEᵀcore(Hcore::Matrix{T1}, F::Matrix{T2}, D::Matrix{T3}) where 
         {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB} = 
2*getE(Hcore, F, D)

# UHF
getEᵀcore(Hcore::Matrix{T1}, Fᵅ::Matrix{T2},   Dᵅ::Matrix{T3}, 
          Fᵝ::Matrix{T4},    Dᵝ::Matrix{T5}) where 
         {TelLB<:T1<:TelUB,  TelLB<:T2<:TelUB, TelLB<:T3<:TelUB, 
          TelLB<:T4<:TelUB,  TelLB<:T5<:TelUB} = 
getE(Hcore, Fᵅ, Dᵅ) + getE(Hcore, Fᵝ, Dᵝ)

# RHF
function getEᵀ(Hcore::Matrix{T1}, HeeI::Array{T2, 4}, C::Matrix{T3}, N::Int) where 
              {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB}
    D = getD(C, N÷2)
    F = getF(Hcore, HeeI, D)
    getEᵀcore(Hcore, F, D)
end

# UHF
function getEᵀ(Hcore::Matrix{T1}, HeeI::Array{T2, 4}, (Cᵅ,Cᵝ)::NTuple{2, Matrix{T3}}, 
               (Nᵅ,Nᵝ)::NTuple{2, Int}) where 
              {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB}
    Dᵅ = getD(Cᵅ, Nᵅ)
    Dᵝ = getD(Cᵝ, Nᵝ)
    Dᵀ = Dᵅ + Dᵝ
    Fᵅ = getF(Hcore, HeeI, Dᵅ, Dᵀ)
    Fᵝ = getF(Hcore, HeeI, Dᵝ, Dᵀ)
    getEᵀcore(Hcore, Fᵅ, Dᵅ, Fᵝ, Dᵝ)
end


function getCFDE(Hcore::Matrix{T1}, HeeI::Array{T2, 4}, X::Matrix{T3}, 
                 Ds::Vararg{Matrix{T4}, N}) where 
                {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB, TelLB<:T4<:TelUB, N}
    Fnew = getF(Hcore, HeeI, Ds)
    Enew = getE(Hcore, Fnew, Ds[1])
    Cnew = getC(X, Fnew)
    [Cnew, Fnew, Ds[1], Enew] # Fnew is based on latest variables.
end


# RHF
function initializeSCF(Hcore::Matrix{T1}, HeeI::Array{T2, 4}, C::Matrix{T3}, N::Int) where 
                      {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB}
    Nˢ = N÷2
    D = getD(C, Nˢ)
    F = getF(Hcore, HeeI, D)
    E = getE(Hcore, F, D)
    HFtempVars(:RHF, Nˢ, C, F, D, E, 2D, 2E)
end

# UHF
function initializeSCF(Hcore::Matrix{T1}, HeeI::Array{T2, 4}, 
                       Cs::NTuple{2, Matrix{T3}}, 
                       Ns::NTuple{2, Int}) where 
                       {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB}
    Ds = getD.(Cs, Ns)
    Dᵀs = [Ds |> sum]
    Fs = getF.(Ref(Hcore), Ref(HeeI), Ds, Ref(Dᵀs[]))
    Es = getE.(Ref(Hcore), Fs, Ds)
    Eᵀs = [Es |> sum]
    res = HFtempVars.(:UHF, Ns, Cs, Fs, Ds, Es)
    res[1].shared.Dtots = res[2].shared.Dtots = Dᵀs
    res[1].shared.Etots = res[2].shared.Etots = Eᵀs
    res |> Tuple
end


const Doc_SCFconfig_OneRowTable = "| `:DIIS`, `:EDIIS`, `:ADIIS` | Subspace size (>1); "*
                                  "Coefficient solver(`:ADMM`-> ADMM solver,"*
                                  " `:LCM` -> Lagrange solver) | "*
                                  "`DIISsize::Int`; `solver::Symbol` | `15`; `:ADMM` |"

"""

    SCFconfig{N} <: ImmutableParameter{SCFconfig, Any}

The `struct` for SCF iteration configurations.

≡≡≡ Field(s) ≡≡≡

`methods::NTuple{N, Symbol}`: The applied methods. The available methods are their 
configurations (in terms of keyword arguments):

| Methods | Configuration(s) | keyword argument(s) | Default value(s) |
| :----   | :---:            | :---:               | ----:            |
| `:DS` | Damping strength: [0,1] | `dampingStrength::Float64` | `0.0` |
$(Quiqbox.Doc_SCFconfig_OneRowTable)

`intervals`: The stopping (skipping) thresholds for the required methods.

`methodConfigs`: The additional keywords arguments for each method stored as `Tuple`s of 
`Pair`s.

`oscillateThreshold`: The threshold for oscillating convergence.

≡≡≡ Initialization Method(s) ≡≡≡

    SCFconfig(methods::Vector{Symbol}, intervals::Vector{Float64}, 
              configs::Dict{Int, <:Vector{<:Pair}}=Dict(1=>Pair[]);
              oscillateThreshold::Float64=1e-5) -> 
    SCFconfig{N}

`methods` and `intervals` are the methods to be applied and their stopping (skipping) 
thresholds respectively; the length of those two `Vector`s should be the same. `configs` 
specifies the additional keyword arguments for each methods by a `Pair` of which the `Int` 
key `i` is for `i`th method and the pointed `Vector{<:Pair}` is the pairs of keyword 
arguments and their values respectively.

≡≡≡ Example(s) ≡≡≡

julia> SCFconfig([:SD, :ADIIS, :DIIS], [1e-4, 1e-12, 1e-13], 
                 Dict(2=>[:solver=>:LCM])
SCFconfig{3}((:SD, :ADIIS, :DIIS), (0.0001, 1.0e-12, 1.0e-13), ((), (:solver => :LCM,), 
()), 1.0e-5)
"""
struct SCFconfig{N} <: ImmutableParameter{SCFconfig, Any}
    methods::NTuple{N, Symbol}
    intervals::NTuple{N, Float64}
    methodConfigs::NTuple{N, Tuple{Vararg{Pair}}}
    oscillateThreshold::Float64

    function SCFconfig(methods::Vector{Symbol}, intervals::Vector{Float64}, 
                       configs::Dict{Int, <:Vector{<:Pair}}=Dict(1=>Pair[]);
                       oscillateThreshold::Float64=1e-5)
        l = length(methods)
        kwPairs = [Pair[] for _=1:l]
        for i in keys(configs)
            kwPairs[i] = configs[i]
        end
        new{length(methods)}(Tuple(methods), Tuple(intervals), Tuple(kwPairs .|> Tuple), 
                             oscillateThreshold)
    end
end


const defaultSCFconfig = SCFconfig([:ADIIS, :DIIS, :ADIIS], [1e-4, 1e-6, 1e-15])


mutable struct HFinterrelatedVars <: HartreeFockintermediateData
    Dtots::Vector{Matrix{T}} where {TelLB<:T<:TelUB}
    Etots::Vector{Float64}

    HFinterrelatedVars() = new()
    HFinterrelatedVars(Dtots, Etots) = new(Dtots, Etots)
end


"""
    HFtempVars{HFtype, N} <: HartreeFockintermediateData

The container to store the intermediate values (only of the same spin configuration) for 
each iteration during the Hartree-Fock SCF procedure. 

≡≡≡ Field(s) ≡≡≡

`Cs::Array{Array{T1, 2}, 1} where {$(TelLB)<:T1<:$(TelUB)}`: Coefficient matrices.

`Fs::Array{Array{T2, 2}, 1} where {$(TelLB)<:T2<:$(TelUB)}`: Fock matrices

`Ds::Array{Array{T3, 2}, 1} where {$(TelLB)<:T3<:$(TelUB)}`: Density matrices corresponding 
to only spin configuration. For RHF each elements means (unconverged) 0.5*Dᵀ.

`Es::Array{Float64, 1}`: Part of Hartree-Fock energy corresponding to only spin 
configuration. For RHF each element means (unconverged) 0.5*E0HF.

`shared.Dtots::Array{Array{T, 2}, 1} where {$(TelLB)<:T<:$(TelUB)}`: The total density 
matrices.

`shared.Etots::Array{Float64, 1}`: The total Hartree-Fock energy.

**NOTE: For UHF, there are 2 `HFtempVars` being updated during the SCF iterations, and 
change the field `shared.Dtots` or `shared.Etots` of one container will affect the other 
one's.**
"""
struct HFtempVars{HFtype, N} <: HartreeFockintermediateData
    Cs::Vector{Matrix{T1}} where {TelLB<:T1<:TelUB}
    Fs::Vector{Matrix{T2}} where {TelLB<:T2<:TelUB}
    Ds::Vector{Matrix{T3}} where {TelLB<:T3<:TelUB}
    Es::Vector{Float64}
    shared::HFinterrelatedVars

    HFtempVars(HFtype::Symbol, N, C, F, D, E, vars...) = 
    new{HFtype, N}([C], [F], [D], [E], HFinterrelatedVars([[x] for x in vars]...))

    HFtempVars(HFtype::Symbol, Nˢ::Int, Cs::Vector{Matrix{T1}}, Fs::Vector{Matrix{T2}}, 
               Ds::Vector{Matrix{T3}}, Es::Vector{<:Real}, Dtots::Vector{Matrix{T4}}, 
               Etots::Vector{<:Real}) where 
              {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB, TelLB<:T4<:TelUB} = 
    new{HFtype, Nˢ}(Cs, Fs, Ds, Es, HFinterrelatedVars(Dtots, Etots))
end


"""

    HFfinalVars{T, N, Nb} <: HartreeFockFinalValue{T}

The container of the final values after a Hartree-Fock SCF procedure.

≡≡≡ Field(s) ≡≡≡

`E0HF::Float64`: Hartree-Fock energy of the electronic Hamiltonian. 

`C::Union{Array{T1, 2}, NTuple{2, Array{T1, 2}}} where {$(TelLB)<:T1<:$(TelUB)}`: 
Coefficient matrix(s) for one spin configuration.

`F::Union{Array{T2, 2}, NTuple{2, Array{T2, 2}}} where {$(TelLB)<:T2<:$(TelUB)}`: Fock 
matrix(s) for one spin configuration.

`D::Union{Array{T3, 2}, NTuple{2, Array{T3, 2}}} where {$(TelLB)<:T3<:$(TelUB)}`: Density 
matrix(s) for one spin configuration.

`Emo::Union{Array{Float64, 1}, NTuple{2, Array{Float64, 1}}}`: Energies of molecular 
orbitals.

`occu::Union{Array{Int, 1}, NTuple{2, Array{Int, 1}}}`: occupation numbers of molecular 
orbitals.

`temp::Union{HFtempVars{T}, NTuple{2, HFtempVars{T}}}` the intermediate values.

`isConverged::Bool`: Whether the SCF procedure is converged in the end.
"""
struct HFfinalVars{T, N, Nb} <: HartreeFockFinalValue{T}
    E0HF::Float64
    C::Union{Matrix{T1}, NTuple{2, Matrix{T1}}} where {TelLB<:T1<:TelUB}
    F::Union{Matrix{T2}, NTuple{2, Matrix{T2}}} where {TelLB<:T2<:TelUB}
    D::Union{Matrix{T3}, NTuple{2, Matrix{T3}}} where {TelLB<:T3<:TelUB}
    Emo::Union{Vector{Float64}, NTuple{2, Vector{Float64}}}
    occu::Union{Vector{Int}, NTuple{2, Vector{Int}}}
    temp::Union{HFtempVars{T}, NTuple{2, HFtempVars{T}}}
    isConverged::Bool

    function HFfinalVars(X::Matrix{T}, vars::HFtempVars{:RHF}, isConverged::Bool) where 
             {TelLB<:T<:TelUB}
        C = vars.Cs[end]
        F = vars.Fs[end]
        D = vars.Ds[end]
        E0HF = vars.shared.Etots[end]
        _, Emo = getC(X, F, outputEmo=true)
        Nˢ = typeof(vars).parameters[2]
        occu = vcat(2*ones(Int, Nˢ), zeros(Int, size(X, 1) - Nˢ))
        temp = vars
        new{:RHF, 2Nˢ, length(Emo)}(E0HF, C, F, D, Emo, occu, temp, isConverged)
    end

    function HFfinalVars(X::Matrix{T}, (αVars, βVars)::NTuple{2, HFtempVars{:UHF}}, 
                         isConverged::Bool) where {TelLB<:T<:TelUB}
        C = (αVars.Cs[end], βVars.Cs[end])
        F = (αVars.Fs[end], βVars.Fs[end])
        D = (αVars.Ds[end], βVars.Ds[end])
        E0HF = αVars.shared.Etots[end]
        res = getC.(Ref(X), F, outputEmo=true)
        Emo = getindex.(res, 2)
        Nˢs = (typeof(αVars).parameters[2], typeof(βVars).parameters[2]) 
        occu = vcat.(ones.(Int, Nˢs), zeros.(Int, size(X, 1) .- Nˢs))
        temp = (αVars, βVars)
        new{:UHF, Nˢs |> sum, length(Emo[1])}(E0HF, C, F, D, Emo, occu, temp, isConverged)
    end
end


"""
    runHF(bs::Union{BasisSetData, Array{<:AbstractGTBasisFuncs, 1}}, 
          nuc::Array{String, 1}, 
          nucCoords::Array{<:AbstractArray, 1}, 
          N::Int=getCharge(nuc); 
          initialC::Union{Array{T, 2}, NTuple{2, Array{T, 2}}, Symbol}=:SAD, 
          HFtype::Symbol=:RHF, 
          scfConfig::SCFconfig=defaultSCFconfig, 
          earlyTermination::Bool=true, 
          printInfo::Bool=true, 
          maxSteps::Int=1000) where {$(TelLB)<:T<:$(TelUB)} -> HFfinalVars

Main function to run Hartree-Fock in Quiqbox.

=== Positional argument(s) ===

`bs::Union{BasisSetData, Array{<:AbstractGTBasisFuncs, 1}}`: Basis set.

`nuc::Array{String, 1}`: The element symbols of the nuclei for the Molecule.

`nucCoords::Array{<:AbstractArray, 1}`: The coordinates of the nuclei.

`N::Int`: The total number of electrons.

=== Keyword argument(s) ===

`initialC::Union{Array{T, 2}, NTuple{2, Array{T, 2}}, Symbol}`: Initial guess of the 
coefficient matrix(s) C of the molecular orbitals.

`HFtype::Symbol`: Hartree-Fock type. Available values are `:RHF` and `:UHF`.

`scfConfig::SCFconfig`: SCF iteration configuration.

`earlyTermination::Bool`: Whether automatically early terminate (skip) a convergence method 
when its performance becomes unstable or poor.

`printInfo::Bool`: Whether print out the information of each iteration step.

`maxSteps::Int`: Maximum allowed iteration steps regardless of whether the SCF converges.
"""
function runHF(bs::Vector{<:AbstractGTBasisFuncs}, 
               nuc::Vector{String}, 
               nucCoords::Vector{<:AbstractArray}, 
               N::Int=getCharge(nuc); 
               initialC::Union{Matrix{T}, NTuple{2, Matrix{T}}, Symbol}=:SAD, 
               HFtype::Symbol=:RHF, 
               scfConfig::SCFconfig=defaultSCFconfig, 
               earlyTermination::Bool=true, 
               printInfo::Bool=true, 
               maxSteps::Int=1000) where {TelLB<:T<:TelUB}
    gtb = GTBasis(bs, false)
    runHF(gtb, nuc, nucCoords, N; initialC, HFtype, scfConfig, 
          earlyTermination, printInfo, maxSteps)
end

function runHF(gtb::BasisSetData, 
               nuc::Vector{String}, 
               nucCoords::Vector{<:AbstractArray}, 
               N::Int=getCharge(nuc); 
               initialC::Union{Matrix{T}, NTuple{2, Matrix{T}}, Symbol}=:SAD, 
               HFtype::Symbol=:RHF, 
               scfConfig::SCFconfig=defaultSCFconfig, 
               earlyTermination::Bool=true, 
               printInfo::Bool=true, 
               maxSteps::Int=1000) where {TelLB<:T<:TelUB}
    @assert length(nuc) == length(nucCoords)
    @assert typeof(gtb).parameters[1] >= ceil(sum(N)/2)
    @assert N > (HFtype == :RHF) "$(HFtype) requires more than $(LBofN) electron to run."
    HFtype == :UHF && (N = (N÷2, N-N÷2))
    Hcore = gtb.getHcore(nuc, nucCoords)
    X = getX(gtb.S)
    initialC isa Symbol && (initialC = guessC(initialC, (HFtype == :UHF), gtb.S, X, Hcore, 
                                              gtb.eeI, gtb.basis, nuc, nucCoords))
    runHFcore(scfConfig, N, Hcore, gtb.eeI, gtb.S, X, initialC; 
              printInfo, maxSteps, earlyTermination)
end


"""

    runHFcore(scfConfig::SCFconfig{L}, 
              N::Union{NTuple{2, Int}, Int}, 
              Hcore::Array{T1, 2}, 
              HeeI::Array{T2, 4}, 
              S::Array{T3, 2}, 
              X::Array{T4, 2}=getX(S), 
              C::Union{Array{T, 2}, NTuple{2, Array{T, 2}}}=
              getCfromGWH(S, Hcore; X, forUHF=(length(N)==2));
              earlyTermination::Bool=true, 
              printInfo::Bool=false, 
              maxSteps::Int=1000) where {$(TelLB)<:T1<:$(TelUB), 
                                         $(TelLB)<:T2<:$(TelUB), 
                                         $(TelLB)<:T3<:$(TelUB), 
                                         $(TelLB)<:T4<:$(TelUB), 
                                         $(TelLB)<:T5<:$(TelUB), L}

The core function of `runHF`.

=== Positional argument(s) ===

`scfConfig::SCFconfig`: SCF iteration configuration.

`N::Union{NTuple{2, Int}, Int}`: The total number of electrons or the numbers of electrons 
with different spins respectively. When the latter is input, an UHF is performed.

`Hcore::Array{T1, 2}`: Core Hamiltonian of electronic Hamiltonian.

`HeeI::Array{T2, 4}`: The electron-electron interaction Hamiltonian which includes both the 
Coulomb interactions and the Exchange Correlations.

`S::Array{T3, 2}`: Overlap matrix of the corresponding basis set.

`X::Array{T4, 2}`: Orthogonal transformation matrix of S. Default value is S^(-0.5).

`C::Union{Array{T, 2}, NTuple{2, Array{T, 2}}}`: Initial guess of the coefficient matrix(s) 
C of the molecular orbitals.

=== Keyword argument(s) ===

`earlyTermination::Bool`: Whether automatically early terminate (skip) a convergence method 
when its performance becomes unstable or poor.

`printInfo::Bool`: Whether print out the information of each iteration step.

`maxSteps::Int`: Maximum allowed iteration steps regardless of whether the SCF converges.
"""
function runHFcore(scfConfig::SCFconfig{L}, 
                   N::Union{NTuple{2, Int}, Int}, 
                   Hcore::Matrix{T1}, 
                   HeeI::Array{T2, 4}, 
                   S::Matrix{T3}, 
                   X::Matrix{T4}=getX(S), 
                   C::Union{Matrix{T5}, NTuple{2, Matrix{T5}}}=
                   getCfromGWH(S, Hcore; X, forUHF=(length(N)==2));
                   earlyTermination::Bool=true, 
                   printInfo::Bool=false, 
                   maxSteps::Int=1000) where {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, 
                                              TelLB<:T3<:TelUB, TelLB<:T4<:TelUB, 
                                              TelLB<:T5<:TelUB, L}
    @assert maxSteps > 0
    vars = initializeSCF(Hcore, HeeI, C, N)
    Etots = (vars isa Tuple) ? vars[1].shared.Etots : vars.shared.Etots
    HFtypeStr =  length(N) == 1 ? "RHF" : "UHF"
    printInfo && println(rpad(HFtypeStr*" | Initial Gauss", 22), "E = $(Etots[end])")
    isConverged = true
    EtotMin = Etots[]
    for (method, kws, breakPoint, i) in 
        zip(scfConfig.methods, scfConfig.methodConfigs, scfConfig.intervals, 1:L)

        while true
            iStep = length(Etots)
            iStep <= maxSteps || (isConverged = false) || break
            HF!(method, N, Hcore, HeeI, S, X, vars; kws...)
            printInfo && (iStep % floor(log(4, iStep) + 1) == 0 || iStep == maxSteps) && 
            println(rpad("Step $(iStep)", 10), 
                    rpad("#$(i) ($(method))", 12), 
                    "E = $(Etots[end])")
            abs(Etots[end]-Etots[end-1]) > breakPoint || (isConverged = true) && break
            flag, Std = isOscillateConverged(Etots, 
                                             10^(log(10, breakPoint)÷2), 
                                             returnStd=true)
            flag && (isConverged = Std > scfConfig.oscillateThreshold ? false : true; break)
            earlyTermination && (Etots[end] - EtotMin) / abs(EtotMin) > 0.2 && 
            (printInfo && println("Early termination of ", method, 
                                  " due to the poor performance."); 
             isConverged = false; break)
        end

    end
    negStr = isConverged ? "is " : "has not "
    printInfo && println("The SCF procedure ", negStr, "converged.\n")
    HFfinalVars(X, vars, isConverged)
end


function SDcore(Nˢ::Int, Hcore::Matrix{T1}, HeeI::Array{T2, 4}, X::Matrix{T3}, 
                F::Matrix{T4}, D::Matrix{T5}; dampingStrength::Float64=0.0, _kws...) where 
               {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB, TelLB<:T4<:TelUB, 
                TelLB<:T5<:TelUB}
    @assert 0 <= dampingStrength <= 1 "The range of `dampingStrength`::Float64 is [0,1]."
    Dnew = getD(X, F, Nˢ)
    (1-dampingStrength)*Dnew + dampingStrength*D
end


function xDIIScore(method::Symbol, Nˢ::Int, Hcore::Matrix{T1}, HeeI::Array{T2, 4}, 
                   S::Matrix{T3}, X::Matrix{T4}, Fs::Vector{Matrix{T5}}, 
                   Ds::Vector{Matrix{T6}},Es::Vector{Float64}; 
                   DIISsize::Int=15, solver::Symbol=:ADMM, 
                   _kws...) where 
                  {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB, TelLB<:T4<:TelUB, 
                   TelLB<:T5<:TelUB, TelLB<:T6<:TelUB}
    DIISmethod, convexConstraint, permuteData = DIISmethods[method]
    is = permuteData ? sortperm(Es) : (:)
    ∇s = (@view Fs[is])[1:end .> end-DIISsize]
    Ds = (@view Ds[is])[1:end .> end-DIISsize]
    Es = (@view Es[is])[1:end .> end-DIISsize]
    vec, B = DIISmethod(∇s, Ds, Es, S)
    c = constraintSolver(vec, B, solver; convexConstraint)
    grad = c.*∇s |> sum
    getD(X, grad |> Hermitian |> Array, Nˢ) # grad == F.
end

const DIISmethods = Dict( :DIIS => ((∇s, Ds,  _, S)->DIIScore(∇s, Ds, S),   false, true),
                         :EDIIS => ((∇s, Ds, Es, _)->EDIIScore(∇s, Ds, Es), true, false),
                         :ADIIS => ((∇s, Ds,  _, _)->ADIIScore(∇s, Ds),     true, false))


function EDIIScore(∇s::Vector{Matrix{T1}}, Ds::Vector{Matrix{T2}}, 
                   Es::Vector{Float64}) where {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB}
    len = length(Ds)
    B = ones(len, len)
    for i=1:len, j=1:len
        B[i,j] = -dot(Ds[i]-Ds[j], ∇s[i]-∇s[j])
    end
    Es, B
end


function ADIIScore(∇s::Vector{Matrix{T1}}, Ds::Vector{Matrix{T2}}) where 
                  {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB}
    len = length(Ds)
    B = ones(len, len)
    vec = [dot(D - Ds[end], ∇s[end]) for D in Ds]
    for i=1:len, j=1:len
        B[i,j] = dot(Ds[i]-Ds[len], ∇s[j]-∇s[len])
    end
    vec, B
end


function DIIScore(∇s::Vector{Matrix{T1}}, Ds::Vector{Matrix{T2}}, S::Matrix{T3}) where 
                 {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB}
    len = length(Ds)
    B = ones(len, len)
    vec = zeros(len)
    for i=1:len, j=1:len
        B[i,j] = dot(∇s[i]*Ds[i]*S - Ds[i]*S*∇s[i], ∇s[j]*Ds[j]*S - Ds[j]*S*∇s[j])
    end
    vec, B
end


const SD = ((Nˢ::Int, Hcore::Matrix{T1}, HeeI::Array{T2, 4}, _dm::Any, X::Matrix{T3}, 
             tVars::HFtempVars; kws...) where 
            {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB}) -> 
      SDcore(Nˢ, Hcore, HeeI, X, tVars.Fs[end], tVars.Ds[end]; kws...)

const xDIIS = (method::Symbol) -> 
              ((Nˢ::Int, Hcore::Matrix{T1}, HeeI::Array{T2, 4}, S::Matrix{T3}, 
                X::Matrix{T4}, tVars::HFtempVars; kws...) where 
               {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, 
                TelLB<:T3<:TelUB, TelLB<:T4<:TelUB}) ->
              xDIIScore(method, Nˢ, Hcore, HeeI, S, X, tVars.Fs, tVars.Ds, tVars.Es; kws...)


const SCFmethods = [:SD, :DIIS, :ADIIS, :EDIIS]

const SCFmethodSelector = Dict(SCFmethods .=> 
                               [SD, xDIIS(:DIIS), xDIIS(:ADIIS), xDIIS(:EDIIS)])


function HF!(SCFmethod::Symbol, N::Union{NTuple{2, Int}, Int}, 
             Hcore::Matrix{T1}, HeeI::Array{T2, 4}, S::Matrix{T3}, X::Matrix{T4}, 
             tVars::Union{HFtempVars{:RHF}, NTuple{2, HFtempVars{:UHF}}}; kws...) where 
            {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB, TelLB<:T4<:TelUB}
    res = HFcore(SCFmethod, N, Hcore, HeeI, S, X, tVars; kws...)
    pushHFtempVars!(tVars, res)
end

# RHF
function HFcore(SCFmethod::Symbol, N::Int, 
                Hcore::Matrix{T1}, HeeI::Array{T2, 4}, S::Matrix{T3}, X::Matrix{T4}, 
                rVars::HFtempVars{:RHF}; kws...) where 
               {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB, TelLB<:T4<:TelUB}
    D = SCFmethodSelector[SCFmethod](N÷2, Hcore, HeeI, S, X, rVars; kws...)
    partRes = getCFDE(Hcore, HeeI, X, D)
    partRes..., 2D, 2partRes[end]
end

function pushHFtempVars!(rVars::HFtempVars, 
                         res::Tuple{Matrix{T1}, Matrix{T2}, Matrix{T3}, Float64, 
                                    Matrix{T4}, Float64}) where 
                        {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB, 
                         TelLB<:T4<:TelUB}
    push!(rVars.Cs, res[1])
    push!(rVars.Fs, res[2])
    push!(rVars.Ds, res[3])
    push!(rVars.Es, res[4])
    push!(rVars.shared.Dtots, res[5])
    push!(rVars.shared.Etots, res[6])
end

# UHF
function HFcore(SCFmethod::Symbol, Ns::NTuple{2, Int}, 
                Hcore::Matrix{T1}, HeeI::Array{T2, 4}, S::Matrix{T3}, X::Matrix{T4}, 
                uVars::NTuple{2, HFtempVars{:UHF}}; kws...) where 
               {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB, TelLB<:T4<:TelUB}
    Ds = SCFmethodSelector[SCFmethod].(Ns, Ref(Hcore), Ref(HeeI), 
                                Ref(S), Ref(X), uVars; kws...)
    Dᵀnew = Ds |> sum
    partRes = getCFDE.(Ref(Hcore), Ref(HeeI), Ref(X), Ds, Ref(Dᵀnew))
    Eᵀnew = partRes[1][end] + partRes[2][end]
    (partRes[1]..., Dᵀnew, Eᵀnew), (partRes[2]..., Dᵀnew, Eᵀnew)
end

function pushHFtempVars!(uVars::NTuple{2, HFtempVars{:UHF}}, 
                         res::NTuple{2, Tuple{Matrix{T1}, Matrix{T2}, Matrix{T3}, Float64, 
                              Matrix{T4}, Float64}}) where 
                        {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB, 
                         TelLB<:T4<:TelUB}
    pushHFtempVars!.(uVars, res)
    pop!(uVars[1].shared.Dtots)
    pop!(uVars[1].shared.Etots)
end


function popHFtempVars!(rVars::HFtempVars)
    pop!(rVars.Cs)
    pop!(rVars.Fs)
    pop!(rVars.Ds)
    pop!(rVars.Es)
    pop!(rVars.shared.Dtots)
    pop!(rVars.shared.Etots)
end

function popHFtempVars!(uVars::NTuple{2, HFtempVars{:UHF}})
    fields = [:Cs, :Fs, :Ds, :Es]
    for field in fields pop!.(getfield.(uVars, field)) end
    pop!(uVars[1].shared.Dtots)
    pop!(uVars[1].shared.Etots)
end


# Default
function ADMMSolver(vec::Vector{Float64}, B::Matrix{Float64}; convexConstraint::Bool=true)
    len = length(vec)
    A = ones(len) |> transpose |> Array
    b = Float64[1.0]
    g = convexConstraint ? fill(indicator(0, 1), len) : fill(indicator(-Inf, Inf), len)
    params = SeparableOptimization.AdmmParams(B, vec, A, b, g)
    settings = SeparableOptimization.Settings(; ρ=ones(1), σ=ones(len), compute_stats=true)
    vars, _ = SeparableOptimization.optimize(params, settings)
    vars.x
end


function CMSolver(vec::Vector, B::Matrix; convexConstraint=true, ϵ::Float64=1e-5)
    len = length(vec)
    getA = (B)->[B  ones(len); ones(1, len) 0]
    b = vcat(-vec, 1)
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


const ConstraintSolvers = Dict(:ADMM=>ADMMSolver, :LCM=>CMSolver)

constraintSolver(vec::Vector{T1}, B::Matrix{T2}, 
                 solver::Symbol=:ADMM; convexConstraint::Bool=true) where 
                {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB} = 
ConstraintSolvers[solver](vec, B; convexConstraint)