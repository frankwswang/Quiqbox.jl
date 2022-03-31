export SCFconfig, runHF, runHFcore

using LinearAlgebra: dot, Hermitian, \, det, I
using PiecewiseQuadratics: indicator
using SeparableOptimization
using Combinatorics: powerset

getXcore1(S::Matrix{Float64}) = S^(-0.5) |> Array

const getXmethods = (m1=getXcore1,)

getX(S::Matrix{Float64}, method::Symbol=:m1) = getfield(getXmethods, method)(S)


getC(X::Matrix{Float64}, F::Matrix{Float64}; 
     outputEmo::Bool=false, outputCx::Bool=false, stabilizeSign::Bool=true) = 
getCcore(X, F, outputEmo, outputCx, stabilizeSign)

function getCcore(X::Matrix{Float64}, F::Matrix{Float64}, outputEmo::Bool=false, 
                  outputCx::Bool=false, stabilizeSign::Bool=true)
    ϵ, Cₓ = eigen(X'*F*X |> Hermitian)
    outC = outputCx ? Cₓ : X*Cₓ
    # Stabilize the sign factor of each column.
    stabilizeSign && for j = 1:size(outC, 2)
       outC[:, j] *= (outC[1,j] < 0 ? -1 : 1)
    end
    outputEmo ? (outC, ϵ) : outC
end


function breakSymOfC(C::Matrix{Float64})
    C2 = copy(C)
    l = min(size(C2)[1], 2)
    C2[1:l, 1:l] .= 0 # Breaking spin symmetry.
    # C2[l, :] .= 0 # Another way.
    (copy(C), C2)
end


function getCfromGWH(::Val{ForUHF}, 
                     S::Matrix{Float64}, Hcore::Matrix{Float64}, X=getX(S)) where {ForUHF}
    l = size(Hcore)[1]
    H = zero(Hcore)
    for j in 1:l, i in 1:l
        H[i,j] = 1.75 * S[i,j] * (Hcore[i,i] + Hcore[j,j]) * 0.5
    end
    C = getCcore(X, H)
    ForUHF ? breakSymOfC(C) : C
end


function getCfromHcore(::Val{ForUHF}, 
                       X::Matrix{Float64}, Hcore::Matrix{Float64}) where {ForUHF}
    C = getCcore(X, Hcore)
    ForUHF ? breakSymOfC(C) : C
end


function getCfromSAD(::Val{ForUHF}, 
                     S::Matrix{Float64}, Hcore::Matrix{Float64}, HeeI::Array{Float64, 4},
                     bs::Vector{<:AbstractGTBasisFuncs}, 
                     nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}, X=getX(S); 
                     scfConfig=SCFconfig((:ADIIS,), (1e-10,))) where {ForUHF}
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
                        (N₁, N₂), h1, HeeI, S, X, getCfromHcore(Val(true), X, h1))
        D₁ += res.D[1]
        D₂ += res.D[2]
        N₁tot += N₁
        N₂tot += N₂
    end
    Dᵀ = D₁ + D₂
    if ForUHF
        getCcore.(Ref(X), getF.(Ref(Hcore), Ref(HeeI), (D₁, D₂), Ref(Dᵀ)))
    else
        getCcore(X, getF(Hcore, HeeI, Dᵀ.*0.5, Dᵀ))
    end
end


const guessCmethods = 
    (  GWH = (ForUHF, S, X, Hcore, _...)->getCfromGWH(ForUHF, S, Hcore, X),
     Hcore = (ForUHF, S, X, Hcore, _...)->getCfromHcore(ForUHF, X, Hcore), 
       SAD = (ForUHF, S, X, Hcore, HeeI, bs, nuc, nucCoords)->
             getCfromSAD(ForUHF, S, Hcore, HeeI, bs, nuc, nucCoords, X))


@inline guessC(::Val{M}, ForUHF::Val{B}, 
               S::Matrix{Float64}, X::Matrix{Float64}, Hcore::Matrix{Float64}, 
               HeeI::Array{Float64, 4}, bs::Vector{<:AbstractGTBasisFuncs}, 
               nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) where {M, B} = 
getfield(guessCmethods, M)(ForUHF, S, X, Hcore, HeeI, bs, nuc, nucCoords)


getD(C::Matrix{Float64}, Nˢ::Int) = @views (C[:,1:Nˢ]*C[:,1:Nˢ]') |> Hermitian |> Array
# Nˢ: number of electrons with the same spin.

@inline getD(X::Matrix{Float64}, F::Matrix{Float64}, Nˢ::Int) = getD(getCcore(X, F), Nˢ)


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
function getEᵀ(Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, C::Matrix{Float64}, N::Int)
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
    Cnew = getCcore(X, Fnew)
    (Cnew, Fnew, Ds[1], Enew) # Fnew is based on latest variables.
end


# RHF
function initializeSCF(Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, C::Matrix{Float64}, 
                       N::Int)
    Nˢ = N÷2
    D = getD(C, Nˢ)
    F = getF(Hcore, HeeI, D)
    E = getE(Hcore, F, D)
    HFtempVars(FunctionType{:RHF}(), Nˢ, C, F, D, E, 2D, 2E)
end

# UHF
function initializeSCF(Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
                       Cs::NTuple{2, Matrix{Float64}}, Ns::NTuple{2, Int})
    Ds = getD.(Cs, Ns)
    Dᵀs = [Ds |> sum]
    Fs = getF.(Ref(Hcore), Ref(HeeI), Ds, Ref(Dᵀs[]))
    Es = getE.(Ref(Hcore), Fs, Ds)
    Eᵀs = [Es |> sum]
    res = HFtempVars.(Ref(FunctionType{:UHF}()), Ns, Cs, Fs, Ds, Es)
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

`methods::NTuple{N, Symbol}`: The applied methods. The available methods and their 
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

julia> SCFconfig((:SD, :ADIIS, :DIIS), (1e-4, 1e-12, 1e-13), Dict(2=>[:solver=>:LCM])
SCFconfig{3}((:SD, :ADIIS, :DIIS), (0.0001, 1.0e-12, 1.0e-13), ((), (:solver => :LCM,), 
()), 1.0e-5)
"""
struct SCFconfig{N} <: ImmutableParameter{SCFconfig, Any}
    methods::NTuple{N, FunctionType}
    intervals::NTuple{N, Float64}
    methodConfigs::NTuple{N, Vector{<:Pair}}
    oscillateThreshold::Float64

    function SCFconfig(methods::NTuple{N, Symbol}, intervals::NTuple{N, Float64}, 
                       configs::Dict{Int, <:Vector{<:Pair}}=Dict(1=>Pair[]);
                       oscillateThreshold::Float64=1e-5) where {N}
        kwPairs = [Pair[] for _=1:N]
        for i in keys(configs)
            kwPairs[i] = configs[i]
        end
        new{N}(FunctionType.(methods), intervals, Tuple(kwPairs), oscillateThreshold)
    end
end


const defaultSCFconfig = SCFconfig((:ADIIS, :DIIS, :ADIIS), (1e-4, 1e-6, 1e-15))


mutable struct HFinterrelatedVars <: HartreeFockintermediateData
    Dtots::Vector{Matrix{Float64}}
    Etots::Vector{Float64}

    HFinterrelatedVars() = new()
    HFinterrelatedVars(Dtots, Etots) = new(Dtots, Etots)
end


"""
    HFtempVars{HFtype} <: HartreeFockintermediateData

The container to store the intermediate values (only of the same spin configuration) for 
each iteration during the Hartree-Fock SCF procedure. 

≡≡≡ Field(s) ≡≡≡

`N::Int`: The number of electrons with the same spin function.

`Cs::Array{Array{Float64, 2}, 1}`: Coefficient matrices.

`Fs::Array{Array{Float64, 2}, 1}`: Fock matrices

`Ds::Array{Array{Float64, 2}, 1}`: Density matrices corresponding 
to only spin configuration. For RHF each elements means (unconverged) 0.5*Dᵀ.

`Es::Array{Float64, 1}`: Part of Hartree-Fock energy corresponding to only spin 
configuration. For RHF each element means (unconverged) 0.5*E0HF.

`shared.Dtots::Array{Array{Float64, 2}, 1}`: The total density 
matrices.

`shared.Etots::Array{Float64, 1}`: The total Hartree-Fock energy.

**NOTE: For UHF, there are 2 `HFtempVars` being updated during the SCF iterations, and 
change the field `shared.Dtots` or `shared.Etots` of one container will affect the other 
one's.**
"""
struct HFtempVars{HFtype} <: HartreeFockintermediateData
    N::Int
    Cs::Vector{Matrix{Float64}}
    Fs::Vector{Matrix{Float64}}
    Ds::Vector{Matrix{Float64}}
    Es::Vector{Float64}
    shared::HFinterrelatedVars
end

HFtempVars(::FunctionType{HFtype}, 
           N::Int, C::Matrix{Float64}, F::Matrix{Float64}, 
           D::Matrix{Float64}, E::Real) where {HFtype} = 
HFtempVars{HFtype}(N, [C], [F], [D], [E], HFinterrelatedVars())

HFtempVars(::FunctionType{HFtype}, 
           N::Int, C::Matrix{Float64}, F::Matrix{Float64}, 
           D::Matrix{Float64}, E::Real, 
           Dtot::Matrix{Float64}, Etot::Real) where {HFtype} = 
HFtempVars{HFtype}(N, [C], [F], [D], [E], HFinterrelatedVars([Dtot], [Etot]))

HFtempVars(::FunctionType{HFtype}, 
           Nˢ::Int, Cs::Vector{Matrix{Float64}}, Fs::Vector{Matrix{Float64}}, 
           Ds::Vector{Matrix{Float64}}, Es::Vector{<:Real}, 
           Dtots::Vector{Matrix{Float64}}, Etots::Vector{<:Real}) where {HFtype} = 
HFtempVars{HFtype}(Nˢ, Cs, Fs, Ds, Es, HFinterrelatedVars(Dtots, Etots))


"""

    HFfinalVars{HFtype} <: HartreeFockFinalValue{HFtype}

The container of the final values after a Hartree-Fock SCF procedure.

≡≡≡ Field(s) ≡≡≡

`E0HF::Float64`: Hartree-Fock energy of the electronic Hamiltonian. 

`N::Int`: The total number of electrons.

`C::Union{Array{Float64, 2}, NTuple{2, Array{Float64, 2}}}`: 
Coefficient matrix(s) for one spin configuration.

`F::Union{Array{Float64, 2}, NTuple{2, Array{Float64, 2}}}`: Fock 
matrix(s) for one spin configuration.

`D::Union{Array{Float64, 2}, NTuple{2, Array{Float64, 2}}}`: Density 
matrix(s) for one spin configuration.

`Emo::Union{Array{Float64, 1}, NTuple{2, Array{Float64, 1}}}`: Energies of molecular 
orbitals.

`occu::Union{Array{Int, 1}, NTuple{2, Array{Int, 1}}}`: occupation numbers of molecular 
orbitals.

`temp::Union{HFtempVars{HFtype}, NTuple{2, HFtempVars{HFtype}}}` the intermediate values.

`isConverged::Bool`: Whether the SCF procedure is converged in the end.
"""
struct HFfinalVars{HFtype} <: HartreeFockFinalValue{HFtype}
    E0HF::Float64
    N::Int
    C::Union{Matrix{Float64}, NTuple{2, Matrix{Float64}}}
    F::Union{Matrix{Float64}, NTuple{2, Matrix{Float64}}}
    D::Union{Matrix{Float64}, NTuple{2, Matrix{Float64}}}
    Emo::Union{Vector{Float64}, NTuple{2, Vector{Float64}}}
    occu::Union{Vector{Int}, NTuple{2, Vector{Int}}}
    temp::Union{HFtempVars{HFtype}, NTuple{2, HFtempVars{HFtype}}}
    isConverged::Bool

    function HFfinalVars(X::Matrix{Float64}, vars::HFtempVars{:RHF}, isConverged::Bool)
        C = vars.Cs[end]
        F = vars.Fs[end]
        D = vars.Ds[end]
        E0HF = vars.shared.Etots[end]
        _, Emo = getCcore(X, F, true)
        N = vars.N
        occu = vcat(2*ones(Int, N), zeros(Int, size(X, 1) - N))
        new{:RHF}(E0HF, 2, C, F, D, Emo, occu, vars, isConverged)
    end

    function HFfinalVars(X::Matrix{Float64}, αβVars::NTuple{2, HFtempVars{:UHF}}, 
                         isConverged::Bool)
        C = last.(getfield.(αβVars, :Cs))
        F = last.(getfield.(αβVars, :Fs))
        D = last.(getfield.(αβVars, :Ds))
        E0HF = αβVars[1].shared.Etots[end]
        Emo = getindex.(getCcore.(Ref(X), F, true), 2)
        Ns = getfield.(αβVars, :N)
        occu = vcat.(ones.(Int, Ns), zeros.(Int, size(X, 1) .- Ns))
        new{:UHF}(E0HF, sum(Ns), C, F, D, Emo, occu, αβVars, isConverged)
    end
end


"""
    runHF(bs::Union{BasisSetData, Array{<:AbstractGTBasisFuncs, 1}}, 
          nuc::Array{String, 1}, 
          nucCoords::Array{<:AbstractArray, 1}, 
          HFtype::Symbol=:RHF; 
          N::Int=getCharge(nuc), 
          initialC::Symbol=:SAD, 
          scfConfig::SCFconfig=defaultSCFconfig, 
          earlyTermination::Bool=true, 
          printInfo::Bool=true, 
          maxSteps::Int=1000) -> HFfinalVars

Main function to run Hartree-Fock in Quiqbox.

=== Positional argument(s) ===

`bs::Union{BasisSetData, Array{<:AbstractGTBasisFuncs, 1}}`: Basis set.

`nuc::Array{String, 1}`: The element symbols of the nuclei for the Molecule.

`nucCoords::Array{<:AbstractArray, 1}`: The coordinates of the nuclei.

`HFtype::Symbol`: Hartree-Fock type. Available values are `:RHF` and `:UHF`.

=== Keyword argument(s) ===

`N::Int`: The total number of electrons.

`initialC::Symbol`: Initial guess of the coefficient matrix(s) C of the molecular orbitals.

`scfConfig::SCFconfig`: SCF iteration configuration.

`earlyTermination::Bool`: Whether automatically early terminate (skip) a convergence method 
when its performance becomes unstable or poor.

`printInfo::Bool`: Whether print out the information of each iteration step.

`maxSteps::Int`: Maximum allowed iteration steps regardless of whether the SCF converges.
"""
function runHF(gtb::BasisSetData{BT}, 
               nuc::Vector{String}, 
               nucCoords::Vector{<:AbstractArray{<:Real}}, 
               ::Val{HFtype}=Val(:RHF); 
               N::Int=getCharge(nuc), 
               initialC::Symbol=:SAD, 
               scfConfig::SCFconfig=defaultSCFconfig, 
               earlyTermination::Bool=true, 
               printInfo::Bool=true, 
               maxSteps::Int=1000) where {HFtype, BT}
    @assert length(nuc) == length(nucCoords)
    @assert length(gtb.basis) >= ceil(N/2)
    @assert N > (HFtype==:RHF) "$(HFtype) requires more than $(HFtype==:RHF) electrons."
    HFtype == :UHF && (N = (N÷2, N-N÷2))
    Hcore = gtb.getHcore(nuc, nucCoords)
    X = getX(gtb.S)
    initialC = guessC(Val(initialC), Val(HFtype==:UHF), gtb.S, X, Hcore, gtb.eeI, gtb.basis, nuc, nucCoords)
    runHFcore(scfConfig, N, Hcore, gtb.eeI, gtb.S, X, initialC; 
              printInfo, maxSteps, earlyTermination)
end

runHF(a1::Vector{<:AbstractGTBasisFuncs}, a2, a3, HFtype::Symbol; kws...) = 
runHF(a1, a2, a3, Val(HFtype); kws...)

runHF(bs::Vector{<:AbstractGTBasisFuncs}, args...; kws...) = 
runHF(GTBasis(bs), args...; kws...)


"""

    runHFcore(scfConfig::SCFconfig{L}, 
              N::Union{NTuple{2, Int}, Int}, 
              Hcore::Array{Float64, 2}, 
              HeeI::Array{Float64, 4}, 
              S::Array{Float64, 2}, 
              X::Array{Float64, 2}=getX(S), 
              C::Union{Array{Float64, 2}, NTuple{2, Array{Float64, 2}}}=
              getCfromGWH(Val(length(N)==2), S, Hcore, X); 
              printInfo::Bool=false, 
              maxSteps::Int=1000, 
              earlyTermination::Bool=true) where {L}

The core function of `runHF`.

=== Positional argument(s) ===

`scfConfig::SCFconfig`: SCF iteration configuration.

`N::Union{NTuple{2, Int}, Int}`: The total number of electrons or the numbers of electrons 
with different spins respectively. When the latter is input, an UHF is performed.

`Hcore::Array{Float64, 2}`: Core Hamiltonian of electronic Hamiltonian.

`HeeI::Array{Float64, 4}`: The electron-electron interaction Hamiltonian which includes both the 
Coulomb interactions and the Exchange Correlations.

`S::Array{Float64, 2}`: Overlap matrix of the corresponding basis set.

`X::Array{Float64, 2}`: Orthogonal transformation matrix of S. Default value is S^(-0.5).

`C::Union{Array{Float64, 2}, NTuple{2, Array{Float64, 2}}}`: Initial guess of the 
coefficient matrix(s) C of the molecular orbitals.

=== Keyword argument(s) ===

`earlyTermination::Bool`: Whether automatically early terminate (skip) a convergence method 
when its performance becomes unstable or poor.

`printInfo::Bool`: Whether print out the information of each iteration step.

`maxSteps::Int`: Maximum allowed iteration steps regardless of whether the SCF converges.
"""
function runHFcore(scfConfig::SCFconfig{L}, 
                   N::Union{NTuple{2, Int}, Int}, 
                   Hcore::Matrix{Float64}, 
                   HeeI::Array{Float64, 4}, 
                   S::Matrix{Float64}, 
                   X::Matrix{Float64}=getX(S), 
                   C::T=getCfromGWH(Val(length(N)==2), S, Hcore, X); 
                   printInfo::Bool=false, 
                   maxSteps::Int=1000, 
                   earlyTermination::Bool=true) where 
                  {L, T<:Union{Matrix{Float64}, NTuple{2, Matrix{Float64}}}}
    vars = initializeSCF(Hcore, HeeI, C, N)
    Etots = (vars isa Tuple) ? vars[1].shared.Etots : vars.shared.Etots
    HFtypeStr =  N isa Int ? "RHF" : "UHF"
    printInfo && println(rpad(HFtypeStr*" | Initial Gauss", 22), "E = $(Etots[end])")
    isConverged = true
    EtotMin = Etots[]
    i = 0
    for (m, kws, breakPoint, l) in 
        zip(scfConfig.methods, scfConfig.methodConfigs, scfConfig.intervals, 1:L)

        while true
            i += 1
            i <= maxSteps || (isConverged = false) || break

            res = HFcore(m, N, Hcore, HeeI, S, X, vars; kws...)
            pushHFtempVars!(vars, res)

            printInfo && (i % floor(log(4, i) + 1) == 0 || i == maxSteps) && 
            println(rpad("Step $i", 10), rpad("#$l ($(m.f))", 12), "E = $(Etots[end])")

            abs(Etots[end]-Etots[end-1]) > breakPoint || (isConverged = true) && break

            flag, Std = isOscillateConverged(Etots, 10^(log(10, breakPoint)÷2))

            flag && (isConverged = Std > scfConfig.oscillateThreshold ? false : true; break)

            if earlyTermination && (Etots[end] - EtotMin) / abs(EtotMin) > 0.2
                printInfo && println("Early termination of ", m, 
                                     " due to the poor performance.")
                isConverged = false
                break
            end
        end

    end
    negStr = isConverged ? "is " : "has not "
    printInfo && println("The SCF procedure ", negStr, "converged.\n")
    HFfinalVars(X, vars, isConverged)
end


function SDcore(Nˢ::Int, Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
                X::Matrix{Float64}, F::Matrix{Float64}, D::Matrix{Float64}; 
                dampingStrength::Float64=0.0, _kws...)
    @assert 0 <= dampingStrength <= 1 "The range of `dampingStrength`::Float64 is [0,1]."
    Dnew = getD(X, F, Nˢ)
    (1-dampingStrength)*Dnew + dampingStrength*D
end


function xDIIScore(::Val{M}, Nˢ::Int, Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
                   S::Matrix{Float64}, X::Matrix{Float64}, Fs::Vector{Matrix{Float64}}, 
                   Ds::Vector{Matrix{Float64}},Es::Vector{Float64}; 
                   DIISsize::Int=15, solver::Symbol=:ADMM, 
                   _kws...) where {M}
    DIISmethod, convexConstraint, permuteData = getfield(DIISmethods, M)
    is = permuteData ? sortperm(Es) : (:)
    ∇s = (@view Fs[is])[1:end .> end-DIISsize]
    Ds = (@view Ds[is])[1:end .> end-DIISsize]
    Es = (@view Es[is])[1:end .> end-DIISsize]
    vec, B = DIISmethod(∇s, Ds, Es, S)
    c = constraintSolver(vec, B, solver; convexConstraint)
    grad = c.*∇s |> sum
    getD(X, grad |> Hermitian |> Array, Nˢ) # grad == F.
end

const DIISmethods = ( DIIS = ((∇s, Ds,  _, S)->DIIScore(∇s, Ds, S),   false, true),
                     EDIIS = ((∇s, Ds, Es, _)->EDIIScore(∇s, Ds, Es), true, false),
                     ADIIS = ((∇s, Ds,  _, _)->ADIIScore(∇s, Ds),     true, false))


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
    vec = [dot(D - Ds[end], ∇s[end]) for D in Ds]
    for j=1:len, i=1:len
        B[i,j] = dot(Ds[i]-Ds[len], ∇s[j]-∇s[len])
    end
    vec, B
end


function DIIScore(∇s::Vector{Matrix{Float64}}, Ds::Vector{Matrix{Float64}}, 
                  S::Matrix{Float64})
    len = length(Ds)
    B = ones(len, len)
    vec = zeros(len)
    for j=1:len, i=1:len
        B[i,j] = dot(∇s[i]*Ds[i]*S - Ds[i]*S*∇s[i], ∇s[j]*Ds[j]*S - Ds[j]*S*∇s[j])
    end
    vec, B
end


@inline SD(Nˢ, Hcore, HeeI, _dm::Any, X, tVars; kws...) = 
        SDcore(Nˢ, Hcore, HeeI, X, tVars.Fs[end], tVars.Ds[end]; kws...)

@inline function xDIIS(::Val{M}) where {M}
    @inline (Nˢ, Hcore, HeeI, S, X, tVars; kws...) ->
            xDIIScore(Val(M), Nˢ, Hcore, HeeI, S, X, tVars.Fs, tVars.Ds, tVars.Es; kws...)
end

const SCFmethodSelector = 
      (SD=SD, DIIS=xDIIS(Val(:DIIS)), ADIIS=xDIIS(Val(:ADIIS)), EDIIS=xDIIS(Val(:EDIIS)))


# RHF
@inline function HFcore(::FunctionType{M}, N::Int, 
                        Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
                        S::Matrix{Float64}, X::Matrix{Float64}, 
                        rVars::HFtempVars{:RHF}; kws...) where {M}
    D = getfield(SCFmethodSelector, M)(N÷2, Hcore, HeeI, S, X, rVars; kws...)
    partRes = getCFDE(Hcore, HeeI, X, D)
    partRes..., 2D, 2partRes[end]
end

@inline function pushHFtempVars!(rVars::HFtempVars, 
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
@inline function HFcore(::FunctionType{M}, Ns::NTuple{2, Int}, 
                        Hcore::Matrix{Float64}, HeeI::Array{Float64, 4}, 
                        S::Matrix{Float64}, X::Matrix{Float64}, 
                        uVars::NTuple{2, HFtempVars{:UHF}}; kws...) where {M}
    Ds = getfield(SCFmethodSelector, M).(Ns, Ref(Hcore), Ref(HeeI), 
                                                 Ref(S), Ref(X), uVars; kws...)
    Dᵀnew = Ds |> sum
    partRes = getCFDE.(Ref(Hcore), Ref(HeeI), Ref(X), Ds, Ref(Dᵀnew))
    Eᵀnew = partRes[1][end] + partRes[2][end]
    (partRes[1]..., Dᵀnew, Eᵀnew), (partRes[2]..., Dᵀnew, Eᵀnew)
end

@inline function pushHFtempVars!(uVars::NTuple{2, HFtempVars{:UHF}}, 
                                 res::NTuple{2, Tuple{Matrix{Float64}, Matrix{Float64}, 
                                                      Matrix{Float64}, Float64, 
                                                      Matrix{Float64}, Float64}})
    pushHFtempVars!.(uVars, res)
    pop!(uVars[1].shared.Dtots)
    pop!(uVars[1].shared.Etots)
end


@inline function popHFtempVars!(rVars::HFtempVars)
    pop!(rVars.Cs)
    pop!(rVars.Fs)
    pop!(rVars.Ds)
    pop!(rVars.Es)
    pop!(rVars.shared.Dtots)
    pop!(rVars.shared.Etots)
end

@inline function popHFtempVars!(uVars::NTuple{2, HFtempVars{:UHF}})
    for field in [:Cs, :Fs, :Ds, :Es] pop!.(getfield.(uVars, field)) end
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
    settings = SeparableOptimization.Settings(; ρ=ones(1), σ=ones(len))
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


const ConstraintSolvers = (ADMM=ADMMSolver, LCM=CMSolver)

constraintSolver(vec::Vector{Float64}, B::Matrix{Float64}, solver::Symbol=:ADMM; 
                 convexConstraint::Bool=true) = 
getfield(ConstraintSolvers, solver)(vec, B; convexConstraint)