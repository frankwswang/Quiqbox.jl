export runHF, runHFcore, SCFconfig

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
              outputCx::Bool=false, outputEmo::Bool=false) where 
             {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB}
    ϵ, Cₓ = eigen(X'*F*X |> Hermitian, sortby=x->x)
    outC = outputCx ? Cₓ : X*Cₓ
    outputEmo ? (outC, ϵ) : outC
end

function getCfromGWH(S::Matrix{T1}, Hcore::Matrix{T2}; K=1.75, X=getX(S), _kws...) where
         {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB}
    l = size(Hcore)[1]
    H = zero(Hcore)
    for i in 1:l, j in 1:l
        H[i,j] = K * S[i,j] * (Hcore[i,i] + Hcore[j,j]) / 2
    end
    getC(X, H)
end

getCfromHcore(S::Matrix{T1}, Hcore::Matrix{T2}; X=getX(S), _kws...) where 
             {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB} = 
             getC(X, Hcore)

const guessCmethods = Dict(:GWH=>getCfromGWH, :Hcore=>getCfromHcore)


guessC(S::Matrix{T1}, Hcore::Matrix{T2}; method::Symbol=:GWH, kws...) where 
      {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB} = 
      guessCmethods[method](S, Hcore; kws...)


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
getF(Hcore::Matrix{T1}, HeeI::Array{T2, 4}, D::Matrix{T3}, Dᵀ::Matrix{T4}) where 
    {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB, TelLB<:T4<:TelUB} = 
getF(Hcore, getG(HeeI, D, Dᵀ))

# UHF
getF(Hcore::Matrix{T1}, HeeI::Array{T2, 4}, D::Matrix{T3}) where 
    {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB} = 
getF(Hcore, getG(HeeI, D))


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
                       Cs::Union{Matrix{T3}, NTuple{2, Matrix{T3}}}, 
                       Ns::NTuple{2, Int}) where 
                       {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, TelLB<:T3<:TelUB}
    if Cs isa Matrix{<:Number}
        C2 = copy(Cs)
        l = min(size(C2)[1], 2)
        C2[1:l, 1:l] .= 0 # Breaking spin symmetry.
        # C2[l, :] .= 0 # Another way.
        Cs = (copy(Cs), C2)
    end
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


struct SCFconfig{N}
    methods::NTuple{N, Symbol}
    intervals::NTuple{N, Float64}
    methodConfigs::NTuple{N, <:Vector{<:Pair}}
    oscillateThreshold::Float64

    function SCFconfig(methods::Vector{Symbol}, intervals::Vector{Float64}, 
                    #    configs::Dict{Int, <:Vector{<:Pair}}; printInfo=true)
                       configs::Dict{Int, <:Vector{<:Any}}=Dict(1=>[]);
                       oscillateThreshold::Float64=1e-5)
        l = length(methods)
        kwPairs = [Pair[] for _=1:l]
        for i in keys(configs)
            kwPairs[i] = configs[i]
        end
        new{length(methods)}(Tuple(methods), Tuple(intervals), Tuple(kwPairs), 
                             oscillateThreshold)
    end
end


mutable struct HFinterrelatedVars <: HartreeFockintermediateData
    Dtots::Vector{Matrix{T}} where {TelLB<:T<:TelUB}
    Etots::Vector{Float64}

    HFinterrelatedVars() = new()
    HFinterrelatedVars(Dtots) = new(Dtots)
    HFinterrelatedVars(Dtots, Etots) = new(Dtots, Etots)
end


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


struct HFfinalVars{T, N, Nb} <: HartreeFockFinalValue{T}
    E0HF::Union{Float64, NTuple{2, Float64}}
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


const defaultSCFconfig = SCFconfig([:ADIIS, :DIIS, :ADIIS], [1e-4, 1e-6, 1e-10])


function runHF(bs::Vector{<:AbstractFloatingGTBasisFunc}, 
               mol::Vector{String}, 
               nucCoords::Vector{<:Vector{<:Real}}, 
               N::Int=getCharge(mol); 
               initialC::Union{Matrix{T}, NTuple{2, Matrix{T}}, Symbol}=:GWH, 
               HFtype::Symbol=:RHF, 
               scfConfig::SCFconfig=defaultSCFconfig, 
               earlyTermination::Bool=true, 
               printInfo::Bool=true, 
               maxSteps::Int=1000) where {TelLB<:T<:TelUB}
    @assert length(mol) == length(nucCoords)
    @assert (basisSize(bs) |> sum) >= ceil(N/2)
    gtb = GTBasis(bs)
    runHF(gtb, mol, nucCoords, N; initialC, scfConfig, 
          HFtype, printInfo, maxSteps, earlyTermination)
end

function runHF(gtb::BasisSetData, 
               mol::Vector{String}, 
               nucCoords::Vector{<:Vector{<:Real}}, 
               N::Union{NTuple{2, Int}, Int}=getCharge(mol); 
               initialC::Union{Matrix{T}, NTuple{2, Matrix{T}}, Symbol}=:GWH, 
               HFtype::Symbol=:RHF, 
               scfConfig::SCFconfig=defaultSCFconfig, 
               earlyTermination::Bool=true, 
               printInfo::Bool=true, 
               maxSteps::Int=1000) where {TelLB<:T<:TelUB}
    @assert length(mol) == length(nucCoords)
    @assert typeof(gtb).parameters[1] >= ceil(N/2)
    Hcore = gtb.getHcore(mol, nucCoords)
    X = getX(gtb.S)
    initialC isa Symbol && (initialC = guessC(gtb.S, Hcore; X, method=initialC))
    runHFcore(N, Hcore, gtb.eeI, gtb.S, X, initialC; 
              scfConfig, printInfo, maxSteps, HFtype, earlyTermination)
end


function runHFcore(N::Union{NTuple{2, Int}, Int}, 
                   Hcore::Matrix{T1}, 
                   HeeI::Array{T2, 4}, 
                   S::Matrix{T3}, 
                   X::Matrix{T4}=getX(S), 
                   C::Union{Matrix{T5}, NTuple{2, Matrix{T5}}}=guessC(S, Hcore; X);
                   HFtype::Symbol=:RHF,  
                   scfConfig::SCFconfig{L}, 
                   earlyTermination::Bool=true, 
                   printInfo::Bool=true, 
                   maxSteps::Int=1000) where {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB, 
                   TelLB<:T3<:TelUB, TelLB<:T4<:TelUB, TelLB<:T5<:TelUB, L}
    @assert maxSteps > 0
    HFtype == :UHF && (N isa Int) && (N = (N÷2, N-N÷2))
    vars = initializeSCF(Hcore, HeeI, C, N)
    Etots = (vars isa Tuple) ? vars[1].shared.Etots : vars.shared.Etots
    printInfo && println(rpad("$(HFtype) Initial Gauss", 22), "E = $(Etots[end])")
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
            (printInfo && println("Early termination of method ", method, 
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
                   DIISsize::Int=15, solver=:default, 
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

const SCFmethods = Dict( [:SD, :DIIS,        :ADIIS,        :EDIIS] .=> 
                         [ SD, xDIIS(:DIIS), xDIIS(:ADIIS), xDIIS(:EDIIS)])


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
    D = SCFmethods[SCFmethod](N÷2, Hcore, HeeI, S, X, rVars; kws...)
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
    Ds = SCFmethods[SCFmethod].(Ns, Ref(Hcore), Ref(HeeI), 
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


const ConstraintSolvers = Dict(:default=>ADMMSolver, :Direct=>CMSolver)

constraintSolver(vec::Vector{T1}, B::Matrix{T2}, 
                 solver::Symbol=:default; convexConstraint::Bool=true) where 
                {TelLB<:T1<:TelUB, TelLB<:T2<:TelUB} = 
ConstraintSolvers[solver](vec, B; convexConstraint)