export runHF, runHFcore

using LinearAlgebra: dot, Hermitian
using PiecewiseQuadratics: indicator
using SeparableOptimization, Convex, COSMO

getXcore(::Val{1}, S) = S^(-0.5) |> Array

getX(S; method::Int=1) = getXcore(Val(method), S)


function getC(X, F; outputCx::Bool=false, outputEmo::Bool=false)
    ϵ, Cₓ = eigen(X'*F*X |> Hermitian, sortby=x->x)
    outC = outputCx ? Cₓ : X*Cₓ
    outputEmo ? (outC, ϵ) : outC
end


getD(C, Nˢ) = @views (C[:,1:Nˢ]*C[:,1:Nˢ]') |> Hermitian |> Array # Nˢ: number of electrons with the same spin

getD(X, F, Nˢ) = getD(getC(X, F), Nˢ)


function getGcore(HeeI, DJ, DK)
    G = zero(DJ)
    l = size(G)[1]
    @views for μ = 1:l, ν = 1:l
        G[μ, ν] = dot(transpose(DJ), HeeI[μ,ν,:,:]) - dot(DK, HeeI[μ,:,:,ν]) # fastest
    end
    G |> Hermitian |> Array
end

@inline getG(::Val{:RHF}, HeeI, D, _=nothing) = getGcore(HeeI, 2D, D)

@inline getG(::Val{:UHF}, HeeI, D, Dᵀ) = getGcore(HeeI, Dᵀ, D)


@inline getF(Hcore, G) = Hcore + G

@inline getF(::Val{:RHF}, Hcore, HeeI, D, _=nothing) = getF(Hcore, getG(Val(:RHF), HeeI, D))

@inline getF(::Val{:UHF}, Hcore, HeeI, D, Dᵀ) = getF(Hcore, getG(Val(:UHF), HeeI, D, Dᵀ))


@inline getE(Hcore, F, D) = dot(transpose(D), 0.5*(Hcore + F))


@inline getEᵀcore(::Val{:RHF}, Hcore, F, D, _1=nothing, _2=nothing) = 2*getE(Hcore, F, D)

@inline getEᵀcore(::Val{:UHF}, Hcore, Fᵅ, Dᵅ, Fᵝ, Dᵝ) = getE(Hcore, Fᵅ, Dᵅ) + getE(Hcore, Fᵝ, Dᵝ)

function getEᵀ(Hcore, HeeI, C::Array{<:Number, 2}, N::Int) # RHF
    D = getD(C, N÷2)
    F = getF(Val(:RHF), Hcore, HeeI, D)
    getEᵀcore(Val(:RHF), Hcore, F, D)
end

function getEᵀ(Hcore, HeeI, (Cᵅ, Cᵝ)::NTuple{2, Array{<:Number, 2}}, (Nᵅ, Nᵝ)::NTuple{2, Int}) # UHF
    Dᵅ = getD(Cᵅ, Nᵅ)
    Dᵝ = getD(Cᵝ, Nᵝ)
    Dᵀ = Dᵅ + Dᵝ
    Fᵅ = getF(Val(:UHF), Hcore, HeeI, Dᵅ, Dᵀ)
    Fᵝ = getF(Val(:UHF), Hcore, HeeI, Dᵝ, Dᵀ)
    getEᵀcore(Val(:UHF), Hcore, Fᵅ, Dᵅ, Fᵝ, Dᵝ)
end


function initializeSCF(Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
                       C::Array{<:Number, 2}, N::Int) # RHF
    Nˢ = N÷2
    D = getD(C, Nˢ)
    F = getF(Val(:RHF), Hcore, HeeI, D)
    E = getE(Hcore, F, D)
    HFtempVars(:RHF, Nˢ, C, F, D, E, 2D, 2E)
end

function initializeSCF(Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
                       Cs::Union{Array{<:Number, 2}, NTuple{2, Array{<:Number, 2}}}, 
                       Ns::NTuple{2, Int}) # UHF
    if Cs isa Array{<:Number, 2}
        C2 = copy(Cs)
        l = min(size(C2)[1], 2)
        C2[1:l, 1:l] .= 0 # Breaking spin symmetry.
        # C2[l, :] .= 0 # Another way.
        Cs = (copy(Cs), C2)
    end
    Ds = getD.(Cs, Ns)
    Dᵀs = [Ds |> sum]
    Fs = getF.(Val(:UHF), Ref(Hcore), Ref(HeeI), Ds, Ref(Dᵀs[]))
    Es = getE.(Ref(Hcore), Fs, Ds)
    Eᵀs = [Es |> sum]
    res = HFtempVars.(:UHF, Ns, Cs, Fs, Ds, Es)
    res[1].shared.Dtots = res[2].shared.Dtots = Dᵀs
    res[1].shared.Etots = res[2].shared.Etots = Eᵀs
    res[1], res[2]
end


struct SCFconfig{N}
    methods::NTuple{N, Symbol}
    intervals::NTuple{N, Float64}
    methodConfigs::NTuple{N, <:Array{<:Pair, 1}}
    oscillateThreshold::Float64

    function SCFconfig(methods::Array{Symbol, 1}, intervals::Array{Float64, 1}, 
                    #    configs::Dict{Int, <:Array{<:Pair, 1}}; printInfo=true)
                       configs::Dict{Int, <:Array{<:Any, 1}}=Dict(1=>[]);
                       oscillateThreshold::Float64=1e-5)
        l = length(methods)
        kwPairs = [Pair[] for _=1:l]
        for i in keys(configs)
            kwPairs[i] = configs[i]
        end
        new{length(methods)}(Tuple(methods), Tuple(intervals), Tuple(kwPairs), oscillateThreshold)
    end
end


mutable struct HFinterrelatedVars <: HartreeFockintermediateData
    Dtots::Array{<:Array{<:Number, 2}, 1}
    Etots::Array{<:Real, 1}

    HFinterrelatedVars() = new()
    HFinterrelatedVars(Dtots) = new(Dtots)
    HFinterrelatedVars(Dtots, Etots) = new(Dtots, Etots)
end


struct HFtempVars{HFtype, N} <: HartreeFockintermediateData
    Cs::Array{<:Array{<:Number, 2}, 1}
    Fs::Array{<:Array{<:Number, 2}, 1}
    Ds::Array{<:Array{<:Number, 2}, 1}
    Es::Array{<:Real, 1}
    shared::HFinterrelatedVars

    HFtempVars(HFtype::Symbol, N, C, F, D, E, vars...) = 
    new{HFtype, N}([C], [F], [D], [E], HFinterrelatedVars([[x] for x in vars]...))
    
    HFtempVars(HFtype::Symbol, N, 
               Cs::Array{<:Array{<:Number, 2}, 1}, Fs::Array{<:Array{<:Number, 2}, 1}, 
               Ds::Array{<:Array{<:Number, 2}, 1}, Es::Array{Float64, 1}, 
               Dtots::Array{<:Array{<:Number, 2}, 1}, Etots::Array{Float64, 1}) = 
    new{HFtype, N}(Cs, Fs, Ds, Es, HFinterrelatedVars(Dtots, Etots))
end


struct HFfinalVars{T, N, Nb} <: HartreeFockFinalValue{T}
    E0HF::Union{Real, NTuple{2, Real}}
    C::Union{Array{<:Number, 2}, NTuple{2, Array{<:Number, 2}}}
    F::Union{Array{<:Number, 2}, NTuple{2, Array{<:Number, 2}}}
    D::Union{Array{<:Number, 2}, NTuple{2, Array{<:Number, 2}}}
    Emo::Union{Array{<:Number, 1}, NTuple{2, Array{<:Number, 1}}}
    occu::Union{Array{Int, 1}, NTuple{2, Array{Int, 1}}}
    temp::Union{HFtempVars{T}, NTuple{2, HFtempVars{T}}}
    isConverged::Bool

    function HFfinalVars(X, vars::HFtempVars{:RHF}, convBool)
        C = vars.Cs[end]
        F = vars.Fs[end]
        D = vars.Ds[end]
        E0HF = vars.shared.Etots[end]
        _, Emo = getC(X, F, outputEmo=true)
        Nˢ = typeof(vars).parameters[2]
        occu = vcat(2*ones(Int, Nˢ), zeros(Int, size(X, 1) - Nˢ))
        temp = vars
        new{:RHF, 2Nˢ, length(Emo)}(E0HF, C, F, D, Emo, occu, temp, convBool)
    end

    function HFfinalVars(X, (αVars, βVars)::NTuple{2, HFtempVars{:UHF}}, convBool)
        C = (αVars.Cs[end], βVars.Cs[end])
        F = (αVars.Fs[end], βVars.Fs[end])
        D = (αVars.Ds[end], βVars.Ds[end])
        E0HF = αVars.shared.Etots[end]
        res = getC.(Ref(X), F, outputEmo=true)
        Emo = getindex.(res, 2)
        Nˢs = (typeof(αVars).parameters[2], typeof(βVars).parameters[2]) 
        occu = vcat.(ones.(Int, Nˢs), zeros.(Int, size(X, 1) .- Nˢs))
        temp = (αVars, βVars)
        new{:UHF, Nˢs |> sum, length(Emo[1])}(E0HF, C, F, D, Emo, occu, temp, convBool)
    end
end


guessC(S, Hcore; method::Symbol=:Hcore, kws...) = guessCcore(Val(method), S, Hcore; kws...)

function guessCcore(::Val{:GWH}, S, Hcore; K=1.75, X=getX(S), _kws...)
    l = size(Hcore)[1]
    H = zero(Hcore)
    for i in 1:l, j in 1:l
        H[i,j] = K * S[i,j] * (Hcore[i,i] + Hcore[j,j]) / 2
    end
    getC(X, H)
end

guessCcore(::Val{:Hcore}, S, Hcore; X=getX(S), _kws...) = getC(X, Hcore)


function runHF(bs::Array{<:AbstractFloatingGTBasisFunc, 1}, 
               mol, nucCoords, N=getCharge(mol); initialC=:default, getXmethod=getX, 
               scfConfig=SCFconfig([:ADIIS, :DIIS, :SD], [1e-4, 1e-8, 1e-12]), 
               HFtype=:RHF, printInfo::Bool=true, maxSteps::Int=1000)
    @assert length(mol) == length(nucCoords)
    @assert (basisSize(bs) |> sum) >= ceil(N/2)
    gtb = GTBasis(bs)
    runHF(gtb, mol, nucCoords, N; initialC, scfConfig, getXmethod, HFtype, printInfo, maxSteps)
end

function runHF(gtb::BasisSetData, 
               mol::Array{String, 1}, 
               nucCoords::Array{<:Array{<:Real, 1}, 1}, 
               N::Union{NTuple{2, Int}, Int}=getCharge(mol); 
               initialC::Union{Array{<:Number, 2}, NTuple{2, Array{<:Number, 2}}, Symbol}=:default, 
               getXmethod::Function=getX, 
               scfConfig::SCFconfig=SCFconfig([:ADIIS, :DIIS, :SD], [1e-4, 1e-8, 1e-12]), 
               HFtype::Symbol=:RHF, printInfo::Bool=true, maxSteps::Int=1000)
    @assert length(mol) == length(nucCoords)
    @assert typeof(gtb).parameters[1] >= ceil(N/2)
    Hcore = gtb.getHcore(mol, nucCoords)
    X = getXmethod(gtb.S)
    initialC == :default && (initialC = guessC(gtb.S, Hcore; X))
    runHFcore(Val(HFtype), N, initialC, Hcore, gtb.eeI, gtb.S, X; scfConfig, printInfo, maxSteps)
end

runHFcore(::Val{:RHF}, N::Int, C, Hcore, HeeI, S, X; scfConfig, printInfo::Bool=true, maxSteps::Int=1000) = 
runHFcore(N, C, Hcore, HeeI, S, X; scfConfig, printInfo, maxSteps)

runHFcore(::Val{:UHF}, N, C, Hcore, HeeI, S, X; scfConfig, printInfo::Bool=true, maxSteps::Int=1000) = 
runHFcore((N isa Tuple ? N : (N÷2, N-N÷2)), C, Hcore, HeeI, S, X; scfConfig, printInfo, maxSteps)

function runHFcore(N::Union{NTuple{2, Int}, Int}, C::Union{Array{<:Number, 2}, NTuple{2, Array{<:Number, 2}}}, 
                   Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
                   S::Array{<:Number, 2}, X::Array{<:Number, 2}; 
                   scfConfig::SCFconfig{L}, printInfo::Bool=true, maxSteps::Int=1000) where {L}
    @assert maxSteps > 0
    vars = initializeSCF(Hcore, HeeI, C, N)
    Etots = (vars isa Tuple) ? vars[1].shared.Etots : vars.shared.Etots
    HFtype = length(N) == 1 ? :RHF : :UHF
    printInfo && println(rpad("$(HFtype) Initial Gauss", 22), "E = $(Etots[end])")
    isConverged = true
    for (method, kws, breakPoint, i) in zip(scfConfig.methods, scfConfig.methodConfigs, scfConfig.intervals, 1:L)
        while true
            iStep = length(Etots)
            iStep <= maxSteps || (isConverged = false) || break
            SCFcore!(method, N, Hcore, HeeI, S, X, vars; kws...)
            printInfo && (iStep % floor(log(4, iStep) + 1) == 0 || iStep == maxSteps) && 
            println(rpad("Step $(iStep)", 10), rpad("#$(i) ($(method))", 12), "E = $(Etots[end])")
            abs(Etots[end]-Etots[end-1]) > breakPoint || (isConverged = true) && break
            flag, Std = isOscillateConverged(Etots, 10^(log(10, breakPoint)÷2), returnStd=true)
            flag && (isConverged = Std > scfConfig.oscillateThreshold ? false : true; break)
        end
    end
    negStr = isConverged ? "is " : "has not "
    printInfo && println("The SCF procedure ", negStr, "converged.\n")
    HFfinalVars(X, vars, isConverged)
end


function SDcore(ValHF::Val, Nˢ::Int, Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
                X::Array{<:Number, 2}, 
                F::Array{<:Number, 2}, 
                D::Array{<:Number, 2}; 
                dampingStrength::Float64=0.0, _kws...)
    @assert 0 <= dampingStrength <= 1 "The range of `dampingStrength`::Float64 is [0,1]."
    Dnew = getD(X, F, Nˢ)
    Dnew = (1-dampingStrength)*Dnew + dampingStrength*D
    f = function (Dᵀnew)
        Fnew = getF(ValHF, Hcore, HeeI, Dnew, Dᵀnew)
        Enew = getE(Hcore, Fnew, Dnew)
        Cnew = getC(X, Fnew)
        [Cnew, Fnew, Dnew, Enew] # Fnew is based on latest variables.
    end
    f, Dnew
end


function xDIIScore(ValHF::Val, ValDIIS::Val, Nˢ::Int, Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
                   S::Array{<:Number, 2}, X::Array{<:Number, 2}, 
                   Fs::Array{<:Array{<:Number, 2}, 1}, 
                   Es::Array{Float64, 1}, 
                   Ds::Array{<:Array{<:Number, 2}, 1}; 
                   DIISsize::Int=15, solver::Symbol=:default, _kws...)
    ∇s = Fs[1:end .> end-DIISsize]
    Es = Es[1:end .> end-DIISsize]
    Ds = Ds[1:end .> end-DIISsize]
    DIISmethod, convexConstraint = DIISselector(ValDIIS)
    vec, B = DIISmethod(Ds, ∇s, Es, S)
    c = constraintSolver(vec, B; convexConstraint, solver)
    grad = c.*∇s |> sum
    Dnew = getD(X, grad |> Hermitian |> Array, Nˢ) # grad == F.
    f = function (Dᵀnew)
        Fnew = getF(ValHF, Hcore, HeeI, Dnew, Dᵀnew)
        Enew = getE(Hcore, Fnew, Dnew)
        Cnew = getC(X, Fnew)
        [Cnew, Fnew, Dnew, Enew] # Fnew is based on latest variables.
    end
    f, Dnew
end


DIISselector(::Val{:DIIS}) = ((Ds, ∇s, _, S) -> DIIScore(Ds, ∇s, S), false)

DIISselector(::Val{:EDIIS}) = ((Ds, ∇s, Es, _) -> EDIIScore(Ds, ∇s, Es), true)

DIISselector(::Val{:ADIIS}) = ((Ds, ∇s, _, _) -> ADIIScore(Ds, ∇s), true)


function EDIIScore(Ds, ∇s, Es)
    len = length(Ds)
    B = ones(len, len)
    for i=1:len, j=1:len
        B[i,j] = -dot(Ds[i]-Ds[j], ∇s[i]-∇s[j])
    end
    Es, B
end


function ADIIScore(Ds, ∇s)
    len = length(Ds)
    B = ones(len, len)
    vec = [dot(D - Ds[end], ∇s[end]) for D in Ds]
    for i=1:len, j=1:len
        B[i,j] = dot(Ds[i]-Ds[len], ∇s[j]-∇s[len])
    end
    vec, B
end


function DIIScore(Ds, ∇s, S)
    len = length(Ds)
    B = ones(len, len)
    vec = zeros(len)
    for i=1:len, j=1:len
        B[i,j] = dot(∇s[i]*Ds[i]*S - Ds[i]*S*∇s[i], ∇s[j]*Ds[j]*S - Ds[j]*S*∇s[j])
    end
    vec, B
end


const SCFmethods = Dict(:SD=>SDcore, :DIIS=>xDIIScore, :ADIIS=>xDIIScore, :EDIIS=>xDIIScore)

SCFmethodSelector(sym::Symbol) = SCFmethodSelector(Val(sym), SCFmethods[sym])

SCFmethodSelector(::Val{:SD}, ::typeof(SDcore)) = 
    (ValHF, Nˢ, Hcore, HeeI, _dm, X, tmpVars::HFtempVars; kws...) -> 
    SDcore(ValHF, Nˢ, Hcore, HeeI, X, tmpVars.Fs[end], tmpVars.Ds[end]; kws...)

SCFmethodSelector(ValDIIS::Val, ::typeof(xDIIScore)) = 
    (ValHF, Nˢ, Hcore, HeeI, S, X, tmpVars::HFtempVars; kws...) -> 
    xDIIScore(ValHF, ValDIIS, Nˢ, Hcore, HeeI, S, X, tmpVars.Fs, tmpVars.Es, tmpVars.Ds; kws...)


function SCFcore!(SCFmethod::Symbol, N::Int, Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
                  S::Array{<:Number, 2}, X::Array{<:Number, 2}, rVars::HFtempVars{:RHF}; kws...) # RHF
    f, D = SCFmethodSelector(SCFmethod)(Val(:RHF), N÷2, Hcore, HeeI, S, X, rVars; kws...)
    Dᵀnew = 2D
    Cnew, Fnew, Dnew, Enew = f(Dᵀnew)
    Eᵀnew = 2Enew
    push!(rVars.Cs, Cnew)
    push!(rVars.Fs, Fnew)
    push!(rVars.Ds, Dnew)
    push!(rVars.Es, Enew)
    push!(rVars.shared.Dtots, Dᵀnew)
    push!(rVars.shared.Etots, Eᵀnew)
end

function SCFcore!(SCFmethod::Symbol, Ns::NTuple{2, Int}, Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
                  S::Array{<:Number, 2}, X::Array{<:Number, 2}, uVars::NTuple{2, HFtempVars{:UHF}}; kws...) # UHF
    fs = Function[]
    Dᵀ = uVars[1].shared.Dtots
    Eᵀ = uVars[1].shared.Etots
    Dᵀnew = zero(Dᵀ[end])
    Eᵀnew = zero(Eᵀ[end])
    for (Nˢ, vars) in zip(Ns, uVars)
        res = SCFmethodSelector(SCFmethod)(Val(:UHF), Nˢ, Hcore, HeeI, S, X, vars; kws...)
        push!(fs, res[1])
        Dᵀnew += res[2]
    end
    for (f, vars) in zip(fs, uVars)
        Cnew, Fnew, Dnew, Enew = f(Dᵀnew)
        push!(vars.Cs, Cnew)
        push!(vars.Fs, Fnew)
        push!(vars.Ds, Dnew)
        push!(vars.Es, Enew)
        Eᵀnew += Enew
    end
    push!(Dᵀ, Dᵀnew)
    push!(Eᵀ, Eᵀnew)
end


constraintSolver(vec, B; convexConstraint=true, solver::Symbol=:default) = 
constraintSolverCore(Val(solver), vec, B; convexConstraint)

function constraintSolverCore(::Val{:default}, vec, B; convexConstraint=true) # Fastest
    len = length(vec)
    A = ones(len) |> transpose |> Array
    b = [1]
    g = convexConstraint ? fill(indicator(0, 1), len) : fill(indicator(-Inf, Inf), len)
    params = SeparableOptimization.AdmmParams(B, vec, A, b, g)
    settings = SeparableOptimization.Settings(; ρ=ones(1), σ=ones(len), compute_stats=true)
    vars, _ = SeparableOptimization.optimize(params, settings)
    vars.x
end

function constraintSolverCore(::Val{:Convex}, vec, B; convexConstraint=true, method=COSMO.Optimizer) # With Convex.jl, more flexible
    len = length(vec)
    c = convexConstraint ? Convex.Variable(len, Positive()) : Convex.Variable(len)
    f = 0.5* Convex.quadform(c, B) + dot(c,vec)
    o = Convex.minimize(f, sum(c)==1)
    Convex.solve!(o, method, silent_solver=true)
    evaluate(c)
end

function constraintSolverCore(ConvexSupportedSolver::Val, vec, B; convexConstraint=true)
    mod = getfield(Quiqbox, string(ConvexSupportedSolver)[6:end-3] |> Symbol)
    method = getfield(mod, :Optimizer)
    constraintSolverCore(Val(:convex), vec, B; convexConstraint, method)
end