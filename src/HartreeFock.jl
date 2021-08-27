export runHF, runHFcore, SCFconfig

using LinearAlgebra: dot, Hermitian
using PiecewiseQuadratics: indicator
using SeparableOptimization, Convex, COSMO

getXcore1(S) = S^(-0.5) |> Array

const getXmethods = Dict(1=>getXcore1)

getX(S; method::Int=1) = getXmethods[method](S)


function getC(X, F; outputCx::Bool=false, outputEmo::Bool=false)
    ϵ, Cₓ = eigen(X'*F*X |> Hermitian, sortby=x->x)
    outC = outputCx ? Cₓ : X*Cₓ
    outputEmo ? (outC, ϵ) : outC
end


getD(C, Nˢ) = @views (C[:,1:Nˢ]*C[:,1:Nˢ]') |> Hermitian |> Array
# Nˢ: number of electrons with the same spin

getD(X, F, Nˢ) = getD(getC(X, F), Nˢ)


function getGcore(HeeI, DJ, DK)
    G = zero(DJ)
    l = size(G)[1]
    for μ = 1:l, ν = 1:l # fastest
        G[μ, ν] = dot(transpose(DJ), @view HeeI[μ,ν,:,:]) - dot(DK, @view HeeI[μ,:,:,ν]) 
    end
    G |> Hermitian |> Array
end

# RHF
@inline getG(HeeI, D) = getGcore(HeeI, 2D, D)

# UHF
@inline getG(HeeI, D, Dᵀ) = getGcore(HeeI, Dᵀ, D)


@inline getF(Hcore, G) = Hcore + G

# RHF
@inline getF(Hcore, HeeI, D::Array{<:Number, 2}, Dᵀ::Array{<:Number, 2}) = getF(Hcore, getG(HeeI, D, Dᵀ))

# UHF
@inline getF(Hcore, HeeI, D::Array{<:Number, 2}) = getF(Hcore, getG(HeeI, D))

@inline getF(Hcore, HeeI, Ds::Tuple{Vararg{Array{<:Number, 2}}}) = getF(Hcore, getG(HeeI, Ds...))


@inline getE(Hcore, F, D) = dot(transpose(D), 0.5*(Hcore + F))


@inline getEᵀcore(Hcore, F, D) = 2*getE(Hcore, F, D)

@inline getEᵀcore(Hcore, Fᵅ, Dᵅ, Fᵝ, Dᵝ) = getE(Hcore, Fᵅ, Dᵅ) + 
                                                        getE(Hcore, Fᵝ, Dᵝ)

function getEᵀ(Hcore, HeeI, C::Array{<:Number, 2}, N::Int) # RHF
    D = getD(C, N÷2)
    F = getF(Hcore, HeeI, D)
    getEᵀcore(Hcore, F, D)
end

# UHF
function getEᵀ(Hcore, HeeI, (Cᵅ,Cᵝ)::NTuple{2, Array{<:Number, 2}}, (Nᵅ,Nᵝ)::NTuple{2, Int})
    Dᵅ = getD(Cᵅ, Nᵅ)
    Dᵝ = getD(Cᵝ, Nᵝ)
    Dᵀ = Dᵅ + Dᵝ
    Fᵅ = getF(Hcore, HeeI, Dᵅ, Dᵀ)
    Fᵝ = getF(Hcore, HeeI, Dᵝ, Dᵀ)
    getEᵀcore(Hcore, Fᵅ, Dᵅ, Fᵝ, Dᵝ)
end


function getCFDE(Hcore, HeeI, X, Ds...)
    Fnew = getF(Hcore, HeeI, Ds)
    Enew = getE(Hcore, Fnew, Ds[1])
    Cnew = getC(X, Fnew)
    [Cnew, Fnew, Ds[1], Enew] # Fnew is based on latest variables.
end


# RHF
function initializeSCF(Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
                       C::Array{<:Number, 2}, N::Int)
    Nˢ = N÷2
    D = getD(C, Nˢ)
    F = getF(Hcore, HeeI, D)
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
    Fs = getF.(Ref(Hcore), Ref(HeeI), Ds, Ref(Dᵀs[]))
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
        new{length(methods)}(Tuple(methods), Tuple(intervals), Tuple(kwPairs), 
                             oscillateThreshold)
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


function guessCcore1(S, Hcore; K=1.75, X=getX(S), _kws...)
    l = size(Hcore)[1]
    H = zero(Hcore)
    for i in 1:l, j in 1:l
        H[i,j] = K * S[i,j] * (Hcore[i,i] + Hcore[j,j]) / 2
    end
    getC(X, H)
end

guessCcore2(S, Hcore; X=getX(S), _kws...) = getC(X, Hcore)

const guessCmethods = Dict(:GWH=>guessCcore1, :Hcore=>guessCcore2)

guessC(S, Hcore; method::Symbol=:GWH, kws...) = guessCmethods[method](S, Hcore; kws...)


const defaultSCFconfig = SCFconfig([:ADIIS, :DIIS, :SD], [1e-4, 1e-8, 1e-12])


function runHF(bs::Array{<:AbstractFloatingGTBasisFunc, 1}, 
               mol, nucCoords, N=getCharge(mol); initialC=:GWH, getXmethod=getX, 
               scfConfig=defaultSCFconfig, 
               HFtype=:RHF, printInfo::Bool=true, maxSteps::Int=1000)
    @assert length(mol) == length(nucCoords)
    @assert (basisSize(bs) |> sum) >= ceil(N/2)
    gtb = GTBasis(bs)
    runHF(gtb, mol, nucCoords, N; initialC, scfConfig, 
          getXmethod, HFtype, printInfo, maxSteps)
end

function runHF(gtb::BasisSetData, 
               mol::Array{String, 1}, 
               nucCoords::Array{<:Array{<:Real, 1}, 1}, 
               N::Union{NTuple{2, Int}, Int}=getCharge(mol); 
               initialC::Union{Array{<:Number, 2}, 
                               NTuple{2, Array{<:Number, 2}}, 
                               Symbol}=:GWH, 
               getXmethod::Function=getX, 
               scfConfig::SCFconfig=defaultSCFconfig, 
               HFtype::Symbol=:RHF, printInfo::Bool=true, maxSteps::Int=1000)
    @assert length(mol) == length(nucCoords)
    @assert typeof(gtb).parameters[1] >= ceil(N/2)
    Hcore = gtb.getHcore(mol, nucCoords)
    X = getXmethod(gtb.S)
    initialC isa Symbol && (initialC = guessC(gtb.S, Hcore; X, method=initialC))
    runHFcore(N, Hcore, gtb.eeI, gtb.S, X, initialC; scfConfig, printInfo, maxSteps, HFtype)
end


function runHFcore(N::Union{NTuple{2, Int}, Int},  
                   Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
                   S::Array{<:Number, 2}, 
                   X::Array{<:Number, 2}=getX(S), 
                   C::Union{Array{<:Number, 2}, 
                            NTuple{2, Array{<:Number, 2}}}=guessC(S, Hcore; X);
                   HFtype::Symbol=:RHF,  
                   scfConfig::SCFconfig{L}, printInfo::Bool=true, 
                   maxSteps::Int=1000) where {L}
    @assert maxSteps > 0
    HFtype == :UHF && (N isa Int) && (N = (N÷2, N-N÷2))
    vars = initializeSCF(Hcore, HeeI, C, N)
    Etots = (vars isa Tuple) ? vars[1].shared.Etots : vars.shared.Etots
    printInfo && println(rpad("$(HFtype) Initial Gauss", 22), "E = $(Etots[end])")
    isConverged = true
    for (method, kws, breakPoint, i) in 
        zip(scfConfig.methods, scfConfig.methodConfigs, scfConfig.intervals, 1:L)
        
        while true
            iStep = length(Etots)
            iStep <= maxSteps || (isConverged = false) || break
            SCF!(method, N, Hcore, HeeI, S, X, vars; kws...)
            printInfo && (iStep % floor(log(4, iStep) + 1) == 0 || iStep == maxSteps) && 
            println(rpad("Step $(iStep)", 10), 
                    rpad("#$(i) ($(method))", 12), 
                    "E = $(Etots[end])")
            abs(Etots[end]-Etots[end-1]) > breakPoint || (isConverged = true) && break
            flag, Std = isOscillateConverged(Etots, 
                                             10^(log(10, breakPoint)÷2), 
                                             returnStd=true)
            flag && (isConverged = Std > scfConfig.oscillateThreshold ? false : true; break)
        end

    end
    negStr = isConverged ? "is " : "has not "
    printInfo && println("The SCF procedure ", negStr, "converged.\n")
    HFfinalVars(X, vars, isConverged)
end


function SDcore(Nˢ::Int, Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
                X::Array{<:Number, 2}, 
                F::Array{<:Number, 2}, 
                D::Array{<:Number, 2}; 
                dampingStrength::Float64=0.0, _kws...)
    @assert 0 <= dampingStrength <= 1 "The range of `dampingStrength`::Float64 is [0,1]."
    Dnew = getD(X, F, Nˢ)
    (1-dampingStrength)*Dnew + dampingStrength*D
end


function xDIIScore(method::Symbol, Nˢ::Int, Hcore::Array{<:Number, 2}, 
                   HeeI::Array{<:Number, 4}, 
                   S::Array{<:Number, 2}, X::Array{<:Number, 2}, 
                   Fs::Array{<:Array{<:Number, 2}, 1}, Es::Array{Float64, 1}, 
                   Ds::Array{<:Array{<:Number, 2}, 1}; 
                   DIISsize::Int=15, solver=:default, _kws...)
    DIISmethod, convexConstraint, permuteData = DIISmethods[method]
    is = permuteData ? sortperm(Es) : (:)
    ∇s = (@view Fs[is])[1:end .> end-DIISsize]
    Es = (@view Es[is])[1:end .> end-DIISsize]
    Ds = (@view Ds[is])[1:end .> end-DIISsize]
    vec, B = DIISmethod(Ds, ∇s, Es, S)
    c = constraintSolver(vec, B, solver; convexConstraint)
    grad = c.*∇s |> sum
    getD(X, grad |> Hermitian |> Array, Nˢ) # grad == F.
end

const DIISmethods = Dict( :DIIS => ((Ds, ∇s,  _, S)->DIIScore(Ds, ∇s, S),   false, true ),
                         :EDIIS => ((Ds, ∇s, Es, _)->EDIIScore(Ds, ∇s, Es), true,  false),
                         :ADIIS => ((Ds, ∇s,  _, _)->ADIIScore(Ds, ∇s),     true,  false))


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


const SD = (Nˢ, Hcore, HeeI, _dm, X, tmpVars::HFtempVars; kws...) -> 
           SDcore(Nˢ, Hcore, HeeI, X, tmpVars.Fs[end], tmpVars.Ds[end]; kws...)

const xDIIS = (method::Symbol) -> (Nˢ, Hcore, HeeI, S, X, tmpVars::HFtempVars; kws...) -> 
                                  xDIIScore(method, Nˢ, Hcore, HeeI, S, X, tmpVars.Fs, 
                                            tmpVars.Es, tmpVars.Ds; kws...)

const SCFmethods = Dict( [:SD, :DIIS,        :ADIIS,        :EDIIS] .=> 
                         [ SD, xDIIS(:DIIS), xDIIS(:ADIIS), xDIIS(:EDIIS)])


# RHF
function RHFcore(SCFmethod::Symbol, N::Int, 
                 Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
                 S::Array{<:Number, 2}, X::Array{<:Number, 2}, 
                 rVars::HFtempVars{:RHF}; kws...)
    D = SCFmethods[SCFmethod](N÷2, Hcore, HeeI, S, X, rVars; kws...)
    partRes = getCFDE(Hcore, HeeI, X, D)
    partRes..., 2D, 2partRes[end]
end

function SCF!(SCFmethod::Symbol, N::Int, 
              Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
              S::Array{<:Number, 2}, X::Array{<:Number, 2}, 
              rVars::HFtempVars{:RHF}; kws...)
    res = RHFcore(SCFmethod, N, Hcore, HeeI, S, X, rVars; kws...)
    push!(rVars.Cs, res[1])
    push!(rVars.Fs, res[2])
    push!(rVars.Ds, res[3])
    push!(rVars.Es, res[4])
    push!(rVars.shared.Dtots, res[5])
    push!(rVars.shared.Etots, res[6])
end


# UHF
function UHFcore(SCFmethod::Symbol, Ns::NTuple{2, Int}, 
                 Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
                 S::Array{<:Number, 2}, X::Array{<:Number, 2}, 
                 uVars::NTuple{2, HFtempVars{:UHF}}; kws...) 
    Ds = SCFmethods[SCFmethod].(Ns, Ref(Hcore), Ref(HeeI), 
                                Ref(S), Ref(X), uVars; kws...)
    Dᵀnew = Ds |> sum
    partRes = getCFDE(Ref(Hcore), Ref(HeeI), Ref(X), Ds, Ref(Dᵀnew))
    partRes..., Dᵀnew, sum(partRes[end])
end

function SCF!(SCFmethod::Symbol, Ns::NTuple{2, Int}, 
              Hcore::Array{<:Number, 2}, HeeI::Array{<:Number, 4}, 
              S::Array{<:Number, 2}, X::Array{<:Number, 2}, 
              uVars::NTuple{2, HFtempVars{:UHF}}; kws...) 
    res = UHFcore(SCFmethod, Ns, Hcore, HeeI, S, X, uVars; kws...)
    fields = [:Cs, :Fs, :Ds, :Es]
    for (field, vars) in zip(fields, @view res[1:4])
        push!.(getfield.(uVars, field), vars)
    end
    push!(uVars[1].shared.Dtots, res[5])
    push!(uVars[1].shared.Etots, res[6])
end


# Fastest
function ADMMSolver(vec, B; convexConstraint=true)
    len = length(vec)
    A = ones(len) |> transpose |> Array
    b = [1]
    g = convexConstraint ? fill(indicator(0, 1), len) : fill(indicator(-Inf, Inf), len)
    params = SeparableOptimization.AdmmParams(B, vec, A, b, g)
    settings = SeparableOptimization.Settings(; ρ=ones(1), σ=ones(len), compute_stats=true)
    vars, _ = SeparableOptimization.optimize(params, settings)
    vars.x
end


# With Convex.jl, more flexible
function ConvexSolver(vec, B; convexConstraint=true, method=COSMO.Optimizer)
    len = length(vec)
    c = convexConstraint ? Convex.Variable(len, Positive()) : Convex.Variable(len)
    f = 0.5* Convex.quadform(c, B) + dot(c,vec)
    o = Convex.minimize(f, sum(c)==1)
    Convex.solve!(o, method, silent_solver=true)
    evaluate(c)
end


const ConstraintSolvers = Dict(:default=>ADMMSolver, :Convex=>ConvexSolver)

constraintSolver(vec, B, solver::Symbol=:default; convexConstraint=true) = 
ConstraintSolvers[solver](vec, B; convexConstraint)

constraintSolver(vec, B, ConvexMethod; convexConstraint=true) = 
ConvexSolver(vec, B; convexConstraint, method=ConvexMethod)