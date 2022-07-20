export gradDescent!, updateParams!, POconfig, optimizeParams!

using LinearAlgebra: norm

"""

    gradDescent!(pars::AbstractVector{T}, grads::AbstractVector{T}, η::T=T(0.001), 
                 clipThreshold::T=2sqrt(length(grad))/(25norm(η))) where {T} -> 
    pars::AbstractVector{T}

Default gradient descent method in used in Quiqbox.
"""
function gradDescent!(pars::AbstractVector{T}, grad::AbstractVector{T}, η::T=T(0.001), 
                      clipThreshold::T=2sqrt(length(grad))/(25norm(η))) where {T}
    gNorm = norm(grad)
    gradNew = ifelse(gNorm > clipThreshold, (clipThreshold / gNorm * grad), grad)
    pars .-= η.*gradNew
end


"""

    updateParams!(pbs::AbstractVector{<:ParamBox{T}}, grads::AbstractVector{T}, 
                  method::F=gradDescent!) where {T, F<:Function} -> 
                  AbstractVector{<:ParamBox{T}}

Given a `AbstractVector` of parameters::`ParamBox{T}` and its gradients with respect to 
each parameter, update the `ParamBox`s and return the updated values.
"""
function updateParams!(pbs::AbstractVector{<:ParamBox{T}}, grads::AbstractVector{T}, 
                       method!::F=gradDescent!) where {T, F<:Function}
    parVals = [i[] for i in pbs]
    method!(parVals, grads)
    for (m,n) in zip(pbs, parVals)
        m[] = n
    end
    parVals
end


const Doc_POconfig_Eg1 = "POconfig{Float64, :HF, HFconfig{:RHF, Val{:SAD}, 3}, "*
                         "typeof(gradDescent!)}(Val{:HF}(), HFconfig{:RHF, Val{:SAD}, 3}"*
                         "(Val{:RHF}(), Val{:SAD}(), SCFconfig{Float64, 3}(interval=("*
                         "0.0001, 1.0e-6, 1.0e-15), oscillateThreshold=1.0e-5, method, "*
                         "methodConfig)[:ADIIS, :DIIS, :ADIIS], 1000, true), NaN, "*
                         "1.0e-5, 500, Quiqbox.gradDescent!)"

const Doc_POconfig_Eg2 = Doc_POconfig_Eg1[1:end-26] * "1" * Doc_POconfig_Eg1[end-24:end]


const OFtypes = (:HF,)

"""

    POconfig{T, M, CBT, F} <: ConfigBox{T, POconfig, M}

≡≡≡ Field(s) ≡≡≡

The mutable container of parameter optimization configuration.

`method::Val{M}`: The method to calculate objective function (e.g., HF energy) for 
optimization. Available values of `M` from Quiqbox are $(string(OFtypes)[2:end-1]).

`config::ConfigBox`: The configuration `struct` for the selected `method`. E.g., 
for `:HF` it's `$(HFconfig)`.

`target::T`: The value of target function aimed to achieve.

`error::T`: The error for the convergence when evaluating difference between 
the latest few energies. When set to `NaN`, there will be no convergence detection.

`maxStep::Int`: Maximum allowed iteration steps regardless of whether the optimization 
iteration converges.

`GD::F1`: Applied gradient descent `Function`. Default method is `$(gradDescent!)`.

≡≡≡ Initialization Method(s) ≡≡≡

    POconfig(;kws...) -> POconfig

    POconfig(t::NamedTuple) -> POconfig

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> POconfig()
$(Doc_POconfig_Eg1)

julia> POconfig(maxStep=100)
$(Doc_POconfig_Eg2)
```
"""
mutable struct POconfig{T, M, CBT<:ConfigBox, F<:Function} <: ConfigBox{T, POconfig, M}
    method::Val{M}
    config::CBT
    target::T
    error::T
    maxStep::Int
    GD::F
end

POconfig(a1::Symbol, args...) = POconfig(Val(a1), args...)

const defaultPOconfigPars = Any[Val(:HF), defaultHFC, NaN, 1e-5, 500, gradDescent!]

POconfig(t::NamedTuple) = genNamedTupleC(:POconfig, defaultPOconfigPars)(t)

POconfig(;kws...) = 
length(kws) == 0 ? POconfig(defaultPOconfigPars...) : POconfig(kws|>NamedTuple)

const defaultPOconfig = POconfig()

const defaultPOconfigStr = "POconfig()"


"""

    genOFmethod(POmethod::Val{:HF}, config::HFconfig=$(defaultHFCStr)) -> 
    NTuple{2, Function}

Generate the functions to calculate the value and gradient respectively of the desired 
objective function. Default method is HF energy. To implement your own method for parameter 
optimization, you can import `genOFmethod` and add new methods with different `POmethod` 
which should have the same value with the field `method` in the corresponding `POconfig`.
"""
function genOFmethod(::Val{:HF}, config::HFconfig{<:Any, HFT}=defaultHFC) where {HFT}
    fVal = @inline function (gtb, nuc, nucCoords, N)
        res = runHF(gtb, nuc, nucCoords, N, config, printInfo=false)
        res.Ehf, res.C
    end
    fVal, gradHFenergy
end


"""

    optimizeParams!(pbs::AbstractVector{<:ParamBox{T1}}, 
                    bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T1, D}}}, 
                              AbstractVector{<:AbstractGTBasisFuncs{T1, D}}}, 
                    nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                    nucCoords::Union{NTuple{NN, NTuple{3, T1}}, 
                                     AbstractVector{<:AbstractVector{T1}}}, 
                    config::POconfig{T2, M, CBT, F}=$(defaultPOconfigStr), 
                    N::Int=getCharge(nuc); 
                    printInfo::Bool=true) where {T1, T2, D, NN, M, CBT, F} -> 
    Es::Vector{T1}, pars::Matrix{T1}, grads::Matrix{T1}

The main function to optimize the parameters of a given basis set.

=== Positional argument(s) ===

`pbs::AbstractVector{<:ParamBox{T}}`: The parameters to be optimized that are extracted 
from the basis set.

`bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
AbstractVector{<:AbstractGTBasisFuncs{T, D}}}`: Basis set.

`nuc::Union{NTuple{NN, String}, AbstractVector{String}}`: The element symbols of the nuclei 
for the studied system.

`nucCoords::Union{NTuple{NN, NTuple{3, T}}, AbstractVector{<:AbstractVector{T}}}`: Nuclei 
coordinates.

`config::POconfig{T, M, CBT, F}`: The Configuration of selected parameter optimization 
method. For more information please refer to `POconfig`.

`N::Int`: Total number of electrons.

=== Keyword argument(s) ===

`printInfo::Bool`: Whether print out the information of iteration steps.
"""
function optimizeParams!(pbs::AbstractVector{<:ParamBox{T1}}, 
                         bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T1, D}}}, 
                                   AbstractVector{<:AbstractGTBasisFuncs{T1, D}}}, 
                         nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                         nucCoords::Union{NTuple{NN, NTuple{3, T1}}, 
                                          AbstractVector{<:AbstractVector{T1}}}, 
                         config::POconfig{T2, M, CBT, F}=defaultPOconfig, 
                         N::Int=getCharge(nuc); printInfo::Bool=true) where 
                        {T1, T2, D, NN, M, CBT, F}
    tAll = @elapsed begin

        i = 0
        Es = T1[]
        pars = zeros(length(pbs), 0)
        grads = zeros(length(pbs), 0)
        error = config.error
        target = config.target
        maxStep = config.maxStep
        gap = min(100, max(maxStep ÷ 200 * 10, 1))
        detectConverge = ifelse(isnan(error), false, true)

        if isnan(target)
            isConverged = (Es) -> isOscillateConverged(Es, error, leastCycles=3)[1]
        else
            isConverged = Es -> (abs(Es[end] - target) < error)
        end

        parsL = [i[] for i in pbs]

        ECmethod, EGmethod = genOFmethod(Val(:HF), config.config)

        while true
            gtb = GTBasis(bs)
            E, C = ECmethod(gtb, nuc, nucCoords, N)

            t = @elapsed begin
                grad = EGmethod(pbs, gtb.basis, gtb.S, C, nuc, nucCoords)
            end

            push!(Es, E)
            pars = hcat(pars, parsL)
            grads = hcat(grads, grad)

            if i%gap == 0 && printInfo
                println(rpad("Step $i: ", 15), rpad("E = $(E)", 26))
                print(rpad("", 10), "params = ")
                println(IOContext(stdout, :limit => true), parsL)
                print(rpad("", 12), "grad = ")
                println(IOContext(stdout, :limit => true), grad)
                println("Step duration: ", t, " seconds.\n")
            end

            !(detectConverge && isConverged(Es)) && i < maxStep || break

            parsL = updateParams!(pbs, grad, config.GD)

            i += 1
        end

    end

    if printInfo
        println("The iteration just ended at")
        println(rpad("Step $(i): ", 15), rpad("E = $(Es[end])", 26))
        print(rpad("", 10), "params = ")
        println(IOContext(stdout, :limit => true), pars[:, end])
        print(rpad("", 12), "grad = ")
        println(IOContext(stdout, :limit => true), grads[:, end])
        println("Optimization duration: ", tAll/60, " minutes.")
        if detectConverge
            println("The result has" * ifelse(isConverged(Es), "", " not") *" converged.")
        end
    end

    Es, pars, grads
end

"""

    optimizeParams!(pbs::AbstractVector{<:ParamBox{T1}}, 
                    bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T1, D}}}, 
                              AbstractVector{<:AbstractGTBasisFuncs{T1, D}}}, 
                    nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                    nucCoords::Union{NTuple{NN, NTuple{3, T1}}, 
                                     AbstractVector{<:AbstractVector{T1}}}, 
                    N::Int=getCharge(nuc), 
                    config::POconfig{T2, M, CBT, F}=$(defaultPOconfigStr); 
                    printInfo::Bool=true) where {T1, T2, D, NN, M, CBT, F} -> 
    Es::Vector{T1}, pars::Matrix{T1}, grads::Matrix{T1}
"""
optimizeParams!(pbs, bs, nuc, nucCoords, N::Int, config=defaultPOconfig; printInfo=true) = 
optimizeParams!(pbs, bs, nuc, nucCoords, config, N; printInfo)