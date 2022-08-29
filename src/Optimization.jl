export gradDescent!, POconfig, optimizeParams!

using LinearAlgebra: norm

const OFtypes = (:HF,)

const defaultPOconfigStr = "POconfig()"

const defaultHFthresholdForHFgrad = getAtolVal(Float64)

const defaultHFforHFgrad = HFconfig(SCF=SCFconfig(threshold=defaultHFthresholdForHFgrad))


"""

    gradDescent!(vars::AbstractVector{T}, grad::AbstractVector{T}, η::T=T(0.001), 
                 threshold::T=2sqrt(length(grad))/(25norm(η))) where {T} -> 
    vars::AbstractVector{T}

Default gradient descent (GD) method used in [`optimizeParams!`](@ref). `vars` are the 
input variables of a function with corresponding gradient `grad`. `η` is the learning rate 
(step size) of the gradient descent. `threshold` is the clipping threshold of `grad` which 
will be renormalized if it's larger then `threshold` to prevent gradient exploding. 
`gradDescent!` modifies the `vars` and returns the updated value. It can be replaced by a 
more advanced GD function through customizing [`POconfig`](@ref).
"""
function gradDescent!(vars::AbstractVector{T}, grad::AbstractVector{T}, η::T=T(0.001), 
                      threshold::T=2sqrt(length(grad))/(25norm(η))) where {T}
    gNorm = norm(grad)
    gradNew = ifelse(gNorm > threshold, (threshold / gNorm * grad), grad)
    vars .-= η*gradNew
end


const defaultPOconfigPars = Any[Val(:HF), defaultHFforHFgrad, NaN, 5e-7, 500, gradDescent!]

"""

    POconfig{T, M, CBT<:ConfigBox, F<:Function} <: ConfigBox{T, POconfig, M}

The mutable container of parameter optimization configurations.

≡≡≡ Field(s) ≡≡≡

`method::Val{M}`: The method to calculate objective function (e.g., HF energy) for 
optimization. Available values of `M` from Quiqbox are $(string(OFtypes)[2:end-1]).

`config::CBT`: The configuration for the selected `method`. E.g., for `:HF` it's 
[`HFconfig`](@ref).

`target::T`: The target value of the objective function. The difference between the 
last-step value and the target value will be used for convergence detection. If it's set to 
`NaN`, the gradient of the latest step and the difference between the function values of 
latest two steps are used instead.

`threshold::T`: The error threshold for the convergence determination; when set to `NaN`, 
there will be no convergence detection.

`maxStep::Int`: Maximum iteration steps allowed regardless if the iteration converges.

`GD::F`: Applied gradient descent `Function`. Default method is [`gradDescent!`](@ref).

≡≡≡ Initialization Method(s) ≡≡≡

    POconfig(;method::Val=$(defaultPOconfigPars[1]), 
              config::ConfigBox=$(defaultHFCStr), 
              target::T=$(defaultPOconfigPars[3]), 
              threshold::T=$(defaultPOconfigPars[4]), 
              maxStep::Int=$(defaultPOconfigPars[5]), 
              GD::Function=$(defaultPOconfigPars[6])) where {T} -> 
    POconfig

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> POconfig();

julia> POconfig(maxStep=100);
```
"""
mutable struct POconfig{T, M, CBT<:ConfigBox, F<:Function} <: ConfigBox{T, POconfig, M}
    method::Val{M}
    config::CBT
    target::T
    threshold::T
    maxStep::Int
    GD::F
end

POconfig(t::NamedTuple) = genNamedTupleC(:POconfig, defaultPOconfigPars)(t)

POconfig(;kws...) = 
length(kws) == 0 ? POconfig(defaultPOconfigPars...) : POconfig(kws|>NamedTuple)

const defaultPOconfig = Meta.parse(defaultPOconfigStr) |> eval


function getGradE(config::POconfig{<:Any, :HF, <:ConfigBox{T, HFconfig}}, 
                  pbs::AbstractVector{<:ParamBox{T}}, gtb::GTBasis{T, D}, 
                  nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{D, T}}, 
                  N::Union{Int, Tuple{Int}, NTuple{2, Int}}) where {T, D, NN}
    res = runHF(gtb, nuc, nucCoords, N, config.config, printInfo=false)
    g = gradOfHFenergy(pbs, res)
    g, res.Ehf
end


"""

    optimizeParams!(pbs, bs, nuc, nucCoords, 
                    config=$(defaultPOconfigStr), N=getCharge(nuc); printInfo=true) -> 
    Tuple{Vector{T}, Matrix{T}, Matrix{T}, Union{Bool, Missing}} where {T}

    optimizeParams!(pbs, bs, nuc, nucCoords, 
                    N=getCharge(nuc), config=$(defaultPOconfigStr); printInfo=true) -> 
    Tuple{Vector{T}, Matrix{T}, Matrix{T}, Union{Bool, Missing}} where {T}

The main function to optimize the parameters of a given basis set. It returns a `Tuple` of 
relevant information. The first three elements are the energies, the parameter values, and 
the gradients from all the iteration steps respectively (For latter two, each column 
corresponds to each step). The last element is the indicator of whether the optimization is 
converged if the convergence detection is on (i.e., `isnan(config.threshold)!=true`), or 
else it's set to `missing`.

=== Positional argument(s) ===

`pbs::AbstractVector{<:ParamBox{T}}`: The parameters to be optimized that are extracted 
from the basis set. If the parameter is marked as "differentiable", the value of its input 
variable will be optimized.

`bs::Union{
    Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
    AbstractVector{<:AbstractGTBasisFuncs{T, D}}
}`: The basis set to be optimized.

`nuc::Union{
    NTuple{NN, String} where NN, 
    AbstractVector{String}
}`: The nuclei in the studied system.

`nucCoords::$(SpatialCoordType)`: The coordinates of corresponding nuclei.

`config::POconfig`: The Configuration of selected parameter optimization method. For more 
information please refer to [`POconfig`](@ref).

`N::Union{Int, Tuple{Int}, NTuple{2, Int}}`: Total number of electrons, or the number(s) of 
electrons with same spin configurations(s).

=== Keyword argument(s) ===

`printInfo::Bool`: Whether print out the information of iteration steps.
"""
function optimizeParams!(pbs::AbstractVector{<:ParamBox{T}}, 
                         bs::Union{Tuple{Vararg{AbstractGTBasisFuncs{T, D}}}, 
                                   AbstractVector{<:AbstractGTBasisFuncs{T, D}}}, 
                         nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                         nucCoords::SpatialCoordType{T, D, NN}, 
                         config::POconfig{<:Any, M, CBT, F}=defaultPOconfig, 
                         N::Union{Int, Tuple{Int}, NTuple{2, Int}}=getCharge(nuc); 
                         printInfo::Bool=true) where {T, D, NN, M, CBT, F}
    tAll = @elapsed begin

        nuc = arrayToTuple(nuc)
        nucCoords = genTupleCoords(T, nucCoords)
        i = 0
        Es = T[]
        pvs = zeros(length(pbs), 0)
        grads = zeros(length(pbs), 0)
        threshold = config.threshold
        target = config.target
        maxStep = config.maxStep
        gap = min(100, max(maxStep ÷ 200 * 10, 1))
        detectConverge = ifelse(isnan(threshold), false, true)
        isConverged = genDetectConvFunc(target, threshold)
        blConv = ifelse(detectConverge, true, missing)

        pvsL = getindex.(pbs)

        while true
            gtb = GTBasis(bs)

            t = @elapsed begin
                grad, E = getGradE(config, pbs, gtb, nuc, nucCoords, N)
            end
            push!(Es, E)
            pvs = hcat(pvs, pvsL)
            grads = hcat(grads, grad)

            if i%gap == 0 && printInfo
                println(rpad("Step $i: ", 15), rpad("E = $(E)", 26))
                print(rpad("", 10), "params = ")
                println(IOContext(stdout, :limit => true), pvsL)
                print(rpad("", 12), "grad = ")
                println(IOContext(stdout, :limit => true), grad)
                println("Step duration: ", t, " seconds.\n")
            end

            !( detectConverge && (blConv = isConverged(Es, grads)) ) && i < maxStep || break

            setindex!.(pbs, config.GD(pvsL, grad))

            i += 1
        end

    end

    if printInfo
        println("The iteration just ended at")
        println(rpad("Step $(i): ", 15), rpad("E = $(Es[end])", 26))
        print(rpad("", 10), "params = ")
        println(IOContext(stdout, :limit => true), pvs[:, end])
        print(rpad("", 12), "grad = ")
        println(IOContext(stdout, :limit => true), grads[:, end])
        println("Optimization duration: ", tAll/60, " minutes.")
        if detectConverge
            println("The result has" * ifelse(blConv, "", " not") *" converged.")
        end
    end

    Es, pvs, grads, blConv
end

optimizeParams!(pbs, bs, nuc, nucCoords, N::Int, config=defaultPOconfig; printInfo=true) = 
optimizeParams!(pbs, bs, nuc, nucCoords, config, N; printInfo)

function genDetectConvFunc(target, threshold)
    if isnan(target)
        function (Es, gs)
            bl = isOscillateConverged(Es, threshold, minimalCycles=4)[1]
            ifelse(!bl || any(g > threshold for g in gs[end]), false, true)
        end
    else
        (Es, _) = Es -> (abs(Es[end] - target) < threshold)
    end
end