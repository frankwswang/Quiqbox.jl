export GDconfig, POconfig, optimizeParams!

using LinearAlgebra: norm
using LineSearches

const OFtypes = (:HFenergy,)
const OFfunctions = Dict([:HFenergy] .=> 
    [
        ( ( runHFcore, x->x[begin][begin].shared.Etots[end] ), 
          ( (tVars, pbs, gtb, nuc, nucCoords, N)->
            gradOfHFenergy(pbs, gtb, last.(getproperty.(tVars[begin], :Cs)), 
                           nuc, nucCoords, N), 
            itself )
        )
    ]
)

const defaultPOconfigStr = "POconfig()"
const defaultHFthresholdForHFgrad = getAtolVal(Float64)
const defaultHFforHFgrad = HFconfig(SCF=SCFconfig(threshold=defaultHFthresholdForHFgrad))


"""

    GDconfig{T, M, ST<:Union{Array{T, 0}, T}} <: ConfigBox{T, GDconfig, M}

The mutable container of configurations for the default gradient descent method used to 
optimize parameters.

≡≡≡ Field(s) ≡≡≡

`lineSearchMethod::M`: The selected line search method to optimize the step size for each 
gradient descent iteration. It can be any algorithm defined in the Julia package 
[LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl), or [`itself`](@ref) 
so that the step size will be fixed to a constant value during all the iterations.

`initialStep::ST`: The value of the initial step size in each iteration. If it's an 
`Array{T, 0}`, then the final step size (optimized by `lineSearchMethod`) in current 
interaction will be set as the initial step size for the next iteration.

`stepBound::NTuple{2, T}`: The lower bound and upper bound of the step size to enforce 
`stepBound[begin] ≤ η ≤ stepBound[end]` (where `η` is the step size in each iteration). 
Whenever the size size is out of the boundary, it will be clipped to a bound value.

`scaleStepBound::Bool`: Determine whether `stepBound` will be scaled so that the norm of 
the gradient descent in each iteration is constrained within the initially configured 
`stepBound`, i.e., `stepBound[begin] ≤ norm(η*∇f) ≤ stepBound[end]` (where `∇f` is the 
gradient of some function `f` in each iteration).

≡≡≡ Initialization Method(s) ≡≡≡

    GDconfig(lineSearchMethod::M=BackTracking(), 
             initialStep::ST=ifelse(M==$(iT), 0.1, 1.0); 
             stepBound::NTuple{2, T}=convert.(eltype(initialStep), (0, Inf)), 
             scaleStepBound::Bool=ifelse(M==typeof($(itself)), true, false)) where 
            {M, T<:AbstractFloat, ST<:Union{Array{T, 0}, T}} -> 
    GDconfig{T, M, ST}
"""
mutable struct GDconfig{T, M, ST<:Union{Array{T, 0}, T}} <: ConfigBox{T, GDconfig, M}
    lineSearchMethod::M
    initialStep::ST
    stepBound::NTuple{2, T}
    scaleStepBound::Bool

    function GDconfig(lineSearchMethod::M=BackTracking(), 
                      initialStep::ST=ifelse(M==iT, 0.1, 1.0); 
                      stepBound::NTuple{2, T}=convert.(eltype(initialStep), (0, Inf)), 
                      scaleStepBound::Bool=ifelse(M==iT, true, false)) where 
                     {M, T<:AbstractFloat, ST<:Union{Array{T, 0}, T}}
        0 < initialStep[] || throw(DomainError(initialStep[], "The initial step size "*
                                   "should be positive.")) 
        stepBound[begin] > stepBound[end] && 
        throw(DomainError(stepBound, "The bound set for initialStep should be a valid "*
              "closed interval."))
        (M == iT || hasproperty(LineSearches, (nameof∘typeof)(lineSearchMethod))) || 
        throw(DomainError(lineSearchMethod, 
              "The input `lineSearchMethod` is not supported."))
        new{T, M, ST}(lineSearchMethod, initialStep, stepBound, scaleStepBound)
    end
end


const defaultPOconfigPars = 
      [Val(:HFenergy), defaultHFforHFgrad, NaN, (5e-7, 5e-6), 200, GDconfig()]

"""

    POconfig{T, M, CBT<:ConfigBox, TH<:Union{T, NTuple{2, T}}, 
             OM} <: ConfigBox{T, POconfig, M}

The mutable container of parameter optimization configurations.

≡≡≡ Field(s) ≡≡≡

`method::Val{M}`: The method to calculate objective function (e.g., HF energy) for 
optimization. Available values of `M` from Quiqbox are $(string(OFtypes)[2:end-1]).

`config::CBT`: The configuration for the selected `method`. E.g., for `:HFenergy` it's 
[`HFconfig`](@ref).

`target::T`: The target value of the objective function. The difference between the 
last-step value and the target value will be used for convergence detection. If it's set to 
`NaN`, the gradient of the latest step and the difference between the function values of 
latest two steps are used instead.

`threshold::TH`: The error threshold/thresholds for the function value difference and 
the gradient both/respectively to determine whether the optimization iteration has 
converged. When it's (or either of them) set to `NaN`, there will be no corresponding 
convergence detection, and when `target` is not `NaN`, the threshold for the gradient won't 
be used because the gradient won't be part of the convergence criteria.

`maxStep::Int`: Maximum iteration steps allowed regardless if the iteration converges.

`optimizer::F`: Applied parameter optimizer. The default setting is [`GDconfig`](@ref)`()`. 
To use a function implemented by the user as the optimizer, it should have the following 
function signature: 

    o!(x::Vector{T}, gx::Vector{T}, fx::T, f::Function, gf::Function) where {T}

where `x` are the initial input variables of `Function` `f`, and the returned value of `f`, 
`f(x)`,  is equal to `fx`. `x` will be optimized, in another word, mutated by the 
optimizer `o!` each time `o!` runs. `gf` is a `Function` such that `gf(x) == (gx, fx)`.

≡≡≡ Initialization Method(s) ≡≡≡

    POconfig(;method::Val=$(defaultPOconfigPars[1]), 
              config::ConfigBox=$(defaultHFCStr), 
              target::T=$(defaultPOconfigPars[3]), 
              threshold::Union{T, NTuple{2, T}}=$(defaultPOconfigPars[4]), 
              maxStep::Int=$(defaultPOconfigPars[5]), 
              optimizer::Function=$(defaultPOconfigPars[6]|>typeof|>nameof)()) where {T} -> 
    POconfig{T}

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> POconfig();

julia> POconfig(maxStep=100);
```
"""
mutable struct POconfig{T, M, CBT<:ConfigBox, TH<:Union{Tuple{T}, NTuple{2, T}}, 
                        OM} <: ConfigBox{T, POconfig, M}
    method::Val{M}
    config::CBT
    target::T
    threshold::TH
    maxStep::Int
    optimizer::OM
end

POconfig(t::NamedTuple) = genNamedTupleC(:POconfig, defaultPOconfigPars)(t)

POconfig(;kws...) = 
length(kws) == 0 ? POconfig(defaultPOconfigPars...) : POconfig(kws|>NamedTuple)

const defaultPOconfig = Meta.parse(defaultPOconfigStr) |> eval


function genLineSearchOpt(GDc::GDconfig{T, M, ST}, 
                          f::Function, gf::Function) where {T, M, ST}
    l = GDc.lineSearchMethod
    η₀ = GDc.initialStep
    lo, up = GDc.stepBound
    @inline function (x, gx, fx)
        ϕForLS(η) = f(x .- η.*gx)
        𝑑ϕForLS(η) = dot(gf(x .- η.*gx)[begin], -gx)
        function ϕ𝑑ϕForLS(η)
            gxNew, ϕ = gf(x .- η.*gx)
            (ϕ, dot(gxNew, -gx))
        end
        η = if GDc.scaleStepBound
            clamp(η₀[], lo, up)
        else
            n = norm(gx)
            clamp(η₀[], lo/n, up/n)
        end
        ηNew = l(ϕForLS, 𝑑ϕForLS, ϕ𝑑ϕForLS, η, fx, -dot(gx, gx))[begin] # new step η
        ST <: Array{T, 0} && (η₀[] = ηNew)
        x .-= ηNew.*gx
    end
end

function genLineSearchOpt(GDc::GDconfig{T, iT, ST}, _::Function, _::Function) where 
                         {T, ST}
    η₀ = GDc.initialStep
    lo, up = GDc.stepBound
    @inline function (x, gx, _)
        η = if GDc.scaleStepBound
            clamp(η₀[], lo, up)
        else
            n = norm(gx)
            clamp(η₀[], lo/n, up/n)
        end
        x .-= η.*gx
    end
end

function convertExternalOpt(o!::F, f::Function, gf::Function) where {F}
    function (x, gx, fx)
        o!(x, gx, fx, f, gf)
        x
    end
end

getOptimizerConstructor(::Type{<:GDconfig}) = genLineSearchOpt

getOptimizerConstructor(::Type{<:Any}) =  convertExternalOpt

@inline function genOptimizer(::Val{M}, Mconfig::ConfigBox{T}, optimizer::O, 
                              pbs::AbstractVector{<:ParamBox{T}}, 
                              bs::AbstractVector{<:AbstractGTBasisFuncs{T, D}}, 
                              nuc::NTuple{NN, String}, 
                              nucCoords::NTuple{NN, NTuple{D, T}}, N) where {M, T, O, NN, D}
    (f0, getOFval), (g0, getOGval) = OFfunctions[M]

    f = function (x)
        xTemp = getindex.(pbs)
        setindex!.(pbs, x)
        gtb = GTBasis(bs)
        res = f0(gtb, nuc, nucCoords, N, Mconfig, printInfo=false)
        setindex!.(pbs, xTemp)
        getOFval(res)
    end

    gf = function (x)
        xTemp = getindex.(pbs)
        setindex!.(pbs, x)
        gtb = GTBasis(bs)
        fRes = f0(gtb, nuc, nucCoords, N, Mconfig, printInfo=false)
        gRes = g0(fRes, pbs, gtb, nuc, nucCoords, N)
        setindex!.(pbs, xTemp)
        getOGval(gRes), getOFval(fRes)
    end

    generator = getOptimizerConstructor(O)
    generator(optimizer, f, gf)
end


function formatTunableParams!(pbs::AbstractVector{<:ParamBox{T}}, 
                              pbcs::AbstractVector{<:ParameterizedContainer{T}}, 
                              filterParsForSafety::Bool=true) where {T}
    filterParsForSafety && getUnique!(pbs, compareFunction=compareParamBox)
    d = Dict{UInt, Array{T, 0}}()
    pbsNew = map(pbs) do p
        res = isDiffParam(p) ? p : changeMapping(outValCopy(p), DI(p.map))
        res.index[] = p.index[]
        res
    end
    for (j, c) in enumerate(pbcs)
        cNew = copyParContainer(c, 
            isDiffParam, 
            itself, 
            function (y)
                idx = findfirst(p->compareParamBoxCore2(y, p), pbs)
                if idx === nothing
                    yN = fullVarCopy(y)
                    data, sym = y.data[]
                    yN.data[] = get!(d, objectid(data), fill(y[])) => sym
                    yN
                else
                    pbsNew[idx]
                end
            end
        )
        pbcs[j] = cNew
    end
    pbs .= pbsNew
    map(x->dataOf(x)[begin], pbsNew) # Vector of parameters to be optimized.
end

function makeAbsLayerForXpnParams(pbs, bs, onlyForNegXpn::Bool=false; 
                                  forceDiffOn::Bool=false, tryJustFlipNegSign::Bool=true)
    xpnFilter = pb->(isOutSymEqual(pb, xpnSym) && (onlyForNegXpn ? pb()<0 : true))
    pbsDiffConfig = getproperty.(pbs, :canDiff)
    newDiffConfig = forceDiffOn ? Ref(fill(true)) : (fill∘getindex).(pbsDiffConfig)
    d = IdDict(pbsDiffConfig .=> newDiffConfig)
    absXpn1 = (pb::ParamBox)->changeMapping( pb, Absolute(pb.map), 
                                             canDiff=get!(d, pb.canDiff, 
                                                          fill(forceDiffOn ? 
                                                               true : pb.canDiff[])) )
    absXpn2 = if tryJustFlipNegSign
        function (pb::ParamBox)
            if (FLevel∘getFLevel)(pb) == IL || pb.map isa DI
                pb[] *= sign(pb[])
                pb
            else
                absXpn1(pb)
            end
        end
    else
        absXpn1
    end
    pbs = copyParContainer.(pbs, xpnFilter, absXpn2, itself)
    bs = copyParContainer.(bs, xpnFilter, absXpn2, itself)
    pbs, bs
end


"""

    optimizeParams!(pbs, bs, nuc, nucCoords, 
                    config=$(defaultPOconfigStr), N=getCharge(nuc); printInfo=true) -> 
    Tuple{Union{Vector{T}, Vector{<:Array{T}}}, 
          Vector{T}, 
          Vector{<:Array{T}}, 
          Union{Bool, Missing}} where {T}

    optimizeParams!(pbs, bs, nuc, nucCoords, 
                    N=getCharge(nuc), config=$(defaultPOconfigStr); printInfo=true) -> 
    Tuple{Union{Vector{T}, Vector{<:Array{T}}}, 
          Vector{T}, 
          Vector{<:Array{T}}, 
          Union{Bool, Missing}} where {T}

The main function to optimize the parameters of a given basis set. It returns a `Tuple` of 
relevant information. The first three elements are the energies, the parameter values, and 
the gradients from all the iteration steps respectively. The last element is the indicator 
of whether the optimization is converged if the convergence detection is on (i.e., 
`config.threshold` is not `NaN`), or else it's set to `missing`.

=== Positional argument(s) ===

`pbs::AbstractVector{<:ParamBox{T}}`: The parameters to be optimized that are extracted 
from the basis set. If the parameter is marked as "differentiable", the value of its input 
variable will be optimized.

`bs::AbstractVector{<:AbstractGTBasisFuncs{T, D}}`: The basis set to be optimized.

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
                         bs::AbstractVector{<:AbstractGTBasisFuncs{T, D}}, 
                         nuc::VectorOrNTuple{String, NN}, 
                         nucCoords::SpatialCoordType{T, D, NN}, 
                         config::POconfig{<:Any, M, CBT, <:Any, F}=defaultPOconfig, 
                         N::Union{Int, Tuple{Int}, NTuple{2, Int}}=getCharge(nuc); 
                         printInfo::Bool=true) where {T, D, NN, M, CBT, F}
    tBegin = time()

    pars = formatTunableParams!(pbs, bs)
    x = getindex.(pars)
    pbsN, bsN = makeAbsLayerForXpnParams(pbs, bs, 
                                         forceDiffOn=true, tryJustFlipNegSign=false)
    parsVals = Vector{T}[]

    i = 0
    Δt₁ = Δt₂ = 0
    threshold = config.threshold
    target = config.target
    maxStep = config.maxStep
    gap = min(100, max(maxStep ÷ 100 * 10, 1))

    target = ifelse(isNaN(target), (threshold[begin], threshold[end]), (threshold[begin],))
    detectConverge = false
    isConverged = map(target) do tar
        if isNaN(tar)
            _->true
        else
            detectConverge |= true
            vrs->isOscillateConverged(vrs, tar*sqrt(vrs[end]|>length), minimalCycles=4)[1]
        end
    end
    detectConverge || (isConverged = (_->false,))
    blConv = false

    nuc = arrayToTuple(nuc)
    nucCoords = genTupleCoords(T, nucCoords)
    f0s, g0s = OFfunctions[M]
    f0config = config.config
    optimize! = genOptimizer(Val(M), f0config, config.optimizer, 
                             pbsN, bsN, nuc, nucCoords, N)

    fx, gx, Δt₁ = optimizeParamsCore(f0s, g0s, pbsN, bsN, nuc, nucCoords, N, f0config)
    fVstr = ndims(fx)==0 ? "𝑓" : "𝒇"
    fVals = [fx]
    grads = [gx]

    while !all(f(x) for (f, x) in zip(isConverged, (fVals, grads))) && i < maxStep

        if i%gap == 0 && printInfo
            println(rpad("Step $(i): ", 11), lpad("$(fVstr) = ", 6), 
                    alignNumSign(fx, roundDigits=nDigitShown))
            print(rpad("", 11), lpad("𝒙 = ", 6))
            println(IOContext(stdout, :limit => true), round.(x, digits=nDigitShown))
            print(rpad("", 11), lpad("∇$(fVstr) = ", 6))
            println(IOContext(stdout, :limit => true), round.(gx, digits=nDigitShown))
            println("Step duration: ", round(Δt₁+Δt₂, digits=6), " seconds.\n")
        end

        t3 = time_ns()
        optimize!(x, gx, fx)
        t4 = time_ns()
        Δt₂ = (t4 - t3) / 1e9
        setindex!.(pars, x)

        fx, gx, Δt₁ = optimizeParamsCore(f0s, g0s, pbsN, bsN, nuc, nucCoords, N, f0config)
        push!(fVals, fx)
        push!(grads, gx)
        push!(parsVals, x)
        i += 1
    end

    tEnd = time()

    if printInfo
        print("The optimization of parameters \n𝒙 := ")
        println(IOContext(stdout, :limit => true), "$((first∘indVarOf).(pbs)) ")
        print("with respect to $(fVstr)(𝒙) from the profile ")
        printstyled(":$M", bold=true)
        println(" just ended at")
        println(rpad("Step $(i): ", 11), lpad("$(fVstr) = ", 6), 
                alignNumSign(fVals[end], roundDigits=nDigitShown))
        print(rpad("", 11), lpad("𝒙 = ", 6))
        println(IOContext(stdout, :limit => true), 
                round.(parsVals[end], digits=nDigitShown))
        print(rpad("", 11), lpad("∇$(fVstr) = ", 6))
        println(IOContext(stdout, :limit => true), 
                round.(grads[end], digits=nDigitShown))
        println("Optimization duration: ", round((tEnd-tBegin)/60, digits=6), 
                " minutes.")
        if detectConverge
            println("The result has" * ifelse(blConv, "", " not") *" converged: ")
            println("∥Δ$(fVstr)∥₂ → ", 
                    round(norm(fVals[end] - fVals[end-1]), digits=nDigitShown), ", ", 
                    "∥vec(∇$(fVstr))∥₂ → ", 
                    round(norm(grads[end]), digits=nDigitShown), ".\n")
        end
    end

    pbs, bs = makeAbsLayerForXpnParams(pbs, bs, true)

    fVals, parsVals, grads, ifelse(detectConverge, blConv, missing)
end

optimizeParams!(pbs, bs, nuc, nucCoords, N::Int, config=defaultPOconfig; printInfo=true) = 
optimizeParams!(pbs, bs, nuc, nucCoords, config, N; printInfo)

function optimizeParamsCore((f0, getOFval), (g0, getOGval), 
                            pbs, bs, nuc, nucCoords, N, config)
    gtb = GTBasis(bs)
    t1 = time_ns()
    fRes = f0(gtb, nuc, nucCoords, N, config, printInfo=false)
    fx = getOFval(fRes)
    gx = (getOGval∘g0)(fRes, pbs, gtb, nuc, nucCoords, N)
    t2 = time_ns()
    fx, gx, (t2 - t1) / 1e9
end