export GDconfig, POconfig, optimizeParams!

using LinearAlgebra: norm
using LineSearches

const OFmethods = [:HFenergy, :DirectRHFenergy]
const defaultPOinfoL = 2
const POinterValMaxStore = 50

const OFconversions = Dict([runHFcore] .=> 
                           [( x->x[begin][begin].shared.Etots[end], 
                              x->last.(getproperty.(x[begin], :Cs)) )
                           ])
const OFfunctions = Dict(OFmethods .=> 
    [
        (
            ( runHFcore, OFconversions[runHFcore][begin] ), 
            ( (tVars, pbs, gtb, nuc, nucCoords, N)->
               gradOfHFenergy(pbs, gtb, OFconversions[runHFcore][end](tVars), 
                              nuc, nucCoords, N), 
            itself )
        ), 
        (
            ( (gtb::GTBasis, nuc, nucCoords, N, config; printInfo) -> 
               config(getEhf, gtb, nuc, nucCoords, N), 
              itself ), 
            ( ((_, pbs, gtb::GTBasis{T, D, BN}, nuc, nucCoords, N) where {T, D, BN}) -> 
              gradOfHFenergy(pbs, gtb, (getXcore1(gtb.S),), nuc, nucCoords, N), 
              itself )
        )
    ]
)

const defaultPOconfigStr = "POconfig()"
const defaultHFthresholdForHFgrad = getAtolVal(Float64)
const defaultHFconfigForPO = HFconfig(SCF=SCFconfig(threshold=defaultHFthresholdForHFgrad))
const defaultDRHFconfigForPO = FuncArgConfig(getEhf)
const defaultOFmethodConfigs = [defaultHFconfigForPO, defaultDRHFconfigForPO]
const defaultOFconfigs = Dict(OFmethods .=> defaultOFmethodConfigs)
const defaultOFmethod = OFmethods[begin]


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
             initialStep::Union{Array{T, 0}, T}=ifelse(M==$(iT), 0.1, 1.0); 
             stepBound::NTuple{2, T}=convert.(eltype(initialStep), (0, Inf)), 
             scaleStepBound::Bool=ifelse(M==typeof($(itself)), true, false)) where 
            {M, T<:AbstractFloat} -> 
    GDconfig{T, M, typeof(initialStep)}
"""
mutable struct GDconfig{T, M, ST<:Union{Array{T, 0}, T}} <: ConfigBox{T, GDconfig, M}
    lineSearchMethod::M
    initialStep::ST
    stepBound::NTuple{2, T}
    scaleStepBound::Bool

    function GDconfig(lineSearchMethod::M=BackTracking(), 
                      initialStep::Union{Array{T, 0}, T}=ifelse(M==iT, 0.1, 1.0); 
                      stepBound::NTuple{2, T}=convert.(eltype(initialStep), (0, Inf)), 
                      scaleStepBound::Bool=ifelse(M==iT, true, false)) where 
                     {M, T<:AbstractFloat}
        0 < initialStep[] || throw(DomainError(initialStep[], "The initial step size "*
                                   "should be positive."))
        stepBound[begin] > stepBound[end] && 
        throw(DomainError(stepBound, "The bound set for initialStep should be a valid "*
              "closed interval."))
        (M == iT || hasproperty(LineSearches, (nameof∘typeof)(lineSearchMethod))) || 
        throw(DomainError(lineSearchMethod, 
              "The input `lineSearchMethod` is not supported."))
        new{T, M, typeof(initialStep)}(lineSearchMethod, initialStep, stepBound, 
                                       scaleStepBound)
    end
end


const partialDefaultPOconfigPars = [NaN, (5e-8, 5e-5), 200, GDconfig(), 
                                    (true, false, false, false)]

const defaultPOconfigPars = begin
    OFmethodVals = Val.(OFmethods)
    cfgs = vcat.( OFmethodVals, getindex.(Ref(defaultOFconfigs), OFmethods), 
                  Ref(partialDefaultPOconfigPars) )
    Dict(vcat(OFmethods.=>cfgs, OFmethodVals.=>cfgs))
end

"""

    POconfig{T, M, CBT<:ConfigBox, TH<:Union{T, NTuple{2, T}}, 
             OM} <: ConfigBox{T, POconfig, M}

The mutable container of parameter optimization configurations.

≡≡≡ Field(s) ≡≡≡

`method::Val{M}`: The method that defines the objective function (e.g., HF energy) for the 
optimization. Available values of `M` from Quiqbox are $(string(OFmethods)[2:end-1]).

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

    optimizer(f::Function, gf::Function, x0::Vector{T}) where {T} -> Function

where `f` is the objective function to be minimized, `gf` is a function that returns 
both the gradient and the returned value of `f` given the input value as a vector. `x0` is 
the initial input value. The output of `optimizer`, if we name it `optimize!`, should have 
the corresponding function signature: 

    optimize!(x::Vector{T}, gx::Vector{T}, fx::T) where {T}

where `x`, `gx`, `fx` are the input value, the gradient, and the returned value of `f` 
respectively at one step. In other words, `(gx, fx) == (gx, f(x)) == gf(x)`. After 
accepting those arguments, `optimizer` should update (i.e. mutate the elements of) `x` so 
that f(x) will have lower returned value.

`saveTrace::NTuple{4, Bool}`: Determine whether saving (by pushing) the intermediate 
information from all the iterations steps to the output of `optimizeParams!`.
The types of relevant information are:

| Sequence | Corresponding Information |
|  :---:   | :---:                     |
| 1 | function value of the applied method |
| 2 | parameter value(s) |
| 3 | function gradient with respect to the parameter(s) |
| 4 | complete output of the applied method |

≡≡≡ Initialization Method(s) ≡≡≡

    POconfig(;method::Union{Val{M}, Symbol}=$(defaultOFmethod), 
              config::ConfigBox=Quiqbox.defaultOFconfigs[method], 
              target::T=$(partialDefaultPOconfigPars[1]), 
              threshold::Union{T, NTuple{2, T}}=$(partialDefaultPOconfigPars[2]), 
              maxStep::Int=$(partialDefaultPOconfigPars[3]), 
              optimizer::Function=$(partialDefaultPOconfigPars[4]|>typeof|>nameof)()) where 
            {T, M} -> 
    POconfig{T, M}

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> POconfig();

julia> POconfig(maxStep=100);
```
"""
mutable struct POconfig{T, M, CBT<:ConfigBox, TH<:Union{T, NTuple{2, T}}, 
                        OM} <: ConfigBox{T, POconfig, M}
    method::Val{M}
    config::CBT
    target::T
    threshold::TH
    maxStep::Int
    optimizer::OM
    saveTrace::NTuple{4, Bool}
end

POconfig(a1::Symbol, args...) = POconfig(Val(a1), args...)

function POconfig(t::NamedTuple)
    bl = hasproperty(t, :method)
    genNamedTupleC(:POconfig, defaultPOconfigPars[bl ? t.method : defaultOFmethod])(t)
end

POconfig(;kws...) = 
isempty(kws) ? POconfig(defaultPOconfigPars[defaultOFmethod]...) : POconfig(kws|>NamedTuple)

const defaultPOconfig = Meta.parse(defaultPOconfigStr) |> eval


function genLineSearchOpt(GDc::GDconfig{T, M, ST}, 
                          f::Function, gf::Function, x0) where {T, M, ST}
    l = GDc.lineSearchMethod
    η₀ = GDc.initialStep
    lo, up = GDc.stepBound
    xBuf = deepcopy(x0)
    gBuf = similar(x0)
    s = similar(gBuf)

    ϕForLS(η) = f(xBuf .+ η.*s)

    function 𝑑ϕForLS!(η)
        gBuf .= gf(xBuf .+ η.*s)[begin]
        dot(gBuf, s)
    end

    function ϕ𝑑ϕForLS!(η)
        gxN, ϕ = gf(xBuf .+ η.*s)
        gBuf .= gxN
        𝑑ϕ = dot(gBuf, s)
        ϕ, 𝑑ϕ
    end

    @inline function (x, gx, fx)
        η = if GDc.scaleStepBound
            n = norm(gx)
            clamp(η₀[], lo/n, up/n)
        else
            clamp(η₀[], lo, up)
        end
        s .= -gx
        gBuf .= gx
        dϕ₀ = dot(s, gBuf)
        ηNew, _ = l(ϕForLS, 𝑑ϕForLS!, ϕ𝑑ϕForLS!, η, fx, dϕ₀) # new step η
        ST <: Array{T, 0} && (η₀[] = ηNew)
        x .-= ηNew.*gx
        xBuf .= x
    end
end

function genLineSearchOpt(GDc::GDconfig{T, iT, ST}, ::Function, ::Function, _) where {T, ST}
    η₀ = GDc.initialStep
    lo, up = GDc.stepBound
    @inline function (x, gx, _)
        η = if GDc.scaleStepBound
            n = norm(gx)
            clamp(η₀[], lo/n, up/n)
        else
            clamp(η₀[], lo, up)
        end
        x .-= η.*gx
    end
end

function convertExternalOpt(genO::F, f::Function, gf::Function, x0) where {F}
    o! = genO(f, gf, x0)
    function (x, gx, fx)
        o!(x, gx, fx)
        x
    end
end

getOptimizerConstructor(::Type{<:GDconfig}) = genLineSearchOpt

getOptimizerConstructor(::Type{<:Any}) =  convertExternalOpt

@inline function genOptimizer(::Val{M}, Mconfig::ConfigBox{T1}, optimizer::O, 
                              pbs::AbstractVector{<:ParamBox{T2}}, 
                              bs::AbstractVector{<:GTBasisFuncs{T2, D}}, 
                              nuc::NTuple{NN, String}, 
                              nucCoords::NTuple{NN, NTuple{D, T2}}, N) where 
                             {M, T1, T2, O, NN, D}
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
    generator(optimizer, f, gf, getindex.(pbs))
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
                    config=$(defaultPOconfigStr), N=getCharge(nuc); 
                    printInfo=true, infoLevel=$(defaultPOinfoL)) -> 
    Vector{Any}

    optimizeParams!(pbs, bs, nuc, nucCoords, 
                    N=getCharge(nuc), config=$(defaultPOconfigStr); 
                    printInfo=true, infoLevel=$(defaultPOinfoL)) -> 
    Vector{Any}

The main function to optimize the parameters of a given basis set. It returns a `Vector` of 
the relevant information. The first element is the indicator of whether the 
optimization is converged if the convergence detection is on (i.e., `config.threshold` is 
not `NaN`), or else it's set to `missing`. More elements may be pushed to the returned 
result in order depending on `config.method`.

=== Positional argument(s) ===

`pbs::AbstractVector{<:ParamBox{T}}`: The parameters to be optimized that are extracted 
from the basis set. If the parameter is marked as "differentiable", the value of its input 
variable will be optimized.

`bs::AbstractVector{<:GTBasisFuncs{T, D}}`: The basis set to be optimized.

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

`infoLevel::Int`: Printed info's level of details when `printInfo=true`. The higher 
(the absolute value of) it is, more intermediate steps will be printed. Once `infoLevel` 
achieve `5`, every step will be printed.
"""
function optimizeParams!(pbs::AbstractVector{<:ParamBox{T}}, 
                         bs::AbstractVector{<:GTBasisFuncs{T, D}}, 
                         nuc::AVectorOrNTuple{String, NN}, 
                         nucCoords::SpatialCoordType{T, D, NN}, 
                         config::POconfig{<:Any, M, CBT, <:Any, F}=defaultPOconfig, 
                         N::Union{Int, Tuple{Int}, NTuple{2, Int}}=getCharge(nuc); 
                         printInfo::Bool=true, 
                         infoLevel::Int=defaultPOinfoL) where {T, D, NN, M, CBT, F}
    tBegin = time()

    pars = formatTunableParams!(pbs, bs)
    x = getindex.(pars)
    pbsN, bsN = makeAbsLayerForXpnParams(pbs, bs, 
                                         forceDiffOn=true, tryJustFlipNegSign=false)

    i = 0
    Δt₁ = Δt₂ = 0
    threshold = config.threshold
    targets = (config.target, 0) # (fTarget, gTarget)
    maxStep = config.maxStep
    adaptStepBl = genAdaptStepBl(infoLevel, maxStep)

    thresholds = ifelse(isNaN(config.target), 
                        [threshold[begin], threshold[end]], [threshold[begin]])
    detectConverge = false
    isConverged = map(targets, thresholds) do target, thd
        if isNaN(thd)
            _->true
        else
            detectConverge |= true
            vrs->isOscillateConverged(vrs, thd*sqrt(vrs[end]|>length), target, 
                                      maxRemains=POinterValMaxStore)[begin]
        end
    end
    detectConverge || (isConverged = (_->false,))
    blConv = false

    nuc = arrayToTuple(nuc)
    nucCoords = genTupleCoords(T, nucCoords)
    f0s, g0s = OFfunctions[M]
    f0config = config.config
    initializeOFconfig!(f0config, bsN, nuc, nucCoords)
    optimize! = genOptimizer(Val(M), f0config, config.optimizer, 
                             pbsN, bsN, nuc, nucCoords, N)

    fx, gx, fRes, Δt₁ = optimizeParamsCore(f0s, g0s, pbsN, bsN, nuc, nucCoords, N, f0config)
    fVstr = ndims(fx)==0 ? "𝑓" : "𝒇"
    fVals = [fx]
    pVals = [x]
    grads = [gx]
    fRess = [fRes]
    allInfo = [fVals, pVals, grads, fRess]
    saveTrace = config.saveTrace

    while !(blConv = all(f(x) for (f, x) in zip(isConverged, (fVals, grads)))) && i<maxStep

        if printInfo && adaptStepBl(i)
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
        updateOFconfig!(f0config, f0s[begin], fRes)

        fx, gx, fRes, Δt₁ = optimizeParamsCore(f0s, g0s, pbsN, bsN, nuc, nucCoords, N, 
                                               f0config)
        i += 1
        saveTrace[begin]   || i < POinterValMaxStore || popfirst!(fVals)
        saveTrace[begin+1] && push!(pVals, x)
        saveTrace[begin+2] || i < POinterValMaxStore || popfirst!(grads)
        saveTrace[end]     && push!(fRess, fRes)
        push!(fVals, fx)
        push!(grads, gx)
    end

    tEnd = time()

    pbsT, bsT = makeAbsLayerForXpnParams(pbs, bs, true)
    pbs .= pbsT
    bs  .= bsT

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
                round.(getindex.(pbs), digits=nDigitShown))
        print("after ", round((tEnd-tBegin)/60, digits=6), " minutes. ")
        if detectConverge
            println("The iteration has" * ifelse(blConv, "", " not") *" converged: ")
            println("∥Δ$(fVstr)∥₂ → ", 
                    round(norm(fVals[end] - fVals[end-1]), digits=nDigitShown), ", ", 
                    "∥vec(∇$(fVstr))∥₂ → ", 
                    round(norm(grads[end]), digits=nDigitShown), ".")
        end
        println()
    end
    res = Any[ifelse(detectConverge, blConv, missing)]
    append!(res, view(allInfo, findall(itself, saveTrace)))
    res
end

optimizeParams!(pbs, bs, nuc, nucCoords, N::Int, config=defaultPOconfig; 
                printInfo=true, infoLevel=defaultPOinfoL) = 
optimizeParams!(pbs, bs, nuc, nucCoords, config, N; printInfo, infoLevel)

function optimizeParamsCore((f0, getOFval), (g0, getOGval), 
                            pbs, bs, nuc, nucCoords, N, config)
    gtb = GTBasis(bs)
    t1 = time_ns()
    fRes = f0(gtb, nuc, nucCoords, N, config, printInfo=false)
    fx = getOFval(fRes)
    gx = (getOGval∘g0)(fRes, pbs, gtb, nuc, nucCoords, N)
    t2 = time_ns()
    fx, gx, fRes, (t2 - t1) / 1e9
end


initializeOFconfig!(ofc::ConfigBox, _, _, _) = itself(ofc)

function initializeOFconfig!(ofc::HFconfig{<:Any, HFT, iT}, bs, nuc, nucCoords) where {HFT}
    bls = iszero.(ofc.C0.mat)
    if any(bls)
        b = GTBasis(bs)
        X = getX(b.S)
        Hcore = coreH(b, nuc, nucCoords)
        C0new = getCfromSAD(Val(HFT), b.S, Hcore, b.eeI, b.basis, nuc, nucCoords, X)
        for (co, cn, bl) in zip(ofc.C0.mat, C0new, bls)
            bl && (co .= cn)
        end
    end
    ofc
end


updateOFconfig!(ofc::ConfigBox, _, _) = itself(ofc)

function updateOFconfig!(ofc::HFconfig{<:Any, <:Any, iT}, f::Function, fRes)
    C0new = OFconversions[f][end](fRes)
    map(ofc.C0.mat, C0new) do co, cn
        co .= cn
    end
    ofc
end