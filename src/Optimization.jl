export gradDescent!, updateParams!, optimizeParams!

using LinearAlgebra: norm

"""

    gradDescent!(pars::Vector{<:Real}, grads::Vector{<:Real}, 
                 η=0.001, clipThreshold=0.08*sqrt(length(grad))/norm(η)) -> 
    pars::Vector{<:Real}

Default gradient descent method in used in Quiqbox.
"""
function gradDescent!(pars::Vector{<:Real}, grad::Vector{<:Real}, η=0.001, 
                      clipThreshold::Real=0.08*sqrt(length(grad))/norm(η))
    @assert length(pars) == length(grad) "The length of gradients and corresponding "*
                                         "parameters should be the same."
    gNorm = norm(grad)
    gradNew = if gNorm > clipThreshold
        clipThreshold / gNorm * grad
    else
        grad
    end

    pars .-= η.*gradNew
end


"""

    updateParams!(pbs::Array{<:ParamBox, 1}, grads::Array{<:Real, 1}, 
                  method::F=gradDescent!) where {F<:Function} -> Array{<:ParamBox, 1}

Given a `Vector` of parameters::`ParamBox` and its gradients with respect to each 
parameter, update the `ParamBox`s and return the updated values.
"""
function updateParams!(pbs::Vector{<:ParamBox}, grads::Vector{<:Real}, 
                       method!::F=gradDescent!) where {F<:Function}
    parVals = [i[] for i in pbs]
    method!(parVals, grads)
    for (m,n) in zip(pbs, parVals)
        m[] = n
    end
    parVals
end


"""

    defaultECmethod(HFtype::Symbol, nuc::Array{String, 1}, 
                    nucCoords::Array{<:AbstractArray, 1}) -> 
    fEC::Function

    defaultECmethod(Ne::NTuple{2, Int}, nuc::Array{String, 1}, 
                    nucCoords::Array{<:AbstractArray, 1}) -> 
    fEC::Function

Return the default `Function` in `optimizeParams!` that will be used to update Hartree-Fock 
energy and coefficient matrix(s):

    fEC(Hcore, HeeI, S) -> E::Float64, 
                           C::Union{Array{Float64, 1}, NTuple{2, Array{Float64, 2}}}

When `HFtype` is set to  `:RHF` or input argument is only a 2-element `Tuple`, the returned 
function is for RHF methods.
"""
function defaultECmethod(HFtype::Symbol, nuc::Vector{String}, 
                         nucCoords::Vector{<:AbstractArray}, Ne::Int=getCharge(nuc))
    HFtype == :UHF && ( Ne = (Ne÷2, Ne-Ne÷2) )
    defaultECmethodCore(Ne, nuc, nucCoords)
end

# Direct specify number of α and β electrons for UHF.
defaultECmethod(Nes::NTuple{2, Int}, 
                nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}) = 
defaultECmethodCore(Nes, nuc, nucCoords)

function defaultECmethodCore(Ne, nuc::Vector{String}, nucCoords::Vector{<:AbstractArray})
    function (Hcore, HeeI, bs, S)
        X = getX(S)
        res = runHFcore(defaultSCFconfig, Ne, Hcore, HeeI, S, X, 
                        guessC(:SAD, (length(Ne)==2), S, X, 
                               Hcore, HeeI, bs, nuc, nucCoords); 
                        printInfo=false)
        res.E0HF, res.C
    end
end


"""

    optimizeParams!(bs::Array{<:FloatingGTBasisFuncs, 1}, pbs::Array{<:ParamBox, 1},
                    nuc::Array{String, 1}, nucCoords::Array{<:AbstractArray, 1};
                    Etarget::Float64=NaN, threshold::Float64=1e-4, maxSteps::Int=2000, 
                    printInfo::Bool=true, GDmethod::F1=gradDescent!, 
                    ECmethod::F2=Quiqbox.defaultECmethod(:RHF, nuc, nucCoords)) where 
                   {F1<:Function, F2<:Function} -> 
    Es::Array{Float64, 1}, pars::Array{Float64, 2}, grads::Array{Float64, 2}

The main function to optimize the parameters of a given basis set.

=== Positional argument(s) ===

`bs::Array{<:FloatingGTBasisFuncs, 1}`: Basis set.

`pbs::Array{<:ParamBox, 1}`: The parameters to be optimized that are extracted from the 
basis set.

`nuc::Array{String, 1}`: The nuclei of the molecule.

`nucCoords::Array{<:AbstractArray, 1}`: The nuclei coordinates.

`ECmethod::F2`: The `Function` used to update Hartree-Fock energy and coefficient matrix(s) 
during the optimization iterations. Default setting is 
`Quiqbox.defaultECmethod(:RHF, nuc, nucCoords)` for RHF and the number of electron is equal 
to nuclei charge.

=== Keyword argument(s) ===

`Etarget::Float64`: The target Hartree-Hock energy intent to achieve.

`threshold::Float64`: The threshold for the convergence when evaluating difference between 
the latest few energies. When set to `NaN`, there will be no convergence detection.

`maxSteps::Int`: Maximum allowed iteration steps regardless of whether the optimization 
iteration converges.

`printInfo::Bool`: Whether print out the information of each iteration step.

`GDmethod::F1`: Applied gradient descent `Function`. Default method is 
`Quiqbox.gradDescent!`.
"""
function optimizeParams!(bs::Vector{<:FloatingGTBasisFuncs}, pbs::Vector{<:ParamBox},
                         nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}, 
                         ECmethod::F2=defaultECmethod(:RHF, nuc, nucCoords);
                         Etarget::Float64=NaN, threshold::Float64=1e-4, maxSteps::Int=500, 
                         printInfo::Bool=true, GDmethod::F1=gradDescent!) where 
                        {F1<:Function, F2<:Function}
    tAll = @elapsed begin

        i = 0
        Es = Float64[]
        pars = zeros(0, length(pbs))
        grads = zeros(0, length(pbs))
        gap = min(100, max(maxSteps ÷ 200 * 5, 1))
        detectConverge = isnan(threshold) ? false : true

        if Etarget === NaN
            isConverged = (Es) -> isOscillateConverged(Es, threshold, leastCycles=3)
        else
            isConverged = Es -> (abs(Es[end] - Etarget) < threshold)
        end

        parsL = [i[] for i in pbs]

        while true
            S = overlaps(bs)
            Hcore = coreH(bs, nuc, nucCoords)
            HeeI = eeInteractions(bs)
            E, C = ECmethod(Hcore, HeeI, bs, S)

            t = @elapsed begin
                grad = gradHFenergy(bs, pbs, C, S, nuc, nucCoords)
            end

            push!(Es, E)
            pars = vcat(pars, parsL |> transpose)
            grads = vcat(grads, grad |> transpose)

            if i%gap == 0 && printInfo
                println(rpad("Step $i: ", 15), rpad("E = $(E)", 26))
                print(rpad("", 10), "params = ")
                println(IOContext(stdout, :limit => true), parsL)
                print(rpad("", 12), "grad = ")
                println(IOContext(stdout, :limit => true), grad)
                println("Step duration: ", t, " seconds.\n")
            end

            parsL = updateParams!(pbs, grad, GDmethod)

            !(detectConverge && isConverged(Es)) && i < maxSteps || break

            i += 1
        end

        popfirst!(Es)

    end

    if printInfo
        println("The iteration just ended at")
        println(rpad("Step $(i): ", 15), rpad("E = $(Es[end])", 26))
        print(rpad("", 10), "params = ")
        println(IOContext(stdout, :limit => true), pars[end, :])
        print(rpad("", 12), "grad = ")
        println(IOContext(stdout, :limit => true), grads[end, :])
        println("Optimization duration: ", tAll/60, " minutes.")
        if detectConverge
            println("The result has" * (isConverged(Es) ? "" : " not") *" converged.")
        end
    end

    Es, pars, grads
end