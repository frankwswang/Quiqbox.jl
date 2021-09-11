export gradDescent!, updateParams!, optimizeParams!


"""

    gradDescent!(pars::Vector{<:Real}, grads::Vector{<:Real}, η=0.001) -> 
    pars::Vector{<:Real}

Default gradient descent method in used in Quiqbox.
"""
function gradDescent!(pars::Vector{<:Real}, grads::Vector{<:Real}, η=0.001)
    @assert length(pars) == length(grads) "The length of gradients and corresponding "*
                                          "parameters should be the same."
    pars .-= η*grads
end


"""

    updateParams!(pbs::Array{<:ParamBox, 1}, grads::Array{<:Real, 1}; 
                  method::F=gradDescent!) where {F<:Function} -> Array{<:ParamBox, 1}

Given a `Vector` of parameters::`ParamBox` and its gradients with respect to each 
parameter, update the `ParamBox`s and return the updated values.
"""
function updateParams!(pbs::Vector{<:ParamBox}, grads::Vector{<:Real}; 
                       method::F=gradDescent!) where {F<:Function}
    parVals = [i[] for i in pbs]
    method(parVals, grads)
    for (m,n) in zip(pbs, parVals)
        m[] = n
    end
    parVals
end


"""

    defaultECmethod(HFtype, Hcore, HeeI, S, Ne) -> 
    E::Float64, C::Union{Array{Float64, 1}, NTuple{2, Array{Float64, 2}}}

The default engine (`Function`) in `optimizeParams!` to update Hartree-Fock energy and 
coefficient matrix(s). 
"""
function defaultECmethod(HFtype, Hcore, HeeI, S, Ne)
    X = getX(S)
    res = runHFcore(Ne, Hcore, HeeI, S, X, guessC(S, Hcore; X); 
                    printInfo=false, HFtype, scfConfig=defaultSCFconfig)
    res.E0HF, res.C
end


"""

    optimizeParams!(bs::Array{<:FloatingGTBasisFuncs, 1}, pbs::Array{<:ParamBox, 1},
                    nuc::Array{String, 1}, nucCoords::Array{<:AbstractArray, 1}, 
                    Ne::Union{NTuple{2, Int}, Int}=getCharge(nuc);
                    Etarget::Float64=NaN, threshold::Float64=1e-4, maxSteps::Int=2000, 
                    printInfo::Bool=true, GDmethod::F1=gradDescent!, HFtype::Symbol=:RHF, 
                    ECmethod::F2=Quiqbox.defaultECmethod) where 
                   {F1<:Function, F2<:Function} -> 
    Es::Array{Float64, 1}, pars::Array{Float64, 2}, grads::Array{Float64, 2}

The main function to optimize the parameters of a given basis set.

=== Positional argument(s) ===

`bs::Array{<:FloatingGTBasisFuncs, 1}`: Basis set.

`pbs::Array{<:ParamBox, 1}`: The parameters to be optimized that are extracted from the 
basis set.

`nuc::Array{String, 1}`: The nuclei of the molecule.

`nucCoords::Array{<:AbstractArray, 1}`: The nuclei coordinates.

`Ne::Union{NTuple{2, Int}, Int}`: The total number of electrons or the numbers of electrons 
with different spins respectively.

=== Keyword argument(s) ===

`Etarget::Float64`: The target Hartree-Hock energy intent to achieve.

`threshold::Float64`: The threshold for the convergence when evaluating difference between 
the latest two energies.

`maxSteps::Int`: Maximum allowed iteration steps regardless of whether the optimization 
iteration converges.

`printInfo::Bool`: Whether print out the information of each iteration step.

`GDmethod::F1`: Applied gradient descent `Function`.

`HFtype::Symbol`: Hartree-Fock type. Available values are `:RHF` and `:UHF`.

`ECmethod::F2`: The `Function` used to update Hartree-Fock energy and coefficient matrix(s) 
during the optimization iterations.
=== Keyword argument(s) ===
"""
function optimizeParams!(bs::Vector{<:FloatingGTBasisFuncs}, pbs::Vector{<:ParamBox},
                         nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}, 
                         Ne::Union{NTuple{2, Int}, Int}=getCharge(nuc);
                         Etarget::Float64=NaN, threshold::Float64=1e-4, maxSteps::Int=2000, 
                         printInfo::Bool=true, GDmethod::F1=gradDescent!, 
                         HFtype::Symbol=:RHF, ECmethod::F2=defaultECmethod) where 
                        {F1<:Function, F2<:Function}
    tAll = @elapsed begin
        
        i = 0
        Es = Float64[]
        pars = zeros(0, length(pbs))
        grads = zeros(0, length(pbs))
        gap = max(5, maxSteps ÷ 200 * 5)

        if Etarget === NaN
            isConverged = (Es) -> isOscillateConverged(Es, threshold, leastCycles=3)
        else
            isConverged = Es -> (abs(Es[end] - Etarget) < threshold)
        end

        parsL = [i[] for i in pbs]

        Npars = length(parsL)

        while true
            S = overlaps(bs)
            Hcore = coreH(bs, nuc, nucCoords)
            HeeI = eeInteractions(bs)
            E, C = ECmethod(HFtype, Hcore, HeeI, S, Ne)

            t = @elapsed begin
                grad = gradHFenegy(bs, pbs, C, S, nuc, nucCoords)
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

            parsL = updateParams!(pbs, grad, method=GDmethod)
            
            !isConverged(Es) && i < maxSteps || break
            
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
        !isConverged(Es) && println("The result has not converged.")
    end

    Es, pars, grads
end