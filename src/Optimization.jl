export gradDescent!, updateParams!, optimizeParams!

function gradDescent!(pars::Array{<:Real, 1}, grads::Array{<:Real, 1}, η=0.001)
    @assert length(pars) == length(grads) "The length of gradients and correponding parameters should be the same."
    pars .-= η*grads
end


function updateParams!(pbs::Array{<:ParamBox, 1}, grads::Array{<:Real, 1}; 
                       method::F=gradDescent!) where {F<:Function}
    parVals = [i[] for i in pbs]
    method(parVals, grads)
    for (m,n) in zip(pbs, parVals)
        m[] = n
    end
    parVals
end


function defaultECmethod(HFtype, Hcore, HeeI, S, Ne)
    X = getX(S)
    res = runHFcore(Ne, Hcore, HeeI, S, X, guessC(S, Hcore; X); 
                    printInfo=false, HFtype, scfConfig=defaultSCFconfig)
    res.E0HF, res.C
end


function optimizeParams!(bs::Array{<:FloatingGTBasisFunc, 1}, pbs::Array{<:ParamBox, 1},
                         mol::Array{String, 1}, nucCoords::Array{<:AbstractArray, 1}, 
                         Ne::Union{NTuple{2, Int}, Int}=getCharge(mol);
                         Etarget::Float64=NaN, threshold::Float64=0.0001, maxSteps::Int=2000, 
                         printInfo::Bool=true, GDmethod::F1=gradDescent!, HFtype=:RHF, 
                         ECmethod::F2=defaultECmethod) where {F1<:Function, F2<:Function}
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
            S = dropdims(overlaps(bs), dims=3)
            Hcore = dropdims(coreH(bs, mol, nucCoords), dims=3)
            HeeI = dropdims(eeInteractions(bs), dims=5)
            E, C = ECmethod(HFtype, Hcore, HeeI, S, Ne)

            t = @elapsed begin
                grad = gradHFenegy(bs, pbs, C, S, mol, nucCoords)
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