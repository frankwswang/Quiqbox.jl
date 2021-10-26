function addToDataChain!(env::Vector{Float64}, atm::Vector{Int32}, bas::Vector{Int32}, 
                         bf::FloatingGTBasisFuncs)
    center = [bf.center[1](), bf.center[2](), bf.center[3]()]
    xpns = Float64[]
    cons = Float64[]
    for i in bf.gauss
        push!(xpns, i.xpn())
        push!(cons, i.con())
    end
    nGauss = bf.gauss |> length
    envEndIndex = length(env)
    gAtmIndex = length(atm) / 6 |> Int32

    append!(env, center)
    append!(atm, Int32[0, envEndIndex, 1, 0, 0, 0])
    envEndIndex += 3

    append!(env, xpns)
    norm = bf.normalizeGTO ? normOfGTOin(bf) : 1.0
    append!(env, cons.*norm)
    append!(bas, Int32[gAtmIndex, AngularMomentumList[bf.subshell], nGauss, 1, 0, 
                       envEndIndex, envEndIndex+nGauss, 0])
    envEndIndex += nGauss*2
    (env, atm, bas)
end


function addToDataChain!(env::Vector{Float64}, atm::Vector{Int32}, 
                         nuclei::Vector{String}, nucleiCoords::Vector{<:AbstractArray})
    @compareLength nuclei nucleiCoords "nuclei" "their coordinates"
    envEndIndex = length(env)
    len = length(nuclei)
    atmsConfig = Int32[]
    for i = 1:len
        append!(env, Float64[nucleiCoords[i][1], nucleiCoords[i][2], nucleiCoords[i][3]])
        append!(atmsConfig, Int32[getCharge(nuclei[i]), envEndIndex, 1, 0, 0, 0])
        envEndIndex += 3
    end
    append!(atm, atmsConfig)
    (env, atm)
end