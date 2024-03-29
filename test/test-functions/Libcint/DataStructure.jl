function Nlα(l, α)
    if l < 2
        sqrt( 2^(2l+3) * factorial(l+1) * 2^(l+1.5) / 
              (factorial(2l+2) * √π) ) * α^(0.5l + 0.75)
    else
        # for higher angular momentum make the upper bound of norms be 1.
        sqrt( 2^(3l+1.5) * factorial(l) / (factorial(2l) * π^1.5) ) * α^(0.5l + 0.75)
    end
end

Nlα(subshell::String, α) = Nlα(Quiqbox.AngMomNumberList[subshell], α)


Nijk(i, j, k) = (2/π)^0.75 * sqrt( 2^(3*(i+j+k)) * factorial(i) * factorial(j) * 
                                   factorial(k) / (factorial(2i) * factorial(2j) * 
                                                   factorial(2k)) )


function Nijkα(i, j, k, α)
    l = i + j + k
    if l < 2
        sqrt( 2^(2l+3) * factorial(l+1) * 2^(l+1.5) / (factorial(2l+2) * √π) ) * 
        α^(0.5l + 0.75)
    else
        # for higher angular momentum make the upper bound of norms be 1.
        Nijk(i, j, k) * α^(0.5l + 0.75)
    end
end


normOfGTOin(b::Quiqbox.FGTBasisFuncs1O) = 
Nijkα.(b.l[1]..., [g.xpn() for g in b.gauss])

normOfGTOin(b::Quiqbox.FloatingGTBasisFuncs) = 
Nlα.(b|>Quiqbox.subshellOf, [g.xpn() for g in b.gauss])

isFull(::Any) = false

isFull(::Quiqbox.FloatingGTBasisFuncs{<:Any, <:Any, 0}) = true

isFull(::Quiqbox.FloatingGTBasisFuncs{<:Any, <:Any, 𝑙, <:Any, <:Any, ON}) where {𝑙, ON} = 
(ON == Quiqbox.SubshellXYZsizes[𝑙+1])

function ijkIndex(b::Quiqbox.FloatingGTBasisFuncs{<:Any, D}) where {D}
    isFull(b) && (return :)
    [Quiqbox.AngMomIndexList[D][ijk] for ijk in b.l]
end


function addToDataChain!(env::Vector{Float64}, atm::Vector{Int32}, bas::Vector{Int32}, 
                         bf::Quiqbox.FloatingGTBasisFuncs{<:Any, <:Any, 𝑙, GN}) where {𝑙, GN}
    center = [bf.center[1](), bf.center[2](), bf.center[3]()]
    xpns = Float64[]
    cons = Float64[]
    for i in bf.gauss
        push!(xpns, i.xpn())
        push!(cons, i.con())
    end
    envEndIndex = length(env)
    gAtmIndex = length(atm) / 6 |> Int32

    append!(env, center)
    append!(atm, Int32[0, envEndIndex, 1, 0, 0, 0])
    envEndIndex += 3

    append!(env, xpns)
    norm = bf.normalizeGTO ? normOfGTOin(bf) : 1.0
    append!(env, cons.*norm)
    append!(bas, Int32[gAtmIndex, 𝑙, GN, 1, 0, envEndIndex, envEndIndex+GN, 0])
    envEndIndex += GN*2
    (env, atm, bas)
end


function addToDataChain!(env::Vector{Float64}, atm::Vector{Int32}, 
                         nuclei::Vector{String}, 
                         nucleiCoords::Vector{<:AbstractArray{<:Real}})
    length(nuclei) == length(nucleiCoords) || 
    throw(AssertionError("The length of `nuclei` and `nucleiCoords` should be the same."))
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