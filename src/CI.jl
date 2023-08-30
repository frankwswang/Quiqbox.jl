struct RSD{𝑚ₛ, ON} <:RCI{𝑚ₛ, ON}
    N::Int
    spinUpOccu::NTuple{ON, Bool}
    spinDnOccu::NTuple{ON, Bool}
    ID::UInt

    function RSD(spinUpOccu::NTuple{ON, Bool}, spinDnOccu::NTuple{ON, Bool}, 
                 basis::NTuple{ON, <:GTBasisFuncs{T, D, 1}})
        Nup = sum(spinUpOccu)
        Ndn = sum(spinDnOccu)
        new{Nup-Ndn, ON}(Nup+Ndn, spinUpOccu, spinDnOccu, objectid(basis))
    end
end

struct CSF{𝑠, 𝑚ₛ, ON} <: RCI{𝑚ₛ, ON}
    N::Int
    spatialOccu::NTuple{ON, Int}
    ID::UInt

    function RSD(spatialOccu::NTuple{ON, Bool}, basis::NTuple{ON, <:GTBasisFuncs{T, D, 1}})
        
        for 
        new{Nup-Ndn, ON}(Nup+Ndn, spinUpOccu, spinDnOccu, objectid(basis))
    end
end




function showbitStr(num, nDigits=4)
    str = num|>bitstring
    str[end-nDigits+1:end]
end

function computeSpinPerm(nUp, nDn) # potential improvement: only iterate over first half.
    u = (1 << nUp) - 1
    siz = binomial(nUp+nDn, nUp)
    v = Array{Int}(undef, siz)
    v[begin] = u
    for i in 2:siz
        t = u | (u - 1)
        t1 = t + 1
        t2 = ((~t & t1) - 1) >> (trailing_zeros(u) + 1)
        u = t1 | t2
        # @show u
        # @show showbitStr.([t, t1, t2], nUp+nDn)
        v[i] = u
    end
    # @show showbitStr.(v, nUp+nDn)
    v
end

## Input: [1, 0, 1, 2]
function genSpinStrings(spatialOrbConfig::Vector{Int})
    ids1 = findall(isone, spatialOrbConfig)
    ids2 = findall(isequal(2), spatialOrbConfig)
    nSingleOccu = length(ids1)
    # nDoubleOccu = sum(ids2)
    nUp = nSingleOccu ÷ 2
    nDn = nSingleOccu - nUp
    permStrs = showbitStr.(computeSpinPerm(nUp, nDn), nSingleOccu)
    spinUpConfigs = collect(zero(spatialOrbConfig) for _ in eachindex(permStrs))
    spinDnConfigs = deepcopy(spinUpConfigs)
    for (perm, SUconfig, SDconfig) in zip(permStrs, spinUpConfigs, spinDnConfigs)
        SUconfig[ids2] .= 1
        SDconfig[ids2] .= 1
        for (spin, idx1) in zip(perm, ids1)
            config = spin=='1' ? SUconfig : SDconfig
            config[idx1] = 1
        end
    end
    spinUpConfigs, spinDnConfigs
end



# RHF state
function formatSpatialOrbConfig((RHFSconfig,)::Tuple{NTuple{ON, String}})
    res = (Int[], Int[], Int[]) # [DoubleOccuIds, SingleOccuIds, UnOccuIds]
    for (i, e) in enumerate(RHFSconfig)
        if e == spinOccupations[end]
            push!(res[1], i)
        elseif e == spinOccupations[begin]
            push!(res[3], i)
        else
            push!(res[2], i)
        end
    end
    res
end

function genSpatialOrbConfigCore(n::Int, refConfig::NTuple{3, Vector{Int}})
    N = 2*length(refConfig[begin]) + length(refConfig[begin+1])
    for i in min(n, N)

    end
end

function cc()
    
end

function promoteElec!(spoConifg, aᵢ, cⱼ)
    spoConifg[aᵢ] -= 1
    spoConifg[cⱼ] += 1
    spoConifg
end

[2,2,2,0,0,0]




function restrictedSDconfig()
    getEhf()
    sum(diag(Hc)[1:n]) + 0.5*sum([(eeI[i,i,j,j] - eeI[i,j,j,i]) for j in 1:n, i in 1:n])
end

function genFCI()

end

function genXCI()

end

function genSCI()

end