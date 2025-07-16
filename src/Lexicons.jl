const PowersOfPi = let
    keyTemp = (:n0d75, :p0d5, :p1d0, :p1d5, :p2d5)
    valTemp = big(π).^ (-0.75, 0.5, 1.0, 1.5, 2.5)
    mapreduce(Base.ImmutableDict, keyTemp, valTemp, 
              init=Base.ImmutableDict{Symbol, BigFloat}()) do key, val
        key=>val
    end
end

const AtomElementNames = Memory{Symbol}([:H,  :He, :Li, :Be, :B,  :C,  :N,  :O,  :F,  :Ne, 
                                         :Na, :Mg, :Al, :Si, :P,  :S,  :Cl, :Ar, :K,  :Ca])

const NuclearChargeDict = Dict{Symbol, Int}(AtomElementNames .=> 1:length(AtomElementNames))

"""

    getCharge(nuc::Union{Tuple{Vararg{Symbol}}, AbstractVector{Symbol}}) -> Int

Return the total electric charge (in 𝑒) of the input nucleus/nuclei.
"""
getCharge(nuc::Symbol) = NuclearChargeDict[nuc]::Int

function getCharge(nuc::Union{Tuple{Vararg{Symbol}}, AbstractVector{Symbol}})
    mapreduce(getCharge, nuc, init=zero(Int))
end