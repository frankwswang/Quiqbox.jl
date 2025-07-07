const PowersOfPi = let
    keyTemp = (:n0d75, :p0d5, :p1d0, :p1d5, :p2d5)
    valTemp = big(π).^ (-0.75, 0.5, 1.0, 1.5, 2.5)
    mapreduce(Base.ImmutableDict, keyTemp, valTemp, 
              init=Base.ImmutableDict{Symbol, BigFloat}()) do key, val
        key=>val
    end
end