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


#! Potentially add a function to generate any possible cartesian angular momentums
const SubshellXYZs::Memory{Memory{NTuple{3, Int}}} = 
      Memory{Memory{NTuple{3, Int}}}([ #> Every XYZs must start with (i, 0, 0)
          Memory{NTuple{3, Int}}([(0,0,0)]),
          Memory{NTuple{3, Int}}([(1,0,0), (0,1,0), (0,0,1)]),
          Memory{NTuple{3, Int}}([(2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)]),
          Memory{NTuple{3, Int}}([(3,0,0), (2,1,0), (2,0,1), (1,2,0), (1,1,1), (1,0,2), 
                                  (0,3,0), (0,2,1), (0,1,2), (0,0,3)]),
          Memory{NTuple{3, Int}}([(4,0,0), (3,1,0), (3,0,1), (2,2,0), (2,1,1), (2,0,2), 
                                  (1,3,0), (1,2,1), (1,1,2), (1,0,3), (0,4,0), (0,3,1), 
                                  (0,2,2), (0,1,3), (0,0,4)]),
          Memory{NTuple{3, Int}}([(5,0,0), (4,1,0), (4,0,1), (3,2,0), (3,1,1), (3,0,2), 
                                  (2,3,0), (2,2,1), (2,1,2), (2,0,3), (1,4,0), (1,3,1), 
                                  (1,2,2), (1,1,3), (1,0,4), (0,5,0), (0,4,1), (0,3,2), 
                                  (0,2,3), (0,1,4), (0,0,5)]),
          Memory{NTuple{3, Int}}([(6,0,0), (5,1,0), (5,0,1), (4,2,0), (4,1,1), (4,0,2), 
                                  (3,3,0), (3,2,1), (3,1,2), (3,0,3), (2,4,0), (2,3,1), 
                                  (2,2,2), (2,1,3), (2,0,4), (1,5,0), (1,4,1), (1,3,2), 
                                  (1,2,3), (1,1,4), (1,0,5), (0,6,0), (0,5,1), (0,4,2), 
                                  (0,3,3), (0,2,4), (0,1,5), (0,0,6)])
      ])


const AngularSubShellDict::Base.ImmutableDict{String, Int} = let
    keyTemp = ["S", "P", "D", "F", "G", "H", "I"]
    valTemp = collect(0 : (length(keyTemp) - 1))
    mapreduce(Base.ImmutableDict, keyTemp, valTemp, 
              init=Base.ImmutableDict{String, Int}()) do key, val
        key=>val
    end
end

const AtomicGTOrbSetDict = Dict(AtomicGTOrbSetNames .=> AtomicGTOrbSetTexts)