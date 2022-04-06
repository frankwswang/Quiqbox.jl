using Test
using Quiqbox
using LinearAlgebra: norm

@testset "Molecule.jl tests" begin

nuc1 = ["H", "H"]
nucCoords1 = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]
bs = genBasisFunc.(nucCoords1, ("STO-3G", "H") |> Ref) |> flatten
resRHF = runHF(bs, nuc1, nucCoords1, printInfo=false)
Molecule(bs, nuc1, nucCoords1, resRHF)

# function nnRepulsions
nuc2 = ["H", "H", "O"]
nucCoords2 = [[-0.7,0.0,0.0], [0.6,0.0,0.0], [0.0, 0.0, 0.0]]

@test nnRepulsions(nuc2, nucCoords2) == 1*1/norm(nucCoords2[1] - nucCoords2[2]) + 
                                        1*8/norm(nucCoords2[1] - nucCoords2[3]) + 
                                        1*8/norm(nucCoords2[2] - nucCoords2[3])

end