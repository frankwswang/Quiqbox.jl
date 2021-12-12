using Test
using Quiqbox
using LinearAlgebra: norm

# function nnRepulsions

nuc = ["H", "H", "O"]
coords = [[-0.7,0.0,0.0], [0.6,0.0,0.0], [0.0, 0.0, 0.0]]

@test nnRepulsions(nuc, coords) == 1*1/norm(coords[1] - coords[2]) + 
                                   1*8/norm(coords[1] - coords[3]) + 
                                   1*8/norm(coords[2] - coords[3])