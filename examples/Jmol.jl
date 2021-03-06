using Quiqbox
using Quiqbox.Molden

mols = [ ["H", "H"], ["N", "H", "H", "H"] ]
molNames = ["H2", "NH3"]
br = 0.529177210903
# Data from CCCBDB: https://cccbdb.nist.gov
molCoords = [ [[0.3705,0.0,0.0], [-0.3705,0.0,0.0]],
              [[0.0, 0.0, 0.1111], [0.0, 0.9316, -0.2592], 
               [0.8068, -0.4658, -0.2592], [-0.8068, -0.4658, -0.2592]]
            ] ./ br

bfCoords = [molCoords..., GridBox(1, 1.2) |> gridCoordOf]
bfs = ["STO-3G"]
bsNames = push!(("-" .*molNames), "-Grid")
prefix = "Example"
for (nuc, nucCoords, molName, iMol) in zip(mols, molCoords, molNames, 1:length(mols)), 
    (bfCoord, bsName) in zip(bfCoords[iMol:end], bsNames[iMol:end]), 
    bf in bfs

    flag = (bfCoord == nucCoords)
    if flag
        bs = genBasisFunc.(bfCoord, bf, nuc) |> flatten
    else
        bs = genBasisFunc.(bfCoord, bf) |> flatten
        bsName = "-Float"*bsName
    end

    # The number of spin-orbitals must not be smaller than the number of electrons.
    if getCharge(nuc) <= 2sum( orbitalNumOf.(bs) )
        HFres = runHF(bs, nuc, nucCoords, printInfo=false)
    else
        continue
    end

    mol = MatterByHF(HFres)
    fn = makeMoldenFile(mol; recordUMO=true, fileName=prefix*"_"*molName*"_"*bf*bsName)
end