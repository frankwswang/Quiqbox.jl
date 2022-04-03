using Quiqbox
using Quiqbox.Molden

mols = [
        ["H", "H"],
        ["N", "H", "H", "H"]
        ]
molNames = [
            "H2", 
            "NH3"
            ]
br = 0.529177210903
# Data from CCCBDB: https://cccbdb.nist.gov
molCoords = [
             [[0.3705,0.0,0.0], [-0.3705,0.0,0.0]],
             [[0.0, 0.0, 0.1111], [0.0, 0.9316, -0.2592], [0.8068, -0.4658, -0.2592], [-0.8068, -0.4658, -0.2592]]
             ] ./ br

bfCoords = [molCoords..., GridBox(1, 1.2) |> gridCoords]
bfs = ["STO-3G"]
bsNames = push!(("-" .*molNames), "-Grid")
prefix = "Example"
for (nuc, nucCoords, molName, iMol) in zip(mols, molCoords, molNames, 1:length(mols)), 
    (bfCoord, bsName) in zip(bfCoords[iMol:end], bsNames[iMol:end]), 
    bf in bfs

    flag = (bfCoord == nucCoords)
    if flag
        nucConfig = [(bf, i) for i in nuc]
        bs = genBasisFunc.(bfCoord, nucConfig) |> flatten
    else
        bs = genBasisFunc.(bfCoord, bf) |> flatten
        bsName = "-Float"*bsName
    end

    # Number of spin-orbitals must not be smaller than numbers of electrons.
    fVars = try
        runHF(bs, nuc, nucCoords, printInfo=false)
    catch
        continue
    end

    mol = Molecule(bs, nuc, nucCoords, fVars)
    fn = makeMoldenFile(mol; recordUMO=true, fileName=prefix*"_"*molName*"_"*bf*bsName)
end