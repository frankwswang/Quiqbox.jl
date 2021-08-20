push!(LOAD_PATH, "./Quiqbox")
using Quiqbox
using Quiqbox.Molden
using Test

@testset "Molden.jl test" begin


mols = [
        ["H", "H"],
        ["H", "F"],
        ["N", "H", "H", "H"]
        ]
molNames = [
            "H2", 
            "HF", 
            "NH3"
            ]
br = 0.529177210903
# Data from CCCBDB: https://cccbdb.nist.gov
molCoords = [
             [[0.3705,0.0,0.0], [-0.3705,0.0,0.0]],
             [[0.4585,0.0,0.0], [-0.4585,0.0,0.0]], 
             [[0.0, 0.0, 0.1111], [0.0, 0.9316, -0.2592], [0.8068, -0.4658, -0.2592], [-0.8068, -0.4658, -0.2592]]
             ] ./ br

bfCoords = [molCoords..., GridBox(1, 2.4).coord]
bfs = ["STO-3G", "STO-2G"]
bsNames = push!(("-" .*molNames), "-Grid")
HFtypes = [:RHF, :UHF]
prefix = "Exmaple"
for (nuc, nucCoords, molName, iMol) in zip(mols, molCoords, molNames, 1:length(mols)), 
    (bfCoord, bsName) in zip(bfCoords[iMol:end], bsNames[iMol:end]), 
    HFtype in HFtypes,
    bf in bfs
    
    flag = (bfCoord == nucCoords)
    if flag
        nucConfig = [(bf, i) for i in nuc]
        bs = BasisFunc.(bfCoord, nucConfig) |> flatten
    else
        bs = BasisFunc.(bfCoord, bf) |> flatten
        bsName = "-Float"*bsName
    end
    
    # Number of spin-orbitals must not be smaller than numbers of electrons.
    fVars = try runHF(bs, nuc, nucCoords; HFtype, printInfo=false) catch; continue end

    mol = Molecule(bs, nuc, nucCoords, fVars)
    fn = makeMoldenFile(mol; recordUMO=true, fileName=prefix*"_"*molName*"_"*bf*bsName*"_"*string(HFtype))
    rm(fn)
end


end