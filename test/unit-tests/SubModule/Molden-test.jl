using Test
using Quiqbox
using Quiqbox.Molden

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
             [[0.0, 0.0, 0.1111], [0.0, 0.9316, -0.2592], 
              [0.8068, -0.4658, -0.2592], [-0.8068, -0.4658, -0.2592]]
             ] ./ br

bfCoords = [molCoords..., GridBox(1, 1.2) |> gridCoords]
bfs = ["STO-3G", "STO-2G"]
bsNames = push!(("-" .*molNames), "-Grid")
dir = @__DIR__
prefix1 = dir*"/"
prefix2 = dir*"/Moldens/"
for ((iNuc, nuc), nucCoords, molName) in zip(enumerate(mols), molCoords, molNames), 
    (bfCoord, bsName) in zip(bfCoords[iNuc:end], bsNames[iNuc:end]), 
    HFtype in Quiqbox.HFtypes,
    bf in bfs

    flag = (bfCoord == nucCoords)
    if flag
        bs = genBasisFunc.(bfCoord, bf, nuc) |> flatten
    else
        bs = genBasisFunc.(bfCoord, bf) |> flatten
        bsName = "-Float"*bsName
    end

    # Number of spin-orbitals must not be smaller than numbers of electrons.
    if getCharge(nuc) <= sum( orbitalNumOf.(bs) )
        fVars = runHF(bs, nuc, nucCoords, HFconfig((HF=HFtype,)), printInfo=false)
    else
        continue
    end

    mol = MatterByHF(fVars)
    fn = "Test_"*molName*"_"*bf*bsName*"_"*string(HFtype)
    fd = makeMoldenFile(mol; roundDigits=5, recordUMO=true, fileName=prefix1*fn)
    str1, str2 = replace.(read.((fd, prefix2*fn*".molden"), String), 
                          r"[0-9]+\.[0-9]{5}(?![0-9])"=>"X.XXXXX")
    str1, str2 = replace.((str1, str2), "-X.XXXXX"=>" X.XXXXX")
    str1, str2 = replace.((str1, str2), "\r\n"=>"\n")
    @test str1 == str2
    rm(fd)
end

bf1 = genBasisFunc(fill(0.0, 3), (0.3, 0.5))
bs1 = (genBasisFunc.(molCoords[1], "STO-3G") |> flatten) .+ bf1
mol1 = runHF(bs1, mols[1], molCoords[1], printInfo=false) |> MatterByHF
@test try makeMoldenFile(mol1) catch; true end

bf2 = genBasisFunc(fill(0.0, 3), "STO-3G", "O")[3][[1,3]]
bs2 = [bf1, bf2]
mol2 = runHF(bs2, mols[1], molCoords[1], printInfo=false) |> MatterByHF
@test try makeMoldenFile(mol2) catch; true end
end