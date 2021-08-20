module Molden

export makeMoldenFile

import ..Quiqbox: Molecule, checkFname, AtomicNumberList, genBasisFuncText, centerOf, flatten, alignSignedNum

function makeMoldenFile(mol::Molecule; recordUMO::Bool=false, fileName::String = "MO")
    nucCoords = mol.nucCoords
    basis = [mol.basis...]
    MOs = mol.orbital
    iNucPoint = 0
    strs = genBasisFuncText(basis)
    strs = split.(strs, "\n", limit=2)
    gCoeffs = getindex.(strs, 2)

    text = """
           [Molden Format]
           made by Quiqbox

           [Atoms] (AU)
           """

    nucCs = [flatten(nucCoords)...]
    basCs = centerOf.(basis) |> unique |> flatten
    if length(nucCs) != length(basCs) || !(nucCs ≈ basCs)
        centers = getindex.(strs, 1)
        for cen in centers
            iNucPoint += 1
            text *= "X    "*rpad("$iNucPoint", 5)*cen*"\n"
        end
    end
    for (n, coord) in zip(mol.nuc, nucCoords)
        cv = round.(coord, sigdigits=15)
        iNucPoint += 1
        text *= rpad("$(n)", 5) * rpad(iNucPoint, 5) * rpad("$(AtomicNumberList[n])", 4)*
                rpad(cv[1]|>alignSignedNum, 20) * rpad(cv[2]|>alignSignedNum, 20) * rpad(cv[3]|>alignSignedNum, 20)*"\n"
    end
    text *= "\n[GTO]"
    for (i, gs) in zip(1:length(gCoeffs), gCoeffs)
        text *= "\n$i 0\n" * gs
    end
    text *= "\n[MO]\n"
    recordUMO ? (l = length(MOs)) : (l = findfirst(isequal(0), [x.occupancy for x in MOs]) - 1)
    for i = 1:l
        text *= "Sym=   $(MOs[i].symmetry)\n"
        text *= "Ene=  "*alignSignedNum(MOs[i].energy)*"\n"
        text *= "Spin=  $(MOs[i].spin)\n"
        text *= "Occup= $(MOs[i].occupancy)\n"
        MOcoeffs = MOs[i].orbitalCoeffs
        text *= join([rpad("   $j", 9)*alignSignedNum(c)*"\n" for (j,c) in zip(1:length(MOcoeffs), MOcoeffs)])
    end

    fn = fileName*".molden" |> checkFname
    open(fn, "w") do f
        write(f, text)
    end

    fn
end

end