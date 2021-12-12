module Molden

export makeMoldenFile

import ..Quiqbox: Molecule, checkFname, AtomicNumberList, centerCoordOf, flatten, groupedSort, 
                  joinConcentricBFuncStr, alignNumSign, alignNum

"""

    makeMoldenFile(mol::Molecule; roundDigits::Int=15, 
                   recordUMO::Bool=false, fileName::String = "MO") -> String

Write the information of input `Molecule` into a **Molden** file. `recordUMO` determines 
whether to include the unoccupied molecular orbitals. `fileName` specifies the name of the 
file, which is also the returned value. If `roundDigits < 0`, there won't be rounding for 
recorded data.
"""
function makeMoldenFile(mol::Molecule; 
                        roundDigits::Int=15, recordUMO::Bool=false, fileName::String = "MO")
    nucCoords = mol.nucCoords |> collect
    nuc = mol.nuc |> collect
    basis = mol.basis |> collect
    MOs = mol.orbital
    iNucPoint = 0
    groups = groupedSort(basis, centerCoordOf)
    strs = joinConcentricBFuncStr.(groups)
    strs = split.(strs, "\n", limit=2)
    gCoeffs = getindex.(strs, 2)
    rpadN = roundDigits < 0 ? 21 : (roundDigits+1)
    lpadN = 8

    text = """
           [Molden Format]
           made by Quiqbox

           [Atoms] (AU)
           """

    centers = getindex.(strs, 1)
    for cen in centers
        iNucPoint += 1
        coord = parse.(Float64, split(cen[5:end]))
        if (i = findfirst(x->prod(isapprox.(x, coord, atol=1e-15)), nucCoords); 
            i !== nothing)
            n = popat!(nuc, i)
            atmName = rpad("$(n)", 5)
            atmNumber = rpad("$(AtomicNumberList[n])", 4)
            popat!(nucCoords, i)
        else
            atmName = "X    "
            atmNumber = "X   "
        end
        roundDigits > 0 && (coord = round.(coord, digits=roundDigits))
        coordStr = alignNum(coord[1], lpadN, rpadN; roundDigits) * 
                   alignNum(coord[2], lpadN, rpadN; roundDigits) * 
                   alignNum(coord[3], lpadN, 0; roundDigits)
        text *= atmName*rpad("$iNucPoint", 5)*atmNumber*coordStr*"\n"
    end
    for (n, coord) in zip(nuc, nucCoords)
        roundDigits >= 0 && (cv = round.(coord, digits=roundDigits))
        iNucPoint += 1
        text *= rpad("$(n)", 5) * rpad(iNucPoint, 5) * rpad("$(AtomicNumberList[n])", 4)*
                alignNum(cv[1], lpadN, rpadN; roundDigits) * 
                alignNum(cv[2], lpadN, rpadN; roundDigits) * 
                alignNum(cv[3], lpadN, 0; roundDigits) * "\n"
    end
    text *= "\n[GTO]"
    for (i, gs) in zip(1:length(gCoeffs), gCoeffs)
        text *= "\n$i 0\n" * gs
    end
    text *= "\n[MO]\n"
    recordUMO ? (l = length(MOs)) : (l = findfirst(isequal(0), 
                                                   [x.occupancy for x in MOs]) - 1)
    for i = 1:l
        text *= "Sym=   $(MOs[i].symmetry)\n"
        moe = MOs[i].energy
        MOcoeffs = MOs[i].orbitalCoeffs
        if roundDigits > 0
            moe = round(moe, digits=roundDigits)
            MOcoeffs = round.(MOcoeffs, digits=roundDigits)
        end
        text *= "Ene=  "*alignNumSign(moe; roundDigits)*"\n"
        text *= "Spin=  $(MOs[i].spin)\n"
        text *= "Occup= $(MOs[i].occupancy)\n"
        text *= join([rpad("   $j", 6)*alignNum(c, lpadN, 0; roundDigits)*
                      "\n" for (j,c) in zip(1:length(MOcoeffs), MOcoeffs)])
    end

    fn = fileName*".molden" |> checkFname
    open(fn, "w") do f
        write(f, text)
    end

    fn
end

end