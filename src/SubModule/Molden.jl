module Molden

export makeMoldenFile

import ..Quiqbox: Molecule, checkFname, AtomicNumberList, centerCoordOf, flatten, groupedSort, 
                  joinConcentricBFuncStr, alignSignedNum

"""

    makeMoldenFile(mol::Molecule; recordUMO::Bool=false, fileName::String = "MO") -> String

Write the information of input `Molecule` into a **Molden** file. `recordUMO` determines 
whether to include the unoccupied molecular orbitals. `fileName` specifies the name of the 
file, which is also the returned value.
"""
function makeMoldenFile(mol::Molecule; recordUMO::Bool=false, fileName::String = "MO")
    nucCoords = mol.nucCoords |> collect
    nuc = mol.nuc |> collect
    basis = mol.basis |> collect
    MOs = mol.orbital
    iNucPoint = 0
    groups = groupedSort(basis, centerCoordOf)
    strs = joinConcentricBFuncStr.(groups)
    strs = split.(strs, "\n", limit=2)
    gCoeffs = getindex.(strs, 2)

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
            coord = round.(coord, sigdigits=15)
            n = popat!(nuc, i)
            atmName = rpad("$(n)", 5)
            atmCoord = rpad("$(AtomicNumberList[n])", 4) * 
                       rpad(coord[1]|>alignSignedNum, 20) * 
                       rpad(coord[2]|>alignSignedNum, 20) * 
                       rpad(coord[3]|>alignSignedNum, 20)
            popat!(nucCoords, i)
        else
            atmName = "X    "
            atmCoord = cen
        end
        text *= atmName*rpad("$iNucPoint", 5)*atmCoord*"\n"
    end
    for (n, coord) in zip(nuc, nucCoords)
        cv = round.(coord, sigdigits=15)
        iNucPoint += 1
        text *= rpad("$(n)", 5) * rpad(iNucPoint, 5) * rpad("$(AtomicNumberList[n])", 4)*
                rpad(cv[1]|>alignSignedNum, 20) * rpad(cv[2]|>alignSignedNum, 20) * 
                rpad(cv[3]|>alignSignedNum, 20)*"\n"
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
        text *= "Ene=  "*alignSignedNum(MOs[i].energy)*"\n"
        text *= "Spin=  $(MOs[i].spin)\n"
        text *= "Occup= $(MOs[i].occupancy)\n"
        MOcoeffs = MOs[i].orbitalCoeffs
        text *= join([rpad("   $j", 9)*alignSignedNum(c)*
                      "\n" for (j,c) in zip(1:length(MOcoeffs), MOcoeffs)])
    end

    fn = fileName*".molden" |> checkFname
    open(fn, "w") do f
        write(f, text)
    end

    fn
end

end