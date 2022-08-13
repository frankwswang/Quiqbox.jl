module Molden

export makeMoldenFile

import ..Quiqbox: CanOrbital, MatterByHF, sortPermBasis, mergeBasisFuncs, getAtolDigits, 
                  isaFullShellBasisFuncs, checkFname, AtomicNumberList, centerCoordOf, 
                  groupedSort, joinConcentricBFuncStr, alignNumSign, alignNum, getAtolVal

const spinStrs = ["Alpha", "Beta"]

getOrbitalType(::CanOrbital) = "A"

"""

    makeMoldenFile(mol::MatterByHF{T, 3}; roundDigits::Int=getAtolDigits(T), 
                   recordUMO::Bool=false, fileName::String = "MO") where {T} -> 
    String

Write the information of `MatterByHF` into a newly created **Molden** file. `recordUMO` 
determines whether to include the unoccupied canonical orbitals. `fileName` specifies the 
name of the file, which is also the returned value. If `roundDigits < 0`, there won't be 
rounding for recorded data.
"""
function makeMoldenFile(mol::MatterByHF{T, 3}; 
                        roundDigits::Int=getAtolDigits(T), 
                        recordUMO::Bool=false, fileName::String = "MO") where {T}
    basis = mol.basis.basis |> collect
    roundAtol = roundDigits<0 ? NaN : exp10(-roundDigits)
    ids = sortPermBasis(basis; roundAtol)
    basis = mergeBasisFuncs(basis[ids]...; roundAtol)
    @assert all(basis .|> isaFullShellBasisFuncs) "The basis set stored in the input " * 
                                                  "`MatterByHF` is not supported by " * 
                                                  "the Molden format."
    occuC = getindex.(mol.occuC, Ref(ids), :)
    unocC = getindex.(mol.unocC, Ref(ids), :)
    nucCoords = mol.nucCoord |> collect
    nuc = mol.nuc |> collect
    if recordUMO
        MOgroups = map(mol.occuOrbital, mol.unocOrbital) do osO, osU
            (osO..., osU...)
        end
        Cgroups = map(occuC, unocC) do cO, cU
            hcat(cO, cU)
        end
    else
        MOgroups = mol.occuOrbital
        Cgroups = occuC
    end
    iNucPoint = 0
    strs = joinConcentricBFuncStr.(groupedSort(basis, centerCoordOf))
    strs = split.(strs, "\n", limit=2)
    gCoeffs = getindex.(strs, 2)
    lpadN = 8

    text = """
           [Molden Format]
           made by Quiqbox

           [Atoms] (AU)
           """

    centers = getindex.(strs, 1)
    for cen in centers
        iNucPoint += 1
        coord = parse.(T, split(cen[5:end]))
        if (i = findfirst(x->all(isapprox.(x, coord, atol=getAtolVal(T))), nucCoords); 
            i !== nothing)
            n = popat!(nuc, i)
            atmName = rpad("$(n)", 5)
            atmNumber = rpad("$(AtomicNumberList[n])", 4)
            popat!(nucCoords, i)
        else
            atmName = "X    "
            atmNumber = "X   "
        end
        coordStr = alignNum(coord[1], lpadN; roundDigits) * 
                   alignNum(coord[2], lpadN; roundDigits) * 
                   alignNum(coord[3], lpadN, 0; roundDigits)
        text *= atmName*rpad("$iNucPoint", 5)*atmNumber*coordStr*"\n"
    end
    for (n, coord) in zip(nuc, nucCoords)
        iNucPoint += 1
        text *= rpad("$(n)", 5) * rpad(iNucPoint, 5) * rpad("$(AtomicNumberList[n])", 4)*
                alignNum(coord[1], lpadN; roundDigits) * 
                alignNum(coord[2], lpadN; roundDigits) * 
                alignNum(coord[3], lpadN, 0; roundDigits) * "\n"
    end
    text *= "\n[GTO]"
    for (i, gs) in enumerate(gCoeffs)
        text *= "\n$i 0\n" * gs
    end
    text *= "\n[MO]\n"
    for (spinIdx, MOs) in enumerate(MOgroups)
        for (i, mo) in enumerate(MOs)
            text *= "Sym=   " * getOrbitalType(mo) * "\n"
            moe = mo.energy
            MOcoeffs = Cgroups[spinIdx][:, i]
            if roundDigits > 0
                moe = round(moe, sigdigits=roundDigits)
                MOcoeffs = round.(MOcoeffs, sigdigits=roundDigits)
            end
            text *= "Ene=  "*alignNumSign(moe; roundDigits)*"\n"
            text *= "Spin=  $(spinStrs[spinIdx])\n"
            text *= "Occup= $(sum(mo.occu)[])\n"
            text *= join([rpad("   $j", 6)*alignNum(c, lpadN, 0; roundDigits)*
                        "\n" for (j,c) in enumerate(MOcoeffs)])
        end
    end

    fn = fileName*".molden" |> checkFname
    open(fn, "w") do f
        write(f, text)
    end

    fn
end

end