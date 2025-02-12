module Molden

export makeMoldenFile

import ..Quiqbox: CanOrbital, MatterByHF, sortPermBasis, mergeSubshellsIn, getAtolDigits, 
                  isaFullShellBasisFuncs, checkFname, AtomicNumberList, centerCoordOf, 
                  groupedSort, joinConcentricBFuncStr, alignNumSign, getAtolVal

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
    basis = mergeSubshellsIn(basis[ids]...; roundAtol)
    all(isaFullShellBasisFuncs(b) for b in basis) || 
    throw(AssertionError("The basis set stored in `mol.basis.basis` is not supported "*
          "by the Molden format."))
    all(b.normalize for b in basis) || 
    throw(AssertionError("`.normalize` must be `true` for every `FloatingBasisFuncs` "*
          "inside `mol.basis.basis`."))
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
    conCentricBfs = groupedSort(basis, centerCoordOf)
    nbfs = length.(conCentricBfs)
    strs = split.(joinConcentricBFuncStr.(conCentricBfs; roundDigits), "\n", limit=2)
    gCoeffs = map(getindex.(strs, 2), nbfs) do str, count
        replace(str, "   true"=>""; count)
    end
    lPad = repeat(" ", 8)

    text = """
           [Molden Format]
           made by Quiqbox

           [Atoms] (AU)
           """

    centers = centerCoordOf.(getindex.(conCentricBfs, 1))
    for cen in centers
        iNucPoint += 1
        # coord = parse.(T, split(cen[5:end]))
        if (i = findfirst(x->all(isapprox(xc, cc, atol=getAtolVal(T)) 
                                 for (xc, cc) in zip(x, cen)), nucCoords); 
            i !== nothing)
            n = popat!(nuc, i)
            atmName = rpad("$(n)", 5)
            atmNumber = rpad("$(AtomicNumberList[n])", 4)
            popat!(nucCoords, i)
        else
            atmName = "X    "
            atmNumber = "X   "
        end
        coordStr = lPad * alignNumSign(cen[1]; roundDigits) * 
                   lPad * alignNumSign(cen[2]; roundDigits) * 
                   lPad * alignNumSign(cen[3], 0; roundDigits)
        text *= atmName * rpad("$iNucPoint", 5) * atmNumber * coordStr * "\n"
    end
    for (n, coord) in zip(nuc, nucCoords)
        iNucPoint += 1
        text *= rpad("$(n)", 5) * rpad(iNucPoint, 5) * rpad("$(AtomicNumberList[n])", 4)*
                lPad * alignNumSign(coord[1]; roundDigits) * 
                lPad * alignNumSign(coord[2]; roundDigits) * 
                lPad * alignNumSign(coord[3], 0; roundDigits) * "\n"
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
            text *= "Ene=  "*alignNumSign(moe; roundDigits)*"\n"
            text *= "Spin=  $(spinStrs[spinIdx])\n"
            text *= "Occup= $(sum(mo.occu)[])\n"
            text *= join([rpad("   $j", 8)*alignNumSign(c, 0; roundDigits)*
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