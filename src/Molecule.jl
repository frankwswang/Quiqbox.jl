export MolOrbital, getMolOrbitals, Molecule, nnRepulsions

using LinearAlgebra: norm

"""

    MolOrbital{N} <: AbstractMolOrbital

`Struct` of molecular orbitals.

≡≡≡ Field(s) ≡≡≡

`symmetry::String`: The symmetry of the orbital. The default value is "A" for being 
antisymmetric.

`energy::Float64`: Molecular energy.

`spin::String`: Spin function of the orbital. Available values: "Alpha", "Beta".

`occupancy::Real`: Occupation number.

`orbitalCoeffs::NTuple{N, Float64}`: coefficients of the basis functions to form the 
molecular orbital.

≡≡≡ Initialization Method(s) ≡≡≡

    MolOrbital(energy::Float64, occupancy::Real, orbitalCoeffs::Array{Float64, 1}, 
               spin::String="Alpha", symmetry::String="A") -> MolOrbital{N}

"""
struct MolOrbital{N} <: AbstractMolOrbital
    symmetry::String
    energy::Float64
    spin::String
    occupancy::Real
    orbitalCoeffs::NTuple{N, Float64}

    function MolOrbital(energy::Float64, occupancy::Real, orbitalCoeffs::Vector{Float64}, 
                        spin::String="Alpha", symmetry::String="A")
        spin != "Alpha" && spin != "Beta" && error("Keyword argument \"spin\" can only"*
                                                   " be \"Alpha\" or \"Beta\"")
        new{length(orbitalCoeffs)}(symmetry, energy, spin, occupancy, orbitalCoeffs|>Tuple)
    end
end


"""

    getMolOrbitals(ens::Array{Float64, 1}, occus::Array{<:Real, 1}, C::Matrix{Float64}, 
                   spins::Array{String, 1}, 
                   symms::Array{String, 1}=repeat(["A"], length(occus))) -> 
    Tuple{Vararg{getMolOrbitals}}

A function that returns the molecular orbitals.
"""
function getMolOrbitals(ens::Vector{Float64}, occus::Vector{<:Real}, C::Matrix{Float64}, 
                        spins::Vector{String}, 
                        symms::Vector{String}=repeat(["A"], length(occus)))
    MolOrbital.(ens, occus, [ C[:,i] for i = 1:size(C, 2) ], spins, symms) |> Tuple
end


"""

    Molecule{Nc, Ne, Nb} <:MolecularHartreeFockCoefficient{Nc, Ne}

Container for the information of a molecule.

≡≡≡ Field(s) ≡≡≡

`nuc::Tuple{Vararg{String}}`: Nuclei of the molecule.

`nucCoords::Tuple{Vararg{NTuple{3, Real}}}`: The coordinates of the nuclei.

`Ne::Int`: The number of electrons.

`orbital::Tuple{Vararg{MolOrbital}}`: Molecular orbitals.

`basis::Tuple{Vararg{FloatingGTBasisFuncs}}`: The basis set for the molecular orbitals.

`E0HF::Float64`: Hartree-Fock energy of the electronic Hamiltonian from the basis set.

`EnnR::Float64`: The nuclear-nuclear repulsion energy.

≡≡≡ Initialization Method(s) ≡≡≡

    Molecule(basis::Array{FloatingGTBasisFuncs, 1}, nuc::Array{String, 1}, 
             nucCoords::Array{<:AbstractArray, 1}, Ne::Int, E0HF::Float64, 
             Emos::Array{Float64, 1}, occus::Array{<:Real, 1}, C::Array{Float64, 2}, 
             spins::Array{String, 1}, 
             symms::Array{String, 1}=repeat(["A"], length(occus))) -> 
    Molecule{<:Any, Ne, <:Any}

`Emos` are the energies of corresponding molecular energies. `occus` are the occupation 
numbers of the orbitals. `C` is the coefficient matrix, which does not need to be a square 
matrix since the number of rows is the size of the (spatial) basis set whereas the number 
of the columns represents the number of molecular orbitals. `spin` specifies the spin 
functions of the orbitals, entries of which can be set to "Alpha" or "Beta". `symms` are 
symmetries of the orbitals where the default entry value is "A" for being antisymmetric.

    Molecule(basis::Array{<:FloatingGTBasisFuncs, 1}, nuc::Array{String, 1}, 
             nucCoords::Array{<:AbstractArray, 1}, HFfVars::HFfinalVars) -> 
    Molecule

Construct a `Molecule` from a basis set, nuclei information, and the result from the 
corresponding Hartree-Fock SCF procedure, specifically a `HFfinalVars` `struct`.
"""
struct Molecule{Nc, Ne, Nb} <:MolecularHartreeFockCoefficient{Nc, Ne}
    nuc::Tuple{Vararg{String}}
    nucCoords::Tuple{Vararg{NTuple{3, Real}}}
    Ne::Int
    orbital::Tuple{Vararg{MolOrbital}}
    basis::Tuple{Vararg{FloatingGTBasisFuncs}}
    E0HF::Float64
    EnnR::Float64

    function Molecule(basis::Vector{<:FloatingGTBasisFuncs}, nuc::Vector{String}, 
                      nucCoords::Vector{<:AbstractArray}, Ne::Int, E0HF::Float64, 
                      Emos::Vector{Float64}, occus::Vector{<:Real}, C::Matrix{Float64}, 
                      spins::Vector{String}, 
                      symms::Vector{String}=repeat(["A"], length(occus)))
        @assert length(nuc) == length(nucCoords)
        Nb = basisSize(basis) |> sum
        coeff = spins |> unique |> length
        @assert (coeff*Nb .== 
                 length.([Emos, occus, C[1,:], spins, symms]) .== 
                 coeff*length(C[:,1])) |> all
        EnnR = nnRepulsions(nuc, nucCoords)
        iSorted = mapPermute(basis, sortBasisFuncs)
        basis = basis[iSorted]
        offset = pushfirst!(accumulate(+, collect(basisSize(basis)) .- 1), 0)
        iSortedExtended = Union{Int, Vector{Int}}[]
        append!(iSortedExtended, iSorted)
        for (i, idx) in zip(iSorted, 1:length(iSorted))
            if basis[i] isa BasisFuncs
                iSortedExtended[idx] = collect(i : i+length(basis[i])-1) .+ offset[i]
            else
                iSortedExtended[idx] += offset[i]
            end
        end
        C = C[vcat(iSortedExtended...), :]
        new{getCharge(nuc), Ne, Nb}(nuc |> Tuple, 
                                    nucCoords .|> Tuple |> Tuple, 
                                    Ne, 
                                    getMolOrbitals(Emos, occus, C, spins, symms), 
                                    deepcopy(basis) |> Tuple,
                                    E0HF, 
                                    EnnR)
    end
end

function Molecule(basis::Vector{<:FloatingGTBasisFuncs}, nuc::Vector{String}, 
                  nucCoords::Vector{<:AbstractArray}, HFfVars::HFfinalVars)
    t = typeof(HFfVars)
    len = t.parameters[3]
    if t <: HFfinalVars{:UHF}
        Emos = HFfVars.Emo |> flatten
        occus = HFfVars.occu |> flatten
        C = hcat(HFfVars.C...)
        spins = vcat(fill("Alpha", len), fill("Beta", len))
    else
        Emos = HFfVars.Emo
        occus = HFfVars.occu
        C = HFfVars.C
        spins = fill("Alpha", len)
    end
    Molecule(basis, nuc, nucCoords, t.parameters[2], HFfVars.E0HF, Emos|>collect, 
             occus|>collect, C, spins, fill("A", length(spins)))
end


"""

    nnRepulsions(nuc::Array{String, 1}, nucCoords::Array{<:AbstractArray, 1}) -> Float64

Calculate the nuclear-nuclear repulsion energy given the nuclei and their coordinates of a 
molecule.
"""
function nnRepulsions(nuc::Vector{String}, nucCoords::Vector{<:AbstractArray})
    E = 0.0
    len = length(nuc)
    for i = 1:len, j=i+1:len
        E += getCharge(nuc[i]) * getCharge(nuc[j]) / norm(nucCoords[i] - nucCoords[j])
    end
    E
end