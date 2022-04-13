export MolOrbital, getMolOrbitals, Molecule, nnRepulsions

using LinearAlgebra: norm

const spinStr = ("α", "β", "α&β")

"""

    MolOrbital{N} <: AbstractMolOrbital

`Struct` of molecular orbitals.

≡≡≡ Field(s) ≡≡≡

`symmetry::String`: The symmetry of the orbital. The default value is "A" for being 
antisymmetric.

`energy::Float64`: Molecular energy.

`spin::String`: Spin function of the orbital. Available values: `"$(spinStr[1])"`, 
`"$(spinStr[2])"`, or `"$(spinStr[3])"` for being double-occupied.

`occupancy::Real`: Occupation number.

`orbitalCoeffs::NTuple{N, Float64}`: coefficients of the basis functions to form the 
molecular orbital.

≡≡≡ Initialization Method(s) ≡≡≡

    MolOrbital(energy::Float64, occupancy::Real, orbitalCoeffs::Vector{Float64}, 
               spin::String="$(spinStr[1])", symmetry::String="A") -> MolOrbital{N}

"""
struct MolOrbital{N} <: AbstractMolOrbital
    symmetry::String
    energy::Float64
    spin::String
    occupancy::Real
    orbitalCoeffs::NTuple{N, Float64}

    function MolOrbital(energy::Float64, occupancy::Real, 
                        orbitalCoeffs::NTuple{N, Float64}, spin::String=spinStr[1], 
                        symmetry::String="A") where {N}
        @assert spin in spinStr "The input spin configuration "*
        "is NOT supported."
        new{N}(symmetry, energy, spin, occupancy, orbitalCoeffs)
    end
end


"""

    getMolOrbitals(ens::Vector{Float64}, occus::Vector{<:Real}, C::Matrix{Float64}, 
                   spins::Vector{String}, 
                   symms::Vector{String}=repeat(["A"], length(occus))) -> 
    Tuple{Vararg{getMolOrbitals}}

A function that returns the molecular orbitals.
"""
function getMolOrbitals(ens::Vector{Float64}, occus::Vector{<:Real}, C::Matrix{Float64}, 
                        spins::Vector{String}, 
                        symms::Vector{String}=repeat(["A"], length(occus)))
    MolOrbital.(ens, occus, [ Tuple(C[:,i]) for i = 1:size(C, 2) ], spins, symms) |> Tuple
end


"""

    Molecule{NN, N, NB} <:MolecularHartreeFockCoefficient{NN, N}

Container for the electronic structure information of a system.

≡≡≡ Field(s) ≡≡≡

`nuc::NTuple{NN, String}`: Nuclei of the system.

`nucCoords::NTuple{NN, NTuple{3,Float64}}`: Nuclei coordinates.

`N::Int`: Total number of electrons.

`orbital::Tuple{Vararg{MolOrbital}}`: Molecular orbitals.

`basis::Tuple{Vararg{FloatingGTBasisFuncs}}`: The basis set for the molecular orbitals.

`Ehf::Float64`: Hartree-Fock energy of the electronic Hamiltonian from the basis set.

`Enn::Float64`: The nuclear repulsion energy.

≡≡≡ Initialization Method(s) ≡≡≡

    Molecule(basis::Vector{<:FloatingGTBasisFuncs}, 
             nuc::NTuple{NN, String}, 
             nucCoords::NTuple{NN, NTuple{3,Float64}}, 
             N::Int, Ehf::Float64, Enn::Float64, 
             Emos::Vector{Float64}, occus::Vector{<:Real}, C::Matrix{Float64}, 
             spins::Vector{String}, 
             symms::Vector{String}=repeat(["A"], length(occus))) -> 
    Molecule{NN, N}

`Emos` are the energies of corresponding molecular energies. `occus` are the occupation 
numbers of the orbitals. `C` is the coefficient matrix, which does not need to be a square 
matrix since the number of rows is the size of the (spatial) basis set whereas the number 
of the columns represents the number of molecular orbitals. `spin` specifies the spin 
functions of the orbitals, entries of which can be set to `"$(spinStr[1])"`, 
`"$(spinStr[2])"`, or `"$(spinStr[3])"` for being double-occupied. `symms` are symmetries 
of the orbitals where the default entry value is "A" for being antisymmetric.

    Molecule(basis::Vector{<:FloatingGTBasisFuncs}, fVars::HFfinalVars) -> Molecule

Construct a `Molecule` from a basis set, and the result from the corresponding 
Hartree-Fock method.
"""
struct Molecule{NN, N, NB} <:MolecularHartreeFockCoefficient{NN, N}
    nuc::NTuple{NN, String}
    nucCoords::NTuple{NN, NTuple{3,Float64}}
    N::Int
    orbital::Tuple{Vararg{MolOrbital}}
    basis::Tuple{Vararg{FloatingGTBasisFuncs}}
    Ehf::Float64
    Enn::Float64

    function Molecule(basis::Vector{<:FloatingGTBasisFuncs}, 
                      nuc::NTuple{NN, String}, 
                      nucCoords::NTuple{NN, NTuple{3,Float64}}, 
                      N::Int, Ehf::Float64, Enn::Float64, 
                      Emos::Vector{Float64}, occus::Vector{<:Real}, C::Matrix{Float64}, 
                      spins::Vector{String}, 
                      symms::Vector{String}=repeat(["A"], length(occus))) where {NN}
        NB = basisSize.(basis) |> sum
        coeff = spins |> unique |> length
        @assert (coeff*NB .== 
                 length.([Emos, occus, C[1,:], spins, symms]) .== 
                 coeff*length(C[:,1])) |> all
        iSorted = mapPermute(basis, sortBasisFuncs)
        basis = basis[iSorted]
        offset = pushfirst!(accumulate(+, collect(basisSize.(basis)) .- 1), 0)
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
        new{NN, N, NB}(nuc, nucCoords, N, getMolOrbitals(Emos, occus, C, spins, symms), 
                       deepcopy(basis) |> Tuple, Ehf, Enn)
    end
end

function Molecule(basis::Vector{<:FloatingGTBasisFuncs}, 
                  fVars::HFfinalVars{<:Any, <:Any, HFTS}) where {HFTS}
    l1 = length(fVars.Emo[1])
    Emos, occus, C, spins = if HFTS == 1
        l2 = l1
        ( fVars.Emo[1], fVars.occu[1], fVars.C[1], fill(spinStr[3], l1) )
    else
        l2 = 2l1
        ( vcat(fVars.Emo[1], fVars.Emo[2]), vcat(fVars.occu[1], fVars.occu[2]), 
          hcat(fVars.C[1], fVars.C[2]), vcat(fill(spinStr[1], l1), fill(spinStr[2], l1)) )
    end
    Molecule(basis, fVars.nuc, fVars.nucCoords, fVars.N, fVars.Ehf, fVars.Enn, Emos, occus, 
             C, spins, fill("A", l2))
end


"""

    nnRepulsions(nuc::Vector{String}, nucCoords::Vector{<:AbstractArray{<:Real}}) -> Float64

Calculate the nuclear repulsion energy given the nuclei and their coordinates.
"""
nnRepulsions(nuc::Vector{String}, nucCoords::Vector{<:AbstractArray{<:Real}}) = 
nnRepulsions(Tuple(nuc), genTupleCoords(nucCoords))

"""

    nnRepulsions(nuc::NTuple{N, String}, 
                 nucCoords::NTuple{N, NTuple{3, Float64}}) where {N} -> 
    Float64

"""
function nnRepulsions(nuc::NTuple{N, String}, 
                      nucCoords::NTuple{N, NTuple{3, Float64}}) where {N}
    E = 0.0
    len = length(nuc)
    for i = 1:len, j=i+1:len
        E += getCharge(nuc[i]) * getCharge(nuc[j]) / norm(nucCoords[i] .- nucCoords[j])
    end
    E
end