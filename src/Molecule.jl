export MolOrbital, getMolOrbitals, Molecule

struct MolOrbital{N} <: AbstractMolOrbital
    symmetry::String
    energy::Float64
    spin::String
    occupancy::Real
    orbitalCoeffs::NTuple{N, Float64}

    function MolOrbital(energy::Float64, occupancy::Real, orbitalCoeffs::Array{Float64, 1}, spin::String="Alpha", symmetry::String="A")
        spin != "Alpha" && spin != "Beta" && error("Keyword arguement \"spin\" can only be \"Alpha\" or \"Beta\"")
        new{length(orbitalCoeffs)}(symmetry, energy, spin, occupancy, orbitalCoeffs |> Tuple)
    end
end


function getMolOrbitals(ens::Array{Float64, 1}, 
                        occus::Array{<:Real, 1}, 
                        C::Array{Float64, 2}, 
                        spins::Array{String, 1}=repeat(["Alpha"], length(occus)), 
                        symms::Array{String, 1}=repeat(["A"], length(occus)))
    MolOrbital.(ens, occus, [ C[:,i] for i = 1:size(C, 2) ], spins, symms) |> Tuple
end


struct Molecule{Nc, Ne, Nb} <:MolecularHartreeFockCoefficient{Nc, Ne}
    nuc::Tuple{Vararg{String}}
    nucCoords::Tuple{Vararg{NTuple{3, Real}}}
    Ne::Int
    orbital::Tuple{Vararg{MolOrbital}}
    basis::Tuple{Vararg{FloatingGTBasisFunc}}

    function Molecule(basis, nuc, nucCoords, Ne, Emos, occus, C, spins, symms)
        @assert length(nuc) == length(nucCoords)
        Nb = basisSize(basis) |> sum
        coeff = spins |> unique |> length
        @assert (coeff*Nb .== length.([Emos, occus, C[1,:], spins, symms]) .== coeff*length(C[:,1])) |> prod
        new{getCharge(nuc), Ne, Nb}(nuc |> Tuple, nucCoords .|> Tuple |> Tuple, Ne, 
                                    getMolOrbitals(Emos, occus, C, spins, symms), deepcopy(basis) |> Tuple)
    end
end


function Molecule(basis, nuc, nucCoords, HFfVars)
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
    Molecule(basis, nuc, nucCoords, t.parameters[2], Emos|>collect, 
             occus|>collect, C, spins, fill("A", length(spins)))
end