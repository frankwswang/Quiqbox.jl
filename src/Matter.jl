export CanOrbital, changeHbasis, MatterByHF, nnRepulsions

using TensorOperations
using LinearAlgebra: norm

"""

    CanOrbital{T, N} <: AbstractSpinOrbital{T, D}

`Struct` of one of a canonical (spin-)orbital set that diagonalizes the Fock matrix of the 
Hartree-Fock state of a many-body system.

≡≡≡ Field(s) ≡≡≡

`energy::T`: eigen energy corresponding to the orbital.

`occu::NTuple{2, Bool}`: Occupation number. 

≡≡≡ Initialization Method(s) ≡≡≡

    CanOrbital(energy::T, occupancy::Real, orbitalCoeffs::NTuple{N, T}, 
               spin::String="$(spinOccupations[1])", symmetry::String="A") where {N, T<:Real} -> 
    CanOrbital{T, N}

"""
struct CanOrbital{T, D, NN} <: AbstractSpinOrbital{T, D}
    energy::T
    index::Int
    nuc::NTuple{NN, String}
    nucCoord::NTuple{NN, NTuple{D, T}}
    occu::NTuple{2, Array{Bool, 0}}
    func::GTBasisFuncs{T, D, 1}

    CanOrbital(::Val{S}, i::Int, fVars::HFfinalVars{T, D, <:Any, NN}) where {S, T, D, NN} = 
    new{T, D, NN}(fVars.Eo[S][i], i, fVars.nuc, fVars.nucCoord, 
                  fill.(SpinOrbitalOccupation[fVars.occu[S][i]]), 
                  mul.(fVars.C[S][:, i], fVars.basis.basis)|>sumOf)
end



function getCanOrbitalsCore(::Val{I}, 
                            fVars::HFfinalVars{<:Any, <:Any, <:Any, <:Any, BN}) where 
                           {I, BN}
    OON = fVars.spin[I]
    rngO = 1:OON
    rngU = (OON+1):BN
    (
        ( CanOrbital.(Val(I), rngO, Ref(fVars)), CanOrbital.(Val(I), rngU, Ref(fVars)) ), 
        ( fVars.C[I][:, 1:OON], fVars.C[I][:, OON+1:BN] )
    )
end

function getCanOrbitalsCore(fVars::HFfinalVars{<:Any, <:Any, :RHF})
    ((a, b), (c, d)) = getCanOrbitalsCore(Val(1), fVars)
    ((a,), (b,)), ((c,), (d,))
end

function getCanOrbitalsCore(fVars::HFfinalVars{<:Any, <:Any, :UHF})
    ((a, b), (c, d)), ((e, f), (g, h)) = getCanOrbitalsCore.((Val(1), Val(2)), Ref(fVars))
    ((a, e), (b, f)), ((c, g), (d, h))
end

getCanOrbitals(fVars::HFfinalVars) = getCanOrbitalsCore(fVars)[1]

# function getCanOrbitals(fVars::HFfinalVars{<:Any, <:Any, :UHF, <:Any, BN}) where {BN}
#     OONα = fVars.N[1]
#     OONβ = fVars.N[2]
#     rngO = 1:OON
#     rngU = (OON+1):BN
#     (
#         vcat(CanOrbital.(Val(1), 1:OONα, Ref(fVars)), 
#              CanOrbital.(Val(2), 1:OONβ, Ref(fVars))), 

#         vcat(CanOrbital.(Val(1), (OONα+1):BN, Ref(fVars)), 
#              CanOrbital.(Val(2), (OONβ+1):BN, Ref(fVars))) 
#     )
# end

# @inline function getCanOrbitalC(fVars::HFfinalVars{<:Any, <:Any, :RHF, <:Any, BN}) where {BN}
#     OON = fVars.N[1]
#     fVars.C[1][:, 1:OON], fVars.C[1][:, OON+1:BN]
# end

# @inline function getCanOrbitalC(fVars::HFfinalVars{<:Any, <:Any, :UHF, <:Any, BN}) where {BN}
#     OONα = fVars.N[1]
#     OONβ = fVars.N[2]
#     fVars.C[1][:, 1:OON], fVars.C[1][:, OON+1:BN]
# end

# """

#     getCanOrbitals(ens::Vector{T}, occus::Vector{<:Real}, C::Matrix{T}, 
#                    spins::Vector{String}, 
#                    symms::Vector{String}=repeat(["A"], length(occus))) where {T<:Real} -> 
#     Tuple{Vararg{getCanOrbitals{<:Any, T}}}

# A function that returns the canonical orbitals.
# """
# function getCanOrbitals(ens::Vector{T}, occus::Vector{<:Real}, C::Matrix{T}, 
#                         spins::Vector{String}, 
#                         symms::Vector{String}=repeat(["A"], length(occus))) where {T<:Real}
#     CanOrbital.(ens, occus, [ Tuple(C[:,i]) for i = 1:size(C, 2) ], spins, symms) |> Tuple
# end


function changeHbasis(b::GTBasis{T, D, BN}, 
                      nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{D, T}}, 
                      C::Matrix{T}) where {T, D, BN, NN}
    @assert all( size(C) .== BN )
    changeHbasis.((coreH(b, nuc, nucCoords), b.eeI), Ref(C))
end

function changeHbasis(oneBoadyInt::Matrix{T}, C::Matrix{T}) where {T}
    pq = Array{T}(undef, size(oneBoadyInt)...)
    @tensor pq[i,j] = oneBoadyInt[a,b] * C[a,i] * C[b,j]
end

function changeHbasis(twoBoadyInt::Array{T, 4}, C::Matrix{T}) where {T}
    pqrs = Array{T}(undef, size(twoBoadyInt)...)
    @tensor pqrs[i,j,k,l] = twoBoadyInt[a,b,c,d] * C[a,i] * C[b,j] * C[c,k] * C[d,l]
end

function changeHbasis(fVars::HFfinalVars)
    map(fVars.C) do C
        changeHbasis(fVars.basis, fVars.nuc, fVars.nucCoord, C)
    end
end

"""

    MatterByHF{T, D, NN, N, BN} <:MatterData{T, NN, N}

Container for the electronic structure information of a system.

≡≡≡ Field(s) ≡≡≡

`nuc::NTuple{NN, String}`: Nuclei of the system.

`nucCoord::NTuple{NN, NTuple{3,Float64}}`: Nuclei coordinates.

`N::Int`: Total number of electrons.

`orbital::Tuple{Vararg{CanOrbital}}`: canonical orbitals.

`basis::Tuple{Vararg{CompositeGTBasisFuncs{T, D}}`: The basis set for canonical orbitals.

`Ehf::Float64`: Hartree-Fock energy of the electronic Hamiltonian from the basis set.

`Enn::Float64`: The nuclear repulsion energy.

≡≡≡ Initialization Method(s) ≡≡≡

    MatterByHF(basis::Vector{<:FloatingGTBasisFuncs}, 
             nuc::NTuple{NN, String}, 
             nucCoords::NTuple{NN, NTuple{3,Float64}}, 
             N::Int, Ehf::Float64, Enn::Float64, 
             Eos::Vector{Float64}, occus::Vector{<:Real}, C::Matrix{Float64}, 
             spins::Vector{String}, 
             symms::Vector{String}=repeat(["A"], length(occus))) -> 
    MatterByHF{NN, N}

`Eos` are the energies of corresponding orbital energies. `occus` are the occupation 
numbers of the orbitals. `C` is the coefficient matrix, which does not need to be a square 
matrix since the number of rows is the size of the (spatial) basis set whereas the number 
of the columns represents the number of canonical orbitals. `spin` specifies the spin 
functions of the orbitals, entries of which can be set to `"$(spinOccupations[1])"`, 
`"$(spinOccupations[2])"`, or `"$(spinOccupations[3])"` for being double-occupied. `symms` are symmetries 
of the orbitals where the default entry value is "A" for being antisymmetric.

    MatterByHF(basis::Vector{<:FloatingGTBasisFuncs}, fVars::HFfinalVars) -> MatterByHF

Construct a `MatterByHF` from a basis set, and the result from the corresponding 
Hartree-Fock method.
"""
struct MatterByHF{T, D, NN, N, BN, HFTS} <:MatterData{T, D, N}
    Ehf::T
    nuc::NTuple{NN, String}
    nucCoord::NTuple{NN, NTuple{D, T}}
    Enn::T
    spin::NTuple{2, Int}
    occuOrbital::NTuple{HFTS, Tuple{Vararg{CanOrbital{T, D, NN}}}}
    unocOrbital::NTuple{HFTS, Tuple{Vararg{CanOrbital{T, D, NN}}}}
    occuC::NTuple{HFTS, Matrix{T}}
    unocC::NTuple{HFTS, Matrix{T}}
    Hcore::NTuple{HFTS, Matrix{T}}
    eeI::NTuple{HFTS, Array{T, 4}}
    basis::GTBasis{T, D, BN}

    # function MatterByHF(basis::Vector{<:CompositeGTBasisFuncs{T, D}}, 
    #                   nuc::NTuple{NN, String}, 
    #                   nucCoords::NTuple{NN, NTuple{3, T}}, 
    #                   N::Int, Ehf::T, Enn::T, 
    #                   Eos::Vector{T}, occus::Vector{<:Real}, C::Matrix{T}, 
    #                   spins::Vector{String}, 
    #                   symms::Vector{String}=repeat(["A"], length(occus))) where {T, D, NN}
    #     BN = basisSize.(basis) |> sum
    #     coeff = spins |> unique |> length
    #     @assert (coeff*BN .== 
    #              length.([Eos, occus, C[1,:], spins, symms]) .== 
    #              coeff*length(C[:,1])) |> all
    #     iSorted = mapPermute(basis, sortBasisFuncs)
    #     basis = basis[iSorted]
    #     offset = pushfirst!(accumulate(+, collect(basisSize.(basis)) .- 1), 0)
    #     iSortedExtended = Union{Int, Vector{Int}}[]
    #     append!(iSortedExtended, iSorted)
    #     for (i, idx) in zip(iSorted, 1:length(iSorted))
    #         if basis[i] isa BasisFuncs
    #             iSortedExtended[idx] = collect(i : i+length(basis[i])-1) .+ offset[i]
    #         else
    #             iSortedExtended[idx] += offset[i]
    #         end
    #     end
    #     C = C[vcat(iSortedExtended...), :]
    #     new{T, D, NN, N, BN}(nuc, nucCoords, N, 
    #                          getCanOrbitals(Eos, occus, C, spins, symms), 
    #                          deepcopy(basis) |> Tuple, Ehf, Enn)
    # end

    function MatterByHF(fVars::HFfinalVars{T, D, <:Any, NN, BN, HFTS}) where 
                       {T, D, NN, BN, HFTS}
        nuc = fVars.nuc
        nucCoords = fVars.nucCoord
        spin = fVars.spin
        basis = fVars.basis
        (osO, osU), (CO, CU) = getCanOrbitalsCore(fVars)
        ints = changeHbasis.(Ref(basis), Ref(nuc), Ref(nucCoords), fVars.C)
        new{T, D, NN, sum(spin), BN, HFTS}(fVars.Ehf, nuc, nucCoords, fVars.Enn, spin, 
                                           Tuple.(osO), Tuple.(osU), CU, CO, 
                                           getindex.(ints, 1), getindex.(ints, 2), basis)
    end
end

# function MatterByHF(basis::Vector{<:CompositeGTBasisFuncs{T, D}}, 
#                   fVars::HFfinalVars{T, D, <:Any, <:Any, HFTS}) where {T, D, HFTS}
#     l1 = length(fVars.Eo[1])
#     Eos, occus, C, spins = if HFTS == 1
#         l2 = l1
#         ( fVars.Eo[1], fVars.occu[1], fVars.C[1], fill(spinOccupations[3], l1) )
#     else
#         l2 = 2l1
#         ( vcat(fVars.Eo[1], fVars.Eo[2]), vcat(fVars.occu[1], fVars.occu[2]), 
#           hcat(fVars.C[1], fVars.C[2]), vcat(fill(spinOccupations[1], l1), fill(spinOccupations[2], l1)) )
#     end
#     MatterByHF(basis, fVars.nuc, fVars.nucCoord, fVars.N, fVars.Ehf, fVars.Enn, Eos, occus, 
#              C, spins, fill("A", l2))
# end


"""

    nnRepulsions(nuc::Union{NTuple{NN, String}, Vector{String}}, 
                 nucCoords::Union{NTuple{NN, NTuple{3, Float64}}, 
                                  Vector{<:AbstractArray{<:Real}}}) where {NN} -> 
    Float64

Calculate the nuclear repulsion energy given the nuclei and their coordinates.
"""
function nnRepulsions(nuc::Union{NTuple{NN, String}, Vector{String}}, 
                      nucCoords::Union{NTuple{NN, NTuple{3, Float64}}, 
                                       Vector{<:AbstractArray{<:Real}}}) where {NN}
    nuc = arrayToTuple(nuc)
    nucCoords = genTupleCoords(nucCoords)
    E = 0.0
    len = length(nuc)
    for i = 1:len, j=i+1:len
        E += getCharge(nuc[i]) * getCharge(nuc[j]) / norm(nucCoords[i] .- nucCoords[j])
    end
    E
end