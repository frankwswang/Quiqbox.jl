export CanOrbital, changeHbasis, MatterByHF, nnRepulsions

using LinearAlgebra: norm
using Tullio: @tullio

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
    orbital::GTBasisFuncs{T, D, 1}

    CanOrbital(::Val{S}, i::Int, fVars::HFfinalVars{T, D, <:Any, NN}) where {S, T, D, NN} = 
    new{T, D, NN}(fVars.Eo[S][i], i, fVars.nuc, fVars.nucCoord, 
                  fill.(SpinOrbitalOccupation[fVars.occu[S][i]]), 
                  mul.(fVars.C[S][:, i], fVars.basis.basis)|>sumOf)
end


function getCanOrbitalsCore(::Val{I}, 
                            fVars::HFfinalVars{<:Any, <:Any, <:Any, <:Any, BN}) where 
                           {I, BN}
    OON = fVars.N[I]
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


"""

    changeHbasis(NBodyInt::Matrix{T}, C::Matrix{T}) where {T} -> Matrix{T}

    changeHbasis(NBodyInt::Array{T, D}, C::Matrix{T}) where {T} -> Array{T, 4}

Change the basis of the input one-body / two-body integrals based on the orbital 
coefficient matrix. 
"""
function changeHbasis(oneBoadyInt::Matrix{T}, C::Matrix{T}) where {T}
    ij = Array{T}(undef, size(C, 2), size(C, 2))
    @tullio ij[i,j] = oneBoadyInt[a,b] * C[a,i] * C[b,j]
end

function changeHbasis(twoBoadyInt::Array{T, 4}, C::Matrix{T}) where {T}
    ijkl = Array{T}(undef, size(C, 2), size(C, 2), size(C, 2), size(C, 2))
    @tullio ijkl[i,j,k,l] = twoBoadyInt[a,b,c,d] * C[a,i] * C[b,j] * C[c,k] * C[d,l]
end

function getJᵅᵝ(twoBoadyInt::Array{T, 4}, (C1, C2)::NTuple{2, Matrix{T}}) where {T}
    iijj = Array{T}(undef, size(C1, 2), size(C2, 2))
    @tullio iijj[i,j] = twoBoadyInt[a,b,c,d] * C1[a,i] * C1[b,i] * C2[c,j] * C2[d,j]
end

"""

    changeHbasis(twoBoadyInt::Array{T, 4}, C1::Matrix{T}, C2::Matrix{T}) where {T} -> 
    Array{T, 4}

Change the basis of the input two-body integrals based on 2 orbital coefficient 
matrices for different spin configurations (i.e., the unrestricted case). The output is a 
3-element `Tuple` of which the first 2 elements are the spatial integrals of each spin 
configurations respectively, while the last element is the coulomb interaction between 
orbitals with different spins.
"""
changeHbasis(twoBoadyInt::Array{T, 4}, C::Vararg{Matrix{T}, 2}) where {T} = 
(changeHbasis.(Ref(twoBoadyInt), C)..., getJᵅᵝ(twoBoadyInt, C))

"""

    changeHbasis(b::GTBasis{T, D}, 
                 nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{D, T}}, 
                 C::Union{Matrix{T}, NTuple{2, Matrix{T}}}) where 
                {T, D, NN} -> 
    NTuple{2, Any}

Return the one-body and two-body integrals after a change of basis based on the input `C`. 
The type of each element in the returned `Tuple` is consistent with the case where the 
first argument of `changeHbasis` is an `Array`.
"""
changeHbasis(b::GTBasis{T, D}, 
             nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{D, T}}, 
             C::Matrix{T}) where {T, D, NN} = 
changeHbasis.((coreH(b, nuc, nucCoords), b.eeI), Ref(C))

function changeHbasis(b::GTBasis{T, D}, 
                      nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{D, T}}, 
                      C::Vararg{Matrix{T}, 2}) where {T, D, NN}
    Hcs = changeHbasis.(Ref(coreH(b, nuc, nucCoords)), C)
    eeIs = changeHbasis(b.eeI, C...)
    Hcs, eeIs
end

"""

    changeHbasis(fVars::HFfinalVars) -> NTuple{2, Any}

"""
changeHbasis(fVars::HFfinalVars) = 
changeHbasis(fVars.basis, fVars.nuc, fVars.nucCoord, fVars.C...)


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
struct MatterByHF{T, D, NN, 𝑁, BN, HFTS} <:MatterData{T, D, 𝑁}
    Ehf::T
    nuc::NTuple{NN, String}
    nucCoord::NTuple{NN, NTuple{D, T}}
    Enn::T
    N::NTuple{2, Int}
    occuOrbital::NTuple{HFTS, Tuple{Vararg{CanOrbital{T, D, NN}}}}
    unocOrbital::NTuple{HFTS, Tuple{Vararg{CanOrbital{T, D, NN}}}}
    occuC::NTuple{HFTS, Matrix{T}}
    unocC::NTuple{HFTS, Matrix{T}}
    coreHsameSpin::NTuple{HFTS, Matrix{T}}
    eeIsameSpin::NTuple{HFTS, Array{T, 4}}
    eeIdiffSpin::Matrix{T}
    basis::GTBasis{T, D, BN}

    function MatterByHF(fVars::HFfinalVars{T, D, <:Any, NN, BN, HFTS}) where 
                       {T, D, NN, BN, HFTS}
        nuc = fVars.nuc
        nucCoords = fVars.nucCoord
        Ns = fVars.N
        basis = fVars.basis
        (osO, osU), (CO, CU) = getCanOrbitalsCore(fVars)
        ints = changeHbasis(fVars)
        if HFTS == 1
            cH = (ints[1],)
            eeI = ints[2]
            Jᵅᵝ = Array{T}(undef, BN, BN)
            for i=1:BN, j=i:BN
               Jᵅᵝ[i,j] = Jᵅᵝ[j,i] = eeI[i,i,j,j]
            end
            eeI = (eeI,)
        elseif HFTS == 2
            cH = ints[1]
            eeI = ints[2][1:2]
            Jᵅᵝ = ints[2][3]
        else
            error("The input data format is not supported: HFTS = $(HFTS).")
        end
        new{T, D, NN, sum(Ns), BN, HFTS}(fVars.Ehf, nuc, nucCoords, fVars.Enn, Ns, 
                                           Tuple.(osO), Tuple.(osU), CO, CU, 
                                           cH, eeI, Jᵅᵝ, basis)
    end
end


"""

    nnRepulsions(nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                 nucCoords::Union{NTuple{NN, NTuple{3, Float64}}, 
                                  AbstractVector{<:AbstractVector{<:Real}}}) where {NN} -> 
    Float64

Calculate the nuclear repulsion energy given the nuclei and their coordinates.
"""
function nnRepulsions(nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                      nucCoords::Union{NTuple{NN, NTuple{D, T}}, 
                                       AbstractVector{<:AbstractVector{<:T}}}) where 
                     {NN, D, T}
    nuc = arrayToTuple(nuc)
    nucCoords = genTupleCoords(T, nucCoords)
    E = T(0)
    len = length(nuc)
    for i = 1:len, j=i+1:len
        E += getCharge(nuc[i]) * getCharge(nuc[j]) / norm(nucCoords[i] .- nucCoords[j])
    end
    E
end