export CanOrbital, getCanOrbitals, changeHbasis, MatterByHF, nnRepulsions

using LinearAlgebra: norm
using Tullio: @tullio
using TensorOperations: @tensor as @TOtensor

"""

    CanOrbital{T, D, NN} <: AbstractSpinOrbital{T, D}

The spatial part (orbital) of a canonical spin-orbital (the set of which diagonalizes the 
Fock matrix of a Hartree-Fock state) with its occupation information. This means the 
maximal occupation number for the mode corresponding to the orbital (namely a canonical 
orbital) equals 2. Please refer to [`getCanOrbitals`](@ref) for the construction of a 
`CanOrbital`.

≡≡≡ Field(s) ≡≡≡

`energy::T`: The eigen energy corresponding to the orbital.

`index::Int`: The index of the orbital within the same spin configuration.

`nuc::NTuple{NN, String}`: The nuclei in the studied system.

`nucCoords::NTuple{NN, NTuple{D, T}}`: The coordinates of corresponding nuclei.

`occu::NTuple{2, Array{Bool, 0}}`: The occupations of two spin configurations.

`orbital::GTBasisFuncs{T, D, 1}`: The spatial orbital part.

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
    OON = fVars.Ns[I]
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

"""

    getCanOrbitals(fVars::HFfinalVars{T, D, <:Any, NN}) where {T, D, NN} -> 
    CanOrbital{T, D, NN}

Generate a set of canonical orbitals from the result of a Hartree-Fock approximation.
"""
getCanOrbitals(fVars::HFfinalVars) = getCanOrbitalsCore(fVars)[1]


"""

    changeHbasis(DbodyInt::AbstractArray{T, D}, C::AbstractMatrix{T}) where {T} -> 
    AbstractArray{T, D}

Change the basis of the input one-body / two-body integrals `DbodyInt` based on the orbital 
coefficient matrix `C`.
"""
changeHbasis(oneBodyInt::AbstractMatrix{T}, C::AbstractMatrix{T}) where {T} = 
@TOtensor ij[i,j] := oneBodyInt[a,b] * C[a,i] * C[b,j]

changeHbasis(twoBodyInt::AbstractArray{T, 4}, C::AbstractMatrix{T}) where {T} = 
@TOtensor ijkl[i,j,k,l] := twoBodyInt[a,b,c,d] * C[a,i] * C[b,j] * C[c,k] * C[d,l]

getJᵅᵝ(twoBodyInt::AbstractArray{T, 4}, (C1, C2)::NTuple{2, AbstractMatrix{T}}) where {T} = 
@tullio iijj[i,j] := twoBodyInt[a,b,c,d] * C1[a,i] * C1[b,i] * C2[c,j] * C2[d,j]

"""

    changeHbasis(twoBodyInt::AbstractArray{T, 4}, 
                 C1::AbstractMatrix{T}, C2::AbstractMatrix{T}) where {T} -> 
    AbstractArray{T, 4}

Change the basis of the input two-body integrals `twoBodyInt` based on two orbital 
coefficient matrices `C1` and `C2` for different spin configurations (e.g., the 
unrestricted case). The output is a 3-element `Tuple` of which the first 2 elements are the 
spatial integrals of each spin configurations respectively, while the last element is the 
Coulomb interactions between orbitals with different spins.
"""
changeHbasis(twoBodyInt::AbstractArray{T, 4}, C::Vararg{AbstractMatrix{T}, 2}) where {T} = 
(changeHbasis.(Ref(twoBodyInt), C)..., getJᵅᵝ(twoBodyInt, C))

"""

    changeHbasis(b::GTBasis{T, D}, 
                 nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{D, T}}, 
                 C::Union{AbstractMatrix{T}, NTuple{2, AbstractMatrix{T}}}) where 
                {T, D, NN} -> 
    NTuple{2, Any}

Return the one-body and two-body integrals after a change of basis based on the input `C`, 
given the basis set information `b`. The type of each element in the returned `Tuple` is 
consistent with the cases where the first argument of `changeHbasis` is an `AbstractArray`.
"""
changeHbasis(b::GTBasis{T, D}, 
             nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{D, T}}, 
             C::AbstractMatrix{T}) where {T, D, NN} = 
changeHbasis.((coreH(b, nuc, nucCoords), b.eeI), Ref(C))

function changeHbasis(b::GTBasis{T, D}, 
                      nuc::NTuple{NN, String}, nucCoords::NTuple{NN, NTuple{D, T}}, 
                      C::Vararg{AbstractMatrix{T}, 2}) where {T, D, NN}
    Hcs = changeHbasis.(Ref(coreH(b, nuc, nucCoords)), C)
    eeIs = changeHbasis(b.eeI, C...)
    Hcs, eeIs
end

"""

    changeHbasis(HFres::HFfinalVars) -> NTuple{2, Any}

Return the one-body and two-body integrals on the basis of the canonical orbitals 
using the result of a Hartree-Fock method `HFres`.
"""
changeHbasis(HFres::HFfinalVars) = 
changeHbasis(HFres.basis, HFres.nuc, HFres.nucCoord, HFres.C...)


"""

    MatterByHF{T, D, NN, N, BN, HFTS} <:MatterData{T, D, N}

Container of the electronic structure information of a quantum system.

≡≡≡ Field(s) ≡≡≡

`Ehf::T`: Hartree-Fock energy of the electronic Hamiltonian.

`nuc::NTuple{NN, String}`: The nuclei in the studied system.

`nucCoord::NTuple{NN, NTuple{D, T}}`: The coordinates of corresponding nuclei.

`Enn::T`: The nuclear repulsion energy.

`Ns::NTuple{2, Int}`: The numbers of two different spins respectively.

`occuOrbital::NTuple{HFTS, Tuple{Vararg{CanOrbital{T, D, NN}}}}`: The occupied canonical 
orbitals.

`unocOrbital::NTuple{HFTS, Tuple{Vararg{CanOrbital{T, D, NN}}}}` The unoccupied canonical 
orbitals.

`occuC::NTuple{HFTS, Matrix{T}}`: Coefficient matrix(s) of occupied canonical orbitals.

`unocC::NTuple{HFTS, Matrix{T}}`: Coefficient matrix(s) of unoccupied canonical orbitals.

`coreHsameSpin::NTuple{HFTS, Matrix{T}}`: Core Hamiltonian(s) (one-body integrals) of the 
canonical orbitals with same spin configuration(s).

`eeIsameSpin::NTuple{HFTS, Array{T, 4}}`: electron-electron interactions (two-body 
integrals) of the canonical orbitals with same spin configuration(s).

`eeIdiffSpin::Matrix{T}`: Coulomb interactions between canonical orbitals with different 
spins.

`basis::GTBasis{T, D, BN}`: The basis set used for the Hartree-Fock approximation.

≡≡≡ Initialization Method(s) ≡≡≡

    MatterByHF(HFres::HFfinalVars{T, D, <:Any, NN, BN, HFTS}) where {T, D, NN, BN, HFTS} -> 
    MatterByHF{T, D, NN, <:Any, BN, HFTS}

Construct a `MatterByHF` from the result of a Hartree-Fock method `HFres`.
"""
struct MatterByHF{T, D, NN, N, BN, HFTS} <:MatterData{T, D, N}
    Ehf::T
    nuc::NTuple{NN, String}
    nucCoord::NTuple{NN, NTuple{D, T}}
    Enn::T
    Ns::NTuple{2, Int}
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
        Ns = fVars.Ns
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
                 nucCoords::$(SpatialCoordType|>typeStrNotUnionAll)) where {NN, D, T} -> 
    T

Return the nuclear repulsion energy given nuclei `nuc` and their coordinates `nucCoords`.
"""
function nnRepulsions(nuc::Union{NTuple{NN, String}, AbstractVector{String}}, 
                      nucCoords::SpatialCoordType{T, D, NN}) where {NN, D, T}
    nuc = arrayToTuple(nuc)
    nucCoords = genTupleCoords(T, nucCoords)
    E = T(0)
    len = length(nuc)
    for i = 1:len, j=i+1:len
        E += getCharge(nuc[i]) * getCharge(nuc[j]) * inv(norm(nucCoords[i] .- nucCoords[j]))
    end
    E
end