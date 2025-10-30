export NuclearCluster, getCharge

struct NuclearCluster{T<:Real, D} <: QueryBox{Pair{ Symbol, NTuple{D, T} }}
    layout::MemoryPair{Symbol, NTuple{D, T}}

    function NuclearCluster(layout::MemoryPair{Symbol, NonEmptyTuple{T, D}}) where 
                           {D, T<:Real}
        checkEmptiness(layout.left, :layout)
        new{T, D+1}(layout)
    end
end

const AbstractRealCoordVector{T<:Real} = Union{
    AbstractVector{<:AbstractVector{T}}, (AbstractVector{NonEmptyTuple{T, D}} where {D})
}

function NuclearCluster(nucSyms::AbstractVector{Symbol}, 
                        nucCoords::AbstractRealCoordVector{T}, 
                        pairwiseSort::Bool=true) where {T<:Real}
    len = checkEmptiness(nucSyms, :nucSyms)
    if len != length(nucCoords)
        throw(AssertionError("`nucSyms` and `nucCoords` should have the same length."))
    end
    dim = (length∘first)(nucCoords)
    nucSymsLocal = Memory{Symbol}(undef, len)
    nucCoordsLocal = Memory{NTuple{dim, T}}(undef, len)
    sortFunc = function (i::Int)
        idx = OneToIndex(i)
        (NuclearChargeDict[getEntry(nucSyms, idx)], getEntry(nucCoords, idx))
    end

    sourceIdxSeq = pairwiseSort ? sort(1:len, by=sortFunc) : OneToRange(len)
    for (i, j, k) in zip(eachindex(nucSymsLocal), eachindex(nucCoordsLocal), sourceIdxSeq)
        idx = OneToIndex(k)
        nucSymsLocal[i] = getEntry(nucSyms, idx)
        nucCoordsLocal[j] = Tuple(getEntry(nucCoords, idx))
    end

    NuclearCluster(MemoryPair(nucSymsLocal, nucCoordsLocal))
end

iterate(nc::NuclearCluster) = iterate(nc.layout)
iterate(nc::NuclearCluster, state) = iterate(nc.layout, state)

length(nc::NuclearCluster) = length(nc.layout)

eltype(::NuclearCluster{T, D}) where {T<:Real, D} = Pair{Symbol, NTuple{D, T}}


"""

    getCharge(nuc::Union{Tuple{Vararg{Symbol}}, AbstractVector{Symbol}}) -> Int

Return the total electric charge (in 𝑒) of the input nucleus/nuclei.
"""
getCharge(nuc::Symbol) = NuclearChargeDict[nuc]::Int

function getCharge(nuc::Union{Tuple{Vararg{Symbol}}, AbstractVector{Symbol}})
    mapreduce(getCharge, +, nuc, init=zero(Int))
end

function getCharge(nucInfo::NuclearCluster)
    mapreduce(+, nucInfo) do pair
        getCharge(pair.first)
    end
end


struct OccupationState{N} <: StateBox{UInt}
    layout::LinearMemory{UInt, N}

    function OccupationState(layout::LinearMemory{UInt, N}) where {N}
        checkPositivity(N)
        new{N}(layout)
    end
end

function OccupationState(input::LinearSequence{<:Integer})
    checkEmptiness(input, :input)
    layout = LinearMemory{UInt}(undef, (Val∘length)(input))
    for (i, ele) in zip(eachindex(layout), input)
        layout[i] = ele
    end
    OccupationState(layout)
end

function iterate(os::OccupationState)
    res = iterate(os.layout)
    if res === nothing
        nothing
    else
        nextItem, nextState = res
        Int(nextItem), nextState
    end
end

function iterate(os::OccupationState, state)
    res = iterate(os.layout, state)
    if res === nothing
        nothing
    else
        nextItem, nextState = res
        Int(nextItem), nextState
    end
end

length(os::OccupationState) = length(os.layout)

eltype(::OccupationState) = Int

getindex(state::OccupationState, index::Int) = 
(Int∘getEntry)(state.layout, OneToIndex(index))

firstindex(::OccupationState) = 1

lastindex(::OccupationState{N}) where {N} = N


getTotalOccupation(state::OccupationState) = (Int∘sum)(state.layout)


function prepareSpinConfiguration(nuc::Union{Symbol, NuclearCluster}, 
                                  difference::MissingOr{Int}=missing; offset::Int=0)
    totalOccu = getCharge(nuc) + offset
    prepareSpinConfigurationCore(totalOccu, difference)
end

function prepareSpinConfigurationCore(totalOccu::Int, 
                                      difference::MissingOr{Int}=missing)
    checkPositivity(totalOccu)

    occuPair = if ismissing(difference)
        nBeta  = totalOccu ÷ 2
        nAlpha = totalOccu - nBeta
        (nAlpha, nBeta)
    else
        if abs(difference) > totalOccu || iseven(difference) != iseven(totalOccu)
            throw(AssertionError("`difference=$difference` results in illegal spin "*
                                 "occupations."))
        end
        nAlpha = (totalOccu + difference) ÷ 2
        nBeta  = (totalOccu - difference) ÷ 2
        (nAlpha, nBeta)
    end

    OccupationState(occuPair)
end

function swapSpinSector!(state::OccupationState{2})
    nAlpha, nBeta = state.layout
    state.layout[begin] = nBeta
    state.layout[ end ] = nAlpha
    state
end

getOccuDifference(state::OccupationState{2}) = first(state.layout) - last(state.layout)

function splitSpinConfiguration(nucInfo::NuclearCluster)
    spinDifference::Int = 0
    atomicOccuStates = Memory{NuclearCluster{2}}(undef, length(nucInfo))

    for (i, pair) in zip(eachindex(atomicOccuStates), nucInfo)
        atomicOccuState = OccupationState(pair.first)
        spinDifference > 0 && swapSpinSector!(atomicOccuState)
        spinDifference += getOccuDifference(atomicOccuState)
        atomicOccuStates[i] = atomicOccuState
    end

    atomicOccuStates #! Need to test summing over each each section returns same result as `prepareSpinConfiguration(nucInfo)`
end


function nucRepulsion(nuc::LinearSequence{Symbol}, 
                          nucCoords::LinearSequence{NonEmptyTuple{T, D}}
                          ) where {T<:Real, D}
    res = T(0)
    for i in eachindex(nuc), j = (i+1):lastindex(nuc)
        res += getCharge(nuc[i]) * getCharge(nuc[j]) / norm(nucCoords[i] .- nucCoords[j])
    end
    res
end

function nucRepulsion(nucInfo::NuclearCluster)
    layout = nucInfo.layout
    nucRepulsion(layout.left, layout.right)
end