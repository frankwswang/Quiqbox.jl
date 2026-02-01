#>/ Custom pseudo LRU (least-recent used) cache \<#
function checkCreditQuota(initial::Union{Integer, OctalNumber}, 
                          ceiling::Union{Integer, OctalNumber})

    if !(1 <= Integer(initial) <= Integer(ceiling) <= 7)
        throw(AssertionError("The range of `quota` $(Integer(initial):Integer(ceiling)) " * 
                             "must be within `1:7`."))
    end

    initial isa Integer && (initial = OctalNumber(initial))
    ceiling isa Integer && (ceiling = OctalNumber(ceiling))

    initial => ceiling
end

checkCreditQuota(quota::Pair) = checkCreditQuota(quota.first, quota.second)


mutable struct CreditPair{K, V}
    @atomic credit::OctalNumber #> ==0: tombstone, >=1: occupied with valid pairs
    const  ceiling::OctalNumber #> ==0: Empty Slot
    const    value::Pair{K, V}

    @generated function CreditPair{K, V}() where {K, V}
        emptyPair = new{K, V}(OUS0, OUS0)
        return :($emptyPair)
    end

    function CreditPair{K, V}(pair::Pair{<:K, <:V}, quota::Pair) where {K, V}
        initial, ceiling = checkCreditQuota(quota)
        new{K, V}(initial, ceiling, pair)
    end
end


function getCreditStatus(pair::CreditPair)
    credit = UInt8(@atomic :acquire pair.credit)
    !isEmptyPair(pair) => credit #> is-slot-occupied => pair-credit
end

function isEmptyPair(pair::CreditPair)
    (iszero∘Integer)(pair.ceiling)
end

function isValidPair(pair::CreditPair)
    isOccupied, credit = getCreditStatus(pair)
    isOccupied && !iszero(credit)
end

function asTombstone(pair::CreditPair)
    isOccupied, credit = getCreditStatus(pair)
    isOccupied && iszero(credit)
end

getPairVal(pair::CreditPair) = pair.value.second
getPairKey(pair::CreditPair) = pair.value.first


mutable struct CacheBlock{K, V}
    @atomic  bump::UInt #> Counter of matched keys
    @atomic  miss::UInt #> Counter of mismatched keys
    @atomic shift::UInt #> Counter of pairs stored away from its hash index
    const   track::Bool
    const   space::Int
    const   mutex::ReentrantLock
    const storage::AtomicMemory{CreditPair{K, V}}

    function CacheBlock{K, V}(space::Integer; track::Bool=true) where {K, V}
        space = (Int∘checkPositivity)(space, true)
        container = AtomicMemory{CreditPair{K, V}}(undef, space)

        for i in eachindex(container)
            setEntry!(container, CreditPair{K, V}(), i)
        end

        new{K, V}(0, 0, 0, track, space, ReentrantLock(), container)
    end
end


struct PseudoLRU{K, V} <: AbstractDict{K, V}
    block::AtomicMemory{CacheBlock{K, V}}
    shape::Pair{Int, Int}                 #> `capacity` => `partition`
    quota::Pair{OctalNumber, OctalNumber} #>  initial   =>  ceiling

    function PseudoLRU{K, V}(capacity::Integer, partition::Integer=32; track::Bool=true, 
                             quota::Pair{<:Integer, <:Integer}=Pair(4, 7)) where {K, V}
        checkPositivity(capacity, true)
        blockSpace = min(checkPositivity(partition, iszero(capacity)), capacity)

        blockCount, tailSpace = capacity > 0 ? fldmod1(capacity, blockSpace) : (0, 0)
        blocks = AtomicMemory{CacheBlock{K, V}}(undef, blockCount)

        for (n, idx) in enumerate(eachindex(blocks))
            space = ifelse(n < blockCount, blockSpace, tailSpace)
            setEntry!(blocks, CacheBlock{K, V}(space; track), idx)
        end

        new{K, V}(blocks, capacity=>blockSpace, checkCreditQuota(quota))
    end
end

function genHashIndex(key::K, maxIndex::Unsigned) where {K}
    maxIndex = (Int∘max)(1, maxIndex)
    hashBranch = ispow2(maxIndex)
    hashVal = convert(UInt, hash(key))
    (hashBranch ? (hashVal & (maxIndex - 1)) : (hashVal % maxIndex)) + 1
end

function locateHashBlock(d::PseudoLRU{K}, key::K) where {K}
    capacity, divider = d.shape
    hashIndex = genHashIndex(key, UInt(capacity))
    secNum, rawIdx = fldmod1(hashIndex, divider)
    getEntry(d.block, OneToIndex(secNum)) => OneToIndex(rawIdx)
end


function dialDn(init::OctalNumber, lower::OctalNumber)
    OctalNumber(UInt8(init) - UInt8(init > lower))
end

function dialUp(init::OctalNumber, upper::OctalNumber)
    OctalNumber(UInt8(init) + UInt8(init < upper))
end


function linearProbe(b::CacheBlock{K}, info::Pair{K, OneToIndex}, 
                     dial!::Bool=true) where {K}
    cPairs = b.storage
    key, index = info
    maxIter = b.space
    maxCredit = typemax(UInt8)
    evictInfo = nothing #> Require `while true` to maintain type stability
    keyMatched = false
    nIter = 0

    while true
        cPair = getEntry(cPairs, index)
        isOccupied, credit = getCreditStatus(cPair)
        nIter += 1

        earlyExit = if iszero(credit) #> Empty slot or tombstone
            !isOccupied #> Tombstone is not still considered an occupied slot
        else #> Valid pair
            keyMatched = isequal(getPairKey(cPair), key)

            if dial!
                if keyMatched
                    @atomic :monotonic cPair.credit dialUp cPair.ceiling
                else #> Won't mark any slot as a tombstone (`.credit > OUS0`)
                    @atomic :monotonic cPair.credit dialDn OPS1
                end
            end

            keyMatched
        end

        if evictInfo === nothing || keyMatched || credit < maxCredit
            keyMatched || (maxCredit = credit)
            evictInfo = (index => cPair)
        end

        (earlyExit || nIter >= maxIter) && break

        index = index.idx < maxIter ? OneToIndex(index, Count(1)) : OneToIndex()
    end

    (keyMatched, evictInfo)
end


function delete!(d::PseudoLRU{K, V}, key::K) where {K, V}
    if d.shape.first > 0
        block, startIdx = locateHashBlock(d, key)
        cPairs = block.storage
        space = block.space
        upperIndex = OneToIndex(space)

        @lock block.mutex begin
            probRes = linearProbe(block, key=>startIdx, false)
            if first(probRes) #> Mark the pair as empty if it is already at the tail
                indexHere, cPairHere = last(probRes)
                indexNext = indexHere < upperIndex ? (indexHere + 1) : OneToIndex()
                cPairNext = getEntry(cPairs, indexNext)

                if isEmptyPair(cPairNext)
                    setEntry!(cPairs, CreditPair{K, V}(), indexHere)

                    for _ in 1:(space - 1) #> Further remove adjacent tombstones
                        indexPrev = indexHere > OneToIndex() ? (indexHere - 1) : upperIndex
                        cPairPrev = getEntry(cPairs, indexPrev)

                        if asTombstone(cPairPrev) #> Raid the tombstone
                            setEntry!(cPairs, CreditPair{K, V}(), indexPrev)
                            indexHere = indexPrev
                        else
                            break
                        end
                    end
                else
                    @atomic :release cPairHere.credit = OUS0
                end
            end
        end
    end

    d
end


function setindex!(d::PseudoLRU{K, V}, val::V, key::K) where {K, V}
    if d.shape.first > 0
        block, startIdx = locateHashBlock(d, key)
        pNew = CreditPair{K, V}(key=>val, d.quota)

        @lock block.mutex begin
            evictIdx = linearProbe(block, key=>startIdx, false)[end].first
            (block.track && evictIdx != startIdx) && (@atomic :monotonic block.shift += 1)
            setEntry!(block.storage, pNew, evictIdx)
        end
    end

    d
end


function haskey(d::PseudoLRU{K}, key::K) where {K}
    if d.shape.first > 0
        block, startIdx = locateHashBlock(d, key)
        cPair = getEntry(block.storage, startIdx)

        if isValidPair(cPair) && getPairKey(cPair) == key
            true
        elseif block.space == 1
            false
        else
            @lock block.mutex linearProbe(block, key=>startIdx, false)[begin]
        end::Bool
    else
        false
    end
end


evalObj(::True, obj::AbstractCallable) = obj()

evalObj(::False, obj) = itself(obj)


function recordInputKey!(b::CacheBlock, hasKey::Bool)
    if hasKey
        @atomic :monotonic b.bump += 1
    else
        @atomic :monotonic b.miss += 1
    end

    nothing
end


function checkEval(callObj::Boolean, obj, d::PseudoLRU{K}, key::K, 
                   trackStatistic::MissingOr{Bool}=missing) where {K}
    d.shape.first > 0 || (return evalObj(callObj, obj))

    block, startIdx = locateHashBlock(d, key)
    cPair = getEntry(block.storage, startIdx)

    val = if isValidPair(cPair) && getPairKey(cPair) == key
        @atomic :monotonic cPair.credit dialUp cPair.ceiling
        matched = true
        getPairVal(cPair)
    elseif block.space == 1
        matched = false
        evalObj(callObj, obj)
    else
        matched, evictInfo = @lock block.mutex linearProbe(block, key=>startIdx, true)
        matched ? getPairVal(evictInfo.second) : evalObj(callObj, obj)
    end

    if (ismissing(trackStatistic) ? block.track : trackStatistic)
        recordInputKey!(block, matched)
    end

    val
end

get(d::PseudoLRU{K}, key::K, default) where {K} = checkEval(False(), default, d, key)

get(f::CommonCallable, d::PseudoLRU{K}, key::K) where {K} = checkEval(True(), f, d, key)


function directSet!(callObj::Boolean, obj, block::CacheBlock{K, V}, 
                    info::Pair{K, OneToIndex}, creditQuota::Pair{OctalNumber, OctalNumber}
                    ) where {K, V}
    key, idx = info
    objVal = evalObj(callObj, obj)
    cPair = CreditPair{K, V}(key=>objVal, creditQuota)
    setEntry!(block.storage, cPair, idx)
    objVal
end

function checkSet!(callObj::Boolean, obj, d::PseudoLRU{K, V}, key::K, 
                   trackStatistic::MissingOr{Bool}=missing) where {K, V}
    d.shape.first > 0 || (return evalObj(callObj, obj))

    block, startIdx = locateHashBlock(d, key)
    cPairs = block.storage
    cPair = getEntry(cPairs, startIdx)
    ismissing(trackStatistic) && (trackStatistic = block.track)

    val = if isValidPair(cPair) && getPairKey(cPair) == key
        @atomic :monotonic cPair.credit dialUp cPair.ceiling
        matched = true
        getPairVal(cPair)
    else
        @lock block.mutex begin
            if block.space == 1
                matched = false
                directSet!(callObj, obj, block, key=>startIdx, d.quota)
            else
                matched, evictInfo = linearProbe(block, key=>startIdx, true)
                evictIdx, pickedPair = evictInfo

                if matched
                    getPairVal(pickedPair)
                else
                    if trackStatistic && evictIdx != startIdx
                        (@atomic :monotonic block.shift += 1)
                    end

                    directSet!(callObj, obj, block, key=>evictIdx, d.quota)
                end
            end
        end
    end

    trackStatistic && recordInputKey!(block, matched)
    val
end


get!(d::PseudoLRU{K}, key::K, default) where {K} = checkSet!(False(), default, d, key)

get!(f::CommonCallable, d::PseudoLRU{K}, key::K) where {K} = checkSet!(True(), f, d, key)


function getindex(d::PseudoLRU{K, V}, key::K) where {K, V}
    finalizer = ()->throw(KeyError("`key = $key` is not found."))
    checkEval(True(), finalizer, d, key, false)::V
end



function countStored(d::PseudoLRU, executor::F=itself) where {F<:AbstractCallable}
    nStored::Int = 0
    blocks = d.block

    for j in eachindex(blocks)
        block = getEntry(blocks, j)
        cPairs = block.storage
        @lock block.mutex begin
            for i in eachindex(cPairs)
                cPair = getEntry(cPairs, i)
                if isValidPair(cPair)
                    nStored += 1
                    executor(OneToIndex(i) => cPair)
                end
            end
        end
    end

    nStored
end

length(d::PseudoLRU) = countStored(d)


function iterate(b::CacheBlock, state::OneToIndex=OneToIndex())
    space = b.space
    if space == 0 || state.idx > space
        nothing
    else
        cPair = getEntry(b.storage, state)
        nextState = OneToIndex(state, Count(1))

        if isValidPair(cPair)
            cPair.value, nextState
        else
            iterate(b, nextState)
        end
    end
end

function iterate(d::PseudoLRU, 
                 state::Pair{OneToIndex, OneToIndex}=( OneToIndex()=>OneToIndex() ))
    blocks = d.block
    blockNum = length(blocks)
    blockIdx, pairIdx = state
    i = blockIdx.idx

    if i > blockNum
        nothing
    else
        block = getEntry(blocks, blockIdx)
        blockRes = iterate(block, pairIdx)

        if blockRes === nothing
            if i < blockNum
                iterate(d, OneToIndex(blockIdx, Count(1))=>OneToIndex())
            else
                nothing
            end
        else
            (first(blockRes), blockIdx=>last(blockRes))
        end
    end
end


function collect(d::PseudoLRU{K, V}) where {K, V}
    container = Pair{K, V}[]
    f = function fillUpContainer(pairInfo::Pair{OneToIndex, CreditPair{K, V}})
        push!(container, pairInfo.second.value)
    end

    countStored(d, f)
    container
end


function empty!(d::PseudoLRU{K, V}) where {K, V}
    blocks = d.block

    for j in eachindex(blocks)
        block = getEntry(blocks, j)
        cPairs = block.storage

        @lock block.mutex begin
            for i in eachindex(cPairs)
                @atomic :release cPairs[i] = CreditPair{K, V}()
            end
        end

        if block.track
            @atomic :release block.bump = 0
            @atomic :release block.miss = 0
            @atomic :release block.shift = 0
        end
    end

    d
end
#>\ Custom pseudo LRU (least-recent used) cache /<#


const OptPseudoLRU{K, V} = Union{EmptyDict{K, V}, PseudoLRU{K, V}}