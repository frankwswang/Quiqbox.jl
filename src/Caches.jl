#>/ Custom LRU (least-recent used) cache \<#
mutable struct StampedPair{K, V}
    stamp::UInt #> Set to `UInt(0)` for idle invalid slot (`.valid=false`)
    const valid::Bool
    const value::Pair{K, V}

    function StampedPair{K, V}() where {K, V}
        new{K, V}(UInt(0), false)
    end

    function StampedPair{K, V}(stamp::UInt, value::Pair{<:K, <:V}) where {K, V}
        new{K, V}(stamp, true, value)
    end
end

mutable struct AtomicLRU{K, V} <: AbstractDict{K, V}
    const direct::Memory{StampedPair{K, V}}
    const hidden::Memory{StampedPair{K, V}}
    const mutex::Pair{AtomicMemory{ReentrantLock}, ReentrantLock}
    const space::Int
    @atomic bump::UInt  #> Counter for successfully getting keys
    @atomic miss::UInt  #> Counter for failing to get keys
    @atomic epoch::UInt #> Counter for updating the stamps
    @atomic clash::UInt #> Counter for hash collisions

    function AtomicLRU{K, V}(directSpace::Int, hiddenSpace::Int=directSpace-1) where {K, V}
        checkPositivity(directSpace)
        checkPositivity(hiddenSpace, true)
        generator = _->StampedPair{K, V}()
        dSec = genMemory(generator, StampedPair{K, V}, directSpace)
        hSec = genMemory(generator, StampedPair{K, V}, hiddenSpace)
        entryLocks = AtomicMemory{ReentrantLock}(undef, directSpace)
        for i in eachindex(entryLocks)
            setEntry!(entryLocks, ReentrantLock(), i)
        end

        new{K, V}(dSec, hSec, Pair(entryLocks, ReentrantLock()), directSpace, 0, 0, 0, 0)
    end
end

function getHashedIndex(d::AtomicLRU{K}, key::K) where {K}
    #> Help with constant propagation
    directSpace = d.space
    hashBranch = ispow2(directSpace)
    hashVal = convert(UInt, hash(key))

    offset = hashBranch ? (hashVal & (directSpace - 1)) : (hashVal % directSpace)

    Int(offset + 1) |> OneToIndex
end

const KeyStatus = Pair{Bool, Tuple{Bool, OneToIndex}} #> `hasKey => (isDirectSec, secIdx)`

function checkHiddenSlot(d::AtomicLRU{K}, keyInfo::Pair{<:K, Tuple{UInt, OneToIndex}}, 
                         finalizer::F=itself) where {K, F<:AbstractCallable}
    key, (directSecStamp, directSecIndex) = keyInfo
    hiddenSector = d.hidden
    @atomic d.clash += 1

    supperStamp::UInt = directSecStamp
    idx::Int = directSecIndex.idx
    toDirectSec::Bool = true
    hasKey::Bool = false

    @lock d.mutex.second begin
        i = 0

        for sPair in hiddenSector
            i += 1
            isValidPair = sPair.valid

            #> `isValidPair` is needed to avoid matching a random key in an invalid `sPair`
            if isValidPair && isequal(key, sPair.value.first)
                toDirectSec = false
                hasKey = true
                idx = i
                break
            elseif !isValidPair #> Get next empty (currently invalid but idle) slot
                toDirectSec = false
                idx = i
                break
            else #> Get the oldest slot as the next one
                localStamp = sPair.stamp
                if localStamp < supperStamp
                    supperStamp = localStamp
                    toDirectSec = false
                    idx = i
                end
            end
        end

        finalizer(KeyStatus( hasKey, (toDirectSec, OneToIndex(idx)) ))
    end
end

function checkCacheSlot(d::AtomicLRU{K}, key::K, finalizer::F=itself) where 
                       {K, F<:AbstractCallable}
    directSecIdx = getHashedIndex(d, key)
    lk = getEntry(d.mutex.first, directSecIdx)

    @lock lk begin
        sPair = getEntry(d.direct, directSecIdx)
        hasKey = false
        marker = (true, directSecIdx)

        if sPair.valid
            if isequal(sPair.value.first, key)
                hasKey = true
            elseif !isempty(d.hidden)
                directSecStamp = sPair.stamp
                return checkHiddenSlot(d, key=>(directSecStamp, directSecIdx), finalizer)
            end
        end

        #> `true` => existing-key slot marker; `false` => next available slot marker
        status = KeyStatus(hasKey, marker)
        finalizer(status)
    end
end

getCacheSector(d::AtomicLRU{K, V}, isDirectSec::Bool) where {K, V} = 
getfield(d, ifelse(isDirectSec, :direct, :hidden))::Memory{StampedPair{K, V}}

function length(d::AtomicLRU)::Int
    res::Int = 0
    for _ in d; res += 1 end
    res
end

function iterateDirectSec(d::AtomicLRU, secIdx::OneToIndex)
    sector = getCacheSector(d, true)

    if secIdx.idx > length(sector)
        iterate(d, ( false, OneToIndex() ))
    else
        nextIdx = secIdx + 1
        lk = getEntry(d.mutex.first, secIdx)

        pair = @lock lk begin
            sPair = getEntry(sector, secIdx)
            sPair.valid ? sPair.value : nothing
        end

        if pair === nothing
            iterateDirectSec(d, nextIdx)
        else
            pair, (true, nextIdx)
        end
    end
end
#> Not thread-safe by itself, needs to be called inside a lock
function iterateHiddenSec(d::AtomicLRU, secIdx::OneToIndex)
    sector = getCacheSector(d, false)
    if secIdx.idx > length(sector)
        nothing
    else
        nextIdx = secIdx + 1
        sPair = getEntry(sector, secIdx)
        if sPair.valid
            sPair.value, (false, nextIdx)
        else
            iterateHiddenSec(d, nextIdx)
        end
    end
end

function iterate(d::AtomicLRU, state::Tuple{Bool, OneToIndex}=( true, OneToIndex() ))
    isDirectSec, secIdx = state
    if isDirectSec
        iterateDirectSec(d, secIdx)
    elseif isempty(d.hidden)
        nothing
    else
        @lock d.mutex.second iterateHiddenSec(d, secIdx)
    end
end

collect(d::AtomicLRU) = [ele for ele in d]

haskey(d::AtomicLRU{K}, key::K) where {K} = checkCacheSlot(d, key, first)

function unsafeBumpExtract!(d::AtomicLRU, marker::Tuple{Bool, OneToIndex}, 
                            bump::Bool=true)
    isDirectSec, secIdx = marker
    sector = getCacheSector(d, isDirectSec)
    bump && (@atomic d.bump += 1)

    sPair = getEntry(sector, secIdx)
    sPair.stamp = (@atomic d.epoch += 1)

    sPair.value.second
end

function getEval(callObj::Boolean, obj, d::AtomicLRU{K}, key::K) where {K}
    finalizer = function getEvalCore(status::KeyStatus)
        if status.first
            unsafeBumpExtract!(d, status.second)
        else
            @atomic d.miss += 1
            evalObj(callObj, obj)
        end
    end

    checkCacheSlot(d, key, finalizer)
end

evalObj(::True, obj::AbstractCallable) = obj()

evalObj(::False, obj) = itself(obj)

get(d::AtomicLRU{K}, key::K, default) where {K} = getEval(False(), default, d, key)

get(f::CommonCallable, d::AtomicLRU{K}, key::K) where {K} = getEval(True(), f, d, key)

function unsafeSetByMarker!(d::AtomicLRU{K, V}, pair::Pair{<:K, <:V}, 
                            marker::Tuple{Bool, OneToIndex}) where {K, V}
    isDirectSec, cacheIndex = marker
    sector = getCacheSector(d, isDirectSec)
    sPair = StampedPair{K, V}((@atomic d.epoch += 1), pair)

    setEntry!(sector, sPair, cacheIndex)

    sPair.value.second
end

function getSet!(callObj::Boolean, obj, d::AtomicLRU{K, V}, key::K) where {K, V}
    finalizer! = function getSetCore!(status::KeyStatus)
        marker = status.second
        if status.first
            unsafeBumpExtract!(d, marker)
        else
            @atomic d.miss += 1
            unsafeSetByMarker!(d, key=>evalObj(callObj, obj), marker)
        end
    end

    checkCacheSlot(d, key, finalizer!)
end

get!(d::AtomicLRU{K}, key::K, default) where {K} = getSet!(False(), default, d, key)

get!(f::CommonCallable, d::AtomicLRU{K}, key::K) where {K} = getSet!(True(), f, d, key)

function getindex(d::AtomicLRU{K, V}, key::K) where {K, V}
    finalizer = function getindexCore(status::KeyStatus)
        if status.first
            unsafeBumpExtract!(d, status.second, false)
        else
            throw(KeyError("`key = $key` is not found."))
        end
    end

    checkCacheSlot(d, key, finalizer)
end

function setindex!(d::AtomicLRU{K, V}, val::V, key::K) where {K, V}
    finalizer! = function setindexCore!(status::KeyStatus)
        unsafeSetByMarker!(d, key=>val, status.second)
    end

    checkCacheSlot(d, key, finalizer!)
    d
end

function empty!(d::AtomicLRU{K, V}) where {K, V}
    atomicLocks = d.mutex.first
    for (i, j) in zip(eachindex(d.direct), eachindex(atomicLocks))
        lk = getEntry(atomicLocks, j)
        @lock lk d.direct[i] = StampedPair{K, V}()
    end

    @lock d.mutex.second begin
        for j in eachindex(d.hidden)
            d.hidden[j] = StampedPair{K, V}()
        end
    end

    @atomic d.bump = 0
    @atomic d.miss = 0
    @atomic d.clash = 0

    d
end
#>\ Custom LRU (least-recent used) cache /<#


const OptAtomicLRU{K, V} = Union{EmptyDict{K, V}, AtomicLRU{K, V}}