using Test
using Quiqbox: PseudoLRU, isValidPair, locateHashBlock, CreditPair, isEmptyPair, asTombstone
using Random: MersenneTwister

@testset "Caches.jl" begin

@testset "`PseudoLRU`" begin
    function genCache(capacity::Int, partition::Int=capacity)
        PseudoLRU{Int, Int}(capacity, partition)
    end

    mutable struct Atomic{T}; @atomic val::T; end

    function getMatchedPair(cache::PseudoLRU, pair::Pair)
        idx = nothing
        for blk in cache.block
            idx = findfirst(x->(isValidPair(x) && x.value==pair), blk.storage)
            if idx !== nothing
                return blk.storage[idx]
            end
        end

        nothing
    end

    function getBump(cache::PseudoLRU)
        mapreduce(+, cache.block, init=UInt(0)) do b
            b.bump
        end
    end

    function getMiss(cache::PseudoLRU)
        mapreduce(+, cache.block, init=UInt(0)) do b
            b.miss
        end
    end

    function getShift(cache::PseudoLRU)
        mapreduce(+, cache.block, init=UInt(0)) do b
            b.shift
        end
    end

    @testset "`haskey`, `getindex`, `setindex!`" begin
        for factor in (1, 2)
            factor = 2
            cache = genCache(factor * 4, 4)
            @test getBump(cache) == 0
            @test getMiss(cache) == 0
            @test getShift(cache) == 0

            cache[1] = 10
            @test getBump(cache) == 0
            @test getMiss(cache) == 0
            @test getShift(cache) == 0
            pair1 = cache.block[end].storage[3]
            @test pair1 === getMatchedPair(cache, 1=>10)
            @test Integer(pair1.credit) == 4
            @test pair1.value == (1 => 10)

            @test length(cache) == 1
            @test getBump(cache) == 0
            @test getMiss(cache) == 0
            @test getShift(cache) == 0
            @test Integer(pair1.credit) == 4

            @test haskey(cache, 1)
            @test getBump(cache) == 0
            @test getMiss(cache) == 0
            @test getShift(cache) == 0
            @test Integer(pair1.credit) == 4

            @test cache[1] == 10
            @test getBump(cache) == 0
            @test getMiss(cache) == 0
            @test getShift(cache) == 0
            @test Integer(pair1.credit) == 5

            #> Overwrite an existing key
            cache[1] = 20
            pair2 = getMatchedPair(cache, 1=>20)
            @test pair1.value != pair2.value
            @test Integer(pair2.credit) == 4
            @test cache[1] == 20
            @test length(cache) == 1

            #> Add more keys
            cache[2] = 30
            cache[3] = 40

            @test haskey(cache, 2)
            @test haskey(cache, 3)
            @test cache[2] == 30
            @test cache[3] == 40
            @test length(cache) == 3

            delete!(cache, 1)
            @test length(cache) == 2
            @test !haskey(cache, 1)
            @test cache[2] == 30
            @test cache[3] == 40
            @test Integer(getMatchedPair(cache, 2=>30).credit) == 6
            @test Integer(getMatchedPair(cache, 3=>40).credit) == 6
        end
    end

    @testset "`delete!`" begin
        for (shape, ids) in zip(( (5, 5), (11, 8) ), ( (3, 4, 10, 17), (6, 12, 16, 21) ))
            a, b, c, d = ids
            cache = genCache(shape...)
            storage = cache.block[begin].storage
            @test locateHashBlock(cache, a) == locateHashBlock(cache, b) == 
                  locateHashBlock(cache, c) == locateHashBlock(cache, d)

            for i in ids
                cache[i] = i
            end

            #> Deleting the tail key resets the slot to be empty
            idxEdge = findfirst(x->x.value==(d=>d), storage)
            for _ in 1:2 #>> The second time should have not effect
                delete!(cache, d)
                @test length(cache) == 3
                @test isEmptyPair(storage[idxEdge])
            end

            #> Deleting the body key marks the slot as a tombstone
            idxBody = findfirst(x->x.value==(b=>b), storage)
            pairBody = storage[idxBody]
            for _ in 1:2
                delete!(cache, b)
                @test length(cache) == 2
                @test storage[idxBody] === pairBody
                @test asTombstone(pairBody)
            end

            #> Refilling the slots
            @assert idxBody < idxEdge
            @test haskey(cache, c)
            cache[d] = d + 1
            @test haskey(cache, d)
            @test storage[idxBody].value == (d => d+1) #>> The tombstone is replaced
            @test length(cache) == 3
            cache[b] = b + 1
            @test haskey(cache, b)
            @test storage[idxEdge].value == (b => b+1) #>> The empty slot is filled
            @test length(cache) == 4

            #> Raiding removable tombstones when deleting keys
            delete!(cache, c)
            delete!(cache, b)
            @test isEmptyPair(storage[idxEdge]) && isEmptyPair(storage[idxEdge-1])
            @test Int(storage[idxEdge].credit) == 0 && Int(storage[idxEdge-1].credit) == 0
        end
    end

    @testset "`get`, `show`" begin
        for factor in (1, 2)
            cache = genCache(factor * 3, 3)
            cache[1] = 1
            cache[2] = 3
            pair = getMatchedPair(cache, 2=>3)
            @test Integer(pair.credit) == 4
            @test get(cache, 2, nothing) == cache[2]
            @test Integer(pair.credit) == 6

            len = length(cache)
            @test get(cache, 999, -1) == -1
            @test get(()->nothing, cache, 999) === nothing
            @test length(cache) == len
            @test !haskey(cache, 999)

            printStr = sprint(print, cache)

            @test printStr == "PseudoLRU(1 => 1, 2 => 3)" || 
                printStr == "PseudoLRU(2 => 3, 1 => 1)"
        end
    end

    @testset "`get!`" begin
        for factor in (1, 2)
            cache = genCache(factor * 4, 4)

            #> `get!(d, key, default)`
            v = get!(cache, 1, 10)
            @test v == 10
            @test cache[1] == 10
            @test length(cache) == 1
            pair1 = getMatchedPair(cache, 1=>10)
            @test Integer(pair1.credit) == 5

            #> Existing keys should not be overwritten by default
            v2 = get!(cache, 1, 99)
            @test Integer(pair1.credit) == 6
            @test v2 == 10
            @test cache[1] == 10
            @test length(cache) == 1

            #> `get!(f, d, key)`
            callCount = Ref(0)
            f2() = (callCount[] += 1; 99)

            v3 = get!(f2, cache, 2)
            @test v3 == 99
            @test callCount[] == 1
            @test cache[2] == 99
            @test length(cache) == 2

            #> On a hit, `f2` must not be called again
            v4 = get!(f2, cache, 2)
            @test v4 == 99
            @test callCount[] == 1
            @test cache[2] == 99
            @test length(cache) == 2
        end
    end

    @testset "`iterate`, `collect`" begin
        for f! in (get!, (d, k, v)->setindex!(d, v, k)), pt in (10, 5)
            cache = genCache(10, pt)

            for i in 1:5
                f!(cache, i, i^2)
            end

            #> `collect` must give a `Vector{Pair{Int, Int}}`
            cachePairs = collect(cache)
            @test length(cachePairs) == 5
            @test cachePairs isa Vector{Pair{Int, Int}}

            #> Order is not assumed: compare as sets
            @test Set(cachePairs) == Set(i => i^2 for i in 1:5)

            #> Iteration via `for (k, v) in cache` must see all pairs
            seen = Dict{Int,Int}()
            for (k, v) in cache
                seen[k] = v
            end
            @test seen == Dict(i => i^2 for i in 1:5)
        end
    end

    @testset "`empty!`" begin
        for pt in (10, 5)
            cache = genCache(10, pt)
            for i in 1:3
                cache[i] = i
            end
            @test length(cache) == 3

            empty!(cache)

            @test length(cache) == 0
            @test !haskey(cache, 1)
            @test getBump(cache) == getMiss(cache) == getShift(cache) == 0

            #> `collect` on the emptied cache after `empty!`
            @test let ps=collect(cache); ps == [] && ps isa Vector{Pair{Int, Int}} end
        end
    end

    @testset "Zero-slot cache" begin
        cache = genCache(0)
        @test length(cache) == 0
        @test get(cache, 1, nothing) === nothing
        @test get!(cache, 1, nothing) === nothing
        cache[1] = 2
        @test !haskey(cache, 1)
        @test length(cache) == 0
        delete!(cache, 1)
        @test length(cache) == 0
        @test isempty(cache|>collect)
    end

    @testset "One-slot cache" begin
        cache = genCache(1)
        @test length(cache) == 0
        @test get(cache, 1, nothing) === nothing
        cache[1] = 2
        @test haskey(cache, 1)
        @test length(cache) == 1
        @test cache[1] == 2
        @test get(cache, 1, nothing) == 2
        @test get!(cache, 1, 1) == 2
        delete!(cache, 1)
        @test length(cache) == 0
        @test get!(cache, 1, 3) == 3
        @test collect(cache)[] == (1 => 3)
    end

    @testset "Max capacity" begin
        cap = 20
        cache = genCache(cap)
        keys = [10, 06, 03, 07, 09, 04, 06, 05, 10, 08, 01, 08, -2, 01, -3, 01, 02, -1, 
                -4, 00, 00, 00, 03, 12, 09, 07, 11, 12, 08, 15, 14, 07, 06, 13, 12]
        vals = rand(-10:20, length(keys))
        for (k, v) in zip(keys, vals)
            get!(cache, k, v)
        end
        @test length(cache) == length(unique(keys)) == cap
    end

    @testset "Empty cache" begin
        for pt in (4, 1)
            cache1 = genCache(4, pt)

            @test length(cache1) == 0
            @test !haskey(cache1, 1)

            #> `collect` and iteration on an empty dict
            cachePairs1 = collect(cache1)
            @test isempty(cachePairs1)
            pairs = Pair{Int, Int}[]
            for p in cache1
                push!(pairs, p)
            end
            @test isempty(pairs)

            #> `getindex` should throw on missing keys
            @test_throws KeyError cache1[1]

            #> `get` should not insert the default
            @test get(cache1, 1, 42) == 42
            @test !haskey(cache1, 1)
            @test length(cache1) == 0

            #> zero-capacity dummy cache
            cache = genCache(0)
            @test length(cache) == 0
            @test !haskey(cache, 1)
            cachePairs = collect(cache)
            @test isempty(cachePairs)
            @test eltype(cachePairs) == Pair{Int, Int}
            @test get(cache, 1, 42) == 42
            @test !haskey(cache, 1)
            @test length(cache) == 0
            @test get!(cache, 1, 42) == 42
            @test !haskey(cache, 1)
            @test length(cache) == 0
            @test setindex!(cache, 1, 42) === cache
            @test !haskey(cache, 1)
            @test length(cache) == 0
        end
    end

    @testset "Eviction of oldest key when over capacity" begin
        cap = 3
        cache = genCache(cap)

        for i in 1:cap
            get!(cache, i, i)
        end
        @test length(cache) == cap

        #> Insert a new key to force eviction
        cache[cap+1] = cap + 1

        #> Capacity must be respected
        @test length(cache) == cap

        #> First inserted key (`1`) should have been evicted
        @test !haskey(cache, 1)
        @test all(haskey(cache, i) for i in 2:(cap+1))
    end

    @testset "Recent access to protect eviction" begin
        cap = 3
        cache = genCache(cap)

        cache[1] = 1
        pair = getMatchedPair(cache, 1=>1)
        @test Integer(pair.credit) == 4
        cache[3] = 3
        cache[2] = 2

        #> Access key `1` and `2`
        cache[1]; cache[1]; cache[2]
        for i in 1:3
            @test Integer(getMatchedPair(cache, i=>i).credit) == 7-i
        end

        #> Trigger eviction
        cache[4] = 4 #> `3` was the least recently/frequently used among (1, 2, 3)
        @test !haskey(cache, 3)
        @test Set(cache) == Set((1, 2, 4) .=> (1, 2, 4))
    end

    @testset "Thread safety" begin
        threadNum = Threads.nthreads()

        if threadNum == 1
            @info "Skipping thread-safety tests due to `Threads.nthreads() == 1`."
        else
            keysPerThread = 10
            @testset "Concurrent `setindex!` to distinct keys" begin
                pt = threadNum * keysPerThread

                for factor in (1, 2)
                    cap = factor * pt
                    cache = genCache(cap, pt)

                    @sync for id in 1:threadNum
                        Threads.@spawn begin
                            kBegin = keysPerThread * (id - 1) + 1
                            kFinal = keysPerThread * id

                            for k in kBegin:kFinal
                                cache[k] = id
                            end
                        end
                    end

                    #> No eviction should have happened
                    @test pt == length(cache)

                    #> Every key written by a thread should be present and have correct `id`
                    @test mapreduce(*, 1:threadNum, init=true) do id
                        kBegin = keysPerThread * (id - 1) + 1
                        kFinal = keysPerThread * id

                        mapreduce(*, kBegin:kFinal, init=true) do k
                            haskey(cache, k) && cache[k] == id
                        end
                    end
                end
            end

            @testset "Concurrent `get!` on repeated keys" begin
                pt = 4
                monoKey = 1

                for factor in (1, 3)
                    cap = factor * pt
                    cache1 = genCache(cap, pt)

                    defaultVal = 12345
                    defaultCount = Atomic(0) #> Counter for calling `getDefaultVal`
                    function getDefaultVal()
                        @atomic defaultCount.val += 1
                        defaultVal
                    end

                    results = Vector{Int}(undef, threadNum)
                    @sync for i in 1:threadNum
                        Threads.@spawn begin
                            results[i] = get!(getDefaultVal, cache1, monoKey)
                        end
                    end

                    #> All results must agree with the value stored in the `cache1`
                    @test haskey(cache1, monoKey)
                    final = cache1[monoKey]
                    @test all(r->(r == final == defaultVal), results)

                    #> After invocations of `get!`, `getDefaultVal` should execute only once
                    @test defaultCount.val == 1

                    #> Capacity is invariant
                    @test length(cache1) == 1

                    #> Test multi-key scenario
                    cache2 = genCache(cap, cap)
                    valPool = rand(1:10, 100)
                    keyPool = rand(-5:5, 100)
                    pairs = keyPool .=> valPool
                    Threads.@threads for pair in pairs
                        get!(cache2, pair.first, pair.second)
                    end
                    @test length(cache2) == min(11, cap)
                    cache2Pairs = collect(cache2)
                    @test issubset(first.(cache2Pairs), Set(-5:5))
                    @test issubset( last.(cache2Pairs), Set(1:10))
                end
            end

            @testset "Concurrent mixed read/write stress test" begin
                cap = max(64, 4 * keysPerThread)
                keyMax = 2 * cap
                niter = 1000

                flag1 = true
                flag2 = true

                #> Repeat the test to reduce the chance of getting false-positive results
                for _ in 1:100, divider in (1, 3)
                    cache = genCache(cap, (cap+2)÷divider)
                    errorCount = Atomic(0)

                    @sync for id in 1:threadNum
                        Threads.@spawn begin
                            rng = MersenneTwister(id)
                            for i in 1:niter
                                k = rand(rng, 1:keyMax)
                                if rand(rng) < 0.5
                                    cache[k] = id #> Write
                                else
                                    v = get(cache, k, 0) #> Read
                                    (0 <= v <= threadNum) || (@atomic errorCount.val += 1)
                                end
                            end
                        end
                    end

                    #> No invalid values should have been observed
                    flag1 *= iszero(errorCount.val)

                    #> Capacity invariant under stress
                    nStored = length(cache)
                    maxCap, minCap = cache.shape
                    bl = if maxCap == minCap
                        nStored <= maxCap
                    else
                        minCap <= nStored <= maxCap
                    end
                    bl || println("nStored = ", nStored)
                    flag2 *= bl
                end

                @test flag1
                @test flag2
            end
        end
    end
end

end