using Test
using Quiqbox: AtomicLRU
using Random: MersenneTwister

@testset "Caches.jl" begin

@testset "`AtomicLRU`" begin
    genSafeCache = capacity -> AtomicLRU{Int, Int}(capacity, capacity-1)
    mutable struct Atomic{T}; @atomic val::T; end

    @testset "`haskey`, `getindex`, `setindex!`" begin
        cache = genSafeCache(4)
        @test cache.bump == 0
        @test cache.miss == 0
        @test cache.epoch == 0
        @test cache.clash == 0

        cache[1] = 10
        @test cache.bump == 0
        @test cache.miss == 0
        @test cache.epoch == 1

        @test length(cache) == 1
        @test cache.bump == 0
        @test cache.miss == 0
        @test cache.epoch == 1

        @test haskey(cache, 1)
        @test cache.bump == 0
        @test cache.miss == 0
        @test cache.epoch == 1

        @test cache[1] == 10
        @test cache.bump == 0
        @test cache.miss == 0
        @test cache.epoch == 2

        #> Overwrite an existing key
        cache[1] = 20
        @test cache.epoch == 3
        @test cache[1] == 20
        @test cache.epoch == 4
        @test length(cache) == 1

        #> Add more keys
        cache[2] = 30
        cache[3] = 40

        @test haskey(cache, 2)
        @test haskey(cache, 3)
        @test cache[2] == 30
        @test cache[3] == 40
        @test length(cache) == 3
    end

    @testset "`get`" begin
        cap = 3
        cache = genSafeCache(3)
        cache[1] = 1
        cache[2] = 2
        f1 = sec->findfirst(x->x.value.second==2, sec)
        idx = f1(cache.direct)
        node = idx === nothing ? cache.hidden[f1(cache.hidden)] : cache.direct[idx]
        @test node.stamp == cache.epoch == 2
        @test get(cache, 2, nothing) == cache[2]
        @test node.stamp == cache.epoch == 4

        len = length(cache)
        @test get(cache, 999, -1) == -1
        @test length(cache) == len
        @test !haskey(cache, 999)
    end

    @testset "`get!`" begin
        cache = genSafeCache(4)

        #> `get!(d, key, default)`
        v = get!(cache, 1, 10)
        @test cache.epoch == 1
        @test v == 10
        @test cache[1] == 10
        @test length(cache) == 1

        #> Existing keys should not be overwritten by default
        v2 = get!(cache, 1, 99)
        @test cache.epoch == 3
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

    @testset "`iterate`, `collect`" begin
        cache = genSafeCache(10)

        for i in 1:5
            cache[i] = i^2
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

    @testset "`empty!`" begin
        cache = genSafeCache(10)
        for i in 1:3
            cache[i] = i
        end
        @test length(cache) == 3

        empty!(cache)

        @test length(cache) == 0
        @test !haskey(cache, 1)
        @test cache.epoch == 3
        @test cache.clash == cache.bump == cache.miss == 0

        #> `collect` on the emptied cache after `empty!`
        @test let ps=collect(cache); ps == [] && ps isa Vector{Pair{Int, Int}} end
    end

    @testset "Empty cache" begin
        cache = genSafeCache(4)

        @test length(cache) == 0
        @test !haskey(cache, 1)

        #> `collect` and iteration on an empty dict
        cachePairs = collect(cache)
        @test isempty(cachePairs)
        pairs = Pair{Int, Int}[]
        for p in cache
            push!(pairs, p)
        end
        @test isempty(pairs)

        #> `getindex` should throw on missing keys
        @test_throws KeyError cache[1]

        #> `get` should not insert the default
        @test get(cache, 1, 42) == 42
        @test !haskey(cache, 1)
        @test length(cache) == 0
    end

    genSpaceStrictCache = capacity -> AtomicLRU{Int, Int}(1, capacity-1)

    @testset "Eviction of oldest key when over capacity" begin
        cap = 3
        cache = genSpaceStrictCache(cap) #> space-strict linear search cache

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
        cache = genSpaceStrictCache(cap)

        cache[1] = 1
        node = cache.direct[findfirst(x->x.value.second==1, cache.direct)]
        @test node.stamp == cache.epoch == 1
        cache[2] = 2
        cache[3] = 3

        #> Access key `1` to make it most recently used
        cache[1]
        @test node.stamp == cache.epoch == 4

        #> Trigger eviction
        cache[4] = 4 #> `2` was the least recently used among (1, 2, 3)
        @test !haskey(cache, 2)
        @test Set(cache) == Set((1, 3, 4) .=> (1, 3, 4))
    end

    @testset "Thread safety" begin
        threadNum = Threads.nthreads()

        if threadNum == 1
            @info "Skipping thread-safety tests due to `Threads.nthreads() == 1`."
        else
            n = 2 * threadNum

            @testset "Concurrent `setindex!` to distinct keys" begin
                keysPerThread = 32
                cap = n * keysPerThread
                cache = genSafeCache(cap)

                @sync for id in 1:n
                    Threads.@spawn begin
                        kBegin = keysPerThread * (id - 1) + 1
                        kFinal = keysPerThread * id

                        for k in kBegin:kFinal
                            cache[k] = id
                        end
                    end
                end

                #> No eviction should have happened
                @test length(cache) == cap

                #> Every key written by a thread should be present and have the correct `id`
                @test mapreduce(*, 1:n, init=true) do id
                    kBegin = keysPerThread * (id - 1) + 1
                    kFinal = keysPerThread * id

                    mapreduce(*, kBegin:kFinal, init=true) do k
                        haskey(cache, k) && cache[k] == id
                    end
                end
            end

            @testset "Concurrent `get!` on repeated keys" begin
                cap = 4
                cache1 = genSafeCache(cap)
                monoKey = 1

                defaultVal = 12345
                defaultCount = Atomic(0) #> Counter for calling `getDefaultVal`
                function getDefaultVal()
                    @atomic defaultCount.val += 1
                    defaultVal
                end

                results = Vector{Int}(undef, n)
                @sync for id in 1:n
                    Threads.@spawn begin
                        results[id] = get!(getDefaultVal, cache1, monoKey)
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
                cache2 = genSafeCache(32)
                valPool = rand(1:10, 100)
                keyPool = rand(-5:5, 100)
                pairs = keyPool .=> valPool
                Threads.@threads for pair in pairs
                    get!(cache2, pair.first, pair.second)
                end
                @test length(cache2) == 11
                @test cache2.epoch == 100
                cache2Pairs = collect(cache2)
                @test issubset(first.(cache2Pairs), Set(-5:5))
                @test issubset( last.(cache2Pairs), Set(1:10))
            end

            @testset "Concurrent mixed read/write stress test" begin
                cap = max(64, 4 * n)
                keyMax = 2 * cap
                niter = 1000

                flag1 = true
                flag2 = true

                #> Repeat the test to reduce the chance of getting false-positive results
                for _ in 1:100
                    cache = genSafeCache(cap)
                    errorCount = Atomic(0)

                    @sync for id in 1:n
                        Threads.@spawn begin
                            rng = MersenneTwister(id)
                            for i in 1:niter
                                k = rand(rng, 1:keyMax)
                                if rand(rng) < 0.5
                                    cache[k] = id #> Write
                                else
                                    v = get(cache, k, 0) #> Read
                                    (0 <= v <= n) || (@atomic errorCount.val += 1)
                                end
                            end
                        end
                    end

                    #> No invalid values should have been observed
                    flag1 *= iszero(errorCount.val)

                    #> Capacity invariant under stress
                    flag2 *= (length(cache) <= 2cap - 1)
                end

                @test flag1
                @test flag2
            end
        end
    end
end

end