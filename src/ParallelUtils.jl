function getIdxChunks(nPoints)
    chunkSize = max(1, fld1(nPoints, max(Threads.nthreads(), 1)))
    Iterators.partition(1:nPoints, chunkSize)
end

adaptiveView(a::AbstractArray, idx) = view(a, idx)
adaptiveView(a::Tuple, idx) = getindex(a, idx)

adaptiveCat(a::AbstractVector{T}, b::AbstractVector{T}) where {T} = vcat(a, b)
adaptiveCat(a::Tuple, b::Tuple) = (a..., b...)

function mapMT(f::F, v::T; minLengthForMT::Int=2Threads.nthreads()) where {F, T}
    if length(v) < max(minLengthForMT, 1)
        map(f, v)
    else
        idxChunks=(getIdxChunks∘length)(v)
        tasks = map(idxChunks) do chunk
            Threads.@spawn map(f, adaptiveView(v, chunk))
        end
        reduce(adaptiveCat, fetch.(tasks))
    end
end

mutable struct Locker{T}
    @atomic val::T
end

getindex(l::Locker) = l.val

const ValuePointer{T} = Union{AbtArray0D{T}, RefVal{T}}

function safelySetVal!(box::Locker{T}, val::T) where {T}
    @atomic box.val = val
end

function safelySetVal!(box::Base.Threads.Atomic{T}, val::T) where {T}
    Threads.atomic_xchg!(box, val)
    val
end

function safelySetVal!(box::B, val::T) where {T, B<:ValuePointer{T}}
    lk = ReentrantLock()
    lock(lk) do
        box[] = val
    end
    val
end

function safelySetVal!(box::AbstractArray{T}, val::Union{T, AbstractArray{T}}) where {T}
    lk = ReentrantLock()
    lock(lk) do
        box .= val
    end
    box
end

function safelyAddVal!(box::Locker{T}, val::T) where {T}
    @atomic box.val += val
end

function safelyAddVal!(box::Base.Threads.Atomic{T}, val::T) where {T}
    Threads.atomic_add!(box, val)
    box[]
end

function safelyAddVal!(box::B, val::T) where {T, B<:ValuePointer{T}}
    lk = ReentrantLock()
    lock(lk) do
        box[] += val
    end
    box[]
end

function safelySubVal!(box::Locker{T}, val::T) where {T}
    @atomic box.val -= val
end

function safelySubVal!(box::Base.Threads.Atomic{T}, val::T) where {T}
    Threads.atomic_sub!(box, val)
    box[]
end

function safelySubVal!(box::B, val::T) where {T, B<:ValuePointer{T}}
    lk = ReentrantLock()
    lock(lk) do
        box[] -= val
    end
    box[]
end