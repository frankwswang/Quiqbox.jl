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