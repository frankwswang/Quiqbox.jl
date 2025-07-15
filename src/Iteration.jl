symmetric2DArrEleNum(n::Int) = n * (n + 1) ÷ 2
symmetric4DArrEleNum(n::Int) = symmetric2DArrEleNum(symmetric2DArrEleNum(n))


# {for j in 1:N, i=1:j} <=> 
# {for n in 1:N(N+1)/2}
function convertIndex1DtoTri2D(n::Int)
    # Using (7 - 0.1) instead of 7 to avoid potential round-off errors.
    j = (floor(Int, (sqrt(8.0n - 6.9) - 1.0)) >> 1) + 1
    i = n - j * (j-1) ÷ 2
    i, j
end

# {for l in 1:N, k in 1:l, j in 1:l, i in 1:ifelse(l==j, k, j)} <=> 
# {for n in (M=N(N+1)÷2; M(M+1)÷2)}
function convertIndex1DtoTri4D(n::Int)
    p, q = convertIndex1DtoTri2D(n)
    i, j = convertIndex1DtoTri2D(p)
    k, l = convertIndex1DtoTri2D(q)
    (i, j, k, l)
end


function sortTensorIndex((i, j)::NTuple{2, Int})
    if i > j
        (j, i)
    else
        (i, j)
    end
end

function sortTensorIndex((i, j, k, l)::NTuple{4, Int})
    pL = sortTensorIndex((i, j))
    pR = sortTensorIndex((k, l))
    if last(pL) > last(pR)
        (pR, pL)
    else
        (pL, pR)
    end
end


function rightCircShift(tpl::NonEmptyTuple)
    body..., tail = tpl
    (tail, body...)
end


function checkAxialIndexStep(arr::AbstractArray, dim::MissingOr{Int}=missing)
    axialIdxRanges = ismissing(dim) ? axes(arr) : (axes(arr, dim),)

    for (i, unitRange) in enumerate(axialIdxRanges)
        if !isequal(oneunit(unitRange|>eltype), 1)
            throw(AssertionError("The indexing step of `arr` at axis $i is not one."))
        end
    end

    nothing
end


struct SymmetricIntRange{S} <: CustomRange

    SymmetricIntRange(::Count{S}) where {S} = new{S}()
end

(::SymmetricIntRange{S})() where {S} = -Int(S) : Int(S)


struct OneToRange <: CustomRange
    endpoint::Int

    function OneToRange(endpoint::Int)
        checkPositivity(endpoint, true)
        new(endpoint)
    end

    function OneToRange(idx::OneToIndex)
        new(idx.idx)
    end
end

function iterate(otr::OneToRange)
    if otr.endpoint == 0
        nothing
    else
        OneToIndex(), OneToIndex(2)
    end
end

function iterate(otr::OneToRange, state::OneToIndex)
    if state.idx > otr.endpoint
        nothing
    else
        state, OneToIndex(state, Count(1))
    end
end

length(otr::OneToRange) = otr.endpoint

eltype(::OneToRange) = OneToIndex


function shiftLinearIndex(arr::AbstractArray, oneToIdx::Int)
    LinearIndices(arr)[begin + oneToIdx - 1]
end

function shiftLinearIndex(arr::Union{Tuple, NamedTuple}, oneToIdx::Int)
    eachindex(arr)[begin + oneToIdx - 1]
end

function shiftLinearIndex(arr::GeneralCollection, uRange::UnitRange{Int})
    offset = shiftLinearIndex(arr, 1) - 1
    (first(uRange) + offset) : (last(uRange) + offset)
end

shiftLinearIndex(arr::GeneralCollection, i::OneToIndex) = shiftLinearIndex(arr, i.idx)


function shiftAxialIndex(arr::AbstractArray, oneToIdx::Int, dim::Int)
    LinearIndices(axes(arr, dim))[begin + oneToIdx - 1]
end

shiftAxialIndex(arr::AbstractArray, i::OneToIndex, dim::Int) =
    shiftAxialIndex(arr, i.idx, dim)

function shiftAxialIndex(arr::AbstractArray{<:Any, N}, oneToIdx::Int) where {N}
    ntuple(Val(N)) do dim
        LinearIndices(axes(arr, dim))[begin + oneToIdx - 1]
    end
end

shiftAxialIndex(arr::AbstractArray, i::OneToIndex) = shiftAxialIndex(arr, i.idx)

function shiftAxialIndex(arr::AbstractArray{<:Any, N}, oneToIds::NTuple{N, Int}) where {N}
    ntuple(Val(N)) do dim
        oneToIdx = getEntry(oneToIds, OneToIndex(dim))
        LinearIndices(axes(arr, dim))[begin + oneToIdx - 1]
    end
end

shiftAxialIndex(arr::AbstractArray{<:Any, N}, ids::NTuple{N, OneToIndex}) where {N} =
shiftAxialIndex(arr, getfield.(ids, :idx))

shiftAxialIndex(::AbstractArray{<:Any, 0}, ::Tuple{}) = ()