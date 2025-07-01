export Count

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