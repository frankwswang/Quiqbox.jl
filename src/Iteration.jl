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


function getLinearFirstIndex(arr::AbstractArray)
    LinearIndices(arr) |> first
end

function getLinearFirstIndex(tpl::GeneralTupleUnion)
    firstindex(tpl)
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


struct Count{N} <: StructuredType

    function Count{N}() where {N}
        checkPositivity(N::Int, true)
        new{N}()
    end
end

const Nil = Count{0}
const One = Count{1}