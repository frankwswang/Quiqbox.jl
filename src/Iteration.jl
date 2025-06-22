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


struct Count{N} <: StructuredType

    function Count{N}() where {N}
        checkPositivity(N::Int, true)
        new{N}()
    end
end

Count(N::Integer) = Count{Int(N)}()

const Nil = Count{0}
const One = Count{1}


struct SymmetricIntRange{S} <: ConfigBox

    SymmetricIntRange(::Count{S}) where {S} = new{S}()
end

(::SymmetricIntRange{S})() where {S} = -Int(S) : Int(S)