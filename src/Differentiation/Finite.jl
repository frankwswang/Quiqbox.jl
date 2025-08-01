# Fornberg's algorithm
## [DOI] 10.1090/S0025-5718-1988-0935077-0
function computeFiniteDiffWeights(order::Int, dis::AbstractVector{T}) where {T<:Real}
    checkPositivity(order, true)
    nGrid = length(dis)
    if order >= nGrid
        throw(AssertionError("The number of the interpolation points `nGrid=$nGrid` "*
                             "must be larger than the order of derivative `order=$order`."))
    end

    c1 = one(T)
    nm = zeros(T, nGrid, order+1)
    nm[begin] = one(T)
    iz = firstindex(dis)
    ir, ic = first.(Base.axes(nm))

    @inbounds for n in 1:(nGrid-1)
        c2 = one(T)
        idxRange = (min(n, order) + ic):-1:(1 + ic)
        ptL = dis[iz+n-1]
        ptR = dis[iz+n]
        p = ir + n

        for v in 0:n-1
            c3 = ptR - dis[iz+v]
            c2 *= c3
            ratio = c1 / c2

            if v == n-1
                for idx in idxRange
                    m = idx - ic
                    nm[p, idx] = ratio * (m * nm[p-1, idx-1] - ptL * nm[p-1, idx])
                end
                nm[p, ic] = -ratio * ptL * nm[p-1, ic]
            end

            q = ir + v
            for idx in idxRange
                m = idx - ic
                nm[q, ic+m] = (ptR * nm[q, idx] - m * nm[q, idx-1]) / c3
            end
            nm[q, ic] *= ptR / c3
        end
        c1 = c2
    end

    res = @view nm[:, end]
    coeffSum = sum(res)
    if order > 0 && !iszero(coeffSum)
        _, idx = findmin(abs, dis)
        res[begin+idx-firstindex(dis)] -= coeffSum
    end

    genMemory(res)
end

function computeFiniteDiffWeights(::Count{M}, sRange::SymmetricIntRange{S}) where {M, S}
    dis = Memory{Rational{Int}}(sRange())
    computeFiniteDiffWeights(M, dis)
end


@generated function getFiniteDiffWeightsCore(::Count{M}, ::SymmetricIntRange{S}
                                             ) where {M, S}
    intGradWeights = computeFiniteDiffWeights(Count(M), SymmetricIntRange( Count(S) ))
    return quote
        copy($intGradWeights)
    end
end


function getInterpolationNumber(diffOrder::Int, accuOrder::Int)
    checkPositivity(diffOrder)
    checkPositivity(accuOrder)
    iseven(accuOrder) || throw(AssertionError("`accuOrder` must be an even number."))
    accuOrder + diffOrder + isodd(diffOrder) - 1
end


function getFiniteDiffAccuOrder(diffOrder::Int, interpNum::Int)
    checkPositivity(diffOrder)
    checkPositivity(interpNum)
    isodd(interpNum) || throw(AssertionError("`interpNum` must be an even number."))
    interpNum + 1 - isodd(diffOrder) - diffOrder
end


@generated function getFiniteDiffWeightsINTERNAL(::Type{T}, ::Count{M}, 
                                                 ::SymmetricIntRange{S}
                                                 ) where {T<:AbstractFloat, M, S}
    sRange = SymmetricIntRange(Count(S))
    weights = getFiniteDiffWeightsCore(Count(M), sRange)
    accuOrder = getFiniteDiffAccuOrder(M, 2S+1)
    spacing = 2eps(T)^inv(1 + accuOrder)
    points = Memory{T}(spacing * sRange())
    scaledWeights = weights ./ spacing^M
    return :($points, $scaledWeights)
end

@generated function getFiniteDiffWeightsINTERNAL(::Type{<:Integer}, ::Count{M}, 
                                                 ::SymmetricIntRange{S}) where {M, S}
    sRange = SymmetricIntRange(Count(S))
    weights = getFiniteDiffWeightsCore(Count(M), sRange)
    points = Memory{T}(sRange())
    return :($points, $weights)
end

function getFiniteDiffWeights(::Type{T}, ::Count{M}, ::SymmetricIntRange{S}) where 
                             {T<:Real, M, S}
    points, weights = getFiniteDiffWeightsINTERNAL(T, Count(M), SymmetricIntRange( Count(S) ))
    copy(points), copy(weights)
end


#M: Order of derivative
#N: Order of finite difference accuracy
struct AxialFiniteDiff{C<:RealOrComplex, D, M, N, F<:Function} <: TypedEvaluator{C}
    f::TypedCarteFunc{C, D, F}
    axis::OneToIndex

    function AxialFiniteDiff(f::TypedCarteFunc{C, D, F}, ::Count{M}, axis::Int, 
                             ::Count{N}=Count(4)) where {C<:RealOrComplex, D, M, F, N}
        checkPositivity(N)
        iseven(N) || throw(AssertionError("`N` must be an even number."))
        new{C, D, M, N, F}(f, OneToIndex(axis))
    end
end

function (f::AxialFiniteDiff{C, D, M, N})(coord::NumberSequence{<:Real}) where 
                                         {T<:Real, C<:RealOrComplex{T}, D, M, N}
    gridRange = SymmetricIntRange(Count(getInterpolationNumber(M, N) ÷ 2))
    points, weights = getFiniteDiffWeightsINTERNAL(T, Count(M), gridRange)
    res = zero(C)
    for (point, weight) in zip(points, weights)
        shiftedCoord = indexedPerturb(+, coord, f.axis=>point)
        res = StableAdd(C)(res, convert(C, weight) * f.f(shiftedCoord))
    end
    res
end

getOutputType(::Type{<:AxialFiniteDiff{C}}) where {C<:RealOrComplex} = C