# Fornberg's algorithm
## [DOI] 10.1090/S0025-5718-1988-0935077-0

function computeFiniteDiffWeightsCore(order::Int, dis::AbstractVector{T}) where {T<:Real}
    checkPositivity(order+1)
    nGrid = length(dis)
    if order >= nGrid
        throw("The order of the derivative (order=$order) must be less than the number "*
              "of the interpolation points: $(nGrid).")
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

    getMemory(@view nm[:, end])
end

function computeFiniteDiffWeights(order::Int, dis::AbstractVector{T}) where {T<:Real}
    res = computeFiniteDiffWeightsCore(order, dis)
    coeffSum = sum(res)
    # Enforce the sum of the coefficients to be zero so that the finite difference 
    # of a constant function will always be zero.
    if order>0 && !iszero(coeffSum)
        _, idx = findmin(abs, dis)
        res[begin+idx-firstindex(dis)] -= coeffSum
    end

    res
end

struct SymmetricIntRange{N} <: ConfigBox

    function SymmetricIntRange(::Val{N}) where {N}
        checkPositivity(N::Int+1)
        new{N}()
    end
end

(::SymmetricIntRange{N})() where {N} = -N : N

@generated function getFiniteDiffWeightsCore(::Val{M}, ::SymmetricIntRange{N}) where {M, N}
    points = Memory{Rational{Int}}(SymmetricIntRange(Val(N))())
    intGradWeights = computeFiniteDiffWeights(M, points)
    return quote
        copy($intGradWeights)
    end
end


@generated function getFiniteDiffWeightsINTERNAL(::Type{T}, ::Val{M}, ::SymmetricIntRange{N}
                                                 ) where {T<:AbstractFloat, M, N}
    sRange = SymmetricIntRange(Val(N))
    weights = getFiniteDiffWeightsCore(Val(M), sRange)
    accuOrder = 2(N + 1) - isodd(M) - M
    spacing = 2eps(T)^inv(1 + accuOrder)
    points = Memory{T}(spacing * sRange())
    scaledWeights = weights ./ spacing^M
    return :($points, $scaledWeights)
end

@generated function getFiniteDiffWeightsINTERNAL(::Type{<:Integer}, ::Val{M}, 
                                                 ::SymmetricIntRange{N}) where {M, N}
    sRange = SymmetricIntRange(Val(N))
    weights = getFiniteDiffWeightsCore(Val(M), sRange)
    points = Memory{T}(sRange())
    return :($points, $weights)
end

function getFiniteDiffWeights(::Type{T}, ::Val{M}, ::SymmetricIntRange{N}) where 
                             {T<:Real, M, N}
    points, weights = getFiniteDiffWeightsINTERNAL(T, Val(M), SymmetricIntRange( Val(N) ))
    copy(points), copy(weights)
end