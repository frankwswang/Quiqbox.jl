using SpecialFunctions: gamma as getGamma, loggamma as getLogGamma, 
                        gamma_inc as getIncompleteGammaPair, erf

#> Reference(s): 
## [DOI] 10.1023/B:JOMC.0000044226.49921.f5

function directComputeBoysFunc(x::T, n::Int) where {T<:AbstractFloat}
    xLocal = eps(T) > eps(Float64) ? Float64(x) : x
    xType = typeof(xLocal)
    z = Int64(n) + 1//2
    p, _ = getIncompleteGammaPair(z, xLocal)
    if iszero(p)
        zero(xType)
    else
        res = if z < 23
                part1 = xType(getGamma(z) * p)
                part2 = xLocal^z
                isinf(part2) ? exp(log(part1) - z*log(xLocal)) : part1 / part2
            else
                exp(getLogGamma(z|>xType) + log(p|>xType) - z*log(xLocal))
        end
        res / 2
    end
end


function initializeUpperOrder(x::T, n::Int) where {T<:AbstractFloat}
    d = getAtolDigits(T)
    res = (Int∘ceil)(d / (abs∘log)(10, ifelse(n==x, T(n), n/x)) + n)
    res + isodd(res)
end

function recursiveGetBoysFuncDn(x::T, orderAndVal::Tuple{Int, T}, 
                                n::Int=first(orderAndVal)-1) where {T<:AbstractFloat}
    m, res = orderAndVal
    for i in (m - 1) : -1 : n
        res = (2x * res + exp(-x)) / muladd(2, i, 1)
    end
    res
end

function ComputeBoysOrderN(x::T, n::Int) where {T<:AbstractFloat}
    if x < getAtolVal(T)
        (T∘inv∘muladd)(2, n, 1)
    else
        nUpper = min(initializeUpperOrder(x, n), 6n)
        fUpper = directComputeBoysFunc(x, nUpper)
        recursiveGetBoysFuncDn(typeof(fUpper)(x), (nUpper, fUpper), n)
    end
end

function computeBoysOrder0(x::T) where {T<:AbstractFloat}
    if x < getAtolVal(T)
        one(T)
    else
        xRoot = sqrt(x)
        T(PowersOfPi[:p0d5]) * erf(xRoot) / (2xRoot)
    end
end


function computeBoysFunc(x::T, n::Int=0) where {T<:AbstractFloat} #> x >= 0, n >= 0
    iszero(n) ? computeBoysOrder0(x) : ComputeBoysOrderN(x, n)
end


function computeBoysSequence(x::T, n::Int, cache!Self::Memory{T}=Memory{T}(undef, n+1)
                             ) where {T<:AbstractFloat}
    iStart = firstindex(cache!Self)
    cache!Self[iStart] = computeBoysFunc(x)
    n > 0 && (cache!Self[iStart+n] = computeBoysFunc(x, n))
    for m in n:-1:2
        i = iStart + m
        @inbounds cache!Self[i-1] = recursiveGetBoysFuncDn(x, (m, cache!Self[i]))
    end
    cache!Self
end