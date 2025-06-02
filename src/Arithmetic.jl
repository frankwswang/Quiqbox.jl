#> Precision reflection on floating-number system
"""

    getAtolDigits(::Type{T}) where {T<:Real} -> Int

Return the maximal number of digits kept after rounding of the input real number type `T`.
"""
function getAtolDigits(::Type{T}) where {T<:Real}
    val = log10(T|>numEps)
    res = max(0, -val) |> floor
    if res > typemax(Int)
        throw(DomainError(res, "This value is too large to be converted to Int."))
    else
        Int(res)
    end
end

function getAtolDigits(num::Real)
    if isnan(num)
        getAtolDigits(num|>typeof)
    else
        str = string(num)
        idx1 = findlast('e', str)
        if idx1 === nothing
            idx2 = findlast('.', str)
            length(str) - idx2
        elseif str[idx1+1] == '-'
            parse(Int, str[idx1+2:end])
        else
            0
        end
    end
end


"""

    getAtolVal(::Type{T}) where {T<:Real} -> Real

Return the absolute precision tolerance of the input real number type `T`.
"""
getAtolVal(::Type{T}) where {T<:Real} = ceil(3numEps(T)/2, sigdigits=1)
getAtolVal(::Type{T}) where {T<:Integer} = one(T)


function roundToMultiOfStep(num::T, step::T) where {T<:Real}
    if iszero(step) || isnan(step)
        num
    else
        invStep = inv(step)
        round(num * invStep, RoundNearest) / invStep
    end
end

function roundToMultiOfStep(num::T, step::T) where {T<:Integer}
    iszero(step) ? num : round(T, num/step) * step
end

roundToMultiOfStep(nums::AbstractArray{T}, step::T) where {T} = 
roundToMultiOfStep.(nums, step)


nearestHalfOf(val::T) where {T<:Real} = 
roundToMultiOfStep(val/2, floor(numEps(T), sigdigits=1))

nearestHalfOf(val::Integer) = itself(val)


getNearestMid(num1::T, num2::T, atol::T) where {T} = 
num1==num2 ? num1 : roundToMultiOfStep((num1+num2)/2, atol)


function unsafeOddFactorial(a::T) where {T<:Integer}
    res = one(T)
    for i in 1:2:a
        res *= i
    end
    res
end

const DefaultOddFactorialCacheSizeLimit = 50
const OddFactorialCache = LRU{Int, BigInt}(maxsize=DefaultOddFactorialCacheSizeLimit)

function oddFactorialBigInt(a::Int) # a * (a-2) * ... * 1
    res = get(OddFactorialCache, a, nothing)
    if res === nothing
        res = unsafeOddFactorial(a|>big)
        OddFactorialCache[a] = res
    end
    res
end

function oddFactorial(a::Int, ::Type{T}=Int128) where {T<:Real}
    if a < 55 # Bound for result to be within typemax(Int128)
        unsafeOddFactorial(a|>Int128)
    else
        oddFactorialBigInt(a)
    end |> T
end

function oddFactorial(a::Int, coeff::T) where {T<:AbstractFloat}
    res = one(T)
    for i in 1:2:a
        res *= i * coeff
    end
    res
end


function computeGaussProd(dxML::T, dxMR::T, lxL::Int, lxR::Int, lx::Int) where {T<:Real}
    lb = max(-lx,  lx - 2lxR)
    ub = min( lx, 2lxL - lx )
    res = zero(T)
    for q in lb:2:ub
        i = (lx + q) >> 1
        j = (lx - q) >> 1
        res += binomial(lxL, i) * binomial(lxR, j) * dxML^(lxL - i) * dxMR^(lxR - j)
    end
    res
end

function computeGaussProd(lxL::Int, lxR::Int, lx::Int)
    lb = max(-lx,  lx - 2lxR)
    ub = min( lx, 2lxL - lx )
    res = zero(Int)
    for q in lb:2:ub
        res += Int(isequal(2lxL, lx+q) * isequal(2lxR, lx-q))
    end
    res
end