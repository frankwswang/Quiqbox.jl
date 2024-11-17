using LinearAlgebra: norm

function oddFactorialCore(a::Int) # a * (a-2) * ... * 1
    factorial(2a) ÷ (2^a * factorial(a))
end


function polyGaussFuncSquaredNorm(α::T, degree::Int) where {T<:Real}
    factor = degree > 0 ? (oddFactorialCore(2degree - 1) / (4α)^degree) : one(T)
    T(πPowers[:p0d5]) / sqrt(2α) * factor
end


function concentricPolyGFOverlap(αLR::NTuple{2, T}, iLR::NTuple{2, Int}) where {T<:Real}
    iSum = sum(iLR)
    isodd(iSum) ? T(0) : polyGaussFuncSquaredNorm(sum(αLR)/2, iSum÷2)
end


struct XpnPair{T<:Real}
    left::T
    right::T
    sum::T
    prod::T

    function XpnPair(l::T, r::T) where {T}
        checkPositivity(l)
        checkPositivity(r)
        new{T}(l, r, l+r, l*r)
    end
end


struct CenPair{T<:Real, D}
    left::NTuple{D, T}
    right::NTuple{D, T}
    dist::T

    CenPair(l::NonEmptyTuple{T, D}, r::NonEmptyTuple{T, D}) where {T, D} = 
    new{T, D+1}(l, r, norm(l .- r))
end


struct AngPair{D}
    left::NTuple{D, Int}
    right::NTuple{D, Int}
    sum::NTuple{D, Int}

    AngPair(l::NonEmptyTuple{Int, D}, r::NonEmptyTuple{Int, D}) where {D} = 
    new{D+1}(l, r, l .+ r)
end

function (cPair::CenPair{T})(xPair::XpnPair{T}) where {T} # gaussProdCore2
    (xPair.left .* cPair.left .+ xPair.right .* cPair.right) ./ xPair.sum
end


function gaussProdCore1(cPair::CenPair{T}, xPair::XpnPair{T}) where {T<:Real}
    if iszero(cPair.dist)
        T(1)
    else
        exp(- xPair.prod / xPair.sum * cPair.dist^2)
    end
end

function gaussProdCore2(x1::T, x2::T, x::T, lx1::Int, lx2::Int, lx::Int) where {T<:Real}
    lb = max(-lx,  lx - 2lx2)
    ub = min( lx, 2lx1 - lx )
    map(lb:2:ub) do q
        i = (lx + q) ÷ 2
        j = (lx - q) ÷ 2
        binomial(lx1,  i) * binomial(lx2,  j) * (x - x1)^(lx1 -i) * (x - x2)^(lx2 -j)
    end |> sum
end


function overlapPGTO(cPair::CenPair{T, D}, xPair::XpnPair{T}, aPair::AngPair{D}) where {T, D}
    α = xPair.sum
    mapreduce(*, cPair.left, cPair.right, cPair(xPair), aPair.left, aPair.right, 
                 aPair.sum) do x1, x2, x, i1, i2, i
        mapreduce(+, 0:(i÷2)) do j
            gaussProdCore2(x1, x2, x, i1, i2, 2j) * polyGaussFuncSquaredNorm(α/2, j)
        end
    end * gaussProdCore1(cPair, xPair)
end


function contrFieldSumSquaredNorm(coeff::AbstractVector{<:Number}, 
                                  overlapMat::AbstractMatrix{<:Number})
    dot(coeff, overlapMat, coeff)
end

function contrCarteGTOSquaredNorm(coeff::Memory{T}, 
                                  αGroups::NonEmptyTuple{Memory{T}, D}, 
                                  degrees::NonEmptyTuple{Int, D}) where {T<:Real, D}
    factor = T(πPowers[:p0d5])^(D+1) * mapreduce(*, degrees) do degree
        degree > 0 ? (oddFactorialCore(2degree-1) / 2^degree) : one(T)
    end
    nCoeff = length(coeff)
    m = zeros(T, nCoeff, nCoeff)
    for i in 1:nCoeff, j in 1:i
        m[i,j] = m[j,i] = 
        (inv∘prod)((getindex.(αGroups, i) .+ getindex.(αGroups, j)) .^ (T(0.5) .+ degrees))
    end
    contrFieldSumSquaredNorm(coeff, m) * T(factor)
end


function getOverlap(o1::PrimGTO{T, D}, o2::PrimGTO{T, D}) where {T, D}
    cPair = CenPair(obtain.(o1.center), obtain.(o2.center))
    fgo1 = o1.body
    fgo2 = o2.body
    xPair = XpnPair(obtain(fgo1.radial.xpn), obtain(fgo2.radial.xpn))
    aPair = AngPair(fgo1.angular.m.tuple, fgo2.angular.m.tuple)
    overlapPGTO(cPair, xPair, aPair)
end

function getOverlap(o1::CompGTO{T, D}, o2::CompGTO{T, D}) where {T, D}
    res = zero(T)
    w1 = obtain(o1.weight)
    w2 = obtain(o2.weight)
    for (i, u) in zip(o1.basis, w1), (j, v) in zip(o2.basis, w2)
        res += getOverlap(i, j) * u * v
    end
    res
end

function genOverlap(o::PolyGaussProd)
    ns = map(x->Base.Fix2(polyGaussFuncSquaredNorm, x), o.angular.m.tuple)
    ChainReduce(*, VectorMemory(ns)), (o.radial.xpn,)
end

function genOverlap(o::CompGTO)
    ns = map(x->Base.Fix2(polyGaussFuncSquaredNorm, x), o.angular.m)
    ChainReduce(*, VectorMemory(ns)), (o.radial.xpn,)
end