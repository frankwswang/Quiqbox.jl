using LinearAlgebra: norm

function oddFactorialCore(a::Int) # a * (a-2) * ... * 1
    factorial(2a) ÷ (2^a * factorial(a))
end


function polyGaussFuncSquaredNorm(α::T, degree::Int) where {T<:Real}
    factor = degree > 0 ? (oddFactorialCore(2degree - 1) / (4α)^degree) : one(T)
    T(πPowers[:p0d5]) / sqrt(2α) * factor
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

function (cPair::CenPair{T})(xPair::XpnPair{T}) where {T}
    (xPair.left .* cPair.left .+ xPair.right .* cPair.right) ./ xPair.sum
end


struct AngPair{D}
    left::NTuple{D, Int}
    right::NTuple{D, Int}
    sum::NTuple{D, Int}

    AngPair(l::NonEmptyTuple{Int, D}, r::NonEmptyTuple{Int, D}) where {D} = 
    new{D+1}(l, r, l .+ r)
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
    res = zero(T)
    for q in lb:2:ub
        i = (lx + q) ÷ 2
        j = (lx - q) ÷ 2
        res += binomial(lx1,  i) * binomial(lx2,  j) * (x - x1)^(lx1 -i) * (x - x2)^(lx2 -j)
    end
    res
end


function overlapPGTO(cPair::CenPair{T, D}, xPair::XpnPair{T}, 
                     aPair::AngPair{D}) where {T, D}
    α = xPair.sum
    mapreduce(*, cPair.left, cPair.right, cPair(xPair), aPair.left, aPair.right, 
                 aPair.sum) do x1, x2, x, i1, i2, i
        mapreduce(+, 0:(i÷2)) do j
            gaussProdCore2(x1, x2, x, i1, i2, 2j) * polyGaussFuncSquaredNorm(α/2, j)
        end
    end * gaussProdCore1(cPair, xPair)
end


function genNormalizer(o::PrimGTOcore{T, D}, paramPtr::PrimOrbParamPtr{T, D}) where {T, D}
    angularFunc = getAngularFunc(o)
    ns = map(x->Base.Fix2(polyGaussFuncSquaredNorm, x), angularFunc.f.m.tuple)
    nCore = OnlyBody( AbsSqrtInv ∘ ChainReduce(StableBinary(*, T), VectorMemory(ns)) )
    ptrTuple = (getXpnPtr(paramPtr.body),)
    ParamFilterFunc(ParamSelectFunc(nCore, ptrTuple), paramPtr.scope)
end


function getXpnPtr(paramPtr::MixedFieldParamPointer{T}) where {T}
    xpnPtr = ChainPointer((:radial, :xpn), TensorType(T))
    paramPtr.core[xpnPtr]
end

function getXpnPtr(paramPtr::MixedFieldParamPointer{T, <:SingleEntryDict}) where {T}
    paramPtr.core.value
end

function getAngularFunc(o::PrimGTOcore)
    o.f.apply.f.right.f
end


function preparePGTOparam!(cache::DimSpanDataCacheBox{T}, o::PrimGTOcore{T, D}, 
                           s::TypedParamInput{T}, p::PrimOrbParamPtr{T, D}) where {T, D}
    # @show p.scope
    sLocal = FilteredObject(s, p.scope)
    cen = map(p.center) do c
        cacheParam!(cache, sLocal, c)
    end
    xpn = cacheParam!(cache, sLocal, getXpnPtr(p.body))
    ang = getAngularFunc(o).f.m.tuple
    cen, xpn, ang
end


function getOverlapCore!(cache::DimSpanDataCacheBox{T}, 
                         (o,)::Tuple{PrimGTOcore{T, D}}, 
                         (s,)::Tuple{TypedParamInput{T}}, 
                         (p,)::Tuple{PrimOrbParamPtr{T, D}}) where {T, D}
    cen, xpn, ang = preparePGTOparam!(cache, o, s, p)
    overlapPGTO(CenPair(cen, cen), XpnPair(xpn, xpn), AngPair(ang, ang))
end

function getOverlapCore!(cache::DimSpanDataCacheBox{T}, 
                         o::NTuple{2, PrimGTOcore{T, D}}, 
                         s::NTuple{2, TypedParamInput{T}}, 
                         p::NTuple{2, PrimOrbParamPtr{T, D}}) where {T, D}
    (cen1, xpn1, ang1), (cen2, xpn2, ang2) = map(o, s, p) do orb, pSet, pPtr
        preparePGTOparam!(cache, orb, pSet, pPtr)
    end
    overlapPGTO(CenPair(cen1, cen2), XpnPair(xpn1, xpn2), AngPair(ang1, ang2))
end