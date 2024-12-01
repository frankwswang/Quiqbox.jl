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
    res = zero(T)
    for q in lb:2:ub
        i = (lx + q) ÷ 2
        j = (lx - q) ÷ 2
        res += binomial(lx1,  i) * binomial(lx2,  j) * (x - x1)^(lx1 -i) * (x - x2)^(lx2 -j)
    end
    res
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


function genNormalizer(o::PrimGTOcore{T, D}, paramPtr::PrimOrbParamPtr{T, D}) where {T, D}
    nCore = (OnlyBody∘genGTOnormalizer∘getAngularFunc)(o)
    ptrTuple = (getXpnPtr(paramPtr.body),)
    PointerFunc(nCore, ptrTuple, paramPtr.sourceID)
end

function genNormalizer(o::PrimitiveOrbCore{T, D}, 
                       paramPtr::PrimOrbParamPtr{T, D}) where {T, D}
    ptrTuple = (Tuple∘values)(paramPtr.body.core)
    fInner = AbsSqrtInv ∘ (OnlyBody∘LeftPartial)(numericalOneBodyInt, Identity(), o.f.apply)
    fCore = function (input, args...)
        fInner(input, buildDict(ptrTuple .=> args))
    end
    PointerFunc(fCore, ptrTuple, paramPtr.sourceID)
end

function getNormCoeff!(cache::DimSpanDataCacheBox{T}, 
                       orbs::NonEmptyTuple{FrameworkOrb{T, D}}) where {T, D} # Optimal
    ptr = ChainPointer((:core, :f, :right))
    paramSets = map(Base.Fix2(getfield, :param), orbs) # better than broadcasting
    normalizers = map(Base.Fix2(getField, ptr), orbs) # better than broadcasting
    mapreduce(StableBinary(*, T), paramSets, normalizers, init=one(T)) do pSet, f
        getNormCoeffCore!(cache, pSet, f)
    end
end

function getNormCoeff!(cache::DimSpanDataCacheBox{T}, orb::FrameworkOrb{T, D}, 
                       degree::Int) where {T, D}
    ptr = ChainPointer((:core, :f, :right))
    normalizer = getField(orb, ptr)
    normCoeff = getNormCoeffCore!(cache, orb.param, normalizer)
    normCoeff^abs(degree)
end

function getNormCoeffCore!(cache::DimSpanDataCacheBox{T}, pSet::FlatParamSet{T}, 
                           normalizer::ReturnTyped{T, <:PointerFunc}) where {T}
    ptrs = normalizer.f.pointer
    pDict = if isempty(ptrs)
        buildDict()
    else
        map(ptrs) do ptr
            ptr => cacheParam!(cache, pSet, ptr)
        end |> buildDict
    end
    normalizer(nothing, pDict)
end

function getNormCoeffCore!(::DimSpanDataCacheBox{T}, ::FlatParamSet{T}, 
                           normalizer::Storage{T}) where {T}
    normalizer.val
end

function overlap(o1::ComposedOrb{T, D}, o2::ComposedOrb{T, D}) where {T, D}
    fo1 = FrameworkOrb(o1)
    o1 === o2 ? overlap(fo1) : overlap(fo1, FrameworkOrb(o2))
end

function overlap(o::FrameworkOrb{T, D}) where {T, D}
    cache = DimSpanDataCacheBox(T)
    res = getOverlapCore!(cache, o.core.f.left, o.param, o.pointer)
    StableBinary(*, T)(res, getNormCoeff!(cache, o, 2))
end

function overlap(o1::FrameworkOrb{T, D}, o2::FrameworkOrb{T, D}) where {T, D}
    if o1 === o2
        overlap(o1)
    else
        cache = DimSpanDataCacheBox(T)
        res = getOverlapCore!(cache, (o1.core.f.left,    o2.core.f.left), 
                                     (o1.param,          o2.param), 
                                     (o1.pointer,        o2.pointer) )
        StableBinary(*, T)(res, getNormCoeff!( cache, (o1, o2) ))
    end
end

const AbsSqrtInv = inv∘sqrt∘abs

function genGTOnormalizer(angularFunc::ReturnTyped{T, <:CartSHarmonics{D}}) where {T, D}
    ns = map(x->Base.Fix2(polyGaussFuncSquaredNorm, x), angularFunc.f.m.tuple)
    AbsSqrtInv ∘ ChainReduce(StableBinary(*, T), VectorMemory(ns))
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

function getOverlapCore!(cache::DimSpanDataCacheBox{T}, o::PrimGTOcore{T, D}, 
                         s::FlatParamSet{T}, p::PrimOrbParamPtr{T, D}) where {T, D}
    overlapPGTO(preparePGTOparam!(cache, o, s, p)...)
end

function getOverlapCore!(cache::DimSpanDataCacheBox{T}, 
                         (o1, o2)::NTuple{2, PrimGTOcore{T, D}}, 
                         (s1, s2)::NTuple{2, FlatParamSet{T}}, 
                         (p1, p2)::NTuple{2, PrimOrbParamPtr{T, D}}) where {T, D}
    overlapPGTO(preparePGTOparam!(cache, o1, s1, p1, o2, s2, p2)...)
end

function getOverlapCore!(cache::DimSpanDataCacheBox{T}, o::CompositeOrbCore{T, D}, 
                        s::FlatParamSet{T}, p::CompOrbParamPtr{T, D}) where {T, D}
    res = zero(T)
    # w = evalField(s, p.weight)
    w = cacheParam!(cache, s, p.weight)
    primOrbs = o.f.chain
    for (i, m, u) in zip(primOrbs, p.basis, w), (j, n, v) in zip(primOrbs, p.basis, w)
        o1core = i.f.left.f.left
        o2core = j.f.left.f.left
        temp = if i === j && m === n
            getOverlapCore!(cache, o1core, s, m)
        else
            getOverlapCore!(cache, (o1core, o2core), (s, s), (m, n))
        end
        res += temp * u * v
    end
    res
end

function getOverlapCore!(cache::DimSpanDataCacheBox{T}, 
                        (o1, o2)::NTuple{2, CompositeOrbCore{T, D}}, 
                        (s1, s2)::NTuple{2, FlatParamSet{T}}, 
                        (p1, p2)::NTuple{2, CompOrbParamPtr{T, D}}) where {T, D}
    res = zero(T)
    pSetBool = (s1 === s2)
    # w1 = evalField(s1, p1.weight)
    # w2 = evalField(s2, p2.weight)
    w1 = cacheParam!(cache, s1, p1.weight)
    w2 = cacheParam!(cache, s2, p2.weight)
    for (i, m, u) in zip(o1.f.chain, p1.basis, w1), 
        (j, n, v) in zip(o2.f.chain, p2.basis, w2)
        o1core = i.f.left.f.left
        o2core = j.f.left.f.left
        temp = if pSetBool && i === j && m === n
            getOverlapCore!(cache, o1core, s1, m)
        else
            getOverlapCore!(cache, (o1core, o2core), (s1, s2), (m, n))
        end
        res += temp * u * v
    end
    res
end

#! Implement memoization of parameter value
function preparePGTOparamCore!(cache::DimSpanDataCacheBox{T}, 
                               o::PrimGTOcore{T, D}, s::FlatParamSet{T}, 
                               p::PrimOrbParamPtr{T, D}) where {T, D}
    cen = map(p.center) do c
        cacheParam!(cache, s, c)
    end
    # cen = evalField.(Ref(s), p.center)
    xpn = cacheParam!(cache, s, getXpnPtr(p.body))
    # xpn = evalField(s, getXpnPtr(p.body))
    ang = getAngularFunc(o).f.m.tuple
    cen, xpn, ang
end

function preparePGTOparam!(cache::DimSpanDataCacheBox{T}, o1::PrimGTOcore{T, D}, 
                           s1::FlatParamSet{T}, p1::PrimOrbParamPtr{T, D}) where {T, D}
    cen1, xpn1, ang1 = preparePGTOparamCore!(cache, o1, s1, p1)
    CenPair(cen1, cen1), XpnPair(xpn1, xpn1), AngPair(ang1, ang1)
end

function preparePGTOparam!(cache::DimSpanDataCacheBox{T}, 
                           o1::PrimGTOcore{T, D}, s1::FlatParamSet{T}, 
                           p1::PrimOrbParamPtr{T, D}, 
                           o2::PrimGTOcore{T, D}, s2::FlatParamSet{T}, 
                           p2::PrimOrbParamPtr{T, D}) where {T, D}
    cen1, xpn1, ang1 = preparePGTOparamCore!(cache, o1, s1, p1)
    cen2, xpn2, ang2 = preparePGTOparamCore!(cache, o2, s2, p2)
    CenPair(cen1, cen2), XpnPair(xpn1, xpn2), AngPair(ang1, ang2)
end


function getOverlapCore!(cache::DimSpanDataCacheBox{T}, o::PrimitiveOrbCore{T, D}, 
                         s::FlatParamSet{T}, p::PrimOrbParamPtr{T, D}) where {T, D}
    parDict = if isempty(p.body.core)
        buildDict()
    else
         map((collect∘keys)(p.body.core)) do ptr
            ptr => cacheParam!(cache, s, ptr)
        end |> buildDict
    end
    # parDict = map(vcat( collect(p.center), (collect∘keys)(p.body.core) )) do ptr
    #     ptr => cacheParam!(cache, s, ptr)
    # end |> buildDict
    numericalOneBodyInt(Identity(), o.f.apply, parDict)
end