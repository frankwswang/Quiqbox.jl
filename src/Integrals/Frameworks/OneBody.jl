function genNormalizer(o::PrimitiveOrbCore{T, D}, ::PrimOrbParamPtr{T, D}) where {T, D}
    fCore = function (x::AbtVecOfAbtArr{T})
        (AbsSqrtInv∘numericalOneBodyInt)(Identity(), (o.f.apply,), (x,))
    end
    OnlyBody(fCore∘getField)
end


function getNormCoeff!(cache::DimSpanDataCacheBox{T}, 
                       orbs::NonEmptyTuple{FrameworkOrb{T, D}}) where {T, D}
    ptr = ChainPointer((:core, :f, :apply, :right))
    localParamSets = map(orbs) do orb # better than broadcasting
        FilteredObject(orb.param, orb.pointer.scope)
    end
    normalizers = map(Base.Fix2(getField, ptr), orbs) # better than broadcasting
    mapreduce(StableBinary(*, T), localParamSets, normalizers, init=one(T)) do pSet, f
        getNormCoeffCore!(cache, pSet, f)
    end
end

function getNormCoeff!(cache::DimSpanDataCacheBox{T}, orb::FrameworkOrb{T, D}, 
                       degree::Int) where {T, D}
    ptr = ChainPointer((:core, :f, :apply, :right, :f))
    normalizer = getField(orb, ptr)
    normCoeff = getNormCoeffCore!(cache, FilteredObject(orb.param, orb.pointer.scope), 
                                  normalizer)
    normCoeff^abs(degree)
end


function getNormCoeffCore!(cache::DimSpanDataCacheBox{T}, pSource::TypedParamInput{T}, 
                           normalizer::EvalOrbNormalizer{T}) where {T}
    paramVal = cacheParam!(cache, pSource)
    normalizer.f(paramVal)::T
end


function overlap(o1::ComposedOrb{T, D}, o2::ComposedOrb{T, D}; 
                 cache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where {T, D}
    fo1 = FrameworkOrb(o1)
    o1 === o2 ? overlap(fo1; cache) : overlap(fo1, FrameworkOrb(o2); cache)
end

function overlap(o::FrameworkOrb{T, D}; 
                 cache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where {T, D}
    res = getOverlapCore!(cache, (o.core.f.apply.left,), (o.param,), (o.pointer,))
    StableBinary(*, T)(res, getNormCoeff!(cache, o, 2))
end

function overlap(o1::FrameworkOrb{T, D}, o2::FrameworkOrb{T, D}; 
                 cache::DimSpanDataCacheBox{T}=DimSpanDataCacheBox(T)) where {T, D}
    if o1 === o2
        overlap(o1; cache)
    else
        res = getOverlapCore!(cache, (o1.core.f.apply.left,    o2.core.f.apply.left), 
                                     (o1.param,                o2.param), 
                                     (o1.pointer,              o2.pointer) )
        StableBinary(*, T)(res, getNormCoeff!( cache, (o1, o2) ))
    end
end


function getOverlapCore!(cache::DimSpanDataCacheBox{T}, 
                         (o,)::Tuple{PrimitiveOrbCore{T, D}}, 
                         (s,)::Tuple{TypedParamInput{T}}, 
                         (p,)::Tuple{PrimOrbParamPtr{T}}) where {T, D}
    evalCore = (o.f.apply,)
    parBlock = cacheParam!(cache, s, p.scope)
    numericalOneBodyInt(Identity(), evalCore, (parBlock,))
end

function getOverlapCore!(cache::DimSpanDataCacheBox{T}, 
                         (o1, o2)::NTuple{2, PrimitiveOrbCore{T, D}}, 
                         (s1, s2)::NTuple{2, TypedParamInput{T}}, 
                         (p1, p2)::NTuple{2, PrimOrbParamPtr{T}}) where {T, D}
    evalCore = (o1.f.apply, o2.f.apply)
    parBlock1 = cacheParam!(cache, s1, p1.scope)
    parBlock2 = cacheParam!(cache, s2, p2.scope)
    numericalOneBodyInt(Identity(), evalCore, (parBlock1, parBlock2))
end


function getIntegrandComponent(o::Memory{<:WeightedPF{T, D}}, 
                               p::Memory{<:PrimOrbParamPtr{T, D}}, 
                               w::ShapedMemory{T, 1}, idx::Int) where {T, D}
    (o[idx].left.f.apply.left, p[idx], w[idx])
end


function getOverlapCore!(cache::DimSpanDataCacheBox{T}, 
                         (o,)::Tuple{CompositeOrbCore{T, D}}, 
                         (s,)::Tuple{FlatParamSet{T}}, 
                         (p,)::Tuple{CompOrbParamPtr{T, D}}) where {T, D}
    res = zero(T)
    sL = FilteredObject(s, p.scope)
    w = cacheParam!(cache, sL, p.weight)
    primOrbs = o.f.chain
    for j in eachindex(primOrbs), i in 1:j
        o1c, ptr1, c1 = getIntegrandComponent(primOrbs, p.basis, w, i)
        if i == j
            res += getOverlapCore!(cache, (o1c,), (sL,), (ptr1,)) * c1^2
        else
            o2c, ptr2, c2 = getIntegrandComponent(primOrbs, p.basis, w, j)
            temp = getOverlapCore!(cache, (o1c, o2c), (sL, sL), (ptr1, ptr2)) * c1 * c2
            res += temp + temp'
        end
    end
    res
end

function getOverlapCore!(cache::DimSpanDataCacheBox{T}, 
                         (o1, o2)::NTuple{2, CompositeOrbCore{T, D}}, 
                         (s1, s2)::NTuple{2, FlatParamSet{T}}, 
                         (p1, p2)::NTuple{2, CompOrbParamPtr{T, D}}) where {T, D}
    res = zero(T)
    s1L = FilteredObject(s1, p1.scope)
    s2L = FilteredObject(s2, p2.scope)
    w1 = cacheParam!(cache, s1L, p1.weight)
    w2 = cacheParam!(cache, s2L, p2.weight)
    primOrbs1 = o1.f.chain
    primOrbs2 = o2.f.chain
    for j in eachindex(primOrbs2), i in eachindex(primOrbs1)
        o1c, ptr1, c1 = getIntegrandComponent(primOrbs1, p1.basis, w1, i)
        o2c, ptr2, c2 = getIntegrandComponent(primOrbs2, p2.basis, w2, j)
        temp = getOverlapCore!(cache, (o1c, o2c), (s1L, s2L), (ptr1, ptr2)) * c1 * c2
        res += temp
    end
    res
end