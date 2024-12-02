function genNormalizer(o::PrimitiveOrbCore{T, D}, 
                       paramPtr::PrimOrbParamPtr{T, D}) where {T, D}
    ptrTuple = (Tuple∘values)(paramPtr.body.core)
    fInner = (OnlyBody∘LeftPartial)(AbsSqrtInv∘numericalOneBodyInt, Identity(), (o.f.apply,))
    fCore = function (input, args...)
        fInner(input, (buildDict(ptrTuple .=> args),))
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


const ParamPtrHolder{T} = Union{FieldPtrDict{T}, NonEmptyTuple{FlatParamSetIdxPtr{T}}}

getParamPointers(input::FieldPtrDict) = (collect∘keys)(input)
getParamPointers(input::NonEmptyTuple{FlatParamSetIdxPtr{T}}) where {T} = itself(input)

function cacheParamDictCore!(cache::DimSpanDataCacheBox{T}, pSet::FlatParamSet{T}, 
                             ptrs::NonEmpTplOrAbtArr{<:FlatParamSetIdxPtr{T}}) where {T}
    map(ptrs) do ptr
        ptr => cacheParam!(cache, pSet, ptr)
    end |> buildDict
end

function cacheParamDict!(cache::DimSpanDataCacheBox{T}, pSet::FlatParamSet{T}, 
                         input::Union{ParamPtrHolder{T}, Tuple{}}) where {T}
    isempty(input) ? buildDict() : cacheParamDictCore!(cache, pSet, getParamPointers(input))
end

function cacheParamDict!(cache::DimSpanDataCacheBox{T}, pSet::FlatParamSet{T}, 
                         input::PrimOrbParamPtr{T}) where {T}
    ptrs = vcat(getParamPointers(input.body.core), input.center...)
    cacheParamDictCore!(cache, pSet, ptrs)
end


function getNormCoeffCore!(cache::DimSpanDataCacheBox{T}, pSet::FlatParamSet{T}, 
                           normalizer::ReturnTyped{T, <:PointerFunc}) where {T}
    pDict = cacheParamDict!(cache, pSet, normalizer.f.pointer)
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
    res = getOverlapCore!(cache, (o.core.f.left,), (o.param,), (o.pointer,))
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


function getOverlapCore!(cache::DimSpanDataCacheBox{T}, 
                         (o,)::Tuple{PrimitiveOrbCore{T, D}}, 
                         (s,)::Tuple{FlatParamSet{T}}, 
                         (p,)::Tuple{PrimOrbParamPtr{T, D}}) where {T, D}
    parDict = cacheParamDict!(cache, s, p.body.core)
    numericalOneBodyInt(Identity(), (o.f.apply,), (parDict,))
end

function getOverlapCore!(cache::DimSpanDataCacheBox{T}, 
                         (o1, o2)::NTuple{2, PrimitiveOrbCore{T, D}}, 
                         (s1, s2)::NTuple{2, FlatParamSet{T}}, 
                         (p1, p2)::NTuple{2, PrimOrbParamPtr{T, D}}) where {T, D}
    parDict1 = cacheParamDict!(cache, s1, p1)
    parDict2 = cacheParamDict!(cache, s2, p2)
    numericalOneBodyInt(Identity(), (o1.f.apply, o2.f.apply), (parDict1, parDict2))
end

function getIntegrandComponent(o::Memory{<:WeightedPF{T, D}}, 
                               p::Memory{<:PrimOrbParamPtr{T, D}}, 
                               w::ShapedMemory{T, 1}, idx::Int) where {T, D}
    (o[idx].f.left.f.left, p[idx], w[idx])
end

function getOverlapCore!(cache::DimSpanDataCacheBox{T}, 
                         (o,)::Tuple{CompositeOrbCore{T, D}}, 
                         (s,)::Tuple{FlatParamSet{T}}, 
                         (p,)::Tuple{CompOrbParamPtr{T, D}}) where {T, D}
    res = zero(T)
    w = cacheParam!(cache, s, p.weight)
    primOrbs = o.f.chain
    for j in eachindex(primOrbs), i in 1:j
        o1core, ptr1, c1 = getIntegrandComponent(primOrbs, p.basis, w, i)
        if i == j
            res += getOverlapCore!(cache, (o1core,), (s,), (ptr1,)) * c1^2
        else
            o2core, ptr2, c2 = getIntegrandComponent(primOrbs, p.basis, w, j)
            temp = getOverlapCore!(cache, (o1core, o2core), (s, s), (ptr1, ptr2)) * c1 * c2
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
    w1 = cacheParam!(cache, s1, p1.weight)
    w2 = cacheParam!(cache, s2, p2.weight)
    primOrbs1 = o1.f.chain
    primOrbs2 = o2.f.chain
    for j in eachindex(primOrbs2), i in eachindex(primOrbs1)
        o1core, ptr1, c1 = getIntegrandComponent(primOrbs1, p1.basis, w1, i)
        o2core, ptr2, c2 = getIntegrandComponent(primOrbs2, p2.basis, w2, j)
        temp = getOverlapCore!(cache, (o1core, o2core), (s1, s2), (ptr1, ptr2)) * c1 * c2
        res += temp
    end
    res
end