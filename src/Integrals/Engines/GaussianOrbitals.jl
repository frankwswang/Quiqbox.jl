using LinearAlgebra: norm
using LRUCache

const DefaultOddFactorialCacheSizeLimit = 25
const OddFactorialCache = LRU{Int, BigInt}(maxsize=DefaultOddFactorialCacheSizeLimit)

function oddFactorial(a::Int) # a * (a-2) * ... * 1
    get!(OddFactorialCache, a) do
        i = BigInt(1)
        for j = 1:2:a
            i *= j
        end
        i
    end
end


function gaussProdCore1(xpnLRsum::T, degree::Int) where {T<:Real}
    factor = degree > 0 ? (T(oddFactorial(2degree - 1)) / (2xpnLRsum)^degree) : one(T)
    T(πPowers[:p0d5]) / sqrt(xpnLRsum) * factor
end

function gaussProdCore2(dxML::T, dxMR::T, lxL::Int, lxR::Int, lx::Int) where {T<:Real}
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

function gaussProdCore2(lxL::Int, lxR::Int, lx::Int)
    lb = max(-lx,  lx - 2lxR)
    ub = min( lx, 2lxL - lx )
    res = zero(Int)
    for q in lb:2:ub
        res += Int(isequal(2lxL, lx+q) * isequal(2lxR, lx-q))
    end
    res
end

function gaussProdCore3(dxLR::T, xpnProdOverSum::T) where {T<:Real}
    exp(- xpnProdOverSum * dxLR^2)
end

function gaussProdCore4(xpnSum::T, xpnRatio::T, dxLR::T) where {T<:Real}
    gaussProdCore1(xpnSum, 0) * gaussProdCore3(dxLR, xpnRatio)
end


struct PrimGaussOrbInfo{T, D}
    cen::NTuple{D, T}
    xpn::T
    ang::NTuple{D, Int}

    PrimGaussOrbInfo(cen::NonEmptyTuple{T, D}, xpn::T, 
                     ang::NonEmptyTuple{Int, D}) where {T, D} = 
    new{T, D+1}(cen, xpn, ang)
end

const PrimGaussOrbConfig{T, D} = PrimGaussOrbInfo{FlatPSetInnerPtr{T}, D}


struct GaussProductInfo{T, D}
    lhs::PrimGaussOrbInfo{T, D}
    rhs::PrimGaussOrbInfo{T, D}
    cen::NTuple{D, T}
    xpn::T

    function GaussProductInfo(oData1::PrimGaussOrbInfo{T, D}, 
                              oData2::PrimGaussOrbInfo{T, D}) where {T, D}
        xpnSum = oData1.xpn + oData2.xpn
        cenNew = (oData1.xpn .* oData1.cen .+ oData2.xpn .* oData2.cen) ./ xpnSum
        new{T, D}(oData1, oData2, cenNew, xpnSum)
    end
end

function GaussProductInfo(configs::NTuple{2, PrimGaussOrbConfig{T, D}}, 
                          parsPair::NTuple{2, FilteredVecOfArr{T}}) where {T, D}
    d1, d2 = map(configs, parsPair) do sector, pars
        cen = getField(pars, sector.cen)
        xpn = getField(pars, sector.xpn)
        PrimGaussOrbInfo(cen, xpn, sector.ang)
    end
    GaussProductInfo(d1, d2)
end


struct NullCache{T} <: CustomCache{T} end

const TupleOf5T2Int{T} = Tuple{T, T, T, T, Int, Int}

struct ZeroAngMomCache{T} <: CustomCache{T} end

const CoreIntCacheBox{T} = Union{CustomCache{T}, LRU{<:Any, T}}


struct OneBodyCoreIntCache{T, D, 
                           F<:NTuple{D, CoreIntCacheBox{T}}} <: SpatialIntegralCache{T, D}
    axis::F

    OneBodyCoreIntCache(axis::NonEmptyTuple{CoreIntCacheBox{T}, D}) where {T, D} = 
    new{T, D+1, typeof(axis)}(axis)
end

OneBodyCoreIntCache(::Type{T}, ::Val{D}) where {T, D} = 
OneBodyCoreIntCache(ntuple( _->NullCache{T}(), Val(D) ))


function overlapPGTOcore(input::TupleOf5T2Int{T}) where {T}
    xpnSum, xpnRatio, xML, xMR, iL, iR = input
    dxLR = xMR - xML
    jRange = 0:((iL + iR) ÷ 2)

    if isequal(dxLR, zero(T))
        mapreduce(+, jRange) do j
            gaussProdCore1(xpnSum, j) * gaussProdCore2(iL, iR, 2j)
        end
    else
        mapreduce(+, jRange) do j
            gaussProdCore1(xpnSum, j) * gaussProdCore2(xML, xMR, iL, iR, 2j)
        end * gaussProdCore3(dxLR, xpnRatio)
    end
end

function overlapPGTOcore(xpn::T, ang::NonEmptyTuple{Int}) where {T}
    mapreduce(*, ang) do i
        gaussProdCore1(2xpn, i)
    end
end

function lazyOverlapPGTO(cache::LRU{TupleOf5T2Int{T}, T}, input::TupleOf5T2Int{T}) where {T}
    res = get(cache, input, nothing) # Fewer allocations than using `get!`
    if res === nothing
        res = overlapPGTOcore(input)
        setindex!(cache, res, input)
    end
    res
end

lazyOverlapPGTO(::NullCache{T}, input::TupleOf5T2Int{T}) where {T} = overlapPGTOcore(input)

lazyOverlapPGTO(::ZeroAngMomCache{T}, input::TupleOf5T2Int{T}) where {T} = 
gaussProdCore4(input[begin], input[begin+1], input[begin+3]-input[begin+2])


const GTOrbOverlapAxialCache{T} = 
      Union{NullCache{T}, ZeroAngMomCache{T}, LRU{TupleOf5T2Int{T}, T}}

const GTOrbOverlapCache{T, D} = 
      OneBodyCoreIntCache{T, D, <:NTuple{D, GTOrbOverlapAxialCache{T}}}

function overlapPGTO(data::GaussProductInfo{T, D}, 
                     cache::GTOrbOverlapCache{T, D}) where {T, D}
    cenL = data.lhs.cen
    cenR = data.rhs.cen
    cenM = data.cen

    xpnS = data.xpn
    xpnRatio = data.lhs.xpn * data.rhs.xpn / xpnS

    angL = data.lhs.ang
    angR = data.rhs.ang

    mapreduce(*, cache.axis, cenL, cenR, cenM, angL, angR) do cache, xL, xR, xM, iL, iR
        lazyOverlapPGTO(cache, (xpnS, xpnRatio, xM-xL, xM-xR, iL, iR))
    end
end


function getAngMomentum(orb::PrimATOcore)
    angFunc = orb.f.apply.f.right.f
    angFunc.f.m
end

function getExponentPtr(orb::PrimGTOcore)
    gFunc = orb.f.apply.f.left.apply
    gFunc.f.select[begin]
end

function getCenCoordPtr(orb::PrimitiveOrbCore)
    orb.f.dress.select
end


function preparePGTOconfig(orb::PrimGTOcore)
    cenIds = getCenCoordPtr(orb)
    xpnIdx = getExponentPtr(orb)
    ang = getAngMomentum(orb).tuple
    PrimGaussOrbInfo(cenIds, xpnIdx, ang)
end

struct PrimGTOrbOverlap{T, D, B<:N12Tuple{PrimGaussOrbConfig{T, D}}
                        } <: OrbitalIntegrator{T, D}
    basis::B
end

const OverlapGTOrbSelf{T, D} = PrimGTOrbOverlap{T, D,  Tuple{    PrimGaussOrbConfig{T, D} }}
const OverlapGTOrbPair{T, D} = PrimGTOrbOverlap{T, D, NTuple{ 2, PrimGaussOrbConfig{T, D} }}

function (f::OverlapGTOrbSelf{T})(pars::FilteredVecOfArr{T}) where {T}
    sector = first(f.basis)
    xpnVal = getField(pars, sector.xpn)
    overlapPGTOcore(xpnVal, sector.ang)
end

function (f::OverlapGTOrbPair{T, D})(pars1::FilteredVecOfArr{T}, 
                                     pars2::FilteredVecOfArr{T}; 
                                     cache::GTOrbOverlapCache{T, D}=
                                     OneBodyCoreIntCache(T, Val(D))) where {T, D}
    data = GaussProductInfo(f.basis, (pars1, pars2))
    overlapPGTO(data, cache)
end


function genGTOrbOverlapFunc(orbCoreData::N12Tuple{PrimGTOcore{T, D}}) where {T, D}
    PrimGTOrbOverlap(preparePGTOconfig.(orbCoreData))
end


const DefaultPGTOrbOverlapCacheSizeLimit = 128

function genPrimGTOrbOverlapCache(::Type{T}, ::Val{D}, ::Val{L}) where {T, D, L}
    ntuple(_->LRU{TupleOf5T2Int{T}, T}(maxsize=DefaultPGTOrbOverlapCacheSizeLimit), Val(D))
end

function genPrimGTOrbOverlapCache(::Type{T}, ::Val{D}, ::Val{0}) where {T, D}
    ntuple(_->ZeroAngMomCache{T}(), Val(D))
end

function getGTOrbOverlapCache((orb1, orb2)::Tuple{TypedPrimGTOcore{T, D, L1}, 
                                                  TypedPrimGTOcore{T, D, L2}}) where 
                             {T, D, L1, L2}
    getGTOrbOverlapCacheCore(T, Val(D), Val(L1+L2)) |> OneBodyCoreIntCache
end

function getGTOrbOverlapCache(::Tuple{TypedPrimGTOcore{T}}) where {T}
    NullCache{T}()
end


function buildNormalizerCore(o::PrimGTOcore{T, D}) where {T, D}
    genGTOrbOverlapFunc((o,))
end