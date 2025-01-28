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

abstract type OrbIntegralComputeCache{T, D, S<:MultiBodyIntegral{D}
                                      } <: IntegralProcessCache{T, D} end

const Orb1BIntegralComputeCache{T, D} = OrbIntegralComputeCache{T, D, OneBodyIntegral{D}}

struct AxialOneBodyIntCompCache{T, D, F<:NTuple{D, CoreIntCacheBox{T}}
                                } <: Orb1BIntegralComputeCache{T, D}
    axis::F

    AxialOneBodyIntCompCache(axis::NonEmptyTuple{CoreIntCacheBox{T}, D}) where {T, D} = 
    new{T, D+1, typeof(axis)}(axis)
end

AxialOneBodyIntCompCache(::Type{T}, ::Val{D}) where {T, D} = 
AxialOneBodyIntCompCache(ntuple( _->NullCache{T}(), Val(D) ))


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

function lazyOverlapPGTO!(cache::LRU{TupleOf5T2Int{T}, T}, 
                          input::TupleOf5T2Int{T}) where {T}
    res = get(cache, input, nothing) # Fewer allocations than using `get!`
    if res === nothing
        res = overlapPGTOcore(input)
        setindex!(cache, res, input)
    end
    res
end

lazyOverlapPGTO!(::NullCache{T}, input::TupleOf5T2Int{T}) where {T} = overlapPGTOcore(input)

lazyOverlapPGTO!(::ZeroAngMomCache{T}, input::TupleOf5T2Int{T}) where {T} = 
gaussProdCore4(input[begin], input[begin+1], input[begin+3]-input[begin+2])


const GTOrbOverlapAxialCache{T} = 
      Union{NullCache{T}, ZeroAngMomCache{T}, LRU{TupleOf5T2Int{T}, T}}

const GTOrbOverlapCache{T, D} = 
      AxialOneBodyIntCompCache{T, D, <:NTuple{D, GTOrbOverlapAxialCache{T}}}

function overlapPGTO!(cache::GTOrbOverlapCache{T, D}, data::GaussProductInfo{T, D}, 
                      ) where {T, D}
    cenL = data.lhs.cen
    cenR = data.rhs.cen
    cenM = data.cen

    xpnS = data.xpn
    xpnRatio = data.lhs.xpn * data.rhs.xpn / xpnS

    angL = data.lhs.ang
    angR = data.rhs.ang

    mapreduce(*, cache.axis, cenL, cenR, cenM, angL, angR) do cache, xL, xR, xM, iL, iR
        lazyOverlapPGTO!(cache, (xpnS, xpnRatio, xM-xL, xM-xR, iL, iR))
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
                                     cache!Self::GTOrbOverlapCache{T, D}=
                                     AxialOneBodyIntCompCache(T, Val(D))) where {T, D}
    data = GaussProductInfo(f.basis, (pars1, pars2))
    overlapPGTO!(cache!Self, data)
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

genGTOrbIntCompCache(::MonomialMul{T, D, L}, 
                     ::Tuple{TypedPrimGTOcore{T, D, L1}, TypedPrimGTOcore{T, D, L2}}) where 
                    {T, D, L, L1, L2} = 
genPrimGTOrbOverlapCache(T, Val(D), Val(L+L1+L2)) |> AxialOneBodyIntCompCache

genGTOrbIntCompCache(::Identity, 
                     ::Tuple{TypedPrimGTOcore{T, D, L1}, TypedPrimGTOcore{T, D, L2}}) where 
                    {T, D, L1, L2} = 
genPrimGTOrbOverlapCache(T, Val(D), Val(L1+L2)) |> AxialOneBodyIntCompCache


function buildNormalizerCore(o::PrimGTOcore{T, D}) where {T, D}
    genGTOrbOverlapFunc((o,))
end


## Multiple Moment ##
function computeMultiMomentGTO(op::MonomialMul{T, D}, 
                               data::PrimGaussOrbInfo{T, D}) where {T, D}
    xpn = data.xpn
    mapreduce(*, op.center, op.degree.tuple, data.center, data.ang) do xMM, n, x, i
        dx = x - xMM
        m = iszero(dx) ? 0 : n
        mapreduce(+, 0:m) do k
            l = n - k
            if isodd(l)
                zero(T)
            else
                binomial(n, k) * dx^k * gaussProdCore1(2xpn, i + (l >> 1))
            end
        end
    end
end


function computeMultiMomentGTO!(op::MonomialMul{T, D}, cache::GTOrbOverlapCache{T, D}, 
                                data::GaussProductInfo{T, D}) where {T, D}

    cenL = data.lhs.cen
    cenR = data.rhs.cen
    cenM = data.cen

    xpnS = data.xpn
    xpnRatio = data.lhs.xpn * data.rhs.xpn / xpnS

    angL = data.lhs.ang
    angR = data.rhs.ang

    mapreduce(*, cache.axis, op.center, op.degree.tuple, 
                 cenL, cenR, cenM, angL, angR) do cache, xMM, n, xL, xR, xM, iL, iR
        dx = xR - xMM #! Consider when xL == xMM
        m = iszero(dx) ? 0 : n
        mapreduce(+, 0:m) do k
            binomial(n, k) * dx^k * 
            lazyOverlapPGTO!(cache, (xpnS, xpnRatio, xM-xL, xM-xR, iL, iR+n-k))
        end
    end
end

computeMultiMomentGTO!(::MonomialMul{T, D, 0}, cache::GTOrbOverlapCache{T, D}, 
                       data::GaussProductInfo{T, D}) where {T, D} = 
overlapPGTO!(cache, data)


struct PrimGTOrbMultiMoment{T, D, L, B<:N12Tuple{PrimGaussOrbConfig{T, D}}
                            } <: OrbitalIntegrator{T, D}
    op::MonomialMul{T, D, L}
    basis::B
end

const MultiMomentGTOrbSelf{T, D, L} = 
      PrimGTOrbMultiMoment{T, D, L,  Tuple{    PrimGaussOrbConfig{T, D} }}
const MultiMomentGTOrbPair{T, D, L} = 
      PrimGTOrbMultiMoment{T, D, L, NTuple{ 2, PrimGaussOrbConfig{T, D} }}

function (f::MultiMomentGTOrbSelf{T})(pars::FilteredVecOfArr{T}) where {T}
    sector = first(f.basis)
    cen = getField(pars, sector.cen)
    xpn = getField(pars, sector.xpn)
    data = PrimGaussOrbInfo(cen, xpn, sector.ang)
    computeMultiMomentGTO(f.op, data)
end

function (f::MultiMomentGTOrbPair{T, D})(pars1::FilteredVecOfArr{T}, 
                                         pars2::FilteredVecOfArr{T}; 
                                         cache!Self::GTOrbOverlapCache{T, D}=
                                         AxialOneBodyIntCompCache(T, Val(D))) where {T, D}
    data = GaussProductInfo(f.basis, (pars1, pars2))
    computeMultiMomentGTO!(f.op, cache!Self, data)
end


function genGTOrbMultiMomentFunc(op::MonomialMul{T, D}, 
                                 orbCoreData::N12Tuple{PrimGTOcore{T, D}}) where {T, D}
    PrimGTOrbMultiMoment(op, preparePGTOconfig.(orbCoreData))
end