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


function polyGaussFuncSquaredNorm(α::T, degree::Int) where {T<:Real}
    factor = degree > 0 ? (T(oddFactorial(2degree - 1)) / (4α)^degree) : one(T)
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


function overlapPGTO(xpn::T, ang::NTuple{D, Int}) where {T, D}
    mapreduce(*, ang) do i
        polyGaussFuncSquaredNorm(xpn, i)
    end
end

function getAngularNums(orb::PrimATOcore)
    angFunc = orb.f.apply.f.right.f
    angFunc.f.m.tuple
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
    ang = getAngularNums(orb)
    cenIds, xpnIdx, ang
end

struct OverlapGTOrbSelf{T, D} <: OrbitalIntegrator{T, D}
    xpn::FlatPSetInnerPtr{T}
    ang::NTuple{D, Int}
end

function (f::OverlapGTOrbSelf{T})(pars::FilteredVecOfArr{T}) where {T}
    xpnVal = getField(pars, f.xpn)
    overlapPGTO(xpnVal, f.ang)
end

function genGTOrbOverlapFunc((orb,)::Tuple{PrimGTOcore{T, D}}) where {T, D}
    _, xpnIdx, ang = preparePGTOconfig(orb)
    OverlapGTOrbSelf(xpnIdx, ang)
end

struct OverlapGTOrbPair{T, D} <: OrbitalIntegrator{T, D}
    cen::NTuple{2, NTuple{ D, FlatPSetInnerPtr{T} }}
    xpn::NTuple{2, FlatPSetInnerPtr{T}}
    ang::NTuple{2, NTuple{D, Int}}
end

function (f::OverlapGTOrbPair{T})(pars1::FilteredVecOfArr{T}, 
                                  pars2::FilteredVecOfArr{T}) where {T}
    cen1 = getField(pars1, f.cen[1])
    cen2 = getField(pars2, f.cen[2])
    xpn1 = getField(pars1, f.xpn[1])
    xpn2 = getField(pars2, f.xpn[2])
    overlapPGTO(CenPair(cen1, cen2), XpnPair(xpn1, xpn2), AngPair(f.ang...))
end

function genGTOrbOverlapFunc(orbs::NTuple{2, PrimGTOcore{T, D}}) where {T, D}
    configs = preparePGTOconfig.(orbs)
    OverlapGTOrbPair(getindex.(configs, 1), getindex.(configs, 2), getindex.(configs, 3))
end


function buildNormalizerCore(o::PrimGTOcore{T, D}) where {T, D}
    genGTOrbOverlapFunc((o,))
end