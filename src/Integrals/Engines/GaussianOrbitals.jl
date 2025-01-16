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


function gaussProdCore1(dxLR::T, xpnProdOverSum::T) where {T<:Real}
    exp(- xpnProdOverSum * dxLR^2)
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


struct PrimGaussOrbData{T, D}
    cen::NTuple{D, T}
    xpn::T
    ang::NTuple{D, Int}

    PrimGaussOrbData(cen::NonEmptyTuple{T, D}, xpn::T, 
                     ang::NonEmptyTuple{Int, D}) where {T, D} = 
    new{T, D+1}(cen, xpn, ang)
end


struct GaussProductData{T, D}
    lhs::PrimGaussOrbData{T, D}
    rhs::PrimGaussOrbData{T, D}
    cen::NTuple{D, T}
    xpn::T

    function GaussProductData(oData1::PrimGaussOrbData{T, D}, 
                              oData2::PrimGaussOrbData) where {T, D}
        xpnSum = oData1.xpn + oData2.xpn
        cenNew = (oData1.xpn .* oData1.cen .+ oData2.xpn .* oData2.cen) ./ xpnSum
        new{T, D}(oData1, oData2, cenNew, xpnSum)
    end
end


function overlapPGTO(data::GaussProductData{T}) where {T}
    cenL = data.lhs.cen
    cenR = data.rhs.cen
    cenM = data.cen

    xpn = data.xpn
    xpnRatio = data.lhs.xpn * data.rhs.xpn / xpn

    angL = data.lhs.ang
    angR = data.rhs.ang

    mapreduce(*, cenL, cenR, cenM, angL, angR) do xL, xR, xM, iL, iR
        dxLR = xL - xR
        jRange = 0:((iL + iR) ÷ 2)
        if isequal(dxLR, zero(T))
            mapreduce(+, jRange) do j
                gaussProdCore2(iL, iR, 2j) * polyGaussFuncSquaredNorm(xpn/2, j)
            end
        else
            mapreduce(+, jRange) do j
                xML = xM - xL
                xMR = xM - xR
                gaussProdCore2(xML, xMR, iL, iR, 2j) * polyGaussFuncSquaredNorm(xpn/2, j)
            end * gaussProdCore1(dxLR, xpnRatio)
        end
    end
end

function overlapPGTO(xpn::T, ang::NonEmptyTuple{Int}) where {T}
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


struct OverlapGTOrbPair{T, D} <: OrbitalIntegrator{T, D}
    cen::NTuple{2, NTuple{ D, FlatPSetInnerPtr{T} }}
    xpn::NTuple{2, FlatPSetInnerPtr{T}}
    ang::NTuple{2, NTuple{D, Int}}
end

function (f::OverlapGTOrbPair{T})(pars1::FilteredVecOfArr{T}, 
                                  pars2::FilteredVecOfArr{T}) where {T}
    c1 = getField(pars1, first(f.cen))
    x1 = getField(pars1, first(f.xpn))
    d1 = PrimGaussOrbData(c1, x1, first(f.ang))
    c2 = getField(pars2,  last(f.cen))
    x2 = getField(pars2,  last(f.xpn))
    d2 = PrimGaussOrbData(c2, x2,  last(f.ang))
    GaussProductData(d1, d2) |> overlapPGTO
end


function genGTOrbOverlapFunc(orbs::NTuple{2, PrimGTOcore{T, D}}) where {T, D}
    configs = preparePGTOconfig.(orbs)
    OverlapGTOrbPair(getindex.(configs, 1), getindex.(configs, 2), getindex.(configs, 3))
end

function genGTOrbOverlapFunc((orb,)::Tuple{PrimGTOcore{T, D}}) where {T, D}
    _, xpnIdx, ang = preparePGTOconfig(orb)
    OverlapGTOrbSelf(xpnIdx, ang)
end


function buildNormalizerCore(o::PrimGTOcore{T, D}) where {T, D}
    genGTOrbOverlapFunc((o,))
end