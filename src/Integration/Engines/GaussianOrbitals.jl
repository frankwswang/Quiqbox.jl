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


struct PrimGaussOrbInfo{T<:Real, D}
    cen::NTuple{D, T}
    xpn::T
    ang::NTuple{D, Int}

    function PrimGaussOrbInfo(cen::NonEmptyTuple{Real, D}, xpn::Real, 
                              ang::NonEmptyTuple{Int, D}) where {D}
        cen..., xpn = promote(cen..., xpn)
        new{typeof(xpn), D+1}(cen, xpn, ang)
    end
end

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


const TupleOf5T2Int{T} = Tuple{T, T, T, T, Int, Int}

function overlapPGTOcore(input::TupleOf5T2Int{T}) where {T}
    xpnSum, xpnRatio, xML, xMR, iL, iR = input
    dxLR = xMR - xML
    if iL == iR == 0
        gaussProdCore4(xpnSum, xpnRatio, dxLR)
    else
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

lazyOverlapPGTO!(::Missing, input::TupleOf5T2Int) = overlapPGTOcore(input)


struct AxialGaussTypeOverlapCache{T, D, C<:NTuple{D, MissingOr{ LRU{TupleOf5T2Int{T}, T} }}
                                  } <: CustomCache{T}
    axis::C

    function AxialGaussTypeOverlapCache(::Type{T}, ::Val{D}) where {T, D}
        axis = ntuple(_->missing, Val(D::Int))
        new{T, D, NTuple{D, Missing}}(axis)
    end

    function AxialGaussTypeOverlapCache(::Type{T}, config::NTuple{D, BoolVal}) where {T, D}
        axialCache = map(config) do axialConfig
            getValData(axialConfig) ? LRU{TupleOf5T2Int{T}, T}(maxsize=128) : missing
        end
        new{T, D, typeof(axialCache)}(axialCache)
    end
end


function overlapPGTO!(cache::AxialGaussTypeOverlapCache{T, D}, 
                      data::GaussProductInfo{T, D}) where {T, D}
    cenL = data.lhs.cen
    cenR = data.rhs.cen
    cenM = data.cen

    xpnS = data.xpn
    xpnRatio = data.lhs.xpn * data.rhs.xpn / xpnS

    angL = data.lhs.ang
    angR = data.rhs.ang

    mapreduce(*, cache.axis, cenL, cenR, cenM, angL, angR) do AxialCache, xL, xR, xM, iL, iR
        lazyOverlapPGTO!(AxialCache, (xpnS, xpnRatio, xM-xL, xM-xR, iL, iR))
    end
end


function prepareOrbitalInfo(orbData::PrimGTOData{T, D}) where {T<:Real, D}
    fCore, paramSet = orbData.body
    gtf = fCore.core.f
    paramFilter = last(gtf.encode).core.core
    cen = convert(NTuple{D, T}, orbData.center)
    pgf = last(first(gtf.binder.f.encode).core.f.binder.f.encode)
    xpn = last(pgf.core.f.binder.f.encode).core(paramSet|>paramFilter).xpn
    ang = last(gtf.binder.f.encode).core.f.binder.f.core.core.m.tuple
    PrimGaussOrbInfo(cen, xpn, ang)
end


function computeGTOrbOverlap(data::Tuple{PrimGTOData})
    formattedData = prepareOrbitalInfo(data|>first)
    overlapPGTOcore(formattedData.xpn, formattedData.ang)
end

function computeGTOrbOverlap(data::NTuple{2, PrimGTOData{T, D}}; 
                             cache!Self::AxialGaussTypeOverlapCache{T, D}=
                                         AxialGaussTypeOverlapCache(T, Val(D))
                             ) where {T, D}
    formattedData = map(data) do sector
        prepareOrbitalInfo(sector)
    end |> Base.Splat(GaussProductInfo)
    overlapPGTO!(cache!Self, formattedData)
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


function computeMultiMomentGTO!(op::MonomialMul{T, D}, 
                                cache::AxialGaussTypeOverlapCache{T, D}, 
                                data::GaussProductInfo{T, D}) where {T, D}
    if op.degree.total == 0
        overlapPGTO!(cache, data)
    else
        cenL = data.lhs.cen
        cenR = data.rhs.cen
        cenM = data.cen

        xpnS = data.xpn
        xpnRatio = data.lhs.xpn * data.rhs.xpn / xpnS

        angL = data.lhs.ang
        angR = data.rhs.ang

        mapreduce(*, cache.axis, op.center, op.degree.tuple, 
                    cenL, cenR, cenM, angL, angR) do AxialCache, xMM, n, xL, xR, xM, iL, iR
            dx = xR - xMM #! Consider when xL == xMM
            m = iszero(dx) ? 0 : n
            mapreduce(+, 0:m) do k
                binomial(n, k) * dx^k * 
                lazyOverlapPGTO!(AxialCache, (xpnS, xpnRatio, xM-xL, xM-xR, iL, iR+n-k))
            end
        end
    end
end


function computeGTOrbMultiMom(op::MonomialMul{T, D}, data::Tuple{PrimGTOData{T, D}}
                              ) where {T, D}
    formattedData = prepareOrbitalInfo(data|>first)
    computeMultiMomentGTO(op, formattedData)
end

function computeGTOrbMultiMom(op::MonomialMul{T, D}, data::NTuple{2, PrimGTOData{T, D}}; 
                              cache!Self::AxialGaussTypeOverlapCache{T, D}=
                                          AxialGaussTypeOverlapCache(T, Val(D))
                              ) where {T, D}
    formattedData = map(data) do sector
        prepareOrbitalInfo(sector)
    end |> Base.Splat(GaussProductInfo)
    computeMultiMomentGTO!(op, cache!Self, formattedData)
end

function getGaussTypeOrbIntegrator(::OneBodyIntegral, ::Identity)
    computeGTOrbOverlap
end

function getGaussTypeOrbIntegrator(::OneBodyIntegral{D}, op::MonomialMul{T, D}) where {T, D}
    LPartial(computeGTOrbMultiMom, (op,))
end


const GaussTypeOrbIntCache{T} = Union{NullCache{T}, AxialGaussTypeOverlapCache{T}}

function getAnalyticIntegral!(::S, cache::GaussTypeOrbIntCache{T}, op::DirectOperator, 
                              data::N12Tuple{PrimGTOData{T, D}}) where 
                             {D, S<:MultiBodyIntegral{D}, T}
    integrator = getGaussTypeOrbIntegrator(S(), op)
    if length(data) > 1
        cache!Self = (cache isa NullCache) ? AxialGaussTypeOverlapCache(T, Val(D)) : cache
        integrator(data; cache!Self)
    else
        integrator(data)
    end
end