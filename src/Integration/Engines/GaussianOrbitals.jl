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


struct PrimGaussOrbInfo{T<:Real, D} <: QueryBox{T}
    cen::NTuple{D, T}
    xpn::T
    ang::NTuple{D, Int}

    function PrimGaussOrbInfo(cen::NonEmptyTuple{Real, D}, xpn::Real, 
                              ang::NonEmptyTuple{Int, D}) where {D}
        cen..., xpn = promote(cen..., xpn)
        new{typeof(xpn), D+1}(cen, xpn, ang)
    end
end

struct GaussProductInfo{T<:Real, D} <: QueryBox{T}
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

function overlapPGTOcore(input::TupleOf5T2Int{T}) where {T<:Real}
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

function overlapPGTOcore(xpn::T, ang::NonEmptyTuple{Int}) where {T<:Real}
    mapreduce(*, ang) do i
        gaussProdCore1(2xpn, i)
    end
end


function lazyOverlapPGTO!(cache::LRU{TupleOf5T2Int{T}, T}, 
                          input::TupleOf5T2Int{T}) where {T<:Real}
    res = get(cache, input, nothing) # Fewer allocations than using `get!`
    if res === nothing
        res = overlapPGTOcore(input)
        setindex!(cache, res, input)
    end
    res
end

lazyOverlapPGTO!(::NullCache{T}, input::TupleOf5T2Int{T}) where {T<:Real} = 
overlapPGTOcore(input)


const PrimGaussOrbInfoCache{T<:Real, D} = 
      LRU{EgalBox{FloatingPolyGaussField{T, D}}, PrimGaussOrbInfo{T, D}}

struct AxialGaussTypeOverlapCache{T<:Real, D, 
                                  M<:NTuple{D, MissingOr{ LRU{TupleOf5T2Int{T}, T} }}
                                  } <: CustomCache{T}
    axis::M
    encode::PrimGaussOrbInfoCache{T, D}

    function AxialGaussTypeOverlapCache(::Type{T}, ::Val{D}) where {T, D}
        axis = ntuple(_->missing, Val(D::Int))
        new{T, D, NTuple{D, Missing}}(axis, PrimGaussOrbInfoCache{T, D}(maxsize=128))
    end

    function AxialGaussTypeOverlapCache(::Type{T}, config::NTuple{D, BoolVal}) where {T, D}
        axialCache = map(config) do axialConfig
            getValData(axialConfig) ? LRU{TupleOf5T2Int{T}, T}(maxsize=128) : missing
        end
        new{T, D, typeof(axialCache)}(axialCache, PrimGaussOrbInfoCache{T, D}(maxsize=128))
    end
end

const GaussTypeOrbIntCache{T<:Real} = Union{NullCache{T}, AxialGaussTypeOverlapCache{T}}


accessAxialCache(cache::AxialGaussTypeOverlapCache, i::Int) = cache.axis[begin+i-1]

accessAxialCache(cache::NullCache, ::Int) = cache


function overlapPGTO!(cache::GaussTypeOrbIntCache{T}, 
                      data::GaussProductInfo{T, D}) where {T<:Real, D}
    cenL = data.lhs.cen
    cenR = data.rhs.cen
    cenM = data.cen

    xpnS = data.xpn
    xpnRatio = data.lhs.xpn * data.rhs.xpn / xpnS

    angL = data.lhs.ang
    angR = data.rhs.ang

    i = 0

    mapreduce(*, cenL, cenR, cenM, angL, angR) do xL, xR, xM, iL, iR
        axialCache = accessAxialCache(cache, (i += 1))
        lazyOverlapPGTO!(axialCache, (xpnS, xpnRatio, xM-xL, xM-xR, iL, iR))
    end
end


function prepareOrbitalInfoCore(field::FloatingPolyGaussField)
    gtf = field.core.f.f # PolyGaussFieldCore
    grf = first(gtf.encode) # RadialFieldFunc
    pgf = last(grf.core.f.binder.f.encode) # GaussFieldFunc
    xpnFormatter = last(pgf.core.f.binder.f.encode) # ParamFormatter
    xpn = xpnFormatter.core(field.param).xpn
    angMomField = last(gtf.encode) # CartAngMomFieldFunc
    amfCore = angMomField.core.f.binder.f.core.f # CartSHarmonics
    ang = amfCore.m.tuple
    PrimGaussOrbInfo(field.center, xpn, ang)
end

function prepareOrbitalInfo!(cache::AxialGaussTypeOverlapCache{T, D}, 
                             orbData::FloatingPolyGaussField{T, D}) where {T<:Real, D}
    get!(cache.encode, EgalBox{FloatingPolyGaussField{T, D}}(orbData)) do
        prepareOrbitalInfoCore(orbData)
    end
end

function prepareOrbitalInfo!(::NullCache{T}, 
                             orbData::FloatingPolyGaussField{T, D}) where {T<:Real, D}
    prepareOrbitalInfoCore(orbData)
end


function computeGTOrbOverlap((data,)::Tuple{FloatingPolyGaussField{T, D}}; 
                             cache!Self::GaussTypeOrbIntCache{T}=
                                         AxialGaussTypeOverlapCache(T, Val(D))
                             ) where {T<:Real, D}
    formattedData = prepareOrbitalInfo!(cache!Self, data)
    overlapPGTOcore(formattedData.xpn, formattedData.ang)
end

function computeGTOrbOverlap(data::NTuple{2, FloatingPolyGaussField{T, D}}; 
                             cache!Self::GaussTypeOrbIntCache{T}=
                                         AxialGaussTypeOverlapCache(T, Val(D))
                             ) where {T<:Real, D}
    formattedData = map(data) do sector
        prepareOrbitalInfo!(cache!Self, sector)
    end |> Base.Splat(GaussProductInfo)
    overlapPGTO!(cache!Self, formattedData)
end


## Multiple Moment ##
function computeMultiMomentGTO(fm::FloatingMonomial{T, D}, 
                               data::PrimGaussOrbInfo{T, D}) where {T<:Real, D}
    xpn = data.xpn
    mapreduce(*, fm.center, fm.degree.tuple, data.center, data.ang) do xMM, n, x, i
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


function computeMultiMomentGTO!(fm::FloatingMonomial{T, D}, cache::GaussTypeOrbIntCache{T}, 
                                data::GaussProductInfo{T, D}) where {T<:Real, D}
    if fm.degree.total == 0
        overlapPGTO!(cache, data)
    else
        cenL = data.lhs.cen
        cenR = data.rhs.cen
        cenM = data.cen

        xpnS = data.xpn
        xpnRatio = data.lhs.xpn * data.rhs.xpn / xpnS

        angL = data.lhs.ang
        angR = data.rhs.ang

        i = 0

        mapreduce(*, fm.center, fm.degree.tuple, 
                     cenL, cenR, cenM, angL, angR) do xMM, n, xL, xR, xM, iL, iR
            dx = xR - xMM #! Consider when xL == xMM
            m = iszero(dx) ? 0 : n
            axialCache = accessAxialCache(cache, (i += 1))
            mapreduce(+, 0:m) do k
                binomial(n, k) * dx^k * 
                lazyOverlapPGTO!(axialCache, (xpnS, xpnRatio, xM-xL, xM-xR, iL, iR+n-k))
            end
        end
    end
end


function computeGTOrbMultiMom(op::MultipoleMomentSampler{T, D}, 
                              (data,)::Tuple{FloatingPolyGaussField{T, D}};
                              cache!Self::GaussTypeOrbIntCache{T}=
                                          AxialGaussTypeOverlapCache(T, Val(D))
                              ) where {T<:Real, D}
    formattedData = prepareOrbitalInfo!(cache!Self, data)
    computeMultiMomentGTO(last(op.dresser).term, formattedData)
end

function computeGTOrbMultiMom(op::MultipoleMomentSampler{T, D}, 
                              data::NTuple{2, FloatingPolyGaussField{T, D}}; 
                              cache!Self::GaussTypeOrbIntCache{T}=
                                          AxialGaussTypeOverlapCache(T, Val(D))
                              ) where {T<:Real, D}
    formattedData = map(data) do sector
        prepareOrbitalInfo!(cache!Self, sector)
    end |> Base.Splat(GaussProductInfo)
    computeMultiMomentGTO!(last(op.dresser).term, cache!Self, formattedData)
end


function getGaussTypeOrbIntegrator(::OneBodyIntegral, ::OverlapSampler)
    computeGTOrbOverlap
end

function getGaussTypeOrbIntegrator(::OneBodyIntegral{D}, op::MultipoleMomentSampler{T, D}
                                   ) where {T<:Real, D}
    LPartial(computeGTOrbMultiMom, (op,))
end


function getAnalyticIntegral!(::S, cache!Self::GaussTypeOrbIntCache{T}, op::DirectOperator, 
                              data::OneBodyOrbIntLayout{PrimGTOData{T, D}}) where 
                             {D, S<:MultiBodyIntegral{D}, T<:Real}
    fields = getfield.(data, :core)
    integrator = getGaussTypeOrbIntegrator(S(), op)
    integrator(fields; cache!Self)
end


#= Additional Method =#
function getAnalyticIntegralCache(::Union{OverlapSampler, MultipoleMomentSampler{T, D}}, 
                                  ::OneBodyOrbIntLayout{PrimGTOData{T, D}}
                                  ) where {T<:Real, D}
    AxialGaussTypeOverlapCache(T, ntuple( _->Val(true), Val(D) ))
end