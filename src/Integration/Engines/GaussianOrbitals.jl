using LinearAlgebra: norm
using LRUCache

const DefaultOddFactorialCacheSizeLimit = 25
const OddFactorialCache = LRU{Int, BigInt}(maxsize=DefaultOddFactorialCacheSizeLimit)

#>-- Basic math formula --<#
function oddFactorial(a::Int) # a * (a-2) * ... * 1
    get!(OddFactorialCache, a) do
        i = BigInt(1)
        for j = 1:2:a
            i *= j
        end
        i
    end
end


function computeGaussProd(dxML::T, dxMR::T, lxL::Int, lxR::Int, lx::Int) where {T<:Real}
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

function computeGaussProd(lxL::Int, lxR::Int, lx::Int)
    lb = max(-lx,  lx - 2lxR)
    ub = min( lx, 2lxL - lx )
    res = zero(Int)
    for q in lb:2:ub
        res += Int(isequal(2lxL, lx+q) * isequal(2lxR, lx-q))
    end
    res
end


function computePGTOrbOverlapAxialFactor(xpnLRsum::T, degree::Int) where {T<:Real}
    factor = degree > 0 ? (T(oddFactorial(2degree - 1)) / (2xpnLRsum)^degree) : one(T)
    T(πPowers[:p0d5]) / sqrt(xpnLRsum) * factor
end

function computePGTOrbOverlapMixedFactor(dxLR::T, xpnProdOverSum::T) where {T<:Real}
    exp(- xpnProdOverSum * dxLR^2)
end



#>-- Basic data structure --<#
struct PrimGaussTypeOrbInfo{T<:Real, D} <: QueryBox{T}
    cen::NTuple{D, T}
    xpn::T
    ang::NTuple{D, Int}

    function PrimGaussTypeOrbInfo(cen::NonEmptyTuple{Real, D}, xpn::Real, 
                                  ang::NonEmptyTuple{Int, D}) where {D}
        cen..., xpn = promote(cen..., xpn)
        new{typeof(xpn), D+1}(cen, xpn, ang)
    end
end

struct GaussProductInfo{T<:Real, D} <: QueryBox{T}
    lhs::PrimGaussTypeOrbInfo{T, D}
    rhs::PrimGaussTypeOrbInfo{T, D}
    cen::NTuple{D, T}
    xpn::T

    function GaussProductInfo(oData1::PrimGaussTypeOrbInfo{T, D}, 
                              oData2::PrimGaussTypeOrbInfo{T, D}) where {T, D}
        xpnSum = oData1.xpn + oData2.xpn
        cenNew = (oData1.xpn .* oData1.cen .+ oData2.xpn .* oData2.cen) ./ xpnSum
        new{T, D}(oData1, oData2, cenNew, xpnSum)
    end
end

#> Union of orbital types that can utilize `AxialGaussOverlapCache`
const GaussProdBasedOrb{T<:Real, D} = Union{
    FloatingPolyGaussField{T, D}
}

const GaussProdBasedOrbCache{T<:Real, D, F<:GaussProdBasedOrb{T, D}} = 
      LRU{EgalBox{F}, PrimGaussTypeOrbInfo{T, D}}

const T4Int2Tuple{T} = Tuple{T, T, T, T, Int, Int}

struct AxialGaussOverlapCache{T<:Real, D, M<:NTuple{D, MissingOr{ LRU{T4Int2Tuple{T}, T} }}, 
                              F<:GaussProdBasedOrb{T, D}} <: CustomCache{T}
    axis::M
    encode::GaussProdBasedOrbCache{T, D, F}

    function AxialGaussOverlapCache(::Type{F}, config::NTuple{D, BoolVal}, 
                                    axialMaxSize::Int=128) where 
                                   {T<:Real, D, F<:GaussProdBasedOrb{T, D}}
        axialCache = map(config) do axialConfig
            getValData(axialConfig) ? LRU{T4Int2Tuple{T}, T}(maxsize=axialMaxSize) : missing
        end
        orbCache = GaussProdBasedOrbCache{T, D, F}(maxsize=axialMaxSize)
        new{T, D, typeof(axialCache), F}(axialCache, orbCache)
    end
end

AxialGaussOverlapCache(::Type{F}, ::NTuple{D, Val{false}}, ::Int=128) where 
                      {T<:Real, D, F<:GaussProdBasedOrb{T, D}} = 
NullCache{T}()


const OptAxialGaussOverlapCache{T<:Real} = 
      Union{NullCache{T}, AxialGaussOverlapCache{T}}

accessAxialCache(cache::AxialGaussOverlapCache, i::Int) = cache.axis[begin+i-1]

accessAxialCache(cache::NullCache, ::Int) = cache



#>-- Gaussian-based orbital info extraction --<#
function prepareOrbitalInfoCore(field::FloatingPolyGaussField{T, D}) where {T<:Real, D}
    gtf = field.core.f.f                            #> `PolyGaussFieldCore`
    grf = first(gtf.encode)                         #> `RadialFieldFunc`
    pgf = last(grf.core.f.binder.f.encode)          #> `GaussFieldFunc`
    xpnFormatter = last(pgf.core.f.binder.f.encode) #> `ParamFormatter`
    xpn = xpnFormatter.core(field.param).xpn
    angMomField = last(gtf.encode)                  #> `CartAngMomFieldFunc`
    amfCore = angMomField.core.f.binder.f.core.f    #> `CartSHarmonics`
    ang = amfCore.m.tuple
    PrimGaussTypeOrbInfo(field.center, xpn, ang)
end

function prepareOrbitalInfo!(cache::AxialGaussOverlapCache{T, D}, 
                             orbData::FloatingPolyGaussField{T, D}) where {T<:Real, D}
    get!(cache.encode, EgalBox{FloatingPolyGaussField{T, D}}(orbData)) do
        prepareOrbitalInfoCore(orbData)
    end
end

function prepareOrbitalInfo!(::NullCache{T}, 
                             orbData::FloatingPolyGaussField{T, D}) where {T<:Real, D}
    prepareOrbitalInfoCore(orbData)
end

function prepareOrbitalInfo!(cache::OptAxialGaussOverlapCache{T}, 
                             orbsData::NTuple{2, FloatingPolyGaussField{T, D}}) where 
                            {T<:Real, D}
    map(orbsData) do orbData
        prepareOrbitalInfo!(cache, orbData)
    end |> Base.Splat(GaussProductInfo)
end



#>-- Cartesian PGTO overlap computation --<#
#> Overlap for 1D Gaussian-function pair
function computeGaussFuncOverlap(xpnSum::T, xpnPOS::T, dxLR::T) where {T<:Real}
    computePGTOrbOverlapAxialFactor(xpnSum, 0) * 
    computePGTOrbOverlapMixedFactor(dxLR, xpnPOS)
end

#> Overlap for concentric axial PGTO pair
function computeAxialPGTOrbOverlap(xpnSum::T, iL::Int, iR::Int) where {T<:Real}
    res = zero(T)
    for j in 0:((iL + iR) ÷ 2)
        res += computeGaussProd(iL, iR, 2j) * computePGTOrbOverlapAxialFactor(xpnSum, j)
    end
    res
end
#> Overlap for arbitrary axial PGTO pair
function computeAxialPGTOrbOverlap(xpnSum::T, xpnPOS::T, xML::T, xMR::T, 
                                   iL::Int, iR::Int) where {T<:Real}
    res = zero(T)
    for j in 0:((iL + iR) ÷ 2)
        res += computeGaussProd(xML, xMR, iL, iR, 2j) * 
               computePGTOrbOverlapAxialFactor(xpnSum, j)
    end
    res * computePGTOrbOverlapMixedFactor(xMR - xML, xpnPOS)
end
#> Adaptive axial-PGTO overlap computation
function computeAxialPGTOrbOverlap(input::T4Int2Tuple{T}) where {T<:Real}
    xpnSum, xpnPOS, xML, xMR, iL, iR = input
    dxLR = xMR - xML
    if iL == iR == 0
        computeGaussFuncOverlap(xpnSum, xpnPOS, dxLR)
    elseif isequal(dxLR, zero(T))
        computeAxialPGTOrbOverlap(xpnSum, iL, iR)
    else
        computeAxialPGTOrbOverlap(xpnSum, xpnPOS, xML, xMR, iL, iR)
    end
end

#> Cache-based axial-PGTO overlap computation
function computeAxialPGTOrbOverlap!(cache::LRU{T4Int2Tuple{T}, T}, 
                                    input::T4Int2Tuple{T}) where {T<:Real}
    res = get(cache, input, nothing) # Fewer allocations than using `get!`
    if res === nothing
        res = computeAxialPGTOrbOverlap(input)
        setindex!(cache, res, input)
    end
    res
end

computeAxialPGTOrbOverlap!(::NullCache{T}, input::T4Int2Tuple{T}) where {T<:Real} = 
computeAxialPGTOrbOverlap(input)

#> Internal overlap computation
function computePGTOrbOverlap(data::PrimGaussTypeOrbInfo{T}) where {T<:Real}
    xpn = data.xpn
    ang = data.ang

    res = one(T)
    for i in ang
        res *= computePGTOrbOverlapAxialFactor(2xpn, i)
    end
    res
end

function computePGTOrbOverlap!(cache::OptAxialGaussOverlapCache{T}, 
                               data::GaussProductInfo{T, D}) where {T<:Real, D}
    cenL = data.lhs.cen
    cenR = data.rhs.cen
    cenM = data.cen
    angL = data.lhs.ang
    angR = data.rhs.ang
    xpnSum = data.xpn
    xpnPOS = data.lhs.xpn * data.rhs.xpn / xpnSum

    i = 0
    res = one(T)
    for (xL, xR, xM, iL, iR) in zip(cenL, cenR, cenM, angL, angR)
        axialCache = accessAxialCache(cache, (i += 1))
        data = (xpnSum, xpnPOS, xM-xL, xM-xR, iL, iR)
        res *= computeAxialPGTOrbOverlap!(axialCache, data)
    end

    res
end



#>-- Cartesian PGTO multipole-moment computation --<#
function computePGTOrbMultipoleMoment(fm::FloatingMonomial{T, D}, 
                                      data::PrimGaussTypeOrbInfo{T, D}) where {T<:Real, D}
    xpn = data.xpn
    res = one(T)
    for (xMM, n, x, i) in zip(fm.center, fm.degree.tuple, data.cen, data.ang)
        dx = x - xMM
        m = iszero(dx) ? 0 : n

        temp = zero(T)
        for k in 0:m
            l = n - k
            if iseven(l)
                factor = binomial(n, k) * dx^k
                temp += factor * computePGTOrbOverlapAxialFactor(2xpn, i + (l >> 1))
            end
        end

        res *= temp
    end
    res
end

function computePGTOrbMultipoleMoment!(fm::FloatingMonomial{T, D}, 
                                       cache::OptAxialGaussOverlapCache{T}, 
                                       data::GaussProductInfo{T, D}) where {T<:Real, D}
    if fm.degree.total == 0
        computePGTOrbOverlap!(cache, data)
    else
        cenL = data.lhs.cen
        cenR = data.rhs.cen
        cenM = data.cen
        angL = data.lhs.ang
        angR = data.rhs.ang
        xpnSum = data.xpn
        xpnPOS = data.lhs.xpn * data.rhs.xpn / xpnSum

        i = 0
        res = one(T)
        for (xMM, n, xL, xR, xM, iL, iR) in zip(fm.center, fm.degree.tuple, 
                                                cenL, cenR, cenM, angL, angR)
            axialCache = accessAxialCache(cache, (i += 1))
            dx = xR - xMM
            m = if iszero(dx)
                0
            elseif isequal(xL, xMM)
                xL, xR = xR, xL
                iL, iR = iR, iL
                0
            else
                n
            end

            temp = zero(T)
            for k in 0:m
                data = (xpnSum, xpnPOS, xM-xL, xM-xR, iL, iR+n-k)
                temp += binomial(n, k) * dx^k * computeAxialPGTOrbOverlap!(axialCache, data)
            end

            res *= temp
        end
        res
    end
end



#>-- Core integral-evaluation function --<#
#> Overlap
function evaluateOverlap((data,)::Tuple{FloatingPolyGaussField{T, D}}; 
                         cache!Self::OptAxialGaussOverlapCache{T}=NullCache{T}()
                         ) where {T<:Real, D}
    formattedData = prepareOrbitalInfo!(cache!Self, data)
    computePGTOrbOverlap(formattedData)
end

function evaluateOverlap(data::NTuple{2, FloatingPolyGaussField{T, D}}; 
                         cache!Self::OptAxialGaussOverlapCache{T}=NullCache{T}()
                         ) where {T<:Real, D}
    formattedData = prepareOrbitalInfo!(cache!Self, data)
    computePGTOrbOverlap!(cache!Self, formattedData)
end

#> Multipole moment
function evaluateMultipoleMoment(op::MultipoleMomentSampler{T, D}, 
                                 (data,)::Tuple{FloatingPolyGaussField{T, D}};
                                 cache!Self::OptAxialGaussOverlapCache{T}=NullCache{T}()
                                 ) where {T<:Real, D}
    formattedData = prepareOrbitalInfo!(cache!Self, data)
    computePGTOrbMultipoleMoment(last(op.dresser).term, formattedData)
end

function evaluateMultipoleMoment(op::MultipoleMomentSampler{T, D}, 
                                 data::NTuple{2, FloatingPolyGaussField{T, D}}; 
                                 cache!Self::OptAxialGaussOverlapCache{T}=NullCache{T}()
                                 ) where {T<:Real, D}
    formattedData = prepareOrbitalInfo!(cache!Self, data)
    computePGTOrbMultipoleMoment!(last(op.dresser).term, cache!Self, formattedData)
end



#>-- Interface with the composite integration framework --<#
getGaussProdBasedIntegrator(::OneBodyIntegral, ::OverlapSampler) = evaluateOverlap

getGaussProdBasedIntegrator(::OneBodyIntegral{D}, op::MultipoleMomentSampler{T, D}
                            ) where {T<:Real, D} = 
LPartial(evaluateMultipoleMoment, (op,))


function getAnalyticIntegral!(::S, cache!Self::OptAxialGaussOverlapCache{T}, 
                              op::DirectOperator, 
                              data::OneBodyOrbIntLayout{PGTOrbData{T, D}}) where 
                             {D, S<:MultiBodyIntegral{D}, T<:Real}
    fields = getfield.(data, :core)
    integrator = getGaussProdBasedIntegrator(S(), op)
    integrator(fields; cache!Self)
end


const AxialGaussOverlapCachedPGTOrbSampler{T<:Real, D} = 
      Union{OverlapSampler, MultipoleMomentSampler{T, D}, DiagDirectionalDiffSampler{T, D}}

#= Additional Method =#
getAnalyticIntegralCache(::AxialGaussOverlapCachedPGTOrbSampler{T, D}, 
                         ::OneBodyOrbIntLayout{PGTOrbData{T, D}}) where {T<:Real, D} = 
AxialGaussOverlapCache(FloatingPolyGaussField{T, D}, ntuple( _->Val(true), Val(D) ))


#> Union of sampler types with corresponding analytic Gaussian integrals implemented
const AnalyticGaussIntegralSampler{T, D} = Union{
    AxialGaussOverlapCachedPGTOrbSampler{T, D}
}