using LinearAlgebra: norm
using LRUCache

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
    params = field.data
    fCenter, gtf = field.core.f.f.encode          #> `PolyGaussFieldCore` (gtf)
    center = last(fCenter.encode).core(params)
    grf = first(gtf.encode)                       #> `RadialFieldFunc`
    pgf = last(grf.core.f.binder.encode)          #> `GaussFieldFunc`
    xpnFormatter = last(pgf.core.f.binder.encode) #> `ParamFormatter`
    xpn = xpnFormatter.core(params).xpn
    angMomField = last(gtf.encode)                #> `CartAngMomentumFunc`
    amfCore = angMomField.core.f.binder.core.f    #> `CartSHarmonics`
    ang = amfCore.m.tuple
    PrimGaussTypeOrbInfo(center, xpn, ang)
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
#> Reusable axial factor
function computePGTOrbOverlapAxialFactor(xpnLRsum::T, degree::Int) where {T<:Real}
    factor = degree > 0 ? oddFactorial(2degree - 1, inv(2xpnLRsum)) : one(T)
    T(πPowers[:p0d5]) / sqrt(xpnLRsum) * factor
end
#> Reusable mixed factor
function computePGTOrbOverlapMixedFactor(dxLR::T, xpnProdOverSum::T) where {T<:Real}
    exp(- xpnProdOverSum * dxLR^2)
end

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



#>-- Cartesian PGTO coordinate-differentiation computation --<#
#> Generator of axial PGTO-pair data with shifted right-side angular momentum component
function shiftRightAngularNum(data::T4Int2Tuple{T}, shift::Int) where {T}
    xpnData1, xpnData2, disData1, disData2, angDataL, angDataR = data
    (xpnData1, xpnData2, disData1, disData2, angDataL, angDataR + shift)
end
#> Coordinate differentiation for arbitrary axial PGTO pair
function computeAxialPGTOrbCoordDiff!(degree::Int, 
                                      cache::LRU{T4Int2Tuple{T}, T}, 
                                      input::T4Int2Tuple{T}) where {T<:Real}
    xpnL, xpnR, xML, xMR, iL, iR = input

    if degree < 1
        xpnSum = xpnL + xpnR
        xpnPOS = xpnL * xpnR / xpnSum
        overlapInput = (xpnSum, xpnPOS, xML, xMR, iL, iR)
        computeAxialPGTOrbOverlap!(cache, overlapInput)
    else
        degreeInner = degree - 1
        termDn = if iR < 1
            zero(T)
        else
            dataDn = shiftRightAngularNum(input, -1)
            computeAxialPGTOrbCoordDiff!(degreeInner, cache, dataDn)
        end
        dataUp = shiftRightAngularNum(input, +1)
        termUp = computeAxialPGTOrbCoordDiff!(degreeInner, cache, dataUp)

        termDn * iR - termUp * 2xpnR
    end
end
#> Coordinate differentiation for concentric axial PGTO pair with equal exponent
function computeAxialPGTOrbCoordDiff(degree::Int, xpn::T, iL::Int, iR::Int) where {T<:Real}
    if degree < 1
        computeAxialPGTOrbOverlap(2xpn, iL, iR)
    else
        degreeInner = degree - 1
        termDn = iR < 1 ? zero(T) : computeAxialPGTOrbCoordDiff(degreeInner, xpn, iL, iR-1)
        termUp = computeAxialPGTOrbCoordDiff(degreeInner, xpn, iL, iR+1)
        termDn * iR - termUp * 2xpn
    end
end

#> Internal coordinate-differentiation computation
function computePGTOrbCoordDiff(degrees::NTuple{D, Int}, 
                                data::PrimGaussTypeOrbInfo{T, D}, 
                                axialWise::Bool=false) where {T<:Real, D}
    xpn = data.xpn
    ang = data.ang

    axialNums = ntuple(itself, Val(D))
    res = ntuple(_->one(T), Val(D))
    idxOffset = firstindex(res) - 1
    for (i, degree) in zip(ang, degrees)
        iHead, iBody... = axialNums
        axialNums = rightCircShift(axialNums)
        tempPosDiffRes = computeAxialPGTOrbCoordDiff(degree, xpn, i, i)
        res = setIndex(res, tempPosDiffRes, iHead+idxOffset, *)

        if !axialWise
            tempOverlapRes = computePGTOrbOverlapAxialFactor(2xpn, i)

            for i in iBody
                res = setIndex(res, tempOverlapRes, i+idxOffset, *)
            end
        end
    end
    res
end

function computePGTOrbCoordDiff!(degrees::NTuple{D, Int}, 
                                 cache::OptAxialGaussOverlapCache{T}, 
                                 data::GaussProductInfo{T, D}, 
                                 axialWise::Bool=false) where {T<:Real, D}
    cenL = data.lhs.cen
    cenR = data.rhs.cen
    cenM = data.cen
    angL = data.lhs.ang
    angR = data.rhs.ang
    xpnL = data.lhs.xpn
    xpnR = data.rhs.xpn
    if !axialWise
        xpnSum = data.xpn
        xpnPOS = xpnL * xpnR / xpnSum
    end

    axialNums = ntuple(itself, Val(D))
    res = ntuple(_->one(T), Val(D))
    idxOffset = firstindex(res) - 1
    for (degree, xL, xR, xM, iL, iR) in zip(degrees, cenL, cenR, cenM, angL, angR)
        iHead, iBody... = axialNums
        axialNums = rightCircShift(axialNums)
        axialCache = accessAxialCache(cache, iHead)

        posDiffData = (xpnL, xpnR, xM-xL, xM-xR, iL, iR)
        tempPosDiffRes = computeAxialPGTOrbCoordDiff!(degree, axialCache, posDiffData)
        res = setIndex(res, tempPosDiffRes, iHead+idxOffset, *)

        if !axialWise
            overlapData = (xpnSum, xpnPOS, xM-xL, xM-xR, iL, iR)
            tempOverlapRes = computeAxialPGTOrbOverlap!(axialCache, overlapData)

            for i in iBody
                res = setIndex(res, tempOverlapRes, i+idxOffset, *)
            end
        end
    end
    res
end



#>-- Core integral-evaluation function --<#
#> Overlap
function computePGTOrbIntegral(::OverlapSampler, 
                               (data,)::Tuple{FloatingPolyGaussField{T, D}}, 
                               cache!Self::OptAxialGaussOverlapCache{T}=NullCache{T}()
                               ) where {T<:Real, D}
    formattedData = prepareOrbitalInfo!(cache!Self, data)
    computePGTOrbOverlap(formattedData)
end

function computePGTOrbIntegral(::OverlapSampler, 
                               data::NTuple{2, FloatingPolyGaussField{T, D}}, 
                               cache!Self::OptAxialGaussOverlapCache{T}=NullCache{T}()
                               ) where {T<:Real, D}
    formattedData = prepareOrbitalInfo!(cache!Self, data)
    computePGTOrbOverlap!(cache!Self, formattedData)
end

#> Multipole moment
function computePGTOrbIntegral(op::MultipoleMomentSampler{T, D}, 
                               (data,)::Tuple{FloatingPolyGaussField{T, D}}, 
                               cache!Self::OptAxialGaussOverlapCache{T}=NullCache{T}()
                               ) where {T<:Real, D}
    formattedData = prepareOrbitalInfo!(cache!Self, data)
    computePGTOrbMultipoleMoment(last(op.dresser).term, formattedData)
end

function computePGTOrbIntegral(op::MultipoleMomentSampler{T, D}, 
                               data::NTuple{2, FloatingPolyGaussField{T, D}}, 
                               cache!Self::OptAxialGaussOverlapCache{T}=NullCache{T}()
                               ) where {T<:Real, D}
    formattedData = prepareOrbitalInfo!(cache!Self, data)
    computePGTOrbMultipoleMoment!(last(op.dresser).term, cache!Self, formattedData)
end

#> Diagonal-directional differentiation (∑ᵢ(cᵢ ⋅ ∂ᵐ/∂xᵢᵐ))
function computePGTOrbIntegral(op::DiagDirectionalDiffSampler{T, D, M}, 
                               (data,)::Tuple{FloatingPolyGaussField{T, D}}, 
                               cache!Self::OptAxialGaussOverlapCache{T}=NullCache{T}()
                               ) where {T<:Real, D, M}
    formattedData = prepareOrbitalInfo!(cache!Self, data)
    diffVec = computePGTOrbCoordDiff(ntuple(_->M, Val(D)), formattedData)
    direction = last(op.dresser).direction
    mapreduce(StableMul(T), StableAdd(T), direction, diffVec)
end


function computePGTOrbIntegral(op::DiagDirectionalDiffSampler{T, D, M}, 
                               data::NTuple{2, FloatingPolyGaussField{T, D}}, 
                               cache!Self::OptAxialGaussOverlapCache{T}=NullCache{T}()
                               ) where {T<:Real, D, M}
    formattedData = prepareOrbitalInfo!(cache!Self, data)
    diffVec = computePGTOrbCoordDiff!(ntuple(_->M, Val(D)), cache!Self, formattedData)
    direction = last(op.dresser).direction
    mapreduce(StableMul(T), StableAdd(T), direction, diffVec)
end



#>-- Interface with the composite integration framework --<#
const AxialGaussOverlapBasedSampler{T<:Real, D} = Union{
    OverlapSampler, 
    MultipoleMomentSampler{T, D}, 
    DiagDirectionalDiffSampler{T, D}
}

function getGaussBasedIntegrationCache(::Type{F}, ::Type{T}, ::Val{D}) where 
                                      {T<:Real, D, F<:AxialGaussOverlapBasedSampler{T, D}}
    AxialGaussOverlapCache(FloatingPolyGaussField{T, D}, ntuple( _->Val(true), Val(D) ))
end

supportGaussBasedIntegration(::Type{T}, ::Val{D}, 
                             ::AxialGaussOverlapBasedSampler{T, D}) where {T<:Real, D} = 
true

supportGaussBasedIntegration(::Type{T}, ::Val{3}, 
                             ::CoulombRepulsionSampler) where {T<:Real} = 
true

supportGaussBasedIntegration(::Type{T}, ::Val{D}, ::DirectOperator) where {T<:Real, D} = 
false

#= Additional Method =#
function evaluateIntegral!(integrator::OrbitalCoreIntegralConfig{T, D, C, N, F}, 
                           pairwiseData::NTuple{N, N12Tuple{ PGTOrbData{T, D} }}) where 
                          {T, C<:RealOrComplex{T}, D, N, F<:DirectOperator}
    op = integrator.operator
    cacheDict = integrator.cache
    if supportGaussBasedIntegration(T, Val(D), op)
        innerData = map(x->getfield.(x, :core), pairwiseData)
        res = if cacheDict isa NullCache
            computePGTOrbIntegral(op, innerData...)
        else
            orbLayout = ntuple(_->(PrimGaussTypeOrb, PrimGaussTypeOrb), Val(N))
            key = (TypeBox(F), orbLayout)::OrbIntLayoutInfo{N}
            cache = get!(cacheDict, key) do
                getGaussBasedIntegrationCache(F, T, Val(D))
            end
            computePGTOrbIntegral(op, innerData..., cache)
        end
        convert(C, res)
    else
        evaluateIntegralCore(integrator, pairwiseData)
    end
end