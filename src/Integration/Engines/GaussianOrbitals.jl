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
    symmetric::Bool

    function GaussProductInfo((oDataL, oDataR)::NTuple{2, PrimGaussTypeOrbInfo{T, D}}, 
                              ) where {T, D}
        xpnSum = oDataL.xpn + oDataR.xpn
        symmetric = if oDataL == oDataR
            cenNew = oDataL.cen
            true
        else
            cenNew = (oDataL.xpn .* oDataL.cen .+ oDataR.xpn .* oDataR.cen) ./ xpnSum
            false
        end
        new{T, D}(oDataL, oDataR, cenNew, xpnSum, symmetric)
    end
end

#> Union of orbital types that can utilize `AxialGaussOverlapCache`
const GaussProdBasedOrb{T<:Real, D} = Union{
    FloatingPolyGaussField{T, D}
}

const GaussProdBasedOrbCache{T<:Real, D, F<:GaussProdBasedOrb{T, D}} = 
      LRU{EgalBox{F}, PrimGaussTypeOrbInfo{T, D}}

const T4Int2Tuple{T} = Tuple{T, T, T, T, Int, Int}

struct AxialGaussOverlapCache{T<:Real, D, M<:NTuple{D, OptionalLRU{T4Int2Tuple{T}, T}}, 
                              } <: QueryCache{T4Int2Tuple{T}, T}
    axis::M

    function AxialGaussOverlapCache(::Type{T}, configs::NTuple{D, Boolean}, 
                                    axialMaxSize::Int=128) where {T<:Real, D}
        axialCache = map(configs) do config
            if evalTypedData(config); LRU{T4Int2Tuple{T}, T}(maxsize=axialMaxSize) else
               EmptyDict{T4Int2Tuple{T}, T}() end
        end
        new{T, D, typeof(axialCache)}(axialCache)
    end
end

function AxialGaussOverlapCache(::Type{T}, ::Count{D}) where {T<:Real, D}
    checkPositivity(D)
    AxialGaussOverlapCache(T, ntuple( _->False(), Val(D) ))
end



#>-- Gaussian-based orbital info extraction --<#
function prepareOrbitalInfoCore(field::FloatingPolyGaussField{T, D}) where {T<:Real, D}
    params = field.data
    shiftedField = field.core.f.f                 #> `ShiftedFieldFuncCore`
    fCenter = shiftedField.inner                  #> `FieldCenterShifter`
    gtf = shiftedField.outer                      #> `PolyGaussFieldCore`
    center = last(fCenter.encode).core(params)
    grf, angMomField = gtf.encode                 #> `RadialFieldFunc` (grf)
    pgf = grf.core.f.binder.outer                 #> `GaussFieldFunc`
    xpnFormatter = last(pgf.core.f.binder.encode) #> `ParamFormatter`
    xpn = xpnFormatter.core(params).xpn
    amfCore = angMomField.core.f.binder.core.f    #> `CartSHarmonics`
    ang = amfCore.m.value
    PrimGaussTypeOrbInfo(center, xpn, ang)
end

function prepareOrbitalInfo((data,)::N1N2Tuple{FloatingPolyGaussField{T, D}}) where 
                           {T<:Real, D}
    lazyMap(data) do orbData
        prepareOrbitalInfoCore(orbData)
    end |> GaussProductInfo
end

function prepareOrbitalInfo((data1, data2)::N2N2Tuple{FloatingPolyGaussField{T, D}}) where 
                           {T<:Real, D}
    i, j, k, l = lazyMap((data1..., data2...)) do orbData
        prepareOrbitalInfoCore(orbData)
    end
    GaussProductInfo((i, j)), GaussProductInfo((k, l))
end



#>-- Cartesian PGTO overlap computation --<#
#> Reusable axial factor
function computePGTOrbOverlapAxialFactor(xpnLRsum::T, degree::Int) where {T<:Real}
    factor = degree > 0 ? oddFactorial(2degree - 1, inv(2xpnLRsum)) : one(T)
    T(PowersOfPi[:p0d5]) / sqrt(xpnLRsum) * factor
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

function computeAxialPGTOrbOverlap!(::EmptyDict{T4Int2Tuple{T}, T}, 
                                    input::T4Int2Tuple{T}) where {T<:Real}
    computeAxialPGTOrbOverlap(input)
end

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

function computePGTOrbOverlap!(cache::AxialGaussOverlapCache{T}, 
                               data::GaussProductInfo{T, D}) where {T<:Real, D}
    if data.symmetric
        computePGTOrbOverlap(data.lhs)
    else
        cenL = data.lhs.cen
        cenR = data.rhs.cen
        cenM = data.cen
        angL = data.lhs.ang
        angR = data.rhs.ang
        xpnSum = data.xpn
        xpnPOS = data.lhs.xpn * data.rhs.xpn / xpnSum

        res = one(T)
        for (xL, xR, xM, iL, iR, sector) in zip(cenL, cenR, cenM, angL, angR, cache.axis)
            data = (xpnSum, xpnPOS, xM-xL, xM-xR, iL, iR)
            res *= computeAxialPGTOrbOverlap!(sector, data)
        end

        res
    end
end



#>-- Cartesian PGTO multipole-moment computation --<#
function computePGTOrbMultipoleMoment(fm::FloatingMonomial{T, D}, 
                                      data::PrimGaussTypeOrbInfo{T, D}) where {T<:Real, D}
    xpn = data.xpn
    res = one(T)
    for (xMM, n, x, i) in zip(fm.center, fm.degree.value, data.cen, data.ang)
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

function computePGTOrbMultipoleMoment!(cache::AxialGaussOverlapCache{T}, 
                                       fm::FloatingMonomial{T, D}, 
                                       data::GaussProductInfo{T, D}) where {T<:Real, D}
    if fm.degree.total == 0
        computePGTOrbOverlap!(cache, data)
    elseif data.symmetric
        computePGTOrbMultipoleMoment(fm, data.lhs)
    else
        cenL = data.lhs.cen
        cenR = data.rhs.cen
        cenM = data.cen
        angL = data.lhs.ang
        angR = data.rhs.ang
        xpnSum = data.xpn
        xpnPOS = data.lhs.xpn * data.rhs.xpn / xpnSum

        res = one(T)
        for (xMM, n, xL, xR, xM, iL, iR, sector) in zip(fm.center, fm.degree.value, 
                                                        cenL, cenR, cenM, angL, angR, 
                                                        cache.axis)
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
                temp += binomial(n, k) * dx^k * computeAxialPGTOrbOverlap!(sector, data)
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
                                      cache::OptionalLRU{T4Int2Tuple{T}, T}, 
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

function computePGTOrbCoordDiff!(cache::AxialGaussOverlapCache{T}, 
                                 degrees::NTuple{D, Int}, 
                                 data::GaussProductInfo{T, D}, 
                                 axialWise::Bool=false) where {T<:Real, D}
    if data.symmetric
        computePGTOrbCoordDiff(degrees, data.lhs, axialWise)
    else
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
        for (degree, xL, xR, xM, iL, iR, sector) in zip(degrees, cenL, cenR, cenM, 
                                                        angL, angR, cache.axis)
            iHead, iBody... = axialNums
            axialNums = rightCircShift(axialNums)

            posDiffData = (xpnL, xpnR, xM-xL, xM-xR, iL, iR)
            tempPosDiffRes = computeAxialPGTOrbCoordDiff!(degree, sector, posDiffData)
            res = setIndex(res, tempPosDiffRes, iHead+idxOffset, *)

            if !axialWise
                overlapData = (xpnSum, xpnPOS, xM-xL, xM-xR, iL, iR)
                tempOverlapRes = computeAxialPGTOrbOverlap!(sector, overlapData)

                for i in iBody
                    res = setIndex(res, tempOverlapRes, i+idxOffset, *)
                end
            end
        end
        res
    end
end



#>-- Cartesian-3D PGTO One-Body Coulomb-field computation based on Obara-Saika scheme --<#
function computePGTOrbMixedFactorProd(xpnPOS::T, cenL::NTuple{N, T}, cenR::NTuple{N, T}
                                      ) where {T<:Real, N}
    mapreduce(*, cenL, cenR) do xL, xR
        computePGTOrbOverlapMixedFactor(xL-xR, xpnPOS)
    end
end
#> notation iNxnPy: (iL-x, n+y)
#>> (iL,   n, iR)                 #>> iN0nP0
#>> (iL-1, n, iR) (iL-1, n+1, iR) #>> iN1nP0 iN1nP1
#>> (iL-2, n, iR) (iL-2, n+1, iR) #>> iN2nP0 iN2nP1
function vertRec((iN1nP0, iN1nP1, iN2nP0, iN2nP1)::NTuple{4, T}, 
                 xpnSum::T, iL::Int, xML::T, xMC::T, factor::T=one(T)) where {T<:Real}
    part1 = xML * iN1nP0 - factor * xMC * iN1nP1
    part2 = (iL-1) * (iN2nP0 - factor * iN2nP1) * inv(2xpnSum)
    part1 + part2
end
function horiRec(xLR::T, iN0nP0::T, iN1nP0::T) where {T<:Real}
    iN0nP0 + xLR * iN1nP0 # [(iL, n, iR-1), (iL-1, n, iR-1)] -> (iL-1, n, iR)
end
#>> n -> n - (iL + iR)
#>> (0, 0, jL, jR, kL, kR) -> (iL+iR, 0, jL, jR, kL, kR)
function verticalFill!(horiBuffer::AbstractVector{T}, xpnSum::T, xML::T, xMC::T, nMax::Int, 
                       factor::T=one(T)) where {T<:Real}
    for n in 1:nMax
        here = zero(T)
        buffer = (horiBuffer[end-n], horiBuffer[end-n+1], zero(T), zero(T))

        for iSum in 1:n
            iN1nP0, iN1nP1, _, _ = buffer
            here = vertRec(buffer, xpnSum, iSum, xML, xMC, factor)
            iSum < n && (buffer = (here, horiBuffer[end-n+iSum+1], iN1nP0, iN1nP1))
            horiBuffer[end-n+iSum] = here
        end
    end

    @view horiBuffer[end-nMax : end]
end


function verticalPush!(segmentNext::AbstractVector{T}, segmentHere::AbstractVector{T}, 
                       xpnSum::T, xML::T, xMC::T, iSum::Int, factor::T=one(T)
                       ) where {T<:Real}
    # @assert iSum+1 == length(segmentNext) == length(segmentHere)
    buffer = (segmentNext[begin], segmentHere[begin], zero(T), zero(T))

    for i in 1:iSum
        here = vertRec(buffer, xpnSum, i, xML, xMC, factor)
        segmentNext[begin+i] = here
        if i < iSum
            a, b, _, _ = buffer
            buffer = (here, segmentHere[begin+i], a, b)
        end
    end

    segmentNext
end
#>> (iL+iR, n, 0) -> (iL, n, iR)
function angularShift!(vertBuffer::AbstractVector{T}, xLR::T, iR::Int) where {T<:Real}
    segment = @view vertBuffer[end-iR:end]

    for y in 1:iR, x in 1:(iR + 1 - y)
        segment[begin+x-1] = horiRec(xLR, segment[begin+x], segment[begin+x-1])
    end

    first(segment)
end

function computePGTOrbPointCoulombField(pointCharge::Pair{Int, NTuple{3, T}}, 
                                        data::GaussProductInfo{T, D}) where {T<:Real, D}
    charge, cenC = pointCharge
    cenL = data.lhs.cen
    cenR = data.rhs.cen
    cenM = data.cen
    drMC = cenM .- cenC
    angL = data.lhs.ang
    angR = data.rhs.ang
    angAxialSum = angL .+ angR
    ijkSum = sum(angAxialSum)
    xpnSum = data.xpn
    xpnPOS = data.lhs.xpn * data.rhs.xpn / xpnSum

    prefactor = computePGTOrbMixedFactorProd(xpnPOS, cenL, cenR) / xpnSum
    factor = -charge * 2T(PowersOfPi[:p1d0]) * prefactor

    horiBuffer = computeBoysSequence(xpnSum * mapreduce(x->x*x, +, drMC), ijkSum)
    vertBuffer = copy(horiBuffer)
    nUpper = ijkSum

    for (nMax, iR, xL, xR, xM, xMC) in zip(angAxialSum, angR, cenL, cenR, cenM, drMC)
        xML = xM - xL
        xLR = xL - xR

        nShiftBound = nUpper - nMax
        vertSegHere = verticalFill!(horiBuffer, xpnSum, xML, xMC, nMax)

        for shiftMin in 1:nShiftBound
            shiftMax = shiftMin + nMax
            vertSegNext = @view vertBuffer[end-shiftMax : end-shiftMin]
            verticalPush!(vertSegNext, vertSegHere, xpnSum, xML, xMC, nMax)
            horiBuffer[end-shiftMin+1] = angularShift!(vertSegHere, xLR, iR)
            vertSegHere = @view horiBuffer[end-shiftMax : end-shiftMin]
            vertSegHere .= vertSegNext
        end

        horiBuffer[end-nShiftBound] = angularShift!(vertSegHere, xLR, iR)
        vertBuffer .= horiBuffer
        nUpper -= nMax
    end

    factor * last(horiBuffer)
end



#>-- Cartesian-3D PGTO One-Body Coulomb-field computation based on Obara-Saika scheme --<#
#>> [(iL,0|oL-2,0), (iL+1,0|oL-1,0), (iL,0|oL-1,0), (iL-1,0|oL-1,0)] -> (iL,0|oL,0)
#>>              (iL, oL-2)               <Head>         iN1oN2
#>> (iL-1, oL-1) (iL, oL-1) (iL+1, oL-1)  <Body>  iN2oN1 iN1oN1 iP0oN1
#>>              (iL, oL  )               <Here>         iN1oP0
function modeTransfer((iN1oN2, iP0oN1, iN1oN1, iN2oN1)::NTuple{4, T}, xLRPair::NTuple{2, T}, 
                      xpnSums::NTuple{2, T}, xpnPair::NTuple{2, T}, angPair::NTuple{2, Int}, 
                      ) where {T<:Real}
    i, o = angPair
    xLR1, xLR2 = xLRPair
    xpnR1, xpnR2 = xpnPair
    xpnSum1, xpnSum2 = xpnSums
    part1 = (i*iN2oN1 + (o - 1)*iN1oN2) / (2xpnSum2)
    part2 = ((xpnR1*xLR1 + xpnR2*xLR2)*iN1oN1 + xpnSum1*iP0oN1) / xpnSum2
    part1 - part2
end

function angularShift!(holder::AbstractMatrix{T}, source::AbstractVector{T}, 
                       xLRPair::NTuple{2, T}, xpnSums::NTuple{2, T}, xpnPair::NTuple{2, T}, 
                       oMax::Int) where {T<:Real}
    angSpace = length(source)
    # @assert 0 <= oMax < angSpace
    # @assert all((oMax+1, angSpace) .<= size(holder))
    activeHolder = @view holder[begin:begin+oMax, begin:begin+angSpace-1]
    activeHolder[begin, :] .= source

    for o in 1:oMax
        n = angSpace - o - 1
        iP0oN1 = activeHolder[begin+o-1, begin+n+1]
        iN1oN1 = activeHolder[begin+o-1, begin+n  ]
        iN2oN1 = n > 0 ? activeHolder[begin+o-1, begin+n-1] : zero(T)
        iN1oN2 = o > 1 ? activeHolder[begin+o-2, begin+n  ] : zero(T)

        for i in n:-1:0
            buffer = (iN1oN2, iP0oN1, iN1oN1, iN2oN1)
            iN1oP0 = modeTransfer(buffer, xLRPair, xpnSums, xpnPair, (i, o))
            activeHolder[begin+o, begin+i] = iN1oP0
            iP0oN1 = iN1oN1
            iN1oN1 = iN2oN1
            iN2oN1 = i > 1 ? activeHolder[begin+o-1, begin+i-2] : zero(T)
            iN1oN2 = (i > 0 && o > 1) ? activeHolder[begin+o-2, begin+i-1] : zero(T)
        end
    end

    @view activeHolder[:, begin:begin+angSpace-oMax-1]
end

function orbitalShift!(buffer::AbstractMatrix{T}, vertSegment::AbstractVector{T}, 
                       xpnSums::NTuple{2, T}, xpnPair::NTuple{2, T}, 
                       xLRPair::NTuple{2, T}, (iR, oR)::NTuple{2, Int}, oMax::Int
                       ) where {T<:Real}
    xLR1, xLR2 = xLRPair
    activeBuffer = angularShift!(buffer, vertSegment, xLRPair, xpnSums, xpnPair, oMax)
    for (n, slot) in zip(eachindex(vertSegment), eachcol(activeBuffer))
        vertSegment[n] = angularShift!(slot, xLR2, oR)
    end
    angRshifted = @view vertSegment[begin : end-oMax]
    # @assert length(angRshifted) == size(activeBuffer, 2)
    angularShift!(angRshifted, xLR1, iR)
end

function computePGTOrbTwoBodyRepulsion(data1::GaussProductInfo{T, D}, 
                                       data2::GaussProductInfo{T, D}) where {T<:Real, D}
    cenL1 = data1.lhs.cen
    cenR1 = data1.rhs.cen
    cenM1 = data1.cen
    angL1 = data1.lhs.ang
    angR1 = data1.rhs.ang
    xpnR1 = data1.rhs.xpn
    xpnSum1 = data1.xpn
    xpnPOS1 = data1.lhs.xpn * xpnR1 / xpnSum1
    ijKAxialSum = angL1 .+ angR1

    cenL2 = data2.lhs.cen
    cenR2 = data2.rhs.cen
    cenM2 = data2.cen
    angL2 = data2.lhs.ang
    angR2 = data2.rhs.ang
    xpnR2 = data2.rhs.xpn
    xpnSum2 = data2.xpn
    xpnPOS2 = data2.lhs.xpn * xpnR2 / xpnSum2
    opqAxialSum = angL2 .+ angR2

    angAxialSum = ijKAxialSum .+ opqAxialSum
    angSum = sum(angAxialSum)
    drM1M2 = cenM1 .- cenM2
    xpnPOS = xpnSum1 * xpnSum2 / (xpnSum1 + xpnSum2)
    xpnFactor = xpnPOS / xpnSum1
    xpnSumPair = (xpnSum1, xpnSum2)
    xpnRPair = (xpnR1, xpnR2)

    prefactor1 = computePGTOrbMixedFactorProd(xpnPOS1, cenL1, cenR1) / xpnSum1
    prefactor2 = computePGTOrbMixedFactorProd(xpnPOS2, cenL2, cenR2) / xpnSum2
    factor = 2T(PowersOfPi[:p2d5]) * prefactor1 * prefactor2 / sqrt(xpnSum1 + xpnSum2)

    horiBuffer = computeBoysSequence(xpnPOS * mapreduce(x->x*x, +, drM1M2), angSum)
    vertBuffer = copy(horiBuffer)
    modeBuffer = ShapedMemory{T}(undef, (maximum(opqAxialSum)+1, maximum(angAxialSum)+1))
    nUpper = angSum

    for (ioMax, oMax, iR, xL1, xR1, oR, xL2, xR2, xM1, xM1M2) in zip(angAxialSum, 
         opqAxialSum, angR1, cenL1, cenR1, angR2, cenL2, cenR2, cenM1, drM1M2)
        xML1 = xM1 - xL1
        angRPair = (iR, oR)
        xLRPair = (xL1 - xR1, xL2 - xR2)

        nShiftBound = nUpper - ioMax
        vertSegHere = verticalFill!(horiBuffer, xpnSum1, xML1, xM1M2, ioMax, xpnFactor)

        for shiftMin in 1:nShiftBound
            shiftMax = shiftMin + ioMax
            vertSegNext = @view vertBuffer[end-shiftMax : end-shiftMin]
            verticalPush!(vertSegNext, vertSegHere, xpnSum1, xML1, xM1M2, ioMax, xpnFactor)
            horiBuffer[end-shiftMin+1] = orbitalShift!(modeBuffer, vertSegHere, xpnSumPair, 
                                                       xpnRPair, xLRPair, angRPair, oMax)
            vertSegHere = @view horiBuffer[end-shiftMax : end-shiftMin]
            vertSegHere .= vertSegNext
        end

        horiBuffer[end-nShiftBound] = orbitalShift!(modeBuffer, vertSegHere, xpnSumPair, 
                                                    xpnRPair, xLRPair, angRPair, oMax)
        vertBuffer .= horiBuffer
        nUpper -= ioMax
    end

    factor * last(horiBuffer)
end



#>-- Core integral-evaluation function --<#
#> Overlap
function computePGTOrbIntegral(::OverlapSampler, 
                               layout::N1N2Tuple{FloatingPolyGaussField{T, D}}, 
                               cache!Self::AxialGaussOverlapCache{T}=
                                           AxialGaussOverlapCache(T, Count(D))
                               ) where {T<:Real, D}
    formattedData = prepareOrbitalInfo(layout)
    computePGTOrbOverlap!(cache!Self, formattedData)
end

#> Multipole moment
function computePGTOrbIntegral(op::MultipoleMomentSampler{T, D}, 
                               layout::N1N2Tuple{FloatingPolyGaussField{T, D}}, 
                               cache!Self::AxialGaussOverlapCache{T}=
                                           AxialGaussOverlapCache(T, Count(D))
                               ) where {T<:Real, D}
    formattedData = prepareOrbitalInfo(layout)
    computePGTOrbMultipoleMoment!(cache!Self, last(op.dresser).term, formattedData)
end

#> Diagonal-directional differentiation (∑ᵢ(cᵢ ⋅ ∂ᵐ/∂xᵢᵐ))
function computePGTOrbIntegral(op::DiagDirectionalDiffSampler{T, D, M}, 
                               layout::N1N2Tuple{FloatingPolyGaussField{T, D}}, 
                               cache!Self::AxialGaussOverlapCache{T}=
                                           AxialGaussOverlapCache(T, Count(D))
                               ) where {T<:Real, D, M}
    formattedData = prepareOrbitalInfo(layout)
    diffVec = computePGTOrbCoordDiff!(cache!Self, ntuple(_->M, Val(D)), formattedData)
    direction = last(op.dresser).direction
    mapreduce(StableMul(T), StableAdd(T), direction, diffVec)
end



#>-- Interface with the composite integration framework --<#
const AxialGaussOverlapSampler{T<:Real, D} = Union{
    OverlapSampler, 
    MultipoleMomentSampler{T, D}, 
    DiagDirectionalDiffSampler{T, D}
}

#= Additional Method =#
function prepareInteComponent!(config::OrbitalIntegrationConfig{T, D, C, N, F}, 
                               ::NTuple{N, NTuple{ 2, FloatingPolyGaussField{T, D} }}
                               ) where {T<:Real, D, C<:RealOrComplex{T}, N, 
                                        F<:AxialGaussOverlapSampler{T, D}}
    if config.cache isa EmptyDict
        AxialGaussOverlapCache(T, Count(D))
    else
        key = (TypeBox(F), ntuple( _->(PrimGaussTypeOrb, PrimGaussTypeOrb), Val(N) ))
        get!(config.cache, key) do
            AxialGaussOverlapCache(T, ntuple( _->True(), Val(D) ))
        end
    end::AxialGaussOverlapCache{T, D}
end

function evaluateIntegralCore!(formattedOp::TypedOperator{C}, 
                               cache::AxialGaussOverlapCache{T, D}, 
                               layout::N12N2Tuple{FloatingPolyGaussField{T, D}}, 
                               ) where {T, C<:RealOrComplex{T}, D}
    convert(C, computePGTOrbIntegral(formattedOp.core, layout, cache))
end