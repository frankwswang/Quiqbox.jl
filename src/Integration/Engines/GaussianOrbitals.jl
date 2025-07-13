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



#>--- Cartesian-3D PGTO Coulomb-field computation based on Obara-Saika scheme ---<#
#>-- Notation convention --<#
#> Electron-pair angular-momentum labels:
#> Electron 1: (iL, jL, kL | iR, jR, kR); Electron 2: (oL, pL, qL | oR, pR, qR)
#> Argument order: (buffer, index, data), (1, 2, L, R, M/C), (coordinate, exponent, angular)
#> [a]P[x]: a + x
#> [b]M[y]: b - y

#>- One-Body Coulomb-integral computation -<#
function computePGTOrbMixedFactorProd(drLR::NTuple{N, T}, xpnPOS::T) where {T<:Real, N}
    mapreduce(*, drLR) do xLR
        computePGTOrbOverlapMixedFactor(xLR, xpnPOS)
    end
end
#>> (n, (iL,   0))                  #>> [here]
#>> (n, (iL-1, 0)) (n+1, (iL-1, 0)) #>> nP0iM1 nP1iM1
#>> (n, (iL-2, 0)) (n+1, (iL-2, 0)) #>> nP0iM2 nP1iM2
function vertTransfer((nP0iM1, nP1iM1, nP0iM2, nP1iM2)::NTuple{4, T}, 
                      iL::Int, xML::T, xMC::T, xpnSum::T, factor::T=one(T)) where {T<:Real}
    part1 = xML * nP0iM1 - factor * xMC * nP1iM1
    part2 = (iL-1) * (nP0iM2 - factor * nP1iM2) * inv(2xpnSum)
    part1 + part2
end
#>> (n, (iL,   iR-1)) (n, (iL-1, iR)) #>> nP0iM0 [here]
#>> (n, (iL-1, iR-1))                 #>> nP0iM1
function horiTransfer((nP0iM0, nP0iM1)::NTuple{2, T}, xLR::T) where {T<:Real}
    nP0iM0 + xLR * nP0iM1
end
#>> <prev>           <here>
#>> (n+iSum, (0, 0)) (n, (iSum, 0))
#>> ...              ...
#>> (n,      (0, 0)) (n, (0,    0))
function verticalFill!(holder::AbstractVector{T}, iSum::Int, xML::T, xMC::T, xpnSum::T, 
                       factor::T=one(T)) where {T<:Real}
    for n in 1:iSum
        here = zero(T)
        buffer = (holder[end-n], holder[end-n+1], zero(T), zero(T))

        for iMax in 1:n
            nP0iM1, nP1iM1, _, _ = buffer
            here = vertTransfer(buffer, iMax, xML, xMC, xpnSum, factor)
            iMax < n && (buffer = (here, holder[end-n+iMax+1], nP0iM1, nP1iM1))
            holder[end-n+iMax] = here
        end
    end

    @view holder[end-iSum : end]
end
#>> <here>         <data>
#>> (n, (iSum, 0)) (n+1, (iSum, 0))
#>> ...            ...
#>> (n, (0,    0)) (n+1, (0,    0))
function verticalPush!(holder::AbstractVector{T}, data::AbstractVector{T}, 
                       iSum::Int, xML::T, xMC::T, xpnSum::T, factor::T=one(T)
                       ) where {T<:Real}
    # @assert iSum+1 == length(holder) == length(data)
    buffer = (holder[begin], data[begin], zero(T), zero(T))

    for i in 1:iSum
        here = vertTransfer(buffer, i, xML, xMC, xpnSum, factor)
        holder[begin+i] = here
        if i < iSum
            a, b, _, _ = buffer
            buffer = (here, data[begin+i], a, b)
        end
    end

    holder
end
#>> <data>            <dump>             <here>
#>> (n, (iSum,    0)) (n, (iSum,    0 ))
#>> ...               ...
#>> (n, (iSum-iR, 0)) (n, (iSum-iR, iR)) (n, (iSum-iR, iR))
#>> ...
#>> (n, (0,       0))
function angularShift!(holder::AbstractVector{T}, iR::Int, xLR::T) where {T<:Real}
    segment = @view holder[end-iR:end]

    for y in 1:iR, x in 1:(iR + 1 - y)
        segment[begin+x-1] = horiTransfer((segment[begin+x], segment[begin+x-1]), xLR)
    end

    first(segment)
end

function computePGTOrbPointCoulombField(pointCharge::Pair{Int, NTuple{3, T}}, 
                                        data::GaussProductInfo{T, D}) where {T<:Real, D}
    charge, cenC = pointCharge
    cenL = data.lhs.cen
    cenR = data.rhs.cen
    cenM = data.cen
    drLR = cenL .- cenR
    drMC = cenM .- cenC
    angL = data.lhs.ang
    angR = data.rhs.ang
    angTpl = angL .+ angR
    angSum = sum(angTpl)
    xpnSum = data.xpn
    xpnPOS = data.lhs.xpn * data.rhs.xpn / xpnSum

    prefactor = computePGTOrbMixedFactorProd(drLR, xpnPOS) / xpnSum
    factor = -charge * 2T(PowersOfPi[:p1d0]) * prefactor

    horiBuffer = computeBoysSequence(xpnSum * mapreduce(x->x*x, +, drMC), angSum)
    vertBuffer = copy(horiBuffer)
    nUpper = angSum

    for (iSum, xL, iR, xM, xLR, xMC) in zip(angTpl, cenL, angR, cenM, drLR, drMC)
        xML = xM - xL
        nShiftBound = nUpper - iSum
        vertSegHere = verticalFill!(horiBuffer, iSum, xML, xMC, xpnSum)

        for shiftMin in 1:nShiftBound
            shiftMax = shiftMin + iSum
            vertSegNext = @view vertBuffer[end-shiftMax : end-shiftMin]
            verticalPush!(vertSegNext, vertSegHere, iSum, xML, xMC, xpnSum)
            horiBuffer[end-shiftMin+1] = angularShift!(vertSegHere, iR, xLR)
            vertSegHere = @view horiBuffer[end-shiftMax : end-shiftMin]
            vertSegHere .= vertSegNext
        end

        horiBuffer[end-nShiftBound] = angularShift!(vertSegHere, iR, xLR)
        vertBuffer .= horiBuffer
        nUpper -= iSum
    end

    factor * last(horiBuffer)
end


#>- Two-Body Coulomb-integral computation -<#
#>>                   (iL, 0|oL-2, 0)                    <head>         iP0oM2
#>> (iL-1, 0|oL-1, 0) (iL, 0|oL-1, 0) (iL+1, 0|oL-1, 0)  <body>  iM1oM1 iP0oM1 iP1oM1
#>>                   (iL, 0|oL,   0)                                   [here]
function modeTransfer((iP0oM2, iP1oM1, iP0oM1, iM1oM1)::NTuple{4, T}, 
                      (i, o)::NTuple{2, Int}, xpnR12::NTuple{2, T}, 
                       xLR12::NTuple{2, T}, xpnSum12::NTuple{2, T}) where {T<:Real}
    xLR1, xLR2 = xLR12
    xpnR1, xpnR2 = xpnR12
    xpnSum1, xpnSum2 = xpnSum12
    part1 = (i*iM1oM1 + (o - 1)*iP0oM2) / (2xpnSum2)
    part2 = ((xpnR1*xLR1 + xpnR2*xLR2)*iP0oM1 + xpnSum1*iP1oM1) / xpnSum2
    part1 - part2
end
#>> (0, 0|oSum, 0) ... (iSum, 0|oSum, 0)                         <here>
#>> ...            ... ...                                       ...
#>> (0, 0|0,    0) ... (iSum, 0|0,    0)                         <here>
#>> (0, 0|0,    0) ... (iSum, 0|0,    0) ... (iSum+oSum, 0|0, 0) <data>
function angularCross!(holder::AbstractMatrix{T}, data::AbstractVector{T}, oSum::Int, 
                       xpnR12::NTuple{2, T}, xLR12::NTuple{2, T}, xpnSum12::NTuple{2, T}
                       ) where {T<:Real}
    angSpace = length(data)
    # @assert 0 <= oSum < angSpace
    # @assert all((oSum+1, angSpace) .<= size(holder))
    activeHolder = @view holder[begin:begin+oSum, begin:begin+angSpace-1]
    activeHolder[begin, :] .= data

    for o in 1:oSum
        n = angSpace - o - 1
        iP1oM1 = activeHolder[begin+o-1, begin+n+1]
        iP0oM1 = activeHolder[begin+o-1, begin+n  ]
        iM1oM1 = n > 0 ? activeHolder[begin+o-1, begin+n-1] : zero(T)
        iP0oM2 = o > 1 ? activeHolder[begin+o-2, begin+n  ] : zero(T)

        for i in n:-1:0
            buffer = (iP0oM2, iP1oM1, iP0oM1, iM1oM1)
            iP0oP0 = modeTransfer(buffer, (i, o), xpnR12, xLR12, xpnSum12)
            activeHolder[begin+o, begin+i] = iP0oP0
            iP1oM1 = iP0oM1
            iP0oM1 = iM1oM1
            iM1oM1 = i > 1 ? activeHolder[begin+o-1, begin+i-2] : zero(T)
            iP0oM2 = (i > 0 && o > 1) ? activeHolder[begin+o-2, begin+i-1] : zero(T)
        end
    end

    @view activeHolder[:, begin:begin+angSpace-oSum-1]
end
#>> <data>                   <dump>                           <here>
#>> (n, (iSum+oSum, 0|0, 0))
#>> ...
#>> (n, (iSum,      0|0, 0)) (n, (iSum,      0 |oSum-oR, oR))
#>> ...                      ...
#>> (n, (iSum-iR,   0|0, 0)) (n, (iSum-iR,   iR|oSum-oR, oR)) (n, (iSum-iR, iR|oSum-oR, oR))
#>> (n, (iSum-iR-1, 0|0, 0)) (n, (iSum-iR-1, 0 |oSum-oR, oR))
#>> ...                      ...
#>> (n, (0,         0|0, 0)) (n, (0,         0 |0,       0 ))
function orbitalShift!(buffer::AbstractMatrix{T}, data::AbstractVector{T}, oSum::Int, 
                       (iR, oR)::NTuple{2, Int}, xpnR12::NTuple{2, T}, 
                       xLR12::NTuple{2, T}, xpnSum12::NTuple{2, T}) where {T<:Real}
    xLR1, xLR2 = xLR12
    activeBuffer = angularCross!(buffer, data, oSum, xpnR12, xLR12, xpnSum12)
    for (n, slot) in zip(eachindex(data), eachcol(activeBuffer))
        data[n] = angularShift!(slot, oR, xLR2)
    end
    angRshifted = @view data[begin : end-oSum]
    # @assert length(angRshifted) == size(activeBuffer, 2)
    angularShift!(angRshifted, iR, xLR1)
end

function computePGTOrbTwoBodyRepulsion(data1::GaussProductInfo{T, D}, 
                                       data2::GaussProductInfo{T, D}) where {T<:Real, D}
    cenL1 = data1.lhs.cen
    cenR1 = data1.rhs.cen
    cenM1 = data1.cen
    drLR1 = cenL1 .- cenR1
    angL1 = data1.lhs.ang
    angR1 = data1.rhs.ang
    ijkTpl = angL1 .+ angR1
    xpnR1 = data1.rhs.xpn
    xpnSum1 = data1.xpn
    xpnPOS1 = data1.lhs.xpn * xpnR1 / xpnSum1

    cenL2 = data2.lhs.cen
    cenR2 = data2.rhs.cen
    cenM2 = data2.cen
    drLR2 = cenL2 .- cenR2
    angL2 = data2.lhs.ang
    angR2 = data2.rhs.ang
    opqTpl = angL2 .+ angR2
    xpnR2 = data2.rhs.xpn
    xpnSum2 = data2.xpn
    xpnPOS2 = data2.lhs.xpn * xpnR2 / xpnSum2

    drM1M2 = cenM1 .- cenM2
    angTpl = ijkTpl .+ opqTpl
    angSum = sum(angTpl)
    xpnR12 = (xpnR1, xpnR2)
    xpnSum12 = (xpnSum1, xpnSum2)
    xpnSumPOS = prod(xpnSum12) / sum(xpnSum12)
    xpnFactor = xpnSumPOS / xpnSum1

    prefactor1 = computePGTOrbMixedFactorProd(drLR1, xpnPOS1) / xpnSum1
    prefactor2 = computePGTOrbMixedFactorProd(drLR2, xpnPOS2) / xpnSum2
    factor = 2T(PowersOfPi[:p2d5]) * prefactor1 * prefactor2 / sqrt(xpnSum1 + xpnSum2)

    horiBuffer = computeBoysSequence(xpnSumPOS * mapreduce(x->x*x, +, drM1M2), angSum)
    vertBuffer = copy(horiBuffer)
    modeBuffer = ShapedMemory{T}(undef, (maximum(opqTpl)+1, maximum(angTpl)+1))
    nUpper = angSum

    for (ioSum, oSum, xL1, iR, oR, xM1, xLR1, xLR2, xM1M2) in zip(angTpl, opqTpl, 
         cenL1, angR1, angR2, cenM1, drLR1, drLR2, drM1M2)
        ioR = (iR, oR)
        xML1 = xM1 - xL1
        xLR12 = (xLR1, xLR2)
        nShiftBound = nUpper - ioSum
        vertSegHere = verticalFill!(horiBuffer, ioSum, xML1, xM1M2, xpnSum1, xpnFactor)

        for shiftMin in 1:nShiftBound
            shiftMax = shiftMin + ioSum
            vertSegNext = @view vertBuffer[end-shiftMax : end-shiftMin]
            verticalPush!(vertSegNext, vertSegHere, ioSum, xML1, xM1M2, xpnSum1, xpnFactor)
            horiBuffer[end-shiftMin+1] = orbitalShift!(modeBuffer, vertSegHere, oSum, ioR, 
                                                       xpnR12, xLR12, xpnSum12)
            vertSegHere = @view horiBuffer[end-shiftMax : end-shiftMin]
            vertSegHere .= vertSegNext
        end

        horiBuffer[end-nShiftBound] = orbitalShift!(modeBuffer, vertSegHere, oSum, ioR, 
                                                    xpnR12, xLR12, xpnSum12)
        vertBuffer .= horiBuffer
        nUpper -= ioSum
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