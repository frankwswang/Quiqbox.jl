export eeIuniqueIndicesOf

using SpecialFunctions: erf
using FastGaussQuadrature: gausslegendre
using LinearAlgebra: dot
using Base: OneTo, Iterators.product

# Reference(s): 
## [DOI] 10.1088/0143-0807/31/1/004

function genFγIntegrand(γ::Int, u::T) where {T}
    function (x)
        ( (x+1)/2 )^(2γ) * exp(-u * (x+1)^2 / 4) / 2
    end
end

@generated function FγCore(γ::Int, u::T, ::Val{GQN}) where {T, GQN}
    GQnodes, GQweights = gausslegendre(GQN)
    return :(dot($GQweights, genFγIntegrand(γ, u).($GQnodes)))
end

for ValI in ValInts[begin:end .<= 1000]
    precompile(FγCore, (Int, Float64, ValI))
end

function F0(u::T) where {T}
    ifelse(u < getAtolVal(T), 
        T(1), 
        begin
            ur = sqrt(u)
            T(πvals[0.5]) * erf(ur) / (2ur)
        end
    )
end

function getGQN(u::T) where {T}
    u = abs(u) + getAtolVal(T)
    res = getAtolDigits(T) + round(0.4u + 2inv(sqrt(u))) + 1
    (Int∘min)(res, typemax(Int) - 1)
end

function Fγ(γ::Int, u::T) where {T}
    if u < getAtolVal(T)
        (inv∘T∘muladd)(2, γ, 1)
    else
        FγCore(γ, u, (getValI∘getGQN)(u))
    end
end

function F₀toFγ(γ::Int, u::T) where {T}
    res = Array{T}(undef, γ+1)
    res[begin] = F0(u)
    γ > 0 && (res[end] = Fγ(γ, u))
    for i in γ:-1:2
        @inbounds res[i] = (expm1(-u) + 2u*res[i+1] + 1) / (2i - 1)
    end
    res
end


function genIntOverlapCore(Δx::T, 
                           i₁::Int, α₁::T, 
                           i₂::Int, α₂::T) where {T}
    res = T(0.0)
    for l₁ in 0:(i₁÷2), l₂ in 0:(i₂÷2)
        Ω = muladd(-2, l₁ + l₂, i₁ + i₂)
        halfΩ = Ω÷2
        oRange = 0:halfΩ
        Δx == 0.0 && (iseven(Ω) ? (oRange = halfΩ:halfΩ) : continue)
        for o in oRange
            res += Δx^(Ω-2o) * 
                   ( α₁^(i₂ - l₁ - 2l₂ - o) / (factorial(l₂) * factorial(i₁-2l₁)) ) * 
                   ( α₂^(i₁ - l₂ - 2l₁ - o) / (factorial(l₁) * factorial(i₂-2l₂)) ) * 
                   ( (-1)^o * factorial(Ω) / 
                     (4^(l₁+ l₂ + o) * factorial(o) * factorial(Ω-2o)) ) * 
                   (α₁ + α₂)^muladd(2, (l₁ + l₂), o)
        end
    end
    res
end

function ∫overlapCore(::Val{3}, 
                      ΔR::NTuple{3, T}, 
                      ijk₁::NTuple{3, Int}, α₁::T, 
                      ijk₂::NTuple{3, Int}, α₂::T) where {T}
    any(n -> n<0, (ijk₁..., ijk₂...)) && (return T(0.0))

    α = α₁ + α₂
    res = T(1)
    for (i₁, i₂, ΔRᵢ) in zip(ijk₁, ijk₂, ΔR)
        int = genIntOverlapCore(ΔRᵢ, i₁, α₁, i₂, α₂)
        iszero(int) && (return T(0))
        res *= (-1)^(i₁) * factorial(i₁) * factorial(i₂) * α^(-i₁-i₂) * int
    end
    res *= sqrt((π/α)^3) * exp(-α₁ / α * α₂* sum(abs2, ΔR))
    res
end

∫overlapCore(::Val{3}, 
             R₁::NTuple{3, T}, R₂::NTuple{3, T}, 
             ijk₁::NTuple{3, Int}, α₁::T, 
             ijk₂::NTuple{3, Int}, α₂::T) where {T} = 
∫overlapCore(Val(3), R₁.-R₂, ijk₁, α₁, ijk₂, α₂)


function ∫elecKineticCore(::Val{3}, 
                          R₁::NTuple{3, T}, R₂::NTuple{3, T}, 
                          ijk₁::NTuple{3, Int}, α₁::T,
                          ijk₂::NTuple{3, Int}, α₂::T) where {T}
    ΔR = R₁ .- R₂
    shifts = ((2,0,0), (0,2,0), (0,0,2))
    resTemp = mapreduce(+, ijk₂, shifts) do l₂, Δl
        2α₂^2 * ∫overlapCore(Val(3), ΔR, ijk₁, α₁, map(+, ijk₂, Δl), α₂) + 
        (l₂-1) * l₂ / 2 * ∫overlapCore(Val(3), ΔR, ijk₁, α₁, map(-, ijk₂, Δl), α₂)
    end
    ∫overlapCore(Val(3), ΔR, ijk₁, α₁, ijk₂, α₂) * α₂ * muladd(2, sum(ijk₂), 3) - resTemp
end


function genIntTerm1(Δx::T1, 
                     l₁::T2, o₁::T2, 
                     l₂::T2, o₂::T2, 
                     i₁::T2, α₁::T1, 
                     i₂::T2, α₂::T1) where {T1, T2<:Integer}
    (r::T2) -> 
        ( Δx^muladd(-2, r, o₁+o₂) / (factorial(r ) * (factorial∘muladd)(-2, r, o₁+o₂)) ) * 
        ( α₁^(o₂-l₁- r) / (factorial(l₁) * factorial(i₁-2l₁-o₁)) ) * 
        ( α₂^(o₁-l₂- r) / (factorial(l₂) * factorial(i₂-2l₂-o₂)) ) * 
        ( (-1)^(o₂+r) * factorial(o₁+o₂) / (4^(l₁+l₂+r) * factorial(o₁) * factorial(o₂)) )
end

function genIntTerm2core(Δx::T1,  μ::T2) where {T1, T2<:Integer}
    (u::T2) -> 
        Δx^(μ-2u) * ( (-1)^u * factorial(μ) / (4^u * factorial(u) * factorial(μ-2u)) )
end

function genIntTerm2(Δx::T1, α::T1, o₁::T2, o₂::T2, μ::T2, r::T2) where {T1, T2<:Integer}
    (u::T2) -> 
        genIntTerm2core(Δx, μ)(u) * α^(r-o₁-o₂-u)
end


function genIntNucAttCore(ΔRR₀::NTuple{3, T}, ΔR₁R₂::NTuple{3, T}, β::T, 
                          ijk₁::NTuple{3, Int}, α₁::T, 
                          ijk₂::NTuple{3, Int}, α₂::T) where {T}
    A = T(0.0)
    i₁, j₁, k₁ = ijk₁
    i₂, j₂, k₂ = ijk₂
    for l₁ in 0:(i₁÷2), m₁ in 0:(j₁÷2), n₁ in 0:(k₁÷2), 
        l₂ in 0:(i₂÷2), m₂ in 0:(j₂÷2), n₂ in 0:(k₂÷2)

        lmn₁ = (l₁, m₁, n₁)
        lmn₂ = (l₂, m₂, n₂)

        for o₁ in 0:(i₁-2l₁), p₁ in 0:(j₁-2m₁), q₁ in 0:(k₁-2n₁), 
            o₂ in 0:(i₂-2l₂), p₂ in 0:(j₂-2m₂), q₂ in 0:(k₂-2n₂)

            opq₁ = (o₁, p₁, q₁)
            opq₂ = (o₂, p₂, q₂)

            μˣ, μʸ, μᶻ = μv = @. ijk₁ + ijk₂ - muladd(2, lmn₁+lmn₂, opq₁+opq₂)
            μsum = sum(μv)
            Fγs = F₀toFγ(μsum, β)
            core1s = genIntTerm1.(ΔR₁R₂, lmn₁, opq₁, lmn₂, opq₂, ijk₁, α₁, ijk₂, α₂)

            for r in 0:((o₁+o₂)÷2), s in 0:((p₁+p₂)÷2), t in 0:((q₁+q₂)÷2)

                rst = (r, s, t)
                tmp = T(0.0)
                core2s = genIntTerm2.(ΔRR₀, α₁+α₂, opq₁, opq₂, μv, rst)

                for u in 0:(μˣ÷2), v in 0:(μʸ÷2), w in 0:(μᶻ÷2)
                    γ = μsum - u - v - w
                    @inbounds tmp += prod((u, v, w) .|> core2s) * 2Fγs[γ+1]
                end

                A += prod(rst .|> core1s) * tmp

            end
        end

    end
    A
end


function ∫nucAttractionCore(::Val{3}, 
                            Z₀::Int, R₀::NTuple{3, T}, 
                            R₁::NTuple{3, T}, R₂::NTuple{3, T}, 
                            ijk₁::NTuple{3, Int}, α₁::T,
                            ijk₂::NTuple{3, Int}, α₂::T) where {T}
    if α₁ == α₂
        α = 2α₁
        R = @. (R₁ + R₂) / 2
        flag = true
    else
        α = α₁ + α₂
        R = @. (α₁*R₁ + α₂*R₂) / α
        flag = false
    end
    ΔRR₀ = R .- R₀
    ΔR₁R₂ = R₁ .- R₂
    β = α * sum(abs2, ΔRR₀)
    genIntNucAttCore(ΔRR₀, ΔR₁R₂, β, ijk₁, α₁, ijk₂, α₂) * 
    (π / α) * exp(-α₁ / α * α₂ * sum(abs2, ΔR₁R₂)) * 
    ( -Z₀ * (-1)^sum(ijk₁.+ijk₂) * prod(factorial.(ijk₁)) * prod(factorial.(ijk₂)) )
end


function genIntTerm3(Δx::T1, 
                     l₁::T2, o₁::T2, 
                     l₂::T2, o₂::T2, 
                     i₁::T2, α₁::T1, 
                     i₂::T2, α₂::T1) where {T1, T2<:Integer}
    (r::T2) -> 
        genIntTerm1(Δx, l₁, o₁, l₂, o₂, i₁, α₁, i₂, α₂)(r) * (α₁+α₂)^muladd(2, l₁+l₂, r)
end

function genIntTerm4(Δx::T1, η::T1, μ::T2) where {T1, T2<:Integer}
    (u::T2) -> 
        genIntTerm2core(Δx, μ)(u) * η^(μ-u)
end


function ∫eeInteractionCore1234(ΔRl::NTuple{3, T}, ΔRr::NTuple{3, T}, 
                                ΔRc::NTuple{3, T}, β::T, η::T, 
                                ijk₁::NTuple{3, Int}, α₁::T, 
                                ijk₂::NTuple{3, Int}, α₂::T, 
                                ijk₃::NTuple{3, Int}, α₃::T, 
                                ijk₄::NTuple{3, Int}, α₄::T) where {T}
    A = T(0.0)
    (i₁, j₁, k₁), (i₂, j₂, k₂), (i₃, j₃, k₃), (i₄, j₄, k₄) = ijk₁, ijk₂, ijk₃, ijk₄

    IJK = @. ijk₁ + ijk₂ + ijk₃ + ijk₄

    for l₁ in 0:(i₁÷2), m₁ in 0:(j₁÷2), n₁ in 0:(k₁÷2), 
        l₂ in 0:(i₂÷2), m₂ in 0:(j₂÷2), n₂ in 0:(k₂÷2), 
        l₃ in 0:(i₃÷2), m₃ in 0:(j₃÷2), n₃ in 0:(k₃÷2), 
        l₄ in 0:(i₄÷2), m₄ in 0:(j₄÷2), n₄ in 0:(k₄÷2)

        lmn₁ = (l₁, m₁, n₁)
        lmn₂ = (l₂, m₂, n₂)
        lmn₃ = (l₃, m₃, n₃)
        lmn₄ = (l₄, m₄, n₄)

        for o₁ in 0:(i₁-2l₁), p₁ in 0:(j₁-2m₁), q₁ in 0:(k₁-2n₁), 
            o₂ in 0:(i₂-2l₂), p₂ in 0:(j₂-2m₂), q₂ in 0:(k₂-2n₂), 
            o₃ in 0:(i₃-2l₃), p₃ in 0:(j₃-2m₃), q₃ in 0:(k₃-2n₃), 
            o₄ in 0:(i₄-2l₄), p₄ in 0:(j₄-2m₄), q₄ in 0:(k₄-2n₄)

            opq₁ = (o₁, p₁, q₁)
            opq₂ = (o₂, p₂, q₂)
            opq₃ = (o₃, p₃, q₃)
            opq₄ = (o₄, p₄, q₄)

            μˣ, μʸ, μᶻ = μv = begin
                @. IJK - muladd(2, lmn₁+lmn₂+lmn₃+lmn₄, opq₁+opq₂+opq₃+opq₄)
            end

            μsum = sum(μv)
            Fγs = F₀toFγ(μsum, β)

            core1s = genIntTerm3.(ΔRl, lmn₁, opq₁, lmn₂, opq₂, ijk₁, α₁, ijk₂, α₂)
            core2s = genIntTerm3.(ΔRr, lmn₄, opq₄, lmn₃, opq₃, ijk₄, α₄, ijk₃, α₃)
            core3s = genIntTerm4.(ΔRc, η, μv)

            for r₁ in 0:((o₁+o₂)÷2), s₁ in 0:((p₁+p₂)÷2), t₁ in 0:((q₁+q₂)÷2), 
                r₂ in 0:((o₃+o₄)÷2), s₂ in 0:((p₃+p₄)÷2), t₂ in 0:((q₃+q₄)÷2)

                rst₁ = (r₁, s₁, t₁)
                rst₂ = (r₂, s₂, t₂)
                tmp = T(0.0)

                for u in 0:(μˣ÷2), v in 0:(μʸ÷2), w in 0:(μᶻ÷2)
                    γ = μsum - u - v - w
                    @inbounds tmp += prod((u, v, w) .|> core3s) * 2Fγs[γ+1]
                end

                A += prod(rst₁ .|> core1s) * prod(rst₂ .|> core2s) * tmp

            end
        end

    end
    A
end


function ∫eeInteractionCore(::Val{3}, 
                            R₁::NTuple{3, T}, ijk₁::NTuple{3, Int}, α₁::T, 
                            R₂::NTuple{3, T}, ijk₂::NTuple{3, Int}, α₂::T,
                            R₃::NTuple{3, T}, ijk₃::NTuple{3, Int}, α₃::T, 
                            R₄::NTuple{3, T}, ijk₄::NTuple{3, Int}, α₄::T) where {T}
    ΔRl = R₁ .- R₂
    ΔRr = R₃ .- R₄
    αl = α₁ + α₂
    αr = α₃ + α₄
    ηl = α₁ / αl * α₂
    ηr = α₃ / αr * α₄
    ΔRc = @. (α₁*R₁ + α₂*R₂) / αl - (α₃*R₃ + α₄*R₄) / αr
    η = αl / (α₁ + α₂ + α₃ + α₄) * αr
    β = η * sum(abs2, ΔRc)
    ∫eeInteractionCore1234(ΔRl, ΔRr, ΔRc, β, η, ijk₁, α₁, ijk₂, α₂, ijk₃, α₃, ijk₄, α₄) * 
    T(πvals[2.5]) / (αl * αr * sqrt(αl + αr)) * 
    exp(-ηl * sum(abs2, ΔRl)) * exp(-ηr * sum(abs2, ΔRr)) * 
    mapreduce(*, ijk₁, ijk₂, ijk₃, ijk₄) do l₁, l₂, l₃, l₄
        (-1)^(l₁+l₂) * factorial(l₁) * factorial(l₂) * factorial(l₃) * factorial(l₄) / 
        (αl^(l₁+l₂) * αr^(l₃+l₄))
    end
end


function reformatIntDataCore(bf::FGTBasisFuncs1O{T, D, 𝑙, GN}) where {T, D, 𝑙, GN}
    R = (centerCoordOf(bf) |> Tuple)::NTuple{D, T}
    ijk = bf.l[begin].tuple
    αds = if hasNormFactor(bf)
        N = getNijk(T, ijk...)
        map(bf.gauss) do x
            xpn, con = outValOf.(x.param)::NTuple{2, T}
            (xpn, con * N * getNα(ijk..., xpn))
        end
    else
        map(x->outValOf.(x.param)::NTuple{2, T}, bf.gauss)
    end
    R, ijk, αds, 𝑙
end

function reformatIntData1((ji,)::Tuple{Bool}, 
                          bfs::NTupleOfFGTBF{2, T, D}) where {T, D}
    if ji
        data1 = reformatIntDataCore(bfs[1])
        (data1, data1)
    else
        (reformatIntDataCore(bfs[1]), 
         reformatIntDataCore(bfs[2]))
    end
end

function reformatIntData1((lk, lj, kj, kiOrji)::NTuple{4, Bool}, 
                          bfs::NTupleOfFGTBF{4, T, D}) where {T, D}
    data4 = reformatIntDataCore(bfs[4])
    data3 = lk ? data4 : reformatIntDataCore(bfs[3])
    data2 = lj ? data4 : (kj ? data3 : reformatIntDataCore(bfs[2]))
    data1 = lj ? (kiOrji ? data3 : reformatIntDataCore(bfs[1])) : 
                 (kiOrji ? data2 : reformatIntDataCore(bfs[1]))
    (data1, data2, data3, data4)
end

function reformatIntData1((lk, _, _, ji)::Tuple{Bool, Val{false}, Val{false}, Bool}, 
                          bfs::NTupleOfFGTBF{4, T, D}) where {T, D}
    data4 = reformatIntDataCore(bfs[4])
    data3 = lk ? data4 : reformatIntDataCore(bfs[3])
    data2 = reformatIntDataCore(bfs[2])
    data1 = ji ? data2 : reformatIntDataCore(bfs[1])
    (data1, data2, data3, data4)
end

function reformatIntData1((_, lj, _, kiOrji)::Tuple{Val{false}, Bool, Val{false}, Bool}, 
                          bfs::NTupleOfFGTBF{4, T, D}) where {T, D}
    data4 = reformatIntDataCore(bfs[4])
    data3 = reformatIntDataCore(bfs[3])
    data2 = lj ? data4 : reformatIntDataCore(bfs[2])
    data1 = (lj && kiOrji) ? data3 : reformatIntDataCore(bfs[1])
    (data1, data2, data3, data4)
end

function reformatIntData1((_, lj, _, _)::Tuple{Val{false}, Bool, Val{false}, Val{false}}, 
                          bfs::NTupleOfFGTBF{4, T, D}) where {T, D}
    data4 = reformatIntDataCore(bfs[4])
    data3 = reformatIntDataCore(bfs[3])
    data2 = lj ? data4 : reformatIntDataCore(bfs[2])
    data1 = reformatIntDataCore(bfs[1])
    (data1, data2, data3, data4)
end

function reformatIntData1((_, _, kj, _)::Tuple{Val{false}, Val{false}, Bool, Val{false}}, 
                          bfs::NTupleOfFGTBF{4, T, D}) where {T, D}
    data4 = reformatIntDataCore(bfs[4])
    data3 = reformatIntDataCore(bfs[3])
    data2 = kj ? data3 : reformatIntDataCore(bfs[2])
    data1 = reformatIntDataCore(bfs[1])
    (data1, data2, data3, data4)
end

function reformatIntData1((_, _, _, ji)::Tuple{Val{false}, Val{false}, Val{false}, Bool}, 
                          bfs::NTupleOfFGTBF{4, T, D}) where {T, D}
    data4 = reformatIntDataCore(bfs[4])
    data3 = reformatIntDataCore(bfs[3])
    data2 = reformatIntDataCore(bfs[2])
    data1 = ji ? data2 : reformatIntDataCore(bfs[1])
    (data1, data2, data3, data4)
end

function reformatIntData1((lk, _, _, _)::Tuple{Bool, Val{false}, Val{false}, Val{false}}, 
                          bfs::NTupleOfFGTBF{4, T, D}) where {T, D}
    data4 = reformatIntDataCore(bfs[4])
    data3 = lk ? data4 : reformatIntDataCore(bfs[3])
    data2 = reformatIntDataCore(bfs[2])
    data1 = reformatIntDataCore(bfs[1])
    (data1, data2, data3, data4)
end

reformatIntData1(::Val{false}, bfs::NTupleOfFGTBF{2, T, D}) where {T, D} = 
(reformatIntDataCore(bfs[1]), reformatIntDataCore(bfs[2]))

reformatIntData1(::Val{false}, bfs::NTupleOfFGTBF{4, T, D}) where {T, D} = 
(reformatIntDataCore(bfs[1]), reformatIntDataCore(bfs[2]), 
 reformatIntDataCore(bfs[3]), reformatIntDataCore(bfs[4]))


reformatIntData2((o1, o2)::NTuple{2, T}, flag::Bool) where {T} = 
( (flag && isless(o2, o1)) ? (o2, o1) : (o1, o2) )

function reformatIntData2((o1, o2, o3, o4)::NTuple{4, T}, flags::NTuple{3, Bool}) where {T}
    l = reformatIntData2((o1, o2), flags[1])
    r = reformatIntData2((o3, o4), flags[2])
    ifelse((flags[3] && isless(r, l)), (r[1], r[2], l[1], l[2]), (l[1], l[2], r[1], r[2]))
end


function getUniquePair!(i::Int, 
                        uniquePairs::Vector{TT}, 
                        uPairCoeffs::Vector{T}, 
                        flag::BL, 
                        psc::TTT, 
                        nFold::Int=1) where 
                       {T, BL<:Union{Bool, NTuple{3, Bool}}, 
                        N, TT<:NTuple{N, T}, TTT<:NTuple{N, NTuple{2, T}}}
    pair = reformatIntData2(first.(psc), flag)
    idx = findfirst(isequal(pair), uniquePairs)
    con = (last.(psc) |> prod) * nFold
    if idx === nothing
        i += 1
        push!(uniquePairs, pair)
        @inbounds uPairCoeffs[i] = con
    else
        @inbounds uPairCoeffs[idx] += con
    end
    i
end


diFoldCount(i::T, j::T) where {T} = ifelse(i==j, 1, 2)

function octaFoldCount(i::T, j::T, k::T, l::T) where {T}
    m = 0
    i != j && (m += 1)
    k != l && (m += 1)
    (i != k || j != l) && (m += 1)
    1 << m
end


function getIntCore11!(n::Int, 
                       uniquePairs::Vector{NTuple{2, T}}, 
                       uPairCoeffs::Vector{T}, 
                       flag::Bool, 
                       ps₁) where {T}
    for (i₁, p₁) in enumerate(ps₁), (i₂, p₂) in zip(OneTo(i₁), ps₁)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flag, (p₁, p₂), diFoldCount(i₁, i₂))
    end
    n
end

function getIntCore12!(n::Int, 
                       uniquePairs::Vector{NTuple{2, T}}, 
                       uPairCoeffs::Vector{T}, 
                       flag::Bool, 
                       (ps₁, ps₂)) where {T}
    for p₁ in ps₁, p₂ in ps₂
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flag, (p₁, p₂))
    end
    n
end


function getOneBodyUniquePairs(flag::Bool, 
                               ps₁::NTuple{GN1, NTuple{2, T}}, 
                               ps₂::NTuple{GN2, NTuple{2, T}}) where {T, GN1, GN2}
    uniquePairs = NTuple{2, T}[]
    uPairCoeffs = Array{T}(undef, GN1*GN2)
    i = 0
    if flag
        if ps₁ == ps₂
            i = getIntCore11!(i, uniquePairs, uPairCoeffs, flag, ps₁)
        else
            psC, ps1, ps2 = tupleDiff(ps₁, ps₂)
            i = getIntCore11!(i, uniquePairs, uPairCoeffs, flag, psC)
            i = getIntCore12!(i, uniquePairs, uPairCoeffs, flag, (ps1, ps₂))
            i = getIntCore12!(i, uniquePairs, uPairCoeffs, flag, (psC, ps2))
        end
    else
        i = getIntCore12!(i, uniquePairs, uPairCoeffs, flag, (ps₁, ps₂))
    end
    uniquePairs, uPairCoeffs
end


function isIntZeroCore(::Val{1}, 
                       R₁::NTuple{D, T}, R₂::NTuple{D, T}, 
                       ijk₁::NTuple{D, Int}, ijk₂::NTuple{D, Int}) where {D, T}
    any(i -> (R₁[i]==R₂[i] && isodd(ijk₁[i] + ijk₂[i])), eachindex(R₁))
end

function isIntZeroCore(::Val{2}, 
                       R₁::NTuple{D, T}, R₂::NTuple{D, T}, 
                       R₃::NTuple{D, T}, R₄::NTuple{D, T}, 
                       ijk₁::NTuple{D, Int}, ijk₂::NTuple{D, Int}, 
                       ijk₃::NTuple{D, Int}, ijk₄::NTuple{D, Int}) where {D, T}
    any(i -> (R₁[i]==R₂[i]==R₃[i]==R₄[i] && isodd(ijk₁[i] + ijk₂[i] + ijk₃[i] + ijk₄[i])), 
        eachindex(R₁))
end

function isIntZeroCore(::Val{:∫nucAttractionCore}, 
                       R₀::NTuple{D, T}, 
                       R₁::NTuple{D, T}, R₂::NTuple{D, T}, 
                       ijk₁::NTuple{D, Int}, ijk₂::NTuple{D, Int}) where {D, T}
    any(i -> (R₀[i]==R₁[i]==R₂[i] && isodd(ijk₁[i] + ijk₂[i])), eachindex(R₁))
end

isIntZero(::Type{typeof(∫overlapCore)}, _, 
          R₁::NTuple{D, T}, R₂::NTuple{D, T}, 
          ijk₁::NTuple{D, Int}, ijk₂::NTuple{D, Int}) where {D, T} = 
isIntZeroCore(Val(1), R₁, R₂, ijk₁, ijk₂)

isIntZero(::Type{typeof(∫elecKineticCore)}, _, 
          R₁::NTuple{D, T}, R₂::NTuple{D, T}, 
          ijk₁::NTuple{D, Int}, ijk₂::NTuple{D, Int}) where {D, T} = 
isIntZeroCore(Val(1), R₁, R₂, ijk₁, ijk₂)

isIntZero(::Type{typeof(∫nucAttractionCore)}, 
          optPosArgs::Tuple{Int, NTuple{D, T}}, 
          R₁::NTuple{D, T}, R₂::NTuple{D, T}, 
          ijk₁::NTuple{D, Int}, ijk₂::NTuple{D, Int}) where {D, T} = 
isIntZeroCore(Val(:∫nucAttractionCore), optPosArgs[2], R₁, R₂, ijk₁, ijk₂)

isIntZero(::Type{typeof(∫eeInteractionCore)}, _, 
          R₁::NTuple{D, T}, R₂::NTuple{D, T}, 
          R₃::NTuple{D, T}, R₄::NTuple{D, T}, 
          ijk₁::NTuple{D, Int}, ijk₂::NTuple{D, Int}, 
          ijk₃::NTuple{D, Int}, ijk₄::NTuple{D, Int}) where {D, T} = 
isIntZeroCore(Val(2), R₁, R₂, R₃, R₄, ijk₁, ijk₂, ijk₃, ijk₄)


const iBlTs = [Tuple{Bool}, NTuple{4, Any}, Val{false}]


function getOneBodyInt(::Type{T}, ::Val{D}, ∫1e::F, @nospecialize(optPosArgs::Tuple), 
                       iBl::Union{iBlTs[1], iBlTs[3]}, bfs::NTupleOfFGTBF{2, T, D}) where 
                       {T, D, F<:Function}
    (R₁, ijk₁, ps₁, 𝑙₁), (R₂, ijk₂, ps₂, 𝑙₂) = reformatIntData1(iBl, bfs)
    𝑙₁==𝑙₂==0 || isIntZero(F, optPosArgs, R₁,R₂, ijk₁,ijk₂) && (return T(0.0))
    uniquePairs, uPairCoeffs = getOneBodyUniquePairs(R₁==R₂ && ijk₁==ijk₂, ps₁, ps₂)
    mapreduce(+, uniquePairs, uPairCoeffs) do x, y
        ∫1e(Val(D), optPosArgs..., R₁, R₂, ijk₁, x[1], ijk₂, x[2])::T * y
    end
end


function getIntCore1111!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         ps₁) where {T}
    for (i₁, p₁) in enumerate(ps₁), (i₂, p₂) in zip(OneTo(i₁), ps₁), 
        (i₃, p₃) in zip(OneTo(i₁), ps₁), (i₄, p₄) in zip((OneTo∘ifelse)(i₃==i₁, i₂,i₃), ps₁)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁,p₂,p₃,p₄), 
                           octaFoldCount(i₁,i₂,i₃,i₄))
    end
    n
end

function getIntCore1122!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         (ps₁, ps₂)) where {T}
    for (i₁, p₁) in enumerate(ps₁), (i₂, p₂) in zip(OneTo(i₁), ps₁), 
        (i₃, p₃) in enumerate(ps₂), (i₄, p₄) in zip(OneTo(i₃), ps₂)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁, p₂, p₃, p₄), 
                           diFoldCount(i₁, i₂)*diFoldCount(i₃, i₄))
    end
    n
end

function getIntCore1212!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         (ps₁, ps₂)) where {T}
    oneSidePairs = product(eachindex(ps₁), eachindex(ps₂))
    for (x, (i₁,i₂)) in enumerate(oneSidePairs), (_, (i₃,i₄)) in zip(OneTo(x), oneSidePairs)
        @inbounds n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, 
                                     (ps₁[i₁], ps₂[i₂], ps₁[i₃], ps₂[i₄]), 
                                     1<<(i₁!=i₃ || i₂!=i₄))
    end
    n
end

function getIntCore1221!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         (ps₁, ps₂)) where {T}
    oneSidePairs = product(eachindex(ps₁), eachindex(ps₂))
    for (x, (i₁,i₂)) in enumerate(oneSidePairs), (_, (i₃,i₄)) in zip(OneTo(x), oneSidePairs)
        @inbounds n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, 
                                     (ps₁[i₁], ps₂[i₂], ps₂[i₄], ps₁[i₃]), 
                                     1<<(i₁!=i₃ || i₂!=i₄))
    end
    n
end

function getIntCore1123!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         (ps₁, ps₂, ps₃)) where {T}
    for (i₁, p₁) in enumerate(ps₁), (i₂, p₂) in zip(OneTo(i₁), ps₁), p₃ in ps₂, p₄ in ps₃
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁, p₂, p₃, p₄), 
                           diFoldCount(i₁, i₂))
    end
    n
end

function getIntCore1233!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         (ps₁, ps₂, ps₃)) where {T}
    for p₁ in ps₁, p₂ in ps₂, (i₃, p₃) in enumerate(ps₃), (i₄, p₄) in zip(OneTo(i₃), ps₃)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁, p₂, p₃, p₄), 
                           diFoldCount(i₃, i₄))
    end
    n
end

function getIntCore1234!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         (ps₁, ps₂, ps₃, ps₄)) where {T}
    for p₁ in ps₁, p₂ in ps₂, p₃ in ps₃, p₄ in ps₄
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁, p₂, p₃, p₄))
    end
    n
end


function getIntX1X1X2X2!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         ps₁, ps₂) where {T}
    if flags[3]
        A, B, C = tupleDiff(ps₁, ps₂)
        if !isempty(A)
            g1111 = (A,)
            g1122 = ((A, C), (B, A), (B, C))
            g1212 = ()
            g1123 = ((A, A, C), (A, C, A), (B, A, C), (B, C, A))
            g1233 = ((A, B, A), (B, A, A), (A, B, C), (B, A, C))
            g1234 = ((A, B, A, C), (A, B, C, A), (A, B, C, A), (B, A, C, A))
            return getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                       (g1111, g1122, g1212, g1123, g1233, g1234))
        end
    end
    getIntCore1122!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂))
end

function getIntX1X2X1X2!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         ps₁, ps₂) where {T}
    if flags[1] && flags[2]
        A, B, C = tupleDiff(ps₁, ps₂)
        if !isempty(A)
            g1111 = (A,)
            g1122 = ()
            g1212 = ((A, C), (B, A), (B, C))
            g1123 = ((A, A, C), (A, B, A), (A, B, C))
            g1233 = ((A, C, A), (B, A, A), (B, C, A))
            g1234 = ((A, C, B, A), (A, C, B, C), (B, A, B, C),
                     (B, A, A, C), (B, C, A, C), (B, C, B, A))
            return getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                       (g1111, g1122, g1212, g1123, g1233, g1234))
        end
    end
    getIntCore1212!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂))
end

function getIntX1X2X2X1!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         ps₁, ps₂) where {T}
    if all(flags)
        A, B, C = tupleDiff(ps₁, ps₂)
        if !isempty(A)
            g1111 = (A,)
            g1122 = ()
            g1212 = ()
            g1123 = ((A, A, B), (A, C, A), (A, C, B))
            g1233 = ((A, C, A), (B, A, A), (B, C, A))
            g1234 = ((A, C, A, B), (A, C, C, B), (B, A, C, A),
                     (B, A, C, B), (B, C, A, B), (B, C, C, A))
            n = getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                    (g1111, g1122, g1212, g1123, g1233, g1234))

            g1221 = ((A, C), (B, A), (B, C))
            for i in g1221
                n = getIntCore1221!(n, uniquePairs, uPairCoeffs, flags, i)
            end
            return n
        end
    end
    getIntCore1221!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂))
end

function getIntX1X1X2X3!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         ps₁, ps₂, ps₃) where {T}
    if flags[2] && flags[3]
        A, B, C, D = tupleDiff(ps₁, ps₂, ps₃)
        if !isempty(A)
            g1111 = (A,)
            g1122 = ((B, A),)
            g1212 = ()
            g1123 = ((A, A, D), (A, C, A), (A, C, D),
                     (B, A, D), (B, C, A), (B, C, D))
            g1233 = ((A, B, A), (B, A, A))
            g1234 = ((A, B, A, D), (A, B, C, A), (A, B, C, D),
                     (B, A, A, D), (B, A, C, A), (B, A, C, D))
            return getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                       (g1111, g1122, g1212, g1123, g1233, g1234))
        end
    end
    getIntCore1123!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃))
end

function getIntX1X2X3X3!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         ps₁, ps₂, ps₃) where {T}
    if flags[1] && flags[3]
        A, B, C, D = tupleDiff(ps₁, ps₂, ps₃)
        if !isempty(A)
            g1111 = (A,)
            g1122 = ((A, D),)
            g1212 = ()
            g1123 = ((A, A, D), (A, D, A))
            g1233 = ((A, C, A), (A, C, D), (B, A, A), 
                     (B, A, D), (B, C, A), (B, C, D))
            g1234 = ((A, C, A, D), (A, C, D, A), (B, A, A, D),
                     (B, A, D, A), (B, C, A, D), (B, C, D, A))
            return getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                       (g1111, g1122, g1212, g1123, g1233, g1234))
        end
    end
    getIntCore1233!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃))
end

function getIntX1X2X3X1!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         ps₁, ps₂, ps₃) where {T}
    if all(flags)
        A, B, C, D = tupleDiff(ps₁, ps₂, ps₃)
        if !isempty(A)
            g1111 = (A,)
            g1122 = ()
            g1212 = ()
            g1123 = ((A, A, B), (A, D, A), (A, D, B))
            g1233 = ((A, C, A), (B, A, A), (B, C, A))
            g1234 = ((A, C, A, B), (A, C, D, A), (A, C, D, B), 
                     (B, A, D, A), (B, A, D, B), 
                     (B, C, A, B), (B, C, D, A), (B, C, D, B))
            n = getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                    (g1111, g1122, g1212, g1123, g1233, g1234))
            return getIntCore1221!(n, uniquePairs, uPairCoeffs, flags, (B,A))
        end
    end
    getIntCore1234!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃, ps₁))
end

function getIntX1X2X2X3!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         ps₁, ps₂, ps₃) where {T}
    if all(flags)
        A, B, C, D = tupleDiff(ps₁, ps₂, ps₃)
        if !isempty(A)
            g1111 = (A,)
            g1122 = ()
            g1212 = ()
            g1123 = ((A, A, D), (A, C, A), (A, C, D))
            g1233 = ((A, C, A), (B, A, A), (B, C, A))
            g1234 = ((A, C, A, D), (A, C, C, D), 
                     (B, A, A, D), (B, A, C, A), (B, A, C, D), 
                     (B, C, A, D), (B, C, C, A), (B, C, C, D))
            n = getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                    (g1111, g1122, g1212, g1123, g1233, g1234))
            return getIntCore1221!(n, uniquePairs, uPairCoeffs, flags, (A,C))
        end
    end
    getIntCore1234!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₂, ps₃))
end

function getIntX1X2X3X4!(n::Int, 
                         uniquePairs::Vector{NTuple{4, T}}, 
                         uPairCoeffs::Vector{T}, 
                         flags::NTuple{3, Bool}, 
                         ps₁, ps₂, ps₃, ps₄) where {T}
    if all(flags)
        A, B, C, D, E = tupleDiff(ps₁, ps₂, ps₃, ps₄)
        if !isempty(A)
            g1111 = (A,)
            g1122 = ()
            g1212 = ()
            g1123 = ((A, A, E), (A, D, A), (A, D, E))
            g1233 = ((A, C, A), (B, A, A), (B, C, A))
            g1234 = ((A, C, A, E), (A, C, D, A), (A, C, D, E),
                     (B, A, A, E), (B, A, D, A), (B, A, D, E), 
                     (B, C, A, E), (B, C, D, A), (B, C, D, E))
            return getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                       (g1111, g1122, g1212, g1123, g1233, g1234))
        end
    end
    getIntCore1234!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃, ps₄))
end


function getIntXAXBXCXDcore!(n::Int, 
                             uniquePairs::Vector{NTuple{4, T}}, 
                             uPairCoeffs::Vector{T}, 
                             flags::NTuple{3, Bool}, 
                             groups::NTuple{6, Any}) where {T}
    for i in groups[1]
        n = getIntCore1111!(n, uniquePairs, uPairCoeffs, flags, i)
    end
    for i in groups[2]
        n = getIntCore1122!(n, uniquePairs, uPairCoeffs, flags, i)
    end
    for i in groups[3]
        n = getIntCore1212!(n, uniquePairs, uPairCoeffs, flags, i)
    end
    for i in groups[4]
        n = getIntCore1123!(n, uniquePairs, uPairCoeffs, flags, i)
    end
    for i in groups[5]
        n = getIntCore1233!(n, uniquePairs, uPairCoeffs, flags, i)
    end
    for i in groups[6]
        n = getIntCore1234!(n, uniquePairs, uPairCoeffs, flags, i)
    end
    n
end


function getTwoBodyUniquePairs(flags::NTuple{5, Bool}, 
                               ps₁::NTuple{GN1, NTuple{2, T}}, 
                               ps₂::NTuple{GN2, NTuple{2, T}}, 
                               ps₃::NTuple{GN3, NTuple{2, T}}, 
                               ps₄::NTuple{GN4, NTuple{2, T}}) where {GN1, GN2, GN3, GN4, T}
    uniquePairs = NTuple{4, T}[]
    uPairCoeffs = Array{T}(undef, GN1*GN2*GN3*GN4)
    flagRijk = flags[1:3]
    i = 0

    if (ps₁ == ps₂ && ps₂ == ps₃ && ps₃ == ps₄ && flags[1] && flags[2] && flags[3])
        getIntCore1111!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁)

    elseif (ps₁ == ps₂ && ps₃ == ps₄ && flags[1] && flags[2])
        getIntX1X1X2X2!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₃)

    elseif (ps₁ == ps₄ && ps₂ == ps₃ && flags[4] && flags[5])
        getIntX1X2X2X1!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₂)

    elseif (ps₁ == ps₃ && ps₂ == ps₄ && flags[3])
        getIntX1X2X1X2!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₂)

    elseif (ps₁ == ps₂ && flags[1])
        getIntX1X1X2X3!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₃, ps₄)

    elseif (ps₃ == ps₄ && flags[2])
        getIntX1X2X3X3!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₂, ps₃)

    elseif (ps₁ == ps₄ && flags[4])
        getIntX1X2X3X1!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₂, ps₃)

    elseif (ps₂ == ps₃ && flags[5])
        getIntX1X2X2X3!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₂, ps₄)
    else
        getIntX1X2X3X4!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₂, ps₃, ps₄)
    end

    uniquePairs, uPairCoeffs
end


function getTwoBodyInt(::Type{T}, ::Val{D}, ∫2e::F, @nospecialize(optPosArgs::Tuple), 
                       iBl::Union{iBlTs[2], iBlTs[3]}, bfs::NTupleOfFGTBF{4, T, D}) where 
                       {T, D, F<:Function}
    (R₁, ijk₁, ps₁, 𝑙₁), (R₂, ijk₂, ps₂, 𝑙₂), (R₃, ijk₃, ps₃, 𝑙₃), (R₄, ijk₄, ps₄, 𝑙₄) = 
    reformatIntData1(iBl, bfs)

    𝑙₁==𝑙₂==𝑙₃==𝑙₄==0 || 
    isIntZero(F, optPosArgs, R₁,R₂,R₃,R₄, ijk₁,ijk₂,ijk₃,ijk₄) && (return T(0.0))

    f1 = (R₁ == R₂ && ijk₁ == ijk₂)
    f2 = (R₃ == R₄ && ijk₃ == ijk₄)
    f3 = (R₁ == R₃ && ijk₁ == ijk₃ && R₂ == R₄ && ijk₂ == ijk₄)
    f4 = (R₁ == R₄ && ijk₁ == ijk₄)
    f5 = (R₂ == R₃ && ijk₂ == ijk₃)

    uniquePairs, uPairCoeffs = getTwoBodyUniquePairs((f1, f2, f3, f4, f5), 
                                                     ps₁, ps₂, ps₃, ps₄)
    map(uniquePairs, uPairCoeffs) do x, y
        ∫2e(Val(D), optPosArgs..., R₁, ijk₁, x[1], 
                                   R₂, ijk₂, x[2], 
                                   R₃, ijk₃, x[3], 
                                   R₄, ijk₄, x[4])::T * y
    end |> sum
end


lAndGNof(::FGTBasisFuncs1O{<:Any, <:Any, 𝑙, GN}) where {𝑙, GN} = (𝑙, GN)

function orderFGTBG((a, b)::NTupleOfFGTBF{2})
    lAndGNof(a) > lAndGNof(b) ? (b, a) : (a, b)
end

function orderFGTBG((a, b, c, d)::NTupleOfFGTBF{4})
    a, b = orderFGTBG((a, b))
    c, d = orderFGTBG((c, d))
    lAndGNof(a) > lAndGNof(c) ? (c, d, a, b) : (a, b, c, d)
end

get1BCompInt(::Type{T}, ::Val{D}, ∫::F, @nospecialize(optPosArgs::Tuple), iBl::BL, 
             ::NTuple{2, Int}, bfs::NTupleOfFGTBF{2, T, D}) where 
            {T, D, F<:Function, BL<:Union{iBlTs[1], iBlTs[3]}} = 
getOneBodyInt(T, Val(D), ∫, optPosArgs, iBl, orderFGTBG(bfs))

get2BCompInt(::Type{T}, ::Val{D}, ∫::F, @nospecialize(optPosArgs::Tuple), iBl::iBlTs[2], 
             ::NTuple{4, Int}, bfs::NTupleOfFGTBF{4, T, D}) where 
            {T, D, F<:Function} = 
getTwoBodyInt(T, Val(D), ∫, optPosArgs, iBl, bfs)

get2BCompInt(::Type{T}, ::Val{D}, ∫::F, @nospecialize(optPosArgs::Tuple), iBl::iBlTs[3], 
             ::NTuple{4, Int}, bfs::NTupleOfFGTBF{4, T, D}) where 
            {T, D, F<:Function} = 
getTwoBodyInt(T, Val(D), ∫, optPosArgs, iBl, orderFGTBG(bfs))

get1BCompInt(::Type{T}, ::Val{D}, ::typeof(∫nucAttractionCore), 
             nucAndCoords::Tuple{Tuple{String, Vararg{String, NNMO}}, 
                                 Tuple{NTuple{D, T}, Vararg{NTuple{D, T}, NNMO}}}, 
             iBl::Union{iBlTs[1], iBlTs[3]}, ::NTuple{2, Int}, 
             bfs::NTupleOfFGTBF{2, T, D}) where {T, D, NNMO} = 
mapreduce(+, nucAndCoords[1], nucAndCoords[2]) do ele, coord
    getOneBodyInt(T, Val(D), ∫nucAttractionCore, (getCharge(ele), coord), iBl, 
                  orderFGTBG(bfs))
end
                          #       j==i      j!=i
const Int1eBIndexLabels = Dict([( true,), (false,)] .=> [Val(:aa), Val(:ab)])

getBF(::Val, ::Type{T}, ::Val{D}, b::SpatialBasis{T, D}, i) where {T, D} = 
@inbounds getindex(b, i)

# When not containing BasisFuncs
getBF(::Val{false}, ::Type{T}, ::Val{D}, b::BasisFuncMix{T, D}, i) where {T, D} = 
@inbounds getindex(b.BasisFunc, i)

# 1e integrals for BasisFuncs/BasisFuncMix-mixed bases
function get1BCompIntCore(::Type{T}, ::Val{D}, ::Val{BL}, ::Val{:aa}, 
                          ∫::F, @nospecialize(optPosArgs::Tuple), 
                          sizes::NTuple{2, Int}, 
                          @nospecialize(bfs::NTuple{2, BT} where 
                                        {TL, DL, BT<:SpatialBasis{TL, DL}})) where 
                         {T, D, BL, F<:Function}
    a = bfs[1]
    BN = if BL
        sizes[1]
    else
        centerNumOf(a)
    end
    res = Array{T}(undef, BN, BN)
    for j in OneTo(BN), i in OneTo(j)
        @inbounds res[j,i] = res[i,j] = 
                  get1BCompInt(T, Val(D), ∫, optPosArgs, (j==i,), sizes, 
                               getBF(Val(BL), T, Val(D), a, i), 
                               getBF(Val(BL), T, Val(D), a, j))
    end
    res
end

function get1BCompIntCore(::Type{T}, ::Val{D}, ::Val{BL}, ::Val{:ab}, 
                          ∫::F, @nospecialize(optPosArgs::Tuple), 
                          sizes::NTuple{2, Int}, 
                          @nospecialize(bfs::NTupleOfSBN{2})) where 
                         {T, D, BL, F<:Function}
    a, b = bfs
    BN1, BN2 = if BL
        sizes
    else
        centerNumOf(a), centerNumOf(b)
    end
    res = Array{T}(undef, BN1, BN2)
    for j in OneTo(BN2), i in OneTo(BN1)
        @inbounds res[i,j] = 
                  get1BCompInt(T, Val(D), ∫, optPosArgs, Val(false), sizes, 
                               getBF(Val(BL), T, Val(D), a, i), 
                               getBF(Val(BL), T, Val(D), b, j))
    end
    res
end

                          # a==b    lk,    lj,    kj, ki/ji    ijkl
const Int2eBIndexLabels = Dict([( true,  true,  true,  true), #1111
                                ( true, false, false,  true), #1122
                                (false,  true, false,  true), #1212
                                (false,  true, false, false), #1323
                                (false, false,  true, false), #1223
                                (false, false,  true,  true), #1112
                                (false, false, false,  true), #1123
                                ( true,  true,  true, false), #1222
                                ( true, false, false, false), #1233
                                (false, false, false, false)  #1234
                               ] .=> 
                               [Val(:aaaa), Val(:aabb), Val(:abab), Val(:acbc), Val(:abbc), 
                                Val(:aabc), Val(:aabc), Val(:abcc), Val(:abcc), Val(:abcd)])

# 2e integrals for BasisFuncs/BasisFuncMix-mixed bases
function get2BCompIntCore(::Type{T}, ::Val{D}, ::Val{BL}, ::Val{:aaaa}, 
                          ∫::F, @nospecialize(optPosArgs::Tuple), 
                          sizes::NTuple{4, Int}, 
                          @nospecialize(bfs::NTuple{4, BT} where 
                                        {TL, DL, BT<:SpatialBasis{TL, DL}})) where 
                         {T, D, BL, F<:Function}
    a = bfs[1]
    BN = if BL
        sizes[1]
    else
        centerNumOf(a)
    end
    res = Array{T}(undef, BN, BN, BN, BN)
    for l in OneTo(BN), k in OneTo(l), j in OneTo(l), i in (OneTo∘ifelse)(l==j, k, j)
        iBl = (l==k, l==j, k==j, ifelse(l==j, k, j)==i)
        @inbounds res[l, k, j, i] = res[k, l, j, i] = res[k, l, i, j] = res[l, k, i, j] = 
                  res[i, j, l, k] = res[j, i, l, k] = res[j, i, k, l] = res[i, j, k, l] = 
                  get2BCompInt(T, Val(D), ∫, optPosArgs, iBl, sizes, 
                               getBF(Val(BL), T, Val(D), a, i), 
                               getBF(Val(BL), T, Val(D), a, j), 
                               getBF(Val(BL), T, Val(D), a, k), 
                               getBF(Val(BL), T, Val(D), a, l))
    end
    res
end

function get2BCompIntCore(::Type{T}, ::Val{D}, ::Val{BL}, ::Val{:aabb}, 
                          ∫::F, @nospecialize(optPosArgs::Tuple), 
                          sizes::NTuple{4, Int}, 
                          @nospecialize(bfs::Tuple{BT1, BT1, BT2, BT2} where 
                                             {TL, DL, BT1<:SpatialBasis{TL, DL}, 
                                                      BT2<:SpatialBasis{TL, DL}})) where 
                         {T, D, BL, F<:Function}
    a, _, b, _ = bfs
    BN1, BN2 = if BL
        sizes[1], sizes[3]
    else
        centerNumOf(a), centerNumOf(b)
    end
    res = Array{T}(undef, BN1, BN1, BN2, BN2)
    for l in OneTo(BN2), k in OneTo(l), j in OneTo(BN1), i in OneTo(j)
        iBl = (l==k, Val(false), Val(false), j==i)
        @inbounds res[i, j, l, k] = res[j, i, l, k] = res[j, i, k, l] = res[i, j, k, l] = 
                  get2BCompInt(T, Val(D), ∫, optPosArgs, iBl, sizes, 
                               getBF(Val(BL), T, Val(D), a, i), 
                               getBF(Val(BL), T, Val(D), a, j), 
                               getBF(Val(BL), T, Val(D), b, k), 
                               getBF(Val(BL), T, Val(D), b, l))
    end
    res
end

function get2BCompIntCore(::Type{T}, ::Val{D}, ::Val{BL}, ::Val{:abab}, 
                          ∫::F, @nospecialize(optPosArgs::Tuple), 
                          sizes::NTuple{4, Int}, 
                          @nospecialize(bfs::Tuple{BT1, BT2, BT1, BT2} where 
                                             {TL, DL, BT1<:SpatialBasis{TL, DL}, 
                                                      BT2<:SpatialBasis{TL, DL}})) where 
                         {T, D, BL, F<:Function}
    a, b, _, _ = bfs
    BN1, BN2 = if BL
        sizes[1], sizes[2]
    else
        centerNumOf(a), centerNumOf(b)
    end
    res = Array{T}(undef, BN1, BN2, BN1, BN2)
    rng = product(OneTo(BN2), OneTo(BN1))
    for (x, (l,k)) in enumerate(rng), (_, (j,i)) in zip(OneTo(x), rng)
        iBl = (Val(false), l==j, Val(false), ifelse(l==j, k==i, false))
        @inbounds res[k, l, i, j] = res[i, j, k, l] = 
                  get2BCompInt(T, Val(D), ∫, optPosArgs, iBl, sizes, 
                               getBF(Val(BL), T, Val(D), a, i), 
                               getBF(Val(BL), T, Val(D), b, j), 
                               getBF(Val(BL), T, Val(D), a, k), 
                               getBF(Val(BL), T, Val(D), b, l))
    end
    res
end

function get2BCompIntCore(::Type{T}, ::Val{D}, ::Val{BL}, ::Val{:aabc}, 
                          ∫::F, @nospecialize(optPosArgs::Tuple), 
                          sizes::NTuple{4, Int}, 
                          @nospecialize(bfs::Tuple{BT1, BT1, BT2, BT3} where 
                                             {TL, DL, BT1<:SpatialBasis{TL, DL}, 
                                                      BT2<:SpatialBasis{TL, DL}, 
                                                      BT3<:SpatialBasis{TL, DL}})) where 
                         {T, D, BL, F<:Function}
    a, _, b, c = bfs
    BN1, BN2, BN3 = if BL
        sizes[1], sizes[3], sizes[4]
    else
        centerNumOf(a), centerNumOf(b), centerNumOf(c)
    end
    res = Array{T}(undef, BN1, BN1, BN2, BN3)
    for l in OneTo(BN3), k in OneTo(BN2), j in OneTo(BN1), i in OneTo(j)
        iBl = (Val(false), Val(false), Val(false), j==i)
        @inbounds res[j, i, k, l] = res[i, j, k, l] = 
                  get2BCompInt(T, Val(D), ∫, optPosArgs, iBl, sizes, 
                               getBF(Val(BL), T, Val(D), a, i), 
                               getBF(Val(BL), T, Val(D), a, j), 
                               getBF(Val(BL), T, Val(D), b, k), 
                               getBF(Val(BL), T, Val(D), c, l))
    end
    res
end

function get2BCompIntCore(::Type{T}, ::Val{D}, ::Val{BL}, ::Val{:abcc}, 
                          ∫::F, @nospecialize(optPosArgs::Tuple), 
                          sizes::NTuple{4, Int}, 
                          @nospecialize(bfs::Tuple{BT1, BT2, BT3, BT3} where 
                                             {TL, DL, BT1<:SpatialBasis{TL, DL}, 
                                                      BT2<:SpatialBasis{TL, DL}, 
                                                      BT3<:SpatialBasis{TL, DL}})) where 
                         {T, D, BL, F<:Function}
    a, b, c, _ = bfs
    BN1, BN2, BN3 = if BL
        sizes[1], sizes[2], sizes[3]
    else
        centerNumOf(a), centerNumOf(b), centerNumOf(c)
    end
    res = Array{T}(undef, BN1, BN2, BN3, BN3)
    for l in OneTo(BN3), k in OneTo(l), j in OneTo(BN2), i in OneTo(BN1)
        iBl = (l==k, Val(false), Val(false), Val(false))
        @inbounds res[i, j, l, k] = res[i, j, k, l] = 
                  get2BCompInt(T, Val(D), ∫, optPosArgs, iBl, sizes, 
                               getBF(Val(BL), T, Val(D), a, i), 
                               getBF(Val(BL), T, Val(D), b, j), 
                               getBF(Val(BL), T, Val(D), c, k), 
                               getBF(Val(BL), T, Val(D), c, l))
    end
    res
end

const IndexABXYbools = 
      Dict([Val{:acbc}, Val{:abbc}, Val{:abcd}] .=> 
          [(j::Int, _,      l::Int) -> (Val(false), l==j, Val(false), Val(false)), 
           (j::Int, k::Int, _     ) -> (Val(false), Val(false), k==j, Val(false)), 
           (_,      _,      _     ) ->  Val(false)])

function get2BCompIntCore(::Type{T}, ::Val{D}, ::Val{BL}, ::IDV, 
                          ∫::F, @nospecialize(optPosArgs::Tuple), 
                          sizes::NTuple{4, Int}, 
                          @nospecialize(bfs::NTupleOfSBN{4})) where 
                         {T, D, BL, IDV<:Union{Val{:acbc}, Val{:abbc}, Val{:abcd}}, 
                          F<:Function}
    a, b, c, d = bfs
    BN1, BN2, BN3, BN4 = if BL
        sizes
    else
        centerNumOf.(bfs)
    end
    res = Array{T}(undef, BN1, BN2, BN3, BN4)
    for l in OneTo(BN4), k in OneTo(BN3), j in OneTo(BN2), i in OneTo(BN1)
        iBl = IndexABXYbools[IDV](j,k,l)
        @inbounds res[i,j,k,l] = 
                  get2BCompInt(T, Val(D), ∫, optPosArgs, iBl, sizes, 
                               getBF(Val(BL), T, Val(D), a, i), 
                               getBF(Val(BL), T, Val(D), b, j), 
                               getBF(Val(BL), T, Val(D), c, k), 
                               getBF(Val(BL), T, Val(D), d, l))
    end
    res
end

getBasisIndexL(::Val{2}, iBl::Tuple{Bool}) = Int1eBIndexLabels[iBl]
getBasisIndexL(::Val{4}, iBl::NTuple{4, Bool}) = Int2eBIndexLabels[iBl]
getBasisIndexL(::Val{2}, ::Val{false}) = Int1eBIndexLabels[(false,)]
getBasisIndexL(::Val{4}, ::Val{false}) = Int2eBIndexLabels[(false, false, false, false)]
getBasisIndexL(::Val{4}, iBl::NTuple{4, Any}) = Int2eBIndexLabels[getBool.(iBl)]

get1BCompInt(::Type{T}, ::Val{D}, ∫::F, @nospecialize(optPosArgs::Tuple), iBl::BL, 
             sizes::NTuple{2, Int}, @nospecialize(bfs::NTupleOfSBN{2})) where 
            {T, D, F<:Function, BL<:Union{iBlTs[1], iBlTs[3]}} = 
get1BCompIntCore(T, Val(D), Val(true), getBasisIndexL(Val(2), iBl), ∫, optPosArgs, sizes, 
                 bfs)

get2BCompInt(::Type{T}, ::Val{D}, ∫::F, @nospecialize(optPosArgs::Tuple), iBl::BL, 
             sizes::NTuple{4, Int}, @nospecialize(bfs::NTupleOfSBN{4})) where 
            {T, D, F<:Function, BL<:Union{iBlTs[2], iBlTs[3]}} = 
get2BCompIntCore(T, Val(D), Val(true), getBasisIndexL(Val(4), iBl), ∫, optPosArgs, sizes, 
                 bfs)

get1BCompInt(::Type{T}, ::Val{D}, ∫::F, @nospecialize(optPosArgs::Tuple), iBl::BL, 
             sizes::NTuple{2, Int}, @nospecialize(bfs::NTupleOfSB1{2})) where 
            {T, D, F<:Function, BL<:Union{iBlTs[1], iBlTs[3]}} = 
if any(bf isa EmptyBasisFunc for bf in bfs)
    zero(T)
else
    (sum∘get1BCompIntCore)(T, Val(D), Val(false), getBasisIndexL(Val(2), iBl), ∫, 
                           optPosArgs, sizes, bfs)
end

get2BCompInt(::Type{T}, ::Val{D}, ∫::F, @nospecialize(optPosArgs::Tuple), iBl::BL, 
             sizes::NTuple{4, Int}, @nospecialize(bfs::NTupleOfSB1{4})) where 
            {T, D, F<:Function, BL<:Union{iBlTs[2], iBlTs[3]}} = 
if any(bf isa EmptyBasisFunc for bf in bfs)
    zero(T)
else
    (sum∘get2BCompIntCore)(T, Val(D), Val(false), getBasisIndexL(Val(4), iBl), ∫, 
                           optPosArgs, sizes, bfs)
end

get1BCompInt(::Type{T}, ::Val{D}, ∫::F, @nospecialize(optPosArgs::Tuple), iBl::BL, 
             sizes::NTuple{2, Int}, @nospecialize(bfs::Vararg{SpatialBasis, 2})) where 
            {T, D, F<:Function, BL<:Union{iBlTs[1], iBlTs[3]}} = 
get1BCompInt(T, Val(D), ∫, optPosArgs, iBl, sizes, bfs)

get2BCompInt(::Type{T}, ::Val{D}, ∫::F, @nospecialize(optPosArgs::Tuple), iBl::BL, 
             sizes::NTuple{4, Int}, @nospecialize(bfs::Vararg{SpatialBasis, 4})) where 
            {T, D, F<:Function, BL<:Union{iBlTs[2], iBlTs[3]}} = 
get2BCompInt(T, Val(D), ∫, optPosArgs, iBl, sizes, bfs)


@inline function update2DarrBlock!(arr::AbstractMatrix{T1}, 
                                   block::Union{T1, AbstractMatrix{T1}}, 
                                   I::T2, J::T2) where {T1, T2<:UnitRange{Int}}
    arr[I, J] .= block
    arr[J, I] .= (J!=I ? transpose(block) : block)
    nothing
end


function getOneBodyInts(∫1e::F, optPosArgs::Tuple, 
                        basisSet::AbstractVector{<:GTBasisFuncs{T, D}}) where 
                       {F<:Function, T, D}
    subSize = orbitalNumOf.(basisSet)
    accuSize = vcat(0, accumulate(+, subSize))
    totalSize = subSize |> sum
    buf = Array{T}(undef, totalSize, totalSize)
    idxShift = firstindex(basisSet) - 1
    BN = length(basisSet)
    Threads.@threads for k in (OneTo∘triMatEleNum)(BN)
        i, j = convert1DidxTo2D(BN, k)
        @inbounds begin
            int = get1BCompInt(T, Val(D), ∫1e, optPosArgs, (j==i,), 
                               (subSize[i],  subSize[j]), 
                               basisSet[i+idxShift], basisSet[j+idxShift])
            rowRange = accuSize[i]+1 : accuSize[i+1]
            colRange = accuSize[j]+1 : accuSize[j+1]
            update2DarrBlock!(buf, int, rowRange, colRange)
        end
    end
    buf
end

function getOneBodyInts(∫1e::F, optPosArgs::Tuple, 
                        basisSet::AbstractVector{<:GTBasisFuncs{T, D, 1}}) where 
                       {F<:Function, T, D}
    BN = length(basisSet)
    buf = Array{T}(undef, BN, BN)
    idxShift = firstindex(basisSet) - 1
    Threads.@threads for k in (OneTo∘triMatEleNum)(BN)
        i, j = convert1DidxTo2D(BN, k)
        @inbounds begin
            int = get1BCompInt(T, Val(D), ∫1e, optPosArgs, (j==i,), 
                               (1, 1), basisSet[i+idxShift], basisSet[j+idxShift])
            buf[j, i] = buf[i, j] = int
        end
    end
    buf
end


permuteDims(arr::AbstractArray{T, N}, order) where {T, N} = permutedims(arr, order)
permuteDims(arr::Number, _) = itself(arr)

@inline function update4DarrBlock!(arr::AbstractArray{T1, 4}, 
                                   block::Union{AbstractArray{T1, 4}, T1}, 
                                   I::T2, J::T2, K::T2, L::T2) where 
                                  {T1, T2<:UnitRange{Int}}
    local blockTemp
    arr[I, J, K, L] .= block
    arr[J, I, K, L] .= (blockTemp = (J!=I ? permuteDims(block, (2,1,3,4)) : block))
    arr[J, I, L, K] .= (blockTemp = (L!=K ? permuteDims(block, (2,1,4,3)) : blockTemp))
    arr[I, J, L, K] .= (blockTemp = (I!=J ? permuteDims(block, (1,2,4,3)) : blockTemp))
    arr[L, K, I, J] .= (blockTemp = ((L, K, I, J) != (I, J, L, K) ? 
                                            permuteDims(block, (4,3,1,2)) : blockTemp))
    arr[K, L, I, J] .= (blockTemp = (K!=L ? permuteDims(block, (3,4,1,2)) : blockTemp))
    arr[K, L, J, I] .= (blockTemp = (J!=I ? permuteDims(block, (3,4,2,1)) : blockTemp))
    arr[L, K, J, I] .= (L!=K ? permuteDims(block, (4,3,2,1)) : blockTemp)
    nothing
end


function getTwoBodyInts(∫2e::F, optPosArgs::Tuple, 
                        basisSet::AbstractVector{<:GTBasisFuncs{T, D}}) where 
                       {F<:Function, T, D}
    subSize = orbitalNumOf.(basisSet)
    accuSize = vcat(0, accumulate(+, subSize)...)
    totalSize = subSize |> sum
    buf = Array{T}(undef, totalSize, totalSize, totalSize, totalSize)
    idxShift = firstindex(basisSet) - 1
    BN = length(basisSet)
    @sync for m in (OneTo∘triMatEleNum∘triMatEleNum)(BN)
        Threads.@spawn begin
            i, j, k, l = convert1DidxTo4D(BN, m)
            iBl = (l==k, l==j, k==j, ifelse(l==j, k, j)==i)
            @inbounds begin
                I = accuSize[i]+1 : accuSize[i+1]
                J = accuSize[j]+1 : accuSize[j+1]
                K = accuSize[k]+1 : accuSize[k+1]
                L = accuSize[l]+1 : accuSize[l+1]
                int = get2BCompInt(T, Val(D), ∫2e, optPosArgs, iBl, 
                                   (subSize[i],  subSize[j],  subSize[k],  subSize[l]), 
                                   basisSet[i+idxShift], basisSet[j+idxShift], 
                                   basisSet[k+idxShift], basisSet[l+idxShift])
                update4DarrBlock!(buf, int, I, J, K, L)
            end
        end
    end
    buf
end

function getTwoBodyInts(∫2e::F, optPosArgs::Tuple, 
                        basisSet::AbstractVector{<:GTBasisFuncs{T, D, 1}}) where 
                       {F<:Function, T, D}
    BN = length(basisSet)
    buf = Array{T}(undef, BN, BN, BN, BN)
    idxShift = firstindex(basisSet) - 1
    @sync for m in (OneTo∘triMatEleNum∘triMatEleNum)(BN)
        Threads.@spawn begin
            i, j, k, l = convert1DidxTo4D(BN, m)
            iBl = (l==k, l==j, k==j, ifelse(l==j, k, j)==i)
            @inbounds begin
                int = get2BCompInt(T, Val(D), ∫2e, optPosArgs, iBl, 
                                   (1, 1, 1, 1), 
                                   basisSet[i+idxShift], basisSet[j+idxShift], 
                                   basisSet[k+idxShift], basisSet[l+idxShift])
                buf[l, k, j, i] = buf[k, l, j, i] = buf[k, l, i, j] = buf[l, k, i, j] = 
                buf[i, j, l, k] = buf[j, i, l, k] = buf[j, i, k, l] = buf[i, j, k, l] = int
            end
        end
    end
    buf
end


"""

    eeIuniqueIndicesOf(basisSetSize::Int) -> Vector{Vector{Int}}

Return the unique matrix element indices (in the chemists' notation) of electron-electron 
interactions given the size of a basis set.
"""
function eeIuniqueIndicesOf(basisSetSize::Int)
    uniqueIdx = fill(Int[0,0,0,0], (triMatEleNum∘triMatEleNum)(basisSetSize))
    index = 1
    for l in OneTo(basisSetSize), k in OneTo(l), 
        j in OneTo(l), i in (OneTo∘ifelse)(l==j, k, j)
        uniqueIdx[index] = [i, j, k, l]
        index += 1
    end
    uniqueIdx
end