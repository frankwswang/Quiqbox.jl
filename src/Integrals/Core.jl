export eeIuniqueIndicesOf

using QuadGK: quadgk
using SpecialFunctions: erf
using LinearAlgebra: dot

# Reference: DOI: 10.1088/0143-0807/31/1/004
factorialL(l::Integer) = FactorialsLs[l+1]

getGKQorder(T::Type{<:Real}) = ifelse(getAtolVal(T) >= getAtolVal(Float64), 13, 26)

πroot(::Type{T}) where {T} = sqrt(π*T(1))

function F0Core(u::T) where {T}
    ur = sqrt(u)
    πroot(T) * erf(ur) / (2ur)
end

function FγCore1(γ::Int, u::T, rtol=0.5getAtolVal(T), order=getGKQorder(T)) where {T}
    quadgk(t -> t^(2γ)*exp(-u*t^2), 0, T(1); order, rtol)[1]
end

function FγCore2(γ::Int, u::T) where {T}
    t = exp(-u) * sum(factorialL(γ-k)/(4^k * factorialL(2*(γ-k)) * u^(k+1)) for k=0:(γ-1))
    T(factorialL(2γ) / (2factorialL(γ)) * (πroot(T)*erf(√u) / (4^γ * u^(γ + T(0.5))) - t))
end

const FγSolverThreshold =  [ 6.6701,  6.6645,  6.6658,  8.0597,  6.8447,  7.6676,  9.3999, 
                             9.3547, 11.3685,  9.1267, 60.9556, 62.7040, 62.7022, 65.9929, 
                            68.3654, 68.3522, 71.1386, 70.8620, 73.3747, 73.8483, 77.8355, 
                            76.9363, 82.0030, 81.9330]

function Fγ(γ::Int, u::T, uEps::Real=getAtolVal(T)) where {T}
    if u < uEps
        T(1 / (2γ + 1))
    elseif γ == 0
        F0Core(u)
    else
        u < FγSolverThreshold[γ] ? FγCore1(γ, u) : FγCore2(γ, u)
    end
end

function F₀toFγCore(γ::Int, u::T, Fγu::T) where {T}
    res = Array{T}(undef, γ+1)
    res[end] = Fγu
    for i in γ:-1:3
        res[i] = (2u*res[i+1] + exp(-u)) / (2i - 1)
    end
    if γ > 0
        res[1] = F0Core(u)
        res[2] = FγCore1(1, u)
    end
    res
end

function F₀toFγ(γ::Int, u::T, uEps::Real=getAtolVal(T)) where {T}
    if uEps >= getAtolVal(Float64)
        F₀toFγCore(γ, u, Fγ(γ, u)) # max(err) < getAtolVal(Float64)
    else
        vcat(F0Core(u), [FγCore1(i, u) for i=1:γ])
    end
end


function genIntOverlapCore(Δx::T, 
                           i₁::Int, α₁::T, 
                           i₂::Int, α₂::T) where {T}
    res = T(0.0)
    for l₁ in 0:(i₁÷2), l₂ in 0:(i₂÷2)
        Ω = i₁ + i₂ - 2*(l₁ + l₂)
        halfΩ = Ω÷2
        oRange = 0:halfΩ
        Δx == 0.0 && (iseven(Ω) ? (oRange = halfΩ:halfΩ) : continue)
        for o in oRange
            res += (-1)^o * factorial(Ω) * 
                    α₁^(i₂ - l₁ - 2l₂ - o) * 
                    α₂^(i₁ - l₂ - 2l₁ - o) * 
                    (α₁ + α₂)^(2 * (l₁ + l₂) + o) * 
                    Δx^(Ω-2o) / 
                    ( 4^(l₁ + l₂ + o) * 
                        factorial(l₁) * 
                        factorial(l₂) * 
                        factorial(o ) * 
                        factorial(i₁ - 2l₁) * 
                        factorial(i₂ - 2l₂) * 
                        factorial(Ω  - 2o ) )
        end
    end
    res
end

∫overlapCore(R₁::NTuple{3, T}, R₂::NTuple{3, T}, 
             ijk₁::NTuple{3, Int}, α₁::T, 
             ijk₂::NTuple{3, Int}, α₂::T) where {T} = 
∫overlapCore(R₁.-R₂, ijk₁, α₁, ijk₂, α₂)


function ∫overlapCore(ΔR::NTuple{3, T}, 
                      ijk₁::NTuple{3, Int}, α₁::T, 
                      ijk₂::NTuple{3, Int}, α₂::T) where {T}
    for n in (ijk₁..., ijk₂...)
        n < 0 && return T(0.0)
    end

    α = α₁ + α₂
    res = (π/α)^T(1.5) * exp(-α₁ * α₂ / α * sum(abs2, ΔR))

        for (i₁, i₂, ΔRᵢ) in zip(ijk₁, ijk₂, ΔR)
            res *= (-1)^(i₁) * factorial(i₁) * factorial(i₂) / α^(i₁+i₂) * 
                   genIntOverlapCore(ΔRᵢ, i₁, α₁, i₂, α₂)
        end

    res
end

function ∫elecKineticCore(R₁::NTuple{3, T}, R₂::NTuple{3, T}, 
                          ijk₁::NTuple{3, Int}, α₁::T,
                          ijk₂::NTuple{3, Int}, α₂::T) where {T}
    ΔR = R₁ .- R₂
    shifts = ((2,0,0), (0,2,0), (0,0,2))
    ( α₂ * (4*sum(ijk₂) + 6) * ∫overlapCore(ΔR, ijk₁, α₁, ijk₂, α₂) - 4 * α₂^2 * 
      sum(∫overlapCore.(Ref(ΔR), Ref(ijk₁), α₁, map.(+, Ref(ijk₂), shifts), α₂)) - 
      sum(ijk₂ .* (ijk₂.-1) .* 
          ∫overlapCore.(Ref(ΔR), Ref(ijk₁), α₁, map.(-, Ref(ijk₂), shifts), α₂)) ) / 2
end

@inline function genIntTerm1(Δx::T, 
                             l₁::Int, o₁::Int, 
                             l₂::Int, o₂::Int, 
                             i₁::Int, α₁::T, 
                             i₂::Int, α₂::T) where {T}
    @inline (r) -> 
        (-1)^(o₂+r) * factorial(o₁+o₂) * α₁^(o₂-l₁-r) * α₂^(o₁-l₂-r) * Δx^(o₁+o₂-2r) / 
        (
            4^(l₁+l₂+r) * 
            factorial(l₁) * factorial(l₂) * factorial(o₁) * factorial(o₂) * 
            factorial(r) * factorial(i₁-2l₁-o₁) * factorial(i₂-2l₂-o₂) * 
            factorial(o₁+o₂-2r)
        )
end

@inline function genIntTerm2(Δx::T, 
                             α::T, 
                             o₁::Int, 
                             o₂::Int, 
                             μ::Int, 
                             r::Int) where {T}
    @inline (u) ->
        (-1)^u * factorial(μ) * Δx^(μ-2u) / 
        (4^u * factorial(u) * factorial(μ-2u) * α^(o₁+o₂-r+u))
end

function genIntNucAttCore1(ΔRR₀::NTuple{3, T}, ΔR₁R₂::NTuple{3, T}, β::T, 
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

            μˣ, μʸ, μᶻ = μv = @. ijk₁ + ijk₂ - 2*(lmn₁ + lmn₂) - (opq₁ + opq₂)
            μsum = sum(μv)
            Fγs = F₀toFγ(μsum, β)
            core1s = genIntTerm1.(ΔR₁R₂, lmn₁, opq₁, lmn₂, opq₂, ijk₁, α₁, ijk₂, α₂)

            for r in 0:((o₁+o₂)÷2), s in 0:((p₁+p₂)÷2), t in 0:((q₁+q₂)÷2)

                rst = (r, s, t)
                tmp = T(0.0)
                core2s = genIntTerm2.(ΔRR₀, α₁+α₂, opq₁, opq₂, μv, rst)

                for u in 0:(μˣ÷2), v in 0:(μʸ÷2), w in 0:(μᶻ÷2)
                    γ = μsum - u - v - w
                    tmp += (((u, v, w) .|> core2s) |> prod) * 2Fγs[γ+1]
                end

                A += ((rst .|> core1s) |> prod) * tmp

            end
        end

    end
    A
end

function ∫nucAttractionCore(Z₀::Int, R₀::NTuple{3, T}, 
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
    res = -Z₀ * π / α * exp(-α₁ * α₂ / α * sum(abs2, ΔR₁R₂))
    res *= (-1)^sum(ijk₁ .+ ijk₂) * (factorial.((ijk₁..., ijk₂...)) |> prod) * 
            genIntNucAttCore1(ΔRR₀, ΔR₁R₂, β, ijk₁, α₁, ijk₂, α₂)
    res
end

@inline function genIntTerm3(Δx, l₁, o₁, l₂, o₂, i₁, α₁, i₂, α₂)
    @inline (r) -> 
        (-1)^(o₂+r) * factorial(o₁+o₂) * α₁^(o₂-l₁-r) * α₂^(o₁-l₂-r) * 
        (α₁+α₂)^(2*(l₁+l₂) + r) * Δx^(o₁+o₂-2r) / 
        (
            4^(l₁+l₂+r) * 
            factorial(l₁) * factorial(l₂) * factorial(o₁) * factorial(o₂) * 
            factorial(r) * factorial(i₁-2l₁-o₁) * factorial(i₂-2l₂-o₂) * 
            factorial(o₁+o₂-2r)
        )
end

@inline function genIntTerm4(Δx, η, μ)
    @inline (u) ->
        (-1)^u * factorial(μ) * η^(μ-u) * Δx^(μ-2u) / 
        (4^u * factorial(u) * factorial(μ-2u))
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
                @. IJK - (lmn₁ + lmn₂ + lmn₃ + lmn₄) * 2 - (opq₁ + opq₂ + opq₃ + opq₄)
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
                    tmp += prod((u, v, w) .|> core3s) * 2Fγs[γ+1]
                end

                A += (rst₁ .|> core1s |> prod) * (rst₂ .|> core2s |> prod) * tmp

            end
        end

    end
    A
end

function ∫eeInteractionCore(R₁::NTuple{3, T}, ijk₁::NTuple{3, Int}, α₁::T, 
                            R₂::NTuple{3, T}, ijk₂::NTuple{3, Int}, α₂::T,
                            R₃::NTuple{3, T}, ijk₃::NTuple{3, Int}, α₃::T, 
                            R₄::NTuple{3, T}, ijk₄::NTuple{3, Int}, α₄::T) where {T}
    ΔRl = R₁ .- R₂
    ΔRr = R₃ .- R₄
    αl = α₁ + α₂
    αr = α₃ + α₄
    ηl = α₁ * α₂ / αl
    ηr = α₃ * α₄ / αr
    ΔRc = @. (α₁*R₁ + α₂*R₂)/αl - (α₃*R₃ + α₄*R₄)/αr
    η = αl * αr / (α₁ + α₂ + α₃ + α₄)
    β = η * sum(abs2, ΔRc)
    res = π^T(2.5) / (αl * αr * sqrt(αl + αr)) * exp(-ηl * sum(abs2, ΔRl)) * 
                                                 exp(-ηr * sum(abs2, ΔRr))
    res *= ( @. (-1)^(ijk₁ + ijk₂) * factorial(ijk₁) * factorial(ijk₂) * 
                factorial(ijk₃) * factorial(ijk₄) / 
                αl^(ijk₁+ijk₂) / αr^(ijk₃+ijk₄) ) |> prod
        J = ∫eeInteractionCore1234(ΔRl, ΔRr, ΔRc, β, η, 
                                   ijk₁, α₁, ijk₂, α₂, ijk₃, α₃, ijk₄, α₄)
    res * J
end


reformatIntData2((o1, o2)::NTuple{2, T}, flag::Bool) where {T} = 
( (flag && isless(o2, o1)) ? (o2, o1) : (o1, o2) )

function reformatIntData2((o1, o2, o3, o4)::NTuple{4, T}, flags::NTuple{3, Bool}) where {T}
    l = reformatIntData2((o1, o2), flags[1])
    r = reformatIntData2((o3, o4), flags[2])
    ifelse((flags[3] && isless(r, l)), (r[1], r[2], l[1], l[2]), (l[1], l[2], r[1], r[2]))
end

function reformatIntData1(bf::FGTBasisFuncs1O{T, D, 𝑙, GN}) where {T, D, 𝑙, GN}
    R = (centerCoordOf(bf) |> Tuple)::NTuple{D, T}
    ijk = bf.l[1].tuple
    αds = if bf.normalizeGTO
        N = getNijk(T, ijk...)
        map(bf.gauss) do x
            xpn = x.xpn()::T
            con = x.con()::T
            (xpn, con * N * getNα(ijk..., xpn))
        end
    else
        map(x->(x.xpn()::T, x.con()::T), bf.gauss)
    end
    R, ijk, αds
end

function reformatIntData1((ji,)::Tuple{Bool}, 
                          bfs::NTuple{2, FGTBasisFuncs1O{T, D}}) where {T, D}
    if ji
        data1 = reformatIntData1(bfs[1])
        (data1, data1)
    else
        reformatIntData1.(bfs)
    end
end

function reformatIntData1((lk, lj, kj, kiOrji)::NTuple{4, Bool}, 
                          bfs::NTuple{4, FGTBasisFuncs1O{T, D}}) where {T, D}
    data4 = reformatIntData1(bfs[4])
    data3 = lk ? data4 : reformatIntData1(bfs[3])
    data2 = lj ? data4 : (kj ? data3 : reformatIntData1(bfs[2]))
    data1 = lj ? (kiOrji ? data3 : reformatIntData1(bfs[1])) : 
                 (kiOrji ? data2 : reformatIntData1(bfs[1]))
    (data1, data2, data3, data4)
end

reformatIntData1(::Val{false}, bfs::Tuple{Vararg{FGTBasisFuncs1O{T, D}}}) where {T, D} = 
reformatIntData1.(bfs)


function isIntZeroCore(::Val{1}, 
                       R₁::NTuple{D, T}, R₂::NTuple{D, T}, 
                       ijk₁::NTuple{D, Int}, ijk₂::NTuple{D, Int}) where {D, T}
    for i in eachindex(R₁)
        R₁[i]==R₂[i] && isodd(ijk₁[i] + ijk₂[i]) && (return true)
    end
    false
end

function isIntZeroCore(::Val{2}, 
                       R₁::NTuple{D, T}, R₂::NTuple{D, T}, 
                       R₃::NTuple{D, T}, R₄::NTuple{D, T}, 
                       ijk₁::NTuple{D, Int}, ijk₂::NTuple{D, Int}, 
                       ijk₃::NTuple{D, Int}, ijk₄::NTuple{D, Int}) where {D, T}
    for i in eachindex(R₁)
        R₁[i]==R₂[i]==R₃[i]==R₄[i] && 
        isodd(ijk₁[i] + ijk₂[i] + ijk₃[i] + ijk₄[i]) && (return true)
    end
    false
end

function isIntZeroCore(::Val{:∫nucAttractionCore}, 
                       R₁::NTuple{D, T}, R₂::NTuple{D, T}, 
                       ijk₁::NTuple{D, Int}, ijk₂::NTuple{D, Int}, 
                       R₀::NTuple{D, T}) where {D, T}
    for i in eachindex(R₁)
        R₀[i]==R₁[i]==R₂[i] && isodd(ijk₁[i] + ijk₂[i]) && (return true)
    end
    false
end

isIntZero(::Type{typeof(∫overlapCore)}, R₁, R₂, ijk₁, ijk₂, _) = 
isIntZeroCore(Val(1), R₁, R₂, ijk₁, ijk₂)

isIntZero(::Type{typeof(∫elecKineticCore)}, R₁, R₂, ijk₁, ijk₂, _) = 
isIntZeroCore(Val(1), R₁, R₂, ijk₁, ijk₂)

isIntZero(::Type{typeof(∫nucAttractionCore)}, R₁, R₂, ijk₁, ijk₂, optArgs) = 
isIntZeroCore(Val(:∫nucAttractionCore), R₁, R₂, ijk₁, ijk₂, optArgs[end])

isIntZero(::Type{typeof(∫eeInteractionCore)}, R₁, R₂, R₃, R₄, ijk₁, ijk₂, ijk₃, ijk₄, _) = 
isIntZeroCore(Val(2), R₁, R₂, R₃, R₄, ijk₁, ijk₂, ijk₃, ijk₄)


function getOneBodyInt(∫1e::F, bls::Union{Tuple{Bool}, Val{false}}, 
                       bf1::FGTBasisFuncs1O{T, D, 𝑙1, GN1}, 
                       bf2::FGTBasisFuncs1O{T, D, 𝑙2, GN2}, 
                       optArgs...) where {F<:Function, T, D, 𝑙1, 𝑙2, GN1, GN2}
    (R₁, ijk₁, ps₁), (R₂, ijk₂, ps₂) = reformatIntData1(bls, (bf1, bf2))
    !(𝑙1==𝑙2==0) && isIntZero(F, R₁, R₂, ijk₁, ijk₂, optArgs) && (return T(0.0))
    uniquePairs, uPairCoeffs = get1BodyUniquePairs(R₁==R₂ && ijk₁==ijk₂, ps₁, ps₂)
    map(uniquePairs, uPairCoeffs) do x, y
        ∫1e(optArgs..., R₁, R₂, ijk₁, x[1], ijk₂, x[2])::T * y
    end |> sum
end

function get1BodyUniquePairs(flag::Bool, 
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

@inline function getIntCore11!(n, uniquePairs, uPairCoeffs, flag, ps₁)
    for (i₁, p₁) in enumerate(ps₁), (i₂, p₂) in zip(1:i₁, ps₁)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flag, (p₁, p₂), diFoldCount(i₁, i₂))
    end
    n
end

@inline function getIntCore12!(n, uniquePairs, uPairCoeffs, flag, (ps₁, ps₂))
    for p₁ in ps₁, p₂ in ps₂
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flag, (p₁, p₂))
    end
    n
end

@inline function getUniquePair!(i, uniquePairs, uPairCoeffs, flag, psc, nFold=1)
    pair = reformatIntData2(getindex.(psc, 1), flag)
    idx = findfirst(x->x==pair, uniquePairs)
    con = (getindex.(psc, 2) |> prod) * nFold
    if idx === nothing
        i += 1
        push!(uniquePairs, pair)
        uPairCoeffs[i] = con
    else
        uPairCoeffs[idx] += con
    end
    i
end


function getTwoBodyInt(∫2e::F, bls::Union{NTuple{4, Bool}, Val{false}}, 
                       bf1::FGTBasisFuncs1O{T, D, 𝑙1, GN1}, 
                       bf2::FGTBasisFuncs1O{T, D, 𝑙2, GN2}, 
                       bf3::FGTBasisFuncs1O{T, D, 𝑙3, GN3}, 
                       bf4::FGTBasisFuncs1O{T, D, 𝑙4, GN4}, 
                       optArgs...) where 
                      {F<:Function, T, D, 𝑙1, 𝑙2, 𝑙3, 𝑙4, GN1, GN2, GN3, GN4}
    (R₁, ijk₁, ps₁), (R₂, ijk₂, ps₂), (R₃, ijk₃, ps₃), (R₄, ijk₄, ps₄) = 
    reformatIntData1(bls, (bf1, bf2, bf3, bf4))

    !(𝑙1==𝑙2==𝑙3==𝑙4==0) && isIntZero(F, R₁, R₂, R₃, R₄, ijk₁, ijk₂, ijk₃, ijk₄, optArgs) && 
    (return T(0.0))

    f1 = (R₁ == R₂ && ijk₁ == ijk₂)
    f2 = (R₃ == R₄ && ijk₃ == ijk₄)
    f3 = (R₁ == R₃ && ijk₁ == ijk₃ && R₂ == R₄ && ijk₂ == ijk₄)
    f4 = (R₁ == R₄ && ijk₁ == ijk₄)
    f5 = (R₂ == R₃ && ijk₂ == ijk₃)

    uniquePairs, uPairCoeffs = get2BodyUniquePairs((f1, f2, f3, f4, f5), ps₁, ps₂, ps₃, ps₄)
    map(uniquePairs, uPairCoeffs) do x, y
        ∫2e(optArgs..., R₁,ijk₁,x[1], R₂,ijk₂,x[2], R₃,ijk₃,x[3], R₄,ijk₄,x[4])::T * y
    end |> sum
end

diFoldCount(i::T, j::T) where {T} = ifelse(i==j, 1, 2)

@inline function octaFoldCount(i::T, j::T, k::T, l::T) where {T}
    m = 0
    i != j && (m += 1)
    k != l && (m += 1)
    (i != k || j != l) && (m += 1)
    2^m
end

function get2BodyUniquePairs(flags::NTuple{5, Bool}, 
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

function getIntX1X1X2X2!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂)
    A, B, C = tupleDiff(ps₁, ps₂)
    if length(A) > 0 && flags[3]
        g1111 = ((A,),)
        g1122 = (((A, C),), ((B, A),), ((B, C),))
        g1212 = ()
        g1123 = (((A, A, C),), ((A, C, A),), ((B, A, C),), ((B, C, A),))
        g1233 = (((A, B, A),), ((B, A, A),), ((A, B, C),), ((B, A, C),))
        g1234 = (((A, B, A, C),), ((A, B, C, A),), ((A, B, C, A),), ((B, A, C, A),))

        n = getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                (g1111, g1122, g1212, g1123, g1233, g1234))
    else
        n = getIntCore1122!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂))
    end
    n
end

function getIntX1X2X1X2!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂)
    A, B, C = tupleDiff(ps₁, ps₂)
    if length(A) > 0 && flags[1] && flags[2]
        g1111 = ((A,),)
        g1122 = ()
        g1212 = (((A, C),), ((B, A),), ((B, C),))
        g1123 = (((A, A, C),), ((A, B, A),), ((A, B, C),))
        g1233 = (((A, C, A),), ((B, A, A),), ((B, C, A),))
        g1234 = (((A, C, B, A),), ((A, C, B, C),), ((B, A, B, C),),
                 ((B, A, A, C),), ((B, C, A, C),), ((B, C, B, A),))
        n = getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                (g1111, g1122, g1212, g1123, g1233, g1234))
    else
        n = getIntCore1212!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂))
    end
    n
end

function getIntX1X2X2X1!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂)
    A, B, C = tupleDiff(ps₁, ps₂)
    if length(A) > 0 && all(flags)
        g1111 = ((A,),)
        g1122 = ()
        g1212 = ()
        g1123 = (((A, A, C),), ((A, C, A),), ((A, C, B),))
        g1233 = (((A, C, A),), ((B, A, A),), ((B, C, A),))
        g1234 = (((A, C, A, B),), ((A, C, C, B),), ((B, A, C, A),),
                 ((B, A, C, B),), ((B, C, A, B),), ((B, C, C, A),))
        n = getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                (g1111, g1122, g1212, g1123, g1233, g1234))

        g1221 = ((A, C), (B, A), (B, C))
        for i in g1221
            n = getIntCore1221!(n, uniquePairs, uPairCoeffs, flags, i)
        end
    else
        n = getIntCore1221!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂))
    end
    n
end

function getIntX1X1X2X3!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂, ps₃)
    A, B, C, D = tupleDiff(ps₁, ps₂, ps₃)
    if length(A) > 0 && flags[2] && flags[3]
        g1111 = ((A,),)
        g1122 = (((B, A),),)
        g1212 = ()
        g1123 = (((A, A, D),), ((A, C, A),), ((A, C, D),),
                 ((B, A, D),), ((B, C, A),), ((B, C, D),))
        g1233 = (((A, B, A),), ((B, A, A),))
        g1234 = (((A, B, A, D),), ((A, B, C, A),), ((A, B, C, D),),
                 ((B, A, A, D),), ((B, A, C, A),), ((B, A, C, D),))
        n = getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                (g1111, g1122, g1212, g1123, g1233, g1234))
    else
        n = getIntCore1123!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃))
    end
    n
end

function getIntX1X2X3X3!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂, ps₃)
    A, B, C, D = tupleDiff(ps₁, ps₂, ps₃)
    if length(A) > 0 && flags[1] && flags[3]
        g1111 = ((A,),)
        g1122 = (((A, D),),)
        g1212 = ()
        g1123 = (((A, A, D),), ((A, D, A),))
        g1233 = (((A, C, A),), ((A, C, D),), ((B, A, A),), 
                 ((B, A, D),), ((B, C, A),), ((B, C, D),))
        g1234 = (((A, C, A, D),), ((A, C, D, A),), ((B, A, A, D),),
                 ((B, A, D, A),), ((B, C, A, D),), ((B, C, D, A),))
        n = getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                (g1111, g1122, g1212, g1123, g1233, g1234))
    else
        n = getIntCore1233!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃))
    end
    n
end

function getIntX1X2X3X1!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂, ps₃)
    A, B, C, D = tupleDiff(ps₁, ps₂, ps₃)
    if length(A) > 0 && all(flags)
        g1111 = ((A,),)
        g1122 = ()
        g1212 = ()
        g1123 = (((A, A, B),), ((A, D, A),), ((A, D, B),))
        g1233 = (((A, C, A),), ((B, A, A),), ((B, C, A),))
        g1234 = (((A, C, A, B),), ((A, C, D, A),), ((A, C, D, B),), 
                 ((B, A, D, A),), ((B, A, D, B),), 
                 ((B, C, A, B),), ((B, C, D, A),), ((B, C, D, B),))
        n = getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                (g1111, g1122, g1212, g1123, g1233, g1234))
        n = getIntCore1221!(n, uniquePairs, uPairCoeffs, flags, (B,A))
    else
        n = getIntCore1234!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃, ps₁))
    end
    n
end

function getIntX1X2X2X3!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂, ps₃)
    A, B, C, D = tupleDiff(ps₁, ps₂, ps₃)
    if length(A) > 0 && all(flags)
        g1111 = ((A,),)
        g1122 = ()
        g1212 = ()
        g1123 = (((A, A, D),), ((A, C, A),), ((A, C, D),))
        g1233 = (((A, C, A),), ((B, A, A),), ((B, C, A),))
        g1234 = (((A, C, A, D),), ((A, C, C, D),), 
                 ((B, A, A, D),), ((B, A, C, A),), ((B, A, C, D),), 
                 ((B, C, A, D),), ((B, C, C, A),), ((B, C, C, D),))
        n = getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                (g1111, g1122, g1212, g1123, g1233, g1234))
        n = getIntCore1221!(n, uniquePairs, uPairCoeffs, flags, (A,C))
    else
        n = getIntCore1234!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₂, ps₃))
    end
    n
end

function getIntX1X2X3X4!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂, ps₃, ps₄)
    A, B, C, D, E = tupleDiff(ps₁, ps₂, ps₃, ps₄)
    if length(A) > 0 && all(flags)
        g1111 = ((A,),)
        g1122 = ()
        g1212 = ()
        g1123 = (((A, A, E),), ((A, D, A),), ((A, D, E),))
        g1233 = (((A, C, A),), ((B, A, A),), ((B, C, A),))
        g1234 = (((A, C, A, E),), ((A, C, D, A),), ((A, C, D, E),),
                 ((B, A, A, E),), ((B, A, D, A),), ((B, A, D, E),), 
                 ((B, C, A, E),), ((B, C, D, A),), ((B, C, D, E),))
        n = getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, 
                                (g1111, g1122, g1212, g1123, g1233, g1234))
    else
        n = getIntCore1234!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃, ps₄))
    end
    n
end

function getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, groups)
    for i in groups[1]
        n = getIntCore1111!(n, uniquePairs, uPairCoeffs, flags, i...)
    end
    for i in groups[2]
        n = getIntCore1122!(n, uniquePairs, uPairCoeffs, flags, i...)
    end
    for i in groups[3]
        n = getIntCore1212!(n, uniquePairs, uPairCoeffs, flags, i...)
    end
    for i in groups[4]
        n = getIntCore1123!(n, uniquePairs, uPairCoeffs, flags, i...)
    end
    for i in groups[5]
        n = getIntCore1233!(n, uniquePairs, uPairCoeffs, flags, i...)
    end
    for i in groups[6]
        n = getIntCore1234!(n, uniquePairs, uPairCoeffs, flags, i...)
    end
    n
end

@inline function getIntCore1111!(n, uniquePairs, uPairCoeffs, flags, ps₁)
    for (i₁, p₁) in enumerate(ps₁), (i₂, p₂) in zip(1:i₁, ps₁), 
        (i₃, p₃) in zip(1:i₁, ps₁), (i₄, p₄) in zip(1:ifelse(i₃==i₁, i₂, i₃), ps₁)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁,p₂,p₃,p₄), 
                           octaFoldCount(i₁,i₂,i₃,i₄))
    end
    n
end

@inline function getIntCore1122!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂))
    for (i₁, p₁) in enumerate(ps₁), (i₂, p₂) in zip(1:i₁, ps₁), 
        (i₃, p₃) in enumerate(ps₂), (i₄, p₄) in zip(1:i₃, ps₂)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁, p₂, p₃, p₄), 
                           diFoldCount(i₁, i₂)*diFoldCount(i₃, i₄))
    end
    n
end

@inline function getIntCore1212!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂))
    oneSidePairs = Iterators.product(eachindex(ps₁), eachindex(ps₂))
    for (x, (i₁,i₂)) in enumerate(oneSidePairs), (_, (i₃,i₄)) in zip(1:x, oneSidePairs)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, 
                           (ps₁[i₁], ps₂[i₂], ps₁[i₃], ps₂[i₄]), 2^(i₁!=i₃ || i₂!=i₄))
    end
    n
end

@inline function getIntCore1221!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂))
    oneSidePairs = Iterators.product(eachindex(ps₁), eachindex(ps₂))
    for (x, (i₁,i₂)) in enumerate(oneSidePairs), (_, (i₃,i₄)) in zip(1:x, oneSidePairs)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, 
                           (ps₁[i₁], ps₂[i₂], ps₂[i₄], ps₁[i₃]), 2^(i₁!=i₃ || i₂!=i₄))
    end
    n
end

@inline function getIntCore1123!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃))
    for (i₁, p₁) in enumerate(ps₁), (i₂, p₂) in zip(1:i₁, ps₁), p₃ in ps₂, p₄ in ps₃
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁, p₂, p₃, p₄), 
                           diFoldCount(i₁, i₂))
    end
    n
end

@inline function getIntCore1233!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃))
    for p₁ in ps₁, p₂ in ps₂, (i₃, p₃) in enumerate(ps₃), (i₄, p₄) in zip(1:i₃, ps₃)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁, p₂, p₃, p₄), 
                           diFoldCount(i₃, i₄))
    end
    n
end

@inline function getIntCore1234!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃, ps₄))
    for p₁ in ps₁, p₂ in ps₂, p₃ in ps₃, p₄ in ps₄
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁, p₂, p₃, p₄))
    end
    n
end


getCompositeInt(∫::F, bls::Union{Tuple{Bool}, Val{false}}, 
                bfs::NTuple{2, FGTBasisFuncs1O{T, D}}, 
                optArgs...) where {F<:Function, T, D} = 
getOneBodyInt(∫, bls, bfs..., optArgs...)

getCompositeInt(∫::F, bls::Union{NTuple{4, Bool}, Val{false}}, 
                bfs::NTuple{4, FGTBasisFuncs1O{T, D}}, 
                optArgs...) where {F<:Function, T, D} = 
getTwoBodyInt(∫, bls, bfs..., optArgs...)

function getCompositeInt(::typeof(∫nucAttractionCore), bls::Union{Tuple{Bool}, Val{false}}, 
                         bfs::NTuple{2, FGTBasisFuncs1O{T, D}}, 
                         nuc::NTuple{NN, String}, 
                         nucCoords::NTuple{NN, NTuple{D, T}}) where {T, D, NN}
    res = T(0.0)
    for (ele, coord) in zip(nuc, nucCoords)
        res += getOneBodyInt(∫nucAttractionCore, bls, bfs..., getCharge(ele), coord|>Tuple)
    end
    res
end
                          #       j==i      j!=i
const Int1eBIndexLabels = Dict([( true,), (false,)] .=> [Val(:aa), Val(:ab)])

getON(::Val{:ContainBasisFuncs}, b::SpatialBasis) = orbitalNumOf(b)
getON(::Val{:WithoutBasisFuncs}, ::CGTBasisFuncs1O{<:Any, <:Any, BN}) where {BN} = BN

getBF(::Val, b::SpatialBasis, i) = getindex(b, i)
getBF(::Val{:WithoutBasisFuncs}, b::BasisFuncMix, i) = getindex(b.BasisFunc, i)

getBFs(::Val{:ContainBasisFuncs}, b::SpatialBasis) = itself(b)
getBFs(::Val{:WithoutBasisFuncs}, b::CGTBasisFuncs1O) = unpackBasis(b)

# 1e integrals for BasisFuncs/BasisFuncMix-mixed bases
function getCompositeIntCore(::Val{T}, ::Val{D}, ::Val{BL}, ::Val{:aa}, ∫::F, 
                             bs::NTuple{2, BT}, optArgs...) where 
                            {T, D, BL, F<:Function, BT<:SpatialBasis{T, D}}
    a = bs[begin]
    ON = getON(Val(BL), a)
    res = zeros(T, ON, ON)
    for j=1:ON, i=1:j
        res[i,j] = res[j,i] = getCompositeInt(∫, (j==i,), 
                                              (getBF(Val(BL), a, i), 
                                               getBF(Val(BL), a, j)), optArgs...)
    end
    res
end

function getCompositeIntCore(::Val{T}, ::Val{D}, ::Val{BL}, ::Val{:ab}, ∫::F, 
                             bs::NTuple{2, SpatialBasis{T, D}}, optArgs...) where 
                            {T, D, BL, F<:Function}
    bfsI, bfsJ = bs = getBFs.(Val(BL), bs)
    ON1, ON2 = length.(bs)
    res = zeros(T, ON1, ON2)
    for (j, bfj) in enumerate(bfsJ), (i, bfi) in enumerate(bfsI)
        res[i,j] = getCompositeInt(∫, Val(false), (bfi, bfj), optArgs...)
    end
    res
end

                          # a==b    lk,    lj,    kj, ki/ji    ijkl
const Int2eBIndexLabels = Dict([( true,  true,  true,  true), #1111
                                ( true, false, false,  true), #1122
                                (false,  true, false,  true), #1212
                                (false,  true, false, false), #1232
                                (false, false,  true, false), #122X
                                (false, false,  true,  true), #1112
                                (false, false, false,  true), #1123
                                ( true,  true,  true, false), #1222
                                ( true, false, false, false), #12XX
                                (false, false, false, false)  #123X
                               ] .=> 
                               [Val(:aaaa), Val(:aabb), Val(:abab), Val(:abcx), Val(:abbx), 
                                Val(:aabc), Val(:aabc), Val(:abxx), Val(:abxx), Val(:abcx)])

# 2e integrals for BasisFuncs/BasisFuncMix-mixed bases
function getCompositeIntCore(::Val{T}, ::Val{D}, ::Val{BL}, ::Val{:aaaa}, ∫::F, 
                             bs::NTuple{4, BT}, optArgs...) where 
                            {T, D, BL, F<:Function, BT<:SpatialBasis{T, D}}
    a = bs[begin]
    ON = getON(Val(BL), a)
    res = zeros(T, ON, ON, ON, ON)
    for l = 1:ON, k = 1:l, j = 1:l, i = 1:ifelse(l==j, k, j)
        bl = (l==k, l==j, k==j, ifelse(l==j, k, j)==i)
        res[i, j, k, l] = res[j, i, k, l] = res[j, i, l, k] = res[i, j, l, k] = 
        res[l, k, i, j] = res[k, l, i, j] = res[k, l, j, i] = res[l, k, j, i] = 
        getCompositeInt(∫, bl, (getBF(Val(BL), a, i), getBF(Val(BL), a, j), 
                                getBF(Val(BL), a, k), getBF(Val(BL), a, l)), optArgs...)
    end
    res
end

function getCompositeIntCore(::Val{T}, ::Val{D}, ::Val{BL}, ::Val{:aabb}, ∫::F, 
                             bs::Tuple{BT1, BT1, BT2, BT2}, optArgs...) where 
                            {T, D, BL, F<:Function, BT1<:SpatialBasis{T, D}, 
                                                    BT2<:SpatialBasis{T, D}}
    a, b = ab = bs[[1, 3]]
    ON1, ON2 = getON.(Val(BL), ab)
    res = zeros(T, ON1, ON1, ON2, ON2)
    for l = 1:ON2, k = 1:l, j = 1:ON1, i = 1:j
        bl = (l==k, false, false, j==i)
        res[i, j, k, l] = res[j, i, k, l] = res[j, i, l, k] = res[i, j, l, k] = 
        getCompositeInt(∫, bl, (getBF(Val(BL), a, i), getBF(Val(BL), a, j), 
                                getBF(Val(BL), b, k), getBF(Val(BL), b, l)), optArgs...)
    end
    res
end

function getCompositeIntCore(::Val{T}, ::Val{D}, ::Val{BL}, ::Val{:abab}, ∫::F, 
                             bs::Tuple{BT1, BT2, BT1, BT2}, optArgs...) where 
                            {T, D, BL, F<:Function, BT1<:SpatialBasis{T, D}, 
                                                    BT2<:SpatialBasis{T, D}}
    a, b = ab = bs[[1, 2]]
    ON1, ON2 = getON.(Val(BL), ab)
    res = zeros(T, ON1, ON2, ON1, ON2)
    rng = Iterators.product(1:ON2, 1:ON1)
    for (x, (l,k)) in enumerate(rng), (_, (j,i)) in zip(1:x, rng)
        bl = (false, l==j, false, ifelse(l==j, k==i, false))
        res[i, j, k, l] = res[k, l, i, j] = 
        getCompositeInt(∫, bl, (getBF(Val(BL), a, i), getBF(Val(BL), b, j), 
                                getBF(Val(BL), a, k), getBF(Val(BL), b, l)), optArgs...)
    end
    res
end

function getCompositeIntCore(::Val{T}, ::Val{D}, ::Val{BL}, ::Val{:abbx}, ∫::F, 
                             bs::Tuple{BT1, BT2, BT2, BT3}, optArgs...) where 
                            {T, D, BL, F<:Function, BT1<:SpatialBasis{T, D}, 
                                                    BT2<:SpatialBasis{T, D}, 
                                                    BT3<:SpatialBasis{T, D}}
    if bs[1] === bs[4]
        a, b = ab = bs[[1, 2]]
        ON1, ON2 = getON.(Val(BL), ab)
        res = zeros(T, ON1, ON2, ON2, ON1)
        rng = Iterators.product(1:ON1, 1:ON2)
        for (x, (l,k)) in enumerate(rng), (_, (i,j)) in zip(1:x, rng)
            bl = (false, false, k==j, false)
            res[i, j, k, l] = res[k, l, i, j] = 
            getCompositeInt(∫, bl, (getBF(Val(BL), a, i), getBF(Val(BL), b, j), 
                                    getBF(Val(BL), b, k), getBF(Val(BL), a, l)), optArgs...)
        end
    else
        res = getCompositeIntCore(Val(T), Val(D), Val(BL), Val(:abcx), ∫, bs, optArgs...)
    end
    res
end

function getCompositeIntCore(::Val{T}, ::Val{D}, ::Val{BL}, ::Val{:aabc}, ∫::F, 
                             bs::Tuple{BT1, BT1, BT2, BT3}, optArgs...) where 
                            {T, D, BL, F<:Function, BT1<:SpatialBasis{T, D}, 
                                                    BT2<:SpatialBasis{T, D}, 
                                                    BT3<:SpatialBasis{T, D}}
    a, b, c = abc = bs[[1, 3, 4]]
    ON1, ON2, ON3 = getON.(Val(BL), abc)
    res = zeros(T, ON1, ON1, ON2, ON3)
    for l=1:ON3, k=1:ON2, j=1:ON1, i=1:j
        bl = (false, false, false, j==i)
        res[i, j, k, l] = res[j, i, k, l] = 
        getCompositeInt(∫, bl, (getBF(Val(BL), a, i), getBF(Val(BL), a, j), 
                                getBF(Val(BL), b, k), getBF(Val(BL), c, l)), optArgs...)
    end
    res
end

function getCompositeIntCore(::Val{T}, ::Val{D}, ::Val{BL}, ::Val{:abxx}, ∫::F, 
                             bs::Tuple{BT1, BT2, BT3, BT3}, optArgs...) where 
                            {T, D, BL, F<:Function, BT1<:SpatialBasis{T, D}, 
                                                    BT2<:SpatialBasis{T, D}, 
                                                    BT3<:SpatialBasis{T, D}}
    a, b, x = abx = bs[[1, 2, 3]]
    ON1, ON2, ON3 = getON.(Val(BL), abx)
    res = zeros(T, ON1, ON2, ON3, ON3)
    for l=1:ON3, k=1:l, j=1:ON2, i=1:ON1
        bl = (l==k, false, false, false)
        res[i, j, k, l] = res[i, j, l, k] = 
        getCompositeInt(∫, bl, (getBF(Val(BL), a, i), getBF(Val(BL), b, j), 
                                getBF(Val(BL), x, k), getBF(Val(BL), x, l)), optArgs...)
    end
    res
end

function getCompositeIntCore(::Val{T}, ::Val{D}, ::Val{BL}, ::Val{:abcx}, ∫::F, 
                             bs::NTuple{4, SpatialBasis{T, D}}, optArgs...) where 
                            {T, D, BL, F<:Function}
    bfsI, bfsJ, bfsK, bfsL = bs = getBFs.(Val(BL), bs)
    ON1, ON2, ON3, ON4 = length.(bs)
    res = zeros(T, ON1, ON2, ON3, ON4)
    for (l, bfl) in enumerate(bfsL), (k, bfk) in enumerate(bfsK), 
        (j, bfj) in enumerate(bfsJ), (i, bfi) in enumerate(bfsI)
        res[i,j,k,l] = getCompositeInt(∫, Val(false), (bfi, bfj, bfk, bfl), optArgs...)
    end
    res
end

getBasisIndexL(::Val{2}, bls::Tuple{Bool}) = Int1eBIndexLabels[bls]
getBasisIndexL(::Val{4}, bls::NTuple{4, Bool}) = Int2eBIndexLabels[bls]
getBasisIndexL(::Val{2}, ::Val{false}) = Int1eBIndexLabels[(false,)]
getBasisIndexL(::Val{4}, ::Val{false}) = Int2eBIndexLabels[(false, false, false, false)]

getCompositeInt(∫::F, bls::Union{Tuple{Vararg{Bool}}, Val{false}}, 
                bs::NTuple{BN, SpatialBasis{T, D}}, 
                optArgs...) where {F<:Function, BN, T, D} = 
getCompositeIntCore(Val(T), Val(D), Val(:ContainBasisFuncs), 
                    getBasisIndexL(Val(BN), bls), ∫, bs, optArgs...)

function getCompositeInt(∫::F, bls::Union{Tuple{Vararg{Bool}}, Val{false}}, 
                         bs::NTuple{BN, SpatialBasis{T, D, 1}}, 
                         optArgs...) where {F<:Function, BN, T, D}
    if any(fieldtypes(typeof(bs)) .<: EmptyBasisFunc)
        zero(T)
    else
        getCompositeIntCore(Val(T), Val(D), Val(:WithoutBasisFuncs), 
                            getBasisIndexL(Val(BN), bls), ∫, bs, optArgs...) |> sum
    end
end


function update2DarrBlock!(arr::AbstractMatrix{T1}, block::T1, 
                           I::T2, J::T2) where {T1, T2<:UnitRange{Int}}
    arr[I, J] .= block
    arr[J, I] .= block
    nothing
end

function update2DarrBlock!(arr::AbstractMatrix{T1}, block::AbstractMatrix{T1}, 
                           I::T2, J::T2) where {T1, T2<:UnitRange{Int}}
    arr[I, J] = block
    arr[J, I] = block |> transpose
    nothing
end

function getOneBodyInts(∫1e::F, basisSet::NTuple{BN, GTBasisFuncs{T, D}}, optArgs...) where 
                       {F<:Function, BN, T, D}
    subSize = orbitalNumOf.(basisSet) |> collect
    accuSize = vcat(0, accumulate(+, subSize))
    len = subSize |> sum
    buf = Array{T}(undef, len, len)
    for j = 1:BN, i = 1:j
        int = getCompositeInt(∫1e, (j==i,), (basisSet[i], basisSet[j]), optArgs...)
        rowRange = accuSize[i]+1 : accuSize[i+1]
        colRange = accuSize[j]+1 : accuSize[j+1]
        update2DarrBlock!(buf, int, rowRange, colRange)
    end
    buf
end

function getOneBodyInts(∫1e::F, basisSet::NTuple{BN, GTBasisFuncs{T, D, 1}}, 
                        optArgs...) where {F<:Function, BN, T, D}
    buf = Array{T}(undef, BN, BN)
    for j = 1:BN, i = 1:j
        int = getCompositeInt(∫1e, (j==i,), (basisSet[i], basisSet[j]), optArgs...)
        buf[i, j] = buf[j, i] = int
    end
    buf
end


permuteArray(arr::AbstractArray{T, N}, order) where {T, N} = PermutedDimsArray(arr, order)
permuteArray(arr::Number, _) = itself(arr)

function update4DarrBlock!(arr::Array{T1, 4}, block::T1, I::T2, J::T2, K::T2, L::T2) where 
                          {T1, T2<:UnitRange{Int}}
    arr[I, J, K, L] .= block
    arr[J, I, K, L] .= block
    arr[J, I, L, K] .= block
    arr[I, J, L, K] .= block
    arr[L, K, I, J] .= block
    arr[K, L, I, J] .= block
    arr[K, L, J, I] .= block
    arr[L, K, J, I] .= block
    nothing
end

function update4DarrBlock!(arr::Array{T1, 4}, block::Array{T1, 4}, 
                           I::T2, J::T2, K::T2, L::T2) where {T1, T2<:UnitRange{Int}}
    arr[I, J, K, L] .= block
    arr[J, I, K, L] = permuteArray(block, (2,1,3,4))
    arr[J, I, L, K] = permuteArray(block, (2,1,4,3))
    arr[I, J, L, K] = permuteArray(block, (1,2,4,3))
    arr[L, K, I, J] = permuteArray(block, (4,3,1,2))
    arr[K, L, I, J] = permuteArray(block, (3,4,1,2))
    arr[K, L, J, I] = permuteArray(block, (3,4,2,1))
    arr[L, K, J, I] = permuteArray(block, (4,3,2,1))
    nothing
end

function getTwoBodyInts(∫2e::F, basisSet::NTuple{BN, GTBasisFuncs{T, D}}) where 
                       {F<:Function, BN, T, D}
    subSize = orbitalNumOf.(basisSet) |> collect
    accuSize = vcat(0, accumulate(+, subSize))
    totalSize = subSize |> sum
    buf = Array{T}(undef, totalSize, totalSize, totalSize, totalSize)
    for l = 1:BN, k = 1:l, j = 1:l, i = 1:ifelse(l==j, k, j)
        I = accuSize[i]+1 : accuSize[i+1]
        J = accuSize[j]+1 : accuSize[j+1]
        K = accuSize[k]+1 : accuSize[k+1]
        L = accuSize[l]+1 : accuSize[l+1]
        bl = (l==k, l==j, k==j, ifelse(l==j, k, j)==i)
        int = getCompositeInt(∫2e, bl, (basisSet[i], basisSet[j], basisSet[k], basisSet[l]))
        update4DarrBlock!(buf, int, I, J, K, L)
    end
    buf
end

function getTwoBodyInts(∫2e::F, basisSet::NTuple{BN, GTBasisFuncs{T, D, 1}}) where 
                       {F<:Function, BN, T, D}
    buf = Array{T}(undef, BN, BN, BN, BN)
    for l = 1:BN, k = 1:l, j = 1:l, i = 1:ifelse(l==j, k, j)
        bl = (l==k, l==j, k==j, ifelse(l==j, k, j)==i)
        int = getCompositeInt(∫2e, bl, (basisSet[i], basisSet[j], basisSet[k], basisSet[l]))
        buf[i, j, k, l] = buf[j, i, k, l] = buf[j, i, l, k] = buf[i, j, l, k] = 
        buf[l, k, i, j] = buf[k, l, i, j] = buf[k, l, j, i] = buf[l, k, j, i] = int
    end
    buf
end


"""

    eeIuniqueIndicesOf(basisSetSize::Int) -> Vector{Vector{Int}}

Return the unique matrix element indices (in the chemists' notation) of electron-electron 
interactions given the size of a basis set.
"""
function eeIuniqueIndicesOf(basisSetSize::Int)
    uniqueIdx = fill(Int[0,0,0,0], (3*binomial(basisSetSize, 4) + 
                                    6*binomial(basisSetSize, 3) + 
                                    4*binomial(basisSetSize, 2) + basisSetSize))
    index = 1
    for i = 1:basisSetSize, j = 1:i, k = 1:i, l = 1:ifelse(k==i, j, k)
        uniqueIdx[index] = [i, j, k, l]
        index += 1
    end
    uniqueIdx
end