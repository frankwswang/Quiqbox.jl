using QuadGK: quadgk
using SpecialFunctions: erf
using LinearAlgebra: dot

# Reference: DOI: 10.1088/0143-0807/31/1/004
FγCore1(γ::Int, u::Float64, rtol=1e-8) = quadgk(t -> t^(2γ)*exp(-u*t^2), 0, 1; rtol)[1]

myFactorial(a) = (a > 20 ? big(a) : a) |> factorial

function FγCore2(γ::Int, u::Float64)
    t = exp(-u) * sum(myFactorial(γ-k)/(4^k * myFactorial(2*(γ-k)) * u^(k+1)) for k=0:(γ-1))
    myFactorial(2γ) / (2 * myFactorial(γ)) * (√π * erf(√u) / (4^γ * u^(γ + 0.5)) - t)
end

const FγSolverThreshold =  [ 1.0e-6,  0.0021,  0.0051,  0.0221,  0.0603,  0.1280,  0.2143, 
                             0.3389,  0.4755,  0.6445, 15.9803, 17.5235, 17.2490, 18.6754, 
                            20.6822, 20.4641, 21.5754, 21.6161, 22.5990, 23.2793, 24.9931, 
                            24.9213, 27.1843, 27.0873]

function Fγ(γ::Int, u::Float64)
    if u < 1e-8
        1.0 / (2γ + 1)
    elseif γ == 0
        factorial(2γ) / (2 * factorial(γ)) * (√π * erf(√u) / (4^γ * u^(γ + 0.5)))
    else
        u < FγSolverThreshold[γ] ? FγCore1(γ, u) : FγCore2(γ, u)
    end
end

function F₀toFγ(γ::Int, u::Float64, Fγu::Float64)
    res = Array{Float64}(undef, γ+1)
    res[end] = Fγu
    for i in γ:-1:1
        res[i] = (2u*res[i+1] + exp(-u)) / (2(i-1) + 1)
    end
    res
end

F₀toFγ(γ::Int, u::Float64) = F₀toFγ(γ, u, Fγ(γ, u))


@inline function genIntOverlapCore(Δx::Float64, 
                                   i₁::Int, α₁::Float64, 
                                   i₂::Int, α₂::Float64)
    res = 0.0
    for l₁ in 0:(i₁÷2), l₂ in 0:(i₂÷2)
        Ω = i₁ + i₂ - 2*(l₁ + l₂)
        oRange = 0:(Ω÷2)
        Δx == 0.0 && (iseven(Ω) ? (oRange = Ω÷2) : continue)
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

∫overlapCore(R₁::NTuple{3, Float64}, R₂::NTuple{3, Float64}, 
             ijk₁::NTuple{3, Int}, α₁::Float64, ijk₂::NTuple{3, Int}, α₂::Float64) = 
∫overlapCore(R₁.-R₂, ijk₁, α₁, ijk₂, α₂)


function ∫overlapCore(ΔR::NTuple{3, Float64}, 
                      ijk₁::NTuple{3, Int}, α₁::Float64, 
                      ijk₂::NTuple{3, Int}, α₂::Float64)
    for n in (ijk₁..., ijk₂...)
        n < 0 && return 0.0
    end

    α = α₁ + α₂
    res = (π/α)^1.5 * exp(-α₁ * α₂ / α * sum(abs2, ΔR))

        for (i₁, i₂, ΔRᵢ) in zip(ijk₁, ijk₂, ΔR)
            res *= (-1.0)^(i₁) * factorial(i₁) * factorial(i₂) / α^(i₁+i₂) * 
                   genIntOverlapCore(ΔRᵢ, i₁, α₁, i₂, α₂)
        end

    res
end

function ∫elecKineticCore(R₁::NTuple{3, Float64}, R₂::NTuple{3, Float64}, 
                          ijk₁::NTuple{3, Int}, α₁::Float64,
                          ijk₂::NTuple{3, Int}, α₂::Float64)
    ΔR = R₁ .- R₂
    shifts = ((2,0,0), (0,2,0), (0,0,2))
    0.5 * (α₂ * (4*sum(ijk₂) + 6) * ∫overlapCore(ΔR, ijk₁, α₁, ijk₂, α₂) - 4.0 * α₂^2 * 
           sum(∫overlapCore.(Ref(ΔR), Ref(ijk₁), α₁, map.(+, Ref(ijk₂), shifts), α₂)) - 
           sum(ijk₂ .* (ijk₂.-1) .* 
               ∫overlapCore.(Ref(ΔR), Ref(ijk₁), α₁, map.(-, Ref(ijk₂), shifts), α₂)))
end

@inline function genIntTerm1(Δx::Float64, 
                             l₁::Int, o₁::Int, 
                             l₂::Int, o₂::Int, 
                             i₁::Int, α₁::Float64, 
                             i₂::Int, α₂::Float64)
    @inline (r) -> 
        (-1)^(o₂+r) * factorial(o₁+o₂) * α₁^(o₂-l₁-r) * α₂^(o₁-l₂-r) * Δx^(o₁+o₂-2r) / 
        (
            4^(l₁+l₂+r) * 
            factorial(l₁) * factorial(l₂) * factorial(o₁) * factorial(o₂) * 
            factorial(r) * factorial(i₁-2l₁-o₁) * factorial(i₂-2l₂-o₂) * 
            factorial(o₁+o₂-2r)
        )
end

@inline function genIntTerm2(Δx::Float64, 
                             α::Float64, 
                             o₁::Int, 
                             o₂::Int, 
                             μ::Int, 
                             r::Int)
    @inline (u) ->
        (-1)^u * factorial(μ) * Δx^(μ-2u) / 
        (4^u * factorial(u) * factorial(μ-2u) * α^(o₁+o₂-r+u))
end

@inline function genIntNucAttCore1(ΔRR₀::NTuple{3, Float64}, ΔR₁R₂::NTuple{3, Float64}, 
                                   β::Float64, 
                                   ijk₁::NTuple{3, Int}, α₁::Float64, 
                                   ijk₂::NTuple{3, Int}, α₂::Float64)
    A = 0.0
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
                tmp = 0.0
                core2s = genIntTerm2.(ΔRR₀, α₁+α₂, opq₁, opq₂, μv, rst)

                for u in 0:(μˣ÷2), v in 0:(μʸ÷2), w in 0:(μᶻ÷2)
                    γ = μsum - u - v - w
                    tmp += (((u, v, w) .|> core2s)::NTuple{3, Float64} |> prod) * 2Fγs[γ+1]
                end

                A += ((rst .|> core1s)::NTuple{3, Float64} |> prod) * tmp

            end
        end

    end
    A
end

function ∫nucAttractionCore(Z₀::Int, R₀::NTuple{3, Real}, 
                            R₁::NTuple{3, Float64}, R₂::NTuple{3, Float64}, 
                            ijk₁::NTuple{3, Int}, α₁::Float64,
                            ijk₂::NTuple{3, Int}, α₂::Float64)
    if α₁ == α₂
        α = 2α₁
        R = @. 0.5 * (R₁ + R₂)
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

function ∫eeInteractionCore1234(ΔRl::NTuple{3, Float64}, ΔRr::NTuple{3, Float64}, 
                                ΔRc::NTuple{3, Float64}, β::Float64, η::Float64, 
                                ijk₁::NTuple{3, Int}, α₁::Float64, 
                                ijk₂::NTuple{3, Int}, α₂::Float64, 
                                ijk₃::NTuple{3, Int}, α₃::Float64, 
                                ijk₄::NTuple{3, Int}, α₄::Float64)
    A = 0.0
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
                tmp = 0.0

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

function ∫eeInteractionCore(R₁::NTuple{3, Float64}, ijk₁::NTuple{3, Int}, α₁::Float64, 
                            R₂::NTuple{3, Float64}, ijk₂::NTuple{3, Int}, α₂::Float64,
                            R₃::NTuple{3, Float64}, ijk₃::NTuple{3, Int}, α₃::Float64, 
                            R₄::NTuple{3, Float64}, ijk₄::NTuple{3, Int}, α₄::Float64)
    ΔRl = R₁ .- R₂
    ΔRr = R₃ .- R₄
    αl = α₁ + α₂
    αr = α₃ + α₄
    ηl = α₁ * α₂ / αl
    ηr = α₃ * α₄ / αr
    ΔRc = @. (α₁*R₁ + α₂*R₂)/αl - (α₃*R₃ + α₄*R₄)/αr
    η = αl * αr / (α₁ + α₂ + α₃ + α₄)
    β = η * sum(abs2, ΔRc)
    res = π^2.5 / (αl * αr * (αl + αr)^0.5) * 
          exp(-ηl * sum(abs2, ΔRl)) * exp(-ηr * sum(abs2, ΔRr))
        res *= (@. (-1.0)^(ijk₁ + ijk₂) * 
                   factorial(ijk₁) * factorial(ijk₂) * factorial(ijk₃) * factorial(ijk₄) / 
                   αl^(ijk₁+ijk₂) / αr^(ijk₃+ijk₄)) |> prod
        J = ∫eeInteractionCore1234(ΔRl, ΔRr, ΔRc, β, η, 
                                   ijk₁, α₁, ijk₂, α₂, ijk₃, α₃, ijk₄, α₄)
    res * J
end

function reformatIntData2((o1, o2)::NTuple{2, T}, flag::Bool) where {T}
    ( (flag && isless(o2, o1)) ? (o2, o1) : (o1, o2) )::NTuple{2, T}
end

function reformatIntData2((o1, o2, o3, o4)::NTuple{4, T}, flag::NTuple{3, Bool}) where {T}
    p1 = (flag[1] && isless(o2, o1)) ? (o2, o1) : (o1, o2)
    p2 = (flag[2] && isless(o4, o3)) ? (o4, o3) : (o3, o4)
    ((flag[3] && isless(p2, p1)) ? (p2..., p1...) : (p1..., p2...) )::NTuple{4, T}
end

function reformatIntData1(bf::FloatingGTBasisFuncs{<:Any, GN, 1}) where {GN}
    R = (centerCoordOf(bf) |> Tuple)::NTuple{3, Float64}
    ijk = bf.ijk[1].tuple
    αds = if bf.normalizeGTO
        N = getNijk(ijk...)
        map(x->(x.xpn()::Float64, x.con() * N * getNα(ijk..., x.xpn())::Float64), 
            bf.gauss::NTuple{GN, GaussFunc})
    else
        map(x->(x.xpn()::Float64, x.con()::Float64), bf.gauss::NTuple{GN, GaussFunc})
    end
    R, ijk, αds
end


function getOneBodyInt(::FunctionType{F}, 
                       bf1::BasisFunc{<:Any, GN1}, bf2::BasisFunc{<:Any, GN2}, 
                       optArgs...) where {F, GN1, GN2}
    (R₁, ijk₁, ps₁), (R₂, ijk₂, ps₂) = reformatIntData1.((bf1, bf2))
    uniquePairs, uPairCoeffs = getOneBodyIntCore(R₁==R₂ && ijk₁==ijk₂, ps₁, ps₂)
    map(uniquePairs, uPairCoeffs) do x, y
        getfield(Quiqbox, F)(optArgs..., R₁, R₂, ijk₁, x[1], ijk₂, x[2])::Float64 * y
    end |> sum
end

@inline function getOneBodyIntCore(flag::Bool, 
                                   ps₁::NTuple{GN1, NTuple{2, Float64}}, 
                                   ps₂::NTuple{GN2, NTuple{2, Float64}}) where {GN1, GN2}
    uniquePairs = NTuple{2, Float64}[]
    sizehint!(uniquePairs, GN1*GN2)
    uPairCoeffs = Array{Float64}(undef, GN1*GN2)
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

function getTwoBodyInt(::FunctionType{F}, 
                       bf1::BasisFunc{<:Any, GN1}, bf2::BasisFunc{<:Any, GN2}, 
                       bf3::BasisFunc{<:Any, GN3}, bf4::BasisFunc{<:Any, GN4}, 
                       optArgs...) where {F, GN1, GN2, GN3, GN4}
    (R₁, ijk₁, ps₁), (R₂, ijk₂, ps₂), (R₃, ijk₃, ps₃), (R₄, ijk₄, ps₄) = 
    reformatIntData1.((bf1, bf2, bf3, bf4))

    f1 = (R₁ == R₂ && ijk₁ == ijk₂)
    f2 = (R₃ == R₄ && ijk₃ == ijk₄)
    f3 = (R₁ == R₃ && ijk₁ == ijk₃ && R₂ == R₄ && ijk₂ == ijk₄)
    f4 = (R₁ == R₄ && ijk₁ == ijk₄ && R₂ == R₃ && ijk₂ == ijk₃)

    uniquePairs, uPairCoeffs = getTwoBodyIntCore((f1, f2, f3, f4), ps₁, ps₂, ps₃, ps₄)
    map(uniquePairs, uPairCoeffs) do x, y
        getfield(Quiqbox, F)(optArgs..., R₁, ijk₁, x[1], R₂, ijk₂, x[2], 
                                         R₃, ijk₃, x[3], R₄, ijk₄, x[4])::Float64 * y
    end |> sum
end

@inline function diFoldCount(i::T, j::T) where {T<:Real}
    i==j ? 1 : 2
end

@inline function octaFoldCount(i::T, j::T, k::T, l::T) where {T<:Real}
    m = 0
    i != j && (m += 1)
    k != l && (m += 1)
    (i != k || j != l) && (m += 1)
    2^m
end

@inline function getTwoBodyIntCore(flags::NTuple{4, Bool}, 
                                   ps₁::NTuple{GN1, NTuple{2, Float64}},
                                   ps₂::NTuple{GN2, NTuple{2, Float64}},
                                   ps₃::NTuple{GN3, NTuple{2, Float64}},
                                   ps₄::NTuple{GN4, NTuple{2, Float64}}) where 
                                  {GN1, GN2, GN3, GN4}
    uniquePairs = NTuple{4, Float64}[]
    sizehint!(uniquePairs, GN1*GN2*GN3*GN4)
    uPairCoeffs = Array{Float64}(undef, GN1*GN2*GN3*GN4)
    flagRijk = flags[1:3]
    i = 0
    if ps₁ == ps₂ && flags[1]
        if ps₃ == ps₄ && flags[2]
            if ps₃ == ps₁ && flags[3]
                i = getIntCore1111!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁)
            else
                i = getIntX1X1X2X2!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₃)
            end
        else
            i = getIntX1X1X2X3!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₃, ps₄)
        end
    elseif ps₃ == ps₄ && flags[2]
        i = getIntX1X2X3X3!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₂, ps₃)
    elseif ps₁ == ps₃ && ps₂ == ps₄ && flags[3]
        i = getIntX1X2X1X2!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₂)
    elseif ps₁ == ps₄ && ps₂ == ps₃ && flags[4]
        i = getIntX1X2X2X1!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₂)
    else
        i = getIntX1X2X3X4!(i, uniquePairs, uPairCoeffs, flagRijk, ps₁, ps₂, ps₃, ps₄)
    end
    uniquePairs, uPairCoeffs
end

@inline function getIntX1X1X2X2!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂)
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

@inline function getIntX1X2X1X2!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂)
    A, B, C = tupleDiff(ps₁, ps₂)
    if length(A) > 0 && all(flags)
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

@inline getIntX1X2X2X1!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂) = 
        getIntX1X2X1X2!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂)

@inline function getIntX1X1X2X3!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂, ps₃)
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

@inline function getIntX1X2X3X3!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂, ps₃)
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

@inline function getIntX1X2X3X4!(n, uniquePairs, uPairCoeffs, flags, ps₁, ps₂, ps₃, ps₄)
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

@inline function getIntXAXBXCXDcore!(n, uniquePairs, uPairCoeffs, flags, groups)
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

@inline function getIntCore1111!(n, uniquePairs, uPairCoeffs, flags, ps₁, nFold=1)
    for (i₁, p₁) in enumerate(ps₁), (i₂, p₂) in zip(1:i₁, ps₁), 
        (i₃, p₃) in zip(1:i₁, ps₁), (i₄, p₄) in zip(1:(i₃==i₁ ? i₂ : i₃), ps₁)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁,p₂,p₃,p₄), 
                           octaFoldCount(i₁,i₂,i₃,i₄)*nFold)
    end
    n
end

@inline function getIntCore1122!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂), 
                                 nFold=1)
    for (i₁, p₁) in enumerate(ps₁), (i₂, p₂) in zip(1:i₁, ps₁), 
        (i₃, p₃) in enumerate(ps₂), (i₄, p₄) in zip(1:i₃, ps₂)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁, p₂, p₃, p₄), 
                           diFoldCount(i₁, i₂)*diFoldCount(i₃, i₄)*nFold)
    end
    n
end

@inline function getIntCore1212!(n, uniquePairs, uPairCoeffs, flags, 
                                 (ps₁, ps₂)::Tuple{NTuple{N1}, NTuple{N2}}, 
                                 nFold=1) where {N1, N2}
    oneSidePairs = Iterators.product(1:N1, 1:N2)
    for (x, (i₁,i₂)) in enumerate(oneSidePairs), (_, (i₃,i₄)) in zip(1:x, oneSidePairs)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, 
                           (ps₁[i₁], ps₂[i₂], ps₁[i₃], ps₂[i₄]), 2^(i₁!=i₃ || i₂!=i₄)*nFold)
    end
    n
end

@inline function getIntCore1123!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃), 
                                 nFold=1)
    for (i₁, p₁) in enumerate(ps₁), (i₂, p₂) in zip(1:i₁, ps₁), p₃ in ps₂, p₄ in ps₃
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁, p₂, p₃, p₄), 
                           diFoldCount(i₁, i₂)*nFold)
    end
    n
end

@inline function getIntCore1233!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃), 
                                 nFold=1)
    for p₁ in ps₁, p₂ in ps₂, (i₃, p₃) in enumerate(ps₃), (i₄, p₄) in zip(1:i₃, ps₃)
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁, p₂, p₃, p₄), 
                           diFoldCount(i₃, i₄)*nFold)
    end
    n
end

@inline function getIntCore1234!(n, uniquePairs, uPairCoeffs, flags, (ps₁, ps₂, ps₃, ps₄), 
                                 nFold=1)
    for p₁ in ps₁, p₂ in ps₂, p₃ in ps₃, p₄ in ps₄
        n = getUniquePair!(n, uniquePairs, uPairCoeffs, flags, (p₁, p₂, p₃, p₄), nFold)
    end
    n
end

getOverlap(bf1::BasisFunc{<:Any, GN1}, bf2::BasisFunc{<:Any, GN2}) where {GN1, GN2} = 
getOneBodyInt(FunctionType{:∫overlapCore}(), bf1, bf2)

getElecKinetic(bf1::BasisFunc{<:Any, GN1}, bf2::BasisFunc{<:Any, GN2}) where {GN1, GN2} = 
getOneBodyInt(FunctionType{:∫elecKineticCore}(), bf1, bf2)

function getNucAttraction(bf1::BasisFunc{<:Any, GN1}, bf2::BasisFunc{<:Any, GN2}, 
                          nuc::AbstractArray{String}, 
                          nucCoords::AbstractArray{<:AbstractArray{<:Real}}) where 
                         {GN1, GN2}
    res = 0.0
    for (ele, coord) in zip(nuc, nucCoords)
        res += getOneBodyInt(FunctionType{:∫nucAttractionCore}(), 
                             bf1, bf2, getCharge(ele), coord|>Tuple)
    end
    res
end

function get2eInteraction(bf1::BasisFunc{<:Any, GN1}, bf2::BasisFunc{<:Any, GN2}, 
                          bf3::BasisFunc{<:Any, GN3}, bf4::BasisFunc{<:Any, GN4}) where 
                         {GN1, GN2, GN3, GN4}
    getTwoBodyInt(FunctionType{:∫eeInteractionCore}(), bf1, bf2, bf3, bf4)
end

@inline function getCompositeInt(::FunctionType{F}, bs::NTuple{N, FloatingGTBasisFuncs}, 
                                 optArgs...) where {F, N}
    range = Iterators.product(bs...)
    map(x->getfield(Quiqbox, F)(x..., optArgs...)::Float64, range)::Array{Float64, N}
end

@inline function getCompositeInt(::FunctionType{F}, 
                                 bs::NTuple{N, CompositeGTBasisFuncs{<:Any, 1}}, 
                                 optArgs...) where {F, N}
    range = Iterators.product(unpackBasisFuncs.(bs)...)
    ( map(x->getfield(Quiqbox, F)(x..., optArgs...)::Float64, range) |> sum )::Float64
end

getOverlap(b1::AbstractGTBasisFuncs, b2::AbstractGTBasisFuncs) = 
getCompositeInt(FunctionType{:getOverlap}(), (b1, b2))

getElecKinetic(b1::AbstractGTBasisFuncs, b2::AbstractGTBasisFuncs) = 
getCompositeInt(FunctionType{:getElecKinetic}(), (b1, b2))

getNucAttraction(b1::AbstractGTBasisFuncs, b2::AbstractGTBasisFuncs, 
                 nuc::AbstractArray{String}, 
                 nucCoords::AbstractArray{<:AbstractArray{<:Real}}) = 
getCompositeInt(FunctionType{:getNucAttraction}(), (b1, b2), nuc, nucCoords)

getCoreHij(b1::AbstractGTBasisFuncs, b2::AbstractGTBasisFuncs, 
           nuc::AbstractArray{String}, 
           nucCoords::AbstractArray{<:AbstractArray{<:Real}}) = 
getElecKinetic(b1, b2) + getNucAttraction(b1, b2, nuc, nucCoords)

get2eInteraction(b1::AbstractGTBasisFuncs, b2::AbstractGTBasisFuncs, 
                 b3::AbstractGTBasisFuncs, b4::AbstractGTBasisFuncs) = 
getCompositeInt(FunctionType{:get2eInteraction}(), (b1, b2, b3, b4))


@inline function getOneBodyInts(::FunctionType{F}, 
                                basisSet::AbstractArray{<:AbstractGTBasisFuncs}, 
                                optArgs...) where {F}
    subSize = basisSize(basisSet) |> collect
    accuSize = vcat(0, accumulate(+, subSize))
    len = subSize |> sum
    buf = Array{Float64}(undef, len, len)
    for i = 1:length(basisSet), j = 1:i
        int = getfield(Quiqbox, F)(basisSet[i], basisSet[j], optArgs...)
        rowRange = accuSize[i]+1 : accuSize[i+1]
        colRange = accuSize[j]+1 : accuSize[j+1]
        buf[rowRange, colRange] .= int
        buf[colRange, rowRange] .= int |> transpose
    end
    buf
end

getOverlaps(BSet::AbstractArray{<:AbstractGTBasisFuncs}) = 
getOneBodyInts(FunctionType{:getOverlap}(), BSet)

getElecKinetics(BSet::AbstractArray{<:AbstractGTBasisFuncs}) = 
getOneBodyInts(FunctionType{:getElecKinetic}(), BSet)

getNucAttractions(BSet::AbstractArray{<:AbstractGTBasisFuncs}, 
                  nuc::AbstractArray{String}, 
                  nucCoords::AbstractArray{<:AbstractArray{<:Real}}) = 
getOneBodyInts(FunctionType{:getNucAttraction}(), BSet, nuc, nucCoords)

getCoreH(BSet::AbstractArray{<:AbstractGTBasisFuncs}, 
         nuc::AbstractArray{String}, nucCoords::AbstractArray{<:AbstractArray{<:Real}}) = 
getOneBodyInts(FunctionType{:getCoreHij}(), BSet, nuc, nucCoords)


permuteArray(arr::AbstractArray{T, N}, order) where {T, N} = PermutedDimsArray(arr, order)
permuteArray(arr::Number, _) = itself(arr)


@inline function getTwoBodyInts(::FunctionType{F}, 
                                basisSet::AbstractArray{<:AbstractGTBasisFuncs}) where {F}
    subSize = basisSize(basisSet) |> collect
    accuSize = vcat(0, accumulate(+, subSize))
    totalSize = subSize |> sum
    buf = Array{Float64}(undef, totalSize, totalSize, totalSize, totalSize)
    for i = 1:length(basisSet), j = 1:i, k = 1:i, l = 1:(k==i ? j : k)
        I = accuSize[i]+1 : accuSize[i+1]
        J = accuSize[j]+1 : accuSize[j+1]
        K = accuSize[k]+1 : accuSize[k+1]
        L = accuSize[l]+1 : accuSize[l+1]
        subBuf = getfield(Quiqbox, F)(basisSet[i], basisSet[j], basisSet[k], basisSet[l])
        buf[I,J,K,L] .= subBuf
        buf[J,I,K,L] .= permuteArray(subBuf, [2,1,3,4])
        buf[J,I,L,K] .= permuteArray(subBuf, [2,1,4,3])
        buf[I,J,L,K] .= permuteArray(subBuf, [1,2,4,3])
        buf[L,K,I,J] .= permuteArray(subBuf, [4,3,1,2])
        buf[K,L,I,J] .= permuteArray(subBuf, [3,4,1,2])
        buf[K,L,J,I] .= permuteArray(subBuf, [3,4,2,1])
        buf[L,K,J,I] .= permuteArray(subBuf, [4,3,2,1])
    end
    buf
end

get2eInteractions(BSet::AbstractArray{<:AbstractGTBasisFuncs}) = 
getTwoBodyInts(FunctionType{:get2eInteraction}(), BSet)

function genUniqueIndices(basisSet::AbstractArray{<:AbstractGTBasisFuncs})
    s = basisSize(basisSet) |> sum
    uniqueIdx = fill(Int[0,0,0,0], (3*binomial(s, 4)+6*binomial(s, 3)+4*binomial(s, 2)+s))
    index = 1
    for i = 1:s, j = 1:i, k = 1:i, l = 1:(k==i ? j : k)
        uniqueIdx[index] = [i, j, k, l]
        index += 1
    end
    uniqueIdx
end