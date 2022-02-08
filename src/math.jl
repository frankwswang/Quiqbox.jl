# FLoops = "cc61a311-1640-44b5-9fba-1b764f453329"
# StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
# using FLoops
# using StaticArrays

using Base.Threads: @threads, @spawn

using SpecialFunctions: erf
using LinearAlgebra: dot

getNijk(i, j, k) = (2/π)^0.75 * ( 2^(3*(i+j+k)) * factorial(i) * factorial(j) * factorial(k) / 
                (factorial(2i) * factorial(2j) * factorial(2k)) )^0.5

getNα(i, j, k, α) = α^(0.5*(i + j + k) + 0.75)

getNijkα(i, j, k, α) = getNijk(i, j, k) * getNα(i, j, k, α)

getNijkα(ijk::Vector, α) = getNijkα(ijk[1], ijk[2], ijk[3], α)


# Reference: DOI: 10.1088/0143-0807/31/1/004
function Fγ(γ::Int, u::Float64)
    u == 0.0 && (return 1 / (2γ + 1))
    if γ > 0
        t = exp(-u) * sum(factorial(γ-k)/(4^k * factorial(2*(γ-k)) * u^(k+1)) for k=0:(γ-1))
    elseif γ == 0
        t = 0
    else
        error("γ must be non-negative.")
    end
    factorial(2γ) / (2 * factorial(γ)) * (√π * erf(√u) / (4^γ * u^(γ + 0.5)) - t)
end

# function F₀toFγ(γ::Int, u::Float64, Fγu::Float64)
#     res = [Fγu]
#     for _ in 1:γ
#         push!(res, (2u*res[end] + exp(-u)) / (2γ + 1))
#     end
#     reverse(res)
# end

function F₀toFγ(γ::Int, u::Float64, Fγu::Float64)
    res = Array{Float64}(undef, γ+1)
    res[end] = Fγu
    for i in γ:-1:1
        res[i] = (2u*res[i+1] + exp(-u)) / (2γ + 1)
    end
    res
end

F₀toFγ(γ::Int, u::Float64) = F₀toFγ(γ, u, Fγ(γ, u))

@inline function genIntOverlapCore2(Δx, i₁, α₁)
    res = 0.0
    for l₁ in 0:(i₁÷2), l₂ in 0:(l₁-1)
        Ω = 2*(i₁ - l₁ - l₂)
        oRange = (Δx == 0.0) ? Ω÷2 : (0:(Ω÷2))
        for o in oRange
            res += (-1)^o * factorial(Ω) * 2^(2 * (l₁ + l₂) + o) * 
                    α₁^(2i₁ - l₂ - l₁ - o) * Δx^(Ω - 2o) / 
                    ( 4^(l₁ + l₂ + o) * 
                        factorial(l₁) * 
                        factorial(l₂) * 
                        factorial(o ) * 
                        factorial(i₁ - 2l₁) * 
                        factorial(i₁ - 2l₂) * 
                        factorial(Ω  - 2o ) )
        end
    end
    res *= 2
    for l₁ in 0:(i₁÷2)
        Ω = 2*(i₁ - 2l₁)
        oRange = (Δx == 0.0) ? Ω÷2 : (0:(Ω÷2))
        for o in oRange
            res += (-1)^o * factorial(Ω) * 2^(4 * l₁ + o) * 
                    α₁^(2i₁ - 2l₁ - o) * Δx^(Ω - 2o) / 
                    ( 4^(2l₁ + o) * 
                        factorial(l₁)^2 * 
                        factorial(o ) * 
                        factorial(i₁ - 2l₁)^2 * 
                        factorial(Ω - 2o ) )
        end
    end
    res
end

@inline function genIntOverlapCore1(Δx, i₁, i₂, α₁, α₂)
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

function ∫overlapCoreCore(ΔR::Vector{Float64}, ijk₁::Vector{Int}, α₁::Float64)
    res = 1.0
    for i in 1:3
        i₁ = ijk₁[i]
        ΔRᵢ = ΔR[i]
        res *= (-1.0)^(i₁) * factorial(i₁)^2 / (2α₁)^(2i₁) * genIntOverlapCore2(ΔRᵢ, i₁, α₁)
    end
    res
end

function ∫overlapCore(R₁::Vector{Float64}, ijk₁::Vector{Int}, α₁::Float64, 
                      R₂::Vector{Float64}, ijk₂::Vector{Int}, α₂::Float64)
    for n in vcat(ijk₁, ijk₂)
        n < 0 && return 0.0
    end
    α = α₁ + α₂
    ΔR = R₁ - R₂
    res = (π/α)^1.5 * exp(-α₁ * α₂ / α * sum(abs2, ΔR))
    α₁ == α₂ && ijk₁ == ijk₂ && ( return ∫overlapCoreCore(ΔR, ijk₁, α₁)*res )
    for i in 1:3
        i₁, i₂ = ijk₁[i], ijk₂[i]
        res *= (-1.0)^(i₁) * factorial(i₁) * factorial(i₂) / α^(i₁+i₂) * 
               genIntOverlapCore1(ΔR[i], i₁, i₂, α₁, α₂)
    end
    res
end

function ∫elecKineticCore(R₁::Vector{Float64}, ijk₁::Vector{Int}, α₁::Float64,
                          R₂::Vector{Float64}, ijk₂::Vector{Int}, α₂::Float64)
    shifts = ([2,0,0], [0,2,0], [0,0,2])
    0.5 * (α₂ * (4*sum(ijk₂) + 6) * ∫overlapCore(R₁, ijk₁, α₁, R₂, ijk₂, α₂) - 4.0 * α₂^2 * 
        sum(∫overlapCore.(Ref(R₁), Ref(ijk₁), α₁, Ref(R₂), Ref(ijk₂).+shifts, α₂)) - 
        sum(ijk₁ .* (ijk₁.-1) .* 
            ∫overlapCore.(Ref(R₁), Ref(ijk₁), α₁, Ref(R₂), Ref(ijk₂).-shifts, α₂)))
end

@inline function genIntTerm1(Δx::Float64, 
                             i₁::Int, i₂::Int, 
                             α₁::Float64, α₂::Float64, 
                             l₁::Int, l₂::Int, 
                             o₁::Int, o₂::Int)
    # if Δx == 0.0
    #     os = o₁ + o₂
    #     if isodd(os)
    #         @inline _ -> 0.0
    #     else
    #         @inline function (r)
    #             if os == 2r
    #                 (-1)^(o₂+r) * factorial(os) * α₁^(o₂-l₁-r) * α₂^(o₁-l₂-r) / 
    #             (
    #                 4^(l₁+l₂+r) * 
    #                 factorial(l₁) * factorial(l₂) * factorial(o₁) * factorial(o₂) * 
    #                 factorial(r) * factorial(i₁-2l₁-o₁) * factorial(i₂-2l₂-o₂)
    #             )
    #             else
    #                 0.0
    #             end
    #         end
    #     end
    # else
        @inline (r) -> 
            (-1)^(o₂+r) * factorial(o₁+o₂) * α₁^(o₂-l₁-r) * α₂^(o₁-l₂-r) * Δx^(o₁+o₂-2r) / 
            (
                4^(l₁+l₂+r) * 
                factorial(l₁) * factorial(l₂) * factorial(o₁) * factorial(o₂) * 
                factorial(r) * factorial(i₁-2l₁-o₁) * factorial(i₂-2l₂-o₂) * 
                factorial(o₁+o₂-2r)
            )
    # end
end

@inline function genIntTerm1(Δx::Float64, 
                             i₁::Int, 
                             α₁::Float64, 
                             l₁::Int, l₂::Int, 
                             o₁::Int, o₂::Int)
    # if Δx == 0.0
    #     os = o₁ + o₂
    #     if isodd(os)
    #         @inline _ -> 0.0
    #     else
    #         @inline function (r)
    #             if os == 2r
    #                 (-1)^(o₂+r) * factorial(os) * α₁^(-l₁-l₂) / 
    #             (
    #                 4^(l₁+l₂+r) * 
    #                 factorial(l₁) * factorial(l₂) * factorial(o₁) * factorial(o₂) * 
    #                 factorial(r) * factorial(i₁-2l₁-o₁) * factorial(i₁-2l₂-o₂)
    #             )
    #             else
    #                 0.0
    #             end
    #         end
    #     end
    # else
        @inline (r) -> 
            (-1)^(o₂+r) * factorial(o₁+o₂) * α₁^(o₁+o₂-l₁-l₂-2r) * Δx^(o₁+o₂-2r) / 
            (
                4^(l₁+l₂+r) * 
                factorial(l₁) * factorial(l₂) * factorial(o₁) * factorial(o₂) * 
                factorial(r) * factorial(i₁-2l₁-o₁) * factorial(i₁-2l₂-o₂) * 
                factorial(o₁+o₂-2r)
            )
    # end
end

@inline function genIntTerm2(Δx::Float64, 
                             α::Float64, 
                             o₁::Int, 
                             o₂::Int, 
                             μ::Int, 
                             r::Int)
    # if Δx == 0.0
    #     genIntTerm2(α, o₁, o₂, μ, r)
    # else
        @inline (u) ->
            (-1)^u * factorial(μ) * Δx^(μ-2u) / 
            (4^u * factorial(u) * factorial(μ-2u) * α^(o₁+o₂-r+u))
    # end
end

@inline function genIntTerm2(α::Float64, 
                             o₁::Int, o₂::Int, 
                             μ::Int, 
                             r::Int)
    # if isodd(μ)
    #     @inline _ -> 0.0
    # else
        @inline function (u)
            if μ-2u == 0
                (-1)^u * factorial(μ) / (4^u * factorial(u) * α^(o₁+o₂-r+u))
            else
                0.0
            end
        end
    # end
end

@inline function genIntNucAttCore2(ΔRR₀::Vector{Float64}, 
                                   ΔR₁R₂::Vector{Float64}, 
                                   ijk₁::Vector{Int}, 
                                   α₁::Float64, 
                                   β::Float64)
    A = 0.0
    i₁, j₁, k₁ = ijk₁
    for l₁ in 0:(i₁÷2), m₁ in 0:(j₁÷2), n₁ in 0:(k₁÷2), 
        l₂ in 0:l₁,     m₂ in 0:m₁,     n₂ in 0:n₁

        lmn₁ = (l₁, m₁, n₁)
        lmn₂ = (l₂, m₂, n₂)

        B = 0.0
        C = 0
        if l₁ != l₂
            C += 1
        end
        if m₁ != m₂
            C += 1
        end
        if n₁ != n₂
            C += 1
        end

        for o₁ in 0:(i₁-2l₁), p₁ in 0:(j₁-2m₁), q₁ in 0:(k₁-2n₁), 
            o₂ in 0:(i₁-2l₂), p₂ in 0:(j₁-2m₂), q₂ in 0:(k₁-2n₂)

            opq₁ = (o₁, p₁, q₁)
            opq₂ = (o₂, p₂, q₂)

            μˣ, μʸ, μᶻ = μv = @. 2*ijk₁ - 2*(lmn₁ + lmn₂) - (opq₁ + opq₂)
            μsum = sum(μv)
            Fγs = F₀toFγ(μsum, β)
            core1s = genIntTerm1.(ΔR₁R₂, ijk₁, α₁, lmn₁, lmn₂, opq₁, opq₂)

            for r in 0:((o₁+o₂)÷2), s in 0:((p₁+p₂)÷2), t in 0:((q₁+q₂)÷2)

                rst = (r, s, t)
                tmp = 0.0
                core2s = genIntTerm2.(ΔRR₀, 2α₁, opq₁, opq₂, μv, rst)

                for u in 0:(μˣ÷2), v in 0:(μʸ÷2), w in 0:(μᶻ÷2)
                    γ = μsum - u - v - w
                    tmp += (((u, v, w) .|> core2s)::Vector{Float64} |> prod) * 2Fγs[γ+1]
                end

                B += ((rst .|> core1s)::Vector{Float64} |> prod) * tmp

            end
        end

        A += B * 2^C

    end
    A
end

@inline function genIntNucAttCore1(ΔRR₀::Vector{Float64}, 
                                   ΔR₁R₂::Vector{Float64}, 
                                   ijk₁::Vector{Int}, 
                                   ijk₂::Vector{Int}, 
                                   α₁::Float64, α₂::Float64, 
                                   β::Float64)
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
            core1s = genIntTerm1.(ΔR₁R₂, 
                                  ijk₁, ijk₂, α₁, α₂, lmn₁, lmn₂, opq₁, opq₂)

            for r in 0:((o₁+o₂)÷2), s in 0:((p₁+p₂)÷2), t in 0:((q₁+q₂)÷2)

                rst = (r, s, t)
                tmp = 0.0
                cores2 = genIntTerm2.(ΔRR₀, α₁+α₂, opq₁, opq₂, μv, rst)

                for u in 0:(μˣ÷2), v in 0:(μʸ÷2), w in 0:(μᶻ÷2)
                    γ = μsum - u - v - w
                    tmp += (((u, v, w) .|> cores2)::Vector{Float64} |> prod) * 2Fγs[γ+1]
                end

                A += ((rst .|> core1s)::Vector{Float64} |> prod) * tmp

            end
        end

    end
    A
end

function ∫nucAttractionCore(Z₀::Int, R₀::Vector{Float64}, 
                            R₁::Vector{Float64}, ijk₁::Vector{Int}, α₁::Float64,
                            R₂::Vector{Float64}, ijk₂::Vector{Int}, α₂::Float64)
    if α₁ == α₂
        α = 2α₁
        R = 0.5 * (R₁ + R₂)
        flag = true
    else
        α = α₁ + α₂
        R = (α₁*R₁ + α₂*R₂) / α
        flag = false
    end
    ΔRR₀ = R - R₀
    ΔR₁R₂ = R₁ - R₂
    β = α * sum(abs2, ΔRR₀)
    res = -Z₀ * π / α * exp(-α₁ * α₂ / α * sum(abs2, R₁-R₂))
    if flag && ijk₁ == ijk₂
        res *= (-1)^sum(2ijk₁) * (factorial.(ijk₁) |> prod)^2 * 
               genIntNucAttCore2(ΔRR₀, ΔR₁R₂, ijk₁, α₁, β)
    else
        res *= (-1)^sum(ijk₁ + ijk₂) * (factorial.(vcat(ijk₁, ijk₂)) |> prod) * 
               genIntNucAttCore1(ΔRR₀, ΔR₁R₂, ijk₁, ijk₂, α₁, α₂, β)
    end
    res
end

@inline function genIntTerm3(Δx, i₁, i₂, α₁, α₂, l₁, l₂, o₁, o₂)
    # if Δx == 0.0
    #     os = o₁ + o₂
    #     if isodd(os)
    #         @inline _ -> 0.0
    #     else
    #         @inline function (r)
    #             if os == 2r
    #                 (-1)^(o₂+r) * factorial(os) * α₁^(o₂-l₁-r) * α₂^(o₁-l₂-r) * 
    #                 (α₁+α₂)^(2*(l₁+l₂) + r) / 
    #             (
    #                 4^(l₁+l₂+r) * 
    #                 factorial(l₁) * factorial(l₂) * factorial(o₁) * factorial(o₂) * 
    #                 factorial(r) * factorial(i₁-2l₁-o₁) * factorial(i₂-2l₂-o₂)
    #             )
    #             else
    #                 0.0
    #             end
    #         end
    #     end
    # else
        @inline (r) -> 
            (-1)^(o₂+r) * factorial(o₁+o₂) * α₁^(o₂-l₁-r) * α₂^(o₁-l₂-r) * 
            (α₁+α₂)^(2*(l₁+l₂) + r) * Δx^(o₁+o₂-2r) / 
            (
                4^(l₁+l₂+r) * 
                factorial(l₁) * factorial(l₂) * factorial(o₁) * factorial(o₂) * 
                factorial(r) * factorial(i₁-2l₁-o₁) * factorial(i₂-2l₂-o₂) * 
                factorial(o₁+o₂-2r)
            )
    # end
end

@inline function genIntTerm3(Δx, i₁, α₁, l₁, l₂, o₁, o₂)
    # if Δx == 0.0
    #     os = o₁ + o₂
    #     if isodd(os)
    #         @inline _ -> 0.0
    #     else
    #         @inline function (r)
    #             if os == 2r
    #                 (-1)^(o₂+r) * factorial(os) * α₁^(l₁+l₂+r) / 
    #             (
    #                 2^r * 
    #                 factorial(l₁) * factorial(l₂) * factorial(o₁) * factorial(o₂) * 
    #                 factorial(r) * factorial(i₁-2l₁-o₁) * factorial(i₁-2l₂-o₂)
    #             )
    #             else
    #                 0.0
    #             end
    #         end
    #     end
    # else
        @inline (r) -> 
            (-1)^(o₂+r) * factorial(o₁+o₂) * α₁^(o₁+o₂+l₁+l₂-r) * Δx^(o₁+o₂-2r) / 
            (
                2^r * 
                factorial(l₁) * factorial(l₂) * factorial(o₁) * factorial(o₂) * 
                factorial(r) * factorial(i₁-2l₁-o₁) * factorial(i₁-2l₂-o₂) * 
                factorial(o₁+o₂-2r)
            )
    # end
end

@inline function genIntTerm4(Δx, η, μ)
    # if Δx == 0.0
    #     genIntTerm4(η, μ)
    # else
        @inline (u) ->
            (-1)^u * factorial(μ) * η^(μ-u) * Δx^(μ-2u) / 
            (4^u * factorial(u) * factorial(μ-2u))
    # end
end

@inline function genIntTerm4(η, μ)
    # if isodd(μ)
    #     @inline _ -> 0.0
    # else
        @inline function (u)
            if μ-2u == 0
                (-1)^u * factorial(μ) * η^(μ-u) / (4^u * factorial(u))
            else
                0.0
            end
        end
    # end
end



# number indicate mark for unique (ijk, α)
function ∫eeInteractionCore1122(ΔRl::Vector{Float64}, ΔRr::Vector{Float64}, ΔRc::Vector{Float64}, 
                                ijk₁::Vector{Int}, α₁::Float64, 
                                ijk₂::Vector{Int}, α₂::Float64, β, η)
    A = 0.0
    (i₁, j₁, k₁), (i₂, j₂, k₂) = ijk₁, ijk₂

    IJK = 2 * (ijk₁ + ijk₂)

    for l₁ in 0:(i₁÷2), m₁ in 0:(j₁÷2), n₁ in 0:(k₁÷2), 
        l₂ in 0:l₁,     m₂ in 0:m₁,     n₂ in 0:n₁, 
        l₃ in 0:(i₂÷2), m₃ in 0:(j₂÷2), n₃ in 0:(k₂÷2), 
        l₄ in 0:l₃,     m₄ in 0:m₃,     n₄ in 0:n₃

        lmn₁ = (l₁, m₁, n₁)
        lmn₂ = (l₂, m₂, n₂)
        lmn₃ = (l₃, m₃, n₃)
        lmn₄ = (l₄, m₄, n₄)

        B = 0.0
        C = 0
        if l₁ != l₂
            C += 1
        end
        if m₁ != m₂
            C += 1
        end
        if n₁ != n₂
            C += 1
        end
        if l₃ != l₄
            C += 1
        end
        if m₃ != m₄
            C += 1
        end
        if n₃ != n₄
            C += 1
        end

        for o₁ in 0:(i₁-2l₁), p₁ in 0:(j₁-2m₁), q₁ in 0:(k₁-2n₁), 
            o₂ in 0:(i₁-2l₂), p₂ in 0:(j₁-2m₂), q₂ in 0:(k₁-2n₂), 
            o₃ in 0:(i₂-2l₃), p₃ in 0:(j₂-2m₃), q₃ in 0:(k₂-2n₃), 
            o₄ in 0:(i₂-2l₄), p₄ in 0:(j₂-2m₄), q₄ in 0:(k₂-2n₄)

            opq₁ = (o₁, p₁, q₁)
            opq₂ = (o₂, p₂, q₂)
            opq₃ = (o₃, p₃, q₃)
            opq₄ = (o₄, p₄, q₄)

            μˣ, μʸ, μᶻ = μv = begin
                @. IJK - (lmn₁ + lmn₂ + lmn₃ + lmn₄) * 2 - (opq₁ + opq₂ + opq₃ + opq₄)
            end

            μsum = sum(μv)
            Fγs = F₀toFγ(μsum, β)

            core1s = genIntTerm3.(ΔRl, ijk₁, α₁, lmn₁, lmn₂, opq₁, opq₂)
            core2s = genIntTerm3.(ΔRr, ijk₂, α₂, lmn₃, lmn₄, opq₃, opq₄)
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

                B += (rst₁ .|> core1s |> prod) * (rst₂ .|> core2s |> prod) * tmp

            end
        end

        A += B * 2^C

    end
    A
end
    # R1-R2 == R3-R4
function ∫eeInteractionCore1111(ΔRl::Vector{Float64}, ΔRr::Vector{Float64}, ΔRc::Vector{Float64}, 
                                ijk₁::Vector{Int}, α₁::Float64, β, η)
    A = 0.0
    (i₁, j₁, k₁) = ijk₁

    IJK = 4ijk₁

    flags = fill(false, 3)
    for i=1:3
        if ΔRl[i] == ΔRr[i]
            flags[i] = true
        end
    end

    for l₁ in 0:(i₁÷2), m₁ in 0:(j₁÷2), n₁ in 0:(k₁÷2), 
        l₂ in 0:l₁,     m₂ in 0:m₁,     n₂ in 0:n₁, 
        l₃ in 0:(flags[1] ? l₁ : (i₁÷2)), 
        m₃ in 0:(flags[2] ? m₁ : (j₁÷2)), 
        n₃ in 0:(flags[3] ? n₁ : (k₁÷2)), 
        l₄ in 0:((flags[1] && l₃==l₁) ? l₂ : l₃), 
        m₄ in 0:((flags[2] && m₃==m₁) ? m₂ : m₃), 
        n₄ in 0:((flags[3] && n₃==n₁) ? n₂ : n₃)

        lmn₁ = (l₁, m₁, n₁)
        lmn₂ = (l₂, m₂, n₂)
        lmn₃ = (l₃, m₃, n₃)
        lmn₄ = (l₄, m₄, n₄)

        B = 0.0
        C = 0
        if l₁ != l₂
            C += 1
        end
        if m₁ != m₂
            C += 1
        end
        if n₁ != n₂
            C += 1
        end
        if l₃ != l₄
            C += 1
        end
        if m₃ != m₄
            C += 1
        end
        if n₃ != n₄
            C += 1
        end
        if flags[1] && (l₁ != l₃ || l₂ != l₄)
            C += 1
        end
        if flags[2] && (m₁ != m₃ || m₂ != m₄)
            C += 1
        end
        if flags[3] && (n₁ != n₃ || n₂ != n₄)
            C += 1
        end

        for o₁ in 0:(i₁-2l₁), p₁ in 0:(j₁-2m₁), q₁ in 0:(k₁-2n₁), 
            o₂ in 0:(i₁-2l₂), p₂ in 0:(j₁-2m₂), q₂ in 0:(k₁-2n₂), 
            o₃ in 0:(i₁-2l₃), p₃ in 0:(j₁-2m₃), q₃ in 0:(k₁-2n₃), 
            o₄ in 0:(i₁-2l₄), p₄ in 0:(j₁-2m₄), q₄ in 0:(k₁-2n₄)

            opq₁ = (o₁, p₁, q₁)
            opq₂ = (o₂, p₂, q₂)
            opq₃ = (o₃, p₃, q₃)
            opq₄ = (o₄, p₄, q₄)

            μˣ, μʸ, μᶻ = μv = begin
                @. IJK - (lmn₁ + lmn₂ + lmn₃ + lmn₄) * 2 - (opq₁ + opq₂ + opq₃ + opq₄)
            end

            μsum = sum(μv)
            Fγs = F₀toFγ(μsum, β)

            core1s = genIntTerm3.(ΔRl, ijk₁, α₁, lmn₁, lmn₂, opq₁, opq₂)
            core2s = genIntTerm3.(ΔRr, ijk₁, α₁, lmn₃, lmn₄, opq₃, opq₄)
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

                B += (rst₁ .|> core1s |> prod) * (rst₂ .|> core2s |> prod) * tmp

            end
        end

        A += B * 2^C

    end
    A
end

function ∫eeInteractionCore1123(ΔRl::Vector{Float64}, ΔRr::Vector{Float64}, ΔRc::Vector{Float64},
                                ijk₁::Vector{Int}, α₁::Float64, 
                                ijk₂::Vector{Int}, α₂::Float64, 
                                ijk₃::Vector{Int}, α₃::Float64, β, η)
    A = 0.0
    (i₁, j₁, k₁), (i₂, j₂, k₂), (i₃, j₃, k₃) = ijk₁, ijk₂, ijk₃

    IJK = 2ijk₁ + ijk₂ + ijk₃

    for l₁ in 0:(i₁÷2), m₁ in 0:(j₁÷2), n₁ in 0:(k₁÷2), 
        l₂ in 0:l₁,     m₂ in 0:m₁,     n₂ in 0:n₁, 
        l₃ in 0:(i₂÷2), m₃ in 0:(j₂÷2), n₃ in 0:(k₂÷2), 
        l₄ in 0:(i₃÷2), m₄ in 0:(j₃÷2), n₄ in 0:(k₃÷2)

        lmn₁ = (l₁, m₁, n₁)
        lmn₂ = (l₂, m₂, n₂)
        lmn₃ = (l₃, m₃, n₃)
        lmn₄ = (l₄, m₄, n₄)

        B = 0.0
        C = 0
        if l₁ != l₂
            C += 1
        end
        if m₁ != m₂
            C += 1
        end
        if n₁ != n₂
            C += 1
        end

        for o₁ in 0:(i₁-2l₁), p₁ in 0:(j₁-2m₁), q₁ in 0:(k₁-2n₁), 
            o₂ in 0:(i₁-2l₂), p₂ in 0:(j₁-2m₂), q₂ in 0:(k₁-2n₂), 
            o₃ in 0:(i₂-2l₃), p₃ in 0:(j₂-2m₃), q₃ in 0:(k₂-2n₃), 
            o₄ in 0:(i₃-2l₄), p₄ in 0:(j₃-2m₄), q₄ in 0:(k₃-2n₄)

            opq₁ = (o₁, p₁, q₁)
            opq₂ = (o₂, p₂, q₂)
            opq₃ = (o₃, p₃, q₃)
            opq₄ = (o₄, p₄, q₄)

            μˣ, μʸ, μᶻ = μv = begin
                @. IJK - (lmn₁ + lmn₂ + lmn₃ + lmn₄) * 2 - (opq₁ + opq₂ + opq₃ + opq₄)
            end

            μsum = sum(μv)
            Fγs = F₀toFγ(μsum, β)

            core1s = genIntTerm3.(ΔRl, ijk₁, α₁, lmn₁, lmn₂, opq₁, opq₂)
            core2s = genIntTerm3.(ΔRr, ijk₂, ijk₃, α₂, α₃, lmn₃, lmn₄, opq₃, opq₄)
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

                B += (rst₁ .|> core1s |> prod) * (rst₂ .|> core2s |> prod) * tmp

            end
        end

        A += B * 2^C

    end
    A
end

∫eeInteractionCore1233(ΔRl::Vector{Float64}, ΔRr::Vector{Float64}, ΔRc::Vector{Float64}, 
                       ijk₁::Vector{Int}, α₁::Float64, 
                       ijk₂::Vector{Int}, α₂::Float64, 
                       ijk₃::Vector{Int}, α₃::Float64, β, η) = 
∫eeInteractionCore1123(ΔRl, ΔRr, ΔRc, ijk₃, α₃, ijk₁, α₁, ijk₂, α₂, β, η)

function ∫eeInteractionCore1234(ΔRl::Vector{Float64}, ΔRr::Vector{Float64}, ΔRc::Vector{Float64}, 
                                ijk₁::Vector{Int}, α₁::Float64, 
                                ijk₂::Vector{Int}, α₂::Float64, 
                                ijk₃::Vector{Int}, α₃::Float64, 
                                ijk₄::Vector{Int}, α₄::Float64, β, η)
    A = 0.0
    (i₁, j₁, k₁), (i₂, j₂, k₂), (i₃, j₃, k₃), (i₄, j₄, k₄) = ijk₁, ijk₂, ijk₃, ijk₄

    IJK = ijk₁ + ijk₂ + ijk₃ + ijk₄

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

            core1s = genIntTerm3.(ΔRl, ijk₁, ijk₂, α₁, α₂, lmn₁, lmn₂, opq₁, opq₂)
            core2s = genIntTerm3.(ΔRr, ijk₃, ijk₄, α₃, α₄, lmn₃, lmn₄, opq₃, opq₄)
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



# R1-R2 == R3-R4
function ∫eeInteractionCore1212(ΔRl::Vector{Float64}, ΔRr::Vector{Float64}, ΔRc::Vector{Float64},
                                ijk₁::Vector{Int}, α₁::Float64, 
                                ijk₂::Vector{Int}, α₂::Float64, β, η)
    A = 0.0
    (i₁, j₁, k₁), (i₂, j₂, k₂) = ijk₁, ijk₂

    IJK = 2 * (ijk₁ + ijk₂)

    flags = fill(false, 3)
    for i=1:3
        if ΔRl[i] == ΔRr[i]
            flags[i] = true
        end
    end

    for l₁ in 0:(i₁÷2), m₁ in 0:(j₁÷2), n₁ in 0:(k₁÷2), 
        l₂ in 0:(i₂÷2), m₂ in 0:(j₂÷2), n₂ in 0:(k₂÷2), 
        l₃ in 0:(flags[1] ? l₁ : (i₁÷2)), 
        m₃ in 0:(flags[2] ? m₁ : (j₁÷2)), 
        n₃ in 0:(flags[3] ? n₁ : (k₁÷2)), 
        l₄ in 0:(flags[1] ? l₂ : (i₂÷2)),
        m₄ in 0:(flags[2] ? m₂ : (j₂÷2)),
        n₄ in 0:(flags[3] ? n₂ : (k₂÷2))

        lmn₁ = (l₁, m₁, n₁)
        lmn₂ = (l₂, m₂, n₂)
        lmn₃ = (l₃, m₃, n₃)
        lmn₄ = (l₄, m₄, n₄)

        B = 0.0
        C = 0
        if flags[1] && l₁ != l₃
            C += 1
        end
        if flags[2] && m₁ != m₃
            C += 1
        end
        if flags[3] && n₁ != n₃
            C += 1
        end
        if flags[1] && l₂ != l₄
            C += 1
        end
        if flags[2] && m₂ != m₄
            C += 1
        end
        if flags[3] && n₂ != n₄
            C += 1
        end

        for o₁ in 0:(i₁-2l₁), p₁ in 0:(j₁-2m₁), q₁ in 0:(k₁-2n₁), 
            o₂ in 0:(i₂-2l₂), p₂ in 0:(j₂-2m₂), q₂ in 0:(k₂-2n₂), 
            o₃ in 0:(i₁-2l₃), p₃ in 0:(j₁-2m₃), q₃ in 0:(k₁-2n₃), 
            o₄ in 0:(i₂-2l₄), p₄ in 0:(j₂-2m₄), q₄ in 0:(k₂-2n₄)

            opq₁ = (o₁, p₁, q₁)
            opq₂ = (o₂, p₂, q₂)
            opq₃ = (o₃, p₃, q₃)
            opq₄ = (o₄, p₄, q₄)

            μˣ, μʸ, μᶻ = μv = begin
                @. IJK - (lmn₁ + lmn₂ + lmn₃ + lmn₄) * 2 - (opq₁ + opq₂ + opq₃ + opq₄)
            end

            μsum = sum(μv)
            Fγs = F₀toFγ(μsum, β)

            core1s = genIntTerm3.(ΔRl, ijk₁, ijk₂, α₁, α₂, lmn₁, lmn₂, opq₁, opq₂)
            core2s = genIntTerm3.(ΔRr, ijk₁, ijk₂, α₁, α₂, lmn₃, lmn₄, opq₃, opq₄)
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

                B += (rst₁ .|> core1s |> prod) * (rst₂ .|> core2s |> prod) * tmp

            end
        end

        A += B * 2^C

    end
    A
end

function ∫eeInteractionCore(R₁::Vector{Float64}, ijk₁::Vector{Int}, α₁::Float64, 
                            R₂::Vector{Float64}, ijk₂::Vector{Int}, α₂::Float64,
                            R₃::Vector{Float64}, ijk₃::Vector{Int}, α₃::Float64, 
                            R₄::Vector{Float64}, ijk₄::Vector{Int}, α₄::Float64)
    ΔRl = R₁ - R₂
    ΔRr = R₃ - R₄
    αl = α₁ + α₂
    αr = α₃ + α₄
    ηl = α₁ * α₂ / αl
    ηr = α₃ * α₄ / αr
    ΔRc = (α₁*R₁ + α₂*R₂)/αl - (α₃*R₃ + α₄*R₄)/αr
    η = αl * αr / (α₁ + α₂ + α₃ + α₄)
    β = η * sum(abs2, ΔRc)
    res = π^2.5 / (αl * αr * (αl + αr)^0.5) * 
          exp(-ηl * sum(abs2, ΔRl)) * exp(-ηr * sum(abs2, ΔRr))
    res *= (@. (-1.0)^(ijk₁ + ijk₂) * 
              factorial(ijk₁) * factorial(ijk₂) * factorial(ijk₃) * factorial(ijk₄) / 
              αl^(ijk₁+ijk₂) / αr^(ijk₃+ijk₄)) |> prod
    # pairs = [(ijk₁, α₁), (ijk₂, α₂), (ijk₃, α₃), (ijk₄, α₄)]
    # marks, list = markUnique(pairs)
    # if marks == [1,1,1,1]
    #     ∫eeInteractionCore1111()
    # elseif marks == [1,1,2,2]
    #     ∫eeInteractionCore1122()
    # elseif marks == [1,2,1,2], [1,2,2,1]
    #     ∫eeInteractionCore1212()
    # elseif marks == [1,1,2,1], [1,1,1,2], [1,1,2,3]
    #     ∫eeInteractionCore1123()
    # elseif marks == [1,2,1,1], [1,2,2,2], [1,2,3,3]
    #     ∫eeInteractionCore1233()
    # else [1,2,3,4], [1,2,3,1], [1,2,1,3]
    #     ∫eeInteractionCore1234()
    # end

    if α₁ == α₂ && ijk₁ == ijk₂
        if α₃ == α₄ && ijk₃ == ijk₄
            if α₃ == α₁ && ijk₃ == ijk₁
                res *= (@. factorial(ijk₁)^4 / αl^(4ijk₁)) |> prod
                J = ∫eeInteractionCore1111(ΔRl, ΔRr, ΔRc, ijk₁, α₁, β, η)
            else
                res *= (@. factorial(ijk₁)^2 * factorial(ijk₃)^2 / 
                           αl^(2ijk₁) / αr^(2ijk₃)) |> prod
                J = ∫eeInteractionCore1122(ΔRl, ΔRr, ΔRc, ijk₁, α₁, ijk₃, α₃, β, η)
            end
        else
            res *= (@. factorial(ijk₁)^2 * factorial(ijk₃) * factorial(ijk₄) / 
                       αl^(2ijk₁) / αr^(ijk₃+ijk₄)) |> prod
            J = ∫eeInteractionCore1123(ΔRl, ΔRr, ΔRc, ijk₁, α₁, ijk₃, α₃, ijk₄, α₄, β, η)
        end
    elseif α₃ == α₄ && ijk₃ == ijk₄
        res *= (@. (-1.0)^(ijk₁ + ijk₂) * 
                   factorial(ijk₁) * factorial(ijk₂) * factorial(ijk₃)^2 / 
                   αl^(ijk₁+ijk₂) / αr^(2ijk₃)) |> prod
        J = ∫eeInteractionCore1233(ΔRl, ΔRr, ΔRc, ijk₁, α₁, ijk₂, α₂, ijk₃, α₃, β, η)
    elseif ((α₃ == α₁ && ijk₃ == ijk₁) || (α₃ == α₂ && ijk₃ == ijk₂)) && 
           ((α₄ == α₁ && ijk₄ == ijk₁) || (α₄ == α₂ && ijk₄ == ijk₂))
        res *= (@. (-1.0)^(ijk₁ + ijk₂) * factorial(ijk₁)^2 * factorial(ijk₂)^2 / 
                   αl^(2*(ijk₁+ijk₂))) |> prod
        J = ∫eeInteractionCore1212(ΔRl, ΔRr, ΔRc, ijk₁, α₁, ijk₂, α₂, β, η)
    else
        res *= (@. (-1.0)^(ijk₁ + ijk₂) * 
                   factorial(ijk₁) * factorial(ijk₂) * factorial(ijk₃) * factorial(ijk₄) / 
                   αl^(ijk₁+ijk₂) / αr^(ijk₃+ijk₄)) |> prod
        J = ∫eeInteractionCore1234(ΔRl, ΔRr, ΔRc, ijk₁, α₁, ijk₂, α₂, ijk₃, α₃, ijk₄, α₄, β, η)
    end
    res * J
end

function loadIntData(bf::FloatingGTBasisFuncs{<:Any, GN, 1}) where {GN}
    cen = centerCoordOf(bf)
    ijk = ijkOrbitalList[bf.ijk[1]]
    [(cen, ijk, g.xpn(), normalizeGTO, g.con()) for g in bf.gauss::NTuple{GN, GaussFunc}]
end

function reformatIntData2(o1::T, o2::T, flag::Bool) where {T}
    ( (flag && isless(o2, o1)) ? (o2, o1) : (o1, o2) )::NTuple{2, T}
end

function reformatIntData2(o1::T, o2::T, o3::T, o4::T, flag::NTuple{3, Bool}) where {T}
    p1 = (flag[1] && isless(o2, o1)) ? (o2, o1) : (o1, o2)
    p2 = (flag[2] && isless(o4, o3)) ? (o4, o3) : (o3, o4)
    ((flag[3] && isless(p2, p1)) ? (p2..., p1...) : (p1..., p2...) )::NTuple{4, T}
end

# loadData1(bfs::FloatingGTBasisFuncs{<:Any, <:Any, 1}...) = 
# [(centerCoordOf(bf), ijkOrbitalList[bf.ijk[1]]) bf in bfs]

# # function loadData2(gfs::GaussFunc...)
# #     [gf.xpn() for gf in gfs]
# # end

# # function loadData3(gfs::GaussFunc...)
# #     [gf.con() for gf in gfs]
# # end

# function reformatDataCore(ijk::Vector{Int}, α::Float64, c::Float64, flag::Bool)
#     flag ? c : c * 
# end


# function reformatData(ijk₁::Vector{Int}, ijk₂::Vector{Int}, 
#                         c₁::Float64,       c₁::Float64, 
#                         f₁::Bool,          f₁::Bool)
#     res = c1 * c2
#     f1 && (res *= )
#     f2 && (res *= )
#     res
# end

# function reformatIntData1_old(bf::FloatingGTBasisFuncs{<:Any, GN, 1}) where {GN}
#     R = centerCoordOf(bf)::Vector{Float64}
#     ijk = ijkOrbitalList[bf.ijk[1]]::Vector{Int}
#     cons = if bf.normalizeGTO
#         N = getNijk(ijk...)
#         Float64[gf.con() * N * getNα(ijk..., gf.xpn()) 
#                 for gf in bf.gauss::NTuple{GN, GaussFunc}]
#     else
#         Float64[gf.con() for gf in bf.gauss::NTuple{GN, GaussFunc}]
#     end
#     R, ijk, cons
# end


# function getOneBodyInt(func::Symbol, 
#                        bf1::FloatingGTBasisFuncs{<:Any, GN1, 1}, 
#                        bf2::FloatingGTBasisFuncs{<:Any, GN2, 1}, 
#                        optArgs...) where {GN1, GN2}
#     R₁, ijk₁, cons₁ = reformatIntData1_old(bf1)
#     R₂, ijk₂, cons₂ = reformatIntData1_old(bf2)

#     αPairs = [reformatIntData2(i.xpn(), j.xpn(), R₁==R₂ && ijk₁==ijk₂)
#               for i in bf1.gauss, j in bf2.gauss]
#     marks, ls = markUnique(αPairs, compareFunction=isequal)

#     ints = map(ls) do x
#         getfield(Quiqbox, func)(optArgs..., R₁, ijk₁, x[1], R₂, ijk₂, x[2])::Float64
#     end

#     map((x,y)->ints[x] * y, marks, cons₁ * transpose(cons₂)) |> sum
# end

# function getTwoBodyInt(func::Symbol, 
#                        bf1::FloatingGTBasisFuncs{<:Any, GN1, 1}, 
#                        bf2::FloatingGTBasisFuncs{<:Any, GN2, 1}, 
#                        bf3::FloatingGTBasisFuncs{<:Any, GN3, 1}, 
#                        bf4::FloatingGTBasisFuncs{<:Any, GN4, 1}, 
#                        optArgs...) where {GN1, GN2, GN3, GN4}
#     R₁, ijk₁, cons₁ = reformatIntData1_old(bf1)
#     R₂, ijk₂, cons₂ = reformatIntData1_old(bf2)
#     R₃, ijk₃, cons₃ = reformatIntData1_old(bf3)
#     R₄, ijk₄, cons₄ = reformatIntData1_old(bf4)

#     f1 = (R₁ == R₂ && ijk₁ == ijk₂)
#     f2 = (R₃ == R₄ && ijk₃ == ijk₄)
#     f3 = (R₁ == R₃ && ijk₁ == ijk₃ && R₂ == R₄ && ijk₂ == ijk₄)

#     αPairs = [reformatIntData2(i.xpn(), j.xpn(), k.xpn(), l.xpn(), (f1, f2, f3)) 
#               for i in bf1.gauss, j in bf2.gauss, k in bf3.gauss, l in bf4.gauss]
#     marks, ls = markUnique(αPairs, compareFunction=isequal)

#     ints = map(ls) do x
#         getfield(Quiqbox, func)(optArgs..., R₁, ijk₁, x[1], R₂, ijk₂, x[2], 
#                                             R₃, ijk₃, x[3], R₄, ijk₄, x[4])
#     end

#     cons = reshape(cons₁, GN1, 1, 1, 1) .* reshape(cons₂, 1, GN2, 1, 1) .* 
#            reshape(cons₃, 1, 1, GN3, 1) .* reshape(cons₄, 1, 1, 1, GN4)

#     (map((x,y)->ints[x] * y, marks, cons) |> sum)::Float64
# end



function reformatIntData1(bf::FloatingGTBasisFuncs{<:Any, GN, 1}) where {GN}
    R = centerCoordOf(bf)::Vector{Float64}
    ijk = ijkOrbitalList[bf.ijk[1]]::Vector{Int}
    # cons = if bf.normalizeGTO
    #     N = getNijk(ijk...)
    #     [(x.xpn()::Float64, x.con() * N * getNα(ijk..., x.xpn())::Float64) 
    #      for x in bf.gauss::NTuple{GN, GaussFunc}]
    # else
    #     [(x.xpn()::Float64, x.con()::Float64) 
    #      for x in bf.gauss::NTuple{GN, GaussFunc}]
    # end
    cons = if bf.normalizeGTO
        N = getNijk(ijk...)
        map(x->(x.xpn()::Float64, x.con() * N * getNα(ijk..., x.xpn())::Float64), 
            bf.gauss::NTuple{GN, GaussFunc})
    else
        map(x->(x.xpn()::Float64, x.con()::Float64), bf.gauss::NTuple{GN, GaussFunc})
    end
    R, ijk, cons
end

# function getOneBodyInt(func::Symbol, 
#                        bf1::FloatingGTBasisFuncs{<:Any, GN1, 1}, 
#                        bf2::FloatingGTBasisFuncs{<:Any, GN2, 1}, 
#                        optArgs...) where {GN1, GN2}
#     GN1 < GN2 ? getOneBodyIntCore(func, bf1, bf2, optArgs...) : 
#                 getOneBodyIntCore(func, bf2, bf1, optArgs...)
# end

function getOneBodyInt(func::Symbol, 
                           bf1::FloatingGTBasisFuncs{<:Any, GN1, 1}, 
                           bf2::FloatingGTBasisFuncs{<:Any, GN2, 1}, 
                           optArgs...) where {GN1, GN2}
    # R₁, ijk₁, cons₁ = reformatIntData1_old(bf1)
    # R₂, ijk₂, cons₂ = reformatIntData1_old(bf2)

    # xpns1 = map(x->x.xpn(), bf1.gauss::NTuple{GN1, GaussFunc})
    # xpns2 = map(x->x.xpn(), bf2.gauss::NTuple{GN2, GaussFunc})

    # uniquePairs = NTuple{2, Float64}[]
    # marks = zeros(Int, GN1, GN2)
    # for (α₁, i) in zip(xpns1::NTuple{GN1, Float64}, 1:GN1), 
    #     (α₂, j) in zip(xpns2::NTuple{GN1, Float64}, 1:GN2)
    #     pair = reformatIntData2(α₁, α₂, R₁==R₂ && ijk₁==ijk₂)
    #     idx = findfirst(x->x==pair, uniquePairs)
    #     if idx === nothing
    #         push!(uniquePairs, pair)
    #         marks[i,j] = lastindex(uniquePairs)
    #     else
    #         marks[i,j] = idx
    #     end
    # end

    # ints = map(uniquePairs) do x
    #     getfield(Quiqbox, func)(optArgs..., R₁, ijk₁, x[1], R₂, ijk₂, x[2])::Float64
    # end

    # map((x,y)->ints[x] * y, marks, cons₁ * transpose(cons₂)) |> sum


    (R₁, ijk₁, ps₁), (R₂, ijk₂, ps₂) = reformatIntData1.((bf1, bf2))
    uniquePairs = NTuple{2, Float64}[]
    sizehint!(uniquePairs, GN1*GN2)
    uPairCoeffs = Array{Float64}(undef, GN1*GN2)
    i = 0

    for p₁ in ps₁::NTuple{GN1, NTuple{2, Float64}}, 
        p₂ in ps₂::NTuple{GN2, NTuple{2, Float64}}
        pair = reformatIntData2(p₁[1], p₂[1], R₁==R₂ && ijk₁==ijk₂)
        idx = findfirst(x->x==pair, uniquePairs)
        con = p₁[2] * p₂[2]
        if idx === nothing
            i += 1
            push!(uniquePairs, pair)
            uPairCoeffs[i] = con
        else
            uPairCoeffs[idx] += con
        end
    end
    map(uniquePairs, uPairCoeffs) do x, y
        getfield(Quiqbox, func)(optArgs..., R₁, ijk₁, x[1], R₂, ijk₂, x[2])::Float64 * y
    end |> sum
end

function getTwoBodyInt(func::Symbol, 
                       bf1::FloatingGTBasisFuncs{<:Any, GN1, 1}, 
                       bf2::FloatingGTBasisFuncs{<:Any, GN2, 1}, 
                       bf3::FloatingGTBasisFuncs{<:Any, GN3, 1}, 
                       bf4::FloatingGTBasisFuncs{<:Any, GN4, 1}, 
                       optArgs...) where {GN1, GN2, GN3, GN4}
    (R₁, ijk₁, ps₁), (R₂, ijk₂, ps₂), (R₃, ijk₃, ps₃), (R₄, ijk₄, ps₄) = 
    reformatIntData1.((bf1, bf2, bf3, bf4))

    f1 = (R₁ == R₂ && ijk₁ == ijk₂)
    f2 = (R₃ == R₄ && ijk₃ == ijk₄)
    f3 = (R₁ == R₃ && ijk₁ == ijk₃ && R₂ == R₄ && ijk₂ == ijk₄)

    uniquePairs = NTuple{4, Float64}[]
    sizehint!(uniquePairs, GN1*GN2*GN3*GN4)
    uPairCoeffs = Array{Float64}(undef, GN1*GN2*GN3*GN4)
    i = 0
    for p₁ in ps₁::NTuple{GN1, NTuple{2, Float64}}, 
        p₂ in ps₂::NTuple{GN2, NTuple{2, Float64}}, 
        p₃ in ps₃::NTuple{GN3, NTuple{2, Float64}}, 
        p₄ in ps₄::NTuple{GN4, NTuple{2, Float64}}
        pair = reformatIntData2(p₁[1], p₂[1], p₃[1], p₄[1], (f1, f2, f3))
        idx = findfirst(x->x==pair, uniquePairs)
        con = p₁[2] * p₂[2] * p₃[2] * p₄[2]
        if idx === nothing
            i += 1
            push!(uniquePairs, pair)
            uPairCoeffs[i] = con
        else
            uPairCoeffs[idx] += con
        end
    end
    map(uniquePairs, uPairCoeffs) do x, y
        getfield(Quiqbox, func)(optArgs..., R₁, ijk₁, x[1], R₂, ijk₂, x[2], 
                                            R₃, ijk₃, x[3], R₄, ijk₄, x[4])::Float64 * y
    end |> sum
end

getOverlap(bf1::FloatingGTBasisFuncs{<:Any, GN1, 1}, 
           bf2::FloatingGTBasisFuncs{<:Any, GN2, 1}) where {GN1, GN2} = 
getOneBodyInt(:∫overlapCore, bf1, bf2)

getElecKinetic(bf1::FloatingGTBasisFuncs{<:Any, GN1, 1}, 
               bf2::FloatingGTBasisFuncs{<:Any, GN2, 1}) where {GN1, GN2} = 
getOneBodyInt(:∫elecKineticCore, bf1, bf2)

function getNucAttraction(bf1::FloatingGTBasisFuncs{<:Any, GN1, 1}, 
                          bf2::FloatingGTBasisFuncs{<:Any, GN2, 1}, 
                          nuc::Vector{String}, 
                          nucCoords::Vector{<:AbstractArray{Float64}}) where {GN1, GN2}
    res = 0.0
    for (ele, coord) in zip(nuc, nucCoords)
        res += getOneBodyInt(:∫nucAttractionCore, bf1, bf2, getCharge(ele), coord)
    end
    res
end

function get2eInteraction(bf1::FloatingGTBasisFuncs, 
                          bf2::FloatingGTBasisFuncs, 
                          bf3::FloatingGTBasisFuncs, 
                          bf4::FloatingGTBasisFuncs)
    getTwoBodyInt(:∫eeInteractionCore, bf1, bf2, bf3, bf4)
end

getOverlap(b1::CompositeGTBasisFuncs{<:Any, 1}, b2::CompositeGTBasisFuncs{<:Any, 1}) = 
[getOverlap(bf1, bf2) for bf1 in unpackBasisFuncs(b1), bf2 in unpackBasisFuncs(b2)] |> sum

getOverlap(b1::CompositeGTBasisFuncs, b2::CompositeGTBasisFuncs) = 
[getOverlap(bf1, bf2) for bf1 in b1, bf2 in b2]