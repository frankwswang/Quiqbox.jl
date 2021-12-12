# Reference: DOI: 10.1088/0143-0807/31/1/004
function getOverlapCore((R₁, α₁, d₁, ijk₁)::Tuple{Vector{Float64}, 
                                                  Float64, Float64, Vector{Int}}, 
                        (R₂, α₂, d₂, ijk₂)::Tuple{Vector{Float64}, 
                                                  Float64, Float64, Vector{Int}})
    α = α₁ + α₂
    res = d₁ * d₂ * (π/α)^1.5 * exp(-α₁ * α₂ / α * sum(abs2, R₁-R₂))
    for i in 1:3
        l₁, l₂ = ijk₁[i], ijk₂[i]
        res *= (-1.0)^(l₁) * factorial(l₁) * factorial(l₂) / α^(l₁+l₂)
        temp = 0
        for i₁ in 0:(l₁÷2), i₂ in 0:(l₂÷2)
            Ω = l₁ + l₂ - 2*(i₁ + i₂)
            for o in 0:(Ω÷2)
                temp += (-1)^o * factorial(Ω) * 
                        α₁^(l₂ - i₁ - 2i₂ - o) * 
                        α₂^(l₁ - i₂ - 2i₁ - o) * 
                         α^(2 * (i₁ + i₂) + o) * 
                        (R₁[i] - R₂[i])^(Ω-2o) / 
                        ( 4^(i₁ + i₂ + o) * 
                          factorial(i₁) * 
                          factorial(i₂) * 
                          factorial(o ) * 
                          factorial(l₁ - 2i₁) * 
                          factorial(l₂ - 2i₂) * 
                          factorial(Ω  - 2o ) )
            end
        end
        res *= temp
    end
    res
end

@inline function getOverlap(bf1::FloatingGTBasisFuncs{<:Any, GN1, 1}, 
                            bf2::FloatingGTBasisFuncs{<:Any, GN2, 1}) where {GN1, GN2}
    R₁, R₂ = (bf1, bf2) .|> centerCoordOf
    ijk₁, ijk₂ = get.(Ref(ijkOrbitalList), getindex.(getfield.((bf1, bf2), :ijk), 1), false)
    res = 0
    for gf1 in bf1.gauss::NTuple{GN1, GaussFunc}, gf2 in bf2.gauss::NTuple{GN2, GaussFunc}
        α₁ = gf1.xpn()
        α₂ = gf2.xpn()
        d₁ = gf1.con()
        d₂ = gf2.con()
        bf1.normalizeGTO && (d₁ *= getNijkα(ijk₁..., α₁))
        bf1.normalizeGTO && (d₁ *= getNijkα(ijk₂..., α₂))
        res += getOverlapCore((R₁, α₁, d₁, ijk₁), (R₂, α₂, d₂, ijk₂))
    end
    res
end

getOverlap(b1::CompositeGTBasisFuncs{<:Any, 1}, b2::CompositeGTBasisFuncs{<:Any, 1}) = 
[getOverlap(bf1, bf2) for bf1 in unpackBasisFuncs(b1), bf2 in unpackBasisFuncs(b2)] |> sum

getOverlap(b1::CompositeGTBasisFuncs, b2::CompositeGTBasisFuncs) = 
[getOverlap(bf1, bf2) for bf1 in b1, bf2 in b2]


function getNijkα(i, j, k, α)
    (2α/π)^0.75 * ( (8α)^(i+j+k) * factorial(i) * factorial(j) * factorial(k) / 
                (factorial(2i) * factorial(2j) * factorial(2k)) )^0.5
end