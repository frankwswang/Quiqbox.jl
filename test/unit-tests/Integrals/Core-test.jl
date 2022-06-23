using Test
using Quiqbox: Fγ
using QuadGK: quadgk

@testset "Core.jl tests" begin

tolerance1 = 1e-8
tolerance2 = 5e-8
perturbStep = rand(-1e-1:2e-3:1e-1)
fNumInt = (γ, u) -> quadgk(t -> t^(2γ)*exp(-u*t^2), 0, 1; rtol=tolerance1)[1]
range = -10:(0.2+perturbStep):2
for γ in 0:24
    @test all([isapprox(fNumInt(γ, 10.0^e), Fγ(γ, 10.0^e), atol=tolerance2) 
                for e in range])
end

end