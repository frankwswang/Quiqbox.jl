using Test
using Quiqbox
using Quiqbox: alignSignedNum
using Symbolics

@testset "Tools.jl" begin
    @test markUnique([1,3,2,2,5]) == ([1,2,3,3,4], [1,3,2,5])

    @test [(1,2), 3, [3,4]] |> flatten == [1,2,3,3,4]

    @test alignSignedNum(-1) == "-1"
    @test alignSignedNum( 1) == " 1"

    # Symbolic representation related functions
    Symbolics.@variables X Y Z
    vec0 = splitTerm(Num(1))
    @test ( string.(vec0) .== string.([Num(1)]) ) |> prod
    vec1 = splitTerm(X)
    @test ( string.(vec1) .== string.([X]) ) |> prod
    vec2 = splitTerm(X^2)
    @test ( string.(vec2) .== string.([X^2]) ) |> prod
    vec3 = splitTerm(X*Y*Z)
    @test ( string.(vec3) .== string.([X*Y*Z]) ) |> prod
    vec4 = splitTerm(X^2 - X*Y - Z^2)
    @test ( string.(vec4) .== string.([X^2, -(Z^2), -X*Y]) ) |> prod
    @test ([vec0, vec1, vec2, vec3, vec4] .|> length) == [1,1,1,1,3]
    @test ([vec0, vec1, vec2, vec3, vec4] |> eltype) ==  Array{Num, 1}
end