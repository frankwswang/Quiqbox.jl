push!(LOAD_PATH, "./Quiqbox")
using Quiqbox
using Quiqbox: alignSignedNum
using Test

@testset "Tools.jl" begin
    @test markUnique([1,3,2,2,5]) == ([1,2,3,3,4], [1,3,2,5])

    @test [(1,2), 3, [3,4]] |> flatten == [1,2,3,3,4]

    @test alignSignedNum(-1) == "-1"
    @test alignSignedNum( 1) == " 1"
end