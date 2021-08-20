push!(LOAD_PATH, "./Quiqbox")
using Quiqbox
using Test

@testset "Tools.jl" begin
    points = [[-2.5, -2.5, -2.5], [-2.5, -2.5, 2.5], [-2.5, 2.5, -2.5], [-2.5, 2.5, 2.5], 
              [2.5, -2.5, -2.5], [2.5, -2.5, 2.5], [2.5, 2.5, -2.5], [2.5, 2.5, 2.5]]
    num = length(points)
    grid = GridBox(1, 5.0)
    
    @test hasEqual(grid, GridBox((1,1,1), 5.0))
    
    @test grid.num == num
    @test grid.len == 5.0
    @test grid.coord == points
    @test map(i-> [j() for j in i], grid.box) == points

    gPoints = gridPoint.(points) |> flatten
    @test [i[] for i in gPoints] == [i() for i in gPoints] == (points |> flatten)
end