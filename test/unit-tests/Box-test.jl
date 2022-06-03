using Test
using Quiqbox

@testset "Tools.jl" begin

points = [[-3.0, -3.0, -3.0], [-3.0, -3.0, 0.0], [-3.0, -3.0, 3.0], 
          [-3.0,  0.0, -3.0], [-3.0,  0.0, 0.0], [-3.0,  0.0, 3.0], 
          [-3.0,  3.0, -3.0], [-3.0,  3.0, 0.0], [-3.0,  3.0, 3.0], 
          [ 0.0, -3.0, -3.0], [ 0.0, -3.0, 0.0], [ 0.0, -3.0, 3.0], 
          [ 0.0,  0.0, -3.0], [ 0.0,  0.0, 0.0], [ 0.0,  0.0, 3.0], 
          [ 0.0,  3.0, -3.0], [ 0.0,  3.0, 0.0], [ 0.0,  3.0, 3.0], 
          [ 3.0, -3.0, -3.0], [ 3.0, -3.0, 0.0], [ 3.0, -3.0, 3.0], 
          [ 3.0,  0.0, -3.0], [ 3.0,  0.0, 0.0], [ 3.0,  0.0, 3.0], 
          [ 3.0,  3.0, -3.0], [ 3.0,  3.0, 0.0], [ 3.0,  3.0, 3.0]]
num = length(points)
grid = GridBox(2, 3.0)

@test hasEqual(grid, GridBox((2,2,2), 3.0))

@test grid.num == num
@test grid.spacing == 3.0
@test gridCoords(grid) == points
@test map(i-> [j() for j in i], grid.box) == Tuple(points)

gPoints = getfield.(getfield.(genSpatialPoint.(points), :point), :param) |> flatten
@test [i[] for i in gPoints] == [i() for i in gPoints] == (points |> flatten)

end