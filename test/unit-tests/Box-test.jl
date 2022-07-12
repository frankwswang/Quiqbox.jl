using Test
using Quiqbox

@testset "Tools.jl" begin

points = [[-3.0, -3.0, -3.0], [ 0.0, -3.0, -3.0], [ 3.0, -3.0, -3.0], [-3.0,  0.0, -3.0], 
          [ 0.0,  0.0, -3.0], [ 3.0,  0.0, -3.0], [-3.0,  3.0, -3.0], [ 0.0,  3.0, -3.0], 
          [ 3.0,  3.0, -3.0], [-3.0, -3.0,  0.0], [ 0.0, -3.0,  0.0], [ 3.0, -3.0,  0.0], 
          [-3.0,  0.0,  0.0], [ 0.0,  0.0,  0.0], [ 3.0,  0.0,  0.0], [-3.0,  3.0,  0.0], 
          [ 0.0,  3.0,  0.0], [ 3.0,  3.0,  0.0], [-3.0, -3.0,  3.0], [ 0.0, -3.0,  3.0], 
          [ 3.0, -3.0,  3.0], [-3.0,  0.0,  3.0], [ 0.0,  0.0,  3.0], [ 3.0,  0.0,  3.0], 
          [-3.0,  3.0,  3.0], [ 0.0,  3.0,  3.0], [ 3.0,  3.0,  3.0]]
num = length(points)
grid = GridBox(2, 3.0)

@test hasEqual(grid, GridBox((2,2,2), 3.0))

@test grid.nPoint == num
@test grid.spacing == 3.0
@test gridCoords(grid) == points
@test map(i-> [j() for j in i], grid.box) == Tuple(points)

gPoints = getproperty.(getproperty.(genSpatialPoint.(points), :point), :param) |> flatten
@test [i[] for i in gPoints] == [i() for i in gPoints] == (points |> flatten)

@test gridCoords( GridBox((1,2), 2.0, [1.0, 1.0]) ) == 
      [[ 0.0, -1.0], [ 2.0, -1.0], [ 0.0,  1.0], [ 2.0,  1.0], [ 0.0,  3.0], [ 2.0,  3.0]]
@test all( collect(getproperty.(GridBox((0,3), 1.0).param, :map))[isodd.(1:end)] .== 
           Quiqbox.itself )
end