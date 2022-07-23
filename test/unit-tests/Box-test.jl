using Test
using Quiqbox

@testset "Tools.jl" begin

points = ([-3.0, -3.0, -3.0], [ 0.0, -3.0, -3.0], [ 3.0, -3.0, -3.0], [-3.0,  0.0, -3.0], 
          [ 0.0,  0.0, -3.0], [ 3.0,  0.0, -3.0], [-3.0,  3.0, -3.0], [ 0.0,  3.0, -3.0], 
          [ 3.0,  3.0, -3.0], [-3.0, -3.0,  0.0], [ 0.0, -3.0,  0.0], [ 3.0, -3.0,  0.0], 
          [-3.0,  0.0,  0.0], [ 0.0,  0.0,  0.0], [ 3.0,  0.0,  0.0], [-3.0,  3.0,  0.0], 
          [ 0.0,  3.0,  0.0], [ 3.0,  3.0,  0.0], [-3.0, -3.0,  3.0], [ 0.0, -3.0,  3.0], 
          [ 3.0, -3.0,  3.0], [-3.0,  0.0,  3.0], [ 0.0,  0.0,  3.0], [ 3.0,  0.0,  3.0], 
          [-3.0,  3.0,  3.0], [ 0.0,  3.0,  3.0], [ 3.0,  3.0,  3.0])
num = length(points)
grid = GridBox(2, 3.0)

@test hasEqual(grid, GridBox((2,2,2), 3.0))

@test grid.nPoint == num
@test grid.spacing == 3.0
@test gridCoordOf(grid) == coordOf.(grid.point) == points
@test map(i-> [j() for j in i], grid.point) == points

gPoints = getproperty.(genSpatialPoint.(points), :param) |> flatten
@test [i[] for i in gPoints] == [i() for i in gPoints] == (points |> flatten |> collect)

@test gridCoordOf( GridBox((1,2), 2.0, [1.0, 1.0]) ) == 
      ([ 0.0, -1.0], [ 2.0, -1.0], [ 0.0,  1.0], [ 2.0,  1.0], [ 0.0,  3.0], [ 2.0,  3.0])
@test all( collect(getproperty.(GridBox((0,3), 1.0).param, :map))[isodd.(1:end)] .== 
           Quiqbox.itself )

grid2 = GridBox((1,0,0), 1.4)
@test gridCoordOf(grid2) == ([-0.7, 0.0, 0.0], [0.7, 0.0, 0.0])

point1, point2 = grid2.point
for i in point1
    i[] = rand()
end
point1Coord, point2Coord = coordOf.(grid2.point)
@test point1Coord[1] == -point2Coord[1]
@test point1Coord[2:3] == point2Coord[2:3]

pointSource = genSpatialPoint([0.0, point2Coord[2:3]...])
point3 = genSpatialPoint([point1[1], pointSource[2:3]...])
point4 = genSpatialPoint([point2[1], pointSource[2:3]...])
@test hasEqual(grid2.point, (point3, point4))

end