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
@test grid.spacing == (3.0, 3.0, 3.0)
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

L1 = fill(1.4)
grid3 = GridBox((1,2,2), L1)
grid4 = GridBox((1,2,2), (1.4, 1.4, 1.4))
grid5 = GridBox((1,2,2), (1.4, L1, L1))
@test hasEqual(grid3, grid4, grid5)
pars3 = markParams!(grid3.point, true)
pars4 = markParams!(grid4.param, true)
pars5 = markParams!(grid5.point, true)
@test length(pars3) == 1
@test length(pars4) == 3
@test length(pars5) == 2
@test hasEqual(pars3[1], pars4[1], pars5[1])
@test hasEqual(pars4[2], pars5[2], changeMapping(pars4[3], pars4[3].map, :Y))
@test pars3[].data[] === pars5[2].data[]

grid6 = GridBox((1,2,0), L1)
grid7 = GridBox((1,2,0), (1.4, 1.4, L1))
grid8 = GridBox((1,2,0), (L1, 1.4, L1))
@test hasEqual(grid6, grid7, grid8)
pars6 = markParams!(grid6.point, true)
pars7 = markParams!(grid7.point, true)
pars8 = markParams!(grid8.point, true)
@test hasEqual(pars5[1], pars6[1], pars7[1], pars8[1])
@test hasEqual(pars5[2], pars7[2], pars8[2])
@test pars6[2][] == pars7[3][] == pars8[3][] == 0.0
@test hasEqual(pars6[2], pars7[3])
@test !hasIdentical(pars7[3], pars8[3]) && hasEqual(pars7[3], pars8[3])
@test hasIdentical(pars6[1], pars8[1])

end