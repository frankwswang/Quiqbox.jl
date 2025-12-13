using Test
using Quiqbox: checkFname, TypePiece, True

@testset "IO.jl tests" begin

#> `checkFname`
suffix = rand(100000000:999999999)
testFname = "__test__"*string(suffix)
@test testFname == checkFname(testFname)
open(testFname, "w") do io end
nFname = checkFname(testFname, showWarning=false)
@test nFname == testFname*"_N"
rm(testFname)

@test sprint(show, TypePiece(True)) == "(::Quiqbox.TypePiece{Quiqbox.True})"

end