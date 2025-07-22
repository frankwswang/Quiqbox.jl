using Test
using Quiqbox
using Quiqbox: checkFname

@testset "IO.jl tests" begin

# function checkFname
suffix = rand(100000000:999999999)
testFname = "__test__"*string(suffix)
@test testFname == checkFname(testFname)
open(testFname, "w") do io end
nFname = checkFname(testFname, showWarning=false)
@test nFname == testFname*"_N"
rm(testFname)

end