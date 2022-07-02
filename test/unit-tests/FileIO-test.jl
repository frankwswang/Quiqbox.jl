using Test
using Quiqbox
using Quiqbox: checkFname, advancedParse, numToSups, superscriptNum, numToSubs, subscriptNum

@testset "FileIO.jl tests" begin

# function checkFname
suffix = rand(100000000:999999999)
testFname = "__test__"*string(suffix)
@test testFname == checkFname(testFname)
open(testFname, "w") do io end
nFname = checkFname(testFname, showWarning=false)
@test nFname == testFname*"_N"
rm(testFname)


# function advancedParse (with adaptiveParse)
@test advancedParse(Float64, "1") === 1.0
@test advancedParse(Float64, "1.0") === 1.0
@test advancedParse(Float64, "1.0 + im") === "1.0 + im"
@test advancedParse(Float64, "1 + 1im") === 1.0 + 1.0im
@test advancedParse(Float64, "1.0 + 1.1im") === 1.0 + 1.1im
@test advancedParse(BigFloat, "1.0") isa BigFloat
@test advancedParse(BigFloat, "1.0") == BigFloat(1.0)


# Function numToSups & numToSubs
ds = [superscriptNum, subscriptNum]
fs = [numToSups, numToSubs]
for (d,f) in zip(ds, fs)
    bl = true
    for i = 0:9
        bl *= (f(i) == d['0' + i]|>string)
    end
    @test bl
    num = rand(100000000:999999999)
    str = ""
    for i in num |> string
        str *= d[i]
    end
    @test str == f(num)
end

end