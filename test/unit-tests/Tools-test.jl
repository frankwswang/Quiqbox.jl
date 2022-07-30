using Test
using Quiqbox: tryIncluding, sizeOf, hasBoolRelation, markUnique, flatten, alignNumSign, 
               replaceSymbol, renameFunc, groupedSort, mapPermute, nameOf, TypedFunction, 
               Pf, itself, getFunc

using Quiqbox
using Suppressor: @capture_out

@testset "Tools.jl" begin

# function tryIncluding
local isIncluded
errStr = @capture_out begin isIncluded = tryIncluding("someMod") end
@test isIncluded == false
pkgDir = @__DIR__
@test length(errStr) > length(@capture_out tryIncluding("someMod", subModulePath=""))


#function sizeOf
nr = rand(1:4)
nc = rand(2:5)
@test sizeOf(rand(nr, nc)) == (nr, nc)
@test sizeOf((rand(nc, nr)...,)) == (nr*nc,)


# function hasBoolRelation
@test  hasBoolRelation(===, 1, 1) == (1 === 1)
@test !hasBoolRelation(===, [1,2], [1,2])
@test  hasBoolRelation(===, [1,2], [1,2], decomposeNumberCollection=true)
@test !hasBoolRelation(===, [1,2], (1,2), decomposeNumberCollection=true)
@test  hasBoolRelation(===, [1,2], (1,2), decomposeNumberCollection=true, 
                       ignoreContainer=true)
@test !hasBoolRelation(===, [1 x->x^2], [1 x->abs(x)])
@test  hasBoolRelation(===, [1 x->x^2], [1 x->abs(x)], ignoreFunction=true)

# function hasEqual
@test hasEqual([1,2], [1.0, 2.0])

# function markUnique
@test markUnique([1,3,2,2,5]) == ([1,2,3,3,4], [1,3,2,5])


# function flatten
@test [(1,2), 3, [3,4]] |> flatten == [1,2,3,3,4]


# function alignNumSign
@test alignNumSign(-1) == "-1"
@test alignNumSign( 1) == " 1"


# function replaceSymbol
@test replaceSymbol(:sombol, "o"=>"y", count=1) == :symbol


# function renameFunc
f1 = renameFunc(:f_renameFunc_1, x->abs(x))
@test f1(-0.1) === 0.1
@test f1(-1) === 1
@test nameof(f1) == :f_renameFunc_1

f2 = renameFunc(:f_renameFunc_2, +, 2)
arg1 = rand()
arg2 = rand()
@test f2(arg1, arg2) === (arg1 + arg2)
@test try f2(1, arg1, arg2) catch; true end

f3 = renameFunc(:f_renameFunc_3, abs, Float64)
@test f3(-0.1) === 0.1
@test try f3(-1) catch; true end
@test typeof(f3) != typeof(abs)

f4 = renameFunc("f_renameFunc_3", x->x+1, Float64)
nameof(f4) == :f_renameFunc_3
@test f3(-0.1) === 0.9


# function groupedSort
a = [rand(1:5, 2) for i=1:20]
a2 = groupedSort(a)
pushfirst!(a2, [[0,0]])
for i in 2:length(a2)
    if length(a2[i]) > 1
        @test hasEqual(a2[i]...)
    end
    @test a2[i][1][1] >= a2[i-1][1][1]
end


# function mapPermute
bl1 = true
bl2 = true
bl3 = true
mapPool = [x->x^2, abs, x->-x]
for i=1:10
    arr = rand(50)
    bs1 = genBasisFunc.([Float64[rand(1:2), rand(1:3), rand(1:3)] for i=1:20], 
                        [(rand(0.5:0.5:2.5), rand(-0.1:0.2:0.5)) for i=1:20])
    bs2 = genBasisFunc.([Float64[rand(1:2), rand(1:3), rand(1:3)] for i=1:20], 
                        [GaussFunc(genExponent(rand(0.5:0.5:2.5), mapPool[rand(1:3)]), 
                                   genContraction(rand(-0.1:0.2:0.5), mapPool[rand(1:3)])) 
                         for i=1:20])
    bl1 *= (sortperm(arr) == mapPermute(arr, sort))
    bl2 *= hasIdentical(bs1[mapPermute(bs1, sortBasisFuncs)], sortBasisFuncs(bs1))
    bl3 *= hasIdentical(bs2[mapPermute(bs2, sortBasisFuncs)], sortBasisFuncs(bs2))
end
@test bl1
@test bl2
@test bl3


# struct TypedFunction
tf1 = TypedFunction(abs)
@test nameOf(tf1) == nameOf(abs) == nameof(abs)
v = -abs(rand())
@test tf1(v) === abs(v)


# struct Pf
pf1 = Pf(-1.5, abs)
@test pf1(-2.0) == -3.0
@test nameOf(pf1) == typeof(pf1)
pf2 = Pf(-1.5, abs)
@test pf2(-2) == Pf(-1.5, tf1)(-2) == -3.0
@test Pf(-1.0, pf2)(-2.0) == 3.0
@test Pf(-1.0, Pf(-1.5, itself))(-2) == -3.0


# function getFunc
@test getFunc(abs) == abs
tff = nameOf(tf1) |> getFunc
@test tff == tf1.f == getFunc(tf1) == abs
@test getFunc(:abs) == abs
@test getFunc(:getTwoBodyInts) == Quiqbox.getTwoBodyInts


# function getAtolVal
@test Quiqbox.getAtolVal(Float64) == 1e-15

end