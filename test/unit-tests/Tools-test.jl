using Test
using Quiqbox
using Quiqbox: getAtolVal, getAtolDigits, roundToMultiOfStep, nearestHalfOf, getNearestMid, 
               isApprox, tryIncluding, sizeOf, hasBoolRelation, flatten, joinTuple, 
               markUnique, getUnique!, itself, themselves, replaceSymbol, groupedSort, 
               mapPermute, getFunc, nameOf, tupleDiff, genIndex, fillObj, arrayToTuple, 
               genTupleCoords, uniCallFunc, mergeMultiObjs, isNaN, getBool, skipIndices
using Suppressor: @capture_out

@testset "Tools.jl" begin

# function getAtolVal getAtolDigits
@test getAtolVal(Float64) == 4e-16
@test getAtolDigits(Float64) == 15


# function roundToMultiOfStep nearestHalfOf getNearestMid
@test roundToMultiOfStep(3811.47123123, 0.01) == 3811.47
@test roundToMultiOfStep(0.1+0.2, 1e-17) == 0.30000000000000004
@test roundToMultiOfStep(0.1+0.2, 1e-16) == 0.3

@test nearestHalfOf(0.1 + 0.2) == 0.15
@test getNearestMid(0.1, 0.2, 1e-16) == 0.15
@test getNearestMid(0.1, 0.2, 1e-17) == (0.1 + 0.2)/2


# function isApprox
v1 = 1/3 + 1e-16
v2 = 1/3
@test isApprox(v1, v2)
@test !isApprox(v1, v2, atol=NaN)


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
@test hasBoolRelation(==, Int, Int)
@test hasBoolRelation(<:, Int, Integer)

# function hasEqual
@test hasEqual([1,2], [1.0, 2.0])

# function hasEqual
@test !hasIdentical([1,2], [1, 2])
@test  hasIdentical([1,2], [1, 2], decomposeNumberCollection=true)
@test !hasIdentical([1,2], [1.0, 2], decomposeNumberCollection=true)
@test !hasIdentical([fill(1), :two], [fill(1), :two])
v0 = fill(1)
@test hasIdentical([v0, :two], [v0, :two])

# function hasApprox
@test  hasApprox(1, 1+5e-16)
@test !hasApprox(1, 1+5e-15)
@test !hasApprox([1,2], [3])


# function flatten
@test [(1,2), (3,3,4)] |> flatten == [1,2,3,3,4]
@test [(1,2), 3, [3,4]] |> flatten == [1,2,3,3,4]


# function joinTuple
@test joinTuple((1,2), (3,4)) == (1,2,3,4)
@test joinTuple((1,2,3,4)) == (1,2,3,4)


# function markUnique
@test markUnique([1,3,2,2,5]) == ([1,2,3,3,4], [1,3,2,5])
emptyArr = Int[]
res1, res2 = markUnique(emptyArr)
res1 === emptyArr == res2


# function getUnique!
arr = rand(1:3, 10)
arr2 = unique(arr)
getUnique!(arr)
@test arr == arr2


#function itself, themselves
x1 = rand(10)
@test itself(x1) === x1 === identity(x1)
@test themselves(x1, x1) === (x1, x1)


# function replaceSymbol
@test replaceSymbol(:sombol, "o"=>"y", count=1) == :symbol


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


# function getFunc
@test getFunc(abs) == abs
@test getFunc(:abs) == abs
@test getFunc(:getTwoBodyInts) == Quiqbox.getTwoBodyInts
@test getFunc(:f1_getFunc)(rand()) isa Missing
@test getFunc(Symbol("x->x^2"))(3) == 3^2


# function nameOf
pf1 = Quiqbox.PF(abs, *, -1.5)
@test nameOf(abs) == :abs
@test nameOf(pf1) == typeof(pf1) == Quiqbox.PF{typeof(abs), typeof(*), Float64}


# function tupleDiff
a = (1,1,2,2,3)
b = (3,2,2,1,3)
c = (4,2,1,3,2)
d = (2,4,5,1,3)
@test tupleDiff(a,b) == ([1, 2, 2, 3], [1], [3])
@test tupleDiff(a,b,c,d) == ([1, 2, 3], [1,2], [2,3], [4,2], [4,5])


# function genIndex
@test genIndex(1) == fill(1)
@test genIndex(nothing) == fill(nothing)


# function fillObj
@test fillObj(v0) === v0
@test fill(1) == v0


# function arrayToTuple
@test arrayToTuple(arr) == Tuple(arr) == arrayToTuple(arr|>Tuple)


# function genTupleCoords
c1 = [rand(3) for _=1:3]
c2 = c1 |> Tuple
c3 = c1 .|> Tuple
c4 = c3 |> Tuple
@test c4 == genTupleCoords(Float64, c1)
@test c4 == genTupleCoords(Float64, c2)
@test c4 == genTupleCoords(Float64, c3)
@test c4 == genTupleCoords(Float64, c4)


# # function callGenFunc
# @test callGenFunc(f1, -1) == 1


# function uniCallFunc
@test uniCallFunc(abs, (1,), -4,2,3) == 4


# function mergeMultiObjs
mergeFunc1 = (x,y; atol) -> ifelse(isapprox(abs(x), abs(y); atol), [abs(x)], abs.([x, y]))
@test mergeMultiObjs(Int, mergeFunc1, -1, -2, 2, 3, -2, -1, 4, -3, atol=1e-10) == [1,2,3,4]


# function isNaN
@test !isNaN(42)
@test  isNaN(NaN)
@test !isNaN("NaN")


# function getBool
@test getBool(true)
@test getBool(Val(true))


# function skipIndices
idsR1 = [3, 6, 7, 11, 12, 15, 18]
idsR2 = [3, 6, 11, 7, 15, 12, 18]
idsT = [1, 2, 6, 5, 6, 7, 8, 4, 9, 10, 3]
@test skipIndices(idsT, idsR1) == skipIndices(idsT, idsR2) == 
      [1, 2, 9, 8, 9, 10, 13, 5, 14, 16, 4]
@test skipIndices(idsT, Int[]) === idsT
@test try skipIndices(idsT, [-1, 2]) catch; true end
@test try skipIndices([-1, 2, 3], idsR1) catch; true end

end