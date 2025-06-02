using Test
using Quiqbox
using Quiqbox: getAtolVal, getAtolDigits, roundToMultiOfStep, nearestHalfOf, getNearestMid, 
               isApprox, sizeOf, markUnique, getUnique!, 
               itself, themselves, replaceSymbol, groupedSort, nameOf, tupleDiff, fillObj, 
               arrayToTuple, genTupleCoords, uniCallFunc, mergeMultiObjs, isNaN, getBool, 
               skipIndices, isOscillateConverged, lazyCollect, asymSign, numEps, 
               genAdaptStepBl, shiftLastEle!, getValParm, fct, triMatEleNum, 
               convertIndex1DtoTri2D, convertIndex1DtoTri4D, mapMapReduce, rmsOf, keepOnly!
using LinearAlgebra: norm
using Random

@testset "Tools.jl" begin

# function isApprox
v1 = 1/3 + 1e-16
v2 = 1/3
@test isApprox(v1, v2)
@test !isApprox(v1, v2, atol=NaN)


#function sizeOf
nr = rand(1:4)
nc = rand(2:5)
@test sizeOf(rand(nr, nc)) == (nr, nc)
@test sizeOf((rand(nc, nr)...,)) == (nr*nc,)


# function markUnique
@test markUnique([1,3,2,2,5]) == ([1,2,3,3,4], [1,3,2,5])
emptyArr = Int[]
res1, res2 = markUnique(emptyArr)
res1 == emptyArr == res2
@test markUnique((1,3,2,2,5)) == ((1,2,3,3,4), [1,3,2,5])
@test markUnique(()) == ((), Union{}[])
@test markUnique((3,)) == ((1,), [3])
markList, cmprList = markUnique(Memory{Int}([3]))
@test markList isa Memory{Int}
@test cmprList isa Vector{Int}
@test (markList, cmprList) == (Memory{Int}([1]), [3])


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
for i in firstindex(a2)+1:lastindex(a2)
    if length(a2[i]) > 1
        h, b... = a2[i]
        for ele in b
          @test isequal(h, ele)
        end
    end
    @test a2[i][1][1] >= a2[i-1][1][1]
end


# function tupleDiff
a = (1,1,2,2,3)
b = (3,2,2,1,3)
c = (4,2,1,3,2)
d = (2,4,5,1,3)
@test tupleDiff(a,b) == ([1, 2, 2, 3], [1], [3])
@test tupleDiff(a,b,c,d) == ([1, 2, 3], [1,2], [2,3], [4,2], [4,5])


# function fillObj
v0 = fill(1)
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


# function uniCallFunc
@test uniCallFunc(abs, (1,), -4,2,3) == 4


# function mergeMultiObjs
mergeFunc1 = (x, y; atol) -> ifelse(isapprox(abs(x), abs(y); atol), [abs(x)], abs.([x, y]))
arr = [-1, -1 + 1e-11, -2, 2, 3, -2, 4, -3]
@test mergeMultiObjs(Float64, mergeFunc1, arr, atol=1e-10) == collect(1.0:4.0)


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


# function isOscillateConverged
Random.seed!(1234)
shift = x-> x+0.01
convVal = 1.215
convAtol = 1e-6
y1 = x -> (1-abs(cos(5x)/(0.7(0.5x+2))))/(shift(3x)+1)^1.25 + convVal + 2e-4randn()/(x+1)
y2 = x -> log(shift(x)+x)/(shift(3x)+1)^1.5 + convVal + 2e-4randn()/(x+1)
y3 = x -> [y1(x), y2(x)]

xs = collect(0:0.01:100)
y1s = y1.(xs)
y3s = y2.(xs)

y1s = Float64[]
convRes1 = []
for x in collect(0:0.01:1000)
    push!(y1s, y1(x))
    bl, resStd = isOscillateConverged(y1s, 1e-7)
    if bl
        @test isapprox(y1(x), convVal, atol=7.5e-4)
        push!(convRes1, bl, resStd)
        break
    end
end
@test convRes1[begin]
@test convRes1[end] < convAtol

y3s = Vector{Float64}[]
convRes2 = []
for x in collect(0:100:5000)
    push!(y3s, y3(x))
    bl, resStd = isOscillateConverged(y3s, 1e-5, convVal)
    if bl
        @test all(isapprox.(y3(x), convVal, atol=1e-4))
        push!(convRes2, bl, resStd)
        break
    end
end
@test convRes2[begin]
@test norm(convRes2[end]) < convAtol*(sqrt∘length)(convRes2[end])


# function lazyCollect
tpl1 = (1,2,3)
arr1 = collect(tpl1)
@test lazyCollect(1) == fill(1)
@test lazyCollect(1:3) == arr1 == collect(1:3)
@test lazyCollect(1:3) !== arr1
@test lazyCollect(tpl1) == arr1
@test lazyCollect(tpl1) !== arr1
@test lazyCollect(arr1) === arr1


# function asymSign
@test asymSign(0) == 1
@test asymSign(1.1) == 1
@test asymSign(-1.1) == -1


# struct FuncArgConfig
fWrapper1 = Quiqbox.FuncArgConfig(isapprox, [20, 20.0005], [:atol=>1e-3])
@test fWrapper1(isapprox)

fWrapper2 = Quiqbox.FuncArgConfig(isapprox, [20, 20+2e-12])
@test fWrapper2(isapprox)

foo1 = ()->true
fWrapper3 = Quiqbox.FuncArgConfig(foo1)
@test fWrapper3(foo1)


# function numEps
@test numEps(Float64) == eps(Float64)
@test numEps(Int64) == one(Int64)
@test numEps(Complex{Float64}) == eps(Float64)


# function genAdaptStepBl
countBl = function (l, N)
    j = 0
    for i = 0:N
        f = genAdaptStepBl(l, N)
        f(i) && (j+=1)
    end
    j
end
maxStep = rand(-1000:1000)
res = hcat([countBl.(collect(0:6), N) for N in collect(0:sign(maxStep):maxStep)]...)
@test all(res[1, :] .== 0)
@test all(sort(c) == c for c in eachcol(res))


# function shiftLastEle!
vect = rand(5)
vectBackup = deepcopy(vect)
shiftVal = rand()
s, signedShift = shiftLastEle!(vect, shiftVal)
@test isapprox(s, sum(vect), atol=2e-15)
residuum =  vect - vectBackup
@test all(residuum[1:4] .== 0)
@test isapprox(residuum[end], signedShift, atol=2e-15)
@test abs(s) >= (abs∘sum)(vectBackup)


# function getValParm
vNum = rand()
v = Val(vNum)
@test getValParm(v) == getValParm(typeof(v)) == vNum


# function triMatEleNum
@test triMatEleNum(6) == 21


# function convertIndex1DtoTri2D, convertIndex1DtoTri4D
test2DIndexing = function (n::Int)
    m = 0
    bl = true
    for j = 1:n, i=1:j
        m += 1
        r1 = (i, j)
        r2 = convertIndex1DtoTri2D(m)
        if r1 != r2
            bl *= false
        end
    end
    @test bl
    m
end

test4DIndexing = function (n::Int)
    m = 0
    bl = true
    for l = 1:n, k = 1:l, j = 1:l, i = 1:ifelse(l==j, k, j)
        m += 1
        r1 = (i, j, k, l)
        r2 = convertIndex1DtoTri4D(m)
        if r1 != r2
            bl *= false
        end
    end
    @test bl
    m
end

for i in (1, 3, 5, 10, 17)
    j = i * (i + 1) ÷ 2
    @test test2DIndexing(i) == j
    @test test4DIndexing(i) == j * (j + 1) ÷ 2
end


# function mapMapReduce
tp = (1, 2, -3)
f1 = x->x^2
@test mapMapReduce(tp, f1) == 36
@test mapMapReduce(tp, f1, +) == 14
@test mapMapReduce(tp, (itself, f1, abs)) == 12
@test mapMapReduce(tp, (f1, itself, abs), +) == 6

# function rmsOf
rmsTest = arr -> sqrt( sum(arr .^ 2) ./ length(arr) )
arr3 = rand(100)
isapprox(rmsTest(arr3), rmsOf(arr3), atol=5e-15)

# function keepOnly!
arr4 = collect(1:5)
idx2 = 3
res3 = keepOnly!(arr4, idx2)
@test arr4[] == res3 == idx2

arr5 = rand(1)
ele = arr5[]
res4 = keepOnly!(arr5, 1)
@test ele == res4

# function intersectMultisets!
a = [1, 3, 2, 1, 3, 1, 1, 3, 3, 3]
b = [1, 2, 3, 3, 3, 3, 3, 1, 2, 1]

a2 = copy(a)
is1 = Quiqbox.intersectMultisets!(a2, a2)
@test is1 == a
@test isempty(a2)

a3 = copy(a)
a4 = copy(a)
is2 = Quiqbox.intersectMultisets!(a3, a4)
@test is2 == a
@test isempty(a3) && isempty(a4)

a5 = copy(a)
b2 = copy(b)
is3 = Quiqbox.intersectMultisets!(a5, b2)
is3t = intersect(a5, b2)
@test isempty(is3t)
@test eltype(is3t) == Int

removeEles! = function (ms::AbstractVector{T}, subms::AbstractVector{T}) where {T}
    for i in subms
        idx = findfirst(isequal(i), ms)
        popat!(ms, idx)
    end
    ms
end

@test removeEles!(copy(a), is3) == a5
@test removeEles!(copy(b), is3) == b2

a6 = copy(a)
b3 = copy(b)
is4 = Quiqbox.intersectMultisets!(b3, a6)
@test is3 != is4
@test sort(is3) == sort(is4)

end