using Test
using Quiqbox: tryIncluding, @compareLength, hasBoolRelation, markUnique, flatten, 
               alignSignedNum, symbolReplace, splitTerm, groupedSort, mapPermute, Pf, itself
using Quiqbox
using Symbolics
using Suppressor: @capture_out

@testset "Tools.jl" begin

    # function tryIncluding
    local isIncluded
    errStr = @capture_out begin isIncluded = tryIncluding("someMod") end
    @test isIncluded == false
    pkgDir = @__DIR__
    @test length(errStr) > length(@capture_out tryIncluding("someMod", subModulePath=""))


    # macro @compareLength
    @test try @compareLength [1] [1,2] "a" "b"; catch; true end


    # function hasBoolRelation
    @test  hasBoolRelation(===, 1, 1) == (1 === 1)
    @test !hasBoolRelation(===, [1,2], [1,2])
    @test  hasBoolRelation(===, [1,2], [1,2], decomposeNumberCollection=true)
    @test !hasBoolRelation(===, [1,2], (1,2), decomposeNumberCollection=true)
    @test  hasBoolRelation(===, [1,2], (1,2), decomposeNumberCollection=true, 
                           ignoreContainer=true)
    @test !hasBoolRelation(===, [1 x->x^2], [1 x->abs(x)])
    @test  hasBoolRelation(===, [1 x->x^2], [1 x->abs(x)], ignoreFunction=true)


    # function markUnique
    @test markUnique([1,3,2,2,5]) == ([1,2,3,3,4], [1,3,2,5])


    # function flatten
    @test [(1,2), 3, [3,4]] |> flatten == [1,2,3,3,4]


    # function alignSignedNum
    @test alignSignedNum(-1) == "-1"
    @test alignSignedNum( 1) == " 1"


    # function symbolReplace
    @test symbolReplace(:sombol, "o"=>"y", count=1) == :symbol


    # function splitTerm
    Symbolics.@variables X Y Z
    vec0 = splitTerm(Num(1))
    @test ( string.(vec0) .== string.([Num(1)]) ) |> prod
    vec1 = splitTerm(X)
    @test ( string.(vec1) .== string.([X]) ) |> prod
    vec2 = splitTerm(X^2)
    @test ( string.(vec2) .== string.([X^2]) ) |> prod
    vec3 = splitTerm(X*Y*Z)
    @test ( string.(vec3) .== string.([X*Y*Z]) ) |> prod
    vec4 = splitTerm(X^2 - X*Y - Z^2)
    @test ( string.(vec4) .== string.([X^2, -(Z^2), -X*Y]) ) |> prod
    @test ([vec0, vec1, vec2, vec3, vec4] .|> length) == [1,1,1,1,3]
    @test ([vec0, vec1, vec2, vec3, vec4] |> eltype) ==  Vector{Num}


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
        bs1 = genBasisFunc.([[rand(1:2), rand(1:3), rand(1:3)] for i=1:20], 
                               [(rand(0.5:0.5:2.5), rand(-0.1:0.2:0.5)) for i=1:20])
        bs2 = genBasisFunc.([[rand(1:2), rand(1:3), rand(1:3)] for i=1:20], 
                            [GaussFunc(   Exponent(rand( 0.5:0.5:2.5), mapPool[rand(1:3)]), 
                                       Contraction(rand(-0.1:0.2:0.5), mapPool[rand(1:3)])) 
                             for i=1:20])
        bl1 *= (sortperm(arr) == mapPermute(arr, sort))
        bl2 *= hasIdentical(bs1[mapPermute(bs1, sortBasisFuncs)], sortBasisFuncs(bs1))
        bl3 *= hasIdentical(bs2[mapPermute(bs2, sortBasisFuncs)], sortBasisFuncs(bs2))
    end
    @test bl1
    @test bl2
    @test bl3

    # struct Pf
    @test Pf(-1.5, abs)(-2) == -3.0
    @test Pf(-1.0, Pf(-1.5, abs))(-2) == 3.0
    @test Pf(1.5, Val(:abs))(-2) == 3.0
    @test Pf(-1.0, Val(Pf{-1.5, :abs}))(-2) == 3.0
    @test typeof(Pf(-1.5, abs))(-2) == -3.0

    @test Pf(-1.0, Pf(-1.5, itself))(-2) == -3.0
    @test Pf(1.5, Val(:itself))(-2) == -3.0
    @test Pf(-1.0, Val(Pf{-1.5, :itself}))(-2) == -3.0
    @test typeof(Pf(-1.5, itself))(-2) == 3.0
end