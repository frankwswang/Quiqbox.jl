using Test
using Quiqbox: tryIncluding, @compareLength, hasBoolRelation, markUnique, flatten, 
               alignSignedNum, symbolReplace, splitTerm
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

end