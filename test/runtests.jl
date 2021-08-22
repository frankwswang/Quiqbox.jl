using Test

@testset "Quiqbox tests" begin

    unit1 = "Support Functions"
    println("Testing $(unit1)...")
    t1 = @elapsed @testset "$(unit1)" begin
        include("unit-tests/Tools-test.jl")
        include("unit-tests/FileIO-test.jl")
    end
    println("$(unit1) test finished in $t1 seconds.\n")

    @testset "Module Function tests" begin

        unit2_1 = "Core Functions"
        println("Testing $(unit2_1)...")
        t2_1 = @elapsed @testset "$(unit2_1)" begin
            include("unit-tests/Basis-test.jl")
            include("unit-tests/Molecule-test.jl")
            include("unit-tests/HartreeFock-test.jl")
            include("unit-tests/Optimization-test.jl")
        end
        println("$(unit2_1) test finished in $t2_1 seconds.\n")

        unit2_2 = "General Function"
        println("Testing $(unit2_2)...")
        t2_2 = @elapsed @testset "$(unit2_2)" begin
            include("unit-tests/Library-test.jl")
            include("unit-tests/Box-test.jl")
        end
        println("$(unit2_2) test finished in $t2_2 seconds.\n")

        unit2_3 = "Sub Modules"
        println("Testing $(unit2_3)...")
        t2_3 = @elapsed @testset "$(unit2_3)" begin
            include("unit-tests/SubModule/Coordinate-test.jl")
            include("unit-tests/SubModule/Molden-test.jl")
        end
        println("$(unit2_3) test finished in $t2_3 seconds.\n")

    end

end