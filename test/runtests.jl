using Test

@testset "Quiqbox tests" begin

    unit1 = "Support Functions"
    println("Testing $(unit1)...")
    t1 = @elapsed @testset "$(unit1)" begin
        include("unit-tests/Tools-test.jl")
        include("unit-tests/FileIO-test.jl")
        include("unit-tests/Overload-test.jl")
    end
    println("$(unit1) test finished in $t1 seconds.\n")

    @testset "Module Functions" begin

        unit2_1 = "Main Functions"
        println("Testing $(unit2_1)...")
        t2_1 = @elapsed @testset "$(unit2_1)" begin
            include("unit-tests/Integration/OneBody-test.jl")
            include("unit-tests/Integration/TwoBody-test.jl")
            include("unit-tests/Integration/math-test.jl")
            include("unit-tests/Basis-test.jl")
            include("unit-tests/Differentiation-test.jl")
        end
        println("$(unit2_1) test finished in $t2_1 seconds.\n")

        unit2_2 = "Data Structure"
        println("Testing $(unit2_2)...")
        t2_2 = @elapsed @testset "$(unit2_2)" begin
            include("unit-tests/Box-test.jl")
            include("unit-tests/Library-test.jl")
            include("unit-tests/Molecule-test.jl")
        end
        println("$(unit2_2) test finished in $t2_2 seconds.\n")

        unit2_3 = "Applications"
        println("Testing $(unit2_3)...")
        t2_3 = @elapsed @testset "$(unit2_3)" begin
            include("unit-tests/HartreeFock-test.jl")
            include("unit-tests/Optimization-test.jl")
        end
        println("$(unit2_3) test finished in $t2_3 seconds.\n")

    end

    @testset "Submodule Functions" begin
        unit3_1 = "Submodule: Molden"
        println("Testing $(unit3_1)...")
        t3_1 = @elapsed @testset "$(unit3_1)" begin
            include("unit-tests/SubModule/Molden-test.jl")
        end
        println("$(unit3_1) test finished in $t3_1 seconds.\n")
    end

end