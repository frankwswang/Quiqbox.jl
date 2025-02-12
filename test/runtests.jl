using Test

@testset "Quiqbox tests" begin

    println("Number of threads used for testing: ", Threads.nthreads(), "\n")

    unit1 = "Utility Components"
    println("Testing $(unit1)...")
    t1 = @elapsed @testset "$(unit1)" begin
        include("unit-tests/Tools-test.jl")
        include("unit-tests/StringIO-test.jl")
    end
    println("$(unit1) test finished in $t1 seconds.\n")

    unit2 = "Core Framework"
    println("Testing $(unit2)...")
    t2 = @elapsed @testset "$(unit2)" begin
        include("unit-tests/Types-test.jl")
        include("unit-tests/Layout-test.jl")
    end
    println("$(unit2) test finished in $t2 seconds.\n")

    unit3 = "Parameterization System"
    println("Testing $(unit3)...")
    t3 = @elapsed @testset "$(unit3)" begin
        include("unit-tests/Parameters-test.jl")
    end
    println("$(unit3) test finished in $t3 seconds.\n")

    unit4 = "Basis-Construction System"
    println("Testing $(unit4)...")
    t4 = @elapsed @testset "$(unit4)" begin
        include("unit-tests/Angular-test.jl")
        include("unit-tests/Spatial-test.jl")
        include("unit-tests/SpatialBasis-test.jl")
    end
    println("$(unit4) test finished in $t4 seconds.\n")

    unit5 = "Differentiation System"
    println("Testing $(unit5)...")
    t5 = @elapsed @testset "$(unit5)" begin
        include("unit-tests/Differentiation/Finite-test.jl")
    end
    println("$(unit5) test finished in $t5 seconds.\n")

    unit6 = "Integration System"
    println("Testing $(unit6)...")
    t6 = @elapsed @testset "$(unit6)" begin
        include("unit-tests/Integration/Interface-test.jl")
        include("unit-tests/Integration/Overlap-test.jl")
    end
    println("$(unit6) test finished in $t6 seconds.\n")

    unit7 = "Aqua.jl"
    println("Running $(unit7) Test...")
    t7 = @elapsed @testset "$(unit7)" begin
        include("quality-tests/Aqua-test.jl")
    end
    println("$(unit7) test finished in $t7 seconds.\n")

end
