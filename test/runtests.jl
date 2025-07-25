using Test
using Documenter

@testset "Quiqbox tests" begin

    println("Number of threads used for testing: ", Threads.nthreads(), "\n")

    unit1 = "Utility Components"
    println("Testing $(unit1)...")
    t1 = @elapsed @testset "$(unit1)" begin
        include("unit-tests/IO-test.jl")
        include("unit-tests/Tools-test.jl")
        include("unit-tests/Strings-test.jl")
        include("unit-tests/Iteration-test.jl")
        include("unit-tests/Collections-test.jl")
    end
    println("$(unit1) test finished in $t1 seconds.\n")

    unit2 = "Core Framework"
    println("Testing $(unit2)...")
    t2 = @elapsed @testset "$(unit2)" begin
        include("unit-tests/Types-test.jl")
        include("unit-tests/Query-test.jl")
    end
    println("$(unit2) test finished in $t2 seconds.\n")

    unit3 = "Function-Composition System"
    println("Testing $(unit3)...")
    t3 = @elapsed @testset "$(unit3)" begin
        include("unit-tests/Mapping-test.jl")
        include("unit-tests/Operators-test.jl")
        include("unit-tests/Parameters-test.jl")
        include("unit-tests/Graphs-test.jl")
    end
    println("$(unit3) test finished in $t3 seconds.\n")

    unit4 = "Numerical-Computation System"
    println("Testing $(unit4)...")
    t4 = @elapsed @testset "$(unit4)" begin
        include("unit-tests/Arithmetic-test.jl")
        include("unit-tests/Angular-test.jl")
        include("unit-tests/Particles-test.jl")
    end
    println("$(unit4) test finished in $t4 seconds.\n")

    unit5 = "Basis-Construction System"
    println("Testing $(unit5)...")
    t5 = @elapsed @testset "$(unit5)" begin
        include("unit-tests/FieldFunctions-test.jl")
        include("unit-tests/OrbitalBases-test.jl")
    end
    println("$(unit5) test finished in $t5 seconds.\n")

    unit6 = "Differentiation System"
    println("Testing $(unit6)...")
    t6 = @elapsed @testset "$(unit6)" begin
        include("unit-tests/Differentiation/Finite-test.jl")
    end
    println("$(unit6) test finished in $t6 seconds.\n")

    unit7 = "Integration System"
    println("Testing $(unit7)...")
    t7 = @elapsed @testset "$(unit7)" begin
        include("unit-tests/Integration/Samplers-test.jl")
        include("unit-tests/Integration/BoysFunction-test.jl")
        include("unit-tests/Integration/Framework-test.jl")
        include("unit-tests/Integration/Interface-test.jl")
        include("unit-tests/Integration/Overlap-test.jl")
        include("unit-tests/Integration/Kinetic-test.jl")
        include("unit-tests/Integration/Coulomb-test.jl")
    end
    println("$(unit7) test finished in $t7 seconds.\n")

    unit8 = "Ab Initio Computation System"
    println("Testing $(unit8)...")
    t8 = @elapsed @testset "$(unit8)" begin
        include("unit-tests/HartreeFock-test.jl")
    end
    println("$(unit8) test finished in $t8 seconds.\n")

    unit9 = "Code Quality"
    println("Testing $(unit9)...")
    t9 = @elapsed @testset "$(unit9)" begin
        include("quality-tests/Aqua-test.jl")
        doctest(Quiqbox)
    end
    println("$(unit9) test finished in $t9 seconds.\n")

end
