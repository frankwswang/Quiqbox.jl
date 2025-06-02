using Test

using Quiqbox: getAtolDigits, getAtolVal, roundToMultiOfStep, nearestHalfOf, getNearestMid, 
               oddFactorial

@testset "Arithmetic.jl" begin

# function  getAtolDigits
@test getAtolDigits(Float32) == 6
@test getAtolDigits(Float64) == 15


# function getAtolVal
@test getAtolVal(Int) == 1
@test getAtolVal(Float64) == 4e-16


# function roundToMultiOfStep nearestHalfOf getNearestMid
@test roundToMultiOfStep(3811.47123123, 0.01) == 3811.47
@test roundToMultiOfStep(0.1+0.2, 1e-17) == 0.30000000000000004
@test roundToMultiOfStep(0.1+0.2, 1e-16) == 0.3
@test roundToMultiOfStep(2, 2) == 2
@test roundToMultiOfStep(2, 3) == 3
@test roundToMultiOfStep(2, 4) == 0
@test roundToMultiOfStep(2, 5) == 0

@test nearestHalfOf(0.1 + 0.2) == 0.15

@test getNearestMid(0.1, 0.2, 1e-16) == 0.15
@test getNearestMid(0.1, 0.2, 1e-17) == (0.1 + 0.2)/2


# function oddFactorial
@test oddFactorial(1) == 1
@test oddFactorial(19) == Int( factorial(20) // (2^10 * factorial(10)) )
@test oddFactorial(21, Int128) === oddFactorial(21)
@test oddFactorial(21, Float64) === oddFactorial(19) * 21.0 === oddFactorial(21, 1.0)
@test oddFactorial(21, 0.5) == oddFactorial(21) / exp2(11)
@test oddFactorial(61, BigInt) == BigInt( factorial(62|>big) // (2^31*factorial(31|>big)) )

end