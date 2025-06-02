using Test
using Quiqbox: computeBoysFunc, computeBoysSequence

@testset "BoysFunction.jl" begin

refPoints = [
    (2.6e-7, 100, 4.97512309732144e-03), 
    (6.4e-5,  45, 1.09883228385254e-02), 
    (1.4e-3,  20, 2.43577075309547e-02), 
    (  6.4,   25, 4.28028518677348e-05), 
    ( 13.0,   25, 8.45734447905704e-08), 
    ( 26.0,   30, 3.57321060811178e-13), 
    ( 27.0,   15, 1.08359515555596e-11), 
    ( 30.0,   20, 1.37585444267909e-13), 
    ( 33.0,  100, 3.42689684943483e-17), 
    ( 50.0,   16, 2.40509456111904e-16), 
    ( 50.0,   64, 5.67024356263279e-24), 
    ( 85.0,   33, 1.74268831008018e-29), 
    (100.0,   36, 3.08919970425521e-33), 
    (120.0,  100, 4.97723065221079e-53), 
    (125.1,  100, 7.75391047694625e-55)
]

for ele in refPoints
    x, order, val = ele
    res = computeBoysFunc(Float64(x), Int64(order))
    @test isapprox(res, val, rtol=(sqrt∘eps)(Float64))
end

fVals = computeBoysSequence(50.0, 100)
@test isapprox(fVals[begin+16], refPoints[begin+ 9][end], rtol=(sqrt∘eps)(Float64))
@test isapprox(fVals[begin+64], refPoints[begin+10][end], rtol=(sqrt∘eps)(Float64))

end