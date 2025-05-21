using Test
using Quiqbox
using Quiqbox: SymmetricIntRange, getInterpolationNumber, getFiniteDiffAccuOrder

@testset "Finite.jl" begin

grid1 = [-8.40927, -5.59942, -2.42758, -2.17322, -1.4906, -9.0402, -9.1059, 3.77213]
fdCoeff1 = [3.5216741758505528, -1.4475057307659966, 38.25843848476915, 
            -50.38822824854306, 14.021518249238527, -20.302726783610808, 
            17.32848738511764, 0.008342467943965013]
fdCoeff2 = [5.905403869232238, -2.3409069379935543, 52.94422768461955, 
            -67.30071929291951, 15.773080673711144, -34.213614445916264, 
            29.215335518827082, 0.017192930439314455]
fdCoeff3 = [-3.6601376379072246, 1.5415689398103254, -31.28161176014497, 
            38.653612434060015, -8.363927960856214, 20.889446806996983, 
            -17.81010097243767, 0.031150150478757854]
fdCoeffRef1 = (fdCoeff1, fdCoeff2, fdCoeff3)
orders = (0, 1, 4)
for (i, r) in zip(orders, fdCoeffRef1)
    @test Quiqbox.computeFiniteDiffWeights(i, grid1) ≈ r
end

@test try SymmetricIntRange(Val(-1))  catch; true end
@test try SymmetricIntRange(Val(1.1)) catch; true end
@test SymmetricIntRange(Val(0))() == 0:0
@test SymmetricIntRange(Val(3))() == -3:3


grid2 = [-3//1, -2//1, -1//1, 0, 1//1, 2//1, 3//1]
fdCoeff4 = [0, 0, 0, 1, 0, 0, 0]
fdCoeff5 = [-1//60, 3//20, -3//4, 0, 3//4, -3//20, 1//60]
fdCoeff6 = [-1//6, 2//1, -13//2, 28//3, -13//2, 2//1, -1//6]
fdCoeffRef2 = (fdCoeff4, fdCoeff5, fdCoeff6)
for (i, r) in zip(orders, fdCoeffRef2)
    @test Quiqbox.computeFiniteDiffWeights(i, grid2) == r
end

xpn_df_test1 = 0.0123456789
f1 = x->exp(-xpn_df_test1 * x^2)
d1 = x->exp(-xpn_df_test1 * x^2) * (-2xpn_df_test1*x)
derivativeOrder = Val(1)
nFiniteDiffGRad = SymmetricIntRange(Val(5))
ps, ws = Quiqbox.getFiniteDiffWeights(Float64, derivativeOrder, nFiniteDiffGRad)
fdVal1 = sum(ws .* f1.(ps .+ 0.5))
@test abs(fdVal1 - d1(0.5)) < 10eps(Float64)

@test getInterpolationNumber(1, 4) == 5
@test getFiniteDiffAccuOrder(1, 5) == 4
@test getInterpolationNumber(2, 4) == 5
@test getFiniteDiffAccuOrder(2, 5) == 4
@test getInterpolationNumber(3, 4) == 7
@test getFiniteDiffAccuOrder(3, 7) == 4
@test getInterpolationNumber(4, 6) == 9
@test getFiniteDiffAccuOrder(4, 9) == 6


f = Quiqbox.TypedCarteFunc(x->x[1]^2 + x[2]*x[1] + x[3]^3/x[1], Float64, Val(3))
df_fd1 = Quiqbox.AxialFiniteDiff(f, Val(1), 1)
df_sd1 = x->(2x[1] + x[2] - x[3]^3/x[1]^2)
testCoord = (1.1, 2.1, -3.2)
@test isapprox(df_fd1(testCoord), df_sd1(testCoord), atol=1e-9)
df_fd2 = Quiqbox.AxialFiniteDiff(f, Val(2), 1)
df_sd2 = x->(2 + 2x[3]^3/x[1]^3)
@test isapprox(df_fd2(testCoord), df_sd2(testCoord), atol=1e-9)

end