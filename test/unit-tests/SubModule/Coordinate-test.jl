push!(LOAD_PATH, "./Quiqbox")
using Quiqbox.Coordinate
using Rotations
using Test

@testset "BoxControl.jl test" begin


x1 = [0, -0.7, 0]
x2 = [0, 0.7, 0]

x1_1, R1 = intrRots(x1, RotY(π/4), RotX(-π/2), RotZ(π/2))
x1_2, R2 = extrRots(x1, RotZ(π/2), RotX(-π/2), RotY(π/4))
@test x1_1 ≈ x1_2
@test R1 ≈ R2
x1_3, _ = extrRots(x1_1, RotZ(π), RotX(-π/2), bodyFrameTransMat=R1)
x1_4, _ = intrRots(x1_1, RotX(-π/2), RotZ(π), bodyFrameTransMat=R1)
x1_5, _ = intrRots(x1_1, RotZ(π/4), RotX(π/2), bodyFrameTransMat=R1)
x1_6, _ = extrRots(x1_2, RotY(π/4), RotX(-π/2))
@test x1 ≈ x1_3 ≈ x1_4 ≈ x1_5 ≈ x1_6
x_7, _ = extrRots([x1_1,x2], RotZ(π), RotX(-π/2), bodyFrameTransMat=R1)
x_8, _ = intrRots([x1_1,x2], RotZ(π/4), RotX(π/2), bodyFrameTransMat=R1)
x_9, _ = extrRots([x1_2,x2], RotY(π/4), RotX(-π/2))
@test x1 ≈ x_7[1] ≈ x_8[1] ≈ x_9[1]


end