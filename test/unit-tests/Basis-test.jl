using Test
using Quiqbox
using Quiqbox: isFull, getBasisFuncs

@testset "Basis.jl" begin

# struct GaussFunc
exp1, con1 = (2, 1.0)
gf1 = GaussFunc(exp1, con1)
gf2 = GaussFunc(exp1, con1)
@test typeof(gf1) == typeof(gf2)
@test (gf1, gf2) isa Tuple{GaussFunc, GaussFunc}
@test (gf1, gf2) isa NTuple{2, GaussFunc}
gf1_ref = gf1
@test gf1.xpn == ParamBox(exp1, :xpn)
@test gf1.con == ParamBox(con1, :con)
@test gf1.param == (gf1.xpn, gf1.con)
@test hasEqual(gf1, gf2)
@test hasIdentical(gf1, gf1)
@test !hasIdentical(gf1, gf2)
@test gf1 !== gf2
@test (gf1 === gf1_ref)
@test hasIdentical(gf1, gf1_ref)
gf1_ref.xpn[] = 3.0
@test gf1.xpn[] == gf1.param[1][] == 3.0
gf1.param[2][] = 4
@test gf1_ref.con[] == gf1_ref.param[2][] == 4
@test gf1_ref === gf1
@test gf2 != gf1
x = [2.0, 1]
for i=1:2 gf1.param[i][] = x[i] end
@test gf1 === gf1_ref !== gf2
@test hasEqual(gf1, gf2)


# struct BasisFunc
cen = [1,2,3]
using Quiqbox: ParamList
cenParNames = [ParamList[:X], ParamList[:Y], ParamList[:Z]]
cenPar = ParamBox.(cen, cenParNames) |> Tuple
bf1 = BasisFunc(cen, [gf1])
bf1_1 = BasisFunc(cen, [gf1])
bf1_2 = BasisFunc(bf1)
bf1_3 = BasisFunc(cenPar, gf1)
bf1_4 = BasisFunc(cenPar, gf1)
bf2 = BasisFunc(cen, [gf2])
@test bf1 !== bf2
@test hasEqual(bf1, bf1_1, bf1_2, bf1_3, bf1_4, bf2)
@test hasIdentical(bf1, bf1_2)
@test !hasIdentical(bf1, bf1_1)
@test !hasIdentical(bf1, bf2)
@test !hasIdentical(bf1, bf1_3)
@test hasIdentical(bf1_3, bf1_4)
@test bf1.subshell == "S"
@test bf1 isa BasisFunc

bf11 = BasisFunc(cen, [gf1, gf1])
bf1_3 = BasisFunc(cen, (exp1, con1))
@test hasEqual(bf1, bf1_3)
@test !hasIdentical(bf1, bf1_3)
bf11_2 = BasisFunc(cen, ([exp1, exp1], [con1, con1]))
@test hasEqual(bf11, bf11_2)
@test !hasIdentical(bf11, bf11_2)
bf2_P_norm2 = BasisFunc(cen, [gf2], "P")
@test bf2_P_norm2.subshell == "P"
@test bf2_P_norm2 isa BasisFuncs

bf3_1 = BasisFunc([0,0,0], (3,1))
bf3_2 = BasisFunc([1,0,0], (2,1))
bf3_3 = BasisFunc([0,0,0], (2,1), "P")
bf3_4 = BasisFunc([0,0,0], (1,1))
@test BasisFunc([bf3_1, bf3_2, bf3_3, bf3_4]) == [bf3_1, bf3_4, bf3_3, bf3_2]


# function isFull
@test isFull(bf3_1) == true
@test isFull(bf3_3) == true
bf4 = BasisFunc([0,0,0], (2,1), [1,0,0])
@test isFull(bf4) == false
@test isFull(1) == false


# struct BasisFuncMix
bfm1 = BasisFuncMix(bf1)
@test bfm1 == BasisFuncMix([bf1])
@test BasisFuncMix(BasisFuncMix(bf3_1)) == BasisFuncMix(bf3_1)
bf4_2 = BasisFunc([0,0,0], (2,1), [[1,0,0]])
bfm2 = BasisFuncMix(bf4)
@test hasEqual(bfm2, BasisFuncMix(bf4_2)[])


# func getBasisFuncs
@test getBasisFuncs(bfm1)[1] == bf1
@test getBasisFuncs(bf1) == (bf1,)
@test getBasisFuncs(0) == ()


# basisSize
@test basisSize("P") == (3,)
@test basisSize(["S", "P", "D"]) == (1, 3, 6)
@test basisSize(bf1) == (1,)
@test basisSize(bfm1) == (1,) == basisSize(bfm2)
@test basisSize([bf1, bf2, bf3_3, bf4]) == (1,1,3,1)


# function genGaussFuncText
bfCoeff = [[6.163845031, 1.097161308], [0.4301284983, 0.6789135305], 
           [0.2459163220, 0.06237087296], [0.04947176920, 0.9637824081],
           [0.2459163220, 0.06237087296], [0.5115407076, 0.6128198961]]
bfCoeff2 = vcat([[bfCoeff[2i+1]'; bfCoeff[2i+2]']' for i=0:2]...)
content = """
S    2   1.0
      6.163845031         0.4301284983
      1.097161308         0.6789135305
S    2   1.0
      0.245916322         0.0494717692
      0.06237087296       0.9637824081
P    2   1.0
      0.245916322         0.5115407076
      0.06237087296       0.6128198961
"""
lines = (content |> IOBuffer |> readlines)
@test map(i->Quiqbox.genGaussFuncText(bfCoeff2[i,:]...), 1:size(bfCoeff2)[1] |> collect) == 
      [lines[2], lines[3], lines[5], lines[6], lines[8], lines[9]].*"\n"


# function assignCenter!
bf5 = BasisFunc("STO-3G")[]
coord = [1,0,0]
bf5_1 = BasisFunc(coord, "STO-3G")[]
@test !hasEqual(bf5, bf5_1)
assignCenter!(coord, bf5)
@test hasEqual(bf5, bf5_1)


# function expressionOf
@test expressionOf(gf1) |> string == "d*exp(-α*(r₁^2 + r₂^2 + r₃^2))"
@test expressionOf(bf1)[] |> string == "d*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))"

expStr = expressionOf(bf5)[] |> string
idx = findfirst('d', expStr)
@test isapprox(parse(Float64, expStr[1:idx-1]), 7.579425332952777, atol=1e-12)
@test expStr[idx:end] == "d*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))*(α^0.75)"

@test expressionOf(bfm1)[] |> string == "d*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))"
@test expressionOf(bfm2)[] |> string == "d*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))"*
                                        "*(r₁ - X)"


end