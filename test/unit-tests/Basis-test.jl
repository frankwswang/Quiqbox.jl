using Test
using Quiqbox
using Quiqbox: isFull, getBasisFuncs, inSymbols, varVal, ElementNames
using Symbolics

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


# struct BasisFunc and function genBasisFunc
cen = [1,2,3]
using Quiqbox: ParamList
cenParNames = [ParamList[:X], ParamList[:Y], ParamList[:Z]]
cenPar = ParamBox.(cen, cenParNames) |> Tuple
bf1 = genBasisFunc(cen, [gf1])
bf1_1 = genBasisFunc(cen, [gf1])
bf1_2 = genBasisFunc(bf1)
bf1_3 = genBasisFunc(cenPar, gf1)
bf1_4 = genBasisFunc(cenPar, gf1)
bf2 = genBasisFunc(cen, [gf2])
@test bf1 !== bf2
@test hasEqual(bf1, bf1_1, bf1_2, bf1_3, bf1_4, bf2)
@test hasIdentical(bf1, bf1_2)
@test !hasIdentical(bf1, bf1_1)
@test !hasIdentical(bf1, bf2)
@test !hasIdentical(bf1, bf1_3)
@test hasIdentical(bf1_3, bf1_4)
@test bf1.subshell == "S"
@test bf1 isa BasisFunc

bf11 = genBasisFunc(cen, [gf1, gf1])
bf1_3 = genBasisFunc(cen, (exp1, con1))
@test hasEqual(bf1, bf1_3)
@test !hasIdentical(bf1, bf1_3)
bf11_2 = genBasisFunc(cen, ([exp1, exp1], [con1, con1]))
@test hasEqual(bf11, bf11_2)
@test !hasIdentical(bf11, bf11_2)
bf2_P_norm2 = genBasisFunc(cen, [gf2], "P")
@test bf2_P_norm2.subshell == "P"
@test bf2_P_norm2 isa BasisFuncs

bf3_1 = genBasisFunc([0,0,0], (3,1))
bf3_2 = genBasisFunc([1,0,0], (2,1))
bf3_3 = genBasisFunc([0,0,0], (2,1), "P")
bf3_4 = genBasisFunc([0,0,0], (1,1))
@test genBasisFunc([bf3_1, bf3_2, bf3_3, bf3_4]) == [bf3_1, bf3_4, bf3_3, bf3_2]


# function isFull
@test isFull(bf3_1) == true
@test isFull(bf3_3) == true
bf4 = genBasisFunc([0,0,0], (2,1), [1,0,0])
@test isFull(bf4) == false
@test isFull(1) == false


# struct BasisFuncMix
using Quiqbox: BasisFuncMix
bfm1 = BasisFuncMix(bf1)
@test bfm1 == BasisFuncMix([bf1])
@test BasisFuncMix(BasisFuncMix(bf3_1)) == BasisFuncMix(bf3_1)
bf4_2 = genBasisFunc([0,0,0], (2,1), [[1,0,0]])
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


# function genBasisFuncText & genBFuncsFromText
randElement = ElementNames[rand(1:length(ElementNames))]
bs1 = genBasisFunc(missing, ("6-31G", "H"))
cens = [rand(3) for _=1:length(bs1)]
assignCenter!.(cens, bs1)
txt1 = genBasisFuncText(bs1, printCenter=false, groupCenters=false) |> join
txt2 = genBasisFuncText(bs1, printCenter=false) |> join
bs2_1 = genBFuncsFromText(txt1)
bs2_2 = genBFuncsFromText(txt2)
assignCenter!.(cens, bs2_1)
assignCenter!.(cens, bs2_2)
@test hasEqual(bs1, bs2_2, ignoreContainerType=true)
@test hasEqual(bs1, bs2_1, ignoreContainerType=true)


# function assignCenter!
bf5 = genBasisFunc(missing, "STO-3G")[]
coord = [1,0,0]
bf5_1 = genBasisFunc(coord, "STO-3G")[]
@test !hasEqual(bf5, bf5_1)
assignCenter!(coord, bf5)
@test hasEqual(bf5, bf5_1)


# function getParams
pb1 = ParamBox(2, :p, canDiff=false)
@test getParams(pb1) == pb1
@test getParams(pb1, :p) == pb1
@test getParams(pb1, :P) === nothing
@test getParams(pb1, onlyDifferentiable=true) === nothing

pb2 = ParamBox(2, :q)
@test getParams([pb1, pb2]) == [pb1, pb2]
@test getParams([pb1, pb2], :p) == [pb1]

gf_pbTest1 = GaussFunc(2,1)
@test getParams(gf_pbTest1) == gf_pbTest1.param |> collect

gf_pbTest2 = GaussFunc(1.5,0.5)
bf_pbTest1 = genBasisFunc([1,0,0], [gf_pbTest1, gf_pbTest2])
@test getParams(bf_pbTest1) == [gf_pbTest1.param..., gf_pbTest2.param..., 
                                bf_pbTest1.center...]

@test getParams([gf_pbTest1, gf_pbTest2, bf_pbTest1.center...]) == getParams(bf_pbTest1)

alpha = Quiqbox.ParamList[:xpn]
@test getParams(bf_pbTest1, alpha) == 
      vcat(getParams(gf_pbTest1, alpha), getParams(gf_pbTest2, alpha)) == 
      [gf_pbTest1.param[1], gf_pbTest2.param[1]]

@test getParams([pb1, bf_pbTest1], [:X, :Y, :Z]) == [nothing, bf_pbTest1.center[1],
                                                     nothing, bf_pbTest1.center[2],
                                                     nothing, bf_pbTest1.center[3]]

# function getVar & getVars
@test getVar(pb1 |> typeof).val.name == :p
@test getVar(pb1)[][1].val.name == :p
@test getVar(pb1)[][2] == 2.0
@test getVar(pb1)[] isa Pair{Num, Float64}

gf11 = GaussFunc(3,1)
gf12 = GaussFunc(3,0.5)
@test getVars([bf1]) == getVars(bf1)
@test getVars([gf11, gf12]) == merge(getVars(gf11), getVars(gf12))
@test getVars(gf1.param |> collect) == Dict(getVar(ParamBox(1, :d))[], getVar(ParamBox(2, :α))[])


# function expressionOf
@test expressionOf(gf1) |> string == "d*exp(-α*(r₁^2 + r₂^2 + r₃^2))"
@test expressionOf(bf1)[] |> string == "d*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))"

expStr = expressionOf(bf5)[] |> string
idx = findfirst('d', expStr)
@test isapprox(parse(Float64, expStr[1:idx-1]), 7.579425332952777, atol=1e-12)
@test expStr[idx:end] == "d*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))*(α^0.75)" || 
      expStr[idx:end] == "d*(α^0.75)*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))"

@test expressionOf(bfm1)[] |> string == "d*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))"
expr_bfm2 =  expressionOf(bfm2)[] |> string
@test expr_bfm2 == "d*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))*(r₁ - X)" || 
      expr_bfm2 == "d*(r₁ - X)*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))"


# function inSymbols
sym1 = :a1
sym2 = :d
syms = [:a, :b, :c]
@test inSymbols(sym1, syms) == :a
@test !inSymbols(sym2, syms)

sym3 = Symbolics.variable(:a)
@test inSymbols(sym3, syms) == :a
@test inSymbols(sym3.val, syms) == :a
@test inSymbols(log(sym3).val, syms) == false

@test inSymbols(abs, syms) == false


# function varVal
vars = @variables X, Y, Z, F(X)
F = vars[end]
vars = vars[1:end-1]
f1(X,Y,Z) = log(X+0.5Y*Z)
expr = f1(X,Y,Z)
vals = [1, 3.334, -0.2]
d1 = Dict(vars .=> vals)
errT = 1e-10
@test varVal.(vars, Ref(d1)) == vals
@test isapprox(varVal(sum(vars), d1), sum(vals), atol=errT)
@test isapprox(varVal(prod(vars), d1), prod(vals), atol=errT)
@test isapprox(varVal(vars[1]^vars[2], d1), vals[1]^vals[2], atol=errT)
@test isapprox(varVal(expr, d1), f1(vals...), atol=errT)

sp = 1.5
gb1 = GridBox(1,sp)
d2 = getVars(gb1.box |> flatten, includeMapping=true)
l = Symbolics.variable(:L,0)
vars2 = [i.val for i in keys(d2)]
diffs2 = Differential(l).(vars2)
exprs2 = map(x -> (x isa SymbolicUtils.Term) ? d2[x] : x, vars2)
diffExprs2 = Symbolics.derivative.(exprs2, Ref(l))
excs2 = Symbolics.build_function.(exprs2, l) .|> eval
vals2 = [f(sp) for f in excs2]
excdiffs2 = Symbolics.build_function.(diffExprs2, l) .|> eval
diffvals2 = [f(sp) for f in excdiffs2]

@test varVal.(vars2, Ref(d2)) == vals2
@test varVal.(diffs2, Ref(d2)) == diffvals2

end