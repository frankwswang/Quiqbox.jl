using Test
using Quiqbox
using Quiqbox: isFull, BasisFuncMix, unpackBasisFuncs, inSymbols, varVal, ElementNames, 
               sortBasisFuncs, ParamList, sumOf, expressionOfCore, mergeGaussFuncs, 
               ijkOrbitalList
using Symbolics
using LinearAlgebra

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
bf3_4 = genBasisFunc([0,0,0], (1,1), "D")
@test genBasisFunc([bf3_1, bf3_4, bf3_3, bf3_2]) == [bf3_1, bf3_3, bf3_4, bf3_2]

bf4_1 = genBasisFunc([0,0,0], "STO-3G")[]
bf4_2 = genBasisFunc([0,0,0], ("STO-3G", "He"))[]
bf4_3 = genBasisFunc([0,0,0], "STO-3G", nucleus = "He")[]
bf4s1 = genBasisFunc([0,0,0], ["STO-3G", "STO-3G"])
bf4s2 = genBasisFunc([0,0,0], ["STO-3G", "STO-3G"], nucleus = "He")
bf4s3 = genBasisFunc([0,0,0], [("STO-3G", "H"), ("STO-3G", "He")])
@test hasEqual.(Ref(bf4_1), bf4s1, Ref(bf4s3[1])) |> prod
@test hasEqual.(Ref(bf4_2), Ref(bf4_3), bf4s2, Ref(bf4s3[2])) |> prod
@test isapprox(1, overlap(bf4_1, bf4_1)[], atol=1e-8)
@test isapprox(1, overlap(bf4_2, bf4_2)[], atol=1e-8)

errorThreshold1 = 1e-11
bf3_2_2 = genBasisFunc([1,0,0], (2,1), normalizeGTO=true)
@test isapprox(1, overlap(bf3_2_2, bf3_2_2)[], atol=errorThreshold1)
@test isapprox(0.0553891828418, overlap(bf3_2, bf3_2)[], atol=errorThreshold1)
bf3_3_2 = genBasisFunc([0,0,0], (2,1), "P", normalizeGTO=true)
@test isapprox(LinearAlgebra.I, overlap(bf3_3_2, bf3_3_2), atol=errorThreshold1)
@test isapprox(0.0207709435653*LinearAlgebra.I, overlap(bf3_3, bf3_3), atol=errorThreshold1)
bf3_4_2 = genBasisFunc([0,0,0], (1,1), "D", normalizeGTO=true)
@test isapprox([1.0 0.0 0.0 1/3 0.0 1/3; 
                0.0 1/3 0.0 0.0 0.0 0.0; 
                0.0 0.0 1/3 0.0 0.0 0.0; 
                1/3 0.0 0.0 1.0 0.0 1/3; 
                0.0 0.0 0.0 0.0 1/3 0.0; 
                1/3 0.0 0.0 1/3 0.0 1.0], 
                overlap(bf3_4_2, bf3_4_2), atol=errorThreshold1)
@test isapprox([0.3691314831028692 0.0 0.0 0.1230438277009564 0.0 0.1230438277009564; 
                0.0 0.1230438277009564 0.0 0.0 0.0 0.0; 
                0.0 0.0 0.1230438277009564 0.0 0.0 0.0; 
                0.1230438277009564 0.0 0.0 0.3691314831028692 0.0 0.1230438277009564; 
                0.0 0.0 0.0 0.0 0.1230438277009564 0.0; 
                0.1230438277009564 0.0 0.0 0.1230438277009564 0.0 0.3691314831028692], 
                overlap(bf3_4, bf3_4), atol=errorThreshold1)


# function sortBasisFuncs
bfs1 = [genBasisFunc([1,1,1], (2,1), [1,0,0]), genBasisFunc([1,1,1], (3,1), [2,0,0]), 
        genBasisFunc([1,1,2], (3,1), [0,0,0]), genBasisFunc([1,1,1], (3,1), "P")]
bfs2 = [genBasisFunc([1,1,1], (2,1), [1,0,0]), genBasisFunc([1,1,1], (3,1), "P"), 
        genBasisFunc([1,1,1], (3,1), [2,0,0]), genBasisFunc([1,1,2], (3,1), [0,0,0])]
bfs3 = sortBasisFuncs(bfs1, groupCenters=true)
@test length.(bfs3) == [3,1]
bfs3 = bfs3 |> flatten
@test !hasEqual(bfs1, bfs2)
@test  hasEqual(bfs3, bfs2)


# function isFull
@test isFull(bf3_1) == true
@test isFull(bf3_3) == true
bf5 = genBasisFunc([0,0,0], (2,1), [1,0,0])
@test isFull(bf5) == false
@test isFull(1) == false


# function centerOf centerCoordOf
@test centerOf(bf5) == bf5.center
@test centerCoordOf(bf5) == [0,0,0]


# struct BasisFuncMix
bfm1 = BasisFuncMix(bf1)
@test bfm1 == BasisFuncMix([bf1])
@test BasisFuncMix(BasisFuncMix(bf3_1)) == BasisFuncMix(bf3_1)
bf5_2 = genBasisFunc([0,0,0], (2,1), [[1,0,0]])
bfm2 = BasisFuncMix(bf5)
@test hasEqual(bfm2, BasisFuncMix(bf5_2)[])

errorThreshold2 = 1e-15
bs1 = genBasisFunc.(gridCoords(GridBox(1,1.5)), Ref(GaussFunc(1.0, 0.5)))
nuc = ["H", "H"]
nucCoords = [rand(3), rand(3)]
bfm = BasisFuncMix(bs1)
S = overlaps([bfm])[]
@test S == overlap(bfm, bfm)[]
@test isapprox(S, overlaps(bs1) |> sum, atol=errorThreshold2)
T = elecKinetics([bfm])[]
@test T == elecKinetic(bfm, bfm)[]
@test isapprox(T, elecKinetics(bs1) |> sum, atol=errorThreshold2)
V = nucAttractions([bfm], nuc, nucCoords)[]
@test V == nucAttraction(bfm, bfm, nuc, nucCoords)[]
@test isapprox(V, nucAttractions(bs1, nuc, nucCoords) |> sum, atol=errorThreshold2)
eeI = eeInteractions([bfm])[]
@test eeI == eeInteraction(bfm, bfm, bfm, bfm)[]
@test isapprox(eeI, eeInteractions(bs1) |> sum, atol=errorThreshold2)


# function sumOf
bs2 = [genBasisFunc([1,1,1], (2,1), [1,0,0], normalizeGTO=true), 
       genBasisFunc([1,1,1], (3,1), [2,0,0], normalizeGTO=true), 
       genBasisFunc([1,1,2], (3,1), [0,0,0], normalizeGTO=true), 
       genBasisFunc([1,1,1], (3,1), [0,1,0], normalizeGTO=true)]
bs2_2 = [genBasisFunc([1,1,1], (2,1), [1,0,0]), 
         genBasisFunc([1,1,1], (3,1), [2,0,0]), 
         genBasisFunc([1,1,2], (3,1), [0,0,0]), 
         genBasisFunc([1,1,1], (3,1), [0,1,0])]
bs2_3 = [genBasisFunc([1,1,1], (2,1), [1,0,0]), 
         genBasisFunc([1,1,1], (3,1), [2,0,0], normalizeGTO=true), 
         genBasisFunc([1,1,2], (3,1), [0,0,0]), 
         genBasisFunc([1,1,1], (3,1), [0,1,0], normalizeGTO=true)]
bs3 = [genBasisFunc([1,1,1], (2,1), [1,0,0], normalizeGTO=true), 
       genBasisFunc([1,1,1], (3,1), [0,1,0], normalizeGTO=true), 
       genBasisFunc([1,1,1], (3,1), [2,0,0], normalizeGTO=true), 
       genBasisFunc([1,1,2], (3,1), [0,0,0], normalizeGTO=true)]
bs3_2 = [genBasisFunc([1,1,1], (2,1), [1,0,0]), 
       genBasisFunc([1,1,1], (3,1), [0,1,0]), 
       genBasisFunc([1,1,1], (3,1), [2,0,0]), 
       genBasisFunc([1,1,2], (3,1), [0,0,0])]

bfm_1 = +(bs2...,)
bfm_2 = sumOf(bs2)
bfm_3 = BasisFuncMix(bs3)
bfm_4 = +(bs2_2...,)
bfm_5 = sumOf(bs2_2)
bfm_6 = BasisFuncMix(bs3_2)
bfm_7 = sumOf([bfm_6])
@test hasEqual(bfm_1, bfm_3)
@test hasEqual(bfm_2, bfm_3)
@test hasEqual(bfm_4, bfm_6)
@test hasEqual(bfm_5, bfm_6)
@test hasEqual(bfm_6, bfm_7)


# function mergeGaussFuncs
gf_merge1 = GaussFunc(2,1)
gf_merge2 = GaussFunc(2,1)
gf_merge3 = GaussFunc(2,1)

@test mergeGaussFuncs(gf_merge1) === gf_merge1

mgf1 = mergeGaussFuncs(gf_merge1, gf_merge1)[]
mgf2 = mergeGaussFuncs(gf_merge1, gf_merge2)[]
@test mgf1.xpn() == 2 == mgf1.con()
@test !hasIdentical(mgf1, mgf2)
gf_merge1_2 = GaussFunc(gf_merge1.xpn, gf_merge2.con)
gf_merge1_3 = GaussFunc(gf_merge2.xpn, gf_merge1.con)
mgf1_2 = mergeGaussFuncs(gf_merge1, gf_merge1_2)[]
mgf1_3 = mergeGaussFuncs(gf_merge1, gf_merge1_3)[]
@test mgf1_2.xpn() == 2 == mgf1_2.con()
@test !hasIdentical(mgf1, mgf1_2)
@test !hasIdentical(mgf1_2, mgf1_3)
@test hasEqual(mgf1, mgf1_2, mgf1_3)

gf_merge3 = GaussFunc(1.5,1)
@test hasIdentical(mergeGaussFuncs(gf_merge1, gf_merge3), [gf_merge1, gf_merge3])


# , add, mul
@test add(bs2[1]) === bs2[1]
bf1s = BasisFuncs(bf1.center, bf1.gauss, [ijkOrbitalList[bf1.ijk[1]]], bf1.normalizeGTO)
@test hasIdentical(add(bf1s), bf1)

for bs in (bs2, bs2_2, bs2_3)
    X = overlaps(bs)^(-0.5)
    bsNew = [mul.(bs, @view X[:,i]) for i in 1:size(X, 2)] .|> sum
    SNew = overlaps(bsNew)
    @test isapprox(SNew, LinearAlgebra.I, atol=1e-14)
end

# TODO: add value test of Gaussian function product theorem.


# function shift
ijk = [1,0,0]
didjdk = [0,1,1]
bf_os1 = genBasisFunc([0,0,0], (2,1), ijk)
bf_os2 = genBasisFunc([0,0,0], (2,1), ijk, normalizeGTO=true)
bf_os1S = genBasisFunc([0,0,0], (2,1), ijk+didjdk)
bf_os2S = genBasisFunc([0,0,0], (2,1), ijk+didjdk, normalizeGTO=true)
@test hasEqual(shift(bf_os1, didjdk), bf_os1S)
@test hasEqual(shift(bf_os2, didjdk), bf_os2S)


# func unpackBasisFuncs
@test unpackBasisFuncs(bfm1)[1] == bf1
@test unpackBasisFuncs(bf1) == [bf1]
@test unpackBasisFuncs(0) == []


# basisSize
@test basisSize("P") == (3,)
@test basisSize(["S", "P", "D"]) == (1, 3, 6)
@test basisSize(bf1) == (1,)
@test basisSize(bfm1) == (1,) == basisSize(bfm2)
@test basisSize([bf1, bf2, bf3_3, bf5]) == (1,1,3,1)


# function genGaussFuncText
bfCoeff = [[6.163845031, 1.097161308], [0.4301284983, 0.6789135305], 
           [0.245916322, 0.06237087296], [0.0494717692, 0.9637824081],
           [0.245916322, 0.06237087296], [0.5115407076, 0.6128198961]]
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
txt1 = genBasisFuncText(bs1, printCenter=false, groupCenters=false) |> join
txt2 = genBasisFuncText(bs1, printCenter=false) |> join
bs2_1 = genBFuncsFromText(txt1)
bs2_2 = genBFuncsFromText(txt2)
assignCenter!.(cens, bs1)
assignCenter!.(cens, bs2_1)
assignCenter!.(cens, bs2_2)
txt3 = genBasisFuncText(bs1) |> join
bs2_3 = genBFuncsFromText(txt3)
@test hasEqual(bs1, bs2_1, ignoreContainer=true)
@test hasEqual(bs1, bs2_2, ignoreContainer=true)
@test hasEqual.(sortBasisFuncs(bs1), bs2_3, ignoreFunction=true) |> prod
@test hasEqual.(bs1, bs2_1, ignoreFunction=true) |> prod
@test hasEqual.(bs1, bs2_2, ignoreFunction=true) |> prod
@test hasEqual.(sortBasisFuncs(bs1), bs2_3, ignoreFunction=true) |> prod


# function assignCenter!
bf6 = genBasisFunc(missing, "STO-3G")[]
coord = [1,0,0]
bf6_1 = genBasisFunc(coord, "STO-3G")[]
@test !hasEqual(bf6, bf6_1)
assignCenter!(coord, bf6)
@test hasEqual(bf6, bf6_1)


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
@test getParams(bf_pbTest1) == [bf_pbTest1.center..., 
                                gf_pbTest1.param..., gf_pbTest2.param...]

alpha = Quiqbox.ParamList[:xpn]
@test getParams(bf_pbTest1, alpha) == 
      vcat(getParams(gf_pbTest1, alpha), getParams(gf_pbTest2, alpha)) == 
      [gf_pbTest1.param[1], gf_pbTest2.param[1]]

cs = [pb1, bf_pbTest1]
ss = [:X, :Y, :Z]
@test [getParams(i, j) for i in cs, j in ss] |> flatten == 
      [nothing, bf_pbTest1.center[1], nothing, bf_pbTest1.center[2], 
       nothing, bf_pbTest1.center[3]]
@test getParams(cs) == vcat(getParams(pb1), getParams(bf_pbTest1))
@test (getParams.(Ref(cs), ss) |> flatten) == (bf_pbTest1.center |> collect)


# function copyBasis
e = genExponent(3.0, x -> x^2 + 1)
c = genContraction(2.0)
gf_dc1 = GaussFunc(e, c)
gf_dc2 = copyBasis(gf_dc1)
@test hasEqual(gf_dc1.con, gf_dc2.con)
@test gf_dc1.xpn() == gf_dc2.xpn() == gf_dc2.xpn[]


# function getVar & getVarDict
# @test getVar(pb1)[][1].val.name == :p
# @test getVar(pb1)[][2] == 2.0
# @test getVar(pb1)[] isa Pair{Num, Float64}

gf11 = GaussFunc(3,1)
gf12 = GaussFunc(3,0.5)
# @test getVarDict([bf1]) == getVarDict(bf1)
# @test getVarDict.([gf11, gf12]) == merge(getVarDict(gf11), getVarDict(gf12))
# @test getVarDict(gf1.param |> collect) == Dict(getVar(ParamBox(1, :d))[], 
#                                             getVar(ParamBox(2, :α))[])


# function expressionOf, expressionOfCore
expr1 = expressionOf(gf1) |> string
@test expr1 == "exp(-2.0(r₁^2) - 2.0(r₂^2) - 2.0(r₃^2))" || 
      expr1 == "exp(-2.0(r₁^2) - (2.0(r₂^2)) - (2.0(r₃^2)))"

expr2 = expressionOf(bf1)[]|>string
@test expr2 == "exp(-2.0((r₁ - 1.0)^2) - 2.0((r₂ - 2.0)^2) - 2.0((r₃ - 3.0)^2))" || 
      expr2 == "exp(-2.0((r₁ - 1.0)^2) - (2.0((r₂ - 2.0)^2)) - (2.0((r₃ - 3.0)^2)))"
@test expressionOfCore(gf1) |> string == "d*exp(-α*(r₁^2 + r₂^2 + r₃^2))"
@test expressionOfCore(bf1)[]|>string == "d*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))"

expStr = expressionOfCore(bf6)[] |> string
idx = findfirst('d', expStr)
@test isapprox(parse(Float64, expStr[1:idx-1]), 7.579425332952777, atol=1e-12)
@test expStr[idx:end] == "d*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))*(α^0.75)" || 
      expStr[idx:end] == "d*(α^0.75)*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))"

@test expressionOfCore(bfm1)[]|>string == "d*exp(-α*((r₁ - X)^2 + (r₂ - Y)^2 + (r₃ - Z)^2))"
expr_bfm2 =  expressionOfCore(bfm2)[] |> string
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

f1 = Symbolics.variable(abs, T=Symbolics.FnType{Tuple{Any}, Real})(Z)
f2 = Symbolics.variable(:abs, T=Symbolics.FnType{Tuple{Any}, Real})(Z)
f1sym = f1.val
f2sym = f2.val
@test varVal(f1sym, d1) == abs(d1[Z])
@test varVal(f2sym, d1) == abs(d1[Z])

sp = 1.5
gb1 = GridBox(1,sp)
d2 = getVarDict(gb1.box |> flatten)
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