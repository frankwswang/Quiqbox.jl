using Test
using Quiqbox
using Quiqbox: BasisFuncMix, unpackBasis, inSymbols, varVal, ElementNames, sortBasisFuncs, 
               ParamList, sumOf, expressionOfCore, mergeGaussFuncs, gaussProd, getNorms
using Symbolics
using LinearAlgebra

@testset "Basis.jl" begin

# struct GaussFunc
xpn1, con1 = (2.0, 1.0)
gf1 = GaussFunc(xpn1, con1)
gf2 = GaussFunc(xpn1, con1)
@test typeof(gf1) == typeof(gf2)
@test (gf1, gf2) isa Tuple{GaussFunc, GaussFunc}
@test (gf1, gf2) isa NTuple{2, GaussFunc}
gf1_ref = gf1
@test gf1.xpn == ParamBox(xpn1, ParamList[:xpn])
@test gf1.con == ParamBox(con1, ParamList[:con])
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


# function genExponent genContraction
m1 = x->x^2
v1 = round(rand(), digits=6)
e1 = genExponent(v1, m1)
@test typeof(e1).parameters[2] == ParamList[:xpn]
@test e1[] == v1
@test e1() == m1(v1)

c1 = genContraction(v1, m1)
@test typeof(c1).parameters[2] == ParamList[:con]
@test c1[] == v1
@test e1() == m1(v1)

e2 = genExponent(e1)
e3 = genExponent(c1)
@test hasIdentical(e1, e2)
@test !hasEqual(e1, e3)
@test e1[] == e3[]
@test e1() == e3()
@test e1.map !== e3.map
v2 = rand()
@test e1.map(v2) == e3.map(v2)

c2 = genExponent(c1)
c3 = genExponent(e1)
@test hasIdentical(c1, c2)
@test !hasEqual(c1, c3)
@test c1[] == c3[]
@test c1() == c3()
@test c1.map !== c3.map
@test c1.map(v2) == c3.map(v2)


# function genSpatialPoint
k = fill(1.0)
v1 = genSpatialPoint([k, 2.0, 3.0])
v2 = genSpatialPoint([k, 2.0, 3.0])
v3 = genSpatialPoint([1.0, 2, 3.0])
v1_2 = genSpatialPoint((k, 2.0, 3.0))
v2_2 = genSpatialPoint((k, 2.0, 3.0))
v3_2 = genSpatialPoint((1.0, 2.0, 3.0))
@test v1 == v2
@test v1_2 == v2_2
@test hasEqual(v1, v2)
@test hasEqual(v1_2, v2_2)
@test hasEqual(v3, v3_2)
@test !hasIdentical(v1, v2)
@test hasIdentical(v1[1], v2[1], v1_2[1], v2_2[1])
@test !hasIdentical(v1[2:3], v2[2:3])


# struct BasisFunc and function genBasisFunc
cen = [1.0, 2.0, 3.0]
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
@test subshellOf(bf1) == "S"
@test bf1 isa BasisFunc

bf11 = genBasisFunc(cen, [gf1, gf1])
bf1_3 = genBasisFunc(cen, (xpn1, con1))
@test hasEqual(bf1, bf1_3)
@test !hasIdentical(bf1, bf1_3)
bf11_2 = genBasisFunc(cen, ([xpn1, xpn1], [con1, con1]))
@test hasEqual(bf11, bf11_2)
@test !hasIdentical(bf11, bf11_2)
bf2_P_norm2 = genBasisFunc(cen, [gf2], "P")
@test subshellOf(bf2_P_norm2) == "P"
@test bf2_P_norm2 isa BasisFuncs

bf3_1 = genBasisFunc([0.0, 0.0, 0.0], (3.0, 1.0))
bf3_2 = genBasisFunc([1.0, 0.0, 0.0], (2.0, 1.0))
bf3_3 = genBasisFunc([0.0, 0.0, 0.0], (2.0, 1.0), "P")
bf3_4 = genBasisFunc([0.0, 0.0, 0.0], (1.0, 1.0), "D")
@test genBasisFunc([bf3_1, bf3_4, bf3_3, bf3_2]) == [bf3_1, bf3_3, bf3_4, bf3_2]

bf4_1 = genBasisFunc(fill(0.0, 3), "STO-3G")[]
bf4_2 = genBasisFunc(fill(0.0, 3), ("STO-3G", "He"))[]
bf4_3 = genBasisFunc(fill(0.0, 3), "STO-3G", nucleus = "He")[]
bf4s1 = genBasisFunc(fill(0.0, 3), ["STO-3G", "STO-3G"])
bf4s2 = genBasisFunc(fill(0.0, 3), ["STO-3G", "STO-3G"], nucleus = "He")
bf4s3 = genBasisFunc(fill(0.0, 3), [("STO-3G", "H"), ("STO-3G", "He")])
@test hasEqual.(Ref(bf4_1), bf4s1, Ref(bf4s3[1])) |> all
@test hasEqual.(Ref(bf4_2), Ref(bf4_3), bf4s2, Ref(bf4s3[2])) |> all
@test isapprox(1, overlap(bf4_1, bf4_1)[], atol=1e-8)
@test isapprox(1, overlap(bf4_2, bf4_2)[], atol=1e-8)

errorThreshold1 = 1e-11; errorThreshold3 = 1e-9
bf3_2_2 = genBasisFunc([1.0, 0.0, 0.0], (2.0, 1.0), normalizeGTO=true)
@test isapprox(1, overlap(bf3_2_2, bf3_2_2)[], atol=errorThreshold1)
@test isapprox(0.6960409996039634, overlap(bf3_2, bf3_2)[], atol=errorThreshold1)
bf3_3_2 = genBasisFunc([0.0, 0.0, 0.0], (2.0, 1.0), "P", normalizeGTO=true)
@test isapprox(LinearAlgebra.I, overlaps((bf3_3_2,)), atol=errorThreshold1)
@test isapprox(0.0870051249505*LinearAlgebra.I, overlaps((bf3_3,)), atol=errorThreshold1)
bf3_4_2 = genBasisFunc([0.0, 0.0, 0.0], (1.0, 1.0), "D", normalizeGTO=true)
@test isapprox(LinearAlgebra.I, overlaps((bf4_2,)), atol=errorThreshold3)
@test isapprox([0.3691314831028692 0.0 0.0 0.1230438277009564 0.0 0.1230438277009564; 
                0.0 0.1230438277009564 0.0 0.0 0.0 0.0; 
                0.0 0.0 0.1230438277009564 0.0 0.0 0.0; 
                0.1230438277009564 0.0 0.0 0.3691314831028692 0.0 0.1230438277009564; 
                0.0 0.0 0.0 0.0 0.1230438277009564 0.0; 
                0.1230438277009564 0.0 0.0 0.1230438277009564 0.0 0.3691314831028692], 
                overlaps((bf3_4,)), atol=errorThreshold1)


# function sortBasisFuncs
bfs1 = [genBasisFunc(fill(1.0, 3), (2.0, 1.0), (1,0,0)), 
        genBasisFunc(fill(1.0, 3), (3.0, 1.0), (2,0,0)), 
        genBasisFunc([1.0, 1.0, 2.0], (3.0, 1.0), (0,0,0)), 
        genBasisFunc(fill(1.0, 3), (3.0, 1.0), "P")]
bfs2 = [genBasisFunc(fill(1.0, 3), (2.0, 1.0), (1,0,0)), 
        genBasisFunc(fill(1.0, 3), (3.0, 1.0), "P"), 
        genBasisFunc(fill(1.0, 3), (3.0, 1.0), (2,0,0)), 
        genBasisFunc([1.0, 1.0, 2.0], (3.0, 1.0), (0,0,0))]
bfs3 = sortBasisFuncs(bfs1, true)
@test length.(bfs3) == [3,1]
bfs3 = bfs3 |> flatten
@test !hasEqual(bfs1, bfs2)
@test  hasEqual(bfs3, bfs2)


# function centerOf centerCoordOf
bf5 = genBasisFunc(fill(0.0, 3), (2.0, 1.0), (1,0,0))
@test centerOf(bf5) == bf5.center
@test centerCoordOf(bf5) == fill(0.0, 3)


# struct BasisFuncMix
bfm1 = BasisFuncMix(bf1)
@test bfm1 == BasisFuncMix([bf1])
@test BasisFuncMix(BasisFuncMix(bf3_1)) == BasisFuncMix(bf3_1)
bf5_2 = genBasisFunc(fill(0.0, 3), (2.0, 1.0), [(1,0,0)])
bfm2 = BasisFuncMix(bf5)
@test hasEqual(bfm2, BasisFuncMix(bf5_2))

errorThreshold2 = 1e-15
bs1 = genBasisFunc.(gridCoords(GridBox(1,1.5)), Ref(GaussFunc(1.0, 0.5)))
nuc = ["H", "H"]
nucCoords = [rand(3), rand(3)]
bfm = BasisFuncMix(bs1)
S = overlaps([bfm])[]
@test S == overlap(bfm, bfm)[]
@test isapprox(S, overlaps(bs1) |> sum, atol=errorThreshold2)
T = eKinetics([bfm])[]
@test T == eKinetic(bfm, bfm)[]
@test isapprox(T, eKinetics(bs1) |> sum, atol=errorThreshold2)
V = neAttractions([bfm], nuc, nucCoords)[]
@test V == neAttraction(bfm, bfm, nuc, nucCoords)[]
@test isapprox(V, neAttractions(bs1, nuc, nucCoords) |> sum, atol=errorThreshold2)
eeI = eeInteractions([bfm])[]
@test eeI == eeInteraction(bfm, bfm, bfm, bfm)[]
@test isapprox(eeI, eeInteractions(bs1) |> sum, atol=errorThreshold2*10)


# function sumOf
bs2 =   [genBasisFunc([1.0, 1.0, 1.0], (2.0, 1.0), (1,0,0), normalizeGTO=true), 
         genBasisFunc([1.0, 1.0, 1.0], (3.0, 1.0), (2,0,0), normalizeGTO=true), 
         genBasisFunc([1.0, 1.0, 2.0], (3.0, 1.0), (0,0,0), normalizeGTO=true), 
         genBasisFunc([1.0, 1.0, 1.0], (3.0, 1.0), (0,1,0), normalizeGTO=true)]
bs2_2 = [genBasisFunc([1.0, 1.0, 1.0], (2.0, 1.0), (1,0,0)), 
         genBasisFunc([1.0, 1.0, 1.0], (3.0, 1.0), (2,0,0)), 
         genBasisFunc([1.0, 1.0, 2.0], (3.0, 1.0), (0,0,0)), 
         genBasisFunc([1.0, 1.0, 1.0], (3.0, 1.0), (0,1,0))]
bs2_3 = [genBasisFunc([1.0, 1.0, 1.0], (2.0, 1.0), (1,0,0)), 
         genBasisFunc([1.0, 1.0, 1.0], (3.0, 1.0), (2,0,0), normalizeGTO=true), 
         genBasisFunc([1.0, 1.0, 2.0], (3.0, 1.0), (0,0,0)), 
         genBasisFunc([1.0, 1.0, 1.0], (3.0, 1.0), (0,1,0), normalizeGTO=true)]
bs3 =   [genBasisFunc([1.0, 1.0, 1.0], (2.0, 1.0), (1,0,0), normalizeGTO=true), 
         genBasisFunc([1.0, 1.0, 1.0], (3.0, 1.0), (0,1,0), normalizeGTO=true), 
         genBasisFunc([1.0, 1.0, 1.0], (3.0, 1.0), (2,0,0), normalizeGTO=true), 
         genBasisFunc([1.0, 1.0, 2.0], (3.0, 1.0), (0,0,0), normalizeGTO=true)]
bs3_2 = [genBasisFunc([1.0, 1.0, 1.0], (2.0, 1.0), (1,0,0)), 
         genBasisFunc([1.0, 1.0, 1.0], (3.0, 1.0), (0,1,0)), 
         genBasisFunc([1.0, 1.0, 1.0], (3.0, 1.0), (2,0,0)), 
         genBasisFunc([1.0, 1.0, 2.0], (3.0, 1.0), (0,0,0))]

bfm_1 = +(bs2...,) # 1.7+ bug
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
gf_merge1 = GaussFunc(2.0, 1.0)
gf_merge2 = GaussFunc(2.0, 1.0)
gf_merge3 = GaussFunc(2.0, 1.0)

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

gf_merge3 = GaussFunc(1.5, 1.0)
@test hasIdentical(mergeGaussFuncs(gf_merge1, gf_merge3), [gf_merge1, gf_merge3])


# function add, mul, gaussProd
@test add(bs2[1]) === bs2[1]
bf1s = BasisFuncs(bf1.center, bf1.gauss, (bf1.ijk[1],), bf1.normalizeGTO)
@test hasIdentical(add(bf1s), bf1)
bfm_add1 = BasisFuncMix(bs2)
@test add(bfm_add1) == sumOf(bs2)
bf_add1 = genBasisFunc([1.0, 2.0, 1.0], (2.0, 1.1))
bf_add2 = genBasisFunc([1.0, 1.0, 1.0], (1.0, 1.2))
bf_add1_2 = BasisFuncMix(bf_add1)
bf_add2_2 = BasisFuncMix(bf_add2)
@test hasEqual(add(bf_add1, bf_add2), add(bf_add2, bf_add1))
@test hasEqual(add(bf_add1_2, bf_add2_2), add(bf_add2_2, bf_add1_2))
@test hasEqual(add(bf_add1_2, bf_add2), add(bf_add1, bf_add2))
@test hasEqual(add(bf_add1, bf_add2_2), add(bf_add1, bf_add2))
@test hasEqual(add(bf_add1_2, bf_add2_2), add(bf_add1, bf_add2))

for bs in (bs2, bs2_2, bs2_3)
    X = overlaps(bs)^(-0.5)
    bsNew = [mul.(bs, @view X[:,i]) for i in 1:size(X, 2)] .|> sum
    SNew = overlaps(bsNew)
    @test isapprox(SNew, LinearAlgebra.I, atol=1e-14)
end
@test hasEqual(mul(1.0, bf1), bf1)
@test hasEqual(mul(1.1, bf1), mul(bf1, 1.1))
xpn2 = genExponent(1.2)
con2 = genContraction(1.2, x->x^2)
bf_pf = genBasisFunc([1.0, 2.0, 3.0], GaussFunc(xpn2, con2))
bf_pf2 = (bf_pf*0.4)*5
@test bf_pf2.gauss[1].con.map isa Quiqbox.Pf{2.0}
@test hasEqual(bf_pf2, mul(bf_pf, 2.0))


α₁, α₂ = rand(1:0.01:10, 2)
d₁, d₂ = rand(2)
R₁ = rand(-2:0.01:2, 3)
R₂ = rand(-2:0.01:2, 3)
xr = -10:0.1:10
yr = -10:0.1:10
zr = -10:0.1:10
gf_mul = mul(GaussFunc(α₁, d₁), GaussFunc(α₂, d₂))
@test isapprox(gf_mul.xpn[], α₁ + α₂, atol=1e-14)
@test isapprox(gf_mul.con[], d₁ * d₂, atol=1e-14)
bl = true
for x in xr, y in yr, z in zr
    xv = [x, y, z]
    α₃, d₃, R₃ = gaussProd((α₁, d₁, R₁), (α₂, d₂, R₂))
    bl *= isapprox(d₁*exp(-α₁*sum(abs2, xv-R₁)) * d₂*exp(-α₂*sum(abs2, xv-R₂)), 
                   d₃*exp(-α₃*sum(abs2, xv-R₃)), atol=1e-15) # limit: atol=1e-20
end
@test bl

@test hasEqual(mul(bf1s, bf1s), [mul(bf1s[1], bf1s[1])])
@test hasEqual(mul(bf1s, bf1s, normalizeGTO=true)[], 
               mul(bf1s[1], bf1s[1], normalizeGTO=true))
bf_mul1 = genBasisFunc([1.0, 0.0, 0.0], (2.0, 3.0))
bf_mul2 = genBasisFunc([1.0, 0.0, 0.0], (1.5, 1.0))
bf_mul2_2 = genBasisFunc([1.0, 0.0, 0.0], (1.5, 1.0), normalizeGTO=true)
@test hasEqual(mul(bf_mul1, bf_mul2), mul(BasisFuncMix([bf_mul1]), BasisFuncMix([bf_mul2])))
@test hasEqual(mul(bf_mul1, bf_mul2, normalizeGTO=true), 
               mul(BasisFuncMix([bf_mul1]), BasisFuncMix([bf_mul2]), normalizeGTO=true))
bf_mul3 = genBasisFunc([1.0, 0.0, 0.0], ([1.0, 1.5], [0.3, 0.4]))
bf_mul4 = mul(bf_mul1, bf_mul2)
bf_mul4_0 = genBasisFunc([1.0, 0.0, 0.0], (3.5, 3.0))
@test hasEqual(bf_mul4, bf_mul4_0)
bf_mul5 = mul(bf_mul1, bf_mul2, normalizeGTO=true)
bf_mul5_0 = genBasisFunc([1.0, 0.0, 0.0], (3.5, 3.0), normalizeGTO=true)
@test hasEqual(bf_mul5, bf_mul5_0)
bf_mul6 = mul(bf_mul1, bf_mul2_2)
gfCoeffs = (3.5, round(3*getNorms(bf_mul2)[], digits=15))
bf_mul6_0 = genBasisFunc([1.0, 0.0, 0.0], gfCoeffs)
@test hasEqual(bf_mul6, bf_mul6_0)
bf_mul7 = mul(bf_mul1, bf_mul2_2, normalizeGTO=true)
bf_mul7_0 = genBasisFunc([1.0, 0.0, 0.0], gfCoeffs, normalizeGTO=true)
@test hasEqual(bf_mul7, bf_mul7_0)
bf_mul8 = mul(bf_mul1, bf_mul3)
bf_mul8_0 = genBasisFunc([1.0, 0.0, 0.0], ([3.0, 3.5], [0.9, 1.2]))
@test hasApprox(bf_mul8, bf_mul8_0)
bf_mul9 = mul(bf_mul1, bf_mul3, normalizeGTO=true)
bf_mul9_0 = genBasisFunc([1.0, 0.0, 0.0], ([3.0, 3.5], [0.9, 1.2]), normalizeGTO=true)
@test hasApprox(bf_mul9, bf_mul9_0)
bf_mul10 = genBasisFunc([1.0, 0.0, 0.0], (1.2, 0.3), (1,0,0))
bf_mul11 = genBasisFunc([0.0, 1.0, 0.0], (1.5, 1.0))
bf_mul12 = genBasisFunc([0.0, 1.0, 0.0], (1.5, 1.0), (1,0,0))
bf_mul13 = genBasisFunc([1.0, 1.0, 1.0], ([1.2, 1.0], [0.2, 0.5]))
bf_mul14 = genBasisFunc([1.0, 0.2, 1.0], ([1.2, 1.0], [0.2, 0.5]), (1,1,1))
bf_mul15 = mul(bf_mul10, BasisFuncMix([bf_mul1, bf_mul2]))
bf_mul15_0 = mul(bf_mul10, BasisFuncMix([bf_mul1, bf_mul2]), normalizeGTO=true)
bf_mul16 = mul(bf_mul15, bf_mul15_0)
bf_mul16_0 = mul(bf_mul15, bf_mul15_0, normalizeGTO=true)

testNorm = function (bf1, bf2)
    bf3 = mul(bf1, bf2, normalizeGTO=false)
    n1 = Quiqbox.getOverlap(bf3, bf3)
    n2 = Quiqbox.getOverlap(mul(bf1, bf1, normalizeGTO=false), 
                            mul(bf2, bf2, normalizeGTO=false))
    @test isapprox(n1, n2, atol=1e-12)
end

testNorm(bf_mul1, bf_mul10)
testNorm(bf_mul1, bf_mul11)
testNorm(bf_mul1, bf_mul12)
testNorm(bf_mul1, bf_mul13)
testNorm(bf_mul1, bf_mul14)
testNorm(bf_mul1, bf_mul15)
testNorm(bf_mul1, bf_mul15_0)
testNorm(bf_mul1, bf_mul16)
testNorm(bf_mul1, bf_mul16_0)
testNorm(bf_mul10, bf_mul11)
testNorm(bf_mul11, bf_mul13)
testNorm(bf_mul12, bf_mul13)
testNorm(bf_mul12, bf_mul14)
testNorm(bf_mul13, bf_mul14)
testNorm(bf_mul12, bf_mul15)
testNorm(bf_mul13, bf_mul15)
testNorm(bf_mul14, bf_mul15)
testNorm(bf_mul15, bf_mul15_0)
testNorm(bf_mul12, bf_mul16)
testNorm(bf_mul13, bf_mul16)
testNorm(bf_mul14, bf_mul16)
testNorm(bf_mul16, bf_mul16_0)
testNorm(bf_mul15_0, bf_mul16_0)


# function getOverlap
@test isapprox(Quiqbox.getOverlap(bf_mul2_2, bf_mul2_2), 1.0, atol=1e-10)
@test isapprox(Quiqbox.getOverlap(bf_mul5_0, bf_mul5_0), 9.0, atol=1e-10)
bf_mul18 = genBasisFunc(rand(3), "STO-3G")[]
@test isapprox(Quiqbox.getOverlap(bf_mul18, bf_mul18), 1.0, atol=1e-10)


# function shift
ijk = (1,0,0)
didjdk = (0,1,1)
bf_os1 = genBasisFunc(fill(0.0, 3), (2.0, 1.0), ijk)
bf_os2 = genBasisFunc(fill(0.0, 3), (2.0, 1.0), ijk, normalizeGTO=true)
bf_os1S = genBasisFunc(fill(0.0, 3), (2.0, 1.0), ijk.+didjdk)
bf_os2S = genBasisFunc(fill(0.0, 3), (2.0, 1.0), ijk.+didjdk, normalizeGTO=true)
bf_os3 = genBasisFunc(fill(0.0, 3), (2.0, 1.0), (2,0,0))
@test hasEqual(shift(bf_os1, didjdk), bf_os1S)
@test hasEqual(shift(bf_os2, didjdk), bf_os2S)
@test shift(bf_os2, didjdk, -) == Quiqbox.EmptyBasisFunc{Float64, 3}()
@test hasEqual(shift(bf_os3, ijk, -), bf_os1)


# function unpackBasis
@test unpackBasis(bfm1)[1] == bf1
@test unpackBasis(bf1) == (bf1,)
@test unpackBasis(Quiqbox.EmptyBasisFunc{Float64, 1}()) == ()


# function decompose
bf_d_1 = genBasisFunc([1.0, 0.0, 0.0], (1.0, 0.5))
bf_d_2 = genBasisFunc([1.0, 0.0, 0.0], ([2.0, 1.0], [0.1, 0.5]))
bf_d_3 = genBasisFunc([1.0, 1.0, 0.0], ([2.0, 1.0], [0.1, 0.2]), "P")
bm_d_1 = BasisFuncMix([bf_d_1, bf_d_2])
dm1 = reshape([bf_d_1], 1, 1)
@test hasIdentical(decompose(bf_d_1), dm1)
@test hasEqual(decompose(bf_d_1, true), dm1)
@test hasEqual(decompose(bf_d_2), reshape([bf_d_2], 1, 1))
dm2 = reshape([genBasisFunc([1.0, 0.0, 0.0], (2.0, 0.1)), bf_d_1], 2, 1)
@test hasEqual(decompose(bf_d_2, true), dm2)
@test hasEqual(decompose(bf_d_3), reshape(bf_d_3[:], 1, 3))
@test hasEqual(decompose(bf_d_3, true), 
               hcat(decompose.(bf_d_3[:], true)...))
@test hasIdentical(decompose(bm_d_1), reshape([bm_d_1], 1, 1))
@test hasEqual(decompose(bm_d_1, true), 
               vcat(decompose.(bm_d_1.BasisFunc, true)...))


# function basisSize
@test basisSize("P") == 3
@test basisSize.(["S", "P", "D"]) == [1, 3, 6]
@test basisSize(bf1) == 1
@test basisSize(bfm1) == 1 == basisSize(bfm2)
@test basisSize.((bf1, bf2, bf3_3, bf5)) == (1, 1, 3, 1)


# function genGaussFuncText
bfCoeff = [[6.163845031, 1.097161308], [0.4301284983, 0.6789135305], 
           [0.245916322, 0.06237087296], [0.0494717692, 0.9637824081],
           [0.245916322, 0.06237087296], [0.5115407076, 0.6128198961]]
bfCoeff2 = vcat([[bfCoeff[2i+1]'; bfCoeff[2i+2]']' for i=0:2]...)
content = """
S    2   1.0
         6.163845031                   0.4301284983
         1.097161308                   0.6789135305
S    2   1.0
         0.245916322                   0.0494717692
         0.06237087296                 0.9637824081
P    2   1.0
         0.245916322                   0.5115407076
         0.06237087296                 0.6128198961
"""
lines = (content |> IOBuffer |> readlines)
@test map(i->Quiqbox.genGaussFuncText(bfCoeff2[i,:]...), 1:size(bfCoeff2)[1] |> collect) == 
         [lines[2], lines[3], lines[5], lines[6], lines[8], lines[9]].*"\n"


# function genBasisFuncText & genBFuncsFromText
randElement = ElementNames[rand(1:length(ElementNames))]
bs1 = genBasisFunc(missing, ("6-31G", "H"), unlinkCenter=true)
cens = [rand(3) for _=1:length(bs1)]
txt1 = genBasisFuncText(bs1, printCenter=false, groupCenters=false) |> join
txt2 = genBasisFuncText(bs1, printCenter=false) |> join
bs2_1 = genBFuncsFromText(txt1)
bs2_2 = genBFuncsFromText(txt2)
assignCenInVal!.(cens, bs1)
assignCenInVal!.(cens, bs2_1)
assignCenInVal!.(cens, bs2_2)
txt3 = genBasisFuncText(bs1) |> join
bs2_3 = genBFuncsFromText(txt3)
@test hasEqual.(bs1, bs2_1, ignoreFunction=true) |> all
@test hasEqual.(bs1, bs2_2, ignoreFunction=true) |> all
@test hasEqual.(sortBasisFuncs(bs1), bs2_3, ignoreFunction=true) |> all


# function assignCenInVal!
bf6 = genBasisFunc(missing, "STO-3G")[]
coord = [1.0, 0.0, 0.0]
bf6_1 = genBasisFunc(coord, "STO-3G")[]
@test !hasEqual(bf6, bf6_1)
assignCenInVal!(coord, bf6)
@test hasEqual(bf6, bf6_1)


# function getParams
pb1 = ParamBox(2, :p)
@test getParams(pb1) == pb1
@test getParams(pb1, :p) == pb1
@test getParams(pb1, :P) === nothing
@test getParams(pb1, onlyDifferentiable=true) === nothing

pb2 = ParamBox(2, :q)
@test getParams([pb1, pb2]) == [pb1, pb2]
@test getParams([pb1, pb2], :p) == [pb1]

gf_pbTest1 = GaussFunc(2.0, 1.0)
@test getParams(gf_pbTest1) == gf_pbTest1.param |> collect

gf_pbTest2 = GaussFunc(1.5, 0.5)
bf_pbTest1 = genBasisFunc([1.0, 0.0, 0.0], [gf_pbTest1, gf_pbTest2])
@test getParams(bf_pbTest1) == [bf_pbTest1.center..., 
                                gf_pbTest1.param..., gf_pbTest2.param...]

alpha = ParamList[:xpn]
@test getParams(bf_pbTest1, alpha) == 
      vcat(getParams(gf_pbTest1, alpha), getParams(gf_pbTest2, alpha)) == 
      [gf_pbTest1.param[1], gf_pbTest2.param[1]]

cs = [pb1, bf_pbTest1]
ss = [:X, :Y, :Z]
@test reshape([getParams(i, j) for i in cs, j in ss], :) |> flatten == 
      [nothing, bf_pbTest1.center[1], nothing, bf_pbTest1.center[2], 
       nothing, bf_pbTest1.center[3]]
@test getParams(cs) == vcat(getParams(pb1), getParams(bf_pbTest1))
@test (getParams.(Ref(cs), ss) |> flatten) == (bf_pbTest1.center |> collect)


# function copyBasis
e = genExponent(3.0, x -> x^2 + 1)
c = genContraction(2.0)
gf_cb1 = GaussFunc(e, c)
gf_cb2 = copyBasis(gf_cb1)
@test !hasEqual(gf_cb1, gf_cb2)
@test hasEqual(gf_cb1.con, gf_cb2.con)
@test gf_cb1.xpn() == gf_cb2.xpn() == gf_cb2.xpn[] != gf_cb1.xpn[]
gf_cb3 = copyBasis(gf_cb1, false)
@test !hasEqual(gf_cb1, gf_cb3)
@test gf_cb1.xpn() != gf_cb3.xpn() == gf_cb3.xpn[] == gf_cb1.xpn[]
cen_cb1 = rand(3)

bf_cb1 = genBasisFunc(cen_cb1, [gf_cb1, gf_cb3])
bf_cb2 = copyBasis(bf_cb1)
bf_cb3 = copyBasis(bf_cb1, false)
testbf_cb = function (bf1, bf2)
    @test bf1.center == bf2.center
    @test bf1.ijk == bf2.ijk
    @test bf1.normalizeGTO == bf2.normalizeGTO
end
testbf_cb(bf_cb1, bf_cb2)
testbf_cb(bf_cb1, bf_cb3)
@test hasEqual(bf_cb2.gauss, (gf_cb2, gf_cb3))
@test hasEqual(bf_cb3.gauss, (gf_cb3, gf_cb3))

bfm_cb1 = BasisFuncMix([bf_cb1, bf_cb3])
bfm_cb2 = copyBasis(bfm_cb1)
bfm_cb3 = copyBasis(bfm_cb1, false)
testbf_cb.(bfm_cb1.BasisFunc, bfm_cb2.BasisFunc)
testbf_cb.(bfm_cb1.BasisFunc, bfm_cb3.BasisFunc)
@test hasEqual(bfm_cb2.BasisFunc |> collect, [bf_cb2, bf_cb3])
@test hasEqual(bfm_cb3.BasisFunc |> collect, [bf_cb3, bf_cb3])


# function markParams! getVar getVarDict
e_gv1 = genExponent(2.0)
c_gv1 = genContraction(1.0)
gf_gv1 = GaussFunc(e_gv1, c_gv1)
e_gv2 = genExponent(2.5)
c_gv2 = genContraction(0.5)
gf_gv2 = GaussFunc(e_gv2, c_gv2)
e_gv3 = genExponent(1.05, x->x^2)
c_gv3 = genContraction(1.5)
gf_gv3 = GaussFunc(e_gv3, c_gv3)
gf_gv12 = GaussFunc(e_gv1, c_gv2)
gf_gv23 = GaussFunc(e_gv2, c_gv3)
gf_gv31 = GaussFunc(e_gv3, c_gv1)
x_gv1 = fill(1.0)
y_gv1 = fill(5.0)
cen_gv1 = genSpatialPoint([x_gv1, 2.0, 3.0])
cen_gv2 = genSpatialPoint([4.0, y_gv1, 6.0])
cen_gv3 = genSpatialPoint([x_gv1, y_gv1, 6.0])

bf_gv1 = genBasisFunc(cen_gv1, gf_gv1)
bf_gv2 = genBasisFunc(cen_gv2, gf_gv2)
bf_gv3 = genBasisFunc(cen_gv2, [gf_gv1, gf_gv2])
bf_gv4 = genBasisFunc(cen_gv1, gf_gv3)
bf_gv5 = genBasisFunc(cen_gv3, gf_gv1)
bf_gv6 = genBasisFunc(cen_gv3, [gf_gv12, gf_gv23, gf_gv31])
bfm_gv = BasisFuncMix([bf_gv2, bf_gv4, bf_gv6])

@test markParams!(bf_gv4) == [cen_gv1..., e_gv3, c_gv3]

pbs_gv1 = markParams!([bf_gv1, bf_gv2])
pbs_gv1_0 = [cen_gv1[1], cen_gv2[1], 
             cen_gv1[2], cen_gv2[2], 
             cen_gv1[3], cen_gv2[3], 
                  e_gv1,      e_gv2, 
                  c_gv1,      c_gv2]
@test pbs_gv1 == pbs_gv1_0
@test hasIdentical(pbs_gv1, pbs_gv1_0)

pbs_gv2 = markParams!([bf_gv2, bf_gv1, bf_gv3])

pbs_gv2_0 = [cen_gv2[1], cen_gv1[1], cen_gv2[1], 
             cen_gv2[2], cen_gv1[2], cen_gv2[2], 
             cen_gv2[3], cen_gv1[3], cen_gv2[3], 
             e_gv2, e_gv1, e_gv1, e_gv2,
             c_gv2, c_gv1, c_gv1, c_gv2]
@test pbs_gv2 == pbs_gv2_0
@test hasIdentical(pbs_gv2, pbs_gv2_0)

bfs_gv = sortBasisFuncs([bf_gv1, bf_gv2, bf_gv3, bf_gv4, bf_gv5, bf_gv6])
pbs_gv3 = markParams!(bfs_gv, true)
pbs_gv4 = markParams!(vcat(bfs_gv, bfm_gv), true)
@test length(pbs_gv3) == 13
@test pbs_gv3 == pbs_gv4
@test hasIdentical(pbs_gv3, pbs_gv4)

pbs_gv0 = [cen_gv1[1], cen_gv2[1], 
           cen_gv1[2], cen_gv2[2], 
           cen_gv1[3], cen_gv2[3], cen_gv3[3], 
           e_gv1, e_gv2, e_gv3, 
           c_gv1, c_gv2, c_gv3]
pbs_gv0_2 = sort(pbs_gv0, by=x->(typeof(x).parameters[2], x.index[]))
pbs_gv3_2 = sort(pbs_gv3, by=x->(typeof(x).parameters[2], x.index[]))
@test pbs_gv3_2 == pbs_gv0_2
@test hasEqual(pbs_gv3_2, pbs_gv0_2)
@test hasIdentical(pbs_gv3_2, pbs_gv0_2)

@test getVar(e_gv1).val.name == :α₁
@test getfield.(getfield.(getVar(bf_gv6), :val), :name) == 
      [:X₁, :Y₂, :Z₂, :α₁, :d₃, :α₃, :d₂, :x_α₂, :d₁]
@test getfield.(getfield.(getVar(bfm_gv), :val), :name) == 
      [:X₁, :Y₁, :Z₁, :x_α₂, :d₂, :X₁, :Y₂, :Z₂, :α₁, 
       :d₃, :α₃, :d₂, :x_α₂, :d₁, :X₂, :Y₂, :Z₃, :α₃, :d₃]

@test getVarDict(e_gv1) == Dict(Symbolics.variable(:α, 1)=>2.0)

pairs_gv1 = sort(getVarDict(e_gv3, includeMapping=true) |> collect, by=x->string(x[1]))
strs_gv1 = getindex.(pairs_gv1, 1) .|> string
strs_gv2 = getindex.(pairs_gv1, 2) .|> string

@test strs_gv1[1][1:3] == "f_α" && strs_gv1[1][end-8:end] == "(x_α₂)"
@test strs_gv1[2:end] == ["x_α₂"]
@test strs_gv2 == ["1.1025", "1.05"]

@test getVarDict(gf_gv1) == Dict( Pair.(Symbolics.variable.([:α, :d], fill(1, 2)), 
                                        [2.0, 1.0]) )

@test getVarDict(bf_gv1) == Dict( Pair.(Symbolics.variable.([:X, :Y, :Z, :α, :d], 
                                                            fill(1, 5)), 
                                        [1.0, 2.0, 3.0, 2.0, 1.0]) )

@test getVarDict(bf_gv1) == getVarDict([bf_gv1, gf_gv1])
@test getVarDict([bf_gv1, bf_gv2]) == 
      merge(getVarDict(bf_gv1), getVarDict(bf_gv2)) ==
      getVarDict([cen_gv1..., cen_gv2..., gf_gv1, gf_gv2]) == 
      getVarDict(vcat(bf_gv1.param|>collect, bf_gv2.param|>collect))
@test getVarDict(bfm_gv) == 
      getVarDict([bf_gv2, bf_gv4, bf_gv6]) == 
      getVarDict(pbs_gv0)


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
@test isapprox(parse(Float64, expStr[1:idx-1]), 2.1381164110649706, atol=1e-12)
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
func1(X,Y,Z) = log(X+0.5Y*Z)
expr = func1(X,Y,Z)
vals = [1, 3.334, -0.2]
d1 = Dict(vars .=> vals)
errT = 1e-10
@test varVal.(vars, Ref(d1)) == vals
@test isapprox(varVal(sum(vars), d1), sum(vals), atol=errT)
@test isapprox(varVal(prod(vars), d1), prod(vals), atol=errT)
@test isapprox(varVal(vars[1]^vars[2], d1), vals[1]^vals[2], atol=errT)
@test isapprox(varVal(expr, d1), func1(vals...), atol=errT)

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