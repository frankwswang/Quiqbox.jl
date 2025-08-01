using Test
using Quiqbox
using LinearAlgebra
using Quiqbox: TypedReduce, TypedExpand, getScreenLevelOptionsCore, getCellOutputLevels, 
               UnitParam, GridParam, getOutputType

@testset "Parameters.jl" begin

pv1 = genTensorVar(1, :a)
pv2 = genTensorVar(1, :a)
@test Quiqbox.ParamMarker(pv1) != Quiqbox.ParamMarker(pv2)
@test !Quiqbox.compareParamBox(pv1, pv2)
pvs = [pv1, pv2]
@test getfield.(Quiqbox.markerOf.(pvs), :index) == [0, 0]
sortParams!(pvs)
@test getfield.(Quiqbox.markerOf.(pvs), :index) == [1, 2]

absTyped1 = TypedReduce(abs, Float64)
tf_data1 = -1.1
@test absTyped1(tf_data1) == abs(tf_data1)
absTyped2 = TypedReduce(absTyped1, Float64)
@test absTyped2 === absTyped1

sumTyped1 = TypedReduce(sum, Float32)
tf_data2 = [1.1, -2.3, 3.3]
@test sumTyped1(tf_data2.|>Float32) == (Float32∘sum)(tf_data2)
sumTyped2 = TypedReduce(sumTyped1, Float32)
@test sumTyped2 === sumTyped1

tf_data3 = [0.055661728153055035 0.3849962858078353 0.7524819058655356; 
            0.17058774687118927 0.3552668837192795 0.5032426334642225]
mapCosTyped1 = TypedExpand(x->cos.(x), eltype(tf_data3), (tf_data3,))
@test mapCosTyped1(tf_data3) == cos.(tf_data3)
mapCosTyped2 = TypedExpand(mapCosTyped1)
@test mapCosTyped2 === mapCosTyped1

matMulTyped1 = TypedExpand(*, Float64, (tf_data3, tf_data2))
tf_data3_2 = rand(5,4)
tf_data2_2 = rand(4)
@test try
    matMulTyped1(tf_data3_2, tf_data2_2)
    false
catch
    true
end
matMulTyped1_2 = TypedExpand(*, Float64, (tf_data3, tf_data2), truncate=true)
@test matMulTyped1_2(tf_data3_2, tf_data2_2) == (tf_data3_2 * tf_data2_2)[begin:begin+1]
matMulTyped2 = TypedExpand(matMulTyped1)
@test matMulTyped2 == matMulTyped1

evenExpand = x-> [x^i for i in 1:5]
evenExpandTyped1 = TypedExpand(evenExpand, Float64, (1.1,))
@test evenExpandTyped1(-1.2) == evenExpand(-1.2)
evenExpandTyped2 = TypedExpand(evenExpandTyped1)
@test evenExpandTyped2 === evenExpandTyped1

@test getScreenLevelOptionsCore(Array) == (0,)
@test getScreenLevelOptionsCore(Tuple{Int}) == (0, 1, 2)
dMemType = Quiqbox.DirectMemory{Int, 2}
@test getScreenLevelOptionsCore(dMemType) == (0, 1, 2)
@test getScreenLevelOptionsCore(Quiqbox.DirectMemory{Tuple{Int}, 1}) == (0, 1, 2)
@test getScreenLevelOptionsCore(Quiqbox.PackedMemory{Int, dMemType}) == (0,)

p1 = genTensorVar(1, :a)
p2 = genTensorVar([2], :b)
p3 = genHeapParam([p1, p1], :c)
p4 = genHeapParam([p2, p2], :d)
p5 = genCellParam(p2, :e)
p6 = genCellParam(p4, :e)
@test getCellOutputLevels((p1, p1)) == (0,)
@test getCellOutputLevels((p1, p2)) == (0,)
@test getCellOutputLevels((p2, p2)) == (0, 1)
@test getCellOutputLevels((p1, p3)) == (0,)
@test getCellOutputLevels((p2, p3)) == (0, 1)
@test getCellOutputLevels((p3, p3)) == (0, 1)
@test !((p1, p4) isa Quiqbox.CoreFixedParIn)
@test getCellOutputLevels((p2, p4)) == (1,)
@test getCellOutputLevels((p3, p4)) == (1,)
@test getCellOutputLevels((p4, p4)) == (1, 2,)
@test getCellOutputLevels((p5, p5)) == (0, 1,)
@test getCellOutputLevels((p5, p6)) == (1,)
@test getCellOutputLevels((p6, p6)) == (1, 2,)

v1Val = 0.5
v1 = genTensorVar(v1Val, :α)
@test symbolOf(v1) == :α
v2Val = 1.1
v2 = genTensorVar(v2Val, :β)
@test obtain(v2) === v2.data[] === v2()
v1_g, s1 = transpileParam(v1)
@test s1.unit[] === v1
@test s1 == Quiqbox.initializeSpanParamSet(v1)
@test s1.unit === first(s1) && s1.grid === last(s1)
v1_f = functionalize(v1_g)
v1_f2 = ParamGraphCaller(v1)
s1v = obtain(s1)
@test try v1_f((unit=nothing, grid=nothing)); catch; true end
@test v1_f(s1v) === obtain(v1) === v1Val
@test v1_f2((unit=nothing, grid=nothing)) === v1_f2(s1v) === v1_f2()
@test v1_f([1.0]) == v1_f2([1.0]) == 1.0 != v1()
a1 = genCellParam(v1, :a)
a1_2 = genCellParam(itself, (v1,), :a1)
@test a1 != a1_2
@test Quiqbox.markObj(a1) == Quiqbox.markObj(a1_2)
@test compareParamBox(a1, a1_2)
@test a1.lambda == a1_2.lambda
@test a1.offset[] === zero(Float64)
@test symbolOf(a1) == :a
@test symbolOf(a1_2) == :a1
@test a1() == v1Val
@test obtain(a1) == v1Val
@test screenLevelOf(a1) == 0

@test compareParamBox(v1, v1)
@test !compareParamBox(v1, genTensorVar(v1Val, :α))
@test compareParamBox(a1, a1)
@test compareParamBox(a1, genCellParam(v1, :a))
@test !compareParamBox(a1, let p = genCellParam(v1, :a); setScreenLevel!(p, 1) end)

a1in, a1mid, a1out, a1self = dissectParam(a1)
@test isempty(a1mid)
@test isempty(a1self)
@test first(a1in)[] === a1in.unit[] === v1
@test a1out[] === a1
@test a1.input isa Tuple{Quiqbox.TensorVar{Float64}}
@test (first∘first∘dissectParam)(a1)[] === first(a1.input)
@test a1.offset[] == 0
a1Offset = 0.25
a1.offset[] = a1Offset
a1Val = v1Val + a1Offset
@test a1() == a1Val
@test obtain(a1) == a1Val
@test obtain(first(a1|>inputOf)) == v1Val

a2 = setScreenLevel!(genCellParam(a1), 1)
@test a2() == a1()
@test screenLevelOf(a2) == 1
@test a1.input == a2.input
@test obtain(inputOf(a2)[1]) != obtain(a2)
a2in, _, a2out, a2self = dissectParam(a2)
@test all(isempty(i) for i in a2in) && isempty(a2out)
@test collect(a2self)[] === a2
@test obtain(a2) == a2.offset[]
@test a2.offset[] == a1Val == a1.offset[] + v1Val

v1ValNew = 0.9
v1_2 = genTensorVar(v1(), symbolOf(v1), true)
@test try setVal!(v1_2, v1ValNew) catch; true end
setVal!(v1, v1ValNew)
@test obtain(v1) == v1ValNew
a1.offset[] = 0.0
@test obtain(a1) == v1ValNew
@test a2.offset[] == obtain(a2) == v1Val + a1Offset
@test obtain(inputOf(a2)[1]) == v1ValNew
@test obtain(a2) == a1Val
setScreenLevel!(a2, 0)
@test a1.input[1] === a2.input[1]
@test obtain(a2) == a1Val
@test a2.offset[] == a1Val - a1() == a2() - v1ValNew
setScreenLevel!(a2, 1)
@test a1.input[1] === a2.input[1]
@test obtain(a2) == a1Val == a2.offset[]
@test obtain(inputOf(a2)[1]) == v1ValNew

t1Val = (1.1, 2.2, 3.3)
t1 = genTensorVar(t1Val, :t1)
map1 = x->x.^2
a3 = genCellParam(map1, (t1,), :T1)
a3Val = obtain(a3)
@test a3Val == map1(t1Val)
@test a3.offset[] == (0.0, 0.0, 0.0)
setScreenLevel!(a3, 2)
@test a3.offset[] == a3Val
@test a3.offset[] == a3Val == obtain(a3)
setScreenLevel!(a3, 1)
@test a3.offset[] == a3Val
@test a3.offset[] == a3Val == obtain(a3)
setScreenLevel!(a3, 0)
@test a3.offset[] == (0.0, 0.0, 0.0)

struct myS{T}
    a::T
end
ms_t1Val = myS.(t1Val)
ms1 = genTensorVar(ms_t1Val, :ms)
map1_2 = x->myS.(getproperty.(x, :a) .^ 2)
a3_2 = genCellParam(map1_2, (ms1,), :MS)
@test getproperty.(obtain(a3_2), :a) == a3Val
@test Quiqbox.checkParamOffsetMethods(typeof(ms_t1Val)) == false
@test try setScreenLevel!(a3_2, 2); false catch; true end

t2Val = -1 .* t1Val
t2 = genTensorVar(t2Val, :t)
map2 = (x, y)-> x .+ y
a4 = genCellParam(map2, (t1, t2), :T)
t12 = genHeapParam([t1, t2], :t12)
a4_2 = genCellParam(x->x[1].+x[2], (t12,), :T)
@test all(obtain(a4) .== 0 .== obtain(a4_2))
n1 = genCellParam(t12, :N)
n1_2 = genCellParam(n1, :N)
@test n1.lambda == n1_2.lambda
@test obtain(n1) == obtain(t12) == [t1Val, t2Val]
n2 = genCellParam(x->[x[2], x[1]], (n1_2,), :N)
@test obtain(n2) == [t2Val, t1Val]
n1_3 = genCellParam(x->map(y->y.^2, x), (n1_2,), :N)
@test n1_3() == map(y->y.^2, obtain.([t1, t2]))

f1 = x->x[1]*x[2]^3+1.0
v12 = genHeapParam([v1, v2], :v12)
b1 = genCellParam(f1, (v12,), :b)
@test b1.offset[] == 0
b1Offset = 0.1
b1.offset[] = b1Offset
f1_bias = (x)->(f1(x) + b1Offset)
b1Val = f1_bias( obtain(v12) )
@test b1() == b1Val
@test obtain(b1) == b1Val
@test screenLevelOf(b1) == 0
@test all(inputOf(b1)[1] === v12)

b2 = setScreenLevel!(genCellParam(b1), 1)
b3 = setScreenLevel!(genCellParam(b1), 2)
@test b1() == b2() == b3()
@test b1Val == b2() == b3()
@test screenLevelOf(b2) == 1
@test screenLevelOf(b3) == 2
@test b2.offset[] == obtain(b2)
@test b3.offset[] == obtain(b3)

f2(x) = 2x[1] * exp(x[1] + x[5]) - x[2]/(1 + (x[3] - x[4])^2)
pars_f2 = genHeapParam([a1, v2, b1, b2, b3], :pars_f2)

c1 = genCellParam(f2, (pars_f2,), :c)
inSet1, midSet1, outSet1, isoSet1 = dissectParam(c1)
@test isempty(isoSet1)
@test outSet1[] === c1
@test isempty(inSet1.grid)
@test all(x==() for x in Quiqbox.getOutputSize.(inSet1.unit))
sortParams!(inSet1.unit)
@test all(inSet1.unit .=== [b2, v1, v2, b3])

b3Val1 = b3()
f2_t = function (x)
    2x[1] * exp(x[1] + b3Val1) - x[2] / (1 + (f1_bias([x[1], x[2]]) - x[3])^2)
end
inSet1_t = [v1, v2, b2, b3]
c1Val = f2_t(obtain.(inSet1_t))
@test obtain(pars_f2) == obtain.([a1, v2, b1, b2, b3])
@test f2(obtain(pars_f2)) == c1Val == c1() == 42.96591869214117
inSet_c1_raw = dissectParam(c1)[begin]
@test inSet_c1_raw.unit == inSet1_t
gn_c1, inSet_c1 = transpileParam(c1)
@test all([b2, v1, v2] .=== inSet_c1.unit)
c1_input1 = obtain(inSet_c1).unit
@test inSet_c1.grid == Quiqbox.genBottomMemory()
@test c1_input1 == map(x->broadcast(Quiqbox.getVertexValue, x), gn_c1.source).unit[1:3]

cm1 = genTensorVar(collect(1:5), :cm, true)
cm2 = genTensorVar(collect(6:10), :cm, true)
cm3 = genTensorVar(collect(11:15), :cm, true)
cm4 = genTensorVar(collect(16:20), :cm, false)
cm5 = genCellParam(+, (cm1, cm2), :cm)
cm6 = genCellParam(+, (cm3, cm4), :cm)
cm7 = genCellParam((x, y)->sum(x.*y), (cm5, cm6), :cm)
@test cm7() == sum(collect(7:2:15) .* collect(27:2:35))
g_cm7, inSet_cm7 = transpileParam(cm7)
@test inSet_cm7.grid[] === cm4

gnf_c1 = functionalize(gn_c1)
@test Quiqbox.getInputSetType(gnf_c1) == Quiqbox.UnitInput
c1Val_2 = gnf_c1((unit=c1_input1, grid=Quiqbox.genBottomMemory()))
gnf_c1_2 = ParamGraphCaller(c1)
@test c1Val_2 == gnf_c1_2(c1_input1) == gnf_c1_2((unit=c1_input1, grid=nothing))
@test gnf_c1_2() == c1() == c1Val_2 == 42.96591869214117

k1 = genTensorVar(1.1, :k1)
k2 = genTensorVar(3.2, :k2)
l1 = genCellParam(x->x^2, (k1,), :l)
l2 = genCellParam(x->2x, (k2,), :l)
l3 = genCellParam(+, (l1, l2), :l)
@test l3() == 1.1^2 + 3.2 * 2
g_l1, g_l1_inSet = transpileParam(l1)
g_l3, g_l3_inSet = transpileParam(l3)
g_l3inVal = (unit=[k1(), k2()], grid=nothing)
@test obtain(g_l3_inSet) == g_l3inVal
g_l3_val = evaluateGraph(g_l3)
gf_l3 = ParamGraphCaller(l3)
@test g_l3_val == gf_l3(g_l3inVal)
@test all(map((x, y)->obtain(x)==y, g_l3_inSet, ([k1(), k2()], [])))
@test gf_l3([1.1, 3.2]) == gf_l3((unit=[1.1, 3.2], grid=nothing)) == 7.61
@test gf_l3([3.2, 1.1]) ≈ 12.44

# Test infinite loop avoidance during recursive function calls
v1a1 = genHeapParam([v1, a1], :va)
b4 = genCellParam(f1, (v1a1,), :b4)
b4Val0 = f1( obtain(v1a1) )
@test inputOf(b4)[begin] === v1a1
v1a1.input[end] = b4
@test try  b4()
catch err
    err isa AssertionError
end

m1Val = [0.8500493898201774 0.8371857655136232 0.8261332197426693; 
         0.8909368725700677 0.028692823163915082 0.8609921365095632; 
         0.15601079124513528 0.8390087083772936 0.539265546021646];
m1 = genTensorVar(m1Val, :m)
@test obtain(m1) == m1Val
g1 = sum
e1 = genCellParam(g1, (m1,), :e)
@test obtain(e1) == g1(m1Val)
m2Val = [0.5878290913996813 0.9722792211623517 0.9553534523311727 0.7753564819171315 0.47273331783546524; 
         0.004178648753904168 0.6065316448778457 0.35980863893594073 0.520655497909379 0.8223591668576677; 
         0.020435742071468366 0.3152758906745252 0.007076181436746043 0.15802943754751997 0.3922026235649726];
m2 = genTensorVar(m2Val, :m)
g2 = ((x::AbstractMatrix{T}, y::AbstractMatrix{T}) where {T}) -> norm(x * y)::T
e2 = genCellParam(g2, (m1, m2), :e)
@test obtain(e2) == g2(m1Val, m2Val)
inSet_e2, midSet_e2, outSet_e2, isoSet_e2 = dissectParam(e2)
@test isempty(inSet_e2.unit)
@test all(inSet_e2.grid .=== [m1, m2])
@test outSet_e2[] == e2
@test isempty(isoSet_e2)

gn_e1, inSet_e1 = transpileParam(e1)
inVal_e1 = obtain(inSet_e1)
@test inVal_e1.unit === nothing && inVal_e1.grid[] == m1Val
pVals_e1_ext = obtain.(inSet_e2.grid)
gnf_e1 = functionalize(gn_e1)
@test evaluateGraph(gn_e1) == gnf_e1(inVal_e1) == 5.82827525296409
@test gnf_e1((unit=nothing, grid=pVals_e1_ext)) == 5.82827525296409
@test gnf_e1(pVals_e1_ext) == gnf_e1(inVal_e1.grid) == 5.82827525296409
f3 = (x, y, z)->log(x^2 + y[1]) * (y[2] - sqrt(norm(exp.(z))))
f3_t = (xv, z)->log(xv[1]^2 + xv[2]) * (xv[3] - sqrt(norm(exp.(z))))
v2a2 = genHeapParam([v2, a2], :v2a2)
d1 = genCellParam(f3, (a1, v2a2, m1), :d)
d2 = setScreenLevel!(genCellParam(d1), 1)
d1_val = d1()
@test d1_val == d2() == f3(a1(), [v2(), a2()], m1())
inSet2, midSet2, outSet2, isoSet2 = dissectParam(d1)
@test all(inSet2.unit .=== [v1, v2, a2])
@test all(inSet2.grid .=== [m1])
@test outSet2[] === d1
@test isempty(isoSet2)

gn_d1, inSet_d1 = transpileParam(d1)
@test IdSet{UnitParam}(inSet_d1.unit) == IdSet{UnitParam}(inSet2.unit)
@test IdSet{GridParam}(inSet_d1.grid) == IdSet{GridParam}(inSet2.grid)
gnf_d1 = functionalize(gn_d1)
inVal_d1 = obtain(inSet_d1)
@test evaluateGraph(gn_d1) == gnf_d1(inVal_d1) == d1()

pVec1 = genHeapParam([c1, d1, a2], :c1d1a2)
apRef1 = genCellParam(pVec1, :ref)
pVec1Val = obtain(pVec1)
@test obtain(apRef1) == pVec1Val == [42.96591869214117, -1.1243289206151053, 0.75]

inSet5r, _, outSet5r, isoSet5r = dissectParam(pVec1.input)
inSet5,  _, outSet5,  isoSet5  = dissectParam(apRef1)

@test isempty(isoSet5)
@test all(outSet5r .=== [c1, d1])
@test isoSet5r[] === a2
@test all(vcat(inSet5r[1], a2) .=== inSet5[1])
@test inSet5r.grid[] === inSet5.grid[]

f3 = x -> x.^2 * ( exp.(x) )'
ap1_1 = genCellParam(f3, (pVec1,), :sq)
ap1_2 = genCellParam(f3, (apRef1,), :sq)
@test try genMeshParam(f3, (pVec1,), :sq); catch; true end
@test obtain(ap1_1) == obtain(ap1_2) == f3(pVec1Val)

f4 = (x, y) -> x^2 * ( exp.(y) )'
mp1 = genMeshParam(f4, (v1, pVec1), :mp1)
@test mp1() == f4(v1(), pVec1())
mp1_2 = genMeshParam(mp1, :mp1)
setScreenLevel!(mp1_2, 1)
@test mp1_2() == mp1()
setVal!(mp1_2, -mp1_2())
@test mp1_2() + mp1() == zero(mp1())

pg1Input = [a1 a2; c1 d1]

pg1 = genHeapParam(pg1Input, :pl)

@test obtain(pg1) == obtain.(pg1.input) == obtain.(pg1Input)
@test compareParamBox(genHeapParam(pg1), pg1)
pg1_2 = genHeapParam(copy(pg1Input), Quiqbox.markerOf(pg1))
pg1_3 = genHeapParam(pg1Input, Quiqbox.markerOf(pg1))
pg1_4 = genHeapParam(pg1)
@test compareParamBox(pg1_2, pg1)
@test compareParamBox(pg1_3, pg1)
@test compareParamBox(pg1_4, pg1)

makeGrid(l, cen, ns) = begin
    l = l[]
    lens = ns .* l
    coordBegin = cen .- lens ./ 2
    iterRange = Iterators.product(range.(0, ns)...)
    map(iterRange) do i
        (coordBegin .+ i .* l)'
    end
end

f5 = (l, c) -> makeGrid(l, c, [1.,2.,3.])
k1 = genTensorVar([1.0], :l)
k2 = genTensorVar([0., 0., 0.], :c)
pm1 = genMeshParam(f5, (k1, k2), :pn)
@test try genCellParam(f5, (k1, k2), :pn); catch; true end
pm1Val = obtain(pm1)
@test  pm1Val == obtain(pm1) == f5(k1(), k2())
@test !isdefined(pm1, :offset)
Quiqbox.ParamMarker(pm1) #> Check if `ParamMarker` can be property instantiated

inSet_pm1, _, outSet_pm1, isoSet_pm1 = dissectParam(pm1)
@test inSet_pm1.grid == [k1, k2]
inSetVal_pm1 = obtain.(inSet_pm1.unit)

gn_pm1, inSet_pm1 = transpileParam(pm1)
gnf_pm1 = functionalize(gn_pm1)
inVal_pm1 = obtain(inSet_pm1)
@test pm1Val == evaluateGraph(gn_pm1) == gnf_pm1(inVal_pm1) == gnf_pm1(inVal_pm1.grid)

val_pm1_1 = obtain(pm1)
@test all(size.(val_pm1_1).== Ref( (1, 3) ))

f6 = (l, c) -> map(norm, f5(l, c))
pm2 = genCellParam(f6, genCellParam.((k1, k2)), :pn)
pm2Val = obtain(pm2)

inSet_pm2_1, _, outSet_pm2, isoSet_pm2 = dissectParam(pm2)
inSetVal_pm2 = obtain(inSet_pm2_1)
gn_pm2, inSet_pm2 = transpileParam(pm2)
gnf_pm2 = ParamGraphCaller(pm2)
inVal_pm2 = obtain(gnf_pm2.source)
@test inVal_pm2 == map(x->(x!==nothing ? sort(x) : x), inSetVal_pm2) == obtain(inSet_pm2)
gnf_pm2core = functionalize(gn_pm2)
@test gnf_pm2.evaluate == gnf_pm2core
@test evaluateGraph(gn_pm2) == gnf_pm2(inVal_pm2) == pm2() == pm2Val
@test gnf_pm2core(inVal_pm2) == gnf_pm2(inVal_pm2.grid) == pm2Val
x1 = genTensorVar(1.0, :x)
x2 = genTensorVar(1.1, :x)
x3 = genTensorVar(1.2, :x)
v1 = genHeapParam([x1, x2, x3], :v)
@test v1() isa getOutputType(v1)
v2 = genTensorVar([2.0], :v)
@test v2() isa getOutputType(v2)
v3 = genHeapParam([v1, v2], :v)
@test v3() isa getOutputType(v3)
v4 = genHeapParam([v3], :v)
@test v4() isa getOutputType(v4)
v5 = genHeapParam([v4], :v)
@test v5() isa getOutputType(v5)
c1 = genCellParam(x1, :c)
c1 |> obtain
c2 = genCellParam(1.1, :c)
c2 |> obtain
c3 = genCellParam(first, (v4,), :c)
c3 |> obtain
c4 = genCellParam((x, y, z)->sum(x)-sum(y) + z, (v1, v2,  x1), :c)
@test (c4 |> obtain) == sum(obtain(v1)) - sum(obtain(v2)) + obtain(x1)

m1 = genCellParam(v1, :m)
@test (m1 |> obtain) == obtain(v1)

m2 = genMeshParam(x->[x, x+1], (c2,), :m)
@test obtain(m2) == [1.1, 2.1]

# indexParam
xMat = genTensorVar(rand(3,3), :x)
xEle = Quiqbox.indexParam(xMat, 5, :xpn5)
f_xEle, ps_xEle = Quiqbox.genParamMapper((pb=xEle,))
ps_xEle == Quiqbox.initializeSpanParamSet(xMat)
f_xEle(obtain(ps_xEle)) == (pb=xMat.data[][5],) == (pb=obtain(xEle),)

xpn1 = genCellParam(1.1, :xpn)
setScreenLevel!(xpn1, 2)
gf1 = Quiqbox.GaussFunc(xpn1)
fCore, pSet = Quiqbox.unpackFunc(sqrt ∘ gf1)
@test fCore isa Quiqbox.VoidParamBindFunc
@test pSet isa Quiqbox.OptSpanParamSet

end