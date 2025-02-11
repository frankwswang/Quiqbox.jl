using Test
using Quiqbox
using LinearAlgebra

@testset "Parameters.jl" begin

tf_data1 = -1.1
absTyped1 = Quiqbox.TypedReduction(abs, tf_data1)
@test absTyped1(tf_data1) == abs(tf_data1)
absTyped2 = Quiqbox.TypedReduction(absTyped1, tf_data1)
@test absTyped2 === absTyped1

tf_data2 = [1.1, -2.3, 3.3]
sumTyped1 = Quiqbox.TypedReduction(sum, tf_data2)
@test sumTyped1(tf_data2) == sum(tf_data2)
sumTyped2 = Quiqbox.TypedReduction(sumTyped1, tf_data2)
@test sumTyped2 === sumTyped1

tf_data3 = [0.055661728153055035 0.3849962858078353 0.7524819058655356; 
            0.17058774687118927 0.3552668837192795 0.5032426334642225]
mapCosTyped1 = Quiqbox.StableMorphism(x->cos.(x), tf_data3)
@test mapCosTyped1(tf_data3) == cos.(tf_data3)
mapCosTyped2 = Quiqbox.StableMorphism(mapCosTyped1, tf_data3)
@test mapCosTyped2 === mapCosTyped1

matMulTyped1 = Quiqbox.StableMorphism(*, tf_data3, tf_data2)
tf_data3_2 = rand(5,4)
tf_data2_2 = rand(4)
@test matMulTyped1(tf_data3_2, tf_data2_2) == tf_data3_2 * tf_data2_2
matMulTyped2 = Quiqbox.StableMorphism(matMulTyped1, tf_data3_2, tf_data2_2)
@test matMulTyped2 == matMulTyped1

evenExpand = x-> [x^i for i in 1:5]
evenExpandTyped1 = Quiqbox.StableMorphism(evenExpand, zero(Float64))
@test evenExpandTyped1(-1.2) == evenExpand(-1.2)
evenExpandTyped2 = Quiqbox.StableMorphism(evenExpandTyped1, zero(Float64))
@test evenExpandTyped2 === evenExpandTyped1

v1Val = 0.5
v1 = TensorVar(v1Val, :α)
@test symOf(v1) == :α
v2Val = 1.1
v2 = TensorVar(v2Val, :β)
@test obtain(v2) === Quiqbox.directObtain(v2.input) === v2()

f1 = x->x[1]*x[2]^3+1.0

a1 = CellParam(v1, :a)
a1_2 = CellParam(identity, v1, :a1)
@test a1.lambda == a1_2.lambda
@test a1.offset === zero(Float64)
@test symOf(a1) == :a
@test symOf(a1_2) == :a1
@test a1() == v1Val
@test obtain(a1) == v1Val
@test screenLevelOf(a1) == 0
a1in, a1out, a1self = markParams!(a1)
@test isempty(a1self)
@test a1in[][] === v1
@test a1out[] === a1
@test inputOf(a1) isa Tuple{TensorVar{Float64}}
@test markParams!(a1)[1][][] === first(inputOf(a1)) === first(a1.input)
@test a1.offset == 0
a1Offset = 0.25
@atomic a1.offset = a1Offset
a1Val = v1Val + a1Offset
@test a1() == a1Val
@test obtain(a1) == a1Val
@test obtain(first(inputOf(a1))) == v1Val

a2 = setScreenLevel(a1, 1)
@test a2() == a1()
@test screenLevelOf(a2) == 1
@test a1.input == a2.input
@test obtain(inputOf(a2)[1]) != obtain(a2)
a2in, a2out, a2self = markParams!(a2)
@test isempty(a2in) && isempty(a2out)
@test a2self[] === a2
@test obtain(a2) == a2.offset
@test a2.offset == a1Val == a1.offset + v1Val

v1ValNew = 0.9
setVal!(v1, v1ValNew)
@test obtain(v1) == v1ValNew
@atomic a1.offset = 0.0
@test obtain(a1) == v1ValNew
@test a2.offset == obtain(a2) == v1Val + a1Offset
@test obtain(inputOf(a2)[1]) == v1ValNew
@test obtain(a2) == a1Val
setScreenLevel!(a2, 0)
@test a1.input[1] === a2.input[1]
@test obtain(a2) == a1Val
@test a2.offset == a1Val - a1() == a2() - v1ValNew
setScreenLevel!(a2, 1)
@test a1.input[1] === a2.input[1]
@test obtain(a2) == a1Val == a2.offset
@test obtain(inputOf(a2)[1]) == v1ValNew

t1Val = (1.1, 2.2, 3.3)
t1 = TensorVar(t1Val, :t1)
map1 = x->x.^2
a3 = CellParam(map1, t1, :T1)
a3Val = obtain(a3)
@test a3Val == map1(t1Val)
@test a3.offset == (0.0, 0.0, 0.0)
@test Quiqbox.directObtain(a3.memory) == map1(t1Val)
setScreenLevel!(a3, 2)
@test a3.offset == a3Val
@test a3.offset == a3Val == obtain(a3)
setScreenLevel!(a3, 1)
@test a3.offset == a3Val
@test a3.offset == a3Val == obtain(a3)
setScreenLevel!(a3, 0)
@test a3.offset == (0.0, 0.0, 0.0)

struct myS{T}
    a::T
end
ms_t1Val = myS.(t1Val)
ms1 = TensorVar(ms_t1Val, :ms)
map1_2 = x->myS.(getproperty.(x, :a) .^ 2)
a3_2 = CellParam(map1_2, ms1, :MS)
@test getproperty.(obtain(a3_2), :a) == a3Val
@test Quiqbox.checkTypedOpMethods(typeof(ms_t1Val)) == false
@test try setScreenLevel!(a3_2, 2); false catch; true end

t2Val = -1 .* t1Val
t2 = TensorVar(t2Val, :t)
map2 = (x, y)-> x .+ y
a4 = CellParam(map2, (t1, t2), :T)
t12 = ParamGrid([t1, t2], :t12)
a4_2 = CellParam(x->x[1].+x[2], t12, :T)
@test all(obtain(a4) .== 0 .== obtain(a4_2))
n1 = GridParam(t12, :N)
n1_2 = GridParam(identity, t12, :N)
n1.lambda == n1_2.lambda
@test obtain(n1) == obtain(t12) == [t1Val, t2Val]
n2 = GridParam(x->[x[2], x[1]], t12, :N)
@test obtain(n2) == [t2Val, t1Val]
n1_3 = GridParam(x->map(y->y.^2, x), t12, :N)
@test n1_3() == map(y->y.^2, obtain.([t1, t2]))

v12 = ParamGrid([v1, v2], :v12)
b1 = CellParam(f1, v12, :b)
@test b1.offset == 0
b1Offset = 0.1
@atomic b1.offset = b1Offset
f1_bias = (x)->(f1(x) + b1Offset)
b1Val = f1_bias( obtain(v12) )
@test b1() == b1Val
@test obtain(b1) == b1Val
@test screenLevelOf(b1) == 0
@test all(inputOf(b1)[1] === v12)

b2 = setScreenLevel(b1, 1)
b3 = setScreenLevel(b1, 2)
@test b1() == b2() == b3()
@test b1Val == b2() == b3()
@test screenLevelOf(b2) == 1
@test screenLevelOf(b3) == 2
@test b2.offset == obtain(b2)
@test b3.offset == obtain(b3)

f2(x) = 2x[1] * exp(x[1] + x[5]) - x[2]/(1 + (x[3] - x[4])^2)
pars_f2 = ParamGrid([a1, v2, b1, b2, b3], :pars_f2)

c1 = CellParam(f2, pars_f2, :c)
inSet1, outSet1, isoSet1 = markParams!(c1)
inSet10D = inSet1[]
@test isempty(isoSet1)
@test outSet1[] == c1
inSet1_t = [v1, v2, b2]
@test all(inSet10D .=== inSet1_t)

b3Val1 = b3()
f2_t = function (x)
    2x[1] * exp(x[1] + b3Val1) - x[2] / (1 + (f1_bias([x[1], x[2]]) - x[3])^2)
end
c1Val = f2_t(obtain.(inSet10D)) #42.96591869214117
@test obtain(pars_f2) == obtain([a1, v2, b1, b2, b3])
@test f2(obtain(pars_f2)) == c1Val == c1() == 42.96591869214117

sortedNodes, marker1_c1, marker2_c1 = topoSort(c1)
sortedNodeSyms = symOf.(sortedNodes)
@test sortedNodeSyms == [:α, :a, :β, :v12, :b, :b, :pars_f2, :c]
pgIds1 = findall(x->(x in symOf.( (v12, pars_f2) )), sortedNodeSyms)
@test marker1_c1[begin:end .∉ Ref(pgIds1)] == [0, 1, 0, 1, 0, 1]
@test marker2_c1[begin:end .∉ Ref(pgIds1)] == [1, 1, 1, 1, 1, 0]
@test marker1_c1[pgIds1] == marker2_c1[pgIds1] == [1, 1]
leafIds = .!marker1_c1 .* marker2_c1
leaves = sortedNodes[leafIds]
@test symOf.(leaves) == [:α, :β, :b]
@test inSet10D == leaves == inSet1_t
gn_c1 = genGraphNode(c1)
@test evaluateNode(gn_c1) == c1() == c1Val

gnf_c1 = compressNode(gn_c1, inSet1)
pVals1 = obtain.(inSet1)
pVals1_2 = Tuple(pVals1)
@test gnf_c1() == c1Val == c1()
pVals1New = [[0.15534854432749634, 0.5391802798413137, 0.6960920197420257]]
setVal!.(inSet10D, pVals1New[])
@test gnf_c1(pVals1New) == c1() != evaluateNode(gn_c1)
setVal!.(inSet10D, pVals1[])
@test gnf_c1(pVals1) == c1Val

# Test infinite loop avoidance during recursive function calls
v1a1 = ParamGrid([v1, a1], :va)
b4 = CellParam(f1, v1a1, :b4)
b4Val0 = f1( obtain(v1a1) )
@test Quiqbox.directObtain(b4.memory) == b4Val0
@test inputOf(b4)[1] === v1a1
v1a1.input[end] = b4
b4Val1 = f1([v1ValNew, Quiqbox.directObtain(b4.memory)])
@test b4() == b4Val1
gn_b4 = genGraphNode(b4)
inSet_b4, _, _ = markParams!(b4)
inSetVal_b4 = obtain.(inSet_b4)
gnf_b4 = compressNode(gn_b4, inSet_b4)
@test b4() == b4Val1 == evaluateNode(gn_b4) == gnf_b4(inSetVal_b4)
@test Quiqbox.directObtain(b4.memory) == b4Val0
b4.memory.value[] = b4Val1
b4Val2 = f1([v1ValNew, Quiqbox.directObtain(b4.memory)])
@test b4Val2 == b4()
@test b4() == b4Val2 == obtain(b4)
@test evaluateNode(gn_b4) == b4Val1 == gnf_b4(inSetVal_b4) != b4Val2
@test Quiqbox.directObtain(b4.memory) == b4Val1
b4fallBack = 1.2345678
b4.memory.value[] = b4fallBack
@test obtain(b4) == f1([v1(), b4fallBack])

pars_f2_b4 = ParamGrid(vcat(inputOf(pars_f2), b4), :pars_f2_b4)
c2 = CellParam(x->norm(exp.(0.001 .* x)), pars_f2_b4, :c)
c2Val = c2() # 2.4542258914367587
@test c2() == c2Val
inSet_c2, outSet_c2, isoSet1_c2 = markParams!(c2)
@test all(inSet_c2[] .=== [v1, v2, b2])
@test outSet_c2[] === c2
@test isempty(isoSet1_c2)
gn_c2 = genGraphNode(c2)
gnf_c2 = compressNode(gn_c2, inSet_c2)
inSet_c2Val  = obtain.(inSet_c2)
gn_c2_Val1 = evaluateNode(gn_c2)
c2_c2_Val2 = gnf_c2(inSet_c2Val)
@test gn_c2_Val1 == c2_c2_Val2 == c2Val == c2()
b4.memory.value[] = Float64(π)
c2Val2 = c2() # 2.465225794459516
@test c2Val == evaluateNode(gn_c2) == gnf_c2(inSet_c2Val) != c2Val2
inSet2_c2, _, _ = markParams!(c2)
gn2_c2 = genGraphNode(c2)
gnf2_c2 = compressNode(gn2_c2, inSet2_c2)
@test evaluateNode(gn2_c2) == gnf2_c2(inSet_c2Val) == c2Val2 == c2()


m1Val = [0.8500493898201774 0.8371857655136232 0.8261332197426693; 
         0.8909368725700677 0.028692823163915082 0.8609921365095632; 
         0.15601079124513528 0.8390087083772936 0.539265546021646];
m1 = TensorVar(m1Val, :m)
@test obtain(m1) == m1Val
g1 = sum
e1 = CellParam(g1, m1, :e)
@test obtain(e1) == g1(m1Val)
m2Val = [0.5878290913996813 0.9722792211623517 0.9553534523311727 0.7753564819171315 0.47273331783546524; 
         0.004178648753904168 0.6065316448778457 0.35980863893594073 0.520655497909379 0.8223591668576677; 
         0.020435742071468366 0.3152758906745252 0.007076181436746043 0.15802943754751997 0.3922026235649726];
m2 = TensorVar(m2Val, :m)
g2 = ((x::Matrix{T}, y::Matrix{T}) where {T}) -> norm(x * y)::T
e2 = CellParam(g2, (m1, m2), :e)
@test obtain(e2) == g2(m1Val, m2Val)
inSet_e2, outSet_e2, isoSet_e2 = markParams!(e2)
@test all(inSet_e2 .=== [m1, m2])
@test outSet_e2[] == e2
@test isempty(isoSet_e2)
gn_e1 = genGraphNode(e1)
pVals_e1 = obtain.(inSet_e2)
gnf_e1 = compressNode(gn_e1, inSet_e2)
@test evaluateNode(gn_e1) == gnf_e1(pVals_e1) == 5.82827525296409

f3 = (x, y, z)->log(x^2 + y[1]) * (y[2] - sqrt(norm(exp.(z))))
f3_t = (xv, z)->log(xv[1]^2 + xv[2]) * (xv[3] - sqrt(norm(exp.(z))))
v2a2 = ParamGrid([v2, a2], :v2a2)
d1 = CellParam(f3, (a1, v2a2, m1), :d)
Quiqbox.ParamMarker(d1)
d2 = setScreenLevel(d1, 1)
d1_val = d1()
@test d1_val == d2() == f3(a1(), [v2(), a2()], m1())
inSet2, outSet2, isoSet2 = markParams!(d1)
@test all(vcat(inSet2[begin] .=== [v1, v2, a2], inSet2[end] === m1))
@test outSet2[] === d1
@test isempty(isoSet2)
gn_d1 = genGraphNode(d1)
pVals2 = obtain.(inSet2)
gnf_d1 = compressNode(gn_d1, inSet2)
@test evaluateNode(gn_d1) == d1_val == gnf_d1(pVals2) == d1()

pVec1 = ParamGrid([c1, d1, a2], :c1d1a2)
apRef1 = GridParam(pVec1, :ref)
pVec1Val = obtain(pVec1) # [42.96591869214117, -1.1243289206151053, 0.75]
@test obtain(apRef1) == pVec1Val

inSet5r, outSet5r, isoSet5r = markParams!(pVec1.input)
inSet5,  outSet5,  isoSet5  = markParams!(apRef1)

@test isempty(isoSet5)
@test all(outSet5r .=== [c1, d1])
@test isoSet5r[] === a2
@test length(inSet5r) == length(inSet5) == 2
@test all(vcat(inSet5r[1], a2) .=== inSet5[1])
@test inSet5r[2] === inSet5[2]

get_gnData = function (inSets, node)
    map(inSets) do inSet
        gn = genGraphNode(node)
        gnf = compressNode(gn, inSet)
        inSetVal = obtain.(inSet)
        @test gnf(inSetVal) == obtain(node) == evaluateNode(gn)
        (gn, gnf, inSetVal)
    end
end

gn_data_5r, gn_data_5 = get_gnData((inSet5r, inSet5), apRef1)

f3 = x -> x.^2 * ( exp.(x) )'
ap1 = GridParam(f3, pVec1, :sq)
ap1Val = obtain(ap1)
@test ap1Val == f3(pVec1Val)

pg1Input = [a1 a2; c1 d1]

pg1 = ParamGrid(pg1Input, :pl)

@test obtain(pg1) == obtain.(Quiqbox.directObtain(pg1.input)) == obtain.(pg1Input)
@test Quiqbox.compareParamBox(ParamGrid(pg1), pg1)
pg1_2 = ParamGrid(copy(pg1Input), pg1.symbol)
pg1_3 = ParamGrid(pg1Input, pg1.symbol)
pg1_4 = ParamGrid(pg1)
@test Quiqbox.compareParamBox(pg1_2, pg1)
@test Quiqbox.compareParamBox(pg1_3, pg1)
@test Quiqbox.compareParamBox(pg1_4, pg1)

makeGrid(l, cen, ns) = begin
    lens = ns .* l
    coordBegin = cen .- lens ./ 2
    iterRange = Iterators.product(range.(0, ns)...)
    map(iterRange) do i
        (coordBegin .+ i .* l)'
    end
end

f4 = (l, c) -> makeGrid(l, c, (1.,2.,3.))
k1 = TensorVar(1.0, :l)
k2 = TensorVar([0., 0., 0.], :c)
pm1 = ParamMesh(f4, (k1, k2), :pn)

pm1Val = obtain(pm1)

@test  pm1Val == obtain(pm1) == f4(k1(), k2())

inSet_pm1, outSet_pm1, isoSet_pm1 = markParams!(pm1)
inSetVal_pm1 = obtain.(inSet_pm1)
gn_pm1 = genGraphNode(pm1)
gnf_pm1 = compressNode(gn_pm1, inSet_pm1)
@test pm1Val == evaluateNode(gn_pm1) == gnf_pm1(inSetVal_pm1)

nps_pm1 = fragment(pm1);
val_pm1_1 = obtain(pm1)
val_pm1_2 = obtain.(nps_pm1)
val_pm1_3 = obtain(nps_pm1)

@test val_pm1_1 == val_pm1_2 == val_pm1_3
@test all(size.(val_pm1_1).== Ref( (1, 3) ))

f5 = (l, c) -> map(norm, f4(l, c))
pm2 = ParamMesh(f5, (k1, k2), :pn)
pm2Val = obtain(pm2)

nps_pm2 = fragment(pm2);

@test obtain.(nps_pm2) == pm2() == obtain(nps_pm2) == f5(k1(), k2()) == pm2Val
pg_pm2 = ParamGrid(nps_pm2, :pm2);
@test obtain(pg_pm2) == obtain(pm2) == obtain(nps_pm2) == obtain.(nps_pm2)

inSet_pm2, outSet_pm2, isoSet_pm2 = markParams!(pm2)
inSetVal_pm2 = obtain.(inSet_pm2)
gn_pm2 = genGraphNode(pm2)
gnf_pm2 = compressNode(gn_pm2, inSet_pm2)
@test evaluateNode(gn_pm2) == gnf_pm2(inSetVal_pm2) == pm2()

inSet_pg_pm2, outSet_pg_pm2, isoSet_pg_pm2 = markParams!(pg_pm2)
inSetVal_pg_pm2 = obtain.(inSet_pg_pm2)
gn_pg_pm2 = genGraphNode(pg_pm2)
gnf_pg_pm2 = compressNode(gn_pg_pm2, inSet_pg_pm2)
@test evaluateNode(gn_pg_pm2) == gnf_pg_pm2(inSetVal_pg_pm2) == pg_pm2()
@test evaluateNode(gn_pg_pm2) == evaluateNode(gn_pm2) == pm2Val
@test gnf_pg_pm2(inSetVal_pg_pm2) == gnf_pm2(inSetVal_pm2) == pm2Val

inSet_nps_pm2, outSet_nps_pm2, isoSet_nps_pm2 = markParams!(nps_pm2)
@test inSet_pm2 == inSet_nps_pm2 == inSet_pg_pm2 == [[k1], k2]
@test isempty(isoSet_pm2) && isempty(isoSet_nps_pm2) && isempty(isoSet_pg_pm2)
@test outSet_pm2[] == pm2
@test outSet_nps_pm2 == vec(nps_pm2)
@test outSet_pg_pm2[] == pg_pm2

end