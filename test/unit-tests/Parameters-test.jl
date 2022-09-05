using Test
using Quiqbox
using Quiqbox: FLevel, getFLevel, compareParamBox

@testset "Parameters.jl" begin

# struct FLevel & function getFLevel
@test FLevel{getFLevel(:itself)} == FLevel(identity) == FLevel{0}
@test FLevel(abs) == FLevel{1}
tf1 = Quiqbox.Sf(2.2, abs)
@test FLevel{getFLevel(tf1)} == FLevel(tf1)
pf1 = Quiqbox.Pf(1.5, tf1)
@test FLevel(pf1) == FLevel{3}
@test getFLevel(FLevel(tf1)) == getFLevel(typeof(tf1)) == getFLevel(tf1) == 2


pb1 = ParamBox(1, :a)
@test inSymOf(pb1) == :x_a
@test outSymOf(pb1) == :a
@test dataOf(pb1)[begin][] == pb1[] == inValOf(pb1)
@test dataOf(pb1)[end] == :x_a
@test pb1[] == pb1() == 1 == outValOf(pb1)
@test pb1.map == Quiqbox.itself == mapOf(pb1)
@test pb1.canDiff[] == false == isDiffParam(pb1)
toggleDiff!(pb1)
@test pb1.canDiff[] == true
@test !disableDiff!(pb1)
@test pb1.canDiff[] == false
@test enableDiff!(pb1)
@test pb1.canDiff[] == true
pb2 = changeMapping(pb1)
@test dataOf(pb1) === dataOf(pb2) === pb1.data[]
pb2_2 = outValCopy(pb1)
@test dataOf(pb1) == dataOf(pb2_2)
@test dataOf(pb1) !== dataOf(pb2_2)

pb3 = ParamBox(-1, :b, abs)
@test inSymOf(pb3) == :x_b
@test outSymOf(pb3) == :b
@test mapOf(pb3) == abs
toggleDiff!(pb3)

pb4 = ParamBox(1.2, :c, x->x^2, :x)
toggleDiff!(pb4)
@test toggleDiff!(pb4)
@test inSymOf(pb4) == :x
@test outSymOf(pb4) == :c
@test startswith(nameof(pb4.map) |> string, "f_c")

pb5 = outValCopy(pb4)
@test pb5() == pb5[] == pb4()
@test pb5.map == Quiqbox.itself

pb5_2 = fullVarCopy(pb4)
@test hasIdentical(pb5_2, pb4)

pb6 = ParamBox(1.1, :p1, abs, :x)
pb7 = changeMapping(pb6, x->x^1.5)
pb8 = changeMapping(pb6, x->x^1.5, :p2)
@test pb8() == pb7() == 1.1^1.5
@test pb8.data[] === pb7.data[] === pb6.data[]
@test pb8.data[][end] == :x
@test typeof(pb7).parameters[2] == :p1
@test typeof(pb8).parameters[2] == :p2


# function indVarOf
@test indVarOf(pb4) == (inSymOf(pb4)=>1.2)
disableDiff!(pb4)
@test (Tuple∘indVarOf)(pb4) == (outSymOf(pb4), pb4()) == (:c, pb4.map(1.2))
@test indVarOf(pb1)[begin] == :x_a
pb9 = ParamBox(pb4[], indVarOf(pb4)[begin], pb4.map, pb4.data[][end], canDiff=false)
@test indVarOf(pb9)[begin] == :c

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
bfm_gv = Quiqbox.BasisFuncMix([bf_gv2, bf_gv4, bf_gv6])

@test indVarOf(e_gv1)[begin] == :α
bfs_gv = sortBasisFuncs([bf_gv1, bf_gv2, bf_gv3, bf_gv4, bf_gv5, bf_gv6])
markParams!(bfs_gv, true)
markParams!(vcat(bfs_gv, bfm_gv), true)
sortTuple = t -> (collect(t) |> sort)
@test sortTuple((first∘indVarOf).(bf_gv6.param)) == [:X₁, :Y₂, :Z₂, 
                                                      :d₁, :d₂, :d₃, :x_α₁, :α₁, :α₂]
@test sortTuple((first∘indVarOf).(bfm_gv.param)) == [:X₁, :X₁, :X₂, :Y₁, :Y₂, :Y₂, 
                                                      :Z₁, :Z₂, :Z₃, 
                                                      :d₁, :d₂, :d₂, :d₃, :d₃, 
                                                      :x_α₁, :x_α₁, :α₁, :α₂, :α₂]


# function enableDiff! & disableDiff! & toggleDiff! with indices
@test Dict(indVarOf(e_gv1)) == Dict(:α₁=>2.0)
e_gv1_2 = changeMapping(e_gv1, x->x+3.2)
@test Dict(indVarOf(e_gv1_2)) == Dict(:α=>5.2)
enableDiff!(e_gv1)
@test e_gv1.index[] === nothing
e_gv1.index[] = 1
@test (enableDiff!(e_gv1) && e_gv1.index[] == 1)
e_gv1_3 = changeMapping(e_gv1, e_gv1_2.map)
@test Dict(indVarOf(e_gv1_3)) == Dict(indVarOf(e_gv1)) == Dict(:x_α₁=>2.0)
@test isDiffParam(e_gv1_3)
disableDiff!(e_gv1_3)
@test indVarOf(e_gv1_3) === indVarOf(e_gv1_2)
e_gv1_3.index[] = 1
@test (disableDiff!(e_gv1_3) || e_gv1_3.index[] == 1)
@test toggleDiff!(e_gv1_3)
e_gv1.index[] = e_gv1_3.index[]
@test Dict(indVarOf(e_gv1_3)) == Dict(indVarOf(e_gv1)) == Dict(:x_α=>2.0)


@test Dict(indVarOf(pb3)) == Dict(:b=>1)
pb4_3 = changeMapping(pb4, x->x+2)
d1 = Dict(indVarOf.(bfm_gv.param))
@test d1 == Dict( vcat( [Symbol(outSymOf(i), Quiqbox.numToSubs(i.index[]))=>i() 
                        for i in bfm_gv.param if !isDiffParam(i)], 
                        [Symbol(inSymOf(i), Quiqbox.numToSubs(i.index[]))=>i[] 
                        for i in bfm_gv.param if isDiffParam(i)] ) )
@test d1 == Dict(indVarOf.(unique(bfm_gv.param)))
@test d1 != Dict(indVarOf.(getUnique!(bfm_gv.param|>collect)))


d2 = Dict(indVarOf.(getUnique!(bfm_gv.param|>collect, compareFunction=hasIdentical)))
d3 = Dict(indVarOf.(getUnique!(bfm_gv.param|>collect, compareFunction=compareParamBox)))
@show d1
@show d2
@show d3
@test d1 == d2
@test d1 == d3


# function compareParamBox
pb1_2 = changeMapping(pb1, x->x+2)
disableDiff!(pb2)
@test !compareParamBox(pb1, pb2) && isDiffParam(pb1) && !isDiffParam(pb2)
@test toggleDiff!(pb2)
@test compareParamBox(pb1, pb2)
toggleDiff!(pb2)
@test compareParamBox(pb1, pb1_2) == compareParamBox(pb1_2, pb1) == true
toggleDiff!(pb1_2)
@test !compareParamBox(pb1, pb1_2)
toggleDiff!(pb1_2)
@test enableDiff!.([pb4_3, pb4]) |> all
@test compareParamBox(pb4, pb4_3)
toggleDiff!(pb4_3)
@test !compareParamBox(pb4, pb4_3)
@test !compareParamBox(pb4_3, ParamBox(Val(:C), pb4_3)) && 
      (Quiqbox.getTypeParams(pb4_3)!=:C) && !isDiffParam(pb4_3)
toggleDiff!(pb4_3)

end