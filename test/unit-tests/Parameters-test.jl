using Test
using Quiqbox
using Quiqbox: getTypeParams, compareParamBox, getFLevel, FLevel, PBFL, addParamBox, 
               mulParamBox, reduceParamBoxes

@testset "Parameters.jl" begin


pb1 = ParamBox(1, :a)
@test getTypeParams(pb1) == (Int, :a, Quiqbox.iT)
@test (FLevel∘getFLevel)(pb1) == FLevel(pb1.map) == FLevel(Quiqbox.iT)
@test inSymOf(pb1) == :x_a && isInSymEqual(pb1, :x_a)
@test outSymOf(pb1) == :a && isOutSymEqual(pb1, :a)
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
@test PBFL((pb1, pb2, pb3)) == PBFL{(0, 0, 1)}

pb4 = ParamBox(1.2, :c, x->x^2, :x)
@test getTypeParams(pb4) == (Float64, :c, typeof(pb4.map))
toggleDiff!(pb4)
@test toggleDiff!(pb4)
@test inSymOf(pb4) == :x
@test outSymOf(pb4) == :c

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
@test d1 != Dict(indVarOf.(getUnique!(bfm_gv.param|>collect)))
d2 = Dict(indVarOf.(getUnique!(bfm_gv.param|>collect, compareFunction=hasIdentical)))
d3 = Dict(indVarOf.(getUnique!(bfm_gv.param|>collect, compareFunction=compareParamBox)))
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
      (getTypeParams(pb4_3)!=:C) && !isDiffParam(pb4_3)
toggleDiff!(pb4_3)


# function addParamBox
pb10 = ParamBox(4, :a, x->x^2)
pb11 = changeMapping(pb10, x->x^3)
pb12 = addParamBox(pb10, pb11)
@test pb12() == (pb10() + pb11())
@test (pb12.data .=== pb10.data) && (pb10.data .=== pb11.data)
@test isDiffParam(pb12) && (pb12.index[] == nothing)
pb13 = ParamBox(3, :a)
pb14 = ParamBox(3, :a)
pb15 = addParamBox(pb13, pb14)
@test getTypeParams(pb15) == (Int, :a, Quiqbox.iT)
@test pb15() == 2pb13[] == pb15[]
@test !isDiffParam(pb15) && (pb15.index[] == nothing)
pb16 = ParamBox(1.5, :a)
pb17 = ParamBox(1.5, :a)
pb18 = addParamBox(pb16, pb17)
@test getTypeParams(pb18) == (Float64, :a, Quiqbox.iT)
@test pb18() == 2pb16[] == pb18[]
@test !isDiffParam(pb18) && (pb18.index[] == nothing)


# function mulParamBox
pb12_2 = mulParamBox(pb10, pb11)
@test pb12_2() == (pb10() * pb11())
@test pb12_2.data .=== pb10.data
@test isDiffParam(pb12_2) && (pb12_2.index[] == nothing)
pb15_2 = mulParamBox(pb13, pb14)
@test getTypeParams(pb15_2) == (Int, :a, Quiqbox.iT)
@test pb15_2() == pb13[]*pb14[] == pb15_2[]
@test !isDiffParam(pb15_2) && (pb15_2.index[] == nothing)
pb18_2 = mulParamBox(pb16, pb17)
@test getTypeParams(pb18_2) == (Float64, :a, Quiqbox.iT)
@test pb18_2() == pb16[]*pb17[] == pb18_2[]
@test !isDiffParam(pb18_2) && (pb18_2.index[] == nothing)

c1 = 2
pb19 = mulParamBox(c1, pb10)
@test pb19() == c1*pb10()
@test pb19.data .=== pb10.data
@test isDiffParam(pb19) && (pb19.index[] == nothing)

c2 = 3
pb20 = mulParamBox(c2, pb13)
@test getTypeParams(pb20) == (Int, :a, Quiqbox.iT)
@test pb20() == c2*pb13() == pb20[]
@test !isDiffParam(pb20) && (pb19.index[] == nothing)


# function reduceParamBoxes
pb10_2 = changeMapping(pb10, pb10.map)
pb10_3 = reduceParamBoxes(pb10, pb10_2)[]
@test pb10_3 === pb10
@test pb10_3 !== pb10_2 && hasIdentical(pb10_3, pb10_2)

pb13_14 = reduceParamBoxes(pb13, pb14)[]
@test !hasIdentical(pb13_14, pb13) && hasEqual(pb13_14, pb13)
@test !hasIdentical(pb13_14, pb14) && hasEqual(pb13_14, pb14)

pb16_a1 = ParamBox(1.5 + 2e-16, :a)
pb16_a2 = ParamBox(1.5 + 1e-10, :a)
pb16_r1 = reduceParamBoxes(pb16_a1, pb16)[]
pb16_r2 = reduceParamBoxes(pb16_a2, pb16)
@test length(pb16_r2) == 2
@test pb16_r2[begin] === pb16_a2 && pb16_r2[end] === pb16

end