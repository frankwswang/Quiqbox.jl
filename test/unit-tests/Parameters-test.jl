using Test
using Quiqbox
using Quiqbox: getVarCore

@testset "Parameters.jl" begin

pb1 = ParamBox(1, :a)
@test inSymOfCore(pb1) == :a == outSymOfCore(pb1)
@test dataOf(pb1)[] == pb1[] == inValOf(pb1)
@test pb1[] == pb1() == 1 == outValOf(pb1)
@test pb1.map == Quiqbox.itself == mapOf(pb1)
@test pb1.canDiff[] == false == isDiffParam(pb1)
toggleDiff!(pb1)
@test pb1.canDiff[] == true
disableDiff!(pb1)
@test pb1.canDiff[] == false
enableDiff!(pb1)
@test pb1.canDiff[] == true
pb2 = inVarCopy(pb1)
@test dataOf(pb1) === dataOf(pb2) === pb1.data
pb2_2 = outValCopy(pb1)
@test dataOf(pb1) == dataOf(pb2_2)
@test dataOf(pb1) !== dataOf(pb2_2)

pb3 = ParamBox(-1, :b, abs)
@test inSymOfCore(pb3) == :x_b
@test outSymOfCore(pb3) == :b
@test mapOf(pb3) == abs

pb4 = ParamBox(1.2, :c, x->x^2, :x)
@test inSymOfCore(pb4) == :x
@test outSymOfCore(pb4) == :c
@test startswith(nameof(pb4.map) |> string, "f_c")
pair1 = inSymValOf(pb4)
@test Tuple(pair1) == (inSymOf(pb4), 1.2) == (:x, 1.2)
pair2 = outSymValOf(pb4)
@test Tuple(pair2) == (outSymOf(pb4), pb4()) == (:c, pb4.map(1.2))
pb4_2 = ParamBox(1.2, :c)
@test inSymOf(pb4_2) == outSymOf(pb4_2) == :c


pb5 = outValCopy(pb4)
@test pb5() == pb5[] == pb4()
@test pb5.map == Quiqbox.itself

pb6 = ParamBox(1.1, :p1, abs, :x)
pb7 = changeMapping(pb6, x->x^1.5)
pb8 = changeMapping(pb6, x->x^1.5, :p2)
@test pb8() == pb7() == 1.1^1.5
@test pb8.data === pb7.data === pb6.data
@test pb8.dataName == pb7.dataName == :x
@test typeof(pb7).parameters[2] == :p1
@test typeof(pb8).parameters[2] == :p2


# function getVarCore getVar getVarDict
@test getVarCore(pb1) == [:a=>1]
@test getVarCore(pb4) == [:c=>1.44, :x=>1.2]
pb9 = ParamBox(pb4[], getVar(pb4), pb4.map, pb4.dataName, canDiff=false)
@test getVarCore(pb9) == [:c=>1.44]

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

@test getVar(e_gv1) == :α
bfs_gv = sortBasisFuncs([bf_gv1, bf_gv2, bf_gv3, bf_gv4, bf_gv5, bf_gv6])
markParams!(bfs_gv, true)
markParams!(vcat(bfs_gv, bfm_gv), true)
@test getVar.(bf_gv6.param) == (:X₁, :Y₂, :Z₂, :α₁, :d₃, :α₃, :d₂, :α₂, :d₁)
@test getVar.(bf_gv6.param, true) == (:X₁, :Y₂, :Z₂, :α₁, :d₃, :α₃, :d₂, :x_α₂, :d₁)
@test getVar.(bfm_gv.param) == (:X₁, :Y₁, :Z₁, :α₂, :d₂, :X₁, :Y₂, :Z₂, :α₁, :d₃, :α₃, :d₂, 
                                :α₂, :d₁, :X₂, :Y₂, :Z₃, :α₃, :d₃)
@test getVar.(bfm_gv.param, true) == (:X₁, :Y₁, :Z₁, :x_α₂, :d₂, :X₁, :Y₂, :Z₂, :α₁, :d₃, 
                                      :α₃, :d₂, :x_α₂, :d₁, :X₂, :Y₂, :Z₃, :α₃, :d₃)


@test getVarDict(e_gv1) == Dict(:α₁=>2.0)
@test getVarDict(pb4) == Dict([:x=>1.2, :c=>1.44])
pb1_2 = changeMapping(pb1, x->x+2)
@test getVarDict(pb1_2) == Dict(:a_a=>1, :a=>3)
@test getVarDict(pb3) == Dict(:x_b=>-1, :b=>1)
pb4_3 = changeMapping(pb4, x->x+2)
@test getVarDict(pb4_3) == Dict(:c=>3.2, :x=>1.2)
@test getVarDict(bf_gv6.param) == Dict( ( (  getVar.(bf_gv6.param) .=> 
                                           outValOf.(bf_gv6.param))..., 
                                          ( inSymOf.(bf_gv6.param) .=> 
                                           getindex.(bf_gv6.param))... ) )
@test getVarDict(bfm_gv.param) == Dict( ( (  getVar.(bfm_gv.param) .=> 
                                           outValOf.(bfm_gv.param))..., 
                                         (  inSymOf.(bfm_gv.param) .=> 
                                           getindex.(bfm_gv.param))... ) )
@test getVarDict(bfm_gv.param) == getVarDict(unique(bfm_gv.param))
@test getVarDict(bfm_gv.param) != getVarDict(getUnique!(bfm_gv.param|>collect))
@test getVarDict(bfm_gv.param) == getVarDict(getUnique!(bfm_gv.param|>collect, 
                                                        compareFunction=hasIdentical))

end