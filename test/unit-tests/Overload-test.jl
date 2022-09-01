using Test
using Quiqbox
using Quiqbox: BasisFuncMix, hasBoolRelation, typeStrOf, getFieldNameStr
using Suppressor: @capture_out

@testset "Overload.jl" begin

# function show
pb1 = ParamBox(-1)
@test (@capture_out show(pb1)) == string(typeof(pb1))*"(-1)[∂][undef]"

pb2 = ParamBox(-1, :a, index=1)
@test (@capture_out show(pb2)) == string(typeof(pb2))*"(-1)[∂][a₁]"

pb3 = ParamBox(-1, :x, abs)
@test (@capture_out show(pb3)) == string(typeof(pb3))*"(-1)[∂][x_x]"

p1 = genSpatialPoint((1.,))
@test (@capture_out show(p1)) == "SpatialPoint{Float64, 1, P1D{Float64, 0}}(param)[1.0][∂]"

p2 = genSpatialPoint((1.,2.))
@test (@capture_out show(p2)) == "SpatialPoint{Float64, 2, P2D{Float64, 0, 0}}(param)"*
                                 "[1.0, 2.0][∂][∂]"

p3 = genSpatialPoint((1.,2.,3.))
@test (@capture_out show(p3)) == "SpatialPoint{Float64, 3, P3D{Float64, 0, 0, 0}}(param)"* 
                                 "[1.0, 2.0, 3.0][∂][∂][∂]"

bf1 = genBasisFunc([1.0, 2.0, 1.0], (2.0, 1.0))
gf1 = bf1.gauss[1]
@test (@capture_out show(gf1)) == string(typeof(gf1))*"(xpn()=$(gf1.xpn()), "*
                                                       "con()=$(gf1.con()), param)"

bFieldStr = getFieldNameStr(bf1)
@test (@capture_out show(bf1)) == string(typeStrOf(bf1))*bFieldStr*"[X⁰Y⁰Z⁰][1.0, 2.0, 1.0]"

bf2 = genBasisFunc(missing, "STO-3G")[]
@test (@capture_out show(bf2)) == string(typeStrOf(bf2))*bFieldStr*"[X⁰Y⁰Z⁰][NaN, NaN, NaN]"

bfs1 = genBasisFunc([0.0, 0.0, 0.0], (2.0, 1.0), "P")
@test (@capture_out show(bfs1)) == string(typeStrOf(bfs1))*bFieldStr*"[3/3][0.0, 0.0, 0.0]"

bfs2 = genBasisFunc([0.0 ,0.0 , 0.0], (2.0, 1.0), [(2,0,0)])
@test (@capture_out show(bfs2)) == string(typeStrOf(bfs2))*bFieldStr*
                                   "[X²Y⁰Z⁰]"*"[0.0, 0.0, 0.0]"
bfs3 = genBasisFunc([0.0, 0.0, 0.0], (2.0, 1.0), [(2,0,0), (1,1,0)])
@test (@capture_out show(bfs3)) == string(typeStrOf(bfs3))*bFieldStr*
                                   "[2/6]"*"[0.0, 0.0, 0.0]"

bfe = Quiqbox.EmptyBasisFunc{Float64, 3}()
@test (@capture_out show(bfe)) == string(typeof(bfe))

box1 = GridBox(2, 1.5)
@test (@capture_out show(box1)) == string(typeStrOf(box1))*getFieldNameStr(box1)

bf3 = genBasisFunc(box1.point[1], (2.0, 1.0))

bfm1 = BasisFuncMix([bf1, bf2, bf3])
@test (@capture_out show(bfm1)) == string(typeStrOf(bfm1))*getFieldNameStr(bfm1)

GTb1 = GTBasis([bf1, bfs2])
@test (@capture_out show(GTb1)) == string(typeof(GTb1))*getFieldNameStr(GTb1)

fVar1 = runHF(GTb1, ["H", "H"], [[0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], HFconfig((C0=:Hcore,)), 
              printInfo=false)

info1 = (@capture_out show(fVar1.temp[1]))
tVarStrP1 = string(typeof(fVar1.temp[1]))
l1 = length(tVarStrP1)
@test info1[1:l1] == tVarStrP1
tVarStrP2 = getFieldNameStr(fVar1.temp[1])
l2 = length(tVarStrP2)
@test info1[l1+1:l1+l2-1] == tVarStrP2[1:end-1]
@test info1[l1+l2:l1+l2+7] == ".Etots=["
@test info1[end-1:end] == "])"

info2 = (@capture_out show(fVar1))
fVarStrP1 = string(typeof(fVar1))
fVarStrP2 = getFieldNameStr(fVar1)
l3 = length(fVarStrP1)
@test info2[1:l3] == fVarStrP1
@test info2[l3+1:l3+4] * info2[l3+17:end] == fVarStrP2

info3 = (@capture_out show(SCFconfig((:DD, :ADIIS, :DIIS), 
                                     (1e-4, 1e-12, 1e-13), Dict(2=>[:solver=>:LCM]))))
@test info3 == "SCFconfig{Float64, 3}(method=(:DD, :ADIIS, :DIIS), "*
               "interval=(0.0001, 1.0e-12, 1.0e-13), methodConfig, oscillateThreshold)"

H2 = MatterByHF(fVar1)
info4 = (@capture_out show(H2))
@test info4 == string(H2|>typeof) * getFieldNameStr(H2)


# function ==, hasBoolRelation
pb4 = deepcopy(pb1)
pb5 = ParamBox(-1, :fa, abs)
pb6 = ParamBox(-1, :x, abs)
pb7 = ParamBox(-1.0, :undef, identity, index=1)
pb8 = ParamBox(-1, :a, abs, index=2)
@test false == (pb3 == pb5)
@test true  == (pb3 == pb6)
@test false  == hasBoolRelation(==, pb1, pb2)
@test true  == hasBoolRelation(==, pb1, pb7)
@test false  == hasBoolRelation(===, pb1, pb7)
toggleDiff!(pb1)
@test true  == (hasBoolRelation(==, pb1, pb7) && pb1.canDiff == pb7.canDiff)
@test true  == hasBoolRelation(==, pb1, pb2, ignoreContainer=true)
@test true  == hasBoolRelation(==, pb1, pb7, ignoreContainer=true)
@test true  == hasBoolRelation(==, pb1, pb4)
@test false == hasBoolRelation(===, pb1, pb4)
@test true  == hasBoolRelation(===, pb1, pb4, ignoreContainer=true)
@test false == hasBoolRelation(==, pb2, pb8)
@test true  == hasBoolRelation(==, pb2, pb8, ignoreFunction = true)


# function +
testAdd = function (a1, a2)
    hasApprox(a2 +  a1 + a1, 
              a1 +  a2 + a1, 
              a1 +  a1 + a2, 
              a1 + (a1 + a2), 
              add(add(a1, a2), a1), 
              add(a1, add(a2, a1)), atol=5e-16)
end

bf3 = genBasisFunc([1.0, 2.0, 1.0], ([2.0, 1.0], [0.2, 0.4]))
bf4 = genBasisFunc([1.0, 1.0, 1.0], (3.0, 0.5))
bfm2 = BasisFuncMix([bf1, bf4, bf1])
bfm3 = BasisFuncMix([bf4, bf4, bf3])

@test gaussCoeffOf.((bf1 + bf3 + bf4).BasisFunc) == ([3.0 0.5], [2.0 1.2; 1.0 0.4])
@test hasEqual(bf1 + bf3 + bf4, 
               bf1 + bf4 + bf3, 
               bf3 + bf1 + bf4, 
               bf3 + bf4 + bf1, 
               bf4 + bf3 + bf1)
@test testAdd(bf1, bf3)
@test testAdd(bf1, bf4)
@test testAdd(bf3, bf4)
@test testAdd(bf1, bfm2)
@test testAdd(bf3, bfm2)
@test testAdd(bf4, bfm2)
@test testAdd(bf1, bfm3)
@test testAdd(bf3, bfm3)
@test testAdd(bf4, bfm3)
@test testAdd(bfm2, bfm3)


# function *
@test hasEqual(gf1 * GaussFunc(0.2, 1.5), GaussFunc(2.2, 1.5))
@test hasEqual(gf1 * π, GaussFunc(2.0, 1π))
@test hasEqual(π * gf1, GaussFunc(2.0, 1π))

testMul = function (a1, a2, ignoreContainer=false)
    r1 = hasApprox( a1*(a1 + a2), 
                    (a1 + a2)*a1, 
                    a1*a1 + a1*a2, 
                    a1*a1 + a2*a1, 
                    mul(a1, add(a1, a2)), 
                    mul(add(a1, a2), a1); ignoreContainer, atol=1e-15)

    r2 = hasApprox( a1 *  a2 * a1, 
                    a1 * (a2 * a1), 
                    mul(mul(a1, a2), a1),
                    mul(a1, mul(a2, a1)); ignoreContainer, atol=1e-15)
    r1 * r2
end

bf5 = genBasisFunc([1.0, 1.0, 1.0], GaussFunc(genExponent(3.0), genContraction(0.2, x->5x)))
bfm4 = BasisFuncMix([bf4, bf5, bf4])

@test testMul(bf1,  bf3)
@test testMul(bf1,  bf4)
@test testMul(bf1,  bf5)
@test testMul(bf3,  bf4)
@test testMul(bf3,  bf5)
@test testMul(bf4,  bf5)
@test testMul(bf1,  bfm2)
@test testMul(bf3,  bfm2)
@test testMul(bf4,  bfm2)
@test testMul(bf1,  bfm3)
@test testMul(bf3,  bfm3)
@test testMul(bf4,  bfm3)
@test testMul(bf1,  bfm4)
@test testMul(bf3,  bfm4)
@test testMul(bf4,  bfm4)
@test testMul(bfm2, bfm3)
@test testMul(bfm2, bfm4)
@test testMul(bfm3, bfm4)

testMul2 = function (a, a1, a2)
    r1 = hasApprox( a*(a1 + a2), 
                    (a1 + a2)*a, 
                    a*a1 + a*a2, 
                    a1*a + a2*a, ignoreContainer=true, atol=1e-15)

    r2 = hasApprox( a  * a1 * a2, 
                    a  * a2 * a1,
                    a1 * a  * a2,
                    a1 * a2 * a ,
                    a2 * a1 * a ,
                    a2 * a  * a1,
                    a  * (a1* a2), 
                    a1 * (a2* a ),
                    a2 * (a * a1), ignoreContainer=true, atol=1e-15)
    r1 * r2
end

c = 2.0

@test testMul2(c, bf1 , bf3 )
@test testMul2(c, bf1 , bf4 )
@test testMul2(c, bf1 , bf5 )
@test testMul2(c, bf3 , bf4 )
@test testMul2(c, bf3 , bf5 )
@test testMul2(c, bf4 , bf5 )
@test testMul2(c, bf1 , bfm2)
@test testMul2(c, bf3 , bfm2)
@test testMul2(c, bf4 , bfm2)
@test testMul2(c, bf1 , bfm3)
@test testMul2(c, bf3 , bfm3)
@test testMul2(c, bf4 , bfm3)
@test testMul2(c, bf1 , bfm4)
@test testMul2(c, bf3 , bfm4)
@test testMul2(c, bf4 , bfm4)
@test testMul2(c, bfm2, bfm3)
@test testMul2(c, bfm2, bfm4)
@test testMul2(c, bfm3, bfm4)

c1 = 1.5
c2 = 2.5

@test testMul2.([bf1, bf3, bf4, bf5, 
                 bfm2.BasisFunc |> sum, bfm3.BasisFunc |> sum, bfm4.BasisFunc |> sum], 
                c1, c2) |> all

@test .!testMul2.([bfm2, bfm3, bfm4], c1, c2) |> all


# function eltype
eltype(pb1) == typeof(pb1[])
eltype(bfs1) == typeof(bfs1[begin])


# function iterate, size, length, ndims
@test iterate(pb1) == (-1, nothing)
@test iterate(gf1) == (gf1, nothing)
@test iterate(bf1) == (bf1, nothing)
bfe = Quiqbox.EmptyBasisFunc{Float64, 2}()
@test iterate(bfe) == (bfe, nothing)

cs = [pb1, gf1, bf1]
@test (iterate.(cs, 1) .=== nothing) |> all
@test (size.(cs) .== Ref(())) |> all
@test (size.(cs, 1) .== 1) |> all
@test (length.(cs) .== 1) |> all

@test iterate(bfm1) == (bfm1, nothing)
@test iterate(bfm1, rand()) === nothing
@test hasEqual(iterate(bfs1), (bfs1[1], 2))
@test hasEqual(iterate(bfs1, 1), (bfs1[1], 2))
@test hasEqual(iterate(bfs1, length(bfs1.l)+1), nothing)

@test size(bfm1) == ()
@test size(bfm1, 1) == 1
@test try size(bfm1, -1) catch; true end
@test size(bfs1) == (length(bfs1.l),)
@test size(bfs1, 1) == length(bfs1.l)
D = dimOf(bf1)
sp1 = bf1.center
@test size(sp1) == (D,)
@test [size(sp1, i) for i in 1:D] == [D, 1, 1]
@test try size(sp1, -1) catch; true end

@test length(bfm1) == 1
@test length(bfs1) == length(bfs1.l)


# function getindex, setindex!, firstindex, lastindex, axes
@test getindex(pb1) == pb1[] == pb1[begin] == pb1[end] == -1
@test (pb1[] = -2; res = (pb1[] == -2); pb1[] = -1; res)
@test axes(pb1) == ()
lt1 = bf1.l[1]
bfm11 = Quiqbox.BasisFuncMix([bf1, bf2, bf1])
for i in bfm11.param[end-8:end-6]
    i[] = rand()
end
@test getindex(gf1) == gf1[] == gf1.param
@test getindex(sp1) == sp1[] == sp1.param
@test getindex(bf1) == bf1[] == bf1.param
@test getindex(lt1) == lt1[] == lt1.tuple
@test getindex(bfm11) == bfm11[] == bfm11.param
bfs1_alter = genBasisFunc.(Ref(fill(0.0, 3)), Ref((2.0,1.0)), [(1,0,0), (0,1,0), (0,0,1)])
for i in eachindex(bfs1)
    @test hasEqual(getindex(bfs1, i), bfs1[i], bfs1_alter[i])
end
@test sp1[begin] === sp1[1]
@test sp1[end] === sp1[dimOf(sp1)]
@test bf1[begin] === bf1 === bf1[end]
@test bfm11[begin] === bfm11 === bfm11[end]
@test hasEqual(bfs1[begin], bfs1[1])
@test hasEqual(bfs1[end], bfs1[orbitalNumOf("P")])

@test firstindex(sp1) == 1
@test lastindex(sp1) == D
@test axes(sp1) == (eachindex(sp1),) == (Base.OneTo(D),)
@test firstindex(lt1) == 1
@test lastindex(lt1) == D
@test axes(lt1) == (eachindex(lt1),) == (Base.OneTo(D),)

@test p3[:] == (p3[1], p3[2], p3[3])
@test bfs1[:] === bfs1[begin:end] === bfs1[[1,2,3]] === bfs1

collection1 = collect(pb1); collection1t = [pb1[]]
@test collection1 == collection1t
collection2 = collect(p3); collection2t = collect(p3[:])
@test collection2 == collection2t
collection3 = collect(bf1); collection3t = [bf1]
@test collection3 == collection3t == [i for i in bf1]
collection4 = collect(bfm1); collection4t = [bfm1]
@test collection4 == collection4t == [i for i in bfm1]
collection5 = collect(bfs1); collection5t = [bfs1[1], bfs1[2], bfs1[3]]
@test collection5 == collection5t == [i for i in bfs1]

@test typeof(collection1) == typeof(collection1t)
@test typeof(collection2) == typeof(collection2t)
@test typeof(collection3) == typeof(collection3t)
@test typeof(collection4) == typeof(collection4t)
@test typeof(collection5) == typeof(collection5t)


# function broadcastable
@test all(getproperty.(pb1, [:data, :map]) .=== [pb1.data, pb1.map])
@test getproperty.(gf1, [:xpn, :con]) == [gf1.xpn, gf1.con]
@test hasEqual(bf1 .* [1.0, 3.0], [bf1*1.0, bf1*3.0])
bfm12 = bf1 + genBasisFunc([1.0, 2.0, 2.0], (2.0, 3.0))
@test hasEqual(bfm12 .* [1.0, 3.0], [bfm12*1.0, bfm12*3.0])
bfs11 = genBasisFunc([1.0, 1.0, 1.0], (2.0, 3.0), "P")
@test hasEqual(bfs11 .* [1.0, 2.0, 3.0], [i*j for (i,j) in zip(bfs11, 1:3)])


# function flatten
bfTf1 = genBasisFunc([1.0, 2.0, 1.0], (2.0, 1.0))
bfTfs = genBasisFunc([1.0, 2.0, 1.0], (3.0, 2.0), "P")
bs1 = [bfTf1, bfTfs]
bs2 = (bfTf1, bfTfs)
bfs = decompose(bfTfs)
@test flatten(bs1) == [bfTf1, bfs...]
@test flatten(bs2) == (bfTf1, bfs...)
bs1 = [bfTf1, bf1]
bs2 = (bfTf1, bf1)
@test flatten(bs1) === bs1
@test flatten(bs2) === bs2

end