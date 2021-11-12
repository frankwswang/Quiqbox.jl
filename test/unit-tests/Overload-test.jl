using Test
using Quiqbox
using Quiqbox: BasisFuncMix, hasBoolRelation
using Suppressor: @capture_out

@testset "Overload.jl" begin

    # function show
    pb1 = ParamBox(-1, canDiff=false)
    @test (@capture_out show(pb1)) == string(typeof(pb1))*"(-1.0)[∂][undef]"

    pb2 = ParamBox(-1, :a, index=1)
    @test (@capture_out show(pb2)) == string(typeof(pb2))*"(-1.0)[∂][a₁]"

    pb3 = ParamBox(-1, :x, abs)
    @test (@capture_out show(pb3)) == string(typeof(pb3))*"(-1.0)[∂][x_x]"

    bf1 = genBasisFunc([1,2,1], (2,1))
    gf1 = bf1.gauss[1]
    @test (@capture_out show(gf1)) == string(typeof(gf1))*"(xpn="*
                                      string(typeof(gf1.param[1]))*"(2.0)[∂][α], con="*
                                      string(typeof(gf1.param[2]))*"(1.0)[∂][d])"
    @test (@capture_out show(bf1)) == string(typeof(bf1))*"(gauss, subshell, center)"*
                                      "[X⁰Y⁰Z⁰][1.0, 2.0, 1.0]"

    bf2 = genBasisFunc(missing, "STO-3G")[]
    @test (@capture_out show(bf2)) == string(typeof(bf2))*"(gauss, subshell, center)"*
                                      "[X⁰Y⁰Z⁰]"*"[NaN, NaN, NaN]"

    bfs1 = genBasisFunc([0,0,0], (2,1), "P")
    @test (@capture_out show(bfs1)) == string(typeof(bfs1))*"(gauss, subshell, center)"*
                                       "[3/3]"*"[0.0, 0.0, 0.0]"

    bfs2 = genBasisFunc([0,0,0], (2,1), [[2,0,0]])
    @test (@capture_out show(bfs2)) == string(typeof(bfs2))*"(gauss, subshell, center)"*
                                       "[X²Y⁰Z⁰]"*"[0.0, 0.0, 0.0]"

    bfm1 = BasisFuncMix([bf1, bf2])
    @test (@capture_out show(bfm1)) == string(typeof(bfm1))*"(BasisFunc, param)"

    GTb1 = GTBasis([bf1, bfs2])
    @test (@capture_out show(GTb1)) == string(typeof(GTb1))*"(basis, S, Te, eeI, getVne, "*
                                       "getHcore)"

    box1 = GridBox(2, 1.5)
    @test (@capture_out show(box1)) == string(typeof(box1))*"(num, len, coord)"

    fVar1 = runHF(GTb1, ["H", "H"], [[0, 0, 0], [1,2,1]], printInfo=false, initialC=:Hcore)

    @test (@capture_out show(fVar1.temp)) == string(typeof(fVar1.temp))*"(shared.Etots="*
                                             "[2.263712269, … , 2.262890932], shared.Dtots"*
                                             ", Cs, Es, Ds, Fs)"

    @test (@capture_out show(fVar1)) == string(typeof(fVar1))*"(E0HF=2.262890932, C, F, D,"*
                                        " Emo, occu, temp, isConverged)"


    # function hasBoolRelation
    pb4 = deepcopy(pb1)
    @test true  == hasBoolRelation(==, pb1, pb4)
    @test false == hasBoolRelation(===, pb1, pb4)
    @test false == hasBoolRelation(==, pb2, pb3)
    @test true  == hasBoolRelation(==, pb2, pb3, ignoreFunction = true)
    @test false == hasBoolRelation(==, pb1, pb2)
    @test true  == hasBoolRelation(==, pb1, pb2, ignoreContainer=true)


    # function +
    testAdd = function (a1, a2)
        hasEqual(a2 +  a1 + a1, 
                 a1 +  a2 + a1, 
                 a1 +  a1 + a2, 
                 a1 + (a1 + a2), 
                 add(add(a1, a2), a1), 
                 add(a1, add(a2, a1)))
    end

    bf3 = genBasisFunc([1,2,1], ([2,1], [0.2, 0.4]))
    bf4 = genBasisFunc([1,1,1], (3,0.5))
    bfm2 = BasisFuncMix([bf1, bf4, bf1])
    bfm3 = BasisFuncMix([bf4, bf4, bf3])

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
    boolFunc(x, y) = (==)(x,y)
    boolFunc(x::Array{<:Real, 0}, y::Array{<:Real, 0}) = isapprox(x, y, atol=1e-12)
    boolFunc(x::Real, y::Real) = isapprox(x, y, atol=1e-12)
    testMul = function (a1, a2, ignoreContainer=false)
        r1 = hasBoolRelation(boolFunc, 
                             a1*(a1 + a2), 
                             (a1 + a2)*a1, 
                             a1*a1 + a1*a2, 
                             a1*a1 + a2*a1, 
                             mul(a1, add(a1, a2)), 
                             mul(add(a1, a2), a1); ignoreContainer)

        r2 = hasBoolRelation(boolFunc, a1 *  a2 * a1, 
                             a1 * (a2 * a1), 
                             mul(mul(a1, a2), a1),
                             mul(a1, mul(a2, a1)); ignoreContainer)
        r1 * r2
    end

    bf5 = genBasisFunc([1,1,1], GaussFunc(genExponent(3), genContraction(0.2, x->5x)))
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
        r1 = hasBoolRelation(boolFunc, 
                             a*(a1 + a2), 
                             (a1 + a2)*a, 
                             a*a1 + a*a2, 
                             a1*a + a2*a, ignoreContainer=true)

        r2 = hasBoolRelation(boolFunc, 
                             a  * a1 * a2, 
                             a  * a2 * a1,
                             a1 * a  * a2,
                             a1 * a2 * a ,
                             a2 * a1 * a ,
                             a2 * a  * a1,
                             a  * (a1* a2), 
                             a1 * (a2* a ),
                             a2 * (a * a1), ignoreContainer=true)
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
                     c1, c2) |> prod

    @test .!testMul2.([bfm2, bfm3, bfm4], c1, c2) |> prod


    # function iterate, size, length, ndims
    @test iterate(pb1) == (-1, nothing)
    @test iterate(gf1) == (gf1, nothing)
    @test iterate(bf1) == (bf1, nothing)

    cs = [pb1, gf1, bf1]
    @test (iterate.(cs, 1) .=== nothing) |> prod
    @test (size.(cs) .== Ref(())) |> prod
    @test (size.(cs, 1) .== 1) |> prod
    @test (length.(cs) .== 1) |> prod
    @test ndims(pb1) == 0

    @test iterate(bfm1) == (bfm1, nothing)
    @test hasEqual(iterate(bfs1), (bfs1[1], 2))
    @test hasEqual(iterate(bfs1, 1), (bfs1[1], 2))
    @test hasEqual(iterate(bfs1, length(bfs1.ijk)+1), nothing)

    @test size(bfm1) == ()
    @test size(bfm1, 1) == 1
    @test size(bfs1) == (length(bfs1.ijk),)
    @test size(bfs1, 1) == length(bfs1.ijk)

    @test length(bfm1) == 1
    @test length(bfs1) == length(bfs1.ijk)


    # function getindex, setindex!, firstindex, lastindex, axes
    @test getindex(pb1) == pb1[] == pb1[begin] == pb1[end] == -1
    @test (pb1[] = -2; res = (pb1[] == -2); pb1[] = -1; res)
    @test axes(pb1) == ()

    @test getindex(gf1) == gf1[] == gf1[begin] == gf1[end] == (gf1.param |> collect)
    @test getindex(bf1) == bf1[] == bf1[begin] == bf1[end] == (bf1.gauss |> collect)
    bfm11 = Quiqbox.BasisFuncMix([bf1, bf2, bf1])
    @test getindex(bfm11) == bfm11[] == bfm11[begin] == bfm11[end] == 
          [bf1.gauss[1], bf2.gauss...]
    bfs1_alter = genBasisFunc.(Ref([0,0,0]), Ref((2,1)), [[1,0,0], [0,1,0], [0,0,1]])
    for i in eachindex(bfs1)
        @test hasEqual(getindex(bfs1, i), bfs1[i], bfs1_alter[i])
    end
    @test hasEqual(bfs1[begin], bfs1[1])
    @test hasEqual(bfs1[end], bfs1[3])


    # function broadcastable
    @test getfield.(pb1, [:data, :map]) == [pb1.data, pb1.map]
    @test getfield.(gf1, [:xpn, :con]) == [gf1.xpn, gf1.con]
    @test hasEqual(bf1 .* [1.0, 3.0], [bf1*1.0, bf1*3.0])
    bfm12 = bf1 + genBasisFunc([1,2,2], (2,3))
    @test hasEqual(bfm12 .* [1.0, 3.0], [bfm12*1.0, bfm12*3.0])
    bfs11 = genBasisFunc([1,1,1], (2,3), "P")
    @test hasEqual(bfs11 .* [1.0, 2.0, 3.0], [i*j for (i,j) in zip(bfs11, 1:3)])
end