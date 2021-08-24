using Test
using Quiqbox
using Suppressor: @capture_out

@testset "Overload.jl" begin

    # function show
    pb1 = ParamBox(-1, canDiff=false)
    @test (@capture_out show(pb1)) == string(typeof(pb1))*"(-1.0)[undef][∂]"

    pb2 = ParamBox(-1, :a, index=1)
    @test (@capture_out show(pb2)) == string(typeof(pb2))*"(-1.0)[a₁][∂]"
    
    pb3 = ParamBox(-1, :x, mapFunction=abs)
    @test (@capture_out show(pb3)) == string(typeof(pb3))*"(-1.0)[x -> abs(x)][∂]"

    bf1 = BasisFunc([1,2,1], (2,1))
    gf1 = bf1.gauss[1]
    @test (@capture_out show(gf1)) == string(typeof(gf1))*"(xpn=ParamBox{:α, Float64}(2.0)"*
                                      "[α][∂], con=ParamBox{:d, Float64}(1.0)[d][∂])"
    @test (@capture_out show(bf1)) == string(typeof(bf1))*"(gauss, subshell, center)"*
                                      "[X⁰Y⁰Z⁰][1.0, 2.0, 1.0]"
    
    bf2 = BasisFunc("STO-3G")[]
    @test (@capture_out show(bf2)) == string(typeof(bf2))*"(gauss, subshell, center)"*
                                      "[X⁰Y⁰Z⁰]"*"[NaN, NaN, NaN]"
    
    bfs1 = BasisFunc([0,0,0], (2,1), "P")
    @test (@capture_out show(bfs1)) == string(typeof(bfs1))*"(gauss, subshell, center)"*
                                       "[3/3]"*"[0.0, 0.0, 0.0]"

    bfs2 = BasisFunc([0,0,0], (2,1), [[2,0,0]])
    @test (@capture_out show(bfs2)) == string(typeof(bfs2))*"(gauss, subshell, center)"*
                                       "[X²Y⁰Z⁰]"*"[0.0, 0.0, 0.0]"

    bfm1 = BasisFuncMix([bf1, bf2])
    @test (@capture_out show(bfm1)) == string(typeof(bfm1))*"(BasisFunc, param)"

    GTb1 = GTBasis([bf1, bfs2])
    @test (@capture_out show(GTb1)) == string(typeof(GTb1))*"(basis, S, Te, eeI, getVne, "*
                                       "getHcore)"
    
    box1 = GridBox(2, 3.5)
    @test (@capture_out show(box1)) == string(typeof(box1))*"(num, len, coord)"

    fVar1 = runHF(GTb1, ["H", "H"], [[0, 0, 0], [1,2,1]], printInfo=false)

    @test (@capture_out show(fVar1.temp)) == string(typeof(fVar1.temp))*"(shared.Etot="*
                                             "[2.263712269, … , 2.262890817], shared.Dtots"*
                                             ", Cs, Es, Ds, Fs)"

    @test (@capture_out show(fVar1)) == string(typeof(fVar1))*"(E0HF=2.262890817, C, F, D,"*
                                        " Emo, occu, temp, isConverged)"

    
    # function hasBoolRelation
    pb4 = deepcopy(pb1)
    @test true  == hasBoolRelation(==, pb1, pb4)
    @test false == hasBoolRelation(===, pb1, pb4)
    @test false == hasBoolRelation(==, pb2, pb3)
    @test true  == hasBoolRelation(==, pb2, pb3, ignoreFunction = true)
    @test false == hasBoolRelation(==, pb1, pb2)
    @test true  == hasBoolRelation(==, pb1, pb2, ignoreContainerType=true)
    
end