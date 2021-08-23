using Test
using Quiqbox
using Suppressor: @capture_out

@testset "Overload.jl" begin

    # function show
    pb1 = ParamBox(-1, canDiff=false)
    @test (@capture_out show(pb1)) == "ParamBox{:undef, Float64}(-1.0)[undef][∂]"
    pb2 = ParamBox(-1, :a, index=1)
    @test (@capture_out show(pb2)) == "ParamBox{:a, Float64}(-1.0)[a₁][∂]"
    pb3 = ParamBox(-1, :x, mapFunction=abs)
    @test (@capture_out show(pb3)) == "ParamBox{:x, Float64}(-1.0)[x -> abs(x)][∂]"

    bf1 = BasisFunc([1,2,1], (2,1))
    @test (@capture_out show(bf1.gauss[1])) == "GaussFunc(xpn=ParamBox{:α, Float64}(2.0)"*
                                                 "[α][∂], con=ParamBox{:d, Float64}(1.0)"*
                                                 "[d][∂])"
    @test (@capture_out show(bf1)) == "BasisFunc{:S, 1}(gauss, subshell, center)[X⁰Y⁰Z⁰]"*
                                      "[1.0, 2.0, 1.0]"
    bf2 = BasisFunc("STO-3G")[]
    @test (@capture_out show(bf2)) == "BasisFunc{:S, 3}(gauss, subshell, center)[X⁰Y⁰Z⁰]"*
                                      "[NaN, NaN, NaN]"
    
    bfs1 = BasisFunc([0,0,0], (2,1), "P")
    @test (@capture_out show(bfs1)) == "BasisFuncs{:P, 1, 3}(gauss, subshell, center)[3/3]"*
                                       "[0.0, 0.0, 0.0]"
    bfs2 = BasisFunc([0,0,0], (2,1), [[2,0,0]])
    @test (@capture_out show(bfs2)) == "BasisFuncs{:D, 1, 1}(gauss, subshell, center)"*
                                       "[X²Y⁰Z⁰][0.0, 0.0, 0.0]"

    bfm1 = BasisFuncMix([bf1, bf2])
    @test (@capture_out show(bfm1)) == "BasisFuncMix{2}(BasisFunc, param)"

    GTb1 = GTBasis([bf1, bfs2])
    @test (@capture_out show(GTb1)) == "GTBasis{2, "*
                                       "Vector{Quiqbox.FloatingGTBasisFunc{Subshell, 1, 1}"*
                                       " where Subshell}}(basis, S, Te, eeI, getVne, getHc"*
                                       "ore)"
    
    box1 = GridBox(2, 3.5)
    @test (@capture_out show(box1)) == "GridBox{2, 2, 2}(num, len, coord)"

    fVar1 = runHF(GTb1, ["H", "H"], [[0, 0, 0], [1,2,1]], printInfo=false)

    @test (@capture_out show(fVar1.temp)) == "Quiqbox.HFtempVars{:RHF, 1}(shared.Etot="*
                                             "[2.263712269, … , 2.262890817], "*
                                             "shared.Dtots, Cs, Es, Ds, Fs)"

    @test (@capture_out show(fVar1)) == "Quiqbox.HFfinalVars{:RHF, 2, 2}(E0HF=2.262890817,"*
                                        " C, F, D, Emo, occu, temp, isConverged)"

    
    # function hasBoolRelation
    pb4 = deepcopy(pb1)
    @test true  == hasBoolRelation(==, pb1, pb4)
    @test false == hasBoolRelation(===, pb1, pb4)
    @test false == hasBoolRelation(==, pb2, pb3)
    @test true  == hasBoolRelation(==, pb2, pb3, ignoreFunction = true)
    @test false == hasBoolRelation(==, pb1, pb2)
    @test true  == hasBoolRelation(==, pb1, pb2, ignoreContainerType=true)
    
end