using Test
using Quiqbox: SubshellXs, SubshellXYs, SubshellXYZs, LTuple, orbitalLin, checkBSList

@testset "Library.jl tests" begin

@test sort(SubshellXs) == SubshellXs
@test sort(SubshellXYs) == SubshellXYs
@test sort.(SubshellXYs.|>collect, rev=true) .|> Tuple == SubshellXYs
@test sort(SubshellXYZs) == SubshellXYZs
@test sort.(SubshellXYZs.|>collect, rev=true) .|> Tuple == SubshellXYZs


# type LTuple
t0 = (0,0,0)
t1 = (1,2,1)
t2 = (2,0,0)
lt0 = LTuple(t0)
lt1 = LTuple(t1)
lt2 = LTuple(t2)
lt3 = lt1 + lt2
@test lt3 == LTuple(3,2,1) == LTuple(lt1, lt2) == LTuple(lt3)
@test lt2+lt0 == lt2 == LTuple(lt2, lt0) == LTuple(lt0, lt2)
@test lt0 == lt0 + lt0 == LTuple(lt0, lt0)
@test length(lt0) == 3 == size(lt0, 1) == typeof(lt0).parameters[1]
@test [i for i in lt3] == [i for i in t1.+t2]
@test lt3 == lt1 + t2 == t1 + lt2
@test Tuple(lt3) === (t1 .+ t2)
@test sum(lt1) == sum(t1) == typeof(lt1).parameters[2]
@test map(x->x^3, lt1) == (1,8,1)
@test map((x,y)->x-y, lt1, lt2) == t1 .- t2
@test map((x,y,z)->x-y+z, lt0, lt1,lt2) == t0 .- t1 .+ t2
@test lt0 < lt1 < lt2
@test lt2 .+ 1 == ( lt2 + LTuple(1,1,1) ).tuple == t2 .+ 1
@test lt2 .- 1 == t2 .- 1

@test orbitalLin("P") == ((1,0,0), (0,1,0), (0,0,1))


@test try checkBSList(); true catch; false end

end