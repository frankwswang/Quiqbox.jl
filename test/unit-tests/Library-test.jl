using Test
using Quiqbox: SubshellXs, SubshellXYs, SubshellXYZs, checkBSList

@testset "Library.jl tests" begin

@test sort(SubshellXs) == SubshellXs
@test sort(SubshellXYs) == SubshellXYs
@test sort.(SubshellXYs.|>collect, rev=true) .|> Tuple == SubshellXYs
@test sort(SubshellXYZs) == SubshellXYZs
@test sort.(SubshellXYZs.|>collect, rev=true) .|> Tuple == SubshellXYZs

@test orbitalLin("P") == ((1,0,0), (0,1,0), (0,0,1))

@test try checkBSList(); true catch; false end

end