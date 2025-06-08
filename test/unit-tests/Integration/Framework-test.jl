using Test
using Quiqbox: BasisIndexList, OneToIndex

@testset "Types.jl" begin

bList1 = BasisIndexList(3)
bList2 = BasisIndexList((3,))
@test length(bList1.index) == length(bList2.index) == 3
@test bList1.endpoint == bList2.endpoint == [OneToIndex(), OneToIndex(4)]
bList3 = BasisIndexList([1, 2])
@test length(bList3.index) == 3
@test bList3.endpoint == [OneToIndex(), OneToIndex(2), OneToIndex(4)]

end