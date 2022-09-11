using Test
using Quiqbox
using Quiqbox: itself, Pf, Sf, FLevel, getFLevel, Absolute, DressedItself

@testset "Mapping.jl" begin

# struct FLevel & function getFLevel
@test FLevel{getFLevel(:itself)} == FLevel(identity) == FLevel{0}
@test FLevel(abs) == FLevel{1}
tf1 = Sf(abs, 2.2)
@test FLevel{getFLevel(tf1)} == FLevel(tf1)
pf1 = Pf(tf1, 1.5)
@test FLevel(pf1) == FLevel{3}
@test getFLevel(FLevel(tf1)) == getFLevel(typeof(tf1)) == getFLevel(tf1) == 2
@test getFLevel(Absolute(itself)) == 1
@test getFLevel(Absolute(abs)) == 2
DressedPf1 = DressedItself(pf1)
@test getFLevel(DressedPf1) == 3
@test DressedPf1 == DressedItself(DressedPf1)


# struct Pf
pf1 = Pf(abs, -1.5)
@test pf1(-2.0) == -3.0
pf2 = Pf(abs, -1.5)
@test pf2(-2) == -3.0
@test Pf(pf2, -1.0)(-2.0) == 3.0
@test Pf(Pf(itself, -1.5), -1.0)(-2) == -3.0


# struct Sf
sf1 = Sf(abs, 2)
@test sf1(-1) == 3
sf2 = Sf(sf1, 3)
@test sf2(-1) == 6

end