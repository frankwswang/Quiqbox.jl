using Test
using Quiqbox
using Quiqbox: FLevel, getFLevel, Absolute, DI, PF, combinePF, RC

@testset "Mapping.jl" begin

# struct FLevel & function getFLevel & struct DI
@test FLevel{getFLevel(itself)} == FLevel(identity) == FLevel{0}
@test FLevel(abs) == FLevel{1}
tf1 = PF(abs, +, 2.2)
@test FLevel{getFLevel(tf1)} == FLevel(tf1)
pf1 = PF(tf1, *, 1.5)
@test FLevel(pf1) == FLevel{2}
@test FLevel(tf1) == FLevel(typeof(tf1)) == FLevel(getFLevel(tf1)) == FLevel{2}
@test getFLevel(Absolute(itself)) == 1
@test getFLevel(Absolute(abs)) == 2
DressedPf1 = DI(pf1)
@test getFLevel(DressedPf1) == 2
@test DressedPf1 == DI(DressedPf1)
a0 = fill(2.2)
@test DressedPf1(a0) === a0


# struct PF and ChainedPF
pf1 = PF(abs, *, -1.5)
@test pf1(-2.0) == -3.0
pf2 = PF(abs, *, -1.5)
@test pf2(-2) == -3.0
pf3 = PF(pf2, *, -1.0)
@test pf3 isa PF{typeof(abs), typeof(*)}
@test pf3(-2.0) == 3.0
@test PF(PF(itself, *, -1.5), *, -1.0)(-2) == -3.0

sf1 = PF(abs, +, 2)
@test sf1(-1) == 3
sf2 = PF(sf1, +, -4)
@test sf2(-1) == -1
@test sf2 isa PF{typeof(abs), typeof(+)}

xf1 = PF(abs, ^, 2)
xf2 = PF(abs, ^, 0.5)
@test xf1(-1) == 1
@test xf2(-4) == 2
xf3 = PF(xf2, ^, -1)
@test xf3 isa PF{typeof(abs), typeof(^)}

cf1 = PF(pf1, +, 0.3)
@test cf1 isa Quiqbox.ChainedPF{typeof(pf1.f), 2, Tuple{typeof(*), typeof(+)}}
@test cf1(-0.33) == 0.33*(-1.5) + 0.3
cf2 = PF(cf1, +, 0.2)
@test cf2 isa Quiqbox.ChainedPF{typeof(pf1.f), 3}
@test cf2(-0.33) == cf1(-0.33) + 0.2

# function combinePF
@test combinePF(+, pf1, pf2) == PF(abs, *, (pf1.c + pf2.c))
pf4 = PF(abs, *, 1.5)
@test combinePF(+, pf1, pf4)(rand()) == 0
@test combinePF(+, sf1, sf2) == PF(abs, *, 2)
sf3 = PF(sf2, +, 5)
@test combinePF(+, sf1, sf3) == PF(PF(abs, *, 2), +, (sf1.c + sf3.c))
@test combinePF(+, pf1, sf1) == combinePF(+, sf1, pf1) == 
      PF(PF(abs, *, pf1.c+1), +, sf1.c)
pf4 = PF(abs, *, -1)
@test combinePF(+, pf4, sf1)(rand()) == combinePF(+, sf1, pf4)(rand()) == sf1.c
pf5 = PF(abs, *, 0)
@test combinePF(+, pf5, sf1) == PF(abs, +, sf1.c)
cf3 = combinePF(+, pf1, sf1)
@test cf3 == PF(PF(abs, *, (1+pf1.c)), +, sf1.c)
cf3_t = combinePF(+, x->abs(x)*pf1.c, x->abs(x)+sf1.c)
rn1 = rand()
@test cf3(rn1) ≈ cf3_t(rn1)

@test combinePF(*, pf1, pf2) == PF(PF(abs, ^, 2), *, pf1.c*pf2.c)
r0 = combinePF(*, pf1, pf5)
@test r0 isa RC{Float64}
@test r0(rand()) == 0
pf6 = PF(abs, *, 1/pf1.c)
@test combinePF(*, pf1, pf6) == PF(abs, ^, 2)
@test combinePF(*, xf1, xf2) == PF(abs, ^, xf1.c+xf2.c)
xf3 = PF(abs, ^, -xf1.c)
r1 = combinePF(*, xf1, xf3)
@test r1 isa RC{Int}
@test r1(rand()) == 1
xf4 = PF(abs, ^, 1-xf1.c)
@test combinePF(*, xf1, xf4) == xf1.f
@test combinePF(*, xf1, pf1) == combinePF(*, pf1, xf1) == PF(PF(abs, ^, xf1.c+1), *, pf1.c)
xf5 = PF(abs, ^, -1)
@test combinePF(*, xf5, pf1)(rand()) == pf1.c
xf6 = PF(abs, ^, 0)
@test combinePF(*, xf6, pf1) == PF(abs, *, pf1.c)


end