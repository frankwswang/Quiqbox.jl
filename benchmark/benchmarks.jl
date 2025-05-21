using BenchmarkTools # `using Pkg; Pkg.add("BenchmarkTools")` to install BenchmarkTools.jl.
using Quiqbox

# Benchmark Structure
const SUITE = BenchmarkGroup()
const CachedCompBSuite = BenchmarkGroup(["Direct", "Lazy"])
const BasisEvalBSuite = BenchmarkGroup(["Direct", "Renormalized"])
const IntegrationBSuite = BenchmarkGroup(["Analytical", "Numerical"])
const DifferentiationBSuite = BenchmarkGroup(["Analytical", "Automatic", "Numerical"])

SUITE["Basis"]["Orbital"] = BasisEvalBSuite
const OrbEvalBSuite = SUITE["Basis"]["Orbital"]
SUITE["Integration"]["Orbital"] = IntegrationBSuite
const OrbInteBSuite = SUITE["Integration"]["Orbital"]
SUITE["CompositeFunction"]["Differentiation"] = DifferentiationBSuite
const DiffFuncBSuite = SUITE["CompositeFunction"]["Differentiation"]

OrbInteBSuite["Overlap"] = CachedCompBSuite
const OrbOvlpBSuite = OrbInteBSuite["Overlap"]


# Benchmark Objects
cen1 = (1.1, 0.5, 1.1)
cen2 = (1.0, 1.5, 1.1)
coord1 = (2.1, 3.1, 3.4)
coord2 = (1.1, 2.1, -3.2)
coord2_2 = [1.1, 2.1, -3.2]

cons1 = [1.5, -0.3]
xpns1 = [1.2,  0.6]

xpns2 = [1.5,  0.6]
cons2 = [1.0,  0.8]

ang1 = (1, 0, 0)

cgto1  = genGaussTypeOrb(cen1, xpns1, cons1, ang1)
cgto1n = genGaussTypeOrb(cen1, xpns1, cons1, ang1, innerRenormalize=true, 
                                                  outerRenormalize=true)
pgto1  = first(cgto1.basis)
pgto1n = PrimitiveOrb(pgto1, renormalize=true)

cgto2 = genGaussTypeOrb(cen2, xpns2, cons2, ang1)
pgto2 = first(cgto2.basis)

stf1DCore = (x::Tuple{Real})->exp(-(x|>first))
stf1D = Quiqbox.EncodedField(TypedReturn(stf1DCore, Float64), Val(1))
pstoCore = Quiqbox.PolyRadialFunc(stf1D, (1, 1, 0))
psto1 = Quiqbox.PrimitiveOrb((1.0, 2.0, 3.0), pstoCore, renormalize=false)
psto1n = Quiqbox.PrimitiveOrb((1.0, 2.0, 3.0), pstoCore, renormalize=true)

func1 = Quiqbox.TypedTupleFunc(x->x[1]^2 + x[2]*x[1] + x[3]^3/x[1], Float64, Val(3))
df_fd1 = Quiqbox.AxialFiniteDiff(func1, Val(1), 1)

# Benchmark Groups
## Orbital-Evaluation Benchmark Group
OrbEvalBSuite["Direct"]["PGTO"] = @benchmarkable ($pgto1)($coord1) evals=1
OrbEvalBSuite["Direct"]["CGTO"] = @benchmarkable ($cgto1)($coord1) evals=1
OrbEvalBSuite["Direct"]["PSTO"] = @benchmarkable ($psto1)($coord1) evals=1

OrbEvalBSuite["Renormalized"]["PGTO"] = @benchmarkable ($pgto1n)($coord1) evals=1
OrbEvalBSuite["Renormalized"]["CGTO"] = @benchmarkable ($cgto1n)($coord1) evals=1
OrbEvalBSuite["Renormalized"]["PSTO"] = @benchmarkable ($psto1n)($coord1) evals=1

# Orbital-Overlap Benchmark Group
OrbOvlpBSuite["Direct"]["PGTO_self_DD"] = @benchmarkable overlap($pgto1,  $pgto1 ) evals=1
OrbOvlpBSuite["Direct"]["PGTO_self_DN"] = @benchmarkable overlap($pgto1,  $pgto1n) evals=1
OrbOvlpBSuite["Direct"]["PGTO_PGTO_DD"] = @benchmarkable overlap($pgto1,  $pgto2 ) evals=1
OrbOvlpBSuite["Direct"]["CGTO_PGTO_DD"] = @benchmarkable overlap($cgto1,  $pgto1 ) evals=1
OrbOvlpBSuite["Direct"]["CGTO_self_DD"] = @benchmarkable overlap($cgto1,  $cgto1 ) evals=1
OrbOvlpBSuite["Direct"]["CGTO_self_DN"] = @benchmarkable overlap($cgto1,  $cgto1n) evals=1
OrbOvlpBSuite["Direct"]["CGTO_CGTO_DD"] = @benchmarkable overlap($cgto1,  $cgto2 ) evals=1
OrbOvlpBSuite["Direct"]["PSTO_PGTO_DD"] = @benchmarkable overlap($psto1,  $pgto1 ) evals=1
OrbOvlpBSuite["Direct"]["PSTO_Self_DD"] = @benchmarkable overlap($psto1,  $psto1 ) evals=1
OrbOvlpBSuite["Direct"]["PSTO_Self_DN"] = @benchmarkable overlap($psto1,  $psto1n) evals=1
OrbOvlpBSuite["Direct"]["PSTO_Self_NN"] = @benchmarkable overlap($psto1n, $psto1n) evals=1

OrbOvlpBSuite["Lazy"]["PGTO_self_DD"] = 
@benchmarkable overlap($pgto1, $pgto1,  lazyCompute=true) evals=1
OrbOvlpBSuite["Lazy"]["PGTO_self_DN"] = 
@benchmarkable overlap($pgto1, $pgto1n, lazyCompute=true) evals=1
OrbOvlpBSuite["Lazy"]["CGTO_self_DD"] = 
@benchmarkable overlap($cgto1, $cgto1,  lazyCompute=true) evals=1
OrbOvlpBSuite["Lazy"]["CGTO_self_DN"] = 
@benchmarkable overlap($cgto1, $cgto1n, lazyCompute=true) evals=1
OrbOvlpBSuite["Lazy"]["CGTO_CGTO_DD"] = 
@benchmarkable overlap($cgto1, $cgto2,  lazyCompute=true) evals=1
OrbOvlpBSuite["Lazy"]["PSTO_Self_DD"] = 
@benchmarkable overlap($psto1, $psto1,  lazyCompute=true) evals=1
OrbOvlpBSuite["Lazy"]["PSTO_Self_DN"] = 
@benchmarkable overlap($psto1, $psto1n, lazyCompute=true) evals=1

DiffFuncBSuite["Numerical"]["df_fd1_tpl"] = @benchmarkable ($df_fd1)($coord2)   evals=1
DiffFuncBSuite["Numerical"]["df_fd1_arr"] = @benchmarkable ($df_fd1)($coord2_2) evals=1


# Finalized Benchmarkable Suite
SUITE # `BenchmarkTools.run(SUITE)` to manually invoke benchmarking.