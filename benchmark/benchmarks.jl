using BenchmarkTools # `using Pkg; Pkg.add("BenchmarkTools")` to install BenchmarkTools.jl.
using Quiqbox

# Benchmark Structure
const SUITE = BenchmarkGroup()
const OrbOvlpBSuite = BenchmarkGroup(["Direct", "Lazy"])
const OrbEvalBSuite = BenchmarkGroup(["Direct", "Renormalized"])
const OrbInteBSuite = BenchmarkGroup(["Analytic", "Numerical"])
const DiffFuncBSuite = BenchmarkGroup(["Symbolic", "Automatic", "Numerical"])

OrbInteBSuite["Overlap"] = OrbOvlpBSuite
SUITE["Basis"]["Orbital"] = OrbEvalBSuite
SUITE["Integration"]["Orbital"] = OrbInteBSuite
SUITE["CompositeFunction"]["Differentiation"] = DiffFuncBSuite


# Benchmark Objects
cen1 = (1.1, 0.5, 1.1)
cen2 = (1.0, 1.5, 1.1)
coord1 = (2.1, 3.1, 3.4)
coord2 = (1.1, 2.1, -3.2)
coord2_2 = [1.1, 2.1, -3.2]
coord3 = (0.5,)
coord4 = (1.1, -2.1)

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

cgto3 = genGaussTypeOrb(cen1[begin:begin+1], xpns2, cons1, ang1[end-1:end])

stf1DCore = (x::Tuple{Real})->exp(-(x|>first))

stf1D = Quiqbox.EncodedField(stf1DCore, Float64, Count(1))
pstf1 = Quiqbox.PolyRadialFunc(stf1D, (1, 1, 0))
psto1 = Quiqbox.PrimitiveOrb((1.0, 2.0, 3.0), pstf1, renormalize=false)
psto1n = Quiqbox.PrimitiveOrb((1.0, 2.0, 3.0), pstf1, renormalize=true)
stf1Dc = Quiqbox.ModularField(stf1DCore, Float64, Count(1))
pstf1c = Quiqbox.PolyRadialFunc(stf1Dc, (1, 1, 0))
psto1c = Quiqbox.PrimitiveOrb((1.0, 2.0, 3.0), pstf1c, renormalize=false)


function gaussianFunc((r,)::Tuple{Real}, params::@NamedTuple{xpn::T}) where {T<:Real}
    exp(-params.xpn * T(r)^2)
end

function genGaussian(cen::NTuple{D, T}, xpns::AbstractVector{T}, 
                     ang::NTuple{D, Int}) where {D, T<:Real}
    center = genCellParam.(cen, (:x, :y, :z)[begin:begin+D-1])
    map(xpns) do xpn
        xpnPar = genCellParam(xpn, :xpn)
        radioFunc = Quiqbox.ModularField(gaussianFunc, Float64, Count(1), (xpn=xpnPar,))
        fieldFunc = PolyRadialFunc(radioFunc, ang)
        PrimitiveOrb(center, fieldFunc)
    end
end

pgtos_gto1c = genGaussian(cen1, xpns1, ang1)
cgto1c = CompositeOrb(pgtos_gto1c, cons1)


func1Core = x->x[1]^2 + x[2]*exp(x[1]) + x[3]^3/log(1.1 + x[1]^2)

func1 = Quiqbox.TypedCarteFunc(func1Core, Float64, Count(3))
df1_fd1 = Quiqbox.AxialFiniteDiff(func1, Count(1), 1)
df1_fd2 = Quiqbox.AxialFiniteDiff(func1, Count(2), 1)
df1_fd3 = Quiqbox.AxialFiniteDiff(func1, Count(3), 1)


gf1D = Quiqbox.GaussFunc(xpns1[begin])
gfOrb = Quiqbox.PrimitiveOrb((1.0,), gf1D, renormalize=false)

ap2D = Quiqbox.AxialProduct((stf1D, gf1D))
apOrb = Quiqbox.PrimitiveOrb((1.0, 2.0), ap2D, renormalize=false)


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

# `ModularField`-Based Orbital Benchmark Group
OrbEvalBSuite["Direct"]["PSTOc"] = @benchmarkable ($psto1c)($coord1) evals=1
OrbEvalBSuite["Direct"]["CGTOc"] = @benchmarkable ($cgto1c)($coord1) evals=1
OrbEvalBSuite["Direct"]["GFOrb"] = @benchmarkable ($gfOrb )($coord3) evals=1
OrbOvlpBSuite["Direct"]["PSTOc_self_DD"] = @benchmarkable overlap($psto1c, $psto1c) evals=1
OrbOvlpBSuite["Direct"]["PSTOc_PSTO_DD"] = @benchmarkable overlap($psto1c, $psto1)  evals=1
OrbOvlpBSuite["Direct"]["CGTOc_self_DD"] = @benchmarkable overlap($cgto1c, $cgto1c) evals=1
OrbOvlpBSuite["Direct"]["CGTOc_CGTO_DD"] = @benchmarkable overlap($cgto1c, $cgto1)  evals=1
OrbOvlpBSuite["Direct"]["GFOrb_self_DD"] = @benchmarkable overlap($gfOrb,  $gfOrb)  evals=1

# Differentiation-Function Benchmark Group
DiffFuncBSuite["Numerical"]["df_fd1_tpl"] = @benchmarkable ($df1_fd1)($coord2)   evals=1
DiffFuncBSuite["Numerical"]["df_fd1_arr"] = @benchmarkable ($df1_fd1)($coord2_2) evals=1
DiffFuncBSuite["Numerical"]["df_fd2_tpl"] = @benchmarkable ($df1_fd2)($coord2)   evals=1
DiffFuncBSuite["Numerical"]["df_fd3_tpl"] = @benchmarkable ($df1_fd3)($coord2)   evals=1

# `ProductField`-Based Orbital Benchmark Group
OrbEvalBSuite["Numerical"]["APOrb"] = @benchmarkable ($apOrb )($coord4) evals=1
OrbOvlpBSuite["Numerical"]["APOrb_self_DD"] = @benchmarkable overlap($apOrb, $apOrb) evals=1
OrbOvlpBSuite["Numerical"]["APOrb_CGTO_DD"] = @benchmarkable overlap($apOrb, $cgto3) evals=1

# Finalized Benchmarkable Suite
SUITE # `BenchmarkTools.run(SUITE)` to manually invoke benchmarking.