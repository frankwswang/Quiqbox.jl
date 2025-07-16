using BenchmarkTools #> `using Pkg; Pkg.add("BenchmarkTools")` to install BenchmarkTools.jl
using Quiqbox

#> Benchmark Structure
const SUITE = BenchmarkGroup(["Basis", "Integration", "CompositeFunction"])
const OrbInteBSuite = BenchmarkGroup()

const OrbEvalBSuite = BenchmarkGroup()
const OvlpInteSuite = BenchmarkGroup()
const ClmbInteSuite = BenchmarkGroup()
const DiffFuncBSuite = BenchmarkGroup(["Symbolic", "Automatic", "Numerical"])

OrbInteBSuite["Overlap"] = OvlpInteSuite
OrbInteBSuite["Coulomb"] = ClmbInteSuite
SUITE["Basis"]["Orbital"] = OrbEvalBSuite
SUITE["Integration"]["Orbital"] = OrbInteBSuite
SUITE["CompositeFunction"]["Differentiation"] = DiffFuncBSuite


#> Benchmark Objects
  cen1 = (1.1,  0.5,  1.1)
  cen2 = (1.0,  1.5,  1.1)
coord1 = (2.1,  3.1,  3.4)
coord2 = (1.1,  2.1, -3.2)
coord3 = (1.1, -2.1)
coord4 = (0.1,  0.2,  0.3)
coord5 = (1.5,  0.2, -0.5)

cons1 = [1.5, -0.3]
xpns1 = [1.2,  0.6]

xpns2 = [1.5,  0.6]
cons2 = [1.0,  0.8]

ang1 = (1, 0, 0)

nucs1 = [:H, :Li]


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
pgto1c = first(cgto1c.basis)


func1Core = x->x[1]^2 + x[2]*exp(x[1]) + x[3]^3/log(1.1 + x[1]^2)

func1 = Quiqbox.TypedCarteFunc(func1Core, Float64, Count(3))
df1_fd1 = Quiqbox.AxialFiniteDiff(func1, Count(1), 1)
df1_fd2 = Quiqbox.AxialFiniteDiff(func1, Count(2), 1)
df1_fd3 = Quiqbox.AxialFiniteDiff(func1, Count(3), 1)


gf1D1 = Quiqbox.GaussFunc(xpns1[begin])
gfo1D1 = Quiqbox.PrimitiveOrb((first(coord3),), gf1D1, renormalize=false)
gf1D2 = Quiqbox.GaussFunc(xpns1[end])
gfo1D2 = Quiqbox.PrimitiveOrb(( last(coord3),), gf1D2, renormalize=false)

ap2D = Quiqbox.AxialProduct((stf1D, gf1D1))
apOrb = Quiqbox.PrimitiveOrb((1.0, 2.0), ap2D, renormalize=false)


#> Benchmark Groups
#>> Orbital-Evaluation Benchmark Group
OrbEvalBSuite["PGTO"]["Direct"] = @benchmarkable ($pgto1 )($coord1) evals=1
OrbEvalBSuite["CGTO"]["Direct"] = @benchmarkable ($cgto1 )($coord1) evals=1
OrbEvalBSuite["PSTO"]["Direct"] = @benchmarkable ($psto1 )($coord1) evals=1

OrbEvalBSuite["PGTO"]["Renorm"] = @benchmarkable ($pgto1n)($coord1) evals=1
OrbEvalBSuite["CGTO"]["Renorm"] = @benchmarkable ($cgto1n)($coord1) evals=1
OrbEvalBSuite["PSTO"]["Renorm"] = @benchmarkable ($psto1n)($coord1) evals=1

OrbEvalBSuite["PGTO"]["Custom"] = @benchmarkable ($pgto1c)($coord1) evals=1
OrbEvalBSuite["CGTO"]["Custom"] = @benchmarkable ($cgto1c)($coord1) evals=1
OrbEvalBSuite["PSTO"]["Custom"] = @benchmarkable ($psto1c)($coord1) evals=1

OrbEvalBSuite["APOrb"]["Direct"] = @benchmarkable ($apOrb )($coord3) evals=1


#>> Orbital-Overlap Benchmark Group
OvlpInteSuite["PGTO_self_DD"]["Cached"] = @benchmarkable overlap($pgto1,  $pgto1 ) evals=1
OvlpInteSuite["PGTO_self_DN"]["Cached"] = @benchmarkable overlap($pgto1,  $pgto1n) evals=1
OvlpInteSuite["PGTO_PGTO_DD"]["Cached"] = @benchmarkable overlap($pgto1,  $pgto2 ) evals=1
OvlpInteSuite["CGTO_PGTO_DD"]["Cached"] = @benchmarkable overlap($cgto1,  $pgto1 ) evals=1
OvlpInteSuite["CGTO_self_DD"]["Cached"] = @benchmarkable overlap($cgto1,  $cgto1 ) evals=1
OvlpInteSuite["CGTO_self_DN"]["Cached"] = @benchmarkable overlap($cgto1,  $cgto1n) evals=1
OvlpInteSuite["CGTO_CGTO_DD"]["Cached"] = @benchmarkable overlap($cgto1,  $cgto2 ) evals=1
OvlpInteSuite["PSTO_PGTO_DD"]["Cached"] = @benchmarkable overlap($psto1,  $pgto1 ) evals=1
OvlpInteSuite["PSTO_Self_DD"]["Cached"] = @benchmarkable overlap($psto1,  $psto1 ) evals=1
OvlpInteSuite["PSTO_Self_DN"]["Cached"] = @benchmarkable overlap($psto1,  $psto1n) evals=1
OvlpInteSuite["PSTO_Self_NN"]["Cached"] = @benchmarkable overlap($psto1n, $psto1n) evals=1

OvlpInteSuite["PGTO_self_DD"]["Direct"] = 
@benchmarkable overlap($pgto1, $pgto1,  lazyCompute=false) evals=1
OvlpInteSuite["PGTO_self_DN"]["Direct"] = 
@benchmarkable overlap($pgto1, $pgto1n, lazyCompute=false) evals=1
OvlpInteSuite["CGTO_self_DD"]["Direct"] = 
@benchmarkable overlap($cgto1, $cgto1,  lazyCompute=false) evals=1
OvlpInteSuite["CGTO_self_DN"]["Direct"] = 
@benchmarkable overlap($cgto1, $cgto1n, lazyCompute=false) evals=1
OvlpInteSuite["CGTO_CGTO_DD"]["Direct"] = 
@benchmarkable overlap($cgto1, $cgto2,  lazyCompute=false) evals=1
OvlpInteSuite["PSTO_Self_DD"]["Direct"] = 
@benchmarkable overlap($psto1, $psto1,  lazyCompute=false) evals=1
OvlpInteSuite["PSTO_Self_DN"]["Direct"] = 
@benchmarkable overlap($psto1, $psto1n, lazyCompute=false) evals=1

OvlpInteSuite["PSTOc_self_DD"]["Cached"] = @benchmarkable overlap($psto1c, $psto1c) evals=1
OvlpInteSuite["PSTOc_PSTO_DD"]["Cached"] = @benchmarkable overlap($psto1c, $psto1)  evals=1
OvlpInteSuite["CGTOc_self_DD"]["Cached"] = @benchmarkable overlap($cgto1c, $cgto1c) evals=1
OvlpInteSuite["CGTOc_CGTO_DD"]["Cached"] = @benchmarkable overlap($cgto1c, $cgto1)  evals=1
OvlpInteSuite["GFO1D_self_DD"]["Cached"] = @benchmarkable overlap($gfo1D1, $gfo1D1) evals=1

OvlpInteSuite["APOrb_self_DD"]["Cached"] = @benchmarkable overlap($apOrb, $apOrb) evals=1
OvlpInteSuite["APOrb_CGTO_DD"]["Cached"] = @benchmarkable overlap($apOrb, $cgto3) evals=1


#>> Orbital-Coulomb Benchmark Group
ClmbInteSuite["NucAttr"]["CGTO"]["Cached"] = 
@benchmarkable neAttraction(nucs1, [coord4, coord5],  cgto1, cgto2) evals=1
ClmbInteSuite["NucAttr"]["CGTOc"]["Cached"] = 
@benchmarkable neAttraction(nucs1, [coord4, coord5], cgto1c, cgto2) evals=1

ClmbInteSuite["ElectRI"]["CGTO"]["Cached"] = 
@benchmarkable eeInteraction(cgto1,   cgto1,  cgto2,  cgto2) evals=1
ClmbInteSuite["ElectRI"]["GFO1D"]["Cached"] = 
@benchmarkable eeInteraction(gfo1D1, gfo1D1, gfo1D2, gfo1D2) evals=1

#>> Differentiation-Function Benchmark Group
DiffFuncBSuite["df1"]["Numerical"] = @benchmarkable ($df1_fd1)($coord2)
DiffFuncBSuite["df2"]["Numerical"] = @benchmarkable ($df1_fd2)($coord2)
DiffFuncBSuite["df3"]["Numerical"] = @benchmarkable ($df1_fd3)($coord2)


#> Finalized Benchmarkable Suite
SUITE #> `BenchmarkTools.run(SUITE)` to manually invoke benchmarking