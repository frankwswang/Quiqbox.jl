using Test
using Quiqbox
using Quiqbox: ∂Basis, defaultHFforHFgrad as DHFO, 
               defaultHFthresholdForHFgrad as DHFOthreshold
using ForwardDiff: derivative as ForwardDerivative

include("../../test/test-functions/Shared.jl")

@testset "Differentiation.jl" begin

# ParamBox indices
bf0 = genBasisFunc(rand(3), (2.2, 1.1))
bf0Pars = bf0.param
outSymOf(bf0Pars[Quiqbox.cxIndex]) == :X
outSymOf(bf0Pars[Quiqbox.cyIndex]) == :Y
outSymOf(bf0Pars[Quiqbox.czIndex]) == :Z
outSymOf(bf0Pars[Quiqbox.xpnIndex]) == :α
outSymOf(bf0Pars[Quiqbox.conIndex]) == :d

# function ∂Basis 𝑑f
fα = x->sqrt(x); gα = x->1/2sqrt(x); vα = 1.2
xpn = genExponent(vα, fα)
fd = x->x^3; gd = x->3x^2; vd = 1.2
con = genContraction(vd, fd, canDiff=false)
gf = GaussFunc(xpn, con)
sgf1 = genBasisFunc([1.0, 0.0, 0.0], gf, (1,0,0))
X, Y, Z, xα, d = markParams!([sgf1])
J = changeMapping(d, fd, :J)
K = changeMapping(X, X.map, :K)
disableDiff!(K)

@test ∂Basis(J, sgf1) == Quiqbox.EmptyBasisFunc{Float64, 3}()
@test ∂Basis(K, sgf1) == Quiqbox.EmptyBasisFunc{Float64, 3}()
enableDiff!(K)
@test ∂Basis(K, sgf1) == Quiqbox.EmptyBasisFunc{Float64, 3}()
enableDiff!(X)
@test hasEqual(∂Basis(K, sgf1), ∂Basis(X, sgf1))
@test hasEqual(∂Basis(X, sgf1), -1*shift(sgf1, (1,0,0), -) + 2xpn()*shift(sgf1, (1,0,0)))
@test hasEqual(∂Basis(Y, sgf1), 2xpn()*shift(sgf1, (0,1,0)))
@test hasEqual(∂Basis(Z, sgf1), 2xpn()*shift(sgf1, (0,0,1)))
@test hasEqual(∂Basis(xα, sgf1), -1*(shift(sgf1, (2,0,0)) + 
                                     shift(sgf1, (0,2,0)) + 
                                     shift(sgf1, (0,0,2)))*gα(vα))
sgfN = genBasisFunc([1.0, 0.0, 0.0], GaussFunc(xpn, genContraction(1/con()*con())), (1,0,0))
@test hasEqual(∂Basis(d, sgf1), sgfN)

con2 = genContraction(con)
enableDiff!(con)
@test !isDiffParam(con2) == isDiffParam(con)
@test con2.data[] === con.data[]
@test ∂Basis(con2, sgf1) == Quiqbox.EmptyBasisFunc{Float64, 3}()
toggleDiff!.([con, con2])
@test ∂Basis(con2, sgf1) == Quiqbox.EmptyBasisFunc{Float64, 3}()
toggleDiff!(con2)
@test hasEqual(∂Basis(con2, sgf1), sgfN)
enableDiff!.([con, con2])
sgfN2 = genBasisFunc([1.0, 0.0, 0.0], GaussFunc(xpn, genContraction(gd(vd))), (1,0,0))
@test hasEqual(∂Basis(con2, sgf1), ∂Basis(con, sgf1), sgfN2)

con3 = genContraction(xpn.data[], fd)
sgf2 = genBasisFunc([1.0, 0.0, 0.0], GaussFunc(xpn, con3), (1,0,0))
@test hasEqual(∂Basis(con3, sgf1), ∂Basis(xpn, sgf1), ∂Basis(xα, sgf1))
@test hasEqual(∂Basis(con3, sgf2), ∂Basis(xpn, sgf2), ∂Basis(xα, sgf2))
@test hasEqual(∂Basis(con3, sgf2), ∂Basis(xα, sgf1)+sgfN2)
disableDiff!(con3)
@test hasEqual(∂Basis(con3, sgf2), sgfN)

f = x->x^3
@test Quiqbox.𝑑f(Quiqbox.DI(f), rand()) == 1
r1 = rand()
@test Quiqbox.𝑑f(f, r1) == ForwardDerivative(f, r1)

sgf3 = genBasisFunc(sgf1, true)
ijk1 = sgf1.l[1].tuple
grad2 = ∂Basis(xpn, sgf3)
grad1 = Quiqbox.getNijkα(ijk1, xpn())*∂Basis(xpn, sgf1) + 
        sgf1 * ForwardDerivative(x->Quiqbox.getNijkα(ijk1, x), xpn()) * gα(vα)
@test hasApprox(overlap(grad1, grad2), overlap(grad1, grad1), overlap(grad2, grad2), 
                atol=1e-15)

for P in (X,Y,Z)
    @test hasApprox(∂Basis(P, sgf3), ∂Basis(P, sgf1) * Quiqbox.getNijkα(ijk1, xpn()), 
                    atol=1e-15)
end

@test hasApprox(absorbNormFactor(∂Basis(con, sgf3)), 
                                 ∂Basis(con, sgf1) * Quiqbox.getNijkα(ijk1, xpn()), 
                                 atol=1e-15)


# function gradOfHFenergy
fDiffOfHFenergyCore! = function (pars, bs, nuc, nucCoords, Δx, config)
    map(pars) do par
        (isDiffParam(par) || Quiqbox.getFLevel(par.map) == 0) || 
        throw(DomainError(par, "This `par` in `pars` is not supported by the finite "*
              "difference method."))
        par[] += Δx
        E = runHF(bs, nuc, nucCoords, config, printInfo=false).Ehf
        par[] -= Δx
        E
    end
end

fDiffOfHFenergy = function (pars, bs, nuc, nucCoords, Δx=1e-6; config=DHFO)
    Eus = fDiffOfHFenergyCore!(pars, bs, nuc, nucCoords, Δx, config)
    Eds = fDiffOfHFenergyCore!(pars, bs, nuc, nucCoords, -Δx, config)
    (Eus - Eds) ./ (2Δx)
end

nuc = ["H", "H"]
nucCoords = [[-0.7,0.0,0.0], [0.7,0.0,0.0]]

grid = GridBox(1, 3.0)
gf1 = GaussFunc(0.7, 1.0)
bs1 = genBasisFunc.(grid.point, Ref([gf1]))
pars1 = markParams!(bs1)[[1, 2, 4, 5]]
S1 = overlaps(bs1)
HFres1 = runHF(bs1, nuc, nucCoords, DHFO, printInfo=false)
grad1 = gradOfHFenergy(pars1, bs1, S1, HFres1.C, nuc, nucCoords)
grad1_fd = fDiffOfHFenergy(pars1, bs1, nuc, nucCoords, 5e-8)
grad1_t = [1.2560795063145092, 1.2560795063145092, 4.050658426012205, 0]
t1 = 1e-14
t2 = 1e-10
t3 = 1e-8
@test isapprox(grad1[1], grad1[2], atol=t1)
compr2Arrays3((grad1=grad1, grad1_t=grad1_t), t2, true)
compr2Arrays3((grad1=grad1, grad1_fd=grad1_fd), 2t3, true)

config = HFconfig(HF=:UHF, SCF=SCFconfig(threshold=DHFOthreshold))
HFres1_2 = runHF(bs1, nuc, nucCoords, config, printInfo=false)
grad1_2 = gradOfHFenergy(pars1, bs1, overlaps(bs1), HFres1.C, nuc, nucCoords)
@test isapprox(grad1_2[1], grad1_2[2], atol=t1)
compr2Arrays3((grad1_2=grad1_2, grad1_t=grad1_t), t2, true)
@test abs(grad1_2[end]) < t1

bfSource = genBasisFunc(missing, "STO-2G", "H")[]
gfs = bfSource.gauss |> collect
cens = genSpatialPoint.(nucCoords)
bs2 = genBasisFunc.(cens, Ref(gfs), normalizeGTO=true)
pars2 = markParams!(bs2, true)
S2 = overlaps(bs2)
HFres2 = runHF(bs2, nuc, nucCoords, DHFO, printInfo=false)
grad2 = gradOfHFenergy(pars2, bs2, S2, HFres2.C, nuc, nucCoords)
grad2_fd = fDiffOfHFenergy(pars2, bs2, nuc, nucCoords, 5e-8)
compr2Arrays3((grad2=grad2, grad2_fd=grad2_fd), 5t3, true)
@test isapprox(grad2[1], -grad2[2], atol=t2)
@test isapprox(grad2[1], -0.0678638313089222, atol=t2)
@test all(grad2[3:6] .== 0)
grad2_tp = [0.006457377706861833, 0.1734869455759258, 
            0.09464147744656537, -0.059960502688765016]
compr2Arrays3((grad2_7toEnd=grad2[7:end], grad2_tp=grad2_tp), t2, true)

bs3 = bs1[[1,5]] .* bs2 # basis set of BasisFuncMix
pars3 = markParams!(bs3, true)
S3 = overlaps(bs3)
HFres3 = runHF(bs3, nuc, nucCoords, DHFO, printInfo=false)
grad3 = gradOfHFenergy(pars3, HFres3)
grad3_fd = fDiffOfHFenergy(pars3, bs3, nuc, nucCoords, 1e-7)
compr2Arrays3((grad3p=grad3[1:end-1], grad3_fdp=grad3_fd[1:end-1]), 10t3, true)
@test isapprox(grad3[end], grad3_fd[end], atol=100t3)
grad3_t = [-0.1606522922402765,    -0.24121983207381031, -0.1480105648704627, 
            0.0047746557202592195, -0.08411039021158562, -0.33217356846754603, 
           -0.4154684787642143,    -0.05739800488612984, -0.3058823967298582, 
           -0.41897421737311447,    0.6569617786306862,   0.10172360606515586, 
            1.2107782336923298,     0.1356557608658473,   1.6060314818099681, 
            0.058838380278514374,   0.7017475810800726,  -1.2886966024268467, 
            2.76294860945643,     -16.536548278392075]
compr2Arrays3((grad3=grad3, grad3_t=grad3_t), 8000t2, true)


# Edge case of accurate gradient requiring high-precision computation
gf2 = GaussFunc(0.7, 1.0)
gf3 = GaussFunc(0.5, 1.0)
grid2 = GridBox(1, 3.0)
bs4 = genBasisFunc.(grid2.point, Ref([gf2])) .+ genBasisFunc(fill(0.0, 3), gf3)
pars4 = markParams!(bs4, true)[1:5]
pVals = [0.06053894364993645, 0.14969526640945469, 0.6609406021835533, 
         0.6053365604566768, 1.582624636863842]
grad4_t = [ 0.00011524226901650934, 0.013112845298709097, 0.008472955326960456, 
           -0.016166185015739958,   0.0061833882951649495]
setindex!.(pars4, pVals)
gtb4 = GTBasis(bs4)
HFres4 = Quiqbox.runHFcore(gtb4, nuc, nucCoords, Quiqbox.defaultHFforHFgrad)
grad4 = gradOfHFenergy(pars4, gtb4, (HFres4[begin][begin].Cs[end],), nuc, nucCoords)
t4 = 5e-8
@test isapprox(grad4, grad4_t, atol=t4)

end