using Test
using Quiqbox

include("../../../test/test-functions/Shared.jl")

@testset "TwoBody.jl tests" begin

errT1 = 1e-10
errT2 = 1e-14

b1 = genBasisFunc([0,1,0], (3,2), (1,0,0), normalizeGTO=true)
b2 = genBasisFunc([1,2,3], (1,2), (0,1,0), normalizeGTO=true)
b3 = genBasisFunc([1,2,3], (1,2), (1,0,0), normalizeGTO=true)
b4 = genBasisFunc([1,1,2], (0.8,0.4))
eeI1 = eeInteraction(b1, b2, b3, b4)
eeI2 = eeInteraction(b2, b1, b3, b4)
eeI3 = eeInteraction(b1, b2, b4, b3)
eeI4 = eeInteraction(b2, b1, b4, b3)
eeI5 = eeInteraction(b3, b4, b1, b2)
eeI6 = eeInteraction(b4, b3, b1, b2)
eeI7 = eeInteraction(b3, b4, b2, b1)
eeI8 = eeInteraction(b4, b3, b2, b1)
@test isapprox(eeI1[], eeI2[], atol=errT2)
@test isapprox(eeI2[], eeI3[], atol=errT2)
@test isapprox(eeI3[], eeI4[], atol=errT2)
@test isapprox(eeI4[], eeI5[], atol=errT2)
@test isapprox(eeI5[], eeI6[], atol=errT2)
@test isapprox(eeI6[], eeI7[], atol=errT2)
@test isapprox(eeI7[], eeI8[], atol=errT2)

nucs = ["H", "H"]
cens = [[-0.7, 0.0, 0.0], [ 0.7, 0.0, 0.0]]
bs = genBasisFunc.(cens, fill("6-31G", 2)) |> flatten
bsiz = basisSize.(bs) |> sum
v1 = [[1.1401448653048112 0.6114018855091737 0.38206058180946845 0.44784188400748576; 
        0.6114018855091737 0.626369765524018 0.34091363711214107 0.4981842896930724; 
        0.38206058180946845 0.34091363711214107 0.6748720646269215 0.4137220019263136; 
        0.44784188400748576 0.4981842896930724 0.4137220019263136 0.5200082440216443],
        [0.6114018855091737 0.34515151828171564 0.21634301778438833 0.25367351679594674; 
        0.34515151828171564 0.38032856861237596 0.2050025372370765 0.30420736880783095; 
        0.21634301778438833 0.2050025372370765 0.4137220019263136 0.2531459572808446; 
        0.25367351679594674 0.30420736880783095 0.2531459572808446 0.32215345737852713],
        [0.38206058180946845 0.21634301778438833 0.17120945350064437 0.16981096699715015; 
        0.21634301778438833 0.23858217564202472 0.16981096699715015 0.20918696249164212; 
        0.17120945350064437 0.16981096699715015 0.38206058180946845 0.21634301778438836; 
        0.16981096699715015 0.20918696249164212 0.21634301778438836 0.2385821756420247],
        [0.44784188400748576 0.25367351679594674 0.16981096699715015 0.1901774302498569; 
        0.25367351679594674 0.2811148954515269 0.1642082324755546 0.23169594480241132; 
        0.16981096699715015 0.1642082324755546 0.34091363711214107 0.2050025372370765; 
        0.1901774302498569 0.23169594480241132 0.2050025372370765 0.25160103976845005],
        [0.6114018855091737 0.34515151828171564 0.21634301778438833 0.25367351679594674; 
        0.34515151828171564 0.38032856861237596 0.2050025372370765 0.30420736880783095; 
        0.21634301778438833 0.2050025372370765 0.4137220019263136 0.2531459572808446; 
        0.25367351679594674 0.30420736880783095 0.2531459572808446 0.32215345737852713],
        [0.626369765524018 0.38032856861237596 0.23858217564202472 0.2811148954515269; 
        0.38032856861237596 0.4829562753692257 0.2516010397684501 0.39183198457193574; 
        0.23858217564202472 0.2516010397684501 0.5200082440216444 0.32215345737852713; 
        0.2811148954515269 0.39183198457193574 0.32215345737852713 0.43088268760282133],
        [0.34091363711214107 0.2050025372370765 0.16981096699715015 0.1642082324755546; 
        0.2050025372370765 0.2516010397684501 0.1901774302498569 0.23169594480241137; 
        0.16981096699715015 0.1901774302498569 0.44784188400748576 0.2536735167959468; 
        0.1642082324755546 0.23169594480241137 0.2536735167959468 0.2811148954515269],
        [0.4981842896930724 0.30420736880783095 0.20918696249164212 0.23169594480241132; 
        0.30420736880783095 0.39183198457193574 0.23169594480241137 0.3372655578635625; 
        0.20918696249164212 0.23169594480241137 0.4981842896930724 0.304207368807831; 
        0.23169594480241132 0.3372655578635625 0.304207368807831 0.39183198457193574],
        [0.38206058180946845 0.21634301778438833 0.17120945350064437 0.16981096699715015; 
        0.21634301778438833 0.23858217564202472 0.16981096699715015 0.20918696249164212; 
        0.17120945350064437 0.16981096699715015 0.38206058180946845 0.21634301778438836; 
        0.16981096699715015 0.20918696249164212 0.21634301778438836 0.2385821756420247],
        [0.34091363711214107 0.2050025372370765 0.16981096699715015 0.1642082324755546; 
        0.2050025372370765 0.2516010397684501 0.1901774302498569 0.23169594480241137; 
        0.16981096699715015 0.1901774302498569 0.44784188400748576 0.2536735167959468; 
        0.1642082324755546 0.23169594480241137 0.2536735167959468 0.2811148954515269],
        [0.6748720646269215 0.4137220019263136 0.38206058180946845 0.34091363711214107; 
        0.4137220019263136 0.5200082440216444 0.44784188400748576 0.4981842896930724; 
        0.38206058180946845 0.44784188400748576 1.1401448653048112 0.6114018855091737; 
        0.34091363711214107 0.4981842896930724 0.6114018855091737 0.626369765524018],
        [0.4137220019263136 0.2531459572808446 0.21634301778438836 0.2050025372370765; 
        0.2531459572808446 0.32215345737852713 0.2536735167959468 0.304207368807831; 
        0.21634301778438836 0.2536735167959468 0.6114018855091737 0.34515151828171564; 
        0.2050025372370765 0.304207368807831 0.34515151828171564 0.38032856861237596],
        [0.44784188400748576 0.25367351679594674 0.16981096699715015 0.1901774302498569; 
        0.25367351679594674 0.2811148954515269 0.1642082324755546 0.23169594480241132; 
        0.16981096699715015 0.1642082324755546 0.34091363711214107 0.2050025372370765; 
        0.1901774302498569 0.23169594480241132 0.2050025372370765 0.25160103976845005],
        [0.4981842896930724 0.30420736880783095 0.20918696249164212 0.23169594480241132; 
        0.30420736880783095 0.39183198457193574 0.23169594480241137 0.3372655578635625; 
        0.20918696249164212 0.23169594480241137 0.4981842896930724 0.304207368807831; 
        0.23169594480241132 0.3372655578635625 0.304207368807831 0.39183198457193574],
        [0.4137220019263136 0.2531459572808446 0.21634301778438836 0.2050025372370765; 
        0.2531459572808446 0.32215345737852713 0.2536735167959468 0.304207368807831; 
        0.21634301778438836 0.2536735167959468 0.6114018855091737 0.34515151828171564; 
        0.2050025372370765 0.304207368807831 0.34515151828171564 0.38032856861237596],
        [0.5200082440216443 0.32215345737852713 0.2385821756420247 0.25160103976845005; 
        0.32215345737852713 0.43088268760282133 0.2811148954515269 0.39183198457193574; 
        0.2385821756420247 0.2811148954515269 0.626369765524018 0.38032856861237596; 
        0.25160103976845005 0.39183198457193574 0.38032856861237596 0.4829562753692257]]
eeIs = reshape(cat(v1..., dims=(3,)), (4,4,4,4))
eeIs1 = eeInteractions(bs)
eeIs2 = [eeInteraction(i...)[] for i in Iterators.product(bs, bs, bs, bs)]

@test compr2Arrays(eeIs1, eeIs, errT1)
@test compr2Arrays(eeIs1, eeIs2, errT1)

eeIs3 = eeInteractions(bs[[1,3,2,4]])
uniqueIdx = Quiqbox.genUniqueIndices(basisSize.(bs) |> sum)
uniqueInts1 = [eeIs1[i...] for i in uniqueIdx]
uniqueInts2 = [eeIs3[i...] for i in uniqueIdx]
@test unique(eeIs1) ⊆ uniqueInts1
@test uniqueInts1  ⊆ unique(eeIs1)
@test isapprox(sort(uniqueInts1), sort(uniqueInts2), atol=errT1)

end