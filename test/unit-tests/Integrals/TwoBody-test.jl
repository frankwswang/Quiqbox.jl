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
v1 = [[1.076566132309008 0.5784702970086923 0.4074124800025758 0.44048862738995587; 
       0.5784702970086923 0.5873958310618852 0.3462701643810019 0.48010659406093475; 
       0.4074124800025758 0.3462701643810019 0.6617232405843492 0.4074796629352414; 
       0.44048862738995587 0.48010659406093475 0.4074796629352414 0.4978431903139518], 
      [0.5784702970086923 0.3294231220508208 0.2317159906644899 0.2517079644476743; 
       0.3294231220508208 0.3613037686256993 0.2097482509136124 0.29691448341455; 
       0.2317159906644899 0.2097482509136124 0.4074796629352414 0.2511787277336205; 
       0.2517079644476743 0.29691448341455 0.2511787277336205 0.3122739827511275], 
      [0.4074124800025758 0.2317159906644899 0.19964269909888754 0.1875349948596763; 
       0.2317159906644899 0.25252562218028296 0.18753499485967629 0.22483142709137904; 
       0.19964269909888754 0.18753499485967629 0.4074124800025758 0.2317159906644899; 
       0.1875349948596763 0.22483142709137904 0.2317159906644899 0.25252562218028296], 
      [0.44048862738995587 0.2517079644476743 0.1875349948596763 0.1958155996210884; 
       0.2517079644476743 0.2775677884136962 0.1731039964226634 0.23443830283513906; 
       0.1875349948596763 0.1731039964226634 0.3462701643810019 0.20974825091361243; 
       0.1958155996210884 0.23443830283513906 0.20974825091361243 0.25247894859933], 
      [0.5784702970086923 0.3294231220508208 0.2317159906644899 0.2517079644476743; 
       0.3294231220508208 0.3613037686256993 0.2097482509136124 0.29691448341455; 
       0.2317159906644899 0.2097482509136124 0.4074796629352414 0.2511787277336205; 
       0.2517079644476743 0.29691448341455 0.2511787277336205 0.3122739827511275], 
      [0.5873958310618852 0.3613037686256993 0.25252562218028296 0.2775677884136962; 
       0.3613037686256993 0.45315032846773895 0.25247894859932996 0.3769474702105444; 
       0.25252562218028296 0.25247894859932996 0.49784319031395186 0.3122739827511275; 
       0.2775677884136962 0.3769474702105444 0.3122739827511275 0.40960981285668097], 
      [0.3462701643810019 0.2097482509136124 0.18753499485967629 0.1731039964226634; 
       0.2097482509136124 0.25247894859932996 0.1958155996210884 0.23443830283513906; 
       0.18753499485967629 0.1958155996210884 0.4404886273899559 0.2517079644476743; 
       0.1731039964226634 0.23443830283513906 0.2517079644476743 0.2775677884136962], 
      [0.48010659406093475 0.29691448341455 0.22483142709137904 0.23443830283513906; 
       0.29691448341455 0.3769474702105444 0.23443830283513906 0.33033903772557727; 
       0.22483142709137904 0.23443830283513906 0.48010659406093475 0.29691448341455; 
       0.23443830283513906 0.33033903772557727 0.29691448341455 0.3769474702105444], 
      [0.4074124800025758 0.2317159906644899 0.19964269909888754 0.1875349948596763; 
       0.2317159906644899 0.25252562218028296 0.18753499485967629 0.22483142709137904; 
       0.19964269909888754 0.18753499485967629 0.4074124800025758 0.2317159906644899; 
       0.1875349948596763 0.22483142709137904 0.2317159906644899 0.25252562218028296], 
      [0.3462701643810019 0.2097482509136124 0.18753499485967629 0.1731039964226634; 
       0.2097482509136124 0.25247894859932996 0.1958155996210884 0.23443830283513906; 
       0.18753499485967629 0.1958155996210884 0.4404886273899559 0.2517079644476743; 
       0.1731039964226634 0.23443830283513906 0.2517079644476743 0.2775677884136962], 
      [0.6617232405843492 0.4074796629352414 0.4074124800025758 0.3462701643810019; 
       0.4074796629352414 0.49784319031395186 0.4404886273899559 0.48010659406093475; 
       0.4074124800025758 0.4404886273899559 1.076566132309008 0.5784702970086923; 
       0.3462701643810019 0.48010659406093475 0.5784702970086923 0.5873958310618852], 
      [0.4074796629352414 0.2511787277336205 0.2317159906644899 0.20974825091361243; 
       0.2511787277336205 0.3122739827511275 0.2517079644476743 0.29691448341455; 
       0.2317159906644899 0.2517079644476743 0.5784702970086923 0.3294231220508208; 
       0.20974825091361243 0.29691448341455 0.3294231220508208 0.3613037686256993], 
      [0.44048862738995587 0.2517079644476743 0.1875349948596763 0.1958155996210884; 
       0.2517079644476743 0.2775677884136962 0.1731039964226634 0.23443830283513906; 
       0.1875349948596763 0.1731039964226634 0.3462701643810019 0.20974825091361243; 
       0.1958155996210884 0.23443830283513906 0.20974825091361243 0.25247894859933], 
      [0.48010659406093475 0.29691448341455 0.22483142709137904 0.23443830283513906; 
       0.29691448341455 0.3769474702105444 0.23443830283513906 0.33033903772557727; 
       0.22483142709137904 0.23443830283513906 0.48010659406093475 0.29691448341455; 
       0.23443830283513906 0.33033903772557727 0.29691448341455 0.3769474702105444],
      [0.4074796629352414 0.2511787277336205 0.2317159906644899 0.20974825091361243; 
       0.2511787277336205 0.3122739827511275 0.2517079644476743 0.29691448341455; 
       0.2317159906644899 0.2517079644476743 0.5784702970086923 0.3294231220508208; 
       0.20974825091361243 0.29691448341455 0.3294231220508208 0.3613037686256993], 
      [0.4978431903139518 0.3122739827511275 0.25252562218028296 0.25247894859933; 
       0.3122739827511275 0.40960981285668097 0.2775677884136962 0.3769474702105444; 
       0.25252562218028296 0.2775677884136962 0.5873958310618852 0.3613037686256993; 
       0.25247894859933 0.3769474702105444 0.3613037686256993 0.45315032846773895]]
eeIs = reshape(cat(v1..., dims=(3,)), (4,4,4,4))
eeIs1 = eeInteractions(bs)
eeIs2 = [eeInteraction(i...)[] for i in Iterators.product(bs, bs, bs, bs)]

@test compr2Arrays1(eeIs1, eeIs, errT1)
@test compr2Arrays1(eeIs1, eeIs2, errT1)

eeIs3 = eeInteractions(bs[[1,3,2,4]])
uniqueIdx = Quiqbox.genUniqueIndices(basisSize.(bs) |> sum)
uniqueInts1 = [eeIs1[i...] for i in uniqueIdx]
uniqueInts2 = [eeIs3[i...] for i in uniqueIdx]
@test unique(eeIs1) ⊆ uniqueInts1
@test uniqueInts1  ⊆ unique(eeIs1)
@test isapprox(sort(uniqueInts1), sort(uniqueInts2), atol=errT1)

end