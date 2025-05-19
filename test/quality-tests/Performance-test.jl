using Test
using Quiqbox
using BenchmarkTools
using Printf


@testset "Performance Test" begin

function getBenchmarkResStr(res::BenchmarkTools.Trial, header::AbstractString="")
    configStr = "(sample=$(res.times|>length), eval/sample=$(res.params.evals))"

    avgInfo = mean(res)
    minInfo = minimum(res)
    maxInfo = maximum(res)

    avgTime = avgInfo.time
    ratioGC = avgInfo.gctime / avgTime

    body = @sprintf("""
        ||Min Time: %.5f ms
        ||Mid Time: %.5f ms
        ||Avg Time: %.5f ± %.5f ms
        ||
        ||Avg GC Time / Avg Time: %.2f%%
        ||Sorted Allocs Estimate: [%d, …, %d]
        ||Sorted Memory Estimate: [%.3f KiB, …, %.3f KiB]
        +-""", 
        minInfo.time / 1e6, 
        median(res).time / 1e6, 
        avgTime / 1e6, std(res).time / 1e6, 
        100ratioGC, 
        minInfo.allocs, maxInfo.allocs, 
        minInfo.memory / 1024, maxInfo.memory / 1024
    )
    "+-<" * header * ">\n|-" * configStr * "\n" * body * "\n"
end

cen1 = (1.1, 0.5, 1.1)
cen2 = (1.0, 1.5, 1.1)

cons1 = [1.5, -0.3]
xpns1 = [1.2, 0.6]

xpns2 = [1.5, 0.6]
cons2 = [1.0, 0.8]

ang = (1, 0, 0)

cgf1 = genGaussTypeOrb(cen1, xpns1, cons1, ang)
cgf2 = genGaussTypeOrb(cen2, xpns2, cons2, ang)
cgf2n = genGaussTypeOrb(cen2, xpns2, cons2, ang, 
                        innerRenormalize=true, outerRenormalize=true)

stf1Core = Quiqbox.TypedReturn((x::Tuple{Real}) -> exp(-(x|>first)), Float64)
stf1 = Quiqbox.EncodedField(stf1Core, Val(1))
sto1 = Quiqbox.PolyRadialFunc(stf1, (1, 1, 0))
stoBasis1 = Quiqbox.PrimitiveOrb((1.0, 2.0, 3.0), sto1, renormalize=false)
stoBasis1n = Quiqbox.PrimitiveOrb((1.0, 2.0, 3.0), sto1, renormalize=true)

begin println("Benchmarking Orbital Evaluation...")
    bRes1 = @benchmark ($stoBasis1)((2.1, 3.1, 3.4)) evals=1
    (println∘getBenchmarkResStr)(bRes1, "stoBasis1((2.1, 3.1, 3.4))")

    bRes2 = @benchmark ($stoBasis1n)((2.1, 3.1, 3.4)) evals=1
    (println∘getBenchmarkResStr)(bRes2, "stoBasis1n((2.1, 3.1, 3.4))")
end

begin
    println("Benchmarking Orbital Integration...")
    bRes3 = @benchmark overlap($cgf1, $cgf1) evals=1
    (println∘getBenchmarkResStr)(bRes3, "overlap(cgf1, cgf1)")

    bRes4 = @benchmark overlap($cgf1, $cgf1, lazyCompute=true) evals=1
    (println∘getBenchmarkResStr)(bRes4, "overlap(cgf1, cgf1, lazyCompute=true)")

    bRes5 = @benchmark overlap($cgf2n, $cgf2) evals=1
    (println∘getBenchmarkResStr)(bRes5, "overlap(cgf2n, cgf2)")

    bRes6 = @benchmark overlap($stoBasis1n, $stoBasis1) evals=1
    (println∘getBenchmarkResStr)(bRes6, "overlap(stoBasis1n, stoBasis1)")
end

end