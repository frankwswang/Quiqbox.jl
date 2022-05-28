function compr2Arrays1(arr1, arr2, errT=1e-12, factor=1)
    res = true
    is = isapprox.(arr1, arr2, atol=errT)
    ids = findall(isequal(0), is)
    if length(ids) == 0
        true
    else
        println()
        for idx in ids
            if factor==1 || !isapprox(arr1[idx], arr2[idx]*factor, atol=errT)
                @show idx arr1[idx] arr2[idx]
                res *= false
            end
        end
        println()
        res
    end
end

function compr2Arrays2(cprTuple::NamedTuple{<:Any, <:NTuple{2, T}}, 
                       cutoffIdx::Int, atol::Float64, atol2::Float64=10*atol1) where {T}
    bools = isapprox.(cprTuple[1], cprTuple[2]; atol)
    ids = findall(isequal(false), bools)
    if length(ids) > 0
        @show ids
        ks = keys(cprTuple)
        for i=1:2
            println(ks[i], "[$(ids)] = ", cprTuple[i][ids])
        end
        @test all(ids .> cutoffIdx) && all(abs.(Et1[ids] - rhfs[ids]) .< atol2)
    else
        @test true
    end
end