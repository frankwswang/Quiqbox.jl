function compr2Arrays(arr1, arr2, errT=1e-12, factor=1)
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