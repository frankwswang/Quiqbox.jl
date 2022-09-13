function compr2Arrays1(arr1::T, arr2::T, errT::Real=1e-12, factor::Real=1) where {T}
    length(arr1) == length(arr2) || 
    throw(AssertionError("`arr1` and `arr2` should have the same length."))
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

function compr2Arrays2(cprTuple::NamedTuple{<:Any, <:NTuple{2}}, 
                       cutoffIdx::Int, atol::Real, atol2::Real=10*atol1, 
                       idxDirecFunc::Function=(>))
    arr1, arr2 = cprTuple
    length(arr1) == length(arr2) || 
    throw(AssertionError("`arr1` and `arr2` should have the same length."))
    bools = isapprox.(arr1, arr2; atol)
    ids = findall(isequal(false), bools)
    if length(ids) > 0
        for (name, val) in zip(keys(cprTuple), cprTuple)
            println(lpad(name, 8), "[$(ids)] = ", val[ids])
        end
        @test all(idxDirecFunc(idx, cutoffIdx) for idx in ids) && 
              all(abs(arr1[idx]-arr2[idx]) < atol2 for idx in ids)
    else
        @test true
    end
end

function compr2Arrays3(cprTuple::NamedTuple{<:Any, <:NTuple{2, T}}, atol::Real, 
                       showAllDiff::Bool=false; additionalInfo::String="") where 
                      {T<:AbstractArray{<:Number}}
    name1, name2 = keys(cprTuple)
    arr1, arr2 = cprTuple
    length(arr1) == length(arr2) || 
    throw(AssertionError("`arr1` and `arr2` should have the same length."))
    res = isapprox.(arr1, arr2; atol) |> all
    if !res
        diff = arr1 - arr2
        showAllDiff && !res && println("$(name1) - $(name2) = ", diff)
        println(additionalInfo)
        v, i  = findmax(abs, diff)
        println("max(abs.($(name1) - $(name2))...) = ", v, "  index = ", i)
    end
    @test res
end