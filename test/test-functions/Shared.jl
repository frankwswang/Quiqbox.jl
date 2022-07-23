function compr2Arrays1(arr1::T, arr2::T, errT::Real=1e-12, factor::Real=1) where {T}
    @assert length(arr1) == length(arr2)
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
                       cutoffIdx::Int, atol::Real, atol2::Real=10*atol1, 
                       idxDirecFunc::Function=(>)) where {T}
    arr1, arr2 = cprTuple
    @assert length(arr1) == length(arr2)
    bools = isapprox.(arr1, arr2; atol)
    ids = findall(isequal(false), bools)
    if length(ids) > 0
        for (name, val) in zip(keys(cprTuple), cprTuple)
            println(lpad(name, 8), "[$(ids)] = ", val[ids])
        end
        @test all(idxDirecFunc.(ids,cutoffIdx)) && 
              all(abs.(arr1[ids]-arr2[ids]) .< atol2)
    else
        @test true
    end
end

function compr2Arrays3(cprTuple::NamedTuple{<:Any, <:NTuple{2, T}}, atol::Real, 
                       showAllDiff::Bool=false) where {T<:AbstractArray{<:Number}}
    name1, name2 = keys(cprTuple)
    arr1, arr2 = cprTuple
    @assert length(arr1) == length(arr2)
    res = isapprox.(arr1, arr2; atol) |> all
    if !res
        diff = arr1 - arr2
        showAllDiff && println("$(name1) - $(name2) = ", diff)
        println("max(abs.($(name1) - $(name2))...) = ", max(abs.(diff)...))
    end
    @test res
end