struct Identity <: DirectOperator end

(::Identity)(f::Function) = itself(f)


struct MonomialMul{T, D, L} <: DirectOperator
    center::NTuple{D, T}
    degree::WeakComp{D, L}

    MonomialMul(center::NonEmptyTuple{T}, degree::WeakComp{D, L}) where {T, D, L} = 
    new{T, D, L}(center, degree)
end

MonomialMul(center::NonEmptyTuple{T, D}, degree::NonEmptyTuple{Int, D}) where {T, D} = 
MonomialMul(center, WeakComp(degree))

abstract type ConstantOperator end
abstract type ParamBoxOperator end

function (f::MonomialMul{T, D})(arg::Union{NTuple{D, T}, AbstractVector{T}}) where {T, D}
    mapreduce(StableMul(T), arg, f.center, f.degree.tuple) do a, c, d
        (a - c)^d
    end
end

function (f::MonomialMul{T, D})(target::F) where {T, D, F<:Function}
    PairCombine(StableMul(T), f, target)
end


# function transform!(::DimSpanDataCacheBox{T}, ::Identity, orb::FrameworkOrb{T}) where {T}
#     itself(orb)
# end