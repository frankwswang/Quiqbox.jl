export ParamBox, inValOf, outValOf, inSymOf, outSymOf, dataOf, mapOf, outValCopy, 
       inVarCopy, enableDiff!, disableDiff!, isDiffParam, toggleDiff!, gradHFenergy

using LinearAlgebra: eigen
using Symbolics: Num

"""

    ParamBox{T, V, F} <: DifferentiableParameter{ParamBox, T}

Parameter container that can enable parameter differentiations.

≡≡≡ Field(s) ≡≡≡

`data::Array{T<:Number, 0}`: The data (parameter) stored in a 0-D `Array` that can be 
accessed by syntax `[]`.

`dataName::Symbol`: The name assigned to the data.

`map::Function`: The mathematical mapping of the data. The mapped result can be accessed by 
syntax `()`.

`canDiff::Array{Bool, 0}`: Indicator that whether this container (mapped variable) is 
marked as "differentiable".

`index::Array{<:Union{Int, Nothing}, 0}`: Additional index assigned to the parameter.

≡≡≡ Initialization Method(s) ≡≡≡

    ParamBox(data::Array{T, 0}, mapFunction::F, canDiff::Array{Bool, 0}, 
             index::Array{<:Union{Int, Nothing}, 0}, name::Symbol=:undef, 
             dataName::Symbol=:undef) where {T<:Number, F<:Function} -> 
    ParamBox{T, name, F}

    ParamBox(data::Array{T, 0}, 
             name::Symbol=:undef, mapFunction::F=itself, dataName::Symbol=:undef; 
             canDiff::Bool=true, 
             index::Union{Int, Nothing}=nothing) where {T<:Number, F<:Function} ->
    ParamBox{T, name, F}

    ParamBox(data::Number, 
             name::Symbol=:undef, mapFunction::F=itself, dataName::Symbol=:undef; 
             canDiff::Bool=true, 
             index::Union{Int, Nothing}=nothing, 
             paramType::Type{<:Number}=Float64) where {F<:Function} ->
    ParamBox{paramType, name, F}

`name` specifies the name of the (mapped) variable the `ParamBox` represents, which helps 
with symbolic representation and automatic differentiation.

`mapFunction`: The (mathematical) mapping of the data, which will be stored in the field 
`map`. It is for the case where the variable represented by the `ParamBox` is dependent on 
another independent variable of which the value is the stored data in the container. After 
initializing a `ParamBox`, e.g `pb1 = ParamBox(x, mapFunction=f)`, `pb[]` returns `x`, and 
`pb()` returns `f(x)`. `mapFunction` is set to `$(itself)` in default, which is a dummy 
function that maps the data to itself.

`canDiff` determines whether the mapped (math) variable is "marked" as differentiable (if 
the mapping is a differentiable function), i.e, whether the mapped variable `ParamBox` 
generates during the automatic differentiation procedure is treated as a dependent variable 
or an independent variable regardless of the mapping relation.

`paramType` specifies the type of the stored parameter to avoid data type mutation.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> ParamBox(1.0)
ParamBox{Float64, :undef, :itself}(1.0)[∂][undef]

julia> ParamBox(1.0, :a)
ParamBox{Float64, :a, :itself}(1.0)[∂][a]

julia> ParamBox(1.0, :a, abs)
ParamBox{Float64, :a, :abs}(1.0)[∂][x_a]
```

**NOTE 1:** The rightmost "`[∂][IV]`" in the printing info indicates the differentiability 
and the name of the represented independent variable `:IV`. When the `ParamBox` is marked 
as "differentiable", "`[∂]`" is in color green (otherwise it's in grey).

**NOTE 2:** It's always the (mapped) variable `V` generated by a `ParamBox{<:Any, V}` that 
is used to construct a basis, whereas the underlying independent variable is used to 
differentiate the basis (in other words, only when `mapFunction = $(itself)` or 
`canDiff = false` is the independent variable same as the mapped variable/represented 
parameter).
"""
struct ParamBox{T, V, F} <: DifferentiableParameter{ParamBox, T}
    data::Array{T, 0}
    dataName::Symbol
    map::Function
    canDiff::Array{Bool, 0}
    index::Array{<:Union{Int, Nothing}, 0}

    function ParamBox(data::Array{T, 0}, mapFunction::F, canDiff, index, 
                      name::Symbol=:undef, dataName::Symbol=:undef) where 
                     {T<:Number, F<:Function}
        flag = (dataName == :undef)
        if typeof(mapFunction) === typeof(itself)
            dName = flag ? name : dataName
        else
            mapFuncStr = mapFunction |> nameOf |> string
            if startswith(mapFuncStr, '#')
                idx = parse(Int, mapFuncStr[2:end])
                fStr = "f_" * string(name) * numToSubs(idx)
                mapFunction = renameFunc(fStr, mapFunction)
            end
            dName = flag  ?  "x_" * string(name) |> Symbol  :  dataName
        end
        f = mapFunction
        new{T, name, nameOf(f)}(data, dName, f, canDiff, index)
    end
end

# (pb::ParamBox)() = Base.invokelatest(pb.map, pb.data[])::Float64
(pb::ParamBox{T})() where {T} = pb.map(pb.data[]::T)
(pb::ParamBox{T, <:Any, :itself})() where {T} = inValOf(pb)

function ParamBox(data::Array{T, 0}, name::Symbol=:undef, mapFunction::F=itself, 
                  dataName::Symbol=:undef; canDiff::Bool=true, 
                  index::Union{Int, Nothing}=nothing) where {T<:Number, F<:Function}
    idx = reshape(Union{Int, Nothing}[0], ()) |> collect
    idx[] = index
    ParamBox(data, mapFunction, fill(canDiff), idx, name, dataName)
end

ParamBox(data::Number, name::Symbol=:undef, mapFunction::F=itself, dataName::Symbol=:undef; 
         canDiff::Bool=true, index::Union{Int, Nothing}=nothing, 
         paramType::Type{<:Number}=Float64) where {F<:Function} = 
ParamBox(fill(data |> paramType), name, mapFunction, dataName; canDiff, index)


"""

    inValOf(pb::ParamBox) -> Number

Return the value of stored data (independent variable) of the input `ParamBox`. Equivalent 
to `pb[]`.
"""
inValOf(pb::ParamBox{T}) where {T} = pb.data[]::T


"""

    inVarValOf(pb::ParamBox{T}) where {T} -> ::Pair{Symbolics.Num, T}

Return a `Pair` of the stored independent variable of the input `ParamBox` and 
its corresponding value.
"""
function inVarValOf(pb::ParamBox{T}) where {T}
    idx = pb.index[]
    hasIdx = idx isa Int
    ivSym = inSymOf(pb)
    ivNum = hasIdx ? Symbolics.variable(ivSym, idx) : Symbolics.variable(ivSym)
    (ivNum => pb.data[])::Pair{Symbolics.Num, T}
end


"""

    outValOf(pb::ParamBox) -> Number

Return the value of mapped data (dependent variable) of the input `ParamBox`. Equivalent to 
`pb()`.
"""
outValOf(pb::ParamBox{T, <:Any, itself}) where {T} = 
(pb::ParamBox{T2, <:Any, :itself} where T<:T2<:T)()
outValOf(pb::ParamBox{T, <:Any, <:Any}) where {T} = (pb::ParamBox{T2} where T<:T2<:T)()


"""

    inSymOf(pb::ParamBox) -> Symbol

Return the `Symbol` of the stored data (independent variable) of the input `ParamBox`.
"""
inSymOf(pb::ParamBox) = pb.dataName


"""

    outSymOf(pb::ParamBox) -> Symbol

Return the `Symbol` of the mapped data (dependent variable) of the input `ParamBox`.
"""
outSymOf(::ParamBox{<:Any, V}) where {V} = V


"""

    dataOf(pb::ParamBox{T}) where {T<:Number} -> Array{T, 0}

Return the 0-D `Array` of the data stored in the input `ParamBox`.
"""
dataOf(pb::ParamBox) = pb.data


"""

    mapOf(pb::ParamBox{<:Number, F}) where {F<:Function} -> F

Return the mapping function of the input `ParamBox`.
"""
mapOf(pb::ParamBox) = pb.map


"""

    outValCopy(pb::ParamBox{<:Number, V}) -> ParamBox{<:Number, V, :itself}

Return a new `ParamBox` of which the stored data is a **deep copy** of the mapped data from 
the input `ParamBox`.
"""
outValCopy(pb::ParamBox) = ParamBox(pb(), outSymOf(pb), canDiff=pb.canDiff[])


"""

    inVarCopy(pb::ParamBox) -> ParamBox{<:Number, <:Any, :itself}

Return a new `ParamBox` of which the stored data is a **shallow copy** of the stored data 
from the input `ParamBox`.

≡≡≡ Example(s) ≡≡≡

```jldoctest; setup = :(push!(LOAD_PATH, "../../src/"); using Quiqbox)
julia> pb1 = ParamBox(-2.0, :a, abs)
ParamBox{Float64, :a, :abs}(-2.0)[∂][x_a]

julia> pb2 = inVarCopy(pb1)
ParamBox{Float64, :x_a, :itself}(-2.0)[∂][x_a]

julia> pb1[] == pb2[] == -2.0
true

julia> pb1[] = 1.1
1.1

julia> pb2[]
1.1
```
"""
inVarCopy(pb::ParamBox) = ParamBox(pb.data, inSymOf(pb), canDiff=pb.canDiff[])


const NoDiffMark = superscriptSym['!']


"""

    enableDiff!(pb::ParamBox) -> ParamBox

Mark the input `ParamBox` as "differentiable" and return the marked `ParamBox`.
"""
function enableDiff!(pb::ParamBox)
    pb.canDiff[] = true
    pb
end


"""

    disableDiff!(pb::ParamBox) -> ParamBox

Mark the input `ParamBox` as "non-differentiable" and return the marked `ParamBox`.
"""
function disableDiff!(pb::ParamBox)
    pb.canDiff[] = false
    pb
end


"""

    isDiffParam(pb::ParamBox) -> Bool

Return the Boolean value of if the input `ParamBox` is "differentiable".
"""
isDiffParam(pb::ParamBox) = pb.canDiff[]


"""

    toggleDiff!(pb::ParamBox) -> Bool

Toggle the differentiability (`pb.canDiff[]`) of the input `ParamBox` and return the 
altered result.
"""
toggleDiff!(pb::ParamBox) = begin pb.canDiff[] = !pb.canDiff[] end


function deriveBasisFunc(bf::CompositeGTBasisFuncs, par::ParamBox) where {N}
    # varDict = getVarDictCore(bf)
    varDict = inVarValOf.(bf |> getParams) |> Dict
    vr = getVar(par)
    info = diffInfo(bf, vr, varDict)
    diffInfoToBasisFunc(bf, info)
end


function oneBodyDerivativeCore(∂bfs::Vector{<:CompositeGTBasisFuncs}, 
                               bfs::Vector{<:CompositeGTBasisFuncs}, 
                               X::Matrix{Float64}, ∂X::Matrix{Float64}, 
                               ʃ::F, isGradient::Bool = false) where {F<:Function}
    dimOfʃ = 1+isGradient*2
    bsSize = ∂bfs |> length
    ∂ʃ = ones(bsSize, bsSize, dimOfʃ)
    ʃab = ones(bsSize, bsSize, dimOfʃ)
    ∂ʃab = ones(bsSize, bsSize, dimOfʃ)
    for i = 1:bsSize, j = 1:i
       ʃab[i,j,:] = ʃab[j,i,:] = ʃ(bfs[i], bfs[j])
    end
    for i = 1:bsSize, j = 1:i
        ∂ʃab[i,j,:] = ∂ʃab[j,i,:] = ʃ(∂bfs[i], bfs[j]) + ʃ(bfs[i], ∂bfs[j])
    end
    @views begin
        for e = 1:dimOfʃ
            for i=1:bsSize, j=1:i
                # X[i,j] == X[j,i]
                ∂ʃ[i,j,e] = ∂ʃ[j,i,e] = 
                transpose( X[:,i]) * ∂ʃab[:,:,e] *  X[:,j] +
                transpose(∂X[:,i]) *  ʃab[:,:,e] *  X[:,j] +
                transpose( X[:,i]) *  ʃab[:,:,e] * ∂X[:,j]
            end
        end
    end
    ∂ʃ
end


function twoBodyDerivativeCore(∂bfs::Vector{<:CompositeGTBasisFuncs}, 
                               bfs::Vector{<:CompositeGTBasisFuncs}, 
                               X::Matrix{Float64}, ∂X::Matrix{Float64}, 
                               ʃ::F, isGradient::Bool = false) where {F<:Function}
    dimOfʃ = 1+isGradient*2
    bsSize = ∂bfs |> length
    ∂ʃ = ones(bsSize, bsSize, bsSize, bsSize, dimOfʃ)
    ʃabcd = ones(bsSize, bsSize, bsSize, bsSize, dimOfʃ)
    ʃ∂abcd = ones(bsSize, bsSize, bsSize, bsSize, dimOfʃ)
    # ʃa∂bcd = similar(ʃabcd)
    # ʃab∂cd = similar(ʃabcd)
    # ʃabc∂d = similar(ʃabcd)
    for i = 1:bsSize, j = 1:i, k = 1:i, l = 1:(k==i ? j : k)
        ʃabcd[i,j,k,l,:] = ʃabcd[j,i,k,l,:] = ʃabcd[j,i,l,k,:] = ʃabcd[i,j,l,k,:] = 
        ʃabcd[l,k,i,j,:] = ʃabcd[k,l,i,j,:] = ʃabcd[k,l,j,i,:] = ʃabcd[l,k,j,i,:] = 
        ʃ(bfs[i],  bfs[j],  bfs[k],  bfs[l])
    end
    for i = 1:bsSize, j=1:bsSize, k=1:bsSize, l=1:k
        ʃ∂abcd[i,j,k,l,:] = ʃ∂abcd[i,j,l,k,:] = ʃ(∂bfs[i], bfs[j],  bfs[k],  bfs[l])
    end
    # Following tensor elements can be provided from above on case.
    # for i = 1:bsSize, j=1:bsSize, k=1:bsSize, l=1:k
    #     ʃa∂bcd[i,j,k,l,:] = ʃa∂bcd[i,j,l,k,:] = ʃ( bfs[i], ∂bfs[j],   bfs[k],  bfs[l])
    # end
    # for i = 1:bsSize, j=1:i, k=1:bsSize, l=1:bsSize
    #     ʃab∂cd[i,j,k,l,:] = ʃab∂cd[j,i,k,l,:] = ʃ( bfs[i],  bfs[j],  ∂bfs[k],  bfs[l])
    # end
    # for i = 1:bsSize, j=1:i, k=1:bsSize, l=1:bsSize
    #     ʃabc∂d[i,j,k,l,:] = ʃabc∂d[j,i,k,l,:] = ʃ( bfs[i],  bfs[j],   bfs[k], ∂bfs[l])
    # end
    for e=1:dimOfʃ
        # [∂ʃ4[i,j,k,l] == ∂ʃ4[j,i,l,k] == ∂ʃ4[j,i,k,l] != ∂ʃ4[l,j,k,i]
        for i = 1:bsSize, j = 1:i, k = 1:i, l = 1:(k==i ? j : k)
            val = 0
            # ʃ∂abcd[i,j,k,l,:] == ʃ∂abcd[i,j,l,k,:] == 
            # ʃab∂cd[l,k,i,j,:] == ʃab∂cd[k,l,i,j,:]
            for a = 1:bsSize, b = 1:bsSize, c = 1:bsSize, d = 1:bsSize
                # Old version: Still correct.
                # val += (  X[a,i]*X[b,j]* X[c,k]*X[d,l] +  X[a,j]*X[b,i]* X[c,k]*X[d,l] + 
                #           X[c,i]*X[d,j]* X[a,k]*X[b,l] +  X[c,i]*X[d,j]* X[a,l]*X[b,k]  ) * 
                #        ʃ∂abcd[a,b,c,d,e] + 
                #        ( ∂X[a,i]*X[b,j]* X[c,k]*X[d,l] + ∂X[a,j]*X[b,i]* X[c,k]*X[d,l] + 
                #           X[a,i]*X[b,j]*∂X[c,k]*X[d,l] +  X[a,i]*X[b,j]*∂X[c,l]*X[d,k]  ) * 
                #        ʃabcd[a,b,c,d,e]
                # New version: Better readability.
                val += (  X[a,i]*X[b,j]*X[c,k]*X[d,l] + X[a,j]*X[b,i]*X[c,k]*X[d,l] + 
                          X[c,i]*X[d,j]*X[a,k]*X[b,l] + X[c,i]*X[d,j]*X[a,l]*X[b,k]  ) * 
                       ʃ∂abcd[a,b,c,d,e] + 
                       ( ∂X[a,i]*X[b,j]* X[c,k]*X[d,l] + X[a,i]*∂X[b,j]*X[c,k]* X[d,l] + 
                          X[a,i]*X[b,j]*∂X[c,k]*X[d,l] + X[a,i]* X[b,j]*X[c,k]*∂X[d,l] ) * 
                       ʃabcd[a,b,c,d,e]
            end
            ∂ʃ[i,j,k,l,e] = ∂ʃ[j,i,k,l,e] = ∂ʃ[j,i,l,k,e] = ∂ʃ[i,j,l,k,e] = 
            ∂ʃ[l,k,i,j,e] = ∂ʃ[k,l,i,j,e] = ∂ʃ[k,l,j,i,e] = ∂ʃ[l,k,j,i,e] = val
        end
    end
    ∂ʃ
end


function derivativeCore(bs::Vector{<:CompositeGTBasisFuncs}, par::ParamBox, 
                        S::Matrix{Float64}; oneBodyFunc::F1, twoBodyFunc::F2, 
                        oneBodyGrad::Bool=false, 
                        twoBodyGrad::Bool=false) where {F1<:Function, F2<:Function}
    # ijkl in chemists' notation of spatial bases (ij|kl).
    ∂bfs = deriveBasisFunc.(bs, Ref(par)) |> flatten
    bfs = decompose.(bs) |> flatten
    bsSize = basisSize(bs) |> sum
    ∂S = ones(bsSize, bsSize)
    ∂X = ones(bsSize, bsSize) # ∂X corresponds to the derivative of X = S^(-0.5)
    ∂X₀ = ones(bsSize, bsSize) # ∂X in its eigen basis
    for i=1:bsSize, j=1:i
        S∂ij = overlap(∂bfs[i], bfs[j])
        Si∂j = overlap(bfs[i], ∂bfs[j])
        ∂S[i,j] = ∂S[j,i] = S∂ij[] + Si∂j[]
    end
    X = S^(-0.5) |> Array
    λ, 𝑣 = eigen(S)
    ∂S2 = transpose(𝑣)*∂S*𝑣
    for i=1:bsSize, j=1:i
        ∂X₀[i,j] = ∂X₀[j,i] = (- ∂S2[i,j] * λ[i]^(-0.5) * λ[j]^(-0.5) * 
                               (λ[i]^0.5 + λ[j]^0.5)^(-1))
    end
    for i=1:bsSize, j=1:bsSize
        ∂X[j,i] = [𝑣[j,k]*∂X₀[k,l]*𝑣[i,l] for k=1:bsSize, l=1:bsSize] |> sum
    end
    ∂ʃ2 = oneBodyDerivativeCore(∂bfs, bfs, X, ∂X, oneBodyFunc, oneBodyGrad)
    ∂ʃ4 = twoBodyDerivativeCore(∂bfs, bfs, X, ∂X, twoBodyFunc, twoBodyGrad)
    ∂ʃ2, ∂ʃ4
end


function ∂HFenergy(bs::Vector{<:CompositeGTBasisFuncs}, par::ParamBox, 
                   C::Union{Matrix{Float64}, NTuple{2, Matrix{Float64}}}, 
                   S::Matrix{Float64}, nuc::Vector{String}, 
                   nucCoords::Vector{<:AbstractArray}, 
                   nElectron::Union{Int, NTuple{2, Int}})
    Xinv = S^(0.5)
    Cₓ = (C isa Tuple) ? (Ref(Xinv) .* C) : (Xinv * C)
    ∂hij, ∂hijkl = derivativeCore(bs, par, S, 
                                  oneBodyFunc=(i, j)->cat(coreHij(i, j, nuc, nucCoords), 
                                                          dims=3), 
                                  twoBodyFunc=(i, j, k, l)->cat(eeInteraction(i, j, k, l), 
                                                                dims=5))
    getEᵀ(dropdims(∂hij, dims=3), dropdims(∂hijkl, dims=5), Cₓ, nElectron)
end


function gradHFenergy(bs::Vector{<:CompositeGTBasisFuncs}, par::Vector{<:ParamBox}, 
                     C::Union{Matrix{Float64}, NTuple{2, Matrix{Float64}}}, 
                     S::Matrix{Float64}, nuc::Vector{String}, 
                     nucCoords::Vector{<:AbstractArray}; 
                     nElectron::Union{Int, NTuple{2, Int}}=getCharge(nuc))
    if length(C) == 2 && nElectron isa Int
        nElectron = (nElectron÷2, nElectron-nElectron÷2)
    end
    ∂HFenergy.(Ref(bs), par, Ref(C), Ref(S), Ref(nuc), Ref(nucCoords), Ref(nElectron))
end

gradHFenergy(bs::Vector{<:CompositeGTBasisFuncs}, par::ParamBox, 
            C::Union{Matrix{Float64}, NTuple{2, Matrix{Float64}}}, S::Matrix{Float64}, 
            nuc::Vector{String}, nucCoords::Vector{<:AbstractArray}; 
            nElectron::Union{Int, NTuple{2, Int}}=getCharge(nuc)) = 
gradHFenergy(bs, [par], C, S, nuc, nucCoords; nElectron)