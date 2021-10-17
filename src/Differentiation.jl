export ParamBox, inValOf, outValOf, inSymOf, outSymOf, dataOf, mapOf, outValCopy, 
       inVarCopy, enableDiff!, disableDiff!, isDiffParam, toggleDiff!, gradHFenegy

using LinearAlgebra: eigen
using Symbolics: Num

# Julia supports 0-D arrays but we need to differentiate parameters that are allowed to be tuned from them.
"""

    ParamBox{V, T} <: DifferentiableParameter{ParamBox, T}

Parameter container that enables parameter differentiations.

≡≡≡ Field(s) ≡≡≡

`data::T`: Stored parameter. It can be accessed through syntax `[]`.

`canDiff::Bool`: Indicator that whether this container should be marked as 
"differentiable", i.e., whether the math function that consists of it can be taken partial 
derivative with respect to it or not.

≡≡≡ Initialization Method(s) ≡≡≡

    ParamBox(data::Number, name::Symbol=:undef; mapFunction::Function=itself, 
             canDiff::Bool=true, paramType::Type{T}=Float64) -> 
    ParamBox{T}

`name` specifies the name of the variable to be stored, which helps with symbolic 
representation and automatic differentiation.

`mapFunction` is for the case to the store the variable that is a dependent variable 
(math function) f(x) of another variable x which is the actually stored in the struct, and 
linked to the f(x) via the `mapFunction`. After initializing the `ParamBox`, e.g 
`pb1 = ParamBox(x, mapFunction=f)`, `pb.data[]` returns `x`, and `pb.data()` returns `f(x)`.

`canDiff` is used to mark the (independent) variable as differentiable when set to `true`, 
otherwise the `ParamBox` will be ignored in any differentiation process like a constant.

`paramType` specifies the type of the stored variable to avoid data type mutation.

≡≡≡ Example(s) ≡≡≡

```
julia> Quiqbox.ParamBox(1.0)
ParamBox{Float64}(1.0)[∂]
```

NOTE: When the parameter inside `x::ParamBox` is marked as "differentiable" (a.k.a. 
`x.canDiff=true`), "`[∂]`" in the printing info is in color green, otherwise it's in grey.
"""
struct ParamBox{T, V, F, IV} <: DifferentiableParameter{ParamBox, T}
    data::Array{T, 0}
    map::Function
    canDiff::Array{Bool, 0}
    index::Array{<:Union{Int, Nothing}, 0}
    function ParamBox(data::Array{T, 0}, map::Function, canDiff, index, name::Symbol=:undef, 
                      dataName::Symbol=:undef) where {T<:Number}
        flag = (dataName == :undef)
        if typeof(map) === typeof(itself)
            dName = flag ? name : dataName
        else
            mapFuncStr = map |> nameOf |> string
            if startswith(mapFuncStr, '#')
                idx = parse(Int, mapFuncStr[2:end])
                fStr = "f_" * string(name) * numToSubs(idx)
                map = renameFunc(fStr, map)
            end
            dName = flag  ?  "x_" * string(name) |> Symbol  :  dataName
        end
        f = map
        new{T, name, nameOf(f), dName}(data, f, canDiff, index)
    end
end

(pb::ParamBox)() = Base.invokelatest(pb.map, pb.data[])

function ParamBox(data::Array{<:Number, 0}, name::Symbol=:undef, mapFunction::F=itself, 
                  dataName::Symbol=:undef; canDiff::Bool=true, 
                  index::Union{Int, Nothing}=nothing) where {F<:Function}
    idx = reshape(Union{Int, Nothing}[0], ()) |> collect
    idx[] = index
    ParamBox(data, mapFunction, fill(canDiff), idx, name, dataName)
end

ParamBox(x::Number, name::Symbol=:undef, mapFunction::F=itself, dataName::Symbol=:undef; 
         canDiff::Bool=true, index::Union{Int, Nothing}=nothing, 
         paramType::Type{<:Number}=Float64) where {F<:Function} = 
ParamBox(fill(x |> paramType), name, mapFunction, dataName; canDiff, index)


"""

    inValOf(pb::ParamBox) -> Number

Equivalent to `pb[]`.
"""
inValOf(pb::ParamBox) = pb.data[]


"""

    outValOf(pb::ParamBox) -> Number

Equivalent to `pb()`.
"""
outValOf(pb::ParamBox) = pb()


"""

    inSymOf(pb::ParamBox) -> Symbol

Return the type (`Symbol`) of the independent variable of the input `ParamBox`.
"""
inSymOf(pb::ParamBox) = typeof(pb).parameters[4]


"""

    outSymOf(pb::ParamBox) -> Symbol

Return the type (`Symbol`) of the dependent variable of the input `ParamBox`.
"""
outSymOf(pb::ParamBox) = typeof(pb).parameters[2]


dataOf(pb::ParamBox) = pb.data

mapOf(pb::ParamBox) = pb.map


function outValCopy(pb::ParamBox{<:Any, V, 
                                 <:Any, <:Any})::ParamBox{<:Any, V, :itself, V} where {V}
    ParamBox(pb(), outSymOf(pb), canDiff=pb.canDiff[])
end

function inVarCopy(pb::ParamBox{T, <:Any, <:Any, IV})::ParamBox{T, IV, :itself, IV} where 
                  {T, IV}
    ParamBox(pb.data, inSymOf(pb), canDiff=pb.canDiff[])
end


const NoDiffMark = superscriptSym['!']


"""

    enableDiff!(pb::ParamBox) -> ParamBox

Mark the input `ParamBox` as "differentiable" (the math function that consists of it can be 
taken partial derivative with respect to it). Return the marked `ParamBox`.
"""
function enableDiff!(pb::ParamBox)
    pb.canDiff[] = true
    pb
end


"""

    disableDiff!(pb::ParamBox) -> ParamBox

Mark the input `ParamBox` as "non-differentiable" (it will be treated as a constant in any 
math function that consists of it). Return the marked `ParamBox`.
"""
function disableDiff!(pb::ParamBox)
    pb.canDiff[] = false
    pb
end


"""

    isDiffParam(pb::ParamBox) -> Bool

Return the Boolean value of if the input `ParamBox` is "differentiable", i.e., whether the 
math function that consists of it can be taken partial derivative with respect.
"""
isDiffParam(pb::ParamBox) = pb.canDiff[]


"""

    toggleDiff!(pb::ParamBox) -> Bool

Toggle the differentiability (`pb.canDiff[]`) of the input `ParamBox` and return the 
altered result.
"""
toggleDiff!(pb::ParamBox) = begin pb.canDiff[] = !pb.canDiff[] end


function deriveBasisFunc(bf::CompositeGTBasisFuncs, par::ParamBox) where {N}
    varDict = getVarDictCore(bf)
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
    for i = 1:bsSize, j = 1:i, k = 1:i, l = 1:(k==i ? j : k)
        ʃabcd[i,j,k,l,:] = ʃabcd[j,i,k,l,:] = ʃabcd[j,i,l,k,:] = ʃabcd[i,j,l,k,:] = 
        ʃabcd[l,k,i,j,:] = ʃabcd[k,l,i,j,:] = ʃabcd[k,l,j,i,:] = ʃabcd[l,k,j,i,:] = 
        ʃ(bfs[i],  bfs[j],  bfs[k],  bfs[l])
    end
    for i = 1:bsSize, j=1:bsSize, k=1:bsSize, l=1:k
        ʃ∂abcd[i,j,k,l,:] = ʃ∂abcd[i,j,l,k,:] = ʃ(∂bfs[i], bfs[j],  bfs[k],  bfs[l])
    end
    for e=1:dimOfʃ
        # [∂ʃ4[i,j,k,l] == ∂ʃ4[j,i,l,k] == ∂ʃ4[j,i,k,l] != ∂ʃ4[l,j,k,i]
        for i = 1:bsSize, j = 1:i, k = 1:i, l = 1:(k==i ? j : k)
            val = 0
            # ʃ∂abcd[i,j,k,l,:] == ʃ∂abcd[i,j,l,k,:] == 
            # ʃab∂cd[l,k,i,j,:] == ʃab∂cd[k,l,i,j,:]
            for a = 1:bsSize, b = 1:bsSize, c = 1:bsSize, d = 1:bsSize
                val += (  X[a,i]*X[b,j]* X[c,k]*X[d,l] +  X[a,j]*X[b,i]* X[c,k]*X[d,l] + 
                          X[c,i]*X[d,j]* X[a,k]*X[b,l] +  X[c,i]*X[d,j]* X[a,l]*X[b,k]  ) * 
                       ʃ∂abcd[a,b,c,d,e] + 
                       ( ∂X[a,i]*X[b,j]* X[c,k]*X[d,l] + ∂X[a,j]*X[b,i]* X[c,k]*X[d,l] + 
                          X[a,i]*X[b,j]*∂X[c,k]*X[d,l] +  X[a,i]*X[b,j]*∂X[c,l]*X[d,k]  ) *  
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
                   S::Matrix{Float64}, mol::Vector{String}, 
                   nucCoords::Vector{<:AbstractArray}, 
                   nElectron::Union{Int, NTuple{2, Int}})
    Xinv = S^(0.5)
    Cₓ = (C isa Tuple) ? (Ref(Xinv) .* C) : (Xinv * C)
    ∂hij, ∂hijkl = derivativeCore(bs, par, S, 
                                  oneBodyFunc=(i,j)->coreHijCore(i,j,mol,nucCoords), 
                                  twoBodyFunc=eeInteractionCore)
    getEᵀ(dropdims(∂hij, dims=3), dropdims(∂hijkl, dims=5), Cₓ, nElectron)
end


function gradHFenegy(bs::Vector{<:CompositeGTBasisFuncs}, par::Vector{<:ParamBox}, 
                     C::Union{Matrix{Float64}, NTuple{2, Matrix{Float64}}}, 
                     S::Matrix{Float64}, mol::Vector{String}, 
                     nucCoords::Vector{<:AbstractArray}; 
                     nElectron::Union{Int, NTuple{2, Int}}=getCharge(mol))
    if length(C) == 2 && nElectron isa Int
        nElectron = (nElectron÷2, nElectron-nElectron÷2)
    end
    ∂HFenergy.(Ref(bs), par, Ref(C), Ref(S), Ref(mol), Ref(nucCoords), nElectron)
end

gradHFenegy(bs::Vector{<:CompositeGTBasisFuncs}, par::ParamBox, 
            C::Union{Matrix{Float64}, NTuple{2, Matrix{Float64}}}, S::Matrix{Float64}, 
            mol::Vector{String}, nucCoords::Vector{<:AbstractArray}; 
            nElectron::Union{Int, NTuple{2, Int}}=getCharge(mol)) = 
gradHFenegy(bs, [par], C, S, mol, nucCoords; nElectron)