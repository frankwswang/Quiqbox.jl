export PrimitiveOrb, CompositeOrb, FrameworkOrb, genGaussTypeOrb

abstract type ComposedOrb{T, D, B} <: OrbitalBasis{T, D, B} end

abstract type EvalComposedOrb{T, D, B} <: TypedEvaluator{T, B} end

(f::OrbitalBasis)(x) = evalFunc(f, x)

(f::EvalComposedOrb)(input, param) = f.f(input, param)


struct ScaledOrbital{T, D, C<:EvalComposedOrb{T, D}, 
                     F<:Function} <: EvalComposedOrb{T, D, C}
    f::PairCombine{StableMul{T}, C, F}
end

function ScaledOrbital(orb::EvalComposedOrb{T}, scalar::Function) where {T}
    (ScaledOrbital∘PairCombine)(StableBinary(*, T), orb, scalar)
end


function normalizeOrbital(fCore::EvalComposedOrb{T}, paramSet::AbstractFlatParamSet, 
                          paramFieldDict::FieldPointerDict) where {T}
    pSetId = objectid(paramSet)
    nCore, nIds = genNormalizerCore(fCore, paramSet, paramFieldDict)
    ScaledOrbital( fCore, ReturnTyped(T, PointerFunc(OnlyBody(nCore), nIds, pSetId)) )
end

const NormFuncType{T} = Union{ReturnTyped{T}, Storage{T}}


function unpackParamFunc!(f::ComposedOrb{T}, paramSet::AbstractFlatParamSet) where {T}
    fEvalCore, _, paramFieldDict = unpackParamFuncCore!(f, paramSet)
    fEval = if f.renormalize
        normalizeOrbital(fEvalCore, paramSet, paramFieldDict)
    else
        ScaledOrbital(fEvalCore, (Storage∘one)(T))
    end
    fEval, paramSet, paramFieldDict
end


function genCellEncoder(::Type{T}, sym::Symbol) where {T}
    function (input)
        if input isa ElementalParam{T}
            input
        else
            p = CellParam(T(input), sym)
            setScreenLevel!(p, 1)
            p
        end
    end
end


struct PrimitiveOrb{T, D, B<:FieldAmplitude{T, D}, 
                    C<:NTuple{D, ElementalParam{T}}} <: ComposedOrb{T, D, B}
    body::B
    center::C
    renormalize::Bool
end

function PrimitiveOrb(body::B, center::NTuple{D, ParamOrValue{T}}; 
                      renormalize::Bool=false) where {T, D, B<:FieldAmplitude{T, D}}
    length(center)!=D && throw(AssertionError("The length of `center` must match `D=$D`."))
    encoder = genCellEncoder(T, :cen)
    PrimitiveOrb(body, encoder.(center); renormalize)
end

PrimitiveOrb(ob::PrimitiveOrb) = itself(ob)

struct PrimitiveOrbCore{T, D, B<:EvalFieldAmp{T, D}} <: EvalComposedOrb{T, D, B}
    f::InsertInward{B, PointerFunc{ShiftByArg{T, D}, NTuple{D, IndexPointer{Data0D}}}}
end

const EvalPrimOrb{T, D, B, F<:NormFuncType{T}} = 
      ScaledOrbital{T, D, PrimitiveOrbCore{T, D, B}, F}

function unpackParamFuncCore!(f::PrimitiveOrb{T, D}, 
                              paramSet::AbstractFlatParamSet) where {T, D}
    pSetId = objectid(paramSet)
    cenIds = locateParam!(paramSet, f.center)
    cenPointers = FieldPointer.(1:D, Ref(T|>TensorType))
    cenDict = Dict(ChainPointer.(:center, cenPointers) .=> cenIds)
    fEvalCore, _, fCoreDict = unpackFunc!(f.body, paramSet)
    fEval = InsertInward(fEvalCore, PointerFunc(ShiftByArg{T, D}(), cenIds, pSetId))
    fEvalDict = anchorFieldPointerDict(fCoreDict, FieldPointer(:body))
    PrimitiveOrbCore(fEval), paramSet, merge(cenDict, fEvalDict)
end


struct CompositeOrb{T, D, B<:FieldAmplitude{T, D}, C<:NTuple{D, ElementalParam{T}}, 
                    W<:FlattenedParam{T, 1}} <: ComposedOrb{T, D, B}
    basis::Memory{PrimitiveOrb{T, D, <:B, <:C}}
    weight::W
    renormalize::Bool

    function CompositeOrb(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                          weight::FlattenedParam{T, 1}, 
                          renormalize::Bool=false) where {T, D}
        basis, weight = getWeightParam(basis, weight)
        B = getfield.(basis, :body) |> eltype
        C = getfield.(basis, :center) |> eltype
        basisMem = Memory{PrimitiveOrb{T, D, <:B, <:C}}(basis)
        new{T, D, B, C, typeof(weight)}(basisMem, weight, renormalize)
    end
end

function getWeightParam(basis::AbstractVector{<:PrimitiveOrb{T, D}}, 
                        weight::W) where {T, D, W<:FlattenedParam{T, 1}}
    getWeightParamCore(basis, weight)
end

function getWeightParam(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                        weight::W) where {T, D, W<:FlattenedParam{T, 1}}
    getWeightParamCore(itself.(basis), weight)
end

function getWeightParamCore(basis::AbstractVector{<:PrimitiveOrb{T, D}}, 
                            weight::W) where {T, D, W<:FlattenedParam{T, 1}}
    if checkEmptiness(basis, :basis) != first(outputTypeOf(weight).size)
        throw(AssertionError("`basis` and `weight` must have the same length."))
    end
    basis, weight
end

function getWeightParamCore(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                            weight::W) where {T, D, W<:FlattenedParam{T, 1}}
    if checkEmptiness(basis, :basis) != first(outputTypeOf(weight).size)
        throw(AssertionError("`basis` and `weight` must have the same length."))
    end
    weightPieces = getEffectiveWeight(basis, weight)
    expandedWeight = ParamGrid(weightPieces, :weight)
    expandedBasis = mapfoldl(decomposeOrb, vcat, basis)
    expandedBasis, expandedWeight
end

decomposeOrb(b::T) where {T<:PrimitiveOrb} = Memory{T}([b])
decomposeOrb(b::CompositeOrb) = b.basis


function getEffectiveWeight(::PrimitiveOrb{T}, weight::FlattenedParam{T, 1}, 
                            idx::Int) where {T}
    [indexParam(weight, idx)]
end

function getEffectiveWeight(o::CompositeOrb{T}, weight::FlattenedParam{T, 1}, 
                            idx::Int) where {T}
    len = first(outputTypeOf(o.weight).size)
    outWeight = indexParam(weight, idx)
    map(i->CellParam(StableBinary(*, T), indexParam(o.weight, i), outWeight, :w), 1:len)
end

function getEffectiveWeight(o::AbstractVector{<:ComposedOrb{T, D}}, 
                            weight::FlattenedParam{T, 1}) where {T, D}
    mapreduce(vcat, enumerate(o)) do (i, x)
        getEffectiveWeight(x, weight, i)
    end
end

function CompositeOrb(basis::AbstractVector{<:ComposedOrb{T, D}}, 
                      weight::AbstractVector{<:ParamOrValue{T}}; 
                      renormalize::Bool=false) where {T, D}
    weightParams = ParamGrid(genCellEncoder(T, :w).(weight), :wBlock)
    CompositeOrb(basis, weightParams, renormalize)
end

CompositeOrb(ob::CompositeOrb) = itself(ob)

const WeightedPF{T, D, F<:EvalPrimOrb{T, D}} = 
      ScaledOrbital{T, D, F, PointOneFunc{OnlyBody{GetIndex}, Data1D}}

function compressWeightedPF(::Type{B}, ::Type{F}) where {T, D, B<:EvalFieldAmp{T, D}, F}
    boolF = isconcretetype(B)
    boolN = isconcretetype(F)
    U = if boolF
        if boolN
            EvalPrimOrb{T, D, B, F}
        else
            EvalPrimOrb{T, D, B, <:F}
        end
    elseif boolN
        EvalPrimOrb{T, D, <:B, F}
    else
        EvalPrimOrb{T, D, <:B, <:F}
    end
    if boolF && boolN
        Memory{WeightedPF{T, D, U}}
    else
        Memory{WeightedPF{T, D, <:U}}
    end
end

struct CompositeOrbCore{T, D, V<:Memory{<:WeightedPF{T, D}}} <: EvalComposedOrb{T, D, V}
    f::ChainReduce{StableAdd{T}, V}
end

const EvalCompOrb{T, D, B, F<:NormFuncType{T}} = 
      ScaledOrbital{T, D, CompositeOrbCore{T, D, B}, F}

function restrainEvalOrbType(weightedFs::AbstractVector{<:WeightedPF{T, D}}) where {T, D}
    fInnerObjs = foldl( ∘, Base.Fix2.(getfield, (:f, :left, :f)) ).(weightedFs)
    fInnerType = eltype(foldl( ∘, Base.Fix2.(getfield, (:apply, :f, :left)) ).(fInnerObjs))
    nInnerType = eltype(getfield.(fInnerObjs, :right))
    V = compressWeightedPF(fInnerType, nInnerType)
    ChainReduce(StableBinary(+, T), V(weightedFs))
end

function unpackParamFuncCore!(f::CompositeOrb{T, D}, 
                              paramSet::AbstractFlatParamSet) where {T, D}
    pSetId = objectid(paramSet)
    weightedFields = WeightedPF{T, D, <:EvalPrimOrb{T, D}}[]
    weightIdx = locateParam!(paramSet, f.weight)
    weightPointer = ChainPointer(:weight, outputTypeOf(f.weight)) => weightIdx
    paramFieldDict = mapfoldl(vcat, eachindex(f.basis), init=weightPointer) do i
        anchor = ChainPointer(:basis, FieldPointer(i))
        fInnerCore, _, innerDict = unpackParamFunc!(f.basis[i], paramSet)
        getIdx = (OnlyBody∘IndexPointer)(i)
        weight = PointerFunc(getIdx, (weightIdx,), pSetId)
        push!(weightedFields, ScaledOrbital(fInnerCore, weight))
        anchorFieldPointerDictCore(innerDict, anchor)
    end |> Dict
    CompositeOrbCore(restrainEvalOrbType(weightedFields)), paramSet, paramFieldDict
end


const PrimGTO{T, D, B<:PolyGaussProd{T, D}, C} = PrimitiveOrb{T, D, B, C}
const CompGTO{T, D, B<:PolyGaussProd{T, D}, C, P} = CompositeOrb{T, D, B, C, P}

const EvalPrimGTO{T, D, B<:EvalPolyGaussProd{T, D}, F} = EvalPrimOrb{T, D, B, F}
const EvalCompGTO{T, D, U<:EvalPrimGTO{T, D}, F} = 
      EvalCompOrb{T, D, <:Memory{<:WeightedPF{T, D, <:U}}, F}


abstract type UnpackedOrb{T, D, B} <: OrbitalBasis{T, D, B} end

struct FrameworkOrb{T, D, B<:EvalComposedOrb{T, D}, P<:AbstractFlatParamSet{T}, 
                    A<:FieldPointerDict} <: UnpackedOrb{T, D, B}
    core::B
    param::P
    anchor::A

    function FrameworkOrb(o::ComposedOrb{T, D}, 
                          paramSet::P=initializeParamSet(FlatParamSet{T})) where 
                         {T, D, P<:AbstractFlatParamSet{T}}
        core, params, indexDict = unpackFunc!(o, paramSet)
        new{T, D, typeof(core), P, typeof(indexDict)}(core, params, indexDict)
    end
end

const FPrimGTO{T, D, B<:EvalPrimGTO{T, D}, P} = FrameworkOrb{T, D, B, P}
const FCompGTO{T, D, B<:EvalCompGTO{T, D}, P} = FrameworkOrb{T, D, B, P}


unpackFunc(b::FrameworkOrb) = (b.core, b.param)


function permitRenormalize!(b::OrbitalBasis)
    b.renormalize = true
end


function forbidRenormalize!(b::OrbitalBasis)
    b.renormalize = false
end


function genGaussTypeOrb(center::NonEmptyTuple{ParamOrValue{T}, D}, 
                         xpn::ParamOrValue{T}, 
                         ijk::NonEmptyTuple{Int, D}=ntuple(_->0, Val(D)); 
                         renormalize::Bool=false) where {T, D}
    encoder1 = genCellEncoder(T, :xpn)
    encoder2 = genCellEncoder(T, :cen)
    gf = xpn |> encoder1 |> GaussFunc
    PrimitiveOrb(PolyRadialFunc(gf, ijk), encoder2.(center), renormalize)
end

function genGaussTypeOrb(center::NonEmptyTuple{ParamOrValue{T}, D}, 
                         xpns::AbstractVector{<:ParamOrValue{T}}, 
                         cons::AbstractVector{<:ParamOrValue{T}}, 
                         ijk::NonEmptyTuple{Int, D}=ntuple(_->0, Val(D)); 
                         renormalize::Bool=false) where {T, D}
    pgtoNum = checkEmptiness(xpns, :xpns)
    if pgtoNum != length(cons)
        throw(AssertionError("`xpns` and `cons` must have the same length."))
    end
    if pgtoNum == 1
        genGaussTypeOrb(center, xpns[], cons[], ijk; renormalize)
    else
        conParams = genCellEncoder(T, :con).(cons)
        pgtos = genGaussTypeOrb.(Ref(center), xpns, Ref(ijk); renormalize)
        CompositeOrb(pgtos, conParams; renormalize)
    end
end


# struct GaussTypeSubshell{T, D, L, B, O} <: OrbitalBatch{T, D, PolyGaussProd{T, D, L, B}, O}
#     seed::PolyGaussProd{T, D, L, B}
#     subshell::NTuple{O, CartSHarmonics{D, L}}

#     function GaussTypeSubshell(center::NonEmptyTuple{ParamOrValue{T}, D}, 
#                                xpns::AbstractVector{<:ParamOrValue{T}}, 
#                                cons::AbstractVector{<:ParamOrValue{T}}, 
#                                l::Int; 
#                                renormalize::Bool=false)
#         l < 0
#     end
# end