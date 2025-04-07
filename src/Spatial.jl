export WrappedField, GaussFunc, AxialProdFunc, PolyRadialFunc

using LinearAlgebra

struct FieldParamFunc{T<:Number, D, F<:AbstractParamFunc} <: AbstractParamFunc
    core::GetParamApply{ParamTupleEncoder{T, D, F}, TaggedSpanSetFilter}

    function FieldParamFunc{T, D}(f::F, scope::TaggedSpanSetFilter) where 
                                 {T, D, F<:AbstractParamFunc}
        new{T, D, F}(EncodeParamApply(ReturnTyped(f, T), scope))
    end
end

(f::FieldParamFunc)(input, params::AbstractSpanValueSet) = f.core(input, params)


getFieldConstraint(::FieldAmplitude{T, 0}) where {T} = Real
getFieldConstraint(::FieldAmplitude{T, D}) where {T, D} = NTuple{D, Real}

function evalFieldAmplitude end

(f::FieldAmplitude)(input) = evalFieldAmplitude(f, input)


function unpackFunc!(f::FieldAmplitude{T, D}, paramSet::AbstractSpanParamSet, 
                     paramSetId::Identifier=Identifier(paramSet)) where {T, D}
    fCore, localParamSet = unpackFieldFunc(f)
    idxFilter = locateParam!(paramSet, localParamSet)
    scope = TaggedSpanSetFilter(idxFilter, paramSetId)
    FieldParamFunc{T, D}(fCore, scope), paramSet
end

function unpackFunc(f::FieldAmplitude{T, D}) where {T, D}
    fCore, paramSet = unpackFieldFunc(f)
    idxFilter = SpanSetFilter(map(length, paramSet)...)
    scope = TaggedSpanSetFilter(idxFilter, Identifier(paramSet))
    FieldParamFunc{T, D}(fCore, scope), paramSet
end


struct WrappedField{T<:Number, D, F<:Function} <: FieldAmplitude{T, D}
    core::ParamTupleEncoder{T, D, F}

    function WrappedField(f::F, ::Type{T}, ::Val{D}) where {F, T, D}
        new{T, D, F}( ParamTupleEncoder(f, T, Bal(D)) )
    end
end

evalFieldAmplitude(f::WrappedField, input, 
                   cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox(Number)) = 
f.core(input::getFieldConstraint(f))

#! Potential Optimization: Add a method for a `WrappedField` that wraps a lucent ParamFunc.
function unpackFieldFunc(f::WrappedField{<:Number, D}) where {D}
    fCore, paramSet = unpackFuncCore!(f.core.f.f)
    TupleHeader(fCore, Val(D)), paramSet
end


const ParamGraphEncoder{F<:EvalParamGraph} = 
      NamedMapEncode{Deref{F}, 1, Tuple{SpanSetFilter}}

function ParamGraphEncoder(f::EvalParamGraph, sFilter::SpanSetFilter, sym::Symbol)
    NamedMapEncode(Deref(f), sFilter, sym)
end


struct ParamExecutor{F<:Function, N, E<:NTuple{N, ParamGraphEncoder}} <: AbstractParamFunc
    apply::EncodeParamApply{F, N, E}

    function ParamExecutor(finalizer::F, 
                              paramMappers::NTuple{N, EvalParamGraph}, 
                              inputFilters::NTuple{N, SpanSetFilter}, 
                              paramSymbols::NTuple{N, Symbol}) where {F<:Function, N}
        paramEncoders = map(paramMappers, inputFilters, paramSymbols) do m, f, s
            NamedMapEncode(m, f, s)
        end
        apply = EncodeParamApply(finalizer, paramEncoders, paramSymbols)
        new{F, N, typeof(paramEncoders)}(apply)
    end
end

ParamExecutor(finalizer::Function, paramMapper::EvalParamGraph, 
              inputFilter::SpanSetFilter, paramSymbol::Symbol) = 
ParamExecutor(finalizer, (paramMapper,), (inputFilter,), (paramSymbol,))

ParamExecutor(finalizer::Function) = ParamExecutor(finalizer, (), (), ())

(f::ParamExecutor)(input, params::AbstractSpanValueSet) = f.apply(input, params)


abstract type FieldAmpBuilder{T<:Number, D} <: CompositeFunction end

isCacheFieldBuilder(::FieldAmpBuilder) = false

const NamedParamTuple{S, N, P<:NTuple{N, ParamBox}} = NamedTuple{S, P}

struct CurriedField{T<:Number, D, F<:FieldAmpBuilder{T, D}, P<:NamedParamTuple
                    } <: FieldAmplitude{T, D}
    core::F
    param::P

    function CurriedField(core::F, params::NamedTuple{S, P}=NamedTuple()) where {T, D, 
                          F<:FieldAmpBuilder{T, D}, S, P<:NamedParamTuple}
        isLucent(core) || throw("`core` must be a lucent function.")
        new{T, D, F, P}(core, params)
    end
end

const EncodedField{T, D, F<:FieldAmpBuilder{T, D}} = CurriedField{T, D, F, @NamedTuple{}}

function evalFieldAmplitude(f::CurriedField, input; 
                            cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox(Number))
    parValArg = f isa EncodedField ? () : (cacheParam!(cache!Self, f.param),)
    cacheArg = isCacheFieldBuilder(f.core) ? (:cache!Self=>cache!Self,) : ()
    f.core(input::getFieldConstraint(f), parValArg...; cacheArg...)
end

function unpackFieldFunc(f::CurriedField{<:Number, D}) where {D}
    fCore, paramSet = unpackFieldFuncCore(f)
    TupleHeader(fCore, Val(D)), paramSet
end


struct GaussFuncBuilder{T<:Real} <: FieldAmpBuilder{T, 0} end

function (::GaussFuncBuilder{T})(input::Real, params::@NamedTuple{xpn::T}) where {T<:Real}
    exp(-params.xpn * input * input)
end

const GaussFunc{T<:Real, P<:UnitParam{T}} = 
      CurriedField{T, 0, GaussFuncBuilder{T}, @NamedTuple{xpn::P}}

function GaussFunc(xpn::UnitOrVal{T}) where {T<:Real}
    CurriedField(GaussFuncBuilder{T}(), (xpn=UnitParamEncoder(T, :xpn, 1)(xpn),))
end

const GaussFuncCore{T<:Function, F<:EvalParamGraph} = 
      ParamExecutor{GaussFuncBuilder{T}, 1, Tuple{ ParamGraphEncoder{F} }}

function unpackFieldFuncCore(f::GaussFunc{T, P}) where {T, P}
    xpnCore, inputSet = compressParam(f.param.xpn)
    paramSet = initializeSpanParamSet(T)
    idxFilter = locateParam!(paramSet, inputSet)
    fCore = ParamExecutor(GaussFuncBuilder{T}(), xpnCore, idxFilter, :xpn)
    fCore::GaussFuncCore{T}, paramSet
end


struct AxialProdFuncBuilder{T, D, B<:NTuple{D, FieldAmplitude{T, 0}}
                            } <: FieldAmpBuilder{T, D}
    axis::B

    function AxialProdFuncBuilder(axes::NonEmptyTuple{FieldAmplitude{T, 0}, D}) where {T, D}
        new{T, D+1, typeof(axes)}(axes)
    end
end

function (f::AxialProdFuncBuilder{T, D})(input::NTuple{D, Real}; 
                                         cache!Self::MultiSpanDataCacheBox=
                                                     MultiSpanDataCacheBox(Number)
                                         ) where {T<:Real, D}
    mapreduce(StableMul(T), f.axis) do fAxial
        evalFieldAmplitude(fAxial, input; cache!Self)
    end
end

isCacheFieldBuilder(::AxialProdFuncBuilder) = true

const AxialProdFunc{T<:Number, D, B<:NTuple{D, FieldAmplitude{T, 0}}} = 
      EncodedField{T, D, AxialProdFuncBuilder{T, D, B}}

AxialProdFunc(compos::NonEmptyTuple{FieldAmplitude{T, 0}}) where {T} = 
(CurriedField∘AxialProdFuncBuilder)(compos)

const AxialFuncCore{T, F<:FieldParamFunc{T, 0}} = 
      ParamPipeline{Tuple{ParamFreeFunc{GetIndex{OneToIndex}}, F}}

const AxialProdFuncCore{T<:Number, D, C<:NTuple{ D, AxialFuncCore{T} }} = 
      ParamCombiner{StableMul{T}, C}

function unpackFieldFuncCore(f::AxialProdFunc{T, D}) where {T, D}
    paramSet = initializeSpanParamSet(T)

    idx = 0
    fAxialCores = map(f.axis) do fAxial
        idx += 1
        fAxialCore = unpackFunc!(fAxial, paramSet, Identifier(nothing)) |> first
        ParamPipeline((ParamFreeFunc(GetIndex{OnrToIndex}(idx)), fAxialCore))
    end

    ParamCombiner(StableMul(T), fAxialCores)::AxialProdFuncCore{T, D}, paramSet
end


struct PolyRadialFuncBuilder{T<:Number, D, B<:FieldAmplitude{T, 0}, 
                             L} <: FieldAmpBuilder{T, D}
    radial::B
    angular::CartSHarmonics{D, L}
end

function (f::PolyRadialFuncBuilder{T, D})(input::NTuple{D, Real}) where {T<:Number, D}
    f.radial(LinearAlgebra.norm(input)) * f.angular(input)
end

const PolyRadialFunc{T, D, B<:FieldAmplitude{T, 0}, L} = 
      EncodedField{T, D, PolyRadialFuncBuilder{T, D, B, L}}

function PolyRadialFunc(radial::FieldAmplitude{T, 0}, angular::NonEmptyTuple{Int}) where {T}
    builder = PolyRadialFuncBuilder(radial, CartSHarmonics(angular))
    CurriedField(builder)
end

const RadialFuncCore{F<:AbstractParamFunc} = 
      ParamPipeline{Tuple{ ParamFreeFunc{typeof(LinearAlgebra.norm)}, TupleHeader{0, F} }}

const PolyRadialFuncCore{T<:Number, D, L, F<:AbstractParamFunc} = 
      ParamCombiner{StableMul{T}, 
                    Tuple{ RadialFuncCore{F}, ParamFreeFunc{CartSHarmonics{D, L}} }}

function unpackFieldFuncCore(f::PolyRadialFunc{T, D}) where {T, D}
    radialCore, paramSet = unpackFieldFunc(f.core.radial)
    binaryOp = StableMul(promote_type(T, Real))
    radial = ParamPipeline((ParamFreeFunc(LinearAlgebra.norm), radialCore))
    fCore = ParamCombiner(binaryOp, ( radial, ParamFreeFunc(f.angular) ))
    fCore::PolyRadialFuncCore{T, D}, paramSet
end


const PolyGaussProd{T, D, L} = PolyRadialFunc{T, D, <:GaussFunc{T}, L}

const EvalGaussFunc{T, C<:GaussFuncCore{T}} = FieldParamFunc{T, 0, C}

const EvalPolyRadialFunc{T, D, L, C<:AbstractParamFunc} = 
      FieldParamFunc{T, D, PolyRadialFuncCore{T, D, L, C}}

const EvalPolyGaussProd{T, D, L, C<:GaussFuncCore{T}} = EvalPolyRadialFunc{T, D, L, C}