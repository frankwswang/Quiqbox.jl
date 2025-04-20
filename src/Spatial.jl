export EncodedField, GaussFunc, AxialProduct, PolyRadialFunc

using LinearAlgebra

(::SelectTrait{InputStyle})(::FieldAmplitude{<:Any, N}) where {N} = TupleInput{Real, N}()


dimOf(::FieldAmplitude{<:Any, N}) where {N} = N

returnTypeOf(::FieldAmplitude{T}) where {T} = T


needFieldAmpEvalCache(::FieldAmplitude) = true

function evalFieldAmplitude(f::FieldAmplitude, input; 
                            cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox(Number))
    formattedInput = formatInput(f, input)
    if needFieldAmpEvalCache(f)
        evalFieldAmplitudeCore(f, formattedInput, cache!Self)
    else
        evalFieldAmplitudeCore(f, formattedInput)
    end
end

(f::FieldAmplitude)(input) = evalFieldAmplitude(f, input)


struct FieldParamFunc{T<:Number, D, F<:AbstractParamFunc} <: AbstractParamFunc
    core::TypedParamFunc{T, FilterParamApply{ TupleHeader{D, F} }}

    function FieldParamFunc{T, D}(f::TupleHeader{D, F}, scope::TaggedSpanSetFilter) where 
                                 {T, D, F<:AbstractParamFunc}
        inner = EncodeParamApply(TupleHeader(f, Val(D)), scope)
        new{T, D, F}(ReturnTyped(inner, T))
    end
end

(f::FieldParamFunc)(input, params::AbstractSpanValueSet) = f.core(input, params)


function unpackFunc!(f::FieldAmplitude{T, D}, paramSet::AbstractSpanParamSet, 
                     paramSetId::Identifier=Identifier(paramSet)) where {T, D}
    fCore, localParamSet = unpackFieldFunc(f)
    idxFilter = locateParam!(paramSet, localParamSet)
    scope = TaggedSpanSetFilter(idxFilter, paramSetId)
    FieldParamFunc{T, D}(fCore, scope)
end

function unpackFunc(f::FieldAmplitude{T, D}) where {T, D}
    fCore, paramSet = unpackFieldFunc(f)
    idxFilter = SpanSetFilter(map(length, paramSet)...)
    scope = TaggedSpanSetFilter(idxFilter, Identifier(paramSet))
    FieldParamFunc{T, D}(fCore, scope), paramSet
end


struct EncodedField{T<:Number, D, F<:Function, E<:Function} <: FieldAmplitude{T, D}
    core::ReturnTyped{T, F}
    encode::TupleHeader{D, E}

    function EncodedField(core::ReturnTyped{T, F}, encode::TupleHeader{D, E}) where 
                         {T<:Number, F<:Function, D, E<:Function}
        if E <: FieldAmplitude
            d = dimOf(encode.f)
            (d == D) || throw(AssertionError("Cannot wrap a $d-dimensional field in $D "*
                                             "dimension."))
        else
            checkArgQuantity(encode.f, 1)
        end

        if F <: FieldAmplitude
            t = returnTypeOf(core.f)
            promote_type(t, T) <: T || 
            throw(AssertionError("Cannot convert the output of `f.f` from `$t` to $T."))
        else
            checkArgQuantity(core.f, 1)
        end

        new{T, D, F, E}(core, encode)
    end
end

function EncodedField(core::ReturnTyped{T, F}, ::Val{D}) where {T, D, F<:Function}
    EncodedField(core, TupleHeader( Val(D) ))
end

function EncodedField(core::FieldAmplitude{T}, dimInfo) where {T}
    EncodedField(ReturnTyped(core, T), dimInfo)
end

needFieldAmpEvalCache(::EncodedField) = true

function evalFieldAmplitudeCore(f::EncodedField{T, D, F, E}, input, 
                                cache!Self::MultiSpanDataCacheBox) where 
                               {T<:Number, D, F<:Function, E<:Function}
    val = formatInput(TupleInput{Real, D}(), input)
    for (caller, type) in zip((f.encode, f.core), (E, F))
        val = if type <: FieldAmplitude
            evalFieldAmplitude(caller.f, val; cache!Self)
        else
            caller(val)
        end
    end
    convert(T, val)
end

function unpackFieldFunc(f::EncodedField{<:Number, D}) where {D}
    paramSet = initializeSpanParamSet()
    fCore = unpackFunc!(f.core, paramSet, Identifier(nothing))
    fEncode = unpackFunc!(f.encode, paramSet, Identifier(nothing))
    TupleHeader(ParamPipeline((fEncode, fCore)), Val(D)), paramSet
end


const WrappedField{T, D, F<:Function} = EncodedField{T, D, F, ItsType}


const RadialField{T, D, F<:FieldAmplitude{T, 1}} = 
      EncodedField{T, D, F, typeof(LinearAlgebra.norm)}

RadialField{T, D}(radial::FieldAmplitude{T, 1}) where {T, D} = 
EncodedField(radial, TupleHeader( LinearAlgebra.norm, Val(D) ))

RadialField(radial::FieldAmplitude{T, 1}, ::Val{D}) where {T, D} = 
RadialField{T, D}(radial)


struct CurriedField{T<:Number, D, F<:Function, P<:NamedParamTuple} <: FieldAmplitude{T, D}
    core::TypedTupleFunc{T, D, ParamFreeFunc{F}}
    param::P

    function CurriedField(core::TypedTupleFunc{T, D, ParamFreeFunc{F}}, 
                          params::P=NamedTuple()) where 
                         {T<:Number, D, F<:Function, P<:NamedParamTuple}
        checkArgQuantity(core.f.f.core, ifelse(isempty(params), 1, 2))
        new{T, D, F, P}(core, params)
    end
end


const NullaryField{T<:Number, D, F<:Function} = CurriedField{T, D, F, @NamedTuple{}}

needFieldAmpEvalCache(::NullaryField) = false

function evalFieldAmplitudeCore(f::NullaryField, input)
    f.core(formatInput(f, input))
end

function unpackFieldFunc(f::NullaryField{<:Number, D}) where {D}
    TupleHeader(InputConverter(f.core.f.f), Val(D)), initializeSpanParamSet(nothing)
end


needFieldAmpEvalCache(::CurriedField) = true

function evalFieldAmplitudeCore(f::CurriedField, input, cache!Self::MultiSpanDataCacheBox)
    paramVals = cacheParam!(cache!Self, f.param)
    f.core(formatInput(f, input), paramVals)
end

function unpackFieldFunc(f::CurriedField{<:Number, D}) where {D}
    params = f.param

    paramEncoder = if length(params) == 1
        sym, par = (first∘pairs)(params)
        encoder, paramSet = compressParam(par)
        NamedMapper(encoder, sym)
    else
        paramSet = initializeSpanParamSet()
        map(params) do param
            encoder, inputSet = compressParam(param)
            inputFilter = locateParam!(paramSet, inputSet)
            encoder ∘ inputFilter
        end |> NamedMapper
    end
    tagFilter = TaggedSpanSetFilter(paramEncoder, Identifier(nothing))
    TupleHeader(EncodeParamApply(f.core.f.f, tagFilter), Val(D)), paramSet
end


function computeGaussFunc((input,)::Tuple{Real}, 
                          params::@NamedTuple{xpn::T}) where {T<:Real}
    exp(-params.xpn * input * input)
end

const GaussFunc{T<:Real, P<:UnitParam{T}} = 
      CurriedField{T, 1, typeof(computeGaussFunc), @NamedTuple{xpn::P}}

function GaussFunc(xpn::UnitOrVal{T}) where {T<:Real}
    core = TypedTupleFunc(ParamFreeFunc(computeGaussFunc), T, Val(1))
    CurriedField(core, (xpn=UnitParamEncoder(T, :xpn, 1)(xpn),))
end

const GaussFuncCore{T<:Real, F<:ComputeGraph{T}} = 
      TupleHeader{1, EncodeParamApply{ typeof(computeGaussFunc), ItsType, 
                                          MonoNMapper{Base.ComposedFunction{F}} }}


struct ProductField{T, D, B<:NonEmptyTuple{ FieldAmplitude{T} }} <: FieldAmplitude{T, D}
    basis::B

    function ProductField(bases::B) where {T, B<:NonEmptyTuple{ FieldAmplitude{T} }}
        new{T, mapreduce(dimOf, +, bases), B}(bases)
    end
end

ProductField(basis::Tuple{FieldAmplitude}) = first(basis)

function evalFieldAmplitude(f::ProductField{T}, input; 
                            cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox(Number), 
                            ) where {T}
    idx = firstindex(input)
    mapreduce(StableMul(T), f.basis) do basis
        iStart = idx
        idx += dimOf(basis)
        evalFieldAmplitude(basis, input[iStart:idx-1]; cache!Self)
    end
end

function unpackFieldFunc(f::ProductField{T, D}) where {T<:Number, D}
    paramSet = initializeSpanParamSet()

    idx = 1
    basisCores = map(f.basis) do basis
        iStart = idx
        basisDim = dimOf(basis)
        idx += basisDim
        getSubIdx = basisDim==1 ? GetIndex{OneToIndex}(idx-1) : GetRange(iStart, idx-1)
        basisCore = unpackFunc!(basis, paramSet, Identifier(nothing))
        ParamPipeline((InputConverter(getSubIdx), basisCore))
    end

    TupleHeader(ParamCombiner(StableMul(T), basisCores), Val(D)), paramSet
end


const AxialProduct{T<:Number, D, B<:NTuple{D, FieldAmplitude{T, 1}}} = ProductField{T, D, B}

AxialProduct(bases::NonEmptyTuple{FieldAmplitude{T, 1}}) where {T} = ProductField(bases)

function AxialProduct(basis::FieldAmplitude{T, 1}, dim::Int) where {T}
    checkPositivity(dim)
    ProductField(ntuple(_->basis, dim))
end


struct CoupledField{T, D, L<:FieldAmplitude{T, D}, R<:FieldAmplitude{T, D}, 
                    F<:Function} <: FieldAmplitude{T, D}
    pair::Tuple{L, R}
    coupler::ParamFreeFunc{F}

    function CoupledField(pair::Tuple{L, R}, coupler::Function) where 
                         {T, D, L<:FieldAmplitude{T, D}, R<:FieldAmplitude{T, D}}
        checkArgQuantity(coupler, 2)
        coupler = ParamFreeFunc(coupler)
        new{T, D, L, R, typeof(coupler.core)}(pair, coupler)
    end
end

function evalFieldAmplitude(f::CoupledField, input; 
                            cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox(Number))
    map(f.pair) do basis
        evalFieldAmplitude(basis, input; cache!Self)
    end |> Base.Splat(f.coupler)
end

function unpackFieldFunc(f::CoupledField{<:Number, D}) where {D}
    fL, fR = f.pair
    paramSet = initializeSpanParamSet()
    fCoreL = unpackFunc!(fL, paramSet, Identifier(nothing))
    fCoreR = unpackFunc!(fR, paramSet, Identifier(nothing))
    TupleHeader(ParamCombiner(f.coupler, (fCoreL, fCoreR)), Val(D)), paramSet
end


const PolyRadialFunc{T, D, B<:RadialField{T, D}, L} = 
      CoupledField{T, D, B, NullaryField{T, D, CartSHarmonics{D, L}}, StableMul{T}}

function PolyRadialFunc(radial::FieldAmplitude{T, 1}, 
                        angular::NonEmptyTuple{Int, D}) where {T, D}
    DimType = Val(D + 1)
    radialCore = RadialField(radial, DimType)
    polyCore = TypedTupleFunc((ParamFreeFunc∘CartSHarmonics)(angular), T, DimType)
    CoupledField((radialCore, CurriedField(polyCore)), StableMul(T))
end

const PolyRadialFuncCore{T<:Number, D, L, F<:Function} = 
      TupleHeader{D, ParamCombiner{ StableMul{T}, 
                                       Tuple{F, TupleHeader{ D, 
                                                                CartSHarmonics{D, L} }} }}


const PolyGaussProd{T, D, L} = PolyRadialFunc{T, D, <:GaussFunc{T}, L}

const EvalGaussFunc{T, C<:GaussFuncCore{T}} = FieldParamFunc{T, 1, C}

const EvalPolyRadialFunc{T, D, L, C} = 
      FieldParamFunc{T, D, PolyRadialFuncCore{T, D, L, C}}

const EvalPolyGaussProd{T, D, L, C<:GaussFuncCore{T}} = EvalPolyRadialFunc{T, D, L, C}