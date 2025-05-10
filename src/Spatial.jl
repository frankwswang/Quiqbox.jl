export EncodedField, GaussFunc, AxialProduct, PolyRadialFunc

using LinearAlgebra

(::SelectTrait{InputStyle})(::FieldAmplitude{<:Any, D}) where {D} = TupleInput{Real, D}()


getOutputType(::FieldAmplitude{T}) where {T} = T


getDimension(::ParticleFunction{D, M}) where {D, M} = Int(D*M)


needFieldAmpEvalCache(::FieldAmplitude) = true

function evalFieldAmplitude(f::FieldAmplitude, input; 
                            cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox())
    formattedInput = formatInput(f, input)
    if needFieldAmpEvalCache(f)
        evalFieldAmplitudeCore(f, formattedInput, cache!Self)
    else
        evalFieldAmplitudeCore(f, formattedInput)
    end
end

(f::FieldAmplitude)(input) = evalFieldAmplitude(f, formatInput(f, input))


struct FieldParamFunc{T<:RealOrComplex, D, F<:AbstractParamFunc, S<:SpanSetFilter
                      } <: AbstractParamFunc
    core::TypedParamFunc{T, ParamFilterApply{TupleHeader{D, F}, S}}

    function FieldParamFunc{T, D}(f::TupleHeader{D, F}, scope::TaggedSpanSetFilter{S}
                                  ) where {T, D, F<:AbstractParamFunc, S<:SpanSetFilter}
        if F <: InputConverter && !(S <: VoidSetFilter)
            throw(AssertionError("The `scope` corresponding to `F<:InputConverter` must "*
                                 "be `$(TaggedSpanSetFilter{VoidSetFilter})`."))
        end
        inner = ContextParamFunc(TupleHeader(f, Val(D)), scope)
        new{T, D, F, S}(ReturnTyped(inner, T))
    end
end

const NullaryFieldFunc{T<:RealOrComplex, D, F<:AbstractParamFunc} = 
      FieldParamFunc{T, D, F, VoidSetFilter}

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


struct EncodedField{T<:RealOrComplex, D, F<:Function, E<:Function} <: FieldAmplitude{T, D}
    core::ReturnTyped{T, F}
    encode::TupleHeader{D, E}

    function EncodedField(core::ReturnTyped{T, F}, encode::TupleHeader{D, E}) where 
                         {T<:RealOrComplex, F<:Function, D, E<:Function}
        if E <: FieldAmplitude
            d = getDimension(encode.f)
            (d == D) || throw(AssertionError("Cannot wrap a $d-dimensional field in $D "*
                                             "dimension."))
        end

        if F <: FieldAmplitude
            t = getOutputType(core.f)
            promote_type(t, T) <: T || 
            throw(AssertionError("Cannot convert the output of `f.f` from `$t` to $T."))
        end

        new{T, D, F, E}(core, encode)
    end
end

function EncodedField(core::ReturnTyped{T, F}, ::Val{D}) where 
                     {T<:RealOrComplex, D, F<:Function}
    EncodedField(core, TupleHeader( Val(D) ))
end

function EncodedField(core::FieldAmplitude{T}, dimInfo) where {T<:RealOrComplex}
    EncodedField(ReturnTyped(core, T), dimInfo)
end

function EncodedField(core::Tuple{Function, T}, dimInfo) where {T<:RealOrComplex}
    EncodedField(ReturnTyped(first(core), T), dimInfo)
end

needFieldAmpEvalCache(::EncodedField) = true

function evalFieldAmplitudeCore(f::EncodedField{T, D, F, E}, input, 
                                cache!Self::MultiSpanDataCacheBox) where 
                               {T<:RealOrComplex, D, F<:Function, E<:Function}
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

function unpackFieldFunc(f::EncodedField{<:RealOrComplex, D}) where {D}
    paramSet = initializeSpanParamSet()
    fCore = unpackFunc!(f.core.f, paramSet, Identifier(nothing))
    fEncode = unpackFunc!(f.encode.f, paramSet, Identifier(nothing))
    TupleHeader(ParamPipeline((fEncode, fCore)), Val(D)), paramSet
end


const WrappedField{T<:RealOrComplex, D, F<:Function} = EncodedField{T, D, F, ItsType}

const EncodedFieldFunc{T<:RealOrComplex, D, E<:AbstractParamFunc, F<:AbstractParamFunc, 
                       S<:SpanSetFilter} = 
      FieldParamFunc{T, D, ParamPipeline{Tuple{E, F}}, S}


const RadialField{T<:RealOrComplex, D, F<:FieldAmplitude{T, 1}} = 
      EncodedField{T, D, F, typeof(LinearAlgebra.norm)}

const RadialFieldFunc{T<:RealOrComplex, D, F<:AbstractParamFunc, S<:SpanSetFilter} = 
      EncodedFieldFunc{T, D, InputConverter{typeof(LinearAlgebra.norm)}, F, S}

RadialField{T, D}(radial::FieldAmplitude{T, 1}) where {T, D} = 
EncodedField(radial, TupleHeader( LinearAlgebra.norm, Val(D) ))

RadialField(radial::FieldAmplitude{T, 1}, ::Val{D}) where {T, D} = 
RadialField{T, D}(radial)


struct CurriedField{T<:RealOrComplex, D, F<:Function, P<:NamedParamTuple} <: FieldAmplitude{T, D}
    core::TypedTupleFunc{T, D, ParamFreeFunc{F}}
    param::P

    function CurriedField(core::TypedTupleFunc{T, D, ParamFreeFunc{F}}, 
                          params::P=NamedTuple()) where 
                         {T<:RealOrComplex, D, F<:Function, P<:NamedParamTuple}
        new{T, D, F, P}(core, params)
    end
end

# If `param` is empty, `.core` should not take `param` as its second argument.
const NullaryField{T<:RealOrComplex, D, F<:Function} = CurriedField{T, D, F, @NamedTuple{}}

needFieldAmpEvalCache(::NullaryField) = false

function evalFieldAmplitudeCore(f::NullaryField, input)
    f.core(formatInput(f, input))
end

function unpackFieldFunc(f::NullaryField{<:RealOrComplex, D}) where {D}
    TupleHeader(InputConverter(f.core.f.f), Val(D)), initializeSpanParamSet(nothing)
end


needFieldAmpEvalCache(::CurriedField) = true

function evalFieldAmplitudeCore(f::CurriedField, input, cache!Self::MultiSpanDataCacheBox)
    paramVals = cacheParam!(cache!Self, f.param)
    f.core(formatInput(f, input), paramVals)
end

function unpackFieldFunc(f::CurriedField{<:RealOrComplex, D}) where {D}
    params = f.param
    paramMapper, paramSet = genParamMapper(params)
    tagFilter = TaggedSpanSetFilter(paramMapper, Identifier(nothing))
    TupleHeader(ContextParamFunc(f.core.f.f, tagFilter), Val(D)), paramSet
end


function computeGaussFunc((input,)::Tuple{Real}, 
                          params::@NamedTuple{xpn::T}) where {T<:Real}
    exp(-params.xpn * input * input)
end

const GaussFunc{T<:Real, P<:UnitParam{T}} = 
      CurriedField{T, 1, typeof(computeGaussFunc), @NamedTuple{xpn::P}}

const ComputeGaussFunc = typeof(computeGaussFunc)

const GaussFieldCore{F<:ParamMapper} = EncodeParamApply{ParamFreeFunc{ComputeGaussFunc}, F}

const GaussFieldFunc{T<:Real, F<:ParamMapper, S<:SpanSetFilter} = 
      FieldParamFunc{T, 1, GaussFieldCore{F}, S}

function GaussFunc(xpn::UnitOrVal{T}) where {T<:Real}
    core = TypedTupleFunc(ParamFreeFunc(computeGaussFunc), T, Val(1))
    CurriedField(core, (xpn=UnitParamEncoder(T, :xpn, 1)(xpn),))
end


struct ProductField{T<:RealOrComplex, D, B<:NonEmptyTuple{FieldAmplitude{T}}
                    } <: FieldAmplitude{T, D}
    basis::B

    function ProductField(bases::B) where {T, B<:NonEmptyTuple{ FieldAmplitude{T} }}
        new{T, mapreduce(getDimension, +, bases), B}(bases)
    end
end

ProductField(basis::Tuple{FieldAmplitude}) = first(basis)

function evalFieldAmplitude(f::ProductField{T}, input; 
                            cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox(), 
                            ) where {T}
    idx = firstindex(input)
    mapreduce(StableMul(T), f.basis) do basis
        iStart = idx
        idx += getDimension(basis)
        evalFieldAmplitude(basis, input[iStart:idx-1]; cache!Self)
    end
end

function unpackFieldFunc(f::ProductField{T, D}) where {T<:RealOrComplex, D}
    paramSet = initializeSpanParamSet()

    idx = 1
    basisCores = map(f.basis) do basis
        iStart = idx
        basisDim = getDimension(basis)
        idx += basisDim
        getSubIdx = basisDim==1 ? GetIndex{OneToIndex}(idx-1) : GetRange(iStart, idx-1)
        basisCore = unpackFunc!(basis, paramSet, Identifier(nothing))
        ParamPipeline((InputConverter(getSubIdx), basisCore))
    end

    TupleHeader(ParamCombiner(StableMul(T), basisCores), Val(D)), paramSet
end


const AxialProduct{T<:RealOrComplex, D, B<:NTuple{D, FieldAmplitude{T, 1}}} = ProductField{T, D, B}

AxialProduct(bases::NonEmptyTuple{FieldAmplitude{T, 1}}) where {T} = ProductField(bases)

function AxialProduct(basis::FieldAmplitude{T, 1}, dim::Int) where {T}
    checkPositivity(dim)
    ProductField(ntuple(_->basis, dim))
end


struct CoupledField{T<:RealOrComplex, D, L<:FieldAmplitude{T, D}, R<:FieldAmplitude{T, D}, 
                    F<:Function} <: FieldAmplitude{T, D}
    pair::Tuple{L, R}
    coupler::ParamFreeFunc{F}

    function CoupledField(pair::Tuple{L, R}, coupler::Function) where 
                         {T, D, L<:FieldAmplitude{T, D}, R<:FieldAmplitude{T, D}}
        coupler = ParamFreeFunc(coupler)
        new{T, D, L, R, typeof(coupler.core)}(pair, coupler)
    end
end

function evalFieldAmplitude(f::CoupledField, input; 
                            cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox())
    map(f.pair) do basis
        evalFieldAmplitude(basis, input; cache!Self)
    end |> Base.Splat(f.coupler)
end

function unpackFieldFunc(f::CoupledField{<:RealOrComplex, D}) where {D}
    fL, fR = f.pair
    paramSet = initializeSpanParamSet()
    fCoreL = unpackFunc!(fL, paramSet, Identifier(nothing))
    fCoreR = unpackFunc!(fR, paramSet, Identifier(nothing))
    TupleHeader(ParamCombiner(f.coupler, (fCoreL, fCoreR)), Val(D)), paramSet
end


const CartAngMomFunc{T<:RealOrComplex, D} = NullaryField{T, D, CartSHarmonics{D}}

const PolyRadialFunc{T<:RealOrComplex, D, F<:FieldAmplitude{T, 1}} = 
      CoupledField{T, D, RadialField{T, D, F}, CartAngMomFunc{T, D}, StableMul{T}}

function PolyRadialFunc(radial::FieldAmplitude{T, 1}, 
                        angular::NonEmptyTuple{Int, D}) where {T, D}
    DimType = Val(D + 1)
    radialCore = RadialField(radial, DimType)
    polyCore = TypedTupleFunc((ParamFreeFunc∘CartSHarmonics)(angular), T, DimType)
    CoupledField((radialCore, CurriedField(polyCore)), StableMul(T))
end

const PolyGaussFunc{T<:Real, D, F<:GaussFunc{T}} = PolyRadialFunc{T, D, F}

const PolyRadialFieldCore{T<:RealOrComplex, D, F<:AbstractParamFunc, S<:SpanSetFilter} = 
      BiParamFuncProd{T, RadialFieldFunc{T, D, F, S}, 
                      NullaryFieldFunc{ T, D, InputConverter{CartSHarmonics{D}} }}

const PolyGaussFieldCore{T<:Real, D, F<:ParamMapper, S<:SpanSetFilter} = 
      PolyRadialFieldCore{T, D, GaussFieldFunc{T, F, S}, S}

const PolyRadialFieldFunc{T<:RealOrComplex, D, F<:PolyRadialFieldCore{T, D}, 
                          S<:SpanSetFilter} = 
      FieldParamFunc{T, D, F, S}

const PolyGaussFieldFunc{T<:Real, D, F<:PolyGaussFieldCore{T, D}, S<:SpanSetFilter} = 
      PolyRadialFieldFunc{T, D, F, S}


strictTypeJoin(TL::Type, TR::Type) = typejoin(TL, TR)

strictTypeJoin(::Type{T}, ::Type{Complex{T}}) where {T<:Real} = RealOrComplex{T}

function strictTypeJoin(::Type{EncodedField{TL, D, FL, EL}}, 
                        ::Type{EncodedField{TR, D, FR, ER}}) where 
                       {D, TL<:RealOrComplex, FL<:Function, EL<:Function, 
                           TR<:RealOrComplex, FR<:Function, ER<:Function}
    p = (;T=strictTypeJoin(TL, TR), D, F=strictTypeJoin(FL, FR), E=strictTypeJoin(EL, ER))
    genParametricType(EncodedField, p)
end

function strictTypeJoin(::Type{CurriedField{TL, D, FL, PL}}, 
                        ::Type{CurriedField{TR, D, FR, PR}}) where 
                       {D, TL<:RealOrComplex, FL<:Function, PL<:NamedParamTuple, 
                           TR<:RealOrComplex, FR<:Function, PR<:NamedParamTuple}
    p = (;T=strictTypeJoin(TL, TR), D, F=strictTypeJoin(FL, FR), P=strictTypeJoin(PL, PR))
    genParametricType(CurriedField, p)
end

function strictTypeJoin(::Type{ProductField{TL, D, BL}}, 
                        ::Type{ProductField{TR, D, BR}}) where 
                       {D, TL<:RealOrComplex, BL<:NonEmptyTuple{ FieldAmplitude{TL} }, 
                           TR<:RealOrComplex, BR<:NonEmptyTuple{ FieldAmplitude{TR} }}
    p = (;T=strictTypeJoin(TL, TR), D, B=strictTypeJoin(BL, BR))
    typeintersect(genParametricType(ProductField, p), ProductField)
end

function strictTypeJoin(::Type{CoupledField{TL, D, LL, RL, FL}}, 
                        ::Type{CoupledField{TR, D, LR, RR, FR}}) where 
                       {D, TL<:RealOrComplex, LL<:FieldAmplitude{TL, D}, 
                           RL<:FieldAmplitude{TL, D}, FL<:Function, 
                           TR<:RealOrComplex, LR<:FieldAmplitude{TR, D}, 
                           RR<:FieldAmplitude{TR, D}, FR<:Function}
    p = (;T=strictTypeJoin(TL, TR), D, L=strictTypeJoin(LL, LR), R=strictTypeJoin(RL, RR), 
          F=strictTypeJoin(FL, FR))
    typeintersect(genParametricType(CoupledField, p), CoupledField)
end


struct FloatingField{T<:Real, D, C<:RealOrComplex{T}, F<:AbstractParamFunc, 
                     P<:OptionalSpanValueSet} <: FieldAmplitude{C, D}
    center::NTuple{D, T}
    core::TypedTupleFunc{C, D, F}
    param::P

    function FloatingField(center::NonEmptyTuple{Real}, f::FieldParamFunc{C, D, F}, 
                           paramSet::AbstractSpanParamSet) where 
                          {C<:RealOrComplex, D, F<:AbstractParamFunc}
        T = extractRealNumberType(C)
        base = f.core.f
        core = ReturnTyped(base.binder, T)
        pValSet = getField(paramSet, last(base.encode).core.core, obtain)
        new{T, D, C, F, typeof(pValSet)}(convert(NTuple{D, T}, center), core, pValSet)
    end
end

function (f::FloatingField{<:RealOrComplex, D})(coord::NTuple{D, Real}) where {D}
    f.core(coord .- f.center, f.param)
end

const FloatingPolyGaussField{T<:Real, D, F<:PolyGaussFieldCore{T, D}} = 
      FloatingField{T, D, T, F}