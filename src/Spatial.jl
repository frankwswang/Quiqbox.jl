export EncodedField, GaussFunc, AxialProduct, PolyRadialFunc

using LinearAlgebra: norm as generalNorm

(::SelectTrait{InputStyle})(::FieldAmplitude{<:Any, D}) where {D} = EuclideanInput{D}()


getOutputType(::FieldAmplitude{C}) where {C<:RealOrComplex} = C


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


struct FieldParamFunc{C<:RealOrComplex, D, F<:AbstractParamFunc, S<:SpanSetFilter
                      } <: AbstractParamFunc
    core::TypedParamFunc{C, ParamFilterApply{EuclideanHeader{D, F}, S}}

    function FieldParamFunc{C, D}(f::EuclideanHeader{D, F}, scope::TaggedSpanSetFilter{S}
                                  ) where {C<:RealOrComplex, D, F<:AbstractParamFunc, 
                                           S<:SpanSetFilter}
        if F <: InputConverter && !(S <: VoidSetFilter)
            throw(AssertionError("The `scope` corresponding to `F<:InputConverter` must "*
                                 "be `$(TaggedSpanSetFilter{VoidSetFilter})`."))
        end
        inner = ContextParamFunc(EuclideanHeader(f, Val(D)), scope)
        new{C, D, F, S}(ReturnTyped(inner, C))
    end
end

(f::FieldParamFunc)(input, params::AbstractSpanValueSet) = f.core(input, params)

const NullaryFieldFunc{C<:RealOrComplex, D, F<:AbstractParamFunc} = 
      FieldParamFunc{C, D, F, VoidSetFilter}


function unpackFunc!(f::FieldAmplitude{C, D}, paramSet::AbstractSpanParamSet, 
                     paramSetId::Identifier=Identifier(paramSet)) where 
                    {C<:RealOrComplex, D}
    fCore, localParamSet = unpackFieldFunc(f)
    idxFilter = locateParam!(paramSet, localParamSet)
    scope = TaggedSpanSetFilter(idxFilter, paramSetId)
    FieldParamFunc{C, D}(fCore, scope)
end

function unpackFunc(f::FieldAmplitude{C, D}) where {C<:RealOrComplex, D}
    fCore, paramSet = unpackFieldFunc(f)
    idxFilter = SpanSetFilter(map(length, paramSet)...)
    scope = TaggedSpanSetFilter(idxFilter, Identifier(paramSet))
    FieldParamFunc{C, D}(fCore, scope), paramSet
end


struct EncodedField{C<:RealOrComplex, D, F<:Function, E<:Function} <: FieldAmplitude{C, D}
    core::ReturnTyped{C, F}
    encode::EuclideanHeader{D, E}

    function EncodedField(core::ReturnTyped{C, F}, encode::EuclideanHeader{D, E}) where 
                         {C<:RealOrComplex, F<:Function, D, E<:Function}
        if E <: FieldAmplitude
            d = getDimension(encode.f)
            (d == D) || throw(AssertionError("Cannot wrap a $d-dimensional field in $D "*
                                             "dimension."))
        end

        if F <: FieldAmplitude
            t = getOutputType(core.f)
            promote_type(t, C) <: C || 
            throw(AssertionError("Cannot convert the output of `f.f` from `$t` to $C."))
        end

        new{C, D, F, E}(core, encode)
    end
end

function EncodedField(core::ReturnTyped{C, F}, ::Val{D}) where 
                     {C<:RealOrComplex, D, F<:Function}
    EncodedField(core, EuclideanHeader( Val(D) ))
end

function EncodedField(core::FieldAmplitude{C}, dimInfo) where {C<:RealOrComplex}
    EncodedField(ReturnTyped(core, C), dimInfo)
end

function EncodedField(core::Tuple{Function, C}, dimInfo) where {C<:RealOrComplex}
    EncodedField(ReturnTyped(first(core), C), dimInfo)
end

needFieldAmpEvalCache(::EncodedField) = true

function evalFieldAmplitudeCore(f::EncodedField{C, D, F, E}, input, 
                                cache!Self::MultiSpanDataCacheBox) where 
                               {C<:RealOrComplex, D, F<:Function, E<:Function}
    val = formatInput(EuclideanInput{D}(), input)
    for (caller, type) in zip((f.encode, f.core), (E, F))
        val = if type <: FieldAmplitude
            evalFieldAmplitude(caller.f, val; cache!Self)
        else
            caller(val)
        end
    end
    convert(C, val)
end

function unpackFieldFunc(f::EncodedField{<:RealOrComplex, D}) where {D}
    paramSet = initializeSpanParamSet()
    fCore = unpackFunc!(f.core.f, paramSet, Identifier(nothing))
    fEncode = unpackFunc!(f.encode.f, paramSet, Identifier(nothing))
    EuclideanHeader(ParamPipeline((fEncode, fCore)), Val(D)), paramSet
end


const WrappedField{C<:RealOrComplex, D, F<:Function} = EncodedField{C, D, F, ItsType}

const EncodedFieldFunc{C<:RealOrComplex, D, E<:AbstractParamFunc, F<:AbstractParamFunc, 
                       S<:SpanSetFilter} = 
      FieldParamFunc{C, D, ParamPipeline{Tuple{E, F}}, S}


const RadialField{C<:RealOrComplex, D, F<:FieldAmplitude{C, 1}} = 
      EncodedField{C, D, F, typeof(generalNorm)}

const RadialFieldFunc{C<:RealOrComplex, D, F<:AbstractParamFunc, S<:SpanSetFilter} = 
      EncodedFieldFunc{C, D, InputConverter{typeof(generalNorm)}, F, S}

RadialField{C, D}(radial::FieldAmplitude{C, 1}) where {C<:RealOrComplex, D} = 
EncodedField(radial, EuclideanHeader( generalNorm, Val(D) ))

RadialField(radial::FieldAmplitude{C, 1}, ::Val{D}) where {C<:RealOrComplex, D} = 
RadialField{C, D}(radial)


struct CurriedField{C<:RealOrComplex, D, F<:Function, P<:NamedParamTuple
                    } <: FieldAmplitude{C, D}
    core::TypedTupleFunc{C, D, ParamFreeFunc{F}}
    param::P

    function CurriedField(core::TypedTupleFunc{C, D, ParamFreeFunc{F}}, 
                          params::P=NamedTuple()) where 
                         {C<:RealOrComplex, D, F<:Function, P<:NamedParamTuple}
        new{C, D, F, P}(core, params)
    end
end

# If `param` is empty, `.core` should not take `param` as its second argument.
const NullaryField{C<:RealOrComplex, D, F<:Function} = CurriedField{C, D, F, @NamedTuple{}}

needFieldAmpEvalCache(::NullaryField) = false

function evalFieldAmplitudeCore(f::NullaryField, input)
    f.core(formatInput(f, input))
end

function unpackFieldFunc(f::NullaryField{<:RealOrComplex, D}) where {D}
    EuclideanHeader(InputConverter(f.core.f.f), Val(D)), initializeSpanParamSet(nothing)
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
    EuclideanHeader(ContextParamFunc(f.core.f.f, tagFilter), Val(D)), paramSet
end


function computeGaussFunc(input::Tuple{Real}, params::@NamedTuple{xpn::T}) where {T<:Real}
    x, = input
    exp(-params.xpn * x * x)
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


struct ProductField{C<:RealOrComplex, D, B<:NonEmptyTuple{FieldAmplitude{C}}
                    } <: FieldAmplitude{C, D}
    basis::B

    function ProductField(bases::B) where 
                         {C<:RealOrComplex, B<:NonEmptyTuple{ FieldAmplitude{C} }}
        new{C, mapreduce(getDimension, +, bases), B}(bases)
    end
end

ProductField(basis::Tuple{FieldAmplitude}) = first(basis)

function evalFieldAmplitude(f::ProductField{C}, input; 
                            cache!Self::MultiSpanDataCacheBox=MultiSpanDataCacheBox(), 
                            ) where {C<:RealOrComplex}
    idx = firstindex(input)
    mapreduce(StableMul(C), f.basis) do basis
        iStart = idx
        idx += getDimension(basis)
        evalFieldAmplitude(basis, input[iStart:idx-1]; cache!Self)
    end
end

function unpackFieldFunc(f::ProductField{C, D}) where {C<:RealOrComplex, D}
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

    EuclideanHeader(ParamCombiner(StableMul(C), basisCores), Val(D)), paramSet
end


const AxialProduct{C<:RealOrComplex, D, B<:NTuple{D, FieldAmplitude{C, 1}}} = 
      ProductField{C, D, B}

AxialProduct(bases::NonEmptyTuple{FieldAmplitude{C, 1}}) where {C<:RealOrComplex} = 
ProductField(bases)

function AxialProduct(basis::FieldAmplitude{C, 1}, dim::Int) where {C<:RealOrComplex}
    checkPositivity(dim)
    ProductField(ntuple(_->basis, dim))
end


struct CoupledField{C<:RealOrComplex, D, L<:FieldAmplitude{C, D}, R<:FieldAmplitude{C, D}, 
                    F<:Function} <: FieldAmplitude{C, D}
    pair::Tuple{L, R}
    coupler::ParamFreeFunc{F}

    function CoupledField(pair::Tuple{L, R}, coupler::Function) where 
                         {C<:RealOrComplex, D, L<:FieldAmplitude{C, D}, 
                          R<:FieldAmplitude{C, D}}
        coupler = ParamFreeFunc(coupler)
        new{C, D, L, R, typeof(coupler.core)}(pair, coupler)
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
    EuclideanHeader(ParamCombiner(f.coupler, (fCoreL, fCoreR)), Val(D)), paramSet
end


const CartAngMomFunc{C<:RealOrComplex, D} = NullaryField{C, D, CartSHarmonics{D}}

const PolyRadialFunc{C<:RealOrComplex, D, F<:FieldAmplitude{C, 1}} = 
      CoupledField{C, D, RadialField{C, D, F}, CartAngMomFunc{C, D}, StableMul{C}}

function PolyRadialFunc(radial::FieldAmplitude{C, 1}, 
                        angular::NonEmptyTuple{Int, D}) where {C<:RealOrComplex, D}
    DimType = Val(D + 1)
    radialCore = RadialField(radial, DimType)
    polyCore = TypedTupleFunc((ParamFreeFunc∘CartSHarmonics)(angular), C, DimType)
    CoupledField((radialCore, CurriedField(polyCore)), StableMul(C))
end

const PolyGaussFunc{T<:Real, D, F<:GaussFunc{T}} = PolyRadialFunc{T, D, F}

const PolyRadialFieldCore{C<:RealOrComplex, D, F<:AbstractParamFunc, S<:SpanSetFilter} = 
      ParamCombiner{ParamFreeFunc{StableMul{C}}, 
                    Tuple{ RadialFieldFunc{C, D, F, S}, 
                           NullaryFieldFunc{C, D, InputConverter{ CartSHarmonics{D} }} }}

const PolyGaussFieldCore{T<:Real, D, F<:ParamMapper, S<:SpanSetFilter} = 
      PolyRadialFieldCore{T, D, GaussFieldFunc{T, F, S}, S}

const PolyRadialFieldFunc{C<:RealOrComplex, D, F<:PolyRadialFieldCore{C, D}, 
                          S<:SpanSetFilter} = 
      FieldParamFunc{C, D, F, S}

const PolyGaussFieldFunc{T<:Real, D, F<:PolyGaussFieldCore{T, D}, S<:SpanSetFilter} = 
      PolyRadialFieldFunc{T, D, F, S}


#= Additional Method =#
function strictTypeJoin(::Type{EncodedField{CL, D, FL, EL}}, 
                        ::Type{EncodedField{CR, D, FR, ER}}) where 
                       {D, CL<:RealOrComplex, FL<:Function, EL<:Function, 
                           CR<:RealOrComplex, FR<:Function, ER<:Function}
    p = (;C=strictTypeJoin(CL, CR), D, F=strictTypeJoin(FL, FR), E=strictTypeJoin(EL, ER))
    genParametricType(EncodedField, p)
end

function strictTypeJoin(::Type{CurriedField{CL, D, FL, PL}}, 
                        ::Type{CurriedField{CR, D, FR, PR}}) where 
                       {D, CL<:RealOrComplex, FL<:Function, PL<:NamedParamTuple, 
                           CR<:RealOrComplex, FR<:Function, PR<:NamedParamTuple}
    p = (;C=strictTypeJoin(CL, CR), D, F=strictTypeJoin(FL, FR), P=strictTypeJoin(PL, PR))
    genParametricType(CurriedField, p)
end

function strictTypeJoin(::Type{ProductField{CL, D, BL}}, 
                        ::Type{ProductField{CR, D, BR}}) where 
                       {D, CL<:RealOrComplex, BL<:NonEmptyTuple{ FieldAmplitude{CL} }, 
                           CR<:RealOrComplex, BR<:NonEmptyTuple{ FieldAmplitude{CR} }}
    p = (;C=strictTypeJoin(CL, CR), D, B=strictTypeJoin(BL, BR))
    typeintersect(genParametricType(ProductField, p), ProductField)
end

function strictTypeJoin(::Type{CoupledField{CL, D, LL, RL, FL}}, 
                        ::Type{CoupledField{CR, D, LR, RR, FR}}) where 
                       {D, CL<:RealOrComplex, LL<:FieldAmplitude{CL, D}, 
                           RL<:FieldAmplitude{CL, D}, FL<:Function, 
                           CR<:RealOrComplex, LR<:FieldAmplitude{CR, D}, 
                           RR<:FieldAmplitude{CR, D}, FR<:Function}
    p = (;C=strictTypeJoin(CL, CR), D, L=strictTypeJoin(LL, LR), R=strictTypeJoin(RL, RR), 
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