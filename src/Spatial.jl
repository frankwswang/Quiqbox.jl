export EncodedField, GaussFunc, AxialProduct, PolyRadialFunc

using LinearAlgebra: norm as generalNorm

(::SelectTrait{InputStyle})(::FieldAmplitude{<:Any, D}) where {D} = CartesianInput{D}()


getDimension(::ParticleFunction{D, M}) where {D, M} = Int(D*M)


needFieldAmpEvalCache(::FieldAmplitude) = false

function evalFieldAmplitude(f::FieldAmplitude, input; 
                            cache!Self::MissingOr{ParamDataCache}=missing)
    formattedInput = formatInput(f, input)
    if needFieldAmpEvalCache(f)
        ismissing(cache!Self) && (cache!Self = initializeParamDataCache())
        evalFieldAmplitudeCore(f, formattedInput, cache!Self)
    else
        evalFieldAmplitudeCore(f, formattedInput)
    end
end

(f::FieldAmplitude)(input) = evalFieldAmplitude(f, formatInput(f, input))


struct FieldParamFunc{T, D, C<:RealOrComplex{T}, F<:AbstractParamFunc, S<:SpanSetFilter
                      } <: TypedParamFunc{C}
    core::TypedReturn{C, ContextParamFunc{ F, CartesianFormatter{D, NTuple{D, T}}, 
                                           GetEntry{TaggedSpanSetFilter{S}} }}

    function FieldParamFunc{T, D, C}(f::F, scope::TaggedSpanSetFilter{S}) where 
                                    {T, C<:RealOrComplex{T}, D, F<:AbstractParamFunc, 
                                     S<:SpanSetFilter}
        checkPositivity(D)
        if F <: InputConverter && !(S <: VoidSetFilter)
            throw(AssertionError("The `scope` corresponding to `F<:InputConverter` must "*
                                 "be `$(TaggedSpanSetFilter{VoidSetFilter})`."))
        end
        inner = ContextParamFunc(f, CartesianFormatter(T, Count(D)), GetEntry(scope))
        new{T, D, C, F, S}(TypedReturn(inner, C))
    end
end

FieldParamFunc(f::TypedCarteFunc{C, D, <:AbstractParamFunc}, 
               scope::TaggedSpanSetFilter{<:SpanSetFilter}) where 
              {T, C<:RealOrComplex{T}, D} = 
FieldParamFunc{T, D, C}(f.f.f, scope)

function evalFieldParamFunc(f::FieldParamFunc{T, D, C, F}, input, params::OptSpanValueSet
                            ) where {T, C<:RealOrComplex{T}, D, F<:AbstractParamFunc}
    f.core(input, params)
end

(f::FieldParamFunc)(input, params::OptSpanValueSet) = evalFieldParamFunc(f, input, params)

getOutputType(::Type{<:FieldParamFunc{T, D, C}}) where {T, D, C<:RealOrComplex{T}} = C

const NullaryFieldFunc{T<:Real, D, C<:RealOrComplex{T}, F<:AbstractParamFunc} = 
      FieldParamFunc{T, D, C, F, VoidSetFilter}


struct StashedField{C<:RealOrComplex, D, F<:AbstractParamFunc, 
                    V<:OptSpanValueSet} <: FieldAmplitude{C, D}
    core::TypedCarteFunc{C, D, F}
    data::V

    function StashedField(f::FieldParamFunc{T, D, C}, paramSet::OptSpanParamSet, 
                          cache!Self::ParamDataCache=initializeParamDataCache()) where 
                         {T, D, C<:RealOrComplex{T}}
        fInner = f.core.f
        core = TypedCarteFunc(fInner.binder, C, Count(D))
        encoder = last(fInner.encode)
        obtainInner = Base.Fix1(obtainCore!, cache!Self)
        paramData = getEntry(paramSet, encoder.core.f.entry, obtainInner)
        new{C, D, typeof(fInner.binder), typeof(paramData)}(core, paramData)
    end
end

(f::StashedField)(input) = f.core(input, f.data)

getOutputType(::Type{<:StashedField{C}}) where {C<:RealOrComplex} = C

needFieldAmpEvalCache(::StashedField) = false

function evalFieldAmplitudeCore(f::StashedField, input)
    f.core(formatInput(f, input), f.data)
end

function unpackFieldFunc(f::F) where {D, C<:RealOrComplex, F<:StashedField{C, D}}
    TypedCarteFunc(RPartial(f.core.f.f, f.data), C, Count(D)), initializeSpanParamSet()
end


function unpackFunc!(f::F, paramSet::OptSpanParamSet, 
                     paramSetId::Identifier=Identifier(paramSet)) where 
                    {T, C<:RealOrComplex{T}, D, F<:FieldAmplitude{C, D}}
    fCore, localParamSet = unpackFieldFunc(f)
    idxFilter = locateParam!(paramSet, localParamSet)
    scope = TaggedSpanSetFilter(idxFilter, paramSetId)
    FieldParamFunc(fCore, scope)
end

function unpackFunc(f::F) where {T, C<:RealOrComplex{T}, D, F<:FieldAmplitude{C, D}}
    fCore, paramSet = unpackFieldFunc(f)
    idxFilter = SpanSetFilter(map(length, paramSet)...)
    scope = TaggedSpanSetFilter(idxFilter, paramSet)
    FieldParamFunc(fCore, scope), paramSet
end


struct EncodedField{C<:RealOrComplex, D, F<:Function, E<:Function} <: FieldAmplitude{C, D}
    core::TypedReturn{C, F}
    encode::CartesianHeader{D, E}

    function EncodedField(core::TypedReturn{C, F}, encode::CartesianHeader{D, E}) where 
                         {C<:RealOrComplex, F<:Function, D, E<:Function}
        checkPositivity(D)
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

function EncodedField(core::TypedReturn{C, F}, ::Count{D}) where 
                     {C<:RealOrComplex, D, F<:Function}
    EncodedField(core, CartesianHeader( Count(D) ))
end

function EncodedField(core::FieldAmplitude{C}, dimInfo) where {C<:RealOrComplex}
    EncodedField(TypedReturn(core, C), dimInfo)
end

function EncodedField(core::Function, ::Type{C}, dimInfo) where {C<:RealOrComplex}
    EncodedField(TypedReturn(core, C), dimInfo)
end

getOutputType(::Type{<:EncodedField{C}}) where {C<:RealOrComplex} = C

needFieldAmpEvalCache(::EncodedField) = true

function evalFieldAmplitudeCore(f::EncodedField{C, D, F, E}, input, 
                                cache!Self::ParamDataCache) where 
                               {C<:RealOrComplex, D, F<:Function, E<:Function}
    val = formatInput(CartesianInput{D}(), input)
    for (caller, type) in zip((f.encode, f.core), (E, F))
        val = if type <: FieldAmplitude
            evalFieldAmplitude(caller.f, val; cache!Self)
        else
            caller(val)
        end
    end
    convert(C, val)
end

function unpackFieldFunc(f::F) where {D, C<:RealOrComplex, F<:EncodedField{C, D}}
    paramSet = initializeSpanParamSet()
    fCore = unpackFunc!(f.core.f, paramSet, Identifier(nothing))
    fEncode = unpackFunc!(f.encode.f, paramSet, Identifier(nothing))
    TypedCarteFunc(ParamPipeline((fEncode, fCore)), C, Count(D)), paramSet
end


const WrappedField{C<:RealOrComplex, D, F<:Function} = EncodedField{C, D, F, ItsType}

const EncodedFieldFunc{T<:Real, D, C<:RealOrComplex{T}, E<:AbstractParamFunc, 
                       F<:AbstractParamFunc, S<:SpanSetFilter} = 
      FieldParamFunc{T, D, C, ParamPipeline{Tuple{E, F}}, S}


const RadialField{C<:RealOrComplex, D, F<:FieldAmplitude{C, 1}} = 
      EncodedField{C, D, F, typeof(generalNorm)}

const RadialFieldFunc{T<:Real, D, C<:RealOrComplex{T}, F<:AbstractParamFunc, 
                      S<:SpanSetFilter} = 
      EncodedFieldFunc{T, D, C, InputConverter{typeof(generalNorm)}, F, S}

RadialField{C, D}(radial::FieldAmplitude{C, 1}) where {C<:RealOrComplex, D} = 
EncodedField(radial, CartesianHeader( generalNorm, Count(D) ))

RadialField(radial::FieldAmplitude{C, 1}, ::Count{D}) where {C<:RealOrComplex, D} = 
RadialField{C, D}(radial)


struct ModularField{C<:RealOrComplex, D, F<:Function, P<:NamedSpanParamTuple
                    } <: FieldAmplitude{C, D}
    core::TypedCarteFunc{C, D, ParamFreeFunc{F}}
    param::P

    function ModularField(core::TypedCarteFunc{C, D, ParamFreeFunc{F}}, 
                          params::P=NamedTuple()) where 
                         {C<:RealOrComplex, D, F<:Function, P<:NamedSpanParamTuple}
        checkPositivity(D)
        new{C, D, F, P}(core, params)
    end
end

function ModularField(core::Function, ::Type{C}, ::Count{D}, 
                      params::NamedSpanParamTuple=NamedTuple()) where {C<:RealOrComplex, D}
    checkPositivity(D)
    typedCore = TypedCarteFunc(ParamFreeFunc(core), C, Count(D))
    ModularField(typedCore, params)
end

getOutputType(::Type{<:ModularField{C}}) where {C<:RealOrComplex} = C

# If `param` is empty, `.core` should not take `param` as its second argument.
const NullaryField{C<:RealOrComplex, D, F<:Function} = ModularField{C, D, F, @NamedTuple{}}

needFieldAmpEvalCache(::NullaryField) = false

function evalFieldAmplitudeCore(f::NullaryField, input)
    f.core(formatInput(f, input))
end

function unpackFieldFunc(f::NullaryField{C, D}) where {C<:RealOrComplex, D}
    TypedCarteFunc(InputConverter(f.core.f.f), C, Count(D)), initializeSpanParamSet()
end


needFieldAmpEvalCache(::ModularField) = true

function evalFieldAmplitudeCore(f::ModularField, input, cache!Self::ParamDataCache)
    paramVals = cacheParam!(cache!Self, f.param)
    f.core(formatInput(f, input), paramVals)
end

function unpackFieldFunc(f::F) where {C<:RealOrComplex, D, F<:ModularField{C, D}}
    params = f.param
    paramMapper, paramSet = genParamMapper(params)
    TypedCarteFunc(ContextParamFunc(f.core.f.f, paramMapper), C, Count(D)), paramSet
end


function computeGaussFunc(input::Tuple{Real}, params::@NamedTuple{xpn::T}) where {T<:Real}
    x, = input
    exp(-params.xpn * x * x)
end

const GaussFunc{T<:Real, P<:UnitParam{T}} = 
      ModularField{T, 1, typeof(computeGaussFunc), @NamedTuple{xpn::P}}

const ComputeGaussFunc = typeof(computeGaussFunc)

const GaussFieldCore{F<:NamedParamMapper} = 
      ContextParamFunc{ParamFreeFunc{ComputeGaussFunc}, ItsType, F}

const GaussFieldFunc{T<:Real, F<:NamedParamMapper, S<:SpanSetFilter} = 
      FieldParamFunc{T, 1, T, GaussFieldCore{F}, S}

function GaussFunc(xpn::UnitOrVal{T}) where {T<:Real}
    core = TypedCarteFunc(ParamFreeFunc(computeGaussFunc), T, Count(1))
    ModularField(core, (xpn=UnitParamEncoder(T, :xpn, 1)(xpn),))
end


struct ProductField{C<:RealOrComplex, D, B<:NonEmptyTuple{FieldAmplitude{C}}
                    } <: FieldAmplitude{C, D}
    basis::B

    function ProductField(bases::B) where 
                         {C<:RealOrComplex, B<:NonEmptyTuple{ FieldAmplitude{C} }}
        dim = mapreduce(getDimension, +, bases)
        checkPositivity(dim)
        new{C, dim, B}(bases)
    end
end

ProductField(basis::Tuple{FieldAmplitude}) = first(basis)

getOutputType(::Type{<:ProductField{C}}) where {C<:RealOrComplex} = C

function evalFieldAmplitude(f::ProductField{C}, input; 
                            cache!Self::ParamDataCache=initializeParamDataCache(), 
                            ) where {C<:RealOrComplex}
    idx = firstindex(input)
    mapreduce(StableMul(C), f.basis) do basis
        iStart = idx
        idx += getDimension(basis)
        evalFieldAmplitude(basis, input[iStart:idx-1]; cache!Self)
    end
end

function unpackFieldFunc(f::F) where {C<:RealOrComplex, D, F<:ProductField{C, D}}
    paramSet = initializeSpanParamSet()

    idx = 1
    basisCores = map(f.basis) do basis
        basisDim = getDimension(basis)
        getSubIdx = ViewOneToRange(idx, Count{basisDim}())
        idx += basisDim
        basisCore = unpackFunc!(basis, paramSet, Identifier(nothing))
        ParamPipeline((InputConverter(getSubIdx), basisCore))
    end

    TypedCarteFunc(ParamCombiner(StableMul(C), basisCores), C, Count(D)), paramSet
end


const AxialProduct{C<:RealOrComplex, D, B<:NTuple{D, FieldAmplitude{C, 1}}} = 
      ProductField{C, D, B}

AxialProduct(bases::NonEmptyTuple{FieldAmplitude{C, 1}}) where {C<:RealOrComplex} = 
ProductField(bases)

function AxialProduct(basis::FieldAmplitude{C, 1}, ::Count{D}) where {C<:RealOrComplex, D}
    checkPositivity(D)
    ProductField(ntuple( _->basis, Val(D) ))
end


struct CoupledField{C<:RealOrComplex, D, L<:FieldAmplitude{C, D}, R<:FieldAmplitude{C, D}, 
                    F<:Function} <: FieldAmplitude{C, D}
    pair::Tuple{L, R}
    coupler::ParamFreeFunc{StableBinary{C, F}}

    function CoupledField(pair::Tuple{L, R}, coupler::Function) where 
                         {C<:RealOrComplex, D, L<:FieldAmplitude{C, D}, 
                          R<:FieldAmplitude{C, D}}
        checkPositivity(D)
        coupler = ParamFreeFunc(StableBinary(coupler, C))
        new{C, D, L, R, typeof(coupler.f.f)}(pair, coupler)
    end
end

getOutputType(::Type{<:CoupledField{C}}) where {C<:RealOrComplex} = C

function evalFieldAmplitude(f::CoupledField, input; 
                            cache!Self::ParamDataCache=initializeParamDataCache())
    map(f.pair) do basis
        evalFieldAmplitude(basis, input; cache!Self)
    end |> Base.Splat(f.coupler)
end

function unpackFieldFunc(f::F) where {D, C<:RealOrComplex, F<:CoupledField{C, D}}
    fL, fR = f.pair
    paramSet = initializeSpanParamSet()
    fCoreL = unpackFunc!(fL, paramSet, Identifier(nothing))
    fCoreR = unpackFunc!(fR, paramSet, Identifier(nothing))
    TypedCarteFunc(ParamCombiner(f.coupler, (fCoreL, fCoreR)), C, Count(D)), paramSet
end


const CartAngMomentum{C<:RealOrComplex, D} = NullaryField{C, D, CartSHarmonics{D}}

const CartAngMomentumFunc{T<:Real, D, C<:RealOrComplex{T}} =
      NullaryFieldFunc{T, D, C, InputConverter{ CartSHarmonics{D} }}

const PolyRadialFunc{C<:RealOrComplex, D, F<:FieldAmplitude{C, 1}} = 
      CoupledField{C, D, RadialField{C, D, F}, CartAngMomentum{C, D}, typeof(*)}

function PolyRadialFunc(radial::FieldAmplitude{C, 1}, 
                        angular::NonEmptyTuple{Int, D}) where {C<:RealOrComplex, D}
    DimType = Count(D + 1)
    radialCore = RadialField(radial, DimType)
    polyCore = TypedCarteFunc((ParamFreeFunc∘CartSHarmonics)(angular), C, DimType)
    CoupledField((radialCore, ModularField(polyCore)), StableMul(C))
end

const PolyGaussFunc{T<:Real, D, F<:GaussFunc{T}} = PolyRadialFunc{T, D, F}

const PolyRadialFieldCore{T<:Real, D, C<:RealOrComplex{T}, F<:AbstractParamFunc, 
                          S<:SpanSetFilter} = 
      ParamCombiner{ParamFreeFunc{StableMul{C}}, 
                    Tuple{ RadialFieldFunc{T, D, C, F, S}, CartAngMomentumFunc{T, D, C} }}

const PolyGaussFieldCore{T<:Real, D, F<:NamedParamMapper, S<:SpanSetFilter} = 
      PolyRadialFieldCore{T, D, T, GaussFieldFunc{T, F, S}, S}

const PolyRadialFieldFunc{T<:Real, D, C<:RealOrComplex{T}, F<:PolyRadialFieldCore{T, D, C}, 
                          S<:SpanSetFilter} = 
      FieldParamFunc{T, D, C, F, S}

const PolyGaussFieldFunc{T<:Real, D, F<:PolyGaussFieldCore{T, D}, S<:SpanSetFilter} = 
      PolyRadialFieldFunc{T, D, T, F, S}


struct ShiftedField{T, D, C<:RealOrComplex{T}, F<:FieldAmplitude{C, D}, 
                    R<:NTuple{ D, UnitParam{T} }} <: FieldAmplitude{C, D}
    center::R
    core::F

    function ShiftedField(center::NTuple{D, UnitOrVal{T}}, core::FieldAmplitude{C, D}
                          ) where {T<:Real, C<:RealOrComplex{T}, D}
        encoder = UnitParamEncoder(T, :cen, 1)
        centerParams = encoder.(center)
        new{T, D, C, typeof(core), typeof(centerParams)}(centerParams, core)
    end
end

getOutputType(::Type{<:ShiftedField{T, D, C}}) where {T, D, C<:RealOrComplex{T}} = C

needFieldAmpEvalCache(::ShiftedField) = true

function evalFieldAmplitudeCore(f::ShiftedField{T, D}, input, 
                                cache!Self::ParamDataCache) where {T<:Real, D}
    centerCoord = cacheParam!(cache!Self, f.center)
    shiftedCoord = StableTupleSub(T, Count(D))(formatInput(f, input), centerCoord)
    f.core(shiftedCoord)
end

function unpackFieldFunc(f::F) where {T, C<:RealOrComplex{T}, D, F<:ShiftedField{C, D}}
    fInner, paramSet = unpackFieldFunc(f.core)
    fCore = fInner.f.f
    mapper, _ = genParamMapper(f.center, paramSet!Self=paramSet)
    shiftCore = StableTupleSub(T, Count(D))
    shifter = ContextParamFunc(shiftCore, CartesianFormatter(T, Count(D)), mapper)
    TypedCarteFunc(ParamPipeline((shifter, fCore)), T, Count(D)), paramSet
end

# const FieldCenterShifter{T<:Real, D, M<:ChainMapper{ <:NTuple{D, Function} }} = 
const FieldCenterShifter{T<:Real, D, M<:ChainMapper{ <:NTuple{D, Function} }} = 
      ContextParamFunc{StableTupleSub{NTuple{D, T}}, CartesianFormatter{D, NTuple{D, T}}, M}

const ShiftedFieldFuncCore{T<:Real, D, F<:AbstractParamFunc, R<:FieldCenterShifter{T, D}} = 
      ParamPipeline{Tuple{R, F}}

const ShiftedFieldFunc{T<:Real, D, C<:RealOrComplex{T}, F<:ShiftedFieldFuncCore{T, D}, 
                       S<:SpanSetFilter} = 
      FieldParamFunc{T, D, C, F, S}

const ShiftedPolyGaussField{T<:Real, D, F<:PolyGaussFunc{T, D}, 
                            R<:NTuple{ D, UnitParam{T} }} = 
      ShiftedField{T, D, T, F, R}

const StashedShiftedFieldFunc{T<:Real, D, C<:RealOrComplex{T}, F<:AbstractParamFunc, 
                              R<:FieldCenterShifter{T, D}, V<:OptSpanValueSet} = 
      StashedField{C, D, ShiftedFieldFuncCore{T, D, F, R}, V}

const FloatingPolyGaussField{T<:Real, D, F<:PolyGaussFieldCore{T, D}, 
                             R<:FieldCenterShifter{T, D}, V<:OptSpanValueSet} = 
      StashedShiftedFieldFunc{T, D, T, F, R, V}



#= Additional Method =#
function strictTypeJoin(::Type{EncodedField{CL, D, FL, EL}}, 
                        ::Type{EncodedField{CR, D, FR, ER}}) where 
                       {D, CL<:RealOrComplex, FL<:Function, EL<:Function, 
                           CR<:RealOrComplex, FR<:Function, ER<:Function}
    p = (;C=strictTypeJoin(CL, CR), D, F=strictTypeJoin(FL, FR), E=strictTypeJoin(EL, ER))
    genParametricType(EncodedField, p)
end

function strictTypeJoin(::Type{ModularField{CL, D, FL, PL}}, 
                        ::Type{ModularField{CR, D, FR, PR}}) where 
                       {D, CL<:RealOrComplex, FL<:Function, PL<:NamedSpanParamTuple, 
                           CR<:RealOrComplex, FR<:Function, PR<:NamedSpanParamTuple}
    p = (;C=strictTypeJoin(CL, CR), D, F=strictTypeJoin(FL, FR), P=strictTypeJoin(PL, PR))
    genParametricType(ModularField, p)
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

function strictTypeJoin(::Type{ShiftedField{TL, D, CL, FL, RL}}, 
                        ::Type{ShiftedField{TR, D, CR, FR, RR}}) where 
                       {D, TL<:Real, CL<:RealOrComplex{TL}, FL<:FieldAmplitude{CL, D}, 
                           RL<:NTuple{ D, UnitParam{TL} }, 
                           TR<:Real, CR<:RealOrComplex{TL}, FR<:FieldAmplitude{CL, D}, 
                           RR<:NTuple{ D, UnitParam{TL} }}
    p = (;T=strictTypeJoin(TL, TR), D, C=strictTypeJoin(CL, CR), F=strictTypeJoin(FL, FR), 
          R=strictTypeJoin(RL, RR))
    typeintersect(genParametricType(ShiftedField, p), ShiftedField)
end