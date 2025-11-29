export PrimitiveOrb, CompositeOrb, genGaussTypeOrb, genGaussTypeOrbSeq

const CONST_typeStrOfRealOrComplex = shortUnionAllString(RealOrComplex)

(::SelectTrait{InputStyle})(::OrbitalBasis{C, D}) where {C<:RealOrComplex, D} = 
CartesianInput{D}()


(f::OrbitalBasis)(input) = evalOrbital(f, formatInput(f, input))


mutable struct PrimitiveOrb{T<:Real, D, C<:RealOrComplex{T}, F<:ShiftedField{T, D, C}
                            } <: OrbitalBasis{C, D, F}
    const field::F
    @atomic renormalize::Bool
end

const PrimGTO{T<:Real, D, F<:ShiftedPolyGaussField{T, D}} = PrimitiveOrb{T, D, T, F}

function PrimitiveOrb(center::NTuple{D, UnitOrVal{<:Real}}, body::FieldAmplitude{C, D}; 
                      renormalize::Bool=false) where {T<:Real, C<:RealOrComplex{T}, D}
    PrimitiveOrb(ShiftedField(center, body), renormalize)
end

PrimitiveOrb(o::PrimitiveOrb; renormalize::Bool=o.renormalize) = 
PrimitiveOrb(o.field, renormalize)

getOutputType(::Type{<:PrimitiveOrb{T, D, C}}) where {T<:Real, D, C<:RealOrComplex{T}} = C


function evalOrbital(orb::PrimitiveOrb{T, D, C}, input; 
                     cache!Self::OptParamDataCache=initializeParamDataCache()) where 
                    {T<:Real, D, C<:RealOrComplex{T}}
    coreValue = evalFieldAmplitude(orb.field, formatInput(orb, input); cache!Self)
    StableMul(C)(coreValue, getNormFactor(orb))
end


function getPrimitiveOrbType(orbs::AbstractArray{B}) where 
                            {T<:Real, D, B<:PrimitiveOrb{T, D}}
    if isempty(orbs) || isconcretetype(B)
        B
    else
        C = Union{}
        fieldType = Union{}

        for orb in orbs
            C = strictTypeJoin(C, getOutputType(orb))
            fieldType = strictTypeJoin(fieldType, typeof(orb.field))
        end

        pseudoType = genParametricType(PrimitiveOrb, (;T, D, C, F=fieldType))
        typeintersect(PrimitiveOrb, pseudoType)
    end
end

mutable struct CompositeOrb{T<:Real, D, C<:RealOrComplex{T}, B<:PrimitiveOrb{T, D}, 
                            W<:GridParam{C, 1}} <: OrbitalBasis{C, D, B}
    const basis::Memory{B}
    const weight::W
    @atomic renormalize::Bool

    function CompositeOrb(basis::Memory{<:PrimitiveOrb{T, D}}, weight::W, renormalize::Bool
                          ) where {T<:Real, C<:RealOrComplex{T}, D, W<:GridParam{C, 1}}
        nPrim = (first∘getOutputSize)(weight)
        checkLengthCore(checkEmptiness(basis, :basis), :basis, nPrim, 
                        "the output length of `weight`")
        formattedBasis = genMemory(basis)
        basisType = getPrimitiveOrbType(formattedBasis)
        new{T, D, C, basisType, W}(formattedBasis, weight, renormalize)
    end
end

const CompGTO{T<:Real, D, B<:PrimGTO{T, D}, W<:GridParam{T, 1}} = 
      CompositeOrb{T, D, T, B, W}

const ComposedOrb{T<:Real, D, C<:RealOrComplex{T}} = 
      Union{PrimitiveOrb{T, D, C}, CompositeOrb{T, D, C}}

function CompositeOrb(basis::AbstractVector{<:PrimitiveOrb{T, D}}, 
                      weight::GridParam{T, 1}; renormalize::Bool=false) where {T<:Real, D}
    CompositeOrb(extractMemory(basis), weight, renormalize)
end

function CompositeOrb(basis::AbstractVector{<:OrbitalBasis{<:RealOrComplex{T}, D}}, 
                      weight::GridParam{C, 1}; renormalize::Bool=false) where 
                     {T<:Real, C<:RealOrComplex{T}, D}
    formattedBasis = genMemory(basis)
    if !(eltype(formattedBasis) <: PrimitiveOrb)
        weight = mapreduce(vcat, enumerate(formattedBasis)) do (idx, basis)
            getEffectiveWeight(basis, weight, idx)
        end |> Base.Fix2(genHeapParam, :weight)
        formattedBasis = mapfoldl(splitOrb, vcat, basis)
    end
    CompositeOrb(formattedBasis, weight, renormalize)
end

splitOrb(o::PrimitiveOrb) = genMemory(o)

splitOrb(o::CompositeOrb) = copy(o.basis)

viewOrb(o::CompositeOrb, onrToIdx::Int) = o.basis[begin+onrToIdx-1]

function viewOrb(o::PrimitiveOrb, onrToIdx::Int)
    onrToIdx == 1 ? itself(o) : throw(BoundsError(o, onrToIdx))
end

function getEffectiveWeight(::PrimitiveOrb{T}, weight::GridParam{C, 1}, idx::Int) where 
                           {T<:Real, C<:RealOrComplex{T}}
    indexParam(weight, idx) |> fill
end

function getEffectiveWeight(o::CompositeOrb{T, D, C}, weight::GridParam{C, 1}, 
                            idx::Int) where {T<:Real, D, C<:RealOrComplex{T}}
    o.renormalize && throw(AssertionError("Merging the weight from a renormalized "*
                           "`CompositeOrb` with another value is prohibited."))
    len = first(getOutputSize(o.weight))
    outerWeight = indexParam(weight, idx)
    map(1:len) do i
        innerWight = indexParam(o.weight, i)
        genCellParam(StableMul(C), (innerWight, outerWeight), :w)
    end
end

function CompositeOrb(basis::AbstractVector{<:OrbitalBasis{<:RealOrComplex{T}, D}}, 
                      weight::AbstractVector{C}; renormalize::Bool=false) where 
                     {T<:Real, C<:RealOrComplex{T}, D}
    weightParam = GridParamEncoder(C, :wBlock, 1)(weight)
    CompositeOrb(basis, weightParam; renormalize)
end

function CompositeOrb(basis::AbstractVector{<:OrbitalBasis{<:RealOrComplex{T}, D}}, 
                      weight::UnitOrValVec{C}; renormalize::Bool=false) where 
                     {T<:Real, C<:RealOrComplex{T}, D}
    encoder = UnitParamEncoder(C, :w, 1)
    weightParams = genHeapParam(map(encoder, weight), :wBlock)
    CompositeOrb(basis, weightParams; renormalize)
end

CompositeOrb(o::CompositeOrb; renormalize::Bool=o.renormalize) = 
CompositeOrb(o.basis, o.weight, renormalize)

getOutputType(::Type{<:CompositeOrb{T, D, C}}) where {T<:Real, D, C<:RealOrComplex{T}} = C


function evalOrbital(orb::CompositeOrb{T, D, C}, input; 
                     cache!Self::OptParamDataCache=initializeParamDataCache()) where 
                    {T<:Real, D, C<:RealOrComplex{T}}
    weightVal = obtainCore!(cache!Self, orb.weight)

    bodyVal = zero(C)
    multiplier = TypedBinary(TypedReturn(*, C), RealOrComplex{T}, C)

    for (basis, w) in zip(orb.basis, weightVal)
        res = evalOrbital(basis, input; cache!Self)
        bodyVal += multiplier(res, w)
    end

    StableMul(C)(convert(C, bodyVal), getNormFactor(orb))
end


function isRenormalized(orb::ComposedOrb)
    orb.renormalize
end


function enforceRenormalize!(b::ComposedOrb)
    @atomic b.renormalize = true
end


function disableRenormalize!(b::ComposedOrb)
    @atomic b.renormalize = false
end


function getNormFactor(orb::ComposedOrb{T, D, C}) where {T<:Real, D, C<:RealOrComplex{T}}
    if isRenormalized(orb)
        constructor = getfield(Quiqbox, nameof(orb))
        orbInner = constructor(orb, renormalize=false)
        overlapVal = computeOrbLayoutIntegral(genOverlapSampler(), (orbInner, orbInner))
        convert(C, absSqrtInv(overlapVal))
    else
        one(C)
    end
end

"""

    genGaussTypeOrb(center::NTuple{D, T}, xpn::T, ijk::NTuple{D, Int}=ntuple(_->0, Val(D)); 
                    renormalize::Bool=false) where {T<:Real, D} -> 
    $PrimitiveOrb{T, D, T}

Generate a `D`-dimensional primitive Gaussian-type orbital (GTO) located at `center` with 
exponent coefficients specified by `xpn` and Cartesian angular momentum specified by `ijk`. 
`renormalize` determines whether the generated orbital should be renormalized.
"""
function genGaussTypeOrb(center::NonEmptyTuple{UnitOrVal{<:Real}, D}, xpn::UnitOrVal{T}, 
                         ijk::NonEmptyTuple{Int, D}=ntuple(_->0, Val(D+1)); 
                         renormalize::Bool=false) where {T<:Real, D}
    gf = GaussFunc(xpn)
    PrimitiveOrb(center, PolyRadialFunc(gf, ijk); renormalize)
end

"""

    genGaussTypeOrb(center::NTuple{D, T}, xpns::AbstractVector{T}, cons::AbstractVector{C}, 
                    ijk::NonEmptyTuple{Int, D}=ntuple(_->0, Val(D)); 
                    innerRenormalize::Bool=false, outerRenormalize::Bool=false) where 
                   {T<:Real, C<:$CONST_typeStrOfRealOrComplex, D} -> 
    $CompositeOrb{T, D, C}

Generate a `D`-dimensional contracted Gaussian-type orbital (GTO) with Cartesian angular 
momentum specified by `ijk`, as a linear combination of con-centric primitive GTOs located 
at `center`, with exponent coefficients specified by `xpns` and contraction coefficients 
specified by `cons`. `innerRenormalize` and `outerRenormalize` determine whether the 
primitive GTOs and the contracted GTO should be renormalized, respectively.
"""
function genGaussTypeOrb(center::NonEmptyTuple{UnitOrVal{<:Real}, D}, 
                         xpns::UnitOrValVec{T}, 
                         cons::Union{UnitOrValVec{C}, GridParam{C, 1}}, 
                         ijk::NonEmptyTuple{Int, D}=ntuple(_->0, Val(D+1)); 
                         innerRenormalize::Bool=false, outerRenormalize::Bool=false) where 
                        {T<:Real, C<:RealOrComplex{T}, D}
    if cons isa GridParam
        conParam = cons
        nPrimOrbs = (first∘getOutputSize)(cons)
    else
        conParam = if cons isa AbstractVector{C}
            GridParamEncoder(C, :con, 1)(cons)
        else
            map(UnitParamEncoder(C, :con, 1), cons)
        end
        nPrimOrbs = length(cons)
    end

    checkLengthCore(checkEmptiness(xpns, :xpns), :xpns, nPrimOrbs, 
                    "the output length of `cons`")

    cenParam = map(UnitParamEncoder(T, :cen, 1), center)

    primGTOs = map(xpns) do xpn
        genGaussTypeOrb(cenParam, xpn, ijk, renormalize=innerRenormalize)
    end
    CompositeOrb(primGTOs, conParam, renormalize=outerRenormalize)
end


const OrbBasisVector{T<:Real, D} = AbstractVector{<:OrbitalBasis{<:RealOrComplex{T}, D}}

const OrbBasisLayout{T<:Real, D} = N24Tuple{OrbitalBasis{<:RealOrComplex{T}, D}}

const OrbBasisCluster{T<:Real, D} = Union{OrbBasisVector{T, D}, OrbBasisLayout{T, D}}


function getOutputType(orbitals::OrbBasisCluster{T, D}) where {T<:Real, D}
    checkEmptiness(orbitals, :orbitals)
    orbType = eltype(orbitals)
    if isconcretetype(orbType)
        getOutputType(orbType)
    else
        mapreduce(getOutputType, strictTypeJoin, orbitals)
    end::Type{<:RealOrComplex{T}}
end


struct PrimOrbPointer{D, C<:RealOrComplex} <: CustomAccessor
    inner::OneToIndex
    renormalize::Bool
end

struct CompOrbPointer{D, C<:RealOrComplex} <: CustomAccessor
    inner::MemoryPair{PrimOrbPointer{D, C}, C}
    renormalize::Bool
end

const OrbitalPointer{D, C<:RealOrComplex} = 
      Union{PrimOrbPointer{D, C}, CompOrbPointer{D, C}}

getOutputType(::OrbitalPointer{D, C}) where {D, C<:RealOrComplex} = C

strictTypeJoin(::Type{CompOrbPointer{D, C}}, ::Type{PrimOrbPointer{D, C}}) where 
              {D, C<:RealOrComplex}= 
OrbitalPointer{D, C}

strictTypeJoin(::Type{PrimOrbPointer{D, C}}, ::Type{CompOrbPointer{D, C}}) where 
              {D, C<:RealOrComplex}= 
OrbitalPointer{D, C}


const StashedFieldPointerDict{T<:Real, D} = 
      EncodedDict{StashedShiftedField{T, D}, OneToIndex, FieldMarker{:StashedField, 2}, 
                 typeof(markObj)}

struct FieldIndexBox{T, D} <: QueryBox{OneToIndex}
    direct::Dict{EgalBox{ShiftedField{T, D}}, OneToIndex}
    latent::StashedFieldPointerDict{T, D}

    function FieldIndexBox{T, D}(maxSize::Int=100) where {T, D}
        directDict = Dict{EgalBox{ShiftedField{T, D}}, OneToIndex}()
        latentDict = EncodedDict{StashedShiftedField{T, D}, OneToIndex}(
            (markObj => TypeBox(FieldMarker{:StashedField, 2})), maxSize
        )
        new{T, D}(directDict, latentDict)
    end
end

mutable struct FieldPoolConfig{T<:Real, D, B<:Boolean, P<:OptSpanParamSet, 
                               M<:OptParamDataCache} <: ConfigBox
    const field::FieldIndexBox{T, D}
    const depth::B
    const param::P
    const cache::M
    count::Int

    function FieldPoolConfig(field::FieldIndexBox{T, D}, depth::B, param::P, paramCache::M
                             ) where {T<:Real, D, B<:Boolean, P<:OptSpanParamSet, 
                                      M<:OptParamDataCache}
        fieldMarkerDict = field.latent.value
        initCount = isempty(fieldMarkerDict) ? 0 : maximum(fieldMarkerDict|>values).idx
        new{T, D, B, P, M}(field, depth, param, paramCache, initCount)
    end
end

function indexGetOrbCore!(config::FieldPoolConfig{T, D}, field::ShiftedField{T, D}
                          ) where {T, D}
    tracker = EgalBox{ShiftedField{T, D}}(field)
    ptrIdxDict = config.field.latent
    trackerDict = config.field.direct

    stashedFieldNew = missing
    idx = get(trackerDict, tracker, nothing)

    if idx === nothing
        paramSet = config.param
        fieldFunc = unpackFunc!(field, paramSet, config.depth)
        fieldCore = StashedField(fieldFunc, paramSet, config.cache)
        idx = get(ptrIdxDict, fieldCore, nothing)
        if idx === nothing
            idx = OneToIndex(config.count += 1)
            stashedFieldNew = fieldCore
            setindex!(ptrIdxDict, idx, fieldCore)
        end
        setindex!(trackerDict, idx, tracker)
    end

    if idx.idx > config.count
        throw(AssertionError("The value of `counter` is lower than the quantity of "*
                             "generated pointers."))
    end

    (idx::OneToIndex) => (stashedFieldNew::MissingOr{StashedShiftedField{T, D}})
end

const PrimOrbPtrSrcPair{T<:Real, D, C<:RealOrComplex{T}, F<:StashedShiftedField{T, D}} = 
      Pair{PrimOrbPointer{D, C}, Memory{F}}

function prepareOrbitalPointer!(::Type{C}, config::FieldPoolConfig{T, D}, 
                                orbital::PrimitiveOrb{T, D}) where 
                               {T<:Real, D, C<:RealOrComplex{T}}
    field = orbital.field
    idx, fieldRes = indexGetOrbCore!(config, field)
    pointer = PrimOrbPointer{D, C}(idx, orbital.renormalize)
    stashedFields = ismissing(fieldRes) ? genBottomMemory() : genMemory(fieldRes)
    (pointer => stashedFields)::PrimOrbPtrSrcPair{T, D, C}
end

const CompOrbPtrSrcPair{T<:Real, D, C<:RealOrComplex{T}, F<:StashedShiftedField{T, D}} = 
      Pair{CompOrbPointer{D, C}, Memory{F}}

function prepareOrbitalPointer!(::Type{C}, config::FieldPoolConfig{T, D}, 
                                orbital::CompositeOrb{T, D}) where 
                               {T<:Real, D, C<:RealOrComplex{T}}
    paramCache = config.cache
    weightValue = Memory{C}(obtainCore!(paramCache, orbital.weight))
    pairs = map(orbital.basis) do primOrb
        prepareOrbitalPointer!(C, config, primOrb)::PrimOrbPtrSrcPair{T, D, C}
    end
    weightedPtrs = MemoryPair(map(first, pairs), weightValue)
    pointer = CompOrbPointer{D, C}(weightedPtrs, orbital.renormalize)
    stashedFields = mapreduce(last, strictVerticalCat, pairs, init=genBottomMemory())
    (pointer => stashedFields)::CompOrbPtrSrcPair{T, D, C}
end

const OrbitalPtrSrcPair{T<:Real, D, C<:RealOrComplex{T}} = 
      Union{PrimOrbPtrSrcPair{T, D, C}, CompOrbPtrSrcPair{T, D, C}}

function cacheOrbitalData!(::Type{C}, 
                           orbitals::OrbBasisCluster{T, D}, directUnpack::Boolean, 
                           cache!Self::OptParamDataCache=initializeParamDataCache(); 
                           maxSize::Int=100) where {T<:Real, D, C<:RealOrComplex{T}}
    checkEmptiness(orbitals, :orbitals)
    paramSet = initializeSpanParamSet()
    idxBox = FieldIndexBox{T, D}(maxSize)
    pairs = let config=FieldPoolConfig(idxBox, directUnpack, paramSet, cache!Self)
        map(orbitals) do orb
            prepareOrbitalPointer!(C, config, orb)::OrbitalPtrSrcPair{T, D, C}
        end
    end
    stashedFields = mapreduce(last, strictVerticalCat, pairs)
    map(first, pairs) => (stashedFields::Memory{<:StashedShiftedField{T, D}})
end


struct OrbCorePointer{D, C<:RealOrComplex}
    inner::MemoryPair{OneToIndex, C}

    function OrbCorePointer(::Count{D}, inner::MemoryPair{OneToIndex, C}) where 
                           {D, C<:RealOrComplex}
        new{D, C}(inner)
    end
end

function OrbCorePointer(pointer::PrimOrbPointer{D, C}, weight::C
                        ) where {D, C<:RealOrComplex}
    innerPair = MemoryPair(genMemory(pointer.inner), genMemory(weight))
    OrbCorePointer(Count(D), innerPair)
end

function OrbCorePointer(pointer::CompOrbPointer{D, C}, weight::AbstractArray{C, 1}
                        ) where {D, C<:RealOrComplex}
    innerPair = MemoryPair(map(x->x.inner, pointer.inner.left), extractMemory(weight))
    OrbCorePointer(Count(D), innerPair)
end


const OrbPairLayoutFormat{D, C} = 
      Union{NTuple{2, OrbitalPointer{D, C}}, NTuple{ 2, OrbCorePointer{D, C} }}

const OrbQuadLayoutFormat{D, C} = 
      Union{NTuple{4, OrbitalPointer{D, C}}, NTuple{ 4, OrbCorePointer{D, C} }}

const OrbitalLayoutFormat{D, C} = 
      Union{OrbPairLayoutFormat{D, C}, OrbQuadLayoutFormat{D, C}}

const OrbitalVectorFormat{D, C} = 
      Union{Memory{<:OrbitalPointer{D, C}}, Memory{ OrbCorePointer{D, C} }}

const OrbFormatCollection{D, C<:RealOrComplex} = 
      Union{OrbitalLayoutFormat{D, C}, OrbitalVectorFormat{D, C}}


struct MultiOrbitalData{T<:Real, D, C<:RealOrComplex{T}, F<:StashedShiftedField{T, D}, 
                        P<:OrbFormatCollection{D, C}} <: QueryBox{F}
    config::Memory{F}
    format::P

    function MultiOrbitalData(orbitals::OrbBasisCluster{T, D}, 
                              directUnpack::Boolean=False(); 
                              cache!Self::OptParamDataCache=initializeParamDataCache()
                              ) where {T<:Real, D}
        outputType = getOutputType(orbitals)
        format, config = cacheOrbitalData!(outputType, orbitals, directUnpack, cache!Self)
        orbitals isa AbstractVector && (format = genMemory(format))
        new{T, D, outputType, eltype(config), typeof(format)}(config, format)
    end

    function MultiOrbitalData(data::MultiOrbitalData{T, D, C}, 
                              format::OrbFormatCollection{D, C}) where 
                             {T<:Real, D, C<:RealOrComplex{T}}
        config = data.config
        new{T, D, C, eltype(config), typeof(format)}(config, format)
    end
end

const OrbitalLayoutData{T<:Real, D, C<:RealOrComplex{T}, F<:StashedShiftedField{T, D}, 
                        P<:OrbitalLayoutFormat{D, C}} = 
      MultiOrbitalData{T, D, C, F, P}

const OrbitalVectorData{T<:Real, D, C<:RealOrComplex{T}, F<:StashedShiftedField{T, D}, 
                        P<:OrbitalVectorFormat{D, C}} = 
      MultiOrbitalData{T, D, C, F, P}

getOutputType(::MultiOrbitalData{T, D, C}) where {T<:Real, D, C<:RealOrComplex{T}} = C

const OrbBasisData{T<:Real, D} = Union{MultiOrbitalData{T, D}, OrbBasisVector{T, D}}


function get3DimPGTOrbNormFactor(xpn::T, carteAng::NTuple{3, Int}) where {T<:Real}
    for i in carteAng; checkPositivity(i, true) end
    i, j, k = carteAng
    angSum = sum(carteAng)
    xpnPart = xpn^(T(2angSum + 3)*T(0.25))
    angPart = T(PowersOfPi[:n0d75]) * (exp2∘T)(1.5angSum + 0.75) * 
              T(sqrt( factorial(i) * factorial(j) * factorial(k) / 
                      (factorial(2i) * factorial(2j) * factorial(2k)) ))
    xpnPart * angPart
end


"""

    genGaussTypeOrbSeq(center::NTuple{3, T}, content::AbstractString; 
                       innerRenormalize::Bool=false, outerRenormalize::Bool=false, 
                       unlinkCenter::Bool=false) where {T<:Real} -> 
    Vector{<:$CompositeOrb{T, 3, T}}

Generate a sequence of Gaussian-type orbitals (GTOs) located at `center` based on the
basis set information provided in `content`, which should the text of an atomic Gaussian 
basis set in the Gaussian (software) format. `innerRenormalize` and `outerRenormalize` 
determine whether the primitive GTOs and the contracted GTOs should be renormalized, 
respectively. `unlinkCenter` determines whether the center parameters of the generated 
contracted GTOs from separate subshells should be unlinked from each other.
"""
function genGaussTypeOrbSeq(center::NTuple{3, UnitOrVal{<:Real}}, content::AbstractString; 
                            innerRenormalize::Bool=false, outerRenormalize::Bool=false, 
                            unlinkCenter::Bool=false)
    Ts = map(x->(x isa UnitParam ? getOutputType(x) : typeof(x)), center)
    T = reduce(typejoin, Ts)
    if !isconcretetype(T)
        throw(AssertionError("The elemental data type of `center`: $Ts must be uniform."))
    end

    cenParams = map(UnitParamEncoder(T, :cen, 1), center)
    formattedContent = replaceSciNotation(content)
    data = map((@view formattedContent[begin : end-1]) |> IOBuffer |> readlines) do line
        advancedParse.(T, split(line))
    end
    idxScope = findall(x -> eltype(x)!=T && length(x)>2 && x[begin]!="X", data)
    R = Tuple{(ifelse(c isa Real, SimpleUnitPar{T}, typeof(c)) for c in center)...}
    F = ShiftedPolyGaussField{T, 3, SimplePolyGaussFunc{T, 3}, R}
    bfs = CompGTO{T, 3, PrimitiveOrb{T, 3, T, F}, SimpleGirdPar{T, 1}}[]

    for j in idxScope
        oInfo = data[j]
        nPGTOrb = Int(oInfo[begin + 1])
        coeffPairs = @view data[j+1 : j+nPGTOrb]
        xpns = first.(coeffPairs)
        subshellStr = first(oInfo)
        cenParamsLocal = unlinkCenter ? deepcopy(cenParams) : cenParams
        angNums = subshellStr == "SP" ? (0, 1) : (AngularSubShellDict[subshellStr],)

        for (i, angNum) in enumerate(angNums)
            for ijk in SubshellXYZs[begin+angNum]
                cons = map(xpns, coeffPairs) do xpn, segment
                    getEntry(segment, OneToIndex(1+i)) * get3DimPGTOrbNormFactor(xpn, ijk)
                end
                push!(bfs, genGaussTypeOrb(cenParamsLocal, xpns, cons, ijk; 
                                           innerRenormalize, outerRenormalize))
            end
        end
    end

    bfs
end

"""

    genGaussTypeOrbSeq(center::NTuple{3, UnitOrVal{T}}, atm::Symbol, basisKey::String; 
                       innerRenormalize::Bool=false, outerRenormalize::Bool=false, 
                       unlinkCenter::Bool=false) where {T<:Real} -> 
    Vector{<:$CompositeOrb{T, 3, T}}

Generate a sequence of Gaussian-type orbitals (GTOs) located at `center` based on the input 
atomic basis set specified by the atomic symbol `atm` and the basis-set name `basisKey`. 

* Atomic symbol options: `$AtomElementNames`.

* Basis-set name options: `$AtomicGTOrbSetNames`.

`innerRenormalize` and `outerRenormalize` determine whether the primitive GTOs and the 
contracted GTOs should be renormalized, respectively. `unlinkCenter` determines whether the 
center parameters of the generated GTOs should be unlinked from each other.
"""
function genGaussTypeOrbSeq(center::NTuple{3, UnitOrVal{<:Real}}, atm::Symbol, 
                            basisKey::String; innerRenormalize::Bool=false, 
                            outerRenormalize::Bool=false, unlinkCenter::Bool=false)
    Ts = map(x->(x isa UnitParam ? getOutputType(x) : typeof(x)), center)
    T = reduce(typejoin, Ts)
    if !isconcretetype(T)
        throw(AssertionError("The elemental data type of `center`: $Ts must be uniform."))
    end

    hasBasis = true
    basisSetFamily = get(AtomicGTOrbSetDict, basisKey, nothing)
    basisStr = if basisSetFamily === nothing
        hasBasis = false
        ""
    else
        atmCharge = get(NuclearChargeDict, atm, nothing)
        if atmCharge === nothing || atmCharge > length(basisSetFamily)
            hasBasis = false
            ""
        else
            res = getEntry(basisSetFamily, atmCharge)
            if res === nothing
                hasBasis = false
                ""
            else
                res
            end
        end
    end
    hasBasis || throw(DomainError((atm, basisKey), 
                      "Quiqbox does not have this basis-set configuration pre-stored."))
    genGaussTypeOrbSeq(center, basisStr; unlinkCenter, innerRenormalize, outerRenormalize)
end