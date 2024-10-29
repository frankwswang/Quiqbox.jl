export PolyGaussFunc, ContractedSum, OriginShifter, AxialFuncProd, PolyGaussProd

(f::FieldAmplitude)(x) = evalFunc(f, x)

abstract type EvalFieldAmp{D, F} <: Evaluator{F} end

evalFunc(f::EvalFieldAmp, input, param) = 
f.f(formatInput(SelectTrait{InputStyle}()(f), input), param)


struct PolyGaussFunc{T<:Real, P<:ElementalParam{T}} <: FieldAmplitude{T, 1}
    exponent::P
    degree::Int
    normalize::Bool

    function PolyGaussFunc(exponent::P, degree::Int, normalize::Bool) where 
                          {T, P<:ElementalParam{T}}
        degree < 0 && throw(AssertionError("`degree` should be non-negative."))
        new{T, P}(exponent, degree, normalize)
    end
end

PolyGaussFunc(exponent::ElementalParam{<:Real}, degree::Int=0; normalize::Bool=true) = 
PolyGaussFunc(exponent, degree, normalize)

struct ComputePGFunc{T} <: CompositeFunction
    degree::Int
    normalize::Bool
end

function (f::ComputePGFunc{T})(r::Real, xpnVal::T) where {T}
    i = f.degree
    factor = ifelse(f.normalize, polyGaussFuncNormFactor(xpnVal, i), one(T))
    exp(-xpnVal * r * r) * r^i * factor
end

struct EvalPolyGaussFunc{F<:PointerFunc{<:ComputePGFunc}} <: EvalFieldAmp{1, PolyGaussFunc}
    f::F
end

function unpackParamFunc!(f::PolyGaussFunc{T, <:ElementalParam{T}}, 
                          paramSet::PBoxAbtArray) where {T}
    fEvalCore = ComputePGFunc{T}(f.degree, f.normalize)
    parIds = (locateParam!(paramSet, f.exponent),)
    fEval = PointerFunc(fEvalCore, parIds, objectid(paramSet))
    EvalPolyGaussFunc(fEval), paramSet
end


struct ContractedSum{T, D, F<:FieldAmplitude{T, D}, 
                     P<:ElementalParam{<:RealOrComplex{T}}} <: FieldAmplitude{T, D}
    basis::Memory{F}
    coeff::Memory{P}

    function ContractedSum(basis::AbstractVector{F}, coeff::AbstractVector{P}) where {T, D, 
                           F<:FieldAmplitude{T, D}, P<:ElementalParam{<:RealOrComplex{T}}}
        if checkEmptiness(basis, :basis) != length(coeff)
            throw(AssertionError("`basis` and `coeff` must have the same length."))
        end
        new{T, D, F, P}(getMemory(basis), getMemory(coeff))
    end
end

struct EvalContractedSum{D, F<:JoinParallel} <: EvalFieldAmp{D, ContractedSum}
    f::F

    EvalContractedSum{D}(f::F) where {D, F} = new{D, F}(f)
end

function unpackParamFunc!(f::ContractedSum{<:Any, D}, paramSet::PBoxAbtArray) where {D}
    pSetId = objectid(paramSet)
    fEvalInners = map(Fix2(unpackFunc!, paramSet), f.basis) .|> first
    fEval = mapreduce(JoinParallel(+), fEvalInners, f.coeff) do ief, coeff
        parIds = (locateParam!(paramSet, coeff),)
        JoinParallel(ief, PointerFunc(OnlyParam(itself), parIds, pSetId), *)
    end
    EvalContractedSum{D}(fEval), paramSet
end


struct OriginShifter{T, D, F<:FieldAmplitude{T, D}, 
                     L<:NTuple{D, ElementalParam{T}}} <: FieldAmplitude{T, D}
    func::F
    shift::L

    OriginShifter(func::F, shift::L) where {T, D, F<:FieldAmplitude{T, D}, 
                                            L<:NonEmptyTuple{ElementalParam{T}}} = 
    new{T, D, F, L}(func, shift)
end

function OriginShifter(func::F, shift::NonEmptyTuple{Real}=ntuple(_->T(0), Val(D))) where 
                      {T, D, F<:FieldAmplitude{T, D}}
    length(shift) != D && throw(AssertionError("The length of `shift` must match `D=$D`."))
    shiftPars = CellParam.(T.(shift), :shift)
    setScreenLevel!.(shiftPars, 2)
    OriginShifter(func, shiftPars)
end

struct ShiftByArg{T<:Real, D} <: FieldlessFunction end

(::ShiftByArg{T, D})(input, arg::Vararg{T, D}) where {T, D} = (input .- arg)

struct EvalOriginShifter{D, F<:InsertInward} <: EvalFieldAmp{D, OriginShifter}
    f::F

    EvalOriginShifter{D}(f::F) where {D, F} = new{D, F}(f)
end

function unpackParamFunc!(f::OriginShifter{T, D}, paramSet::PBoxAbtArray) where {T, D}
    pSetId = objectid(paramSet)
    parIds = locateParam!(paramSet, f.shift)
    fEvalOuter = unpackFunc!(f.func, paramSet) |> first
    fEval = InsertInward(fEvalOuter, PointerFunc(ShiftByArg{T, D}(), parIds, pSetId))
    EvalOriginShifter{D}(fEval), paramSet
end


struct AxialFuncProd{T, D, B<:NTuple{D, FieldAmplitude{T, 1}}} <: FieldAmplitude{T, D}
    component::B

    AxialFuncProd(component::B) where {T, B<:NonEmptyTuple{FieldAmplitude{T, 1}}} = 
    new{T, length(component), B}(component)
end

AxialFuncProd((b,)::Tuple{FieldAmplitude{<:Any, 1}}) = itself(b)

AxialFuncProd(compos::AbstractVector{<:FieldAmplitude{T, 1}}) where {T} = 
AxialFuncProd( Tuple(compos) )

function AxialFuncProd(b::FieldAmplitude{<:Any, 1}, dim::Int)
    dim < 1 && throw(AssertionError("`dim` must be a positive integer."))
    (AxialFuncProd∘Tuple∘fill)(b, dim)
end

struct EvalAxialFuncProd{D, F<:JoinParallel} <: EvalFieldAmp{D, AxialFuncProd}
    f::F

    EvalAxialFuncProd{D}(f::F) where {D, F} = new{D, F}(f)
end

function unpackParamFunc!(f::AxialFuncProd{<:Any, D}, paramSet::PBoxAbtArray) where {D}
    fEvalComps = map(Fix2(unpackFunc!, paramSet), f.component) .|> first
    fEval = mapreduce(JoinParallel(*), fEvalComps, 1:D) do efc, idx
        InsertInward(efc, (OnlyInput∘Base.Fix2)(getindex, idx))
    end
    EvalAxialFuncProd{D}(fEval), paramSet
end


struct PolyGaussProd{T, D, L, B<:NTuple{D, PolyGaussFunc{T}}} <: FieldAmplitude{T, D}
    product::AxialFuncProd{T, D, B}
    angular::CartSHarmonics{D, L}

    function PolyGaussProd(product::AxialFuncProd{T, D, B}) where 
                          {T, D, B<:NTuple{D, PolyGaussFunc{T}}}
        angular = CartSHarmonics(getfield.(product.component, :degree))
        new{T, D, azimuthalNumOf(angular), B}(product, angular)
    end
end

function PolyGaussProd(pgfs::NonEmptyTuple{PolyGaussFunc{T}}) where {T}
    length(pgfs) < 2 && throw(AssertionError("`pgfs` must contain at least two elements."))
    (PolyGaussProd∘AxialFuncProd)(pgfs)
end

PolyGaussProd(xpns::NonEmptyTuple{ElementalParam{T}, D}, 
              ijk::NonEmptyTuple{Int, D}; normalize::Bool=true) where {T, D} = 
PolyGaussFunc.(xpns, ijk; normalize) |> PolyGaussProd

PolyGaussProd(xpn::ElementalParam{T}, 
              ijk::NonEmptyTuple{Int, D}; normalize::Bool=true) where {T, D} = 
PolyGaussProd((Tuple∘fill)(xpn, D+1), ijk; normalize)

struct EvalPolyGaussProd{D, F<:EvalAxialFuncProd} <: EvalFieldAmp{D, PolyGaussProd}
    f::F

    EvalPolyGaussProd{D}(f::F) where {D, F} = new{D, F}(f)
end

function unpackParamFunc!(f::PolyGaussProd{<:Any, D}, paramSet::PBoxAbtArray) where {D}
    fEval, pSet = unpackParamFunc!(f.product, paramSet)
    EvalPolyGaussProd{D}(fEval), pSet
end