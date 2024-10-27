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

struct EvalPolyGaussFunc{F<:PointerFunc{<:ComputePGFunc}} <: EvalFieldAmp{1, PolyGaussFunc}
    f::F
end

function (f::ComputePGFunc{T})(r::Real, xpnVal::T) where {T}
    i = f.degree
    factor = ifelse(f.normalize, polyGaussFuncNormFactor(xpnVal, i), one(T))
    exp(-xpnVal * r * r) * r^i * factor
end

function unpackParamFunc!(f::PolyGaussFunc{T, <:ElementalParam{T}}, 
                          paramSet::PBoxCollection) where {T}
    evalFuncCore = ComputePGFunc{T}(f.degree, f.normalize)
    parIds = (locateParam!(paramSet, f.exponent),)
    evalFunc = PointerFunc(evalFuncCore, parIds, objectid(paramSet))
    EvalPolyGaussFunc(evalFunc), paramSet
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

function unpackParamFunc!(f::ContractedSum{<:Any, D}, paramSet::PBoxCollection) where {D}
    pSetId = objectid(paramSet)
    innerEvalFuncs = map(Fix2(unpackFunc!, paramSet), f.basis) .|> first
    evalFunc = mapreduce(JoinParallel(+), innerEvalFuncs, f.coeff) do ief, coeff
        parIds = (locateParam!(paramSet, coeff),)
        JoinParallel(ief, PointerFunc(OnlyParam(itself), parIds, pSetId), *)
    end
    EvalContractedSum{D}(evalFunc), paramSet
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

struct EvalOriginShifter{D, F<:InsertInward} <: EvalFieldAmp{D, OriginShifter}
    f::F

    EvalOriginShifter{D}(f::F) where {D, F} = new{D, F}(f)
end

function unpackParamFunc!(f::OriginShifter{T, D}, paramSet::PBoxCollection) where {T, D}
    pSetId = objectid(paramSet)
    parIds = locateParam!(paramSet, f.shift)
    outerEvalFunc = unpackFunc!(f.func, paramSet) |> first
    encoder = (inputOrigin, pVal::Vararg{T, D}) -> (inputOrigin .- pVal)
    evalFunc = InsertInward(outerEvalFunc, PointerFunc(encoder, parIds, pSetId))
    EvalOriginShifter{D}(evalFunc), paramSet
end


struct AxialProduct{T, D, B<:NTuple{D, FieldAmplitude{T, 1}}} <: FieldAmplitude{T, D}
    component::B

    AxialProduct(component::B) where {T, B<:NonEmptyTuple{FieldAmplitude{T, 1}}} = 
    new{T, length(component), B}(component)
end

AxialProduct((b,)::Tuple{FieldAmplitude{<:Any, 1}}) = itself(b)

AxialProduct(compos::AbstractVector{<:FieldAmplitude{T, 1}}) where {T} = 
AxialProduct( Tuple(compos) )

struct EvalAxialProduct{D, F<:JoinParallel} <: EvalFieldAmp{D, AxialProduct}
    f::F

    EvalAxialProduct{D}(f::F) where {D, F} = new{D, F}(f)
end

function unpackParamFunc!(f::AxialProduct{<:Any, D}, paramSet::PBoxCollection) where {D}
    evalFuncComps = map(Fix2(unpackFunc!, paramSet), f.component) .|> first
    evalFunc = mapreduce(JoinParallel(*), evalFuncComps, 1:D) do efc, idx
        InsertInward(efc, OnlyInput(input->getindex(input, idx)))
    end
    EvalAxialProduct{D}(evalFunc), paramSet
end


struct CartGaussOrb{T, D, L, B<:NTuple{D, PolyGaussFunc{T}}} <: FieldAmplitude{T, D}
    product::AxialProduct{T, D, B}
    angular::CartSHarmonics{D, L}

    function CartGaussOrb(product::AxialProduct{T, D, B}) where 
                         {T, D, B<:NTuple{D, PolyGaussFunc{T}}}
        angular = CartSHarmonics(getfield.(product.component, :degree))
        new{T, D, azimuthalNumOf(angular), B}(product, angular)
    end
end

struct EvalCartGaussOrb{D, F<:EvalAxialProduct} <: EvalFieldAmp{D, CartGaussOrb}
    f::F

    EvalCartGaussOrb{D}(f::F) where {D, F} = new{D, F}(f)
end

function unpackParamFunc!(f::CartGaussOrb{<:Any, D}, paramSet::PBoxCollection) where {D}
    evalFunc, pSet = unpackParamFunc!(f.product, paramSet)
    EvalCartGaussOrb{D}(evalFunc), pSet
end