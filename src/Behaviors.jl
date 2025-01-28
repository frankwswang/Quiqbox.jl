(::SelectTrait{ParameterizationStyle})(::F) where {F<:Function} = 
GenericFunction()

(::SelectTrait{ParameterizationStyle})(::ParamBoxFunction{T}) where {T} = 
TypedParamFunc{T}()


# formatInput
(::SelectTrait{InputStyle})(::Type{<:SphericalHarmonics{D}}) where {D} = 
TupleInput{D}()

(::SelectTrait{InputStyle})(::Type{<:EvalFieldAmp{<:Any, D}}) where {D} = 
TupleInput{D}()

(::SelectTrait{InputStyle})(::Type{<:EvalFieldAmp{<:Any, 0}}) = 
ScalarInput()

formatInput(::ScalarInput, x::Any) = itself(x)

formatInput(::VectorInput, x::AbstractArray) = vec(x)
formatInput(::VectorInput, x::Tuple) = collect(x)

formatInput(::TupleInput{N}, x::Tuple{Vararg{Any, N}}) where {N} = itself(x)
formatInput(::TupleInput{N}, x::AbstractArray) where {N} = 
formatInput(TupleInput{N}(), Tuple(x))