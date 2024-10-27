(::SelectTrait{ParameterStyle})(::F) where {F<:Function} = 
ifelse( hasmethod(unpackParamFunc!, Tuple{F, PBoxCollection}), 
        IsParamFunc(), NotParamFunc() )

# formatInput
(::SelectTrait{InputStyle})(::Type{<:SphericalHarmonics{D}}) where {D} = 
TupleInput{D}()

(::SelectTrait{InputStyle})(::Type{<:EvalFieldAmp{D}}) where {D} = 
TupleInput{D}()

formatInput(::VectorInput, x::AbstractArray) = vec(x)
formatInput(::VectorInput, x::Tuple) = collect(x)

formatInput(::TupleInput{1}, x::Any) = itself(x)
formatInput(::TupleInput{1}, x::Tuple{Any}) = first(x)
formatInput(::TupleInput{N}, x::Tuple{Vararg{Any, N}}) where {N} = itself(x)
formatInput(::TupleInput{N}, x::AbstractArray) where {N} = 
formatInput(TupleInput{N}(), Tuple(x))