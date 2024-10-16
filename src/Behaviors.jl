# getParams
# (::SelectTrait{ParamBoxAccess})(::Function) = WithoutParamBox()

# (::SelectTrait{ParamBoxAccess})(::SpatialAmplitude) = ContainParamBox()

# getParams(::WithoutParamBox, ::Any, ::SymOrMiss) = ParamBox[]

# function getParams(::ContainParamBox, obj, sym::SymOrMiss)
#     map( fieldnames(pf) ) do field
#         getParams(getproperty(pf, sym), field)
#     end
#     len = length.(res) |> sum
#     isequal(len, 0) ? ParamBox[] : reduce(vcat, res)
# end


(::SelectTrait{ParameterStyle})(::F) where {F<:Function} = 
ifelse(hasmethod(unpackParamFunc, Tuple{F}), IsParamFunc(), NotParamFunc())


# formatInput
(::SelectTrait{InputStyle})(::SpatialAmplitude{<:Any, 0, 1}) = MagnitudeInput()

(::SelectTrait{InputStyle})(::SpatialAmplitude{<:Any, D, 1}) where {D} = TupleInput{D}()

(::SelectTrait{InputStyle})(::SphericalHarmonics{D}) where {D} = TupleInput{D}()

formatInputCall(::VectorInput, f::Function, x::AbstractArray) = f(x|>vec)

formatInputCall(::VectorInput, f::Function, x::Tuple) = f(x|>collect)

formatInputCall(::TupleInput{N}, f::Function, x::Tuple{Vararg{Any, N}}) where {N} = f(x)

formatInputCall(::TupleInput{N}, f::Function, x::AbstractArray) where {N} = 
formatInputCall(TupleInput{N}(), f, Tuple(x))

formatInputCall(::MagnitudeInput, f::Function, x) = f(x|>norm)