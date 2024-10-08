# getParams
(::SelectTrait{ParamBoxAccess})(::SpatialAmplitude) = ContainParamBox()

getParams(::WithoutParamBox, ::Any, ::SymOrMiss) = ParamBox[]

function getParams(::ContainParamBox, obj, sym::SymOrMiss)
    map( fieldnames(pf) ) do field
        getParams(getproperty(pf, sym), field)
    end
    len = length.(res) |> sum
    isequal(len, 0) ? ParamBox[] : reduce(vcat, res)
end