function checkFuncReturn(f::F, fName::Symbol, ::Type{TArg}, ::Type{TReturn}) where 
                        {F<:Function, TArg, TReturn}
    Base.return_types(f, (TArg,))[] == TReturn || 
    throw(ArgumentError("`$fName`: `$f` should return a `$TReturn` given an input "*
                        "argument of type `$TArg`."))
end


function checkCollectionMinLen(data, dataSym::Symbol, minLen::Int)
    minLen = abs(minLen)
    str = minlen > 1 ? "s." : "."
    dataLen = length(data)
    dataLen < minLen && 
    throw(ArgumentError("`$dataSym`: $data should contain at least $minLen element"* str))
    dataLen == minLen
end

function checkEmptiness(obj, name::Symbol)
    isempty(obj) && throw(AssertionError("`$name` must not be empty."))
    length(obj)
end

function checkLength(obj, name::Symbol, len::Int)
    if len <= 0
        checkEmptiness(obj, name)
    elseif length(obj) != len
        throw(AssertionError("The length of `$name` must match `len=$len`."))
    end
    nothing
end