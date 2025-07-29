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

function checkEmptiness(obj, name::Symbol; reverseCheck::Bool=false)
    str = reverseCheck ? "" : " not"
    if ifelse(reverseCheck, !, itself)(obj|>isempty)
        throw(AssertionError("`$name` must$str be empty."))
    end
    length(obj)
end

function checkLengthCore(objLen::Int, objName::Symbol, 
                         len::Int, lenName::Union{Missing, String}=missing)
    if objLen != len
        str = ismissing(lenName) ? "be equal to $len" : "match $lenName: $len"
        throw(AssertionError("The length of `$objName` must " * str * "."))
    end
    nothing
end

function checkLength(obj, name::Symbol, len::Int, lenName::Union{Missing, String}=missing)
    checkLengthCore(length(obj), name, len, lenName)
end
#! Standardize return type
function checkPositivity(num::Real, allowZero::Bool=false)
    subStr = ifelse(allowZero, "non-negative", "positive")
    (num + Int(allowZero)) > 0 || throw(AssertionError("`num` should be $subStr."))
    nothing
end

function checkBottomType(::Type{T}) where {T}
    T <: Union{} && throw(AssertionError("`T` must not be `Union{}`."))
    nothing
end


function checkIntLevelMismatch(actualLevel::Int, targetLevelSet::NonEmptyTuple{Int}, 
                               levelPrefix::AbstractString="")
    if length(targetLevelSet) == 1
        targetLevel = first(targetLevelSet)
        errorFlag = (actualLevel != targetLevel)
        errorStr = "`$targetLevel`."
    else
        errorFlag = !(actualLevel in targetLevelSet)
        errorStr = "within `$targetLevelSet`."
    end

    if errorFlag
        levelStr = ifelse(length(levelPrefix)==0, "level", "$levelPrefix level")
        throw(AssertionError("The $levelStr is $actualLevel, which should have been " * 
                             errorStr))
    end

    actualLevel
end


function checkBottomArray(arr::AbstractArray)
    if eltype(arr) <:Union{} && !isempty(arr)
        throw(AssertionError("`arr` must be empty if its element type is `Union{}`."))
    end
    nothing
end