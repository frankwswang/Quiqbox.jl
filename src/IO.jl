"""

    checkFname(Fname::String; showWarning::Bool=true) -> String

Check if there is a file with the same name in the current directory. If so, add a `"_N"` 
at the end of `Fname`. `showWarning` determines whether to print out the WARNING info when 
there is a file with the same name.
"""
function checkFname(Fname::String; showWarning::Bool=true)
    FnameN = Fname
    while isfile(FnameN) == true
        i = contains(FnameN, ".") ? (findlast(".", FnameN)|>last) : (lastindex(FnameN)+1)
        FnameN = FnameN[1:i-1] * "_N" * FnameN[i:end]
    end
    FnamePrint, FnameNPrint = map([Fname, FnameN]) do f
        contains(f, "/") ? f[(findlast("/", f) |> last)+1 : end] : f
    end
    FnameN != Fname && showWarning && (@warn """The file expected to create already exists. 
                                                Adding another suffix.
                                                Old name: $(FnamePrint)
                                                New name: $(FnameNPrint)""")
    FnameN
end


function getObjNameStr(objName::Symbol, objAlias::AbstractString=string(objName))
    (Base.isexported(Quiqbox, objName) ? "" : string(nameof(Quiqbox), ".")) * objAlias
end

function enableCompatShowFormat(::B, io::IO) where {B<:BoolVal}
    get(io, :compact, false)::Bool || getValData(B)
end


#>= Custom pretty-printing =<#
Base.show(io::IO, ::MIME"text/plain", s::IndexedSym) = showIndexedSym(io, s)

function showIndexedSym(io::IO, s::IndexedSym)
    idx = s.index
    print(io, s.name, ( ismissing(idx) ? "" : numToSubs(idx) ))
end


Base.show(io::IO, p::ParamBox) = showParamBox(True(), io, p)

Base.show(io::IO, ::MIME"text/plain", p::ParamBox) = showParamBox(False(), io, p)

function showParamBox(::B, io::IO, p::ParamBox) where {B<:BoolVal}
    pType = typeof(p)
    pName = nameof(pType)
    pSymbol = indexedSymOf(p)
    aliasStr = getTypeAliasStr(p)
    outerType = get(io, :typeinfo, Any)
    if pType <: outerType <: ParamBox
        if outerType == pType
            show(io, MIME("text/plain"), pSymbol)
        else
            print(io, "(")
            show(io, MIME("text/plain"), pSymbol)
            print(io, "::", getObjNameStr(pName), ")")
        end
    elseif enableCompatShowFormat(B(), io)
        print(io, aliasStr)
        print(io, "(…)")
    else
        print(io, "(")
        show(io, MIME("text/plain"), pSymbol)
        print(io, "::", getObjNameStr(pName), ")")
        level = screenLevelOf(p)
        relationStr = (level==0 ? " ==> " : (level==1 ? " <=> " : " <== "))
        print(io, relationStr, "(")
        if level > 0
            show(IOContext(io, :compact=>true), obtain(p))
        else
            print(io, "::", getOutputType(p))
        end
        print(io, ")")
    end
end


Base.show(io::IO, f::CompositeFunction) = 
showCompositeFunc(True(), io, f)

Base.show(io::IO, ::MIME"text/plain", f::CompositeFunction) = 
showCompositeFunc(False(), io, f)

function showCompositeFunc(::B, io::IO, f::CompositeFunction) where {B<:BoolVal}
    nMethod = getMethodNum(f)
    aliasStr = getTypeAliasStr(f)
    if enableCompatShowFormat(B(), io)
        print(io,  "(::", aliasStr, ")")
    else
        typeStr = string("(::", aliasStr, ")")
        methodStr = string("(", nMethod, " method", (nMethod > 1 ? "s" : ""), ")")
        print(io,  typeStr, " ", methodStr)
    end
end


function getTypeAliasStr(obj)
    objType = typeof(obj)
    if hasmethod(getTypeAliasStrCore, (objType,))
        objName = objType <: Function ? nameof(obj) : nameof(objType)
        getObjNameStr(objName, getTypeAliasStrCore(obj))
    else
        string(obj|>typeof)
    end
end

getTypeAliasStrCore(::StableAdd{T}) where {T} = string("StableAdd{$T}")

getTypeAliasStrCore(::StableMul{T}) where {T} = string("StableMul{$T}")

function getTypeAliasStrCore(::ScreenParam{T, E, P}) where {T, E<:Pack{T}, P<:ParamBox{T}}
    tailStr = P <: PrimitiveParam ? string(P) : "…"
    string("ScreenParam", "{", T, ", ", E, ", " , tailStr, "}")
end

getTypeAliasStrCore(::ReduceParam{T, E}) where {T, E<:Pack{T}} = 
string("ReduceParam", "{", T, ", ", E, ", ", "…}")

getTypeAliasStrCore(::ExpandParam{T, E, N}) where {T, E<:Pack{T}, N} = 
string("ExpandParam", "{", T, ", ", E, ", ", N, ", ", "…}")