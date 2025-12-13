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


function getOwnedObjNameStr(objName::Symbol, objAlias::AbstractString=string(objName))
    addPrefix = if isdefined(Quiqbox, objName)
        isequal((parentmodule∘getfield)(Quiqbox, objName), Quiqbox) && 
                !Base.isexported(Quiqbox, objName)
    else
        false
    end
    (addPrefix ? string(nameof(Quiqbox), ".") : "") * objAlias
end

function enableCompatShowFormat(::B, io::IO) where {B<:Boolean}
    get(io, :compact, false)::Bool || evalTypedData(B)
end


#>= Custom pretty-printing =<#
Base.show(io::IO, ::MIME"text/plain", s::IndexedSym) = showIndexedSym(io, s)

function showIndexedSym(io::IO, s::IndexedSym)
    idx = s.index
    print(io, "#", s.name, ( iszero(idx) ? "" : "{$idx}" ))
end


Base.show(io::IO, p::ParamBox) = showParamBox(True(), io, p)

Base.show(io::IO, ::MIME"text/plain", p::ParamBox) = showParamBox(False(), io, p)

function showParamBox(::B, io::IO, p::ParamBox) where {B<:Boolean}
    pType = typeof(p)
    pName = nameof(pType)
    pSymbol = markerOf(p)
    outerType = get(io, :typeinfo, Any)
    if pType <: outerType <: ParamBox
        if outerType == pType
            show(io, MIME("text/plain"), pSymbol)
        else
            print(io, "(")
            show(io, MIME("text/plain"), pSymbol)
            print(io, " :: ", getOwnedObjNameStr(pName), ")")
        end
    elseif enableCompatShowFormat(B(), io)
        customShow(io, pType)
        print(io, "(…)")
    else
        print(io, "(")
        show(io, MIME("text/plain"), pSymbol)
        print(io, " ::", getOwnedObjNameStr(pName), ")")
        level = screenLevelOf(p)
        relationStr = (level==0 ? " ==> " : (level==1 ? " <=> " : " <== "))
        print(io, relationStr, "(")
        outputType = if level > 0
            output = obtain(p)
            formattedOutput = (output isa AbstractArray) ? vec(output) : output
            show(IOContext(io, :compact=>true), formattedOutput)
            print(io, " ::")
            typeof(output)
        else
            print(io, "::")
            getOutputType(p)
        end
        customShow(io, outputType)
        print(io, ")")
    end
end


function show(io::IO, ::TypePiece{T}) where {T}
    print(io, getOwnedObjNameStr(nameof(T), "TypePiece{"))
    customShow(io, T)
    print(io, "}")
end


Base.show(io::IO, f::CompositeFunction) = 
showCompositeFunc(True(), io, f)

Base.show(io::IO, ::MIME"text/plain", f::CompositeFunction) = 
showCompositeFunc(False(), io, f)

function showCompositeFunc(::B, io::IO, f::CompositeFunction) where {B<:Boolean}
    print(io,  "(::")
    customShow(io, typeof(f))
    print(io,  ")")
    if !enableCompatShowFormat(B(), io)
        nMethod = getMethodNum(f)
        methodStr = string("(", nMethod, " method", (nMethod > 1 ? "s" : ""), ")")
        print(io,  " ", methodStr)
    end
end


Base.show(io::IO, ::Type{GetAxisEntry}) = 
print(io, getOwnedObjNameStr(nameof(GetAxisEntry), "GetAxisEntry"))

Base.show(io::IO, ::Type{GetUnitEntry}) = 
print(io, getOwnedObjNameStr(nameof(GetUnitEntry), "GetUnitEntry"))

Base.show(io::IO, ::Type{GetGridEntry}) = 
print(io, getOwnedObjNameStr(nameof(GetGridEntry), "GetGridEntry"))

Base.show(io::IO, ::Type{VoidSetFilter}) = 
print(io, getOwnedObjNameStr(nameof(VoidSetFilter), "VoidSetFilter"))

Base.show(io::IO, ::Type{UnitSetFilter}) = 
print(io, getOwnedObjNameStr(nameof(UnitSetFilter), "UnitSetFilter"))

Base.show(io::IO, ::Type{GridSetFilter}) = 
print(io, getOwnedObjNameStr(nameof(GridSetFilter), "GridSetFilter"))

Base.show(io::IO, ::Type{FullSetFilter}) = 
print(io, getOwnedObjNameStr(nameof(FullSetFilter), "FullSetFilter"))

#>> Cannot directly overload `show` of parameterized `Type` because type parameters 
#>> are not individually dispatched on. Otherwise, it is possible for unbound type 
#>> parameters in `Type{T}` where `T` is a `UnionAll` leak into the function body.
function customShow(io::IO, typeInfo) #> `typeInfo` can be a non-`Type` type parameter
    if (typeInfo isa Type) && isconcretetype(typeInfo)
        print(io, getOwnedObjNameStr(nameof(typeInfo), ""))
        customShowCore(io, typeInfo)
    else
        show(io, typeInfo)
    end
end

customShowCore(io::IO, type::Type) = show(io, type)

function customShowCore(io::IO, ::Type{StableAdd{T}}) where {T}
    print(io, "StableAdd{")
    customShow(io, T)
    print(io, "}")
end

function customShowCore(io::IO, ::Type{StableMul{T}}) where {T}
    print(io, "StableMul{")
    customShow(io, T)
    print(io, "}")
end

function customShowCore(io::IO, ::Type{StableTupleSub{T}}) where {T<:Tuple}
    print(io, "StableTupleSub{")
    customShow(io, T)
    print(io, "}")
end

function customShowCore(io::IO, ::Type{Typed{T}}) where {T}
    print(io, "Typed{")
    customShow(io, T)
    print(io, "}")
end

function customShowCore(io::IO, ::Type{TypedCarteFunc{T, D, F}}) where {T, D, F<:Function}
    print(io, "TypedCarteFunc{")
    customShow(io, T)
    print(io, ", ")
    customShow(io, D)
    print(io, ", ")
    customShow(io, F)
    print(io, "}")
end

function customShowCore(io::IO, ::Type{ContextParamFunc{B, E, F}}) where 
                       {B<:Function, E<:Function, F<:Function}
    print(io, "ContextParamFunc{")
    customShow(io, B)
    print(io, ", ")
    customShow(io, E)
    print(io, ", ")
    customShow(io, F)
    print(io, "}")
end

function customShowCore(io::IO, ::Type{CartesianFormatter{N, R}}) where 
                       {N, R<:NTuple{N, Real}}
    print(io, "CartesianFormatter{")
    customShow(io, N)
    print(io, ", ")
    customShow(io, R)
    print(io, "}")
end

function customShowCore(io::IO, ::Type{DirectMemory{T, N}}) where {T, N}
    print(io, "DirectMemory{")
    customShow(io, T)
    print(io, ", ")
    customShow(io, N)
    print(io, "}")
end

function customShowCore(io::IO, ::Type{NestedMemory{T, E, N}}) where 
                       {T, E<:PackedMemory{T}, N}
    print(io, "NestedMemory{")
    customShow(io, T)
    print(io, ", ")
    customShow(io, E)
    print(io, ", ")
    customShow(io, N)
    print(io, "}")
end