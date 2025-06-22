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
        print(io, pType, "(…)")
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
    print(io,  "(::", typeof(f), ")")
    if !enableCompatShowFormat(B(), io)
        nMethod = getMethodNum(f)
        methodStr = string("(", nMethod, " method", (nMethod > 1 ? "s" : ""), ")")
        print(io,  " ", methodStr)
    end
end


show(io::IO, ::Type{GetEntry{T}}) where {T<:AbstractAccessor} = 
print(io, getObjNameStr(nameof(GetEntry), "GetEntry{$T}"))

show(io::IO, ::Type{GetAxisEntry}) = 
print(io, getObjNameStr(nameof(GetAxisEntry), "GetAxisEntry"))

show(io::IO, ::Type{GetUnitEntry}) = 
print(io, getObjNameStr(nameof(GetUnitEntry), "GetUnitEntry"))

show(io::IO, ::Type{GetGridEntry}) = 
print(io, getObjNameStr(nameof(GetGridEntry), "GetGridEntry"))

show(io::IO, ::Type{StableAdd{T}}) where {T} = 
print(io, getObjNameStr(nameof(StableAdd), "StableAdd{$T}"))

show(io::IO, ::Type{StableMul{T}}) where {T} = 
print(io, getObjNameStr(nameof(StableMul), "StableMul{$T}"))

show(io::IO, ::Type{StableTupleSub{T}}) where {T<:Tuple} = 
print(io, getObjNameStr(nameof(StableTupleSub), "StableTupleSub{$T}"))

show(io::IO, ::Type{TypedCarteFunc{T, D, F}}) where {T, D, F} = 
print(io, getObjNameStr(nameof(TypedCarteFunc), "TypedCarteFunc{$T, $D, $F}"))

show(io::IO, ::Type{CartesianFormatter{N, R}}) where {N, R<:NTuple{N, Real}} = 
print(io, getObjNameStr(nameof(CartesianFormatter), "CartesianFormatter{$N, $R}"))

show(io::IO, ::Type{VoidSetFilter}) = 
print(io, getObjNameStr(nameof(VoidSetFilter), "VoidSetFilter"))

show(io::IO, ::Type{UnitSetFilter}) = 
print(io, getObjNameStr(nameof(UnitSetFilter), "UnitSetFilter"))

show(io::IO, ::Type{GridSetFilter}) = 
print(io, getObjNameStr(nameof(GridSetFilter), "GridSetFilter"))

show(io::IO, ::Type{FullSetFilter}) = 
print(io, getObjNameStr(nameof(FullSetFilter), "FullSetFilter"))

function show(io::IO, ::Type{ContextParamFunc{B, E, F}}) where 
             {B<:Function, E<:Function, F<:Function}
    print(io, string("ContextParamFunc", "{", B, ", ", E, ", ", F, "}"))
end