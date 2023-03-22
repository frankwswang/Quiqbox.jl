using Printf: @sprintf

const superscriptNum = Dict(['0'=>'⁰', '1'=>'¹', '2'=>'²', '3'=>'³', '4'=>'⁴', '5'=>'⁵', 
                             '6'=>'⁶', '7'=>'⁷', '8'=>'⁸', '9'=>'⁹'])
const subscriptNum   = Dict(['0'=>'₀', '1'=>'₁', '2'=>'₂', '3'=>'₃', '4'=>'₄', '5'=>'₅', 
                             '6'=>'₆', '7'=>'₇', '8'=>'₈', '9'=>'₉'])
const superscriptSym = Dict(['+'=>'⁺', '-'=>'⁻', '('=>'⁽', ')'=>'⁾', '!'=>'ꜝ'])

const subscripts = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']


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


function advancedParse(::Type{T}, content::AbstractString, 
                       ParseFunc::F=adaptiveParse) where {T, F<:Function}
    res = ParseFunc(T, content)
    res === nothing && (res = content)
    res
end


function adaptiveParse(::Type{T}, content::AbstractString) where {T<:AbstractFloat}
    res = tryparse(T, content)
    res === nothing && (res = tryparse(Complex{T}, content))
    res
end


function numToSups(num::Int)
    str = string(num)
    [superscriptNum[i] for i in str] |> prod
end

numToSups(::Nothing) = ""

function numToSubs(num::Int)
    str = string(num)
    [subscriptNum[i] for i in str] |> prod
end

numToSubs(::Nothing) = ""


function alignNum(x::T, lpadN::Int=8, rpadN::Union{Int, Missing}=missing; 
                  roundDigits::Int=-1) where {T<:Real}
    if rpadN isa Missing
        rpadN = getAtolDigits(T)+2
        if roundDigits >= 0
            rpadN = min(rpadN, roundDigits+2)
        end
    end
    if roundDigits < 0
        str = x |> string
    else
        format = "%.$(roundDigits)f"
        str = :(@sprintf($format, $x)) |> eval
    end
    body = split(str, '.')
    if length(body) == 2
        head, tail = body
        tail = "."*tail
        rpadN += 1
    else
        head = body[]
        tail = ""
    end
    lpad(head, lpadN) * rpad(tail, rpadN)
end

function alignNumSign(c::Real; roundDigits::Int=-1)
    if c < 0
        alignNum(c, 0, 0; roundDigits)
    else
        " "*alignNum(c, 0, 0; roundDigits)
    end
end


function inSymbol(sym::Symbol, src::Symbol)
    symStr, srcStr = (sym, src) .|> string
    bl = false
    for i in subscripts
        i == symStr[end] && (bl = true; break)
    end
    l1 = sizeof(symStr)
    l2 = sizeof(srcStr)
    bl ? (l1 == l2 && symStr == srcStr) : (l1 <= l2 && startswith(srcStr, symStr))
end


function typeStrNotUnionAll(::Type{T}) where {T}
    strT = string(T)
    rng = findlast("where", strT)
    if rng === nothing
        strT
    else
        strT[1:rng[1]-2]
    end
end


function findFirstEnclosureRange(str::String, startIdx::Int=1, 
                                 bracketPair::NTuple{2, Char}=('{', '}'))
    strLeft = str[startIdx:end]
    opBkt = bracketPair[begin]
    clBkt = bracketPair[end]
    idxBegin = findfirst(opBkt, strLeft)
    idxEnd = idxBegin-1
    idx = idxBegin
    offset = 1
    while offset > 0 && idx < lastindex(str)
        idx = nextind(strLeft, idx)
        strLeft[idx] == opBkt && (offset+=1)
        strLeft[idx] == clBkt && (offset-=1; idxEnd=idx)
    end
    (startIdx-1) .+ (idxBegin:idxEnd)
end


function ShortenStrClip(str::String, clip::AbstractString)
    clip = string(clip)
    tempSym = "#tempSym#"
    while (ids1 = findfirst(clip, str); ids1 !== nothing)
        ids2 = findFirstEnclosureRange(str, ids1[end])
        idxL = lastindex(str)
        ids2L = ids2[end]
        tail = if str[thisind(str, min(idxL, ids2L+1)):thisind(str, min(idxL, ids2L+7))] == 
                  " where "
            tailStartIdx = if str[ids2L+8] == '{'
                findFirstEnclosureRange(str, ids2L+7)[end] + 1
            else
                id1, id2 = findnext.((',', '}'), str, ids2L+7)
                id1 === nothing && (id1 = idxL)
                id2 === nothing && (id2 = idxL)
                min(id1, id2)
            end
            str[tailStartIdx:end]
        else
            str[ids2L+1:end]
        end
        abbrev = ifelse(str[nextind(str, ids1[end])] == '{', "{…}", "")
        str = str[begin:prevind(str, ids1[begin])] * tempSym * abbrev * tail
    end
    replace(str, tempSym=>clip)
end