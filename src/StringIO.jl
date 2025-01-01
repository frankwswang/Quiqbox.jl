export symbolFrom

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

function getSNxpn(x::Real)
    x = abs(x)
    if x == 0
        0
    else
        floor(Int, log10(x))
    end
end

function getSigDecimal(x)
    xabsStr = string(x|>abs)
    idx = findfirst('e', xabsStr)
    idx = if idx === nothing
        length(xabsStr)
    else
        idx - 1
    end
    xDigitsStr = xabsStr[begin:idx]
    if xDigitsStr[begin] == '0'
        xDigitsStr = xDigitsStr[3:end]
        idx = findfirst(x->x!='0', xDigitsStr)
        idx === nothing && (idx = 0)
        idx = length(xDigitsStr) - idx + 1
    elseif endswith(xDigitsStr, ".0")
        idx -= 2
        while xDigitsStr[idx] == '0'
            idx -= 1
        end
    else
        idx -= 1
    end
    idx - 1
end

function alignNum(x::T, lPad::Int=8, rPad::Union{Int, Missing}=missing; 
                  roundDigits::Int=-1) where {T<:Real}
    notNumber = isnan(x)
    noRounding = roundDigits < 0
    xTypeDigits = ifelse(notNumber, -3, getAtolDigits(T))
    rPad = ifelse(rPad isa Missing, ifelse(noRounding, xTypeDigits+5, roundDigits), rPad)
    str = if x isa Integer || notNumber || noRounding
        str = string(x)
    else
        if 1 <= abs(x) < 10
            format = "%.$(roundDigits)f"
        else
            xpn = getSNxpn(x)
            xpnDigits = max(2, (ndigits∘abs)(xpn)) + 2
            sigDecimal = getSigDecimal(x)
            minDigits = roundDigits - xpnDigits
            format = if roundDigits >= xpnDigits && 
                        min(sigDecimal, roundDigits+xpn+1) <= minDigits && 
                        (-xpn >= xpnDigits || xpn >= max(lPad-signbit(x), 1))
                "%.$(minDigits)e"
            elseif xpn > 0
                maxDigits = xTypeDigits - xpn
                maxDigits < 0 && (x = (BigFloat∘string)(x))
                "%.$(min(roundDigits, max(0, maxDigits)))f"
            else
                maxDigits = max(sigDecimal + 1 - xpn, xTypeDigits+1)
                "%.$(min(roundDigits, maxDigits))f"
            end
        end
        :(@sprintf($format, $x)) |> eval
    end
    body = split(str, '.')
    if length(body) == 2
        head, tail = body
        tail = "."*tail
        rPad += 1
    else
        head = body[]
        tail = ""
    end
    lpad(head, lPad) * rpad(tail, rPad)
end

function alignNumSign(c::Real, rPad::Int=0; roundDigits::Int=-1)
    cStr = alignNum(c, 0, 0; roundDigits)
    if cStr[begin] != '-'
        cStr = " "*cStr
    end
    rpad(cStr, rPad)
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


function cropStrR(str::String, maxLen::Int)
    strLen = length(str)
    if maxLen > strLen
        rpad(str, maxLen)
    else
        i = 0
        mapreduce(*, 1:maxLen) do _
            i = nextind(str, i)
            str[i]
        end
    end
end


function setNumDigits(::Type{T1}, num::T2) where {T1<:Real, T2<:Real}
    d1 = getAtolDigits(T1)
    min(ifelse(isnan(num), d1, getAtolDigits(num)), d1)
end


function genTimeStr(ns::Real, ratioTons::Real=1; 
                    autoUnit::Bool=true, roundDigits::Int=TimerDigits)
    t = ratioTons / 1e9 * ns
    unit = "second"
    if !autoUnit || 1e-2 <= t <=60
    elseif t > 3600
        t /= 3600
        unit = "hour"
    elseif t > 60
        t /= 60
        unit = "minute"
    else
        t *= 1000
        unit = "millisecond"
    end
    unit = ifelse(t>1, unit*'s', unit)
    alignNum(t, 0; roundDigits) * " " * unit
end

function defSymbolIndex(sym::Symbol)
    let sym=sym
        function (index::Int)
            index < 0 && throw(DomainError(index, "`index` should be non-negative `Int`."))
            Symbol(sym, index)
        end
    end
end

genSymbolTuple(sym::Symbol, len::Int) = ntuple(defSymbolIndex(sym), len)


function convertSciNotation(str::AbstractString)
    replace(str, r"D(?=[\+\-])" => "e")
end


mutable struct IndexedSym
    const name::Symbol
    index::MissingOr{Int}

    IndexedSym(name::Symbol, index::MissingOr{Int}=missing) = new(name, index)
end

IndexedSym(idxSym::IndexedSym) = IndexedSym(idxSym.name, idxSym.index)

IndexedSym(idxSym::IndexedSym, sym::Symbol) = 
IndexedSym(Symbol(idxSym.name, sym), idxSym.index)

IndexedSym(sym::Symbol, idxSym::IndexedSym) = 
IndexedSym(Symbol(sym, idxSym.name), idxSym.index)

function symbolFrom(is::IndexedSym)
    idx = is.index
    if ismissing(idx)
        is.name
    else
        Symbol(is.name, '_', idx)
    end
end