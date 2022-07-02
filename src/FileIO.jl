using Printf: @sprintf

const superscriptNum = Dict(['0'=>'⁰', '1'=>'¹', '2'=>'²', '3'=>'³', '4'=>'⁴', '5'=>'⁵', 
                             '6'=>'⁶', '7'=>'⁷', '8'=>'⁸', '9'=>'⁹'])
const subscriptNum   = Dict(['0'=>'₀', '1'=>'₁', '2'=>'₂', '3'=>'₃', '4'=>'₄', '5'=>'₅', 
                             '6'=>'₆', '7'=>'₇', '8'=>'₈', '9'=>'₉'])
const superscriptSym = Dict(['+'=>'⁺', '-'=>'⁻', '('=>'⁽', ')'=>'⁾', '!'=>'ꜝ'])

const subscripts = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']
"""

    checkFname(Fname::String; showWarning::Bool=true) -> String

Check if there is a file with the same name in the current directory. If so, will add an 
`"_N"` at the end of the file name `String`. `showWarning` determines whether prints out 
the WARNING info when there is a file with the same name.
"""
function checkFname(Fname::String; showWarning::Bool=true)
    FnameN = Fname
    while isfile(FnameN) == true
        i = contains(FnameN, ".") ? ( findlast(".", FnameN)|>last ) : ( (FnameN|>length)+1 )
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


function alignNum(x::Number, lpadN::Int=8, rpadN::Int=21; roundDigits::Int=-1)
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
    l1 = length(symStr)
    l2 = length(srcStr)
    bl ? (l1 == l2 && symStr == srcStr) : (l1 <= l2 && symStr == srcStr[1:l1])
end