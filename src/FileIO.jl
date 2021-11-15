using Printf: @sprintf

const superscriptNum = Dict(['0'=>'⁰', '1'=>'¹', '2'=>'²', '3'=>'³', '4'=>'⁴', '5'=>'⁵', 
                             '6'=>'⁶', '7'=>'⁷', '8'=>'⁸', '9'=>'⁹'])
const subscriptNum   = Dict(['0'=>'₀', '1'=>'₁', '2'=>'₂', '3'=>'₃', '4'=>'₄', '5'=>'₅', 
                             '6'=>'₆', '7'=>'₇', '8'=>'₈', '9'=>'₉'])
const superscriptSym = Dict(['+'=>'⁺', '-'=>'⁻', '('=>'⁽', ')'=>'⁾', '!'=>'ꜝ'])


"""

    checkFname(Fname::String; showWarning::Bool=true) -> String

Check if there is a file with the same name in the current directory. If so, will add an 
`"_N"` at the end of the file name `String`. `showWarning` determines whether prints out 
the WARNING info when there is a file with the same name.
"""
function checkFname(Fname::String; showWarning::Bool=true)
    FnameN = Fname
    while isfile(FnameN) == true
        i=0
        contains(FnameN, ".") ? i=findlast(".", FnameN)|>last : i=(FnameN|>length)+1
        FnameN = FnameN[1:i-1]*"_N"*FnameN[i:end]
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


function advancedParse(content::AbstractString, 
                       ParseFunc::F=adaptiveParse) where {F<:Function}
    res = ParseFunc(content)
    res === nothing && (res = content)
    res
end


function adaptiveParse(content::AbstractString)
    res = tryparse(Int, content)
    res === nothing && (res = tryparse(Float64, content))
    res === nothing && (res = tryparse(Complex{Float64}, content))
    res
end


function numToSups(num::Int)
    str = string(num)
    [superscriptNum[i] for i in str] |> prod
end


function numToSubs(num::Int)
    str = string(num)
    [subscriptNum[i] for i in str] |> prod
end


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