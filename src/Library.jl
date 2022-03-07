export getCharge, ParamList

const ElementNames = 
[
"H",
"He",
"Li",
"Be",
"B",
"C",
"N",
"O",
"F",
"Ne",
"Na",
"Mg",
"Al",
"Si",
"P",
"S",
"Cl",
"Ar",
"K",
"Ca"
]


const SubshellNames = 
[
"S",
"P",
"D",
"F",
"G",
"H",
"I"
]


# const SubshellAngularMomentums = 
# [
# ("X⁰Y⁰Z⁰",),
# ("X¹Y⁰Z⁰", "X⁰Y¹Z⁰", "X⁰Y⁰Z¹"),
# ("X²Y⁰Z⁰", "X¹Y¹Z⁰", "X¹Y⁰Z¹", "X⁰Y²Z⁰", "X⁰Y¹Z¹", "X⁰Y⁰Z²"),
# ("X³Y⁰Z⁰", "X²Y¹Z⁰", "X²Y⁰Z¹", "X¹Y²Z⁰", "X¹Y¹Z¹", "X¹Y⁰Z²", "X⁰Y³Z⁰", "X⁰Y²Z¹", "X⁰Y¹Z²", "X⁰Y⁰Z³"),
# ("X⁴Y⁰Z⁰", "X³Y¹Z⁰", "X³Y⁰Z¹", "X²Y²Z⁰", "X²Y¹Z¹", "X²Y⁰Z²", "X¹Y³Z⁰", "X¹Y²Z¹", "X¹Y¹Z²", "X¹Y⁰Z³", "X⁰Y⁴Z⁰", "X⁰Y³Z¹", "X⁰Y²Z²", "X⁰Y¹Z³", "X⁰Y⁰Z⁴"),
# ("X⁵Y⁰Z⁰", "X⁴Y¹Z⁰", "X⁴Y⁰Z¹", "X³Y²Z⁰", "X³Y¹Z¹", "X³Y⁰Z²", "X²Y³Z⁰", "X²Y²Z¹", "X²Y¹Z²", "X²Y⁰Z³", "X¹Y⁴Z⁰", "X¹Y³Z¹", "X¹Y²Z²", "X¹Y¹Z³", "X¹Y⁰Z⁴", "X⁰Y⁵Z⁰", "X⁰Y⁴Z¹", "X⁰Y³Z²", "X⁰Y²Z³", "X⁰Y¹Z⁴", "X⁰Y⁰Z⁵"),
# ("X⁶Y⁰Z⁰", "X⁵Y¹Z⁰", "X⁵Y⁰Z¹", "X⁴Y²Z⁰", "X⁴Y¹Z¹", "X⁴Y⁰Z²", "X³Y³Z⁰", "X³Y²Z¹", "X³Y¹Z²", "X³Y⁰Z³", "X²Y⁴Z⁰", "X²Y³Z¹", "X²Y²Z²", "X²Y¹Z³", "X²Y⁰Z⁴", "X¹Y⁵Z⁰", "X¹Y⁴Z¹", "X¹Y³Z²", "X¹Y²Z³", "X¹Y¹Z⁴", "X¹Y⁰Z⁵", "X⁰Y⁶Z⁰", "X⁰Y⁵Z¹", "X⁰Y⁴Z²", "X⁰Y³Z³", "X⁰Y²Z⁴", "X⁰Y¹Z⁵", "X⁰Y⁰Z⁶")
# ]

struct XYZTuple{S}
    tuple::NTuple{3, Int}

    function XYZTuple{S}(t::NTuple{3, Int}) where {S}
        @assert all(t .>= 0)
        new{sum(t)}(t)
    end

    function XYZTuple(xyz1::XYZTuple{S1}, xyz2::XYZTuple{S2}) where {S1, S2}
        new{S1+S2}(xyz1.tuple .+ xyz2.tuple)
    end
end

XYZTuple(t::NTuple{3, Int}) = XYZTuple{sum(t)}(t)
XYZTuple(args::Vararg{Int, 3}) = XYZTuple(args)
XYZTuple(a::Vector{Int}) = XYZTuple(a...)

import Base: iterate, size, length, ndims, +, isless, Tuple
iterate(snt::XYZTuple, args...) = iterate(snt.tuple, args...)
size(snt::XYZTuple, args...) = size(snt.tuple, args...)
length(snt::XYZTuple) = length(snt.tuple)
ndims(snt::XYZTuple) = ndims(snt.tuple)
+(xyz1::XYZTuple{S1}, xyz2::XYZTuple{S2}) where {S1, S2} = XYZTuple(xyz1, xyz2)
isless(xyz1::XYZTuple, xyz2::XYZTuple) = isless(xyz1.tuple, xyz2.tuple)
Tuple(xyz::XYZTuple) = xyz.tuple

const SubshellXYZs = 
[ # Every XYZs must start with (l, 0, 0)
(XYZTuple(0,0,0),),
(XYZTuple(1,0,0), XYZTuple(0,1,0), XYZTuple(0,0,1)),
(XYZTuple(2,0,0), XYZTuple(1,1,0), XYZTuple(1,0,1), XYZTuple(0,2,0), XYZTuple(0,1,1), XYZTuple(0,0,2)),
(XYZTuple(3,0,0), XYZTuple(2,1,0), XYZTuple(2,0,1), XYZTuple(1,2,0), XYZTuple(1,1,1), XYZTuple(1,0,2), XYZTuple(0,3,0), XYZTuple(0,2,1), XYZTuple(0,1,2), XYZTuple(0,0,3)),
(XYZTuple(4,0,0), XYZTuple(3,1,0), XYZTuple(3,0,1), XYZTuple(2,2,0), XYZTuple(2,1,1), XYZTuple(2,0,2), XYZTuple(1,3,0), XYZTuple(1,2,1), XYZTuple(1,1,2), XYZTuple(1,0,3), XYZTuple(0,4,0), XYZTuple(0,3,1), XYZTuple(0,2,2), XYZTuple(0,1,3), XYZTuple(0,0,4)),
(XYZTuple(5,0,0), XYZTuple(4,1,0), XYZTuple(4,0,1), XYZTuple(3,2,0), XYZTuple(3,1,1), XYZTuple(3,0,2), XYZTuple(2,3,0), XYZTuple(2,2,1), XYZTuple(2,1,2), XYZTuple(2,0,3), XYZTuple(1,4,0), XYZTuple(1,3,1), XYZTuple(1,2,2), XYZTuple(1,1,3), XYZTuple(1,0,4), XYZTuple(0,5,0), XYZTuple(0,4,1), XYZTuple(0,3,2), XYZTuple(0,2,3), XYZTuple(0,1,4), XYZTuple(0,0,5)),
(XYZTuple(6,0,0), XYZTuple(5,1,0), XYZTuple(5,0,1), XYZTuple(4,2,0), XYZTuple(4,1,1), XYZTuple(4,0,2), XYZTuple(3,3,0), XYZTuple(3,2,1), XYZTuple(3,1,2), XYZTuple(3,0,3), XYZTuple(2,4,0), XYZTuple(2,3,1), XYZTuple(2,2,2), XYZTuple(2,1,3), XYZTuple(2,0,4), XYZTuple(1,5,0), XYZTuple(1,4,1), XYZTuple(1,3,2), XYZTuple(1,2,3), XYZTuple(1,1,4), XYZTuple(1,0,5), XYZTuple(0,6,0), XYZTuple(0,5,1), XYZTuple(0,4,2), XYZTuple(0,3,3), XYZTuple(0,2,4), XYZTuple(0,1,5), XYZTuple(0,0,6))
]

function ijkToStr(ijk::XYZTuple)
    res = ""
    xyz = ("X", "Y", "Z")
    for (i, j) in zip(xyz, ijk)
        res *= i * superscriptNum[j + '0']
    end
    res
end

const BasisFuncNames = 
[
"STO-2G", 
"STO-3G", 
"STO-6G", 
"3-21G", 
"6-31G", 
"cc-pVDZ", 
"cc-pVTZ", 
"cc-pVQZ"
]


const SciNotMarker = r"D(?=[\+\-])"
const sciNotReplace = (txt)->replace(txt, SciNotMarker => "e")
const BStextEndingMarker = "****"
const BasisSetList = Dict(BasisFuncNames .=> BasisFuncTexts)
const AtomicNumberList = Dict(ElementNames .=> collect(1 : length(ElementNames)))
const AngularMomentumList = Dict(SubshellNames .=> collect(0 : length(SubshellNames)-1))
const SubshellSuborderList = Dict(SubshellNames .=> SubshellXYZs)
const SubshellDimList = Dict(SubshellNames .=> length.(SubshellXYZs))
# const ijkOrderList = Dict(SubshellNames .=> SubshellAngularMomentums)
const ijkIndexList = Dict(flatten(SubshellXYZs) .=> flatten([collect(1:length(i)) for i in SubshellXYZs]))
# const ijkOrbitalList = Dict(flatten(SubshellAngularMomentums)  .=> flatten(SubshellXYZs))
# const ijkStringList = Dict(flatten(SubshellXYZs) .=> flatten(SubshellAngularMomentums))
# const ParamNames = [:𝑋, :𝑌, :𝑍, :𝑑, :𝛼, :𝐿]
const ParamNames = [:X, :Y, :Z, :d, :α, :L]
const ParamSymbols = [:X, :Y, :Z, :con, :xpn, :spacing]
const ParamList = Dict(ParamSymbols .=> ParamNames)

getCharge(nucs::Vector{String}) = getCharge.(nucs) |> sum

getCharge(nucStr::String) = AtomicNumberList[nucStr]


function checkBSList(;printInfo::Bool=false)
    texts = vcat(BasisFuncTexts...)
    for text in texts
        if text !== nothing
            sBool = startswith(text, r"[A-Z][a-z]?     [0-9]\n")
            eBool = endswith(text, BStextEndingMarker*"\n")
            @assert (sBool && eBool) """\n
            The format of 'Basis functions' is NOT CORRECT!
            "[A-Z][a-z]?     [0-9]\\n" (regex) should exit as the 1st line of the string!
            "$(BStextEndingMarker)\\n" should exit as the last line of the string!
            The incorrect text content is:\n$(text)
            """
        end
    end
    printInfo && println("Basis function list checked.")
end