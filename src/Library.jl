export XYZTuple, getCharge, ParamList

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


struct XYZTuple{L}
    tuple::NTuple{3, Int}

    function XYZTuple{L}(t::NTuple{3, Int}) where {L}
        @assert all(t .>= 0)
        new{sum(t)}(t)
    end

    XYZTuple(xyz::XYZTuple{L}) where {L} = new{L}(xyz.tuple)

    XYZTuple(xyz1::XYZTuple{L1}, xyz2::XYZTuple{L2}) where {L1, L2} = 
    new{L1+L2}(xyz1.tuple .+ xyz2.tuple)

    XYZTuple(xyz::XYZTuple{L}, ::XYZTuple{0}) where {L} = new{L}(xyz.tuple)

    XYZTuple(::XYZTuple{0}, xyz::XYZTuple{L}) where {L} = new{L}(xyz.tuple)

    XYZTuple(::XYZTuple{0}, ::XYZTuple{0}) = new{0}((0, 0, 0))
end

XYZTuple(t::NTuple{3, Int}) = XYZTuple{sum(t)}(t)
XYZTuple(args::Vararg{Int, 3}) = XYZTuple(args)
XYZTuple(a::Vector{Int}) = XYZTuple(a...)

import Base: iterate, size, length, ndims, +, isless, Tuple, sum, map, broadcastable
iterate(snt::XYZTuple, args...) = iterate(snt.tuple, args...)
size(snt::XYZTuple, args...) = size(snt.tuple, args...)
length(snt::XYZTuple) = length(snt.tuple)
ndims(snt::XYZTuple) = ndims(snt.tuple)
+(xyz1::XYZTuple{L1},  xyz2::XYZTuple{L2}  ) where {L1, L2} = XYZTuple(xyz1, xyz2)
+( xyz::XYZTuple{L},      t::NTuple{3, Int}) where {L} = xyz + XYZTuple{sum(t)}(t)
+(   t::NTuple{3, Int}, xyz::XYZTuple{L}   ) where {L} = +(xyz, t)
@inline isless(xyz1::XYZTuple, xyz2::XYZTuple) = isless(xyz1.tuple, xyz2.tuple)
@inline Tuple(xyz::XYZTuple) = xyz.tuple
@inline sum(::XYZTuple{L}) where {L} = L
@inline sum(f, xyz::XYZTuple) = sum(f, xyz.tuple)
@inline map(f, x::XYZTuple{L1}, y::XYZTuple{L2}) where {L1, L2} = map(f, x.tuple, y.tuple)
@inline map(f, xyzs::Vararg{XYZTuple, N}) where {N} = map(f, getfield.(xyzs, :tuple)...)
Base.broadcastable(xyz::XYZTuple) = Base.broadcastable(xyz.tuple)

const SubshellXYZs = 
[ # Every XYZs must start with (l, 0, 0)
(XYZTuple(0,0,0),),
(XYZTuple(1,0,0), XYZTuple(0,1,0), XYZTuple(0,0,1)),
(XYZTuple(2,0,0), XYZTuple(1,1,0), XYZTuple(1,0,1), XYZTuple(0,2,0), XYZTuple(0,1,1), 
 XYZTuple(0,0,2)),
(XYZTuple(3,0,0), XYZTuple(2,1,0), XYZTuple(2,0,1), XYZTuple(1,2,0), XYZTuple(1,1,1), 
 XYZTuple(1,0,2), XYZTuple(0,3,0), XYZTuple(0,2,1), XYZTuple(0,1,2), XYZTuple(0,0,3)),
(XYZTuple(4,0,0), XYZTuple(3,1,0), XYZTuple(3,0,1), XYZTuple(2,2,0), XYZTuple(2,1,1), 
 XYZTuple(2,0,2), XYZTuple(1,3,0), XYZTuple(1,2,1), XYZTuple(1,1,2), XYZTuple(1,0,3), 
 XYZTuple(0,4,0), XYZTuple(0,3,1), XYZTuple(0,2,2), XYZTuple(0,1,3), XYZTuple(0,0,4)),
(XYZTuple(5,0,0), XYZTuple(4,1,0), XYZTuple(4,0,1), XYZTuple(3,2,0), XYZTuple(3,1,1), 
 XYZTuple(3,0,2), XYZTuple(2,3,0), XYZTuple(2,2,1), XYZTuple(2,1,2), XYZTuple(2,0,3), 
 XYZTuple(1,4,0), XYZTuple(1,3,1), XYZTuple(1,2,2), XYZTuple(1,1,3), XYZTuple(1,0,4), 
 XYZTuple(0,5,0), XYZTuple(0,4,1), XYZTuple(0,3,2), XYZTuple(0,2,3), XYZTuple(0,1,4), 
 XYZTuple(0,0,5)),
(XYZTuple(6,0,0), XYZTuple(5,1,0), XYZTuple(5,0,1), XYZTuple(4,2,0), XYZTuple(4,1,1), 
 XYZTuple(4,0,2), XYZTuple(3,3,0), XYZTuple(3,2,1), XYZTuple(3,1,2), XYZTuple(3,0,3), 
 XYZTuple(2,4,0), XYZTuple(2,3,1), XYZTuple(2,2,2), XYZTuple(2,1,3), XYZTuple(2,0,4), 
 XYZTuple(1,5,0), XYZTuple(1,4,1), XYZTuple(1,3,2), XYZTuple(1,2,3), XYZTuple(1,1,4), 
 XYZTuple(1,0,5), XYZTuple(0,6,0), XYZTuple(0,5,1), XYZTuple(0,4,2), XYZTuple(0,3,3), 
 XYZTuple(0,2,4), XYZTuple(0,1,5), XYZTuple(0,0,6))
]

const SubshellXYZsizes = length.(SubshellXYZs)


function ijkToStr(ijk::XYZTuple)
    res = ""
    xyz = ("X", "Y", "Z")
    for (i, j) in zip(xyz, ijk)
        res *= i * superscriptNum[j + '0']
    end
    res
end


const BasisSetNames = 
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
const BasisSetList = Dict(BasisSetNames .=> BasisFuncTexts)
const AtomicNumberList = Dict(ElementNames .=> collect(1 : length(ElementNames)))
const AngularMomentumList = Dict(SubshellNames .=> collect(0 : length(SubshellNames)-1))
const SubshellOrientationList = Dict(SubshellNames .=> SubshellXYZs)
const SubshellSizeList = Dict(SubshellNames .=> SubshellXYZsizes)
const ijkIndexList = Dict(flatten(SubshellXYZs) .=> 
                     flatten([collect(1:length(i)) for i in SubshellXYZs]))
# const ParamNames = [:𝑋, :𝑌, :𝑍, :𝑑, :𝛼, :𝐿]
const ParamNames = [:X, :Y, :Z, :d, :α, :L]
const ParamSymbols = [:X, :Y, :Z, :con, :xpn, :spacing]
const ParamList = Dict(ParamSymbols .=> ParamNames)

const αParamSym = ParamList[:xpn]
const dParamSym = ParamList[:con]
const XParamSym = ParamList[:X]
const YParamSym = ParamList[:Y]
const ZParamSym = ParamList[:Z]



"""

    getCharge(nucs::Union{Vector{String}, Tuple{Vararg{String}}}) -> Int

Return the total electric charge (in 𝑒) of the input nuclei.
"""
getCharge(nucs::Union{Vector, Tuple}) = getCharge.(nucs) |> sum

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