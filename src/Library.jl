export LTuple, orbitalLin, getCharge, ParamList

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


struct LTuple{D, L}
    tuple::NTuple{D, Int}

    function LTuple{D, L}(t::NTuple{D, Int}) where {D, L}
        @assert all(t .>= 0)
        new{D, sum(t)}(t)
    end

    LTuple(xyz::LTuple{D, L}) where {D, L} = new{D, L}(xyz.tuple)

    LTuple(xyz1::LTuple{D, L1}, xyz2::LTuple{D, L2}) where {D, L1, L2} = 
    new{D, L1+L2}(xyz1.tuple .+ xyz2.tuple)

    LTuple(xyz::LTuple{D, L}, ::LTuple{D, 0}) where {D, L} = new{D, L}(xyz.tuple)

    LTuple(::LTuple{D, 0}, xyz::LTuple{D, L}) where {D, L} = new{D, L}(xyz.tuple)

    LTuple(::LTuple{D, 0}, ::LTuple{D, 0}) where {D} = new{D, 0}((0, 0, 0))
end

LTuple(t::NTuple{D, Int}) where {D} = LTuple{D, sum(t)}(t)
LTuple(args::Vararg{Int}) = LTuple(args)
LTuple(a::Vector{Int}) = LTuple(a...)

import Base: iterate, size, length, eltype
iterate(snt::LTuple, args...) = iterate(snt.tuple, args...)
size(snt::LTuple, args...) = size(snt.tuple, args...)
length(::LTuple{D}) where {D} = D
eltype(::LTuple) = Int

import Base: +
+(xyz1::LTuple{D, L1},  xyz2::LTuple{D, L2} ) where {D, L1, L2} = LTuple(xyz1, xyz2)
+( xyz::LTuple{D, L},      t::NTuple{D, Int}) where {D, L} = xyz + LTuple{D, sum(t)}(t)
+(   t::NTuple{D, Int},  xyz::LTuple{D, L}  ) where {D, L} = +(xyz, t)

import Base: Tuple, sum, map, isless
Tuple(xyz::LTuple) = xyz.tuple
sum(::LTuple{<:Any, L}) where {L} = L
sum(f, xyz::LTuple) = sum(f, xyz.tuple)
map(f, x::LTuple{D, L1}, y::LTuple{D, L2}) where {D, L1, L2} = map(f, x.tuple, y.tuple)
map(f, xyzs::Vararg{LTuple{D}, N}) where {D, N} = map(f, getproperty.(xyzs, :tuple)...)
isless(xyz1::LTuple{D}, xyz2::LTuple{D}) where {D} = isless(xyz1.tuple, xyz2.tuple)

import Base: broadcastable
Base.broadcastable(xyz::LTuple) = Base.broadcastable(xyz.tuple)

const SubshellXs = 
[(LTuple(0,),),(LTuple(1,),),(LTuple(2,),),(LTuple(3,),),(LTuple(4,),),(LTuple(5,),),
 (LTuple(6,),)]

const SubshellXYs = 
[ # Every XYs must start with (l, 0)
(LTuple(0,0),),
(LTuple(1,0), LTuple(0,1)),
(LTuple(2,0), LTuple(1,1), LTuple(0,2)),
(LTuple(3,0), LTuple(2,1), LTuple(1,2), LTuple(0,3)),
(LTuple(4,0), LTuple(3,1), LTuple(2,2), LTuple(1,3), LTuple(0,4)),
(LTuple(5,0), LTuple(4,1), LTuple(3,2), LTuple(2,3), LTuple(1,4), LTuple(0,5)),
(LTuple(6,0), LTuple(5,1), LTuple(4,2), LTuple(3,3), LTuple(2,4), LTuple(1,5), LTuple(0,6))
]

const SubshellXYZs = 
[ # Every XYZs must start with (l, 0, 0)
(LTuple(0,0,0),),
(LTuple(1,0,0), LTuple(0,1,0), LTuple(0,0,1)),
(LTuple(2,0,0), LTuple(1,1,0), LTuple(1,0,1), LTuple(0,2,0), LTuple(0,1,1), LTuple(0,0,2)),
(LTuple(3,0,0), LTuple(2,1,0), LTuple(2,0,1), LTuple(1,2,0), LTuple(1,1,1), LTuple(1,0,2), 
 LTuple(0,3,0), LTuple(0,2,1), LTuple(0,1,2), LTuple(0,0,3)),
(LTuple(4,0,0), LTuple(3,1,0), LTuple(3,0,1), LTuple(2,2,0), LTuple(2,1,1), LTuple(2,0,2), 
 LTuple(1,3,0), LTuple(1,2,1), LTuple(1,1,2), LTuple(1,0,3), LTuple(0,4,0), LTuple(0,3,1), 
 LTuple(0,2,2), LTuple(0,1,3), LTuple(0,0,4)),
(LTuple(5,0,0), LTuple(4,1,0), LTuple(4,0,1), LTuple(3,2,0), LTuple(3,1,1), LTuple(3,0,2), 
 LTuple(2,3,0), LTuple(2,2,1), LTuple(2,1,2), LTuple(2,0,3), LTuple(1,4,0), LTuple(1,3,1), 
 LTuple(1,2,2), LTuple(1,1,3), LTuple(1,0,4), LTuple(0,5,0), LTuple(0,4,1), LTuple(0,3,2), 
 LTuple(0,2,3), LTuple(0,1,4), LTuple(0,0,5)),
(LTuple(6,0,0), LTuple(5,1,0), LTuple(5,0,1), LTuple(4,2,0), LTuple(4,1,1), LTuple(4,0,2), 
 LTuple(3,3,0), LTuple(3,2,1), LTuple(3,1,2), LTuple(3,0,3), LTuple(2,4,0), LTuple(2,3,1), 
 LTuple(2,2,2), LTuple(2,1,3), LTuple(2,0,4), LTuple(1,5,0), LTuple(1,4,1), LTuple(1,3,2), 
 LTuple(1,2,3), LTuple(1,1,4), LTuple(1,0,5), LTuple(0,6,0), LTuple(0,5,1), LTuple(0,4,2), 
 LTuple(0,3,3), LTuple(0,2,4), LTuple(0,1,5), LTuple(0,0,6))
]

const SubshellLs = (SubshellXs, SubshellXYs, SubshellXYZs)

const SubshellXsizes = length.(SubshellXs)
const SubshellXYsizes = length.(SubshellXYs)
const SubshellXYZsizes = length.(SubshellXYZs)
const SubshellSizes =(SubshellXsizes, SubshellXYsizes, SubshellXYZsizes)

function LtoStr(lt::LTuple{D}) where {D}
    res = ""
    ls = ["X", "Y", "Z"][1:D]
    for (i, j) in zip(ls, lt)
        res *= i * superscriptNum[j + '0']
    end
    res
end


const SciNotMarker = r"D(?=[\+\-])"
const sciNotReplace = (txt)->replace(txt, SciNotMarker => "e")
const BStextEndingMarker = "****"
const BasisSetList = Dict(BasisSetNames .=> BasisFuncTexts)
const AtomicNumberList = Dict(ElementNames .=> collect(1 : length(ElementNames)))
const AngularMomentumList = Dict(SubshellNames .=> collect(0 : length(SubshellNames)-1))
const SubshellOrientationList = [Dict(SubshellNames .=> SubshellLs[1]), 
                                 Dict(SubshellNames .=> SubshellLs[2]), 
                                 Dict(SubshellNames .=> SubshellLs[3])]
const SubshellSizeList = [Dict(SubshellNames .=> SubshellXsizes),
                          Dict(SubshellNames .=> SubshellXYsizes),
                          Dict(SubshellNames .=> SubshellXYZsizes)]
const AngMomIndexList = [Dict(flatten(SubshellLs[1]) .=> 
                              flatten([collect(1:length(i)) for i in SubshellLs[1]])),
                         Dict(flatten(SubshellLs[2]) .=> 
                              flatten([collect(1:length(i)) for i in SubshellLs[2]])),
                         Dict(flatten(SubshellLs[3]) .=> 
                              flatten([collect(1:length(i)) for i in SubshellLs[3]]))]
# const ParamSyms = [:𝑋, :𝑌, :𝑍, :𝑑, :𝛼, :𝐿]
const SpatialParams = (:X, :Y, :Z)
const SpatialParamSyms = (:X, :Y, :Z)
const GaussFuncParams = (:xpn, :con)
const GaussFuncParamSyms = (:α, :d)
const OtherParams = [:spacing]
const OtherParamSyms = [:L]
const ParamSyms = vcat(SpatialParamSyms..., GaussFuncParamSyms..., OtherParamSyms)
const ParamAcrs = vcat(SpatialParams..., GaussFuncParams..., OtherParams)
const ParamList = Dict(ParamAcrs .=> ParamSyms)

const xpnSym = ParamList[:xpn]
const conSym = ParamList[:con]
const cxSym = SpatialParamSyms[1]
const cySym = SpatialParamSyms[2]
const czSym = SpatialParamSyms[3]

const spinOccupations = ("0", "↿", "⇂", "↿⇂")
const OrbitalOccupation = ((false, false), (true, false), (false, true), (true, true))
const SpinOrbitalOccupation = Dict(spinOccupations .=> OrbitalOccupation)
const SpinOrbitalSpinConfig = Dict(OrbitalOccupation .=> spinOccupations)


const HFtypes = (:RHF, :UHF)
const HFsizes = (1, 2)
const HFtypeSizeList = Dict(HFtypes .=> HFsizes)


"""

    orbitalLin(subshell::String, D::Int=3) -> Tuple{Vararg{NTuple{3, Int}}}

Return all the possible angular momentum configuration(s) within the input `subshell` of 
`D` dimension.
"""
orbitalLin(subshell::String, D::Int=3) = 
getproperty.(SubshellLs[D][AngularMomentumList[subshell]+1], :tuple)


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
            "[A-Z][a-z]?     [0-9]\\n" (regex) should exit as the first line of the string!
            "$(BStextEndingMarker)\\n" should exit as the last line of the string!
            The incorrect text content is:\n$(text)
            """
        end
    end
    printInfo && println("Basis function list checked.")
end


const ValInts = Val.(collect(0:5000))

function getValI(i::Int)
    i = abs(i)
    i < length(ValInts) ? ValInts[i+1] : ValInts[end]
end