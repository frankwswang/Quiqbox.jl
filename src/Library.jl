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
"G"
]


const SubshellAngularMomentums = 
[
["X⁰Y⁰Z⁰"],
["X¹Y⁰Z⁰", "X⁰Y¹Z⁰", "X⁰Y⁰Z¹"],
["X²Y⁰Z⁰", "X¹Y¹Z⁰", "X¹Y⁰Z¹", "X⁰Y²Z⁰", "X⁰Y¹Z¹", "X⁰Y⁰Z²"],
["X³Y⁰Z⁰", "X²Y¹Z⁰", "X²Y⁰Z¹", "X¹Y²Z⁰", "X¹Y¹Z¹", "X¹Y⁰Z²", "X⁰Y³Z⁰", "X⁰Y²Z¹", "X⁰Y¹Z²", "X⁰Y⁰Z³"],
["X⁴Y⁰Z⁰", "X³Y¹Z⁰", "X³Y⁰Z¹", "X²Y²Z⁰", "X²Y¹Z¹", "X²Y⁰Z²", "X¹Y³Z⁰", "X¹Y²Z¹", "X¹Y¹Z²", "X¹Y⁰Z³", "X⁰Y⁴Z⁰", "X⁰Y³Z¹", "X⁰Y²Z²", "X⁰Y¹Z³", "X⁰Y⁰Z⁴"]
]


const SubshellOrbitals = 
[ # Every XYZs must start with [l, 0, 0]
[[0,0,0]],
[[1,0,0], [0,1,0], [0,0,1]],
[[2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2]],
[[3,0,0], [2,1,0], [2,0,1], [1,2,0], [1,1,1], [1,0,2], [0,3,0], [0,2,1], [0,1,2], [0,0,3]],
[[4,0,0], [3,1,0], [3,0,1], [2,2,0], [2,1,1], [2,0,2], [1,3,0], [1,2,1], [1,1,2], [1,0,3], [0,4,0], [0,3,1], [0,2,2], [0,1,3], [0,0,4]]
]


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
const SubshellSuborderList = Dict(SubshellNames .=> SubshellOrbitals)
const ijkOrderList = Dict(SubshellNames .=> SubshellAngularMomentums)
const ijkIndexList = Dict(flatten(SubshellAngularMomentums) .=> flatten([collect(1:length(i)) for i in SubshellAngularMomentums]))
const ijkOrbitalList = Dict(flatten(SubshellAngularMomentums)  .=> flatten(SubshellOrbitals))
const ijkStringList = Dict(flatten(SubshellOrbitals) .=> flatten(SubshellAngularMomentums))
const SubshellDimList = Dict(SubshellNames .=> length.(SubshellOrbitals))
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