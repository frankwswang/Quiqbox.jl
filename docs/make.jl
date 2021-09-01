push!(LOAD_PATH,"../src/")
using Documenter, Quiqbox

makedocs(
    sitename="Quiqbox.jl",
    modules = [Quiqbox],
    authors="Weishi Wang",
    pages=[
        "Home"=>"index.md"
        "Manual"=>[
            "basis.md"
            "SCF.md"
            "optimization.md"
        ]
        "Base"=>[
            "coreFunction.md"
            "toolFunction.md"
            "coreType.md"
        ]
        "Submodule"=>[
            "molden.md"
        ]
        "Index"=>"list.md"
    ]
)

deploydocs(repo="github.com/frankwswang/Quiqbox.jl.git", 
           branch = "gh-pages",
           devbranch = "main")