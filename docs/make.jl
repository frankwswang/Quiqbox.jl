push!(LOAD_PATH,"../src/")
using Documenter, Quiqbox

makedocs(
    sitename="Quiqbox.jl",
    modules = [Quiqbox],
    authors="Weishi Wang",
    pages=[
        "Home"=>"index.md"
        "Manual"=>[
            "SCF.md"
        ]
        "Base"=>[
            "coreFunction.md"
            "coreType.md"
        ]
        "Index"=>"list.md"
    ]
)

deploydocs(repo="github.com/frankwswang/Quiqbox.jl.git", 
           devbranch = "main",
           target = "build",
           push_preview = true)