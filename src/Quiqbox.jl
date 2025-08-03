module Quiqbox

include("Types.jl")
include("Dispatch.jl")

include("../lib/BasisSets/BasisSets.jl")
include("Lexicons.jl")

include("Tools.jl")
include("Strings.jl")
include("Exception.jl")
include("ParallelUtils.jl")

include("Query.jl")
include("Iteration.jl")
include("Collections.jl")
include("Mapping.jl")
include("Operators.jl")
include("Parameters.jl")
include("Graphs.jl")
include("Arithmetic.jl")

include("Angular.jl")
include("Particles.jl")

include("FieldFunctions.jl")
include("OrbitalBases.jl")

include("Differentiation/Finite.jl")

include("Integration/Samplers.jl")
include("Integration/Framework.jl")
include("Integration/Engines/Numerical.jl")
include("Integration/Engines/BoysFunction.jl")
include("Integration/Engines/GaussianOrbitals.jl")
include("Integration/Interface.jl")

include("HartreeFock.jl")

include("IO.jl")

include("Precompilation.jl")

end

## Development comment guideline
#!! Urgent issue
#! Non-urgent issue
#? Consideration for change or new implementation
#> Code mechanism explanation
#- Code method reference

## Footnote comment template
#= Additional Method =#